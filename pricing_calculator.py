#!/usr/bin/env python3
import requests
import time
import json
import os
import numpy as np
import asyncio
import aiohttp
from transformers import AutoTokenizer

def wait_for_server(base_url, timeout = 300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM server did not start within timeout period")

# ----------------------------
# Workload stats loader
# ----------------------------
def load_wildchat_workload(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    isl_bins = obj["isl_bins"]
    osl_bins = obj["osl_bins"]
    P = np.array(obj["probabilities"], dtype=float)

    k = float(obj["arrival_k"])
    theta = float(obj["arrival_theta"])

    return isl_bins, osl_bins, P, k, theta

# ----------------------------
# Prompt utils 
# ----------------------------
def generate_random_prompt(tokenizer, num_tokens):
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    return prompt

def bin_midpoints(edges):
    return [(edges[i] + edges[i+1]) // 2 for i in range(len(edges)-1)]

def make_sampler_wildchat(isl_bins, osl_bins, P):
    """Sample a (bin_i, bin_j) by probability, return (isl_mid, osl_mid)."""
    isl_mids = bin_midpoints(isl_bins)
    osl_mids = bin_midpoints(osl_bins)
    flat = P.ravel()
    idx = np.arange(flat.size)
    W = P.shape[1]
    def sampler():
        k = np.random.choice(idx, p=flat)
        i, j = divmod(k, W)                 # map 1D index -> (row i, col j)
        return isl_mids[i], osl_mids[j]
    return sampler

def make_sampler_generic(workloads):
    """Uniform sampler over fixed (isl, osl) tuples."""
    def sampler():
        return workloads[np.random.randint(0, len(workloads))]
    return sampler

# ----------------------------
# Inference call
# ----------------------------

async def send_request(session, base_url, model, prompt, osl, request_id):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "stop": [],
        "ignore_eos": True,
        "stream": True,
    }
    
    started = time.perf_counter()
    first_arrival = last_arrival = None

    try:
        async with session.post(f"{base_url}/v1/completions", json=payload, timeout=120) as response:
            if response.status == 200:
                async for line in response.content:
                    data = line.decode('utf-8').strip()[6:]  # Remove 'data: ' prefix
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter()
                    if first_arrival is None:
                        first_arrival = now
                        isl = len(obj["choices"][0]["prompt_token_ids"])
                    last_arrival = now
                prefill = first_arrival - started
                decode = last_arrival - first_arrival
                print(f"  Request {request_id}: Prefill time: {prefill}, Decode time: {decode}, Input tokens: {isl}, Output tokens: {osl}")
                return prefill, decode
            else:
                print(f"  Warning: Request {request_id} failed with status {response.status}")
                return 0, 0
    except Exception as e:
        print(f"  Error in request {request_id}: {e}")
        return 0, 0

# ----------------------------
# Benchmark loop
# ----------------------------
async def benchmark_throughput(base_url, model, num_requests, sampler, k, theta):
    print(f"\nBenchmarking with {num_requests} requests...")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = {} # Prompt cache keyed by ISL

    total_in_tok = total_out_tok = 0
    total_prefill = total_decode = 0.0

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            isl, osl = sampler()
            if isl not in prompts:
                prompts[isl] = generate_random_prompt(tokenizer, isl)

            # If k & theta provided, add gaps; else fire immediately
            if (k is not None) and (theta is not None) and i > 0:
                gap = np.random.gamma(shape=k, scale=theta)
                await asyncio.sleep(gap)
            tasks.append(asyncio.create_task(
                send_request(session, base_url, model, prompts[isl], osl, i + 1)
            ))

            total_in_tok += isl
            total_out_tok += osl

        results = await asyncio.gather(*tasks)
    
    for (prefill, decode) in results:
        total_prefill += prefill
        total_decode += decode
    
    tps_in = total_in_tok / total_prefill
    tps_out = total_out_tok / total_decode
    return tps_in, tps_out

# ----------------------------
# Pricing
# ----------------------------
def calculate_pricing(tps_in, tps_out, gpu_cost_per_hour):
    gpu_cost_per_sec = gpu_cost_per_hour / 3600.0
    price_in_tok = gpu_cost_per_sec / tps_in
    price_out_tok = gpu_cost_per_sec / tps_out

    return {
        "tps_in": tps_in,
        "tps_out": tps_out,
        "cost_per_input_token": price_in_tok,
        "cost_per_output_token": price_out_tok,
        "cost_per_1k_input_tokens": price_in_tok * 1_000,
        "cost_per_1k_output_tokens": price_out_tok * 1_000,
        "cost_per_1m_input_tokens": price_in_tok * 1_000_000,
        "cost_per_1m_output_tokens": price_out_tok * 1_000_000
    }

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    model = os.environ.get('MODEL')
    gpu_type = os.environ.get('GPU_TYPE')
    gpu_cost = float(os.environ.get('GPU_COST'))
    num_requests = int(os.environ.get('NUM_REQUESTS'))
    port = os.environ.get('PORT')
    base_url = f"http://localhost:{port}"
    
    wait_for_server(base_url)

    mode = os.environ.get("WORKLOAD_MODE", "generic").lower()

    # Display config
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hour)")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    if mode == "generic":
        workloads = [
            (1024, 1024),  # chat
            (1024, 8192),  # reasoning
            (8192, 1024),  # summarising
        ]
        sampler = make_sampler_generic(workloads)
        k = theta = None

    elif mode == "wildchat":
        workload_stats_path = os.environ.get("WORKLOAD_STATS_PATH")
        if not workload_stats_path or not os.path.exists(workload_stats_path):
            raise FileNotFoundError("WORKLOAD_STATS_PATH is missing.")

        isl_bins, osl_bins, P, k, theta = load_wildchat_workload(workload_stats_path)
        sampler = make_sampler_wildchat(isl_bins, osl_bins, P)
    else:
        raise ValueError(f"Unknown WORKLOAD_MODE: {mode}")
     
    (tps_in, tps_out) = asyncio.run(
        benchmark_throughput(base_url, model, num_requests, sampler, k, theta
    ))
    
    pricing = calculate_pricing(tps_in, tps_out, gpu_cost)
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Throughput:in={tps_in:.2f} tok/s | out={tps_out:.2f} tok/s")
    print(f"\nPRICING:")
    print(f"  Input tokens:")
    print(f"    Per token:     ${pricing['cost_per_input_token']:.10f}")
    print(f"    Per 1K tokens: ${pricing['cost_per_1k_input_tokens']:.8f}")
    print(f"    Per 1M tokens: ${pricing['cost_per_1m_input_tokens']:.4f}")
    print(f"  Output tokens:")
    print(f"    Per token:     ${pricing['cost_per_output_token']:.10f}")
    print(f"    Per 1K tokens: ${pricing['cost_per_1k_output_tokens']:.8f}")
    print(f"    Per 1M tokens: ${pricing['cost_per_1m_output_tokens']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()