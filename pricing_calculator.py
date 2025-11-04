#!/usr/bin/env python3
import requests
import time
import json
import os
import numpy as np
import asyncio
import aiohttp
from transformers import AutoTokenizer

def wait_for_server(base_url: str = "http://localhost:8000", timeout: int = 300):
    """Wait for vLLM server to be ready"""
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
# Prompt gen 
# ----------------------------

def load_joint_probs(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    isl_bins = obj["isl_bins"]
    osl_bins = obj["osl_bins"]
    P = np.array(obj["probabilities"], dtype=float)
    return isl_bins, osl_bins, P

def generate_random_prompt(tokenizer, num_tokens: int) -> str:
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    return prompt

def bin_midpoints(edges):
    mids = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mids.append((lo + hi) // 2)
    return mids

def make_sampler_generic(workloads):
    """Uniform sampler over fixed (isl, osl) tuples."""
    def sampler():
        return workloads[np.random.randint(0, len(workloads))]
    return sampler

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
    }
    
    start_time = time.time()
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload, timeout=120) as response:
            end_time = time.time()
            if response.status == 200:
                result = await response.json()
                tokens_input = result['usage']['prompt_tokens']
                tokens_generated = result['usage']['completion_tokens']
                print(f"  Request {request_id}: Input tokens: {tokens_input}, Output tokens: {tokens_generated}")
                return tokens_generated, end_time - start_time
            else:
                print(f"  Warning: Request {request_id} failed with status {response.status}")
                return 0, 0
    except Exception as e:
        print(f"  Error in request {request_id}: {e}")
        return 0, 0
    

def get_prompt_for_isl(tokenizer, cache, isl):
    """Return a cached random prompt of isl tokens, generating if needed."""
    p = cache.get(isl)
    if p is None:
        p = generate_random_prompt(tokenizer, isl)
        cache[isl] = p
    return p

# ----------------------------
# Benchmark loop
# ----------------------------

async def benchmark_throughput(
    base_url: str,
    model: str,
    num_requests: int,
    arrival_rate: float,
    arrival_burstiness: float,
    sampler
) -> float:
    print(f"\nBenchmarking with {num_requests} requests...")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Lazy prompt cache keyed by ISL
    prompts_by_isl = {}

    overall_start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            isl, osl = sampler()
            prompt = get_prompt_for_isl(tokenizer, prompts_by_isl, int(isl))

            if arrival_rate > 0 and i > 0:
                k = max(arrival_burstiness, 1e-6)
                theta = 1.0 / (arrival_rate * k)
                gap = np.random.gamma(shape=k, scale=theta)
                await asyncio.sleep(gap)
            tasks.append(asyncio.create_task(
                send_request(session, base_url, model, prompt, osl, i + 1)
            ))

        results = await asyncio.gather(*tasks)
    
    overall_end = time.time()
    
    total_tokens = sum(tokens for tokens, _ in results)
    total_time = overall_end - overall_start
    throughput = total_tokens / total_time
    return throughput

# ----------------------------
# Pricing
# ----------------------------

def calculate_pricing(throughput: float, gpu_cost_per_hour: float):
    tokens_per_hour = throughput * 3600
    cost_per_token = gpu_cost_per_hour / tokens_per_hour
    
    return {
        "throughput_tokens_per_sec": throughput,
        "tokens_per_hour": tokens_per_hour,
        "cost_per_token": cost_per_token,
        "cost_per_1k_tokens": cost_per_token * 1000,
        "cost_per_1m_tokens": cost_per_token * 1000000,
    }

# ----------------------------
# Entrypoint
# ----------------------------

def main():
    model = os.environ.get('MODEL')
    gpu_type = os.environ.get('GPU_TYPE')
    gpu_cost = float(os.environ.get('GPU_COST'))
    num_requests = int(os.environ.get('NUM_REQUESTS'))
    arrival_rate = float(os.environ.get('ARRIVAL_RATE'))
    arrival_burstiness = float(os.environ.get('ARRIVAL_BURSTINESS'))
    port = os.environ.get('PORT')
    base_url = f"http://localhost:{port}"
    
    wait_for_server(base_url)
    
    # Display config
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hour)")
    print(f"{'='*60}")

    mode = os.environ.get("WORKLOAD_MODE", "generic").lower()

    if mode == "generic":
        workloads = [
            (1024, 1024),  # chat
            (1024, 8192),  # reasoning
            (8192, 1024),  # summarising
        ]
        sampler = make_sampler_generic(workloads)
    elif mode == "wildchat":
        joint_probs_path = os.environ.get("JOINT_PROBS_PATH")
        if not joint_probs_path or not os.path.exists(joint_probs_path):
            raise FileNotFoundError("WORKLOAD_MODE=wildchat but JOINT_PROBS_PATH is missing.")
        isl_bins, osl_bins, P = load_joint_probs(joint_probs_path)
        sampler = make_sampler_wildchat(isl_bins, osl_bins, P)
    else:
        raise ValueError(f"Unknown WORKLOAD_MODE: {mode}")

    
    throughput = asyncio.run(benchmark_throughput(
        base_url=base_url,
        model=model,
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        arrival_burstiness=arrival_burstiness,
        sampler=sampler
    ))
    
    pricing = calculate_pricing(throughput, gpu_cost)
    
    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Throughput: {pricing['throughput_tokens_per_sec']:.2f} tokens/second")
    print(f"Tokens per hour: {pricing['tokens_per_hour']:.0f}")
    print(f"\nPRICING:")
    print(f"  Per token:     ${pricing['cost_per_token']:.10f}")
    print(f"  Per 1K tokens: ${pricing['cost_per_1k_tokens']:.8f}")
    print(f"  Per 1M tokens: ${pricing['cost_per_1m_tokens']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()