#!/usr/bin/env python3
import time
import json
import os
import numpy as np
import asyncio
import aiohttp
from transformers import AutoTokenizer

from utils import (
    wait_for_server,
    load_wildchat_workload,
    bin_midpoints,
    generate_random_prompt,
)

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
async def benchmark_throughput(base_url, model, num_requests, sampler, k, theta, concurrency_cap=None):
    print(f"\nBenchmarking with {num_requests} requests...")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = {} # Prompt cache keyed by ISL

    jobs = []
    total_in_tok = total_out_tok = 0
    for _ in range(num_requests):
        isl, osl = sampler()
        jobs.append((isl, osl))
        total_in_tok += isl
        total_out_tok += osl
        if isl not in prompts:
            prompts[isl] = generate_random_prompt(tokenizer, isl)

    total_prefill = total_decode = 0.0

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Case 1: concurrency cap (wildchat, calibrated)
        if concurrency_cap is not None:
            K = min(concurrency_cap, num_requests)
            in_flight = set()
            results = []
            next_idx = 0

            # Fill the pipeline
            while len(in_flight) < K:
                isl, osl = jobs[next_idx]
                prompt = prompts[isl]
                task = asyncio.create_task(
                    send_request(session, base_url, model, prompt, osl, next_idx + 1)
                )
                in_flight.add(task)
                next_idx += 1

            # Closed-loop: whenever one finishes, start another
            while in_flight:
                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    results.append(task.result())
                    if next_idx < num_requests:
                        isl, osl = jobs[next_idx]
                        prompt = prompts[isl]
                        in_flight.add(
                            asyncio.create_task(
                                send_request(session, base_url, model, prompt, osl, next_idx + 1)
                            )
                        )
                        next_idx += 1

        # Case 2: original gap-based open-loop (used only when not wildchat)
        else:
            tasks = []
            for idx, (isl, osl) in enumerate(jobs):
                if (k is not None) and (theta is not None) and idx > 0:
                    gap = np.random.gamma(shape=k, scale=theta)
                    await asyncio.sleep(gap)
                tasks.append(asyncio.create_task(
                    send_request(session, base_url, model, prompts[isl], osl, idx + 1)
                ))
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
        
        calib_path = os.environ.get("WORKLOAD_CALIB_PATH")
        if not calib_path or not os.path.exists(calib_path):
            raise FileNotFoundError("WORKLOAD_CALIB_PATH is missing.")

        isl_bins, osl_bins, P, k, theta = load_wildchat_workload(workload_stats_path)
        sampler = make_sampler_wildchat(isl_bins, osl_bins, P)

        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        concurrency_cap = int(calib["concurrency_cap"])
        print(f"Using calibrated concurrency cap (wildchat): {concurrency_cap}")

    else:
        raise ValueError(f"Unknown WORKLOAD_MODE: {mode}")
     
    (tps_in, tps_out) = asyncio.run(
        benchmark_throughput(base_url, model, num_requests, sampler, k, theta, concurrency_cap)
    )
    
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
