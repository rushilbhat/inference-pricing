#!/usr/bin/env python3
import requests
import time
import json
import os
import numpy as np
import asyncio
import aiohttp
from transformers import AutoTokenizer

WORKLOADS = [
    (1024, 1024),  # chat
    (1024, 8192),  # reasoning
    (8192, 1024),  # summarising
]

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

def generate_random_prompt(tokenizer, num_tokens: int) -> str:
    """Generate a random prompt by sampling random token IDs and detokenizing"""
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    return prompt

async def send_request(session, base_url, model, prompt, osl, request_id):
    """Send a single async request"""
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

async def benchmark_throughput(
    base_url: str,
    model: str,
    num_requests: int,
    arrival_rate: float,
    arrival_burstiness: float
) -> float:
    """Benchmark the model and return throughput in tokens/second"""    
    print(f"\nBenchmarking with {num_requests} requests...")
    print("  Mixed workloads (equal probability):")
    print("   - 1024 ISL / 1024 OSL (chat)")
    print("   - 1024 ISL / 8192 OSL (reasoning)")
    print("   - 8192 ISL / 1024 OSL (summarising)\n")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Pre-generate one prompt per distinct ISL to avoid re-decoding every time
    distinct_isl = sorted({isl for isl, _ in WORKLOADS})
    prompts_by_isl = {isl: generate_random_prompt(tokenizer, isl) for isl in distinct_isl}

    # Record overall start time
    overall_start = time.time()
    
    # Send all requests asynchronously
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            # Equal probability pick of workload
            isl, osl = WORKLOADS[np.random.randint(0, len(WORKLOADS))]
            prompt = prompts_by_isl[isl]
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
    
    # Calculate totals
    total_tokens = sum(tokens for tokens, _ in results)
    total_time = overall_end - overall_start
    throughput = total_tokens / total_time
    return throughput

def calculate_pricing(throughput: float, gpu_cost_per_hour: float):
    """Calculate pricing based on throughput and GPU cost"""
    tokens_per_hour = throughput * 3600
    cost_per_token = gpu_cost_per_hour / tokens_per_hour
    
    return {
        "throughput_tokens_per_sec": throughput,
        "tokens_per_hour": tokens_per_hour,
        "cost_per_token": cost_per_token,
        "cost_per_1k_tokens": cost_per_token * 1000,
        "cost_per_1m_tokens": cost_per_token * 1000000,
    }

def main():
    # Read configuration from environment variables
    model = os.environ.get('MODEL')
    gpu_type = os.environ.get('GPU_TYPE')
    gpu_cost = float(os.environ.get('GPU_COST'))
    num_requests = int(os.environ.get('NUM_REQUESTS'))
    arrival_rate = float(os.environ.get('ARRIVAL_RATE'))
    arrival_burstiness = float(os.environ.get('ARRIVAL_BURSTINESS'))
    port = os.environ.get('PORT')
    
    base_url = f"http://localhost:{port}"
    
    # Wait for server to be ready
    wait_for_server(base_url)
    
    # Display config
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hour)")
    print(f"{'='*60}")
    
    # Benchmark throughput
    throughput = asyncio.run(benchmark_throughput(
        base_url=base_url,
        model=model,
        num_requests=num_requests,
        arrival_rate=arrival_rate,
        arrival_burstiness=arrival_burstiness
    ))
    
    # Calculate pricing
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
    
    # Save results to JSON
    results = {
        "model": model,
        "gpu_type": gpu_type,
        "gpu_cost_per_hour": gpu_cost,
        "benchmark_config": {
            "workloads": [
                {"input_tokens": isl, "output_tokens": osl} for isl, osl in WORKLOADS
            ],
            "num_requests": num_requests,
            "arrival_rate": arrival_rate,
            "arrival_burstiness": arrival_burstiness
        },
        "pricing": pricing
    }
    
    with open('/workspace/pricing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: pricing_results.json")

if __name__ == "__main__":
    main()