#!/usr/bin/env python3
import time
import json
import os
import asyncio
import aiohttp
from transformers import AutoTokenizer

from utils import wait_for_server, generate_random_prompt, WARMUP_REQUESTS
from traffic_profile import TrafficProfile
from concurrency_estimator import ConcurrencyEstimator


async def benchmark_request(session, base_url, model, prompt, isl, osl, request_id):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": osl,
        "stop": [],
        "ignore_eos": True,
        "stream": True,
    }
    
    start_t = time.perf_counter()
    first_token_t = None

    try:
        async with session.post(f"{base_url}/v1/completions", json=payload, timeout=120) as response:
            if response.status == 200:
                async for line in response.content:
                    data = line.decode('utf-8').strip()[6:]  # Remove 'data: ' prefix
                    if data == "[DONE]":break
                    if first_token_t is None: first_token_t = time.perf_counter()

                end_t = time.perf_counter()
                prefill = first_token_t - start_t
                decode = end_t - first_token_t
                print(f"  Request {request_id}: Prefill time: {prefill}, Decode time: {decode}, Input tokens: {isl}, Output tokens: {osl}")
                return prefill, decode
            else:
                print(f"  Warning: Request {request_id} failed with status {response.status}")
                return 0, 0
    except Exception as e:
        print(f"  Error in request {request_id}: {e}")
        return 0, 0

async def run_benchmark(base_url, model, tokenizer, num_requests, isl, osl, concurrency_cap):
    print(f"\nBenchmarking (Requests={num_requests}, Cap={concurrency_cap})")

    prompt = generate_random_prompt(tokenizer, isl)
    total_prefill = total_decode = 0.0

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(WARMUP_REQUESTS):
            await benchmark_request(session, base_url, model, prompt, isl, osl, f"warmup-{i + 1}")

        K = min(concurrency_cap, num_requests)
        in_flight = set()
        idx = 0

        # Fill the pipeline
        while len(in_flight) < K:
            in_flight.add(
                asyncio.create_task(benchmark_request(session, base_url, model, prompt, isl, osl, idx + 1))
            )
            idx += 1

        # Closed-loop: whenever one finishes, start another
        while in_flight:
            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                p, d = task.result()
                total_prefill += p
                total_decode += d
                if idx < num_requests:
                    in_flight.add(
                        asyncio.create_task(benchmark_request(session, base_url, model, prompt, isl, osl, idx + 1))
                    )
                    idx += 1

    tps_in = (isl * num_requests) / total_prefill
    tps_out = (osl * num_requests) / total_decode
    return tps_in, tps_out

async def main():
    MODEL = os.environ.get('MODEL')
    GPU_TYPE = os.environ.get('GPU_TYPE')
    GPU_COST = float(os.environ.get('GPU_COST'))
    NUM_REQUESTS = int(os.environ.get('NUM_REQUESTS'))
    PORT = os.environ.get('PORT')
    BASE_URL = f"http://localhost:{PORT}"
    
    #Print config

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    profile = TrafficProfile(tokenizer)
    metrics = profile.compute_metrics()
    print(f"\nTraffic Profile Metrics:")
    print(f"    ISL: {metrics['avg_isl']}")
    print(f"    OSL: {metrics['avg_osl']}")
    print(f"    Arrival Rate: {metrics['arrival_rate']:.4f} req/s")

    await wait_for_server(BASE_URL)

    estimator = ConcurrencyEstimator(BASE_URL, MODEL, tokenizer)
    capacity = await estimator.estimate(
        isl=metrics["avg_isl"], 
        osl=metrics["avg_osl"],
        arrival_rate=metrics["arrival_rate"]
    )

    tps_in, tps_out = await run_benchmark(
        base_url=BASE_URL,
        model=MODEL,
        tokenizer=tokenizer,
        num_requests=NUM_REQUESTS,
        isl=metrics["avg_isl"],
        osl=metrics["avg_osl"],
        concurrency_cap=capacity["concurrency_cap"]
    )

    gpu_cost_per_sec = GPU_COST / 3600.0
    util = capacity['utilisation']
    price_in = gpu_cost_per_sec / (util * tps_in)
    price_out = gpu_cost_per_sec / (util * tps_out)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Throughput:in={tps_in:.2f} tok/s | out={tps_out:.2f} tok/s")
    print(f"\nPRICING:")
    print(f"  Input tokens:")
    print(f"    Per token:     ${price_in:.10f}")
    print(f"    Per 1K tokens: ${price_in * 1000:.8f}")
    print(f"    Per 1M tokens: ${price_in * 1000000:.4f}")
    print(f"  Output tokens:")
    print(f"    Per token:     ${price_out:.10f}")
    print(f"    Per 1K tokens: ${price_out * 1000:.8f}")
    print(f"    Per 1M tokens: ${price_out * 1000000:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())