#!/usr/bin/env python3
import os
import json
import time
import asyncio
import numpy as np
import aiohttp
from transformers import AutoTokenizer

from utils import (
    wait_for_server,
    load_wildchat_workload,
    generate_random_prompt,
)

async def send_request_cal(session, base_url, model, prompt, osl):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "stop": [],
        "ignore_eos": True,
    }

    start_time = time.perf_counter()
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload, timeout=120) as response:
            end_time = time.perf_counter()
            if response.status == 200:
                # await response.json()
                return end_time - start_time
            else:
                print(f" Warning: status={response.status}")
                return 0.0
    except Exception as e:
        print(f" Error: {e}")
        return 0.0

async def estimate_avg_response_time(base_url, model, isl_toks, osl_toks, samples):
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = generate_random_prompt(tokenizer, isl_toks)

    durations = []
    async with aiohttp.ClientSession() as session:
        print(f"Sampling response time at ISL={isl_toks}, OSL={osl_toks}")
        for s in range(samples):
            dur = await send_request_cal(
                session=session,
                base_url=base_url,
                model=model,
                prompt=prompt,
                osl=osl_toks,
            )
            print(f"  Sample {s+1}: t={dur:.3f}s")
            if dur > 0:
                durations.append(dur)

    if not durations:
        print("Warning: no successful calibration samples, returning 0.")
        return 0.0

    return float(np.mean(durations))

def main():
    model = os.environ.get("MODEL")
    port = os.environ.get("PORT")
    base_url = f"http://localhost:{port}"

    stats_path = os.environ.get("WORKLOAD_STATS_PATH")
    calib_path = os.environ.get("WORKLOAD_CALIB_PATH")

    print(f"\n=== Calibration ===")
    print(f"Model: {model}")
    print(f"============================\n")

    wait_for_server(base_url)

    avg_isl, avg_osl, mean_gap, arrival_rate = load_wildchat_workload(stats_path)

    print("Calibrating average response time at mean tokens...")
    avg_R = asyncio.run(
        estimate_avg_response_time(
            base_url=base_url, 
            model=model, 
            isl_toks=avg_isl,
            osl_toks=avg_osl,
            samples=3,
        )
    )
    print(f"\nEstimated avg response time: {avg_R:.3f} s")

    natural_conc = arrival_rate * avg_R
    concurrency_cap = max(int(np.ceil(natural_conc)), 1)


    print("\nConcurrency summary:")
    print(f"  mean gap            : {mean_gap:.3f} s")
    print(f"  λ (arrival rate)    : {arrival_rate:.6f} req/s")
    print(f"  natural concurrency : {natural_conc:.3f}")
    print(f"  concurrency cap     : {concurrency_cap}\n")

    calib = {
        "model": model,
        "avg_response_time": avg_R,
        "mean_gap": mean_gap,
        "arrival_rate": arrival_rate,
        "natural_concurrency": natural_conc,
        "concurrency_cap": concurrency_cap,
    }

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration to {calib_path}\n")

if __name__ == "__main__":
    main()
