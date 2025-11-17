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
    bin_midpoints,
    generate_random_prompt,
)

# ----------------------------
# Calibration request (non-streaming, simple)
# ----------------------------

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

# ----------------------------
# Bin-based average response time
# ----------------------------

async def estimate_avg_response_time(base_url, model, isl_bins, osl_bins, P, samples_per_bin):
    isl_mids = bin_midpoints(isl_bins)
    osl_mids = bin_midpoints(osl_bins)

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts_by_isl = {}

    total_weighted_R = 0.0

    async with aiohttp.ClientSession() as session:
        nonzero = np.argwhere(P > 0)
        for i, j in nonzero:
            prob_ij = float(P[i, j])
            isl_mid = int(isl_mids[i])
            osl_mid = int(osl_mids[j])

            # cached prompt per ISL midpoint
            if isl_mid not in prompts_by_isl:
                prompts_by_isl[isl_mid] = generate_random_prompt(tokenizer, isl_mid)
            prompt = prompts_by_isl[isl_mid]

            bin_durations = []
            print(f"bin({i},{j}): input tokens={isl_mid}, output tokens={osl_mid}")
            for s in range(samples_per_bin):
                dur = await send_request_cal(
                    session=session,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    osl=osl_mid,
                )
                print(f"  Sample {s+1}: t={dur:.3f}s")
                if dur > 0:
                    bin_durations.append(dur)

            avg_bin_R = float(np.mean(bin_durations))
            total_weighted_R += prob_ij * avg_bin_R

    return total_weighted_R

# ----------------------------
# Entrypoint
# ----------------------------

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

    isl_bins, osl_bins, P, k, theta = load_wildchat_workload(stats_path)

    print("Calibrating average response time over WildChat bins...")
    avg_R = asyncio.run(
        estimate_avg_response_time(
            base_url=base_url, 
            model=model, 
            isl_bins=isl_bins, 
            osl_bins=osl_bins, 
            P=P, 
            samples_per_bin=3
        )
    )
    print(f"\nEstimated avg response time: {avg_R:.3f} s")

    mean_gap = k * theta
    arrival_rate = 1.0 / mean_gap  # req/s
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
