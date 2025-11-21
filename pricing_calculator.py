#!/usr/bin/env python3
import time
import json
import os
import asyncio
import aiohttp
from transformers import AutoTokenizer
import numpy as np

from utils import wait_for_server, generate_random_prompt
from traffic_profile import TrafficProfile

TARGET_UTILISATION = 0.85

class Benchmarker:
    def __init__(self, base_url, model_name, tokenizer):
        self.base_url = base_url
        self.model = model_name
        self.tokenizer = tokenizer
        self.headers = {"Content-Type": "application/json"}

        self.batch_schedule = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    async def _measure_single_request(self, session, payload_bytes):
        start_t = time.perf_counter()
        first_token_t = end_t = None

        try:
            async with session.post(f"{self.base_url}/v1/completions", data=payload_bytes, headers=self.headers, timeout=120) as response:
                if response.status != 200: return None
                async for line in response.content:
                    data = line.decode('utf-8').strip()[6:]  # Remove 'data: ' prefix
                    if data == "[DONE]":break
                    if first_token_t is None: first_token_t = time.perf_counter()
                end_t = time.perf_counter()
                return {
                    "start_t": start_t,
                    "first_token_t": first_token_t,
                    "end_t": end_t,
                }
        except Exception as e:
            return None

    async def run_sweep(self, isl, osl):
        prompt = generate_random_prompt(self.tokenizer, isl)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": osl,
            "stop": [],
            "ignore_eos": True,
            "stream": True 
        }
        payload_bytes = json.dumps(payload).encode('utf-8')

        perf_measurements = []
        
        print(f"\n[Benchmarker] Sweeping for Saturation (ISL={isl}, OSL={osl})")
        
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup
            await self._measure_single_request(session, payload_bytes)

            for b in self.batch_schedule:
                tasks = [self._measure_single_request(session, payload_bytes) for _ in range(b)]
                results = await asyncio.gather(*tasks)
                valid = [r for r in results if r is not None]

                if len(valid) < b * 0.8:
                    print(f"Batch: {b:<6} | FAIL (High Error Rate)")
                    break

                start_times = [r['start_t'] for r in valid]
                first_token_times = [r['first_token_t'] for r in valid]
                end_times = [r['end_t'] for r in valid]

                batch_start_t = min(start_times)
                batch_prefill_end_t = max(first_token_times)
                batch_end_t = max(end_times)

                prefill = batch_prefill_end_t - batch_start_t
                decode = batch_end_t - batch_prefill_end_t

                input_tps = (len(valid) * isl) / prefill
                output_tps = (len(valid) * osl) / decode

                ttfts = [(r['first_token_t'] - r['start_t']) for r in valid]
                tpots = [(r['end_t'] - r['first_token_t']) / osl for r in valid]
                p90_ttft = np.percentile(ttfts, 90)
                p90_tpot = np.percentile(tpots, 90)

                print(f"Batch: {b:<6} | Input TPS: {input_tps:<15.5f} | Output TPS: {output_tps:<15.5f} | P90 TTFT: {p90_ttft:<15.5f} | P90 TPOT: {(p90_tpot*1000):<15.5f}ms")
                # for i, r in enumerate(valid):
                #     print(f" Batch: {b} | ID: {i} | Start: {r['start_t']:.10f} | First: {r['first_token_t']:.10f} | End: {r['end_t']:.10f} | Prefill: {r['first_token_t'] - r['start_t']:.10f} | Decode: {r['end_t'] - r['first_token_t']:.10f} | Input TPS: {len(valid)*isl/(r['first_token_t'] - r['start_t']):.10f} |  Output TPS: {len(valid) *osl/(r['end_t'] - r['first_token_t']):.10f}")
                # print("")

                perf_measurements.append({
                    "batch_size": b,
                    "input_tps": input_tps,
                    "output_tps": output_tps,
                    "p90_ttft": p90_ttft,
                    "p90_tpot": p90_tpot
                })


        return perf_measurements
    

def calculate_prices(measurements, gpu_cost):
    slos = [
        ("Baseline", 1.0, 0.05), # 20 tok/s
        ("Fast",     0.5, 0.0167), # 60 tok/s
        ("Instant", 0.25, 0.01), # 100 tok/s
    ]

    gpu_cost_per_sec = gpu_cost / 3600.0
    
    print(f"\n{'='*110}")
    print(f"PRICING MATRIX (Hardware Cost: ${gpu_cost:.2f}/hr)")
    print(f"{'='*110}")
    print(f"{'Scenario':<22} | {'SLO (TTFT/TPOT)':<18} | {'Max Batch':<10} | {'In TPS':<10} | {'Out TPS':<10} | {'In $/1M':<10} | {'Out $/1M':<10}")
    print(f"{'-'*110}")

    for name, max_ttft, max_tpot in slos:
        valid_runs = [
            run for run in measurements 
            if run['p90_ttft'] <= max_ttft and run['p90_tpot'] <= max_tpot
        ]
        
        slo_desc = f"<{max_ttft}s / <{int(max_tpot*1000)}ms"

        if not valid_runs:
            print(f"{name:<22} | {slo_desc:<18} | {'None':<10} | {'-':<10} | {'-':<10} | {'-':<10}")
            continue

        best_run = max(valid_runs, key=lambda x: x['output_tps'])        
        price_in_token = gpu_cost_per_sec / (best_run['input_tps'] * TARGET_UTILISATION)
        price_out_token = gpu_cost_per_sec / (best_run['output_tps'] * TARGET_UTILISATION)
        price_in_1m = price_in_token * 1_000_000
        price_out_1m = price_out_token * 1_000_000
        print(f"{name:<22} | {slo_desc:<18} | {best_run['batch_size']:<10} | {best_run['input_tps']:<10.0f} | {best_run['output_tps']:<10.0f} | {price_in_1m:<10.4f} | {price_out_1m:<10.4f}")

    print(f"{'='*110}")
    print(f"* Assumes {TARGET_UTILISATION*100}% utilisation.")


async def main():
    MODEL = os.environ.get('MODEL')
    GPU_TYPE = os.environ.get('GPU_TYPE')
    GPU_COST = float(os.environ.get('GPU_COST'))
    PORT = os.environ.get('PORT')
    BASE_URL = f"http://localhost:{PORT}"
    
    #Print config

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    profile = TrafficProfile(tokenizer)
    metrics = profile.compute_metrics()
    isl, osl = metrics['avg_isl'], metrics['avg_osl']
    print(f"Traffic Profile: avg ISL={isl}, avg OSL={osl}")

    await wait_for_server(BASE_URL)

    benchmarker = Benchmarker(BASE_URL, MODEL, tokenizer)
    measurements = await benchmarker.run_sweep(isl, osl)
    if measurements: calculate_prices(measurements, GPU_COST)

if __name__ == "__main__":
    asyncio.run(main())