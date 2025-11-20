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

class CapacityEstimator:
    def __init__(self, base_url, model_name, tokenizer):
        self.base_url = base_url
        self.model = model_name
        self.tokenizer = tokenizer
        self.headers = {"Content-Type": "application/json"}

        # SLO Constraints
        self.max_ttft = 1.0 
        self.max_tpot = 0.2

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

    async def find_peak_throughput(self, isl, osl):
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

        best = {
            "batch_size": 1,
            "input_tps": 0,
            "output_tps": 0
        }
        
        print(f"\n[ConcurrencyEstimator] Sweeping for Saturation (ISL={isl}, OSL={osl})")
        
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

                if p90_ttft > self.max_ttft:
                    print(f"SLO LIMIT REACHED: TTFT {p90_ttft:.5f}s > {self.max_ttft}s")
                    break
                if p90_tpot > self.max_tpot:
                    print(f"SLO LIMIT REACHED: TPOT {p90_tpot*1000:.5f}ms > {self.max_tpot*1000:.5f}ms")
                    break

                if output_tps > best["output_tps"]:
                    best = {
                        "batch_size": b,
                        "input_tps": input_tps,
                        "output_tps": output_tps
                    }
                print(f"Diff: {start_times[-1] - start_times[0]}")

        return best

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

    estimator = CapacityEstimator(BASE_URL, MODEL, tokenizer)
    capacity = await estimator.find_peak_throughput(isl, osl)
    tps_in = capacity["input_tps"]
    tps_out = capacity["output_tps"]

    if tps_in == 0 or tps_out == 0:
        print("Benchmark failed to get valid throughput.")
        return

    gpu_cost_per_sec = GPU_COST / 3600.0
    price_in = gpu_cost_per_sec / (tps_in * TARGET_UTILISATION)
    price_out = gpu_cost_per_sec / (tps_out * TARGET_UTILISATION)
    
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