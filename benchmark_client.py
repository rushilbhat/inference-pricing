#!/usr/bin/env python3
import time
import json
import os
import asyncio
import aiohttp
from transformers import AutoTokenizer
import numpy as np

from wildchat.traffic_profile import TrafficProfile
import config

class BenchmarkClient:
    def __init__(self, base_url, model_path, tokenizer):
        self.base_url = base_url
        self.model = model_path
        self.tokenizer = tokenizer
        self.batch_schedule = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    async def wait_for_server(self, timeout=300):
        print(f"\nWaiting for vLLM at {self.base_url}...")
        start = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start < timeout:
                try:
                    async with session.get(f"{self.base_url}/v1/models") as resp:
                        if resp.status == 200:
                            print("Server is ready.")
                            return True
                except:
                    pass
                await asyncio.sleep(2)
        raise TimeoutError("vLLM server did not start within timeout period")
    
    def generate_random_prompt(self, num_tokens):
        vocab_size = self.tokenizer.vocab_size
        random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
        return self.tokenizer.decode(random_token_ids, skip_special_tokens=True)
    
    async def get_vllm_metrics(self, session):
        try:
            async with session.get(f"{self.base_url}/metrics") as resp:
                if resp.status != 200: return {}
                text = await resp.text()

                keys = [
                    "vllm:prompt_tokens_total",
                    "vllm:generation_tokens_total",
                    "vllm:time_to_first_token_seconds_sum",
                    "vllm:time_to_first_token_seconds_count",
                    "vllm:inter_token_latency_seconds_sum",
                    "vllm:inter_token_latency_seconds_count"
                ]
                
                data = {}
                for line in text.splitlines():
                    if line.startswith("#"): continue
                    for k in keys:
                        if k in line:
                            try:
                                val = float(line.split()[-1])
                                data[k] = val
                            except:
                                pass
                return data
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {}


    async def send_request(self, session, isl, osl):
        prompt = self.generate_random_prompt(isl)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": osl,
            "stop": [],
            "ignore_eos": True,
            "stream": False
        }
        try:
            async with session.post(f"{self.base_url}/v1/completions", json=payload) as response:
                return response.status == 200
        except:
            return False


    async def run_sweep(self, isl, osl):
        print(f"\n[BenchmarkClient] ISL={isl}, OSL={osl}")
        
        perf_measurements = []
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup
            await self.send_request(session, isl, osl)

            for b in self.batch_schedule:
                m_start = await self.get_vllm_metrics(session)
                t_start = time.perf_counter()

                tasks = [self.send_request(session, isl, osl) for _ in range(b)]
                results = await asyncio.gather(*tasks)
                
                t_end = time.perf_counter()
                m_end = await self.get_vllm_metrics(session)
                
                if not all(results):
                    print(f"Batch: {b:<6} | FAIL (Errors detected)")
                    break

                duration = t_end - t_start

                def get_delta(key):
                    return m_end.get(key) - m_start.get(key)
                
                d_prompt_toks = get_delta('vllm:prompt_tokens_total')
                d_gen_toks = get_delta('vllm:generation_tokens_total')
                
                d_ttft_sum = get_delta('vllm:time_to_first_token_seconds_sum')
                d_ttft_count = get_delta('vllm:time_to_first_token_seconds_count')
                
                d_tpot_sum = get_delta('vllm:inter_token_latency_seconds_sum')
                d_tpot_count = get_delta('vllm:inter_token_latency_seconds_count')

                input_tps = d_prompt_toks / duration
                output_tps = d_gen_toks / duration

                avg_ttft = (d_ttft_sum / d_ttft_count)
                avg_tpot = (d_tpot_sum / d_tpot_count)

                print(f"Batch: {b:<6} | In TPS: {input_tps:<10.1f} | Out TPS: {output_tps:<10.1f} | Avg TTFT: {avg_ttft:<8.4f}s | Avg TPOT: {(avg_tpot*1000):<8.2f}ms")
                
                perf_measurements.append({
                    "batch_size": b,
                    "input_tps": input_tps,
                    "output_tps": output_tps,
                    "avg_ttft": avg_ttft,
                    "avg_tpot": avg_tpot
                })

        return perf_measurements
    

def print_pricing(measurements, server_cost_per_hr):
    server_cost_per_sec = server_cost_per_hr / 3600.0
    
    print(f"\n{'='*110}")
    print(f"PRICING MATRIX (Hardware Cost: ${server_cost_per_hr}/hr)")
    print(f"{'='*110}")
    print(f"{'Scenario':<22} | {'SLO (TTFT/TPOT)':<18} | {'Max Batch':<10} | {'In TPS':<10} | {'Out TPS':<10} | {'In $/1M':<10} | {'Out $/1M':<10}")
    print(f"{'-'*110}")

    for name, max_ttft, max_tpot in config.SLO_LEVELS:
        valid_runs = [
            run for run in measurements 
            if run['avg_ttft'] <= max_ttft and run['avg_tpot'] <= max_tpot
        ]
        slo_desc = f"<{max_ttft}s / <{int(max_tpot*1000)}ms"
        if not valid_runs:
            print(f"{name:<22} | {slo_desc:<18} | {'None':<10} | {'-':<10} | {'-':<10} | {'-':<10}")
            continue

        best_run = max(valid_runs, key=lambda x: x['output_tps'])        
        price_in_token = server_cost_per_sec / (best_run['input_tps'] * config.TARGET_UTILISATION)
        price_out_token = server_cost_per_sec / (best_run['output_tps'] * config.TARGET_UTILISATION)
        price_in_1m = price_in_token * 1_000_000
        price_out_1m = price_out_token * 1_000_000
        print(f"{name:<22} | {slo_desc:<18} | {best_run['batch_size']:<10} | {best_run['input_tps']:<10.0f} | {best_run['output_tps']:<10.0f} | {price_in_1m:<10.4f} | {price_out_1m:<10.4f}")

    print(f"{'='*110}")
    print(f"* Assumes {config.TARGET_UTILISATION*100}% utilisation.")


async def main(): 
    QUANTIZATION = os.environ.get('QUANTIZATION')
    TP = os.environ.get('TP')

    MODEL_PATH = config.QUANTIZATION_MAP[QUANTIZATION]
    BASE_URL = f"http://localhost:{config.PORT}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    profile = TrafficProfile(tokenizer, start_date=config.TRAFFIC_START_DATE, end_date=config.TRAFFIC_END_DATE)
    metrics = profile.compute_metrics()
    isl, osl = metrics['avg_isl'], metrics['avg_osl']

    benchmarker = BenchmarkClient(BASE_URL, MODEL_PATH, tokenizer)
    await benchmarker.wait_for_server()
    measurements = await benchmarker.run_sweep(isl, osl)

    if measurements: 
        print_pricing(measurements, config.SERVER_COST_PER_HR)

        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)
        entry = {"id": f"{QUANTIZATION}_{TP}", "measurements": measurements}
        try:
            with open(config.RESULTS_FILENAME, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    asyncio.run(main())