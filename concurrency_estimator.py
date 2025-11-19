import time
import asyncio
import aiohttp
import numpy as np

from utils import generate_random_prompt, WARMUP_REQUESTS, ESTIMATE_SAMPLES

class ConcurrencyEstimator:
    def __init__(self, base_url, model_name, tokenizer):
        self.base_url = base_url
        self.model = model_name
        self.tokenizer = tokenizer

    async def _measure_response_time(self, session, prompt, osl):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": osl,
            "stop": [],
            "ignore_eos": True,
        }
        
        start = time.perf_counter()
        try:
            async with session.post(f"{self.base_url}/v1/completions", json=payload, timeout=120) as response:
                if response.status == 200:
                    await response.read() # Ensure full body is received
                    return time.perf_counter() - start
        except Exception as e:
            print(f"[ConcurrencyEstimator] Request failed: {e}")
        return None

    async def estimate(self, isl, osl, arrival_rate):
        prompt = generate_random_prompt(self.tokenizer, isl)
        latencies = []
        async with aiohttp.ClientSession() as session:
            print(f"\n[ConcurrencyEstimator] Sampling response time at ISL={isl}, OSL={osl}")
            for _ in range(WARMUP_REQUESTS):
                await self._measure_response_time(session, prompt, osl)

            for i in range(ESTIMATE_SAMPLES):
                lat = await self._measure_response_time(session, prompt, osl)
                if lat: 
                    latencies.append(lat)
                    print(f"  Sample {i+1}: {lat:.4f}s")

        # Fallback if calibration fails
        if not latencies:
            return {"concurrency_cap": 1, "utilisation": 1.0}

        avg_response_time = np.mean(latencies)
        average_concurrency = arrival_rate * avg_response_time
        concurrency_cap = max(int(np.ceil(average_concurrency)), 1)
        utilisation = min(1.0, average_concurrency)

        print(f"[ConcurrencyEstimator] Avg Response: {avg_response_time:.3f}s")
        print(f"[ConcurrencyEstimator] Average Concurrency: {average_concurrency:.3f}")
        print(f"[ConcurrencyEstimator] Concurrency Cap: {concurrency_cap}")

        return {
            "concurrency_cap": concurrency_cap,
            "utilisation": utilisation
        }