#!/usr/bin/env python3
import requests
import time
import json
import os
import numpy as np
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

def generate_random_prompt(tokenizer, num_tokens: int) -> str:
    """Generate a random prompt by sampling random token IDs and detokenizing"""
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    prompt = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    return prompt

def benchmark_throughput(
    base_url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_requests: int
) -> float:
    """Benchmark the model and return throughput in tokens/second"""
    print(f"\nBenchmarking with {num_requests} requests...")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print("")
    
    # Load tokenizer and generate random prompt
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt = generate_random_prompt(tokenizer, input_tokens)
    
    total_tokens = 0
    total_time = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": output_tokens,
                "temperature": 0.0,
                "stop": [],
                "ignore_eos": True,
            },
            timeout=120
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            tokens_generated = result['usage']['completion_tokens']
            tokens_input = result['usage']['prompt_tokens']
            total_tokens += tokens_generated
            total_time += (end_time - start_time)
            
            print(f"  Request {i + 1}: Input tokens: {tokens_input}, Output tokens: {tokens_generated}")
        else:
            print(f"  Warning: Request {i + 1} failed with status {response.status_code}")
    
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
    input_tokens = int(os.environ.get('INPUT_TOKENS'))
    output_tokens = int(os.environ.get('OUTPUT_TOKENS'))
    num_requests = int(os.environ.get('NUM_REQUESTS'))
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
    throughput = benchmark_throughput(
        base_url=base_url,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_requests=num_requests
    )
    
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
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_requests": num_requests,
        },
        "pricing": pricing
    }
    
    with open('/workspace/pricing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: pricing_results.json")

if __name__ == "__main__":
    main()