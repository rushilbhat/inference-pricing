#!/usr/bin/env python3
import requests
import time
import yaml
import json
from typing import Dict, List

def load_config() -> Dict:
    """Load configuration from config.yml"""
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

def wait_for_server(base_url: str = "http://localhost:8000", timeout: int = 300):
    """Wait for vLLM server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✓ vLLM server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM server did not start within timeout period")

def benchmark_throughput(
    base_url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_requests: int
) -> float:
    """
    Benchmark the model and return throughput in tokens/second
    """
    print(f"\nBenchmarking with {num_requests} requests...")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print("")
    
    # Create a prompt that's approximately the right length
    # Rough estimate: 1 token ≈ 4 characters
    prompt = "Explain the concept of machine learning. " * (input_tokens // 10)
    
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
            },
            timeout=120
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            tokens_generated = result['usage']['completion_tokens']
            total_tokens += tokens_generated
            total_time += (end_time - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests...")
        else:
            print(f"  Warning: Request {i + 1} failed with status {response.status_code}")
    
    throughput = total_tokens / total_time
    return throughput

def calculate_pricing(throughput: float, gpu_cost_per_hour: float) -> Dict[str, float]:
    """
    Calculate pricing based on throughput and GPU cost
    
    Formula: Cost per token = GPU hourly cost / (throughput * 3600 seconds)
    """
    tokens_per_hour = throughput * 3600
    cost_per_token = gpu_cost_per_hour / tokens_per_hour
    cost_per_1k_tokens = cost_per_token * 1000
    cost_per_1m_tokens = cost_per_token * 1000000
    
    return {
        "throughput_tokens_per_sec": throughput,
        "tokens_per_hour": tokens_per_hour,
        "cost_per_token": cost_per_token,
        "cost_per_1k_tokens": cost_per_1k_tokens,
        "cost_per_1m_tokens": cost_per_1m_tokens,
    }

def main():
    # Load configuration
    config = load_config()
    
    model = config['model']
    input_tokens = config['input_tokens']
    output_tokens = config['output_tokens']
    num_requests = config['num_requests']
    gpu_type = config['gpu_type']
    gpu_cost = config['gpu_costs'][gpu_type]
    
    base_url = "http://localhost:8000"
    
    # Wait for server to be ready
    wait_for_server(base_url)
    
    # Benchmark throughput
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"GPU: {gpu_type} (${gpu_cost}/hour)")
    print(f"{'='*60}")
    
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
    
    with open('pricing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: pricing_results.json")

if __name__ == "__main__":
    main()