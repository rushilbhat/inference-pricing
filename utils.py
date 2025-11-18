import json
import time
import requests
import numpy as np


def wait_for_server(base_url, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM server did not start within timeout period")


def load_wildchat_workload(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    avg_isl = obj["avg_isl"]
    avg_osl = obj["avg_osl"]
    mean_gap_seconds = obj["mean_gap_seconds"]
    arrival_rate_per_second = obj["arrival_rate_per_second"]

    return avg_isl, avg_osl, mean_gap_seconds, arrival_rate_per_second


def bin_midpoints(edges):
    return [(edges[i] + edges[i + 1]) // 2 for i in range(len(edges) - 1)]


def generate_random_prompt(tokenizer, num_tokens):
    vocab_size = tokenizer.vocab_size
    random_token_ids = np.random.randint(0, vocab_size, size=num_tokens)
    return tokenizer.decode(random_token_ids, skip_special_tokens=True)
