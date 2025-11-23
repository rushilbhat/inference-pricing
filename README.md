# LLM Inference Pricing Calculator

## 1. Installation

1.  Set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Usage

To run the benchmark and generate a price, execute the runner script:

**Command:**
```bash
python3 run_benchmark.py [quantization] [tp_size]
```

*   `quantization`: e.g., `base`, `w8a8`, `w4a16`, `w8a16` (Configured in `config.py`)
*   `tp_size`: No. parallel workers.

**Example:**
```bash
python3 run_benchmark.py w4a16 2
```

### Output & Pricing
The script outputs a pricing matrix to the console. It calculates the cost based on the server hourly rate divided by the maximum throughput achievable without violating latency SLOs wrt TTFT and TPOT.

## 3. Methodology

*   **Traffic Profiling (`wildchat/traffic_profile.py`):** The profiling script analyzes WildChat dataset traffic from March 17–23, 2024 to determine average input and output sequence lengths (ISL/OSL) for realistic benchmarking prompts.
    *   *Note:* A cached file is provided (`wildchat/cache.jsonl`) to bypass the process of downloading and filtering the full dataset.
*   **Benchmarking (`benchmark_client.py`):** The client executes a sweep of increasing batch sizes to measure throughput and latency under varying levels of concurrency.

## 4. Results & Visualization

Llama-3.1-8B-Instruct was benchmarked on an NVIDIA DGX Station A100 (4x 80GB PCIe), hosted by DataCrunch via PrimeIntellect. Results are stored in the `benchmark_results/` folder.

The optimal configuration was `w8a8` quantization with `4`-way TP. Based on the strictest SLO adherence (TTFT < 0.25s, TPOT < 10ms), the recommended pricing is:

*   **Input Price:** $0.2890 / 1M tokens
*   **Output Price:** $1.2144 / 1M tokens

To generate a plot of the throughput and latency metrics from the stored logs:

```bash
python3 plot_results.py benchmark_results/example_Meta-Llama-3.1-8B-Instruct_NVIDIA_DGX_Station_A100_80GB.jsonl
```