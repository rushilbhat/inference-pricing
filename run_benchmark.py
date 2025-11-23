#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import config


@contextmanager
def running_vllm_server(quantization, tp):
    model_path = config.QUANTIZATION_MAP[quantization]
    print(f"[run_benchmark] Launching vLLM server for {model_path} with TP={tp}")

    cmd = [
        "vllm", "serve",
        model_path,
        "--port", str(config.PORT),
        "--tensor-parallel-size", str(tp),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print(f"[run_benchmark] vLLM server PID={proc.pid}")

    try:
        yield proc
    finally:
        print("[run_benchmark] Stopping vLLM server...")
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM pricing benchmark")
    parser.add_argument("quantization", choices=config.QUANTIZATION_MAP.keys())
    parser.add_argument("tp", type=int, default=1)
    args = parser.parse_args()

    print(
        f"\nModel family: {config.MODEL_FAMILY} | Quantization: {args.quantization} | TP: {args.tp}"
    )

    env = os.environ.copy()
    env["QUANTIZATION"] = args.quantization
    env["TP"] = str(args.tp)

    with running_vllm_server(args.quantization, args.tp):
        time.sleep(5)
        rc = subprocess.call([sys.executable, "benchmark_client.py"], env=env)

    sys.exit(rc)

if __name__ == "__main__":
    main()