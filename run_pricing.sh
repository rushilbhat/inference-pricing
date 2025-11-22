#!/bin/bash

MODE=${1:-base}
TP=${2:-1}
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
PORT=8000

TOTAL_COST=$(python3 -c "print(f'{$GPU_COST * $TP:.2f}')")

case "$MODE" in
  "base")  MODEL="unsloth/Meta-Llama-3.1-8B-Instruct" ;;
  "w8a8")  MODEL="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8" ;;
  "w4a16") MODEL="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16" ;;
  *)       echo "Error: Invalid mode. Use [base|w8a8|w4a16]"; exit 1 ;;
esac

echo "Model: $MODEL | Mode: $MODE | TP: $TP | Total Cost: \$$TOTAL_COST/hr"

sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \

CLIENT_CMD="pip install -q requests transformers numpy aiohttp datasets && python3 pricing_calculator.py"

sudo docker run --rm --name pricing-client \
    --network host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e MODEL="$MODEL" -e GPU_TYPE="$GPU_TYPE" -e GPU_COST="$TOTAL_COST" -e PORT="$PORT" \
    python:3.12-slim \
    bash -c "$CLIENT_CMD"

echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"