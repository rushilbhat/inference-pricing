#!/bin/bash

# MODEL="Qwen/Qwen2.5-0.5B"
MODEL="unsloth/Meta-Llama-3.1-8B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
PORT=8000

sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --port "$PORT" \
    --override-generation-config '{"max_new_tokens": 8192}'

CLIENT_CMD="pip install -q requests transformers numpy aiohttp datasets && python3 pricing_calculator.py"

sudo docker run --rm --name pricing-client \
    --network host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e MODEL="$MODEL" -e GPU_TYPE="$GPU_TYPE" -e GPU_COST="$GPU_COST" -e PORT="$PORT" \
    python:3.12-slim \
    bash -c "$CLIENT_CMD"

echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"