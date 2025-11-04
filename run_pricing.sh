#!/bin/bash

MODEL="Qwen/Qwen2.5-0.5B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
NUM_REQUESTS=30
PORT=8000

WORKLOAD_MODE="wildchat"   # or "generic"
JOINT_PROBS_PATH="/workspace/isl_osl_jointprobs_first_1000.json"
ARRIVAL_JSON_PATH="/workspace/wildchat_arrival_first_1000.json"

sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model $MODEL \
    --port $PORT \
    --override-generation-config '{"max_new_tokens": 8192}'


sudo docker run --rm --name pricing-client \
    --network host \
    -v $(pwd):/workspace \
    -w /workspace \
    -e MODEL="$MODEL" \
    -e GPU_TYPE="$GPU_TYPE" \
    -e GPU_COST="$GPU_COST" \
    -e NUM_REQUESTS="$NUM_REQUESTS" \
    -e PORT="$PORT" \
    -e WORKLOAD_MODE="$WORKLOAD_MODE" \
    -e JOINT_PROBS_PATH="$JOINT_PROBS_PATH" \
    -e ARRIVAL_JSON_PATH="$ARRIVAL_JSON_PATH" \
    python:3.12-slim \
    bash -c "pip install -q requests transformers numpy aiohttp && python3 pricing_calculator.py"

echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"