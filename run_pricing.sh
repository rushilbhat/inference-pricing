#!/bin/bash

MODEL="Qwen/Qwen2.5-0.5B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
NUM_REQUESTS=30
PORT=8000

sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --port "$PORT" \
    --override-generation-config '{"max_new_tokens": 8192}'

env_args=(
    -e MODEL="$MODEL"
    -e GPU_TYPE="$GPU_TYPE"
    -e GPU_COST="$GPU_COST"
    -e NUM_REQUESTS="$NUM_REQUESTS"
    -e PORT="$PORT"
)

WORKLOAD_STATS_PATH="/workspace/wildchat_workload_~first_1000.json"
WORKLOAD_CALIB_PATH="/workspace/wildchat_calibration_~first_1000.json"

env_args+=(
    -e WORKLOAD_STATS_PATH="$WORKLOAD_STATS_PATH"
    -e WORKLOAD_CALIB_PATH="$WORKLOAD_CALIB_PATH"
)

CLIENT_CMD="pip install -q requests transformers numpy aiohttp datasets; "

CLIENT_CMD+="python3 process_wildchat.py; "
CLIENT_CMD+="python3 calibrate_workload.py; "
CLIENT_CMD+="python3 pricing_calculator.py"

sudo docker run --rm --name pricing-client \
    --network host \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "${env_args[@]}" \
    python:3.12-slim \
    bash -c "$CLIENT_CMD"

echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"
