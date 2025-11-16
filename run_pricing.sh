#!/bin/bash

MODEL="Qwen/Qwen2.5-0.5B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
NUM_REQUESTS=30
PORT=8000

WORKLOAD_MODE="wildchat"   # or "generic"

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
    -e WORKLOAD_MODE="$WORKLOAD_MODE"
)

CLIENT_CMD="pip install -q requests transformers numpy aiohttp datasets; "

if [[ "$WORKLOAD_MODE" == "wildchat" ]]; then
    WORKLOAD_STATS_PATH="/workspace/wildchat_workload_~first_1000.json"
    env_args+=(-e WORKLOAD_STATS_PATH="$WORKLOAD_STATS_PATH")
    CLIENT_CMD+="python3 process_wildchat.py \"$WORKLOAD_STATS_PATH\"; "
else
    CLIENT_CMD+="echo 'Skipping process_wildchat.py (generic workload)'; "
fi

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
