#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2.5-0.5B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
INPUT_TOKENS=1024
OUTPUT_TOKENS=1024
NUM_REQUESTS=30
ARRIVAL_RATE=0.0
ARRIVAL_BURSTINESS=1.0
PORT=8000

# Start server
sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model $MODEL --port $PORT

# Run client (minimal Python image, no GPU needed!)
sudo docker run --rm --name pricing-client \
    --network host \
    -v $(pwd):/workspace \
    -w /workspace \
    -e MODEL="$MODEL" \
    -e GPU_TYPE="$GPU_TYPE" \
    -e GPU_COST="$GPU_COST" \
    -e INPUT_TOKENS="$INPUT_TOKENS" \
    -e OUTPUT_TOKENS="$OUTPUT_TOKENS" \
    -e NUM_REQUESTS="$NUM_REQUESTS" \
    -e ARRIVAL_RATE="$ARRIVAL_RATE" \
    -e ARRIVAL_BURSTINESS="$ARRIVAL_BURSTINESS" \
    -e PORT="$PORT" \
    python:3.12-slim \
    bash -c "pip install -q requests transformers numpy aiohttp && python3 pricing_calculator.py"

# Cleanup
echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"