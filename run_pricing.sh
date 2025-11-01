#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2.5-0.5B"
GPU_TYPE="nvidia_a100_40gb"
GPU_COST=2.50
INPUT_TOKENS=512
OUTPUT_TOKENS=512
NUM_REQUESTS=20
PORT=8000

echo "=== vLLM Pricing Engine ==="
echo "Model: $MODEL"
echo "GPU: $GPU_TYPE (\$$GPU_COST/hour)"
echo ""

# Start server
sudo docker run -d --name pricing-server \
    --runtime nvidia --gpus all \
    --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model $MODEL --port $PORT

# Wait for server
echo "Waiting for vLLM server..."
until curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; do
    sleep 2
done
echo "✓ Server ready!"

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
    -e PORT="$PORT" \
    python:3.12-slim \
    bash -c "pip install -q requests && python3 pricing_calculator.py"

# Cleanup
echo ""
echo "Cleaning up..."
sudo docker stop pricing-server >/dev/null 2>&1
sudo docker rm pricing-server >/dev/null 2>&1
echo "Done!"