#!/bin/bash
set -e

echo "=== vLLM Pricing Engine ==="
echo ""

# Read model from config
MODEL=$(grep "model:" config.yml | awk '{print $2}' | tr -d '"')

echo "Starting vLLM server with model: $MODEL"
echo ""

# Start vLLM server in background
sudo docker run -d --name pricing-vllm \
    --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model $MODEL

echo "Waiting for vLLM server to be ready..."
echo ""

# Run the pricing calculator (it will wait for server and benchmark)
python3 pricing_calculator.py

# Cleanup
echo ""
echo "Cleaning up..."
docker stop pricing-vllm >/dev/null 2>&1
docker rm pricing-vllm >/dev/null 2>&1

echo "Done!"