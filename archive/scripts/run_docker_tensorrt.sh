#!/bin/bash

# Build and run Docker container with TensorRT support

echo "Building GPU OCR Server with TensorRT support..."

# Build the Docker image
docker build -t gpu-ocr-tensorrt:latest .

if [ $? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

echo "Build successful! Starting container..."

# Stop any existing container
docker stop gpu-ocr-tensorrt 2>/dev/null
docker rm gpu-ocr-tensorrt 2>/dev/null

# Run the container with GPU support
docker run -d \
    --name gpu-ocr-tensorrt \
    --runtime=nvidia \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/src:/app/src:ro \
    -v $(pwd)/model_cache:/app/model_cache \
    -v $(pwd)/cache:/app/cache \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/testpdfs:/app/testpdfs:ro \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e PYTHONUNBUFFERED=1 \
    -e USE_TENSORRT=true \
    -e FLAGS_use_tensorrt=1 \
    -e FLAGS_tensorrt_precision_mode=FP16 \
    -e FLAGS_tensorrt_workspace_size=4096 \
    gpu-ocr-tensorrt:latest

if [ $? -eq 0 ]; then
    echo "Container started successfully!"
    echo "Waiting for service to initialize..."
    sleep 10
    
    # Check if service is healthy
    echo "Checking service health..."
    curl -s http://localhost:8000/health | jq .
    
    echo ""
    echo "View logs with: docker logs -f gpu-ocr-tensorrt"
    echo "Stop with: docker stop gpu-ocr-tensorrt"
else
    echo "Failed to start container!"
    exit 1
fi