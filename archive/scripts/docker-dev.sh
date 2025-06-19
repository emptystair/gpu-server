#!/bin/bash

# GPU OCR Server Docker Development Script
# Provides convenient commands for building, running, and testing the Docker container

set -e

# Configuration
PROJECT_NAME="gpu-ocr-server"
IMAGE_NAME="gpu-ocr-server:latest"
CONTAINER_NAME="gpu-ocr-server"
NETWORK_NAME="gpu-server01_ocr-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if nvidia-docker is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_error "NVIDIA Docker runtime not available. Please install nvidia-docker2."
        print_info "Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
}

# Build the Docker image
build() {
    print_info "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME .
    print_success "Docker image built successfully"
}

# Run GPU test
gpu_test() {
    print_info "Testing GPU access in Docker container..."
    check_nvidia_docker
    
    docker run --rm \
        --gpus all \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        $IMAGE_NAME \
        python -c "
import paddle
print('PaddlePaddle version:', paddle.__version__)
print('CUDA available:', paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print('GPU count:', paddle.device.cuda.device_count())
    print('GPU 0 name:', paddle.device.cuda.get_device_name(0))
    print('GPU 0 memory:', paddle.device.cuda.get_device_properties(0))
"
    print_success "GPU test completed"
}

# Run in development mode
dev() {
    print_info "Starting in development mode..."
    
    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Create network if it doesn't exist
    docker network create $NETWORK_NAME 2>/dev/null || true
    
    # Start Redis first
    print_info "Starting Redis container..."
    docker run -d \
        --name ocr-redis \
        --network $NETWORK_NAME \
        --restart unless-stopped \
        -v redis-data:/data \
        redis:7-alpine \
        redis-server --appendonly yes || true
    
    # Run the GPU OCR server
    print_info "Starting GPU OCR server..."
    docker run -d \
        --name $CONTAINER_NAME \
        --gpus all \
        --runtime=nvidia \
        --ipc=host \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864:67108864 \
        --network $NETWORK_NAME \
        -p 8000:8000 \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e NVIDIA_DISABLE_REQUIRE=true \
        -e HOST=0.0.0.0 \
        -e PORT=8000 \
        -e DEFAULT_DPI=120 \
        -e MAX_BATCH_SIZE=50 \
        -e TENSORRT_PRECISION=FP16 \
        -e GPU_MEMORY_BUFFER_MB=500 \
        -e PYTHONUNBUFFERED=1 \
        -e LOG_LEVEL=INFO \
        -e ENVIRONMENT=development \
        -v $(pwd)/src:/app/src:ro \
        -v $(pwd)/tests:/app/tests:ro \
        -v ocr-cache:/app/cache \
        -v ocr-models:/app/model_cache \
        -v ocr-logs:/app/logs \
        -v ocr-uploads:/app/uploads \
        --shm-size=8g \
        --restart unless-stopped \
        $IMAGE_NAME
    
    print_success "Development server started"
    print_info "Server available at: http://localhost:8000"
    print_info "API docs available at: http://localhost:8000/docs"
    print_info "View logs: docker logs -f $CONTAINER_NAME"
}

# Run with Docker Compose
compose_up() {
    print_info "Starting with Docker Compose..."
    docker-compose up -d
    print_success "Services started with Docker Compose"
    print_info "Server available at: http://localhost:8000"
    print_info "API docs available at: http://localhost:8000/docs"
}

# Stop all services
stop() {
    print_info "Stopping services..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker stop ocr-redis 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    print_success "Services stopped"
}

# View logs
logs() {
    docker logs -f $CONTAINER_NAME
}

# Run tests inside container
test() {
    print_info "Running tests inside container..."
    docker exec -it $CONTAINER_NAME python /app/tests/test_paddle_ocr.py
}

# Shell into container
shell() {
    print_info "Opening shell in container..."
    docker exec -it $CONTAINER_NAME /bin/bash
}

# Clean up containers and volumes
clean() {
    print_warning "This will remove containers and volumes. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Cleaning up..."
        docker-compose down -v 2>/dev/null || true
        docker rm -f $CONTAINER_NAME ocr-redis 2>/dev/null || true
        docker volume rm ocr-cache ocr-models ocr-logs ocr-uploads redis-data 2>/dev/null || true
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Show help
help() {
    echo "GPU OCR Server Docker Development Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  gpu         Test GPU access inside container"
    echo "  dev         Run in development mode with volume mounts"
    echo "  compose     Start with Docker Compose"
    echo "  stop        Stop all services"
    echo "  logs        View container logs"
    echo "  test        Run tests inside container"
    echo "  shell       Open shell in container"
    echo "  clean       Remove containers and volumes"
    echo "  help        Show this help message"
}

# Main script
check_docker

case "$1" in
    build)
        build
        ;;
    gpu)
        gpu_test
        ;;
    dev)
        dev
        ;;
    compose)
        compose_up
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    test)
        test
        ;;
    shell)
        shell
        ;;
    clean)
        clean
        ;;
    help|"")
        help
        ;;
    *)
        print_error "Unknown command: $1"
        help
        exit 1
        ;;
esac