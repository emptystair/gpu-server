#!/bin/bash

# Docker-based test runner for GPU OCR Server
# This script handles building and running tests in Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
MAX_FILES=""
BUILD_IMAGE=false
KEEP_RUNNING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build           Build Docker image before running tests"
            echo "  --test-type TYPE  Type of test: all, direct, api, verify (default: all)"
            echo "  --max-files N     Maximum number of files to test"
            echo "  --keep-running    Keep containers running after tests"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build Docker image
build_image() {
    print_info "Building Docker image..."
    docker compose -f docker-compose.test.yml build
    print_success "Docker image built successfully"
}

# Function to cleanup containers
cleanup() {
    if [ "$KEEP_RUNNING" = false ]; then
        print_info "Cleaning up containers..."
        docker compose -f docker-compose.test.yml down -v
    else
        print_info "Keeping containers running (use 'docker compose -f docker-compose.test.yml down' to stop)"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Function to run verification
run_verification() {
    print_info "Running setup verification..."
    docker compose -f docker-compose.test.yml run --rm test-runner python tests/verify_test_setup.py
}

# Function to run direct OCR tests
run_direct_tests() {
    print_info "Running direct OCR service tests..."
    
    CMD="python tests/test_pdf_comprehensive.py"
    if [ ! -z "$MAX_FILES" ]; then
        CMD="$CMD --max-files $MAX_FILES"
    fi
    
    docker compose -f docker-compose.test.yml run --rm direct-test $CMD
}

# Function to run API tests
run_api_tests() {
    print_info "Starting OCR server for API tests..."
    
    # Start the server
    docker compose -f docker-compose.test.yml up -d ocr-server
    
    # Wait for server to be healthy
    print_info "Waiting for server to be ready..."
    for i in {1..30}; do
        if docker compose -f docker-compose.test.yml exec ocr-server curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            print_success "Server is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo
    
    # Run API tests
    print_info "Running API tests..."
    CMD="python tests/test_api_pdf_processing.py --server http://ocr-server:8000"
    if [ ! -z "$MAX_FILES" ]; then
        CMD="$CMD --max-files $MAX_FILES"
    fi
    
    docker compose -f docker-compose.test.yml run --rm test-runner $CMD
}

# Function to run all tests
run_all_tests() {
    run_verification
    echo
    run_direct_tests
    echo
    run_api_tests
}

# Function to copy test results
copy_results() {
    print_info "Copying test results..."
    
    # Create results directory
    mkdir -p test-results
    
    # Copy results from containers
    docker cp ocr-test-runner:/app/test_results_comprehensive.json ./test-results/ 2>/dev/null || true
    docker cp ocr-test-runner:/app/test_api_results.json ./test-results/ 2>/dev/null || true
    
    if [ -f "./test-results/test_results_comprehensive.json" ] || [ -f "./test-results/test_api_results.json" ]; then
        print_success "Test results copied to ./test-results/"
    fi
}

# Main execution
echo -e "${GREEN}=== GPU OCR Server Docker Test Suite ===${NC}"
echo -e "Test type: $TEST_TYPE"

# Build image if requested
if [ "$BUILD_IMAGE" = true ]; then
    build_image
fi

# Check if image exists
if ! docker images | grep -q "gpu-ocr-server.*test"; then
    print_warning "Docker image not found. Building..."
    build_image
fi

# Run tests based on type
case $TEST_TYPE in
    verify)
        run_verification
        ;;
    direct)
        run_direct_tests
        copy_results
        ;;
    api)
        run_api_tests
        copy_results
        ;;
    all)
        run_all_tests
        copy_results
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

print_success "Tests completed!"

# Show results summary
if [ -d "./test-results" ]; then
    echo -e "\n${YELLOW}Test results available in:${NC}"
    ls -la ./test-results/*.json 2>/dev/null || echo "  No results files found"
fi