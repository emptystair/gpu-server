#!/bin/bash

# Script to run comprehensive PDF tests
# This script can start the server and run various test suites

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SERVER_PORT=8000
TEST_TYPE="all"
MAX_FILES=""
START_SERVER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-server)
            START_SERVER=true
            shift
            ;;
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="--max-files $2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --start-server    Start the OCR server before running tests"
            echo "  --test-type TYPE  Type of test to run: all, direct, api (default: all)"
            echo "  --max-files N     Maximum number of files to test"
            echo "  --port PORT       Server port (default: 8000)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to check if server is running
check_server() {
    echo -e "${YELLOW}Checking if server is running on port $SERVER_PORT...${NC}"
    if curl -s "http://localhost:$SERVER_PORT/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not running${NC}"
        return 1
    fi
}

# Function to start server
start_server() {
    echo -e "${YELLOW}Starting OCR server...${NC}"
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Start server in background
    python -m uvicorn src.main:app --host 0.0.0.0 --port $SERVER_PORT &
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 5
    
    # Check if server started successfully
    if check_server; then
        echo -e "${GREEN}✓ Server started successfully (PID: $SERVER_PID)${NC}"
    else
        echo -e "${RED}✗ Failed to start server${NC}"
        exit 1
    fi
}

# Function to run direct OCR tests
run_direct_tests() {
    echo -e "\n${YELLOW}Running direct OCR service tests...${NC}"
    python tests/test_pdf_comprehensive.py $MAX_FILES
}

# Function to run API tests
run_api_tests() {
    echo -e "\n${YELLOW}Running API endpoint tests...${NC}"
    python tests/test_api_pdf_processing.py --server "http://localhost:$SERVER_PORT" $MAX_FILES
}

# Function to cleanup
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        echo -e "\n${YELLOW}Stopping server (PID: $SERVER_PID)...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Main execution
echo -e "${GREEN}=== GPU OCR Server PDF Test Suite ===${NC}"
echo -e "Test type: $TEST_TYPE"
echo -e "PDF folder: /home/ryanb/Projects/gpu-server0.1/tests/testpdfs"

# Start server if requested
if [ "$START_SERVER" = true ]; then
    start_server
else
    # Check if server is already running
    if ! check_server; then
        echo -e "${YELLOW}Server is not running. Use --start-server to start it automatically.${NC}"
        exit 1
    fi
fi

# Run tests based on type
case $TEST_TYPE in
    direct)
        run_direct_tests
        ;;
    api)
        run_api_tests
        ;;
    all)
        run_direct_tests
        echo -e "\n${YELLOW}Waiting before API tests...${NC}"
        sleep 2
        run_api_tests
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=== All tests completed ===${NC}"

# Show summary files
echo -e "\n${YELLOW}Test results saved to:${NC}"
[ -f "test_results_comprehensive.json" ] && echo "  - test_results_comprehensive.json"
[ -f "test_api_results.json" ] && echo "  - test_api_results.json"