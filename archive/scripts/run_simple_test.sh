#!/bin/bash

# Simple test runner using the production Docker setup

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== GPU OCR Server - Simple PDF Test ===${NC}"
echo

# Check if server is already running
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is already running${NC}"
    SERVER_RUNNING=true
else
    echo -e "${YELLOW}Server is not running. Starting with docker compose...${NC}"
    
    # Use the standard docker-compose
    docker compose up -d
    
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server started successfully${NC}"
            SERVER_RUNNING=false
            break
        fi
        echo -n "."
        sleep 2
    done
    echo
fi

# Run a simple test
echo -e "\n${YELLOW}Testing PDF processing...${NC}"

# Pick first PDF from the folder
PDF_FILE=$(ls /home/ryanb/Projects/gpu-server0.1/tests/testpdfs/*.pdf | head -1)
PDF_NAME=$(basename "$PDF_FILE")

echo "Testing with: $PDF_NAME"

# Test the API
echo -e "\n${YELLOW}Sending OCR request...${NC}"
curl -X POST http://localhost:8000/api/v1/ocr/pdf \
  -F "file=@$PDF_FILE" \
  -F "language=en" \
  -F "output_format=json" \
  -F "merge_pages=true" \
  -o test_result.json \
  -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"

# Check result
if [ -f test_result.json ]; then
    echo -e "\n${GREEN}✓ Test completed successfully${NC}"
    echo "Result saved to: test_result.json"
    
    # Show summary
    if command -v jq > /dev/null 2>&1; then
        echo -e "\n${YELLOW}Result summary:${NC}"
        jq '{
            request_id: .request_id,
            status: .status,
            pages: (.pages | length),
            processing_time: .processing_time,
            average_confidence: .average_confidence
        }' test_result.json
    else
        echo -e "\n${YELLOW}First 500 characters of result:${NC}"
        head -c 500 test_result.json
        echo "..."
    fi
else
    echo -e "\n${RED}✗ Test failed - no result file created${NC}"
fi

# Check GPU status
echo -e "\n${YELLOW}GPU Status:${NC}"
curl -s http://localhost:8000/api/v1/gpu/status | jq '.' 2>/dev/null || echo "Unable to get GPU status"

# Cleanup option
if [ "$SERVER_RUNNING" = false ]; then
    echo -e "\n${YELLOW}Server was started for this test.${NC}"
    echo "To stop it: docker compose down"
fi

echo -e "\n${GREEN}=== Test Complete ===${NC}"