#!/bin/bash

# Direct Docker run for GPU OCR Server testing

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== GPU OCR Server - Direct Docker Test ===${NC}"
echo

# Check if server is already running on port 8000
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${RED}Port 8000 is already in use. Please stop the existing server.${NC}"
    exit 1
fi

# Run the server directly with Docker
echo -e "${YELLOW}Starting OCR server with Docker...${NC}"
docker run -d \
  --name gpu-ocr-test \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DEFAULT_DPI=150 \
  -e LOG_LEVEL=INFO \
  gpu-ocr-server:latest

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server started successfully${NC}"
        break
    fi
    echo -n "."
    sleep 2
done
echo

# Check server health
echo -e "\n${YELLOW}Server Health Check:${NC}"
curl -s http://localhost:8000/api/v1/health | jq '.' 2>/dev/null || curl http://localhost:8000/api/v1/health

# Run PDF test
echo -e "\n${YELLOW}Testing PDF processing...${NC}"

# Pick first PDF
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
if [ -f test_result.json ] && [ -s test_result.json ]; then
    echo -e "\n${GREEN}✓ Test completed successfully${NC}"
    echo "Result saved to: test_result.json"
    
    # Show summary if jq available
    if command -v jq > /dev/null 2>&1; then
        echo -e "\n${YELLOW}Result summary:${NC}"
        jq '{
            request_id: .request_id,
            status: .status,
            pages: (.pages | length),
            processing_time: .processing_time,
            average_confidence: .average_confidence
        }' test_result.json 2>/dev/null || echo "Could not parse JSON result"
    else
        echo -e "\n${YELLOW}First 500 characters of result:${NC}"
        head -c 500 test_result.json
        echo "..."
    fi
else
    echo -e "\n${RED}✗ Test failed - no valid result${NC}"
fi

# Check GPU status
echo -e "\n${YELLOW}GPU Status:${NC}"
curl -s http://localhost:8000/api/v1/gpu/status | jq '.' 2>/dev/null || echo "Unable to get GPU status"

# Show logs
echo -e "\n${YELLOW}Recent server logs:${NC}"
docker logs --tail 20 gpu-ocr-test

# Cleanup
echo -e "\n${YELLOW}Stopping test server...${NC}"
docker stop gpu-ocr-test
docker rm gpu-ocr-test

echo -e "\n${GREEN}=== Test Complete ===${NC}"