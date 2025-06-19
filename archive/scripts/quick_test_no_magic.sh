#!/bin/bash

# Quick test without python-magic dependency

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Quick PDF Test (No Magic) ===${NC}"

# Create a temporary patch for ocr_service.py
echo -e "${YELLOW}Patching ocr_service.py to remove magic dependency...${NC}"

# Backup original
cp src/ocr_service.py src/ocr_service.py.bak

# Create patched version that doesn't use magic
cat > src/ocr_service_patch.py << 'EOF'
# Temporary patch to remove magic dependency
import os
import sys

# Mock magic module
class MockMagic:
    def from_buffer(self, data, mime=False):
        # Simple mime type detection based on file header
        if data.startswith(b'%PDF'):
            return 'application/pdf' if mime else 'PDF document'
        elif data.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg' if mime else 'JPEG image'
        elif data.startswith(b'\x89PNG'):
            return 'image/png' if mime else 'PNG image'
        else:
            return 'application/octet-stream' if mime else 'data'

sys.modules['magic'] = MockMagic()

# Now import the rest
EOF

# Prepend the patch to ocr_service.py
cat src/ocr_service_patch.py > src/ocr_service_temp.py
tail -n +2 src/ocr_service.py >> src/ocr_service_temp.py
mv src/ocr_service_temp.py src/ocr_service.py

# Run the server
echo -e "${YELLOW}Starting server with patched code...${NC}"
docker run -d \
  --name gpu-ocr-test \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  -e NVIDIA_VISIBLE_DEVICES=all \
  gpu-ocr-server:latest

# Wait for server
echo "Waiting for server..."
sleep 10

# Check if running
if docker ps | grep gpu-ocr-test > /dev/null; then
    echo -e "${GREEN}✓ Server is running${NC}"
    
    # Test health
    echo -e "\n${YELLOW}Testing health endpoint...${NC}"
    curl -s http://localhost:8000/api/v1/health | head -20
    
    # Test PDF
    echo -e "\n${YELLOW}Testing PDF processing...${NC}"
    PDF_FILE=$(ls tests/testpdfs/*.pdf | head -1)
    
    curl -X POST http://localhost:8000/api/v1/ocr/process \
      -F "file=@$PDF_FILE" \
      -F "strategy=speed" \
      -o test_result.json \
      -w "\nHTTP Status: %{http_code}\n"
    
    if [ -f test_result.json ]; then
        echo -e "${GREEN}✓ Got response${NC}"
        echo "First 200 chars:"
        head -c 200 test_result.json
    fi
else
    echo -e "${RED}✗ Server failed to start${NC}"
    echo "Logs:"
    docker logs gpu-ocr-test
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
docker stop gpu-ocr-test 2>/dev/null
docker rm gpu-ocr-test 2>/dev/null

# Restore original file
mv src/ocr_service.py.bak src/ocr_service.py
rm -f src/ocr_service_patch.py

echo -e "\n${GREEN}Done!${NC}"