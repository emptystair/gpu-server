#!/bin/bash

# Test with CPU-only mode to bypass CUDNN issues

echo "Testing OCR with CPU mode..."

# Test with a simple image request
PDF_FILE=$(ls tests/testpdfs/*.pdf | head -1)
PDF_NAME=$(basename "$PDF_FILE")

echo "Testing PDF: $PDF_NAME"

# Try the basic process endpoint with CPU
curl -X POST http://localhost:8000/api/v1/ocr/process \
  -F "file=@$PDF_FILE" \
  -F "strategy=speed" \
  -F "dpi=120" \
  -o test_cpu_result.json \
  -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" \
  -v

if [ -f test_cpu_result.json ]; then
    echo -e "\nResult preview:"
    head -c 500 test_cpu_result.json
    echo -e "\n\nFull result saved to: test_cpu_result.json"
fi