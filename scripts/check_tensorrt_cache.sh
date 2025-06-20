#!/bin/bash
# Check for TensorRT cache files and directories

echo "=== Checking TensorRT Cache Locations ==="
echo

echo "1. Shape files in /root/.paddleocr:"
docker exec gpu-ocr-server find /root/.paddleocr -name "*shape*.txt" -type f -exec ls -lh {} \;
echo

echo "2. Paddle cache directory:"
docker exec gpu-ocr-server du -sh /root/.cache/paddle 2>/dev/null || echo "Not found"
docker exec gpu-ocr-server find /root/.cache/paddle -type f 2>/dev/null | head -10
echo

echo "3. TensorRT engine files:"
docker exec gpu-ocr-server find / -name "*.engine" -o -name "*.trt" -o -name "*.plan" 2>/dev/null | grep -v "/proc" | head -10
echo

echo "4. Temporary paddle files:"
docker exec gpu-ocr-server find /tmp -name "*paddle*" -o -name "*trt*" 2>/dev/null | head -10
echo

echo "5. Model cache directory:"
docker exec gpu-ocr-server ls -la /app/model_cache/ 2>/dev/null || echo "Not found"
echo

echo "6. Check for any large files created during initialization:"
docker exec gpu-ocr-server find /root /tmp /app -type f -size +10M -mmin -60 2>/dev/null | grep -v ".tar" | head -10
echo

echo "=== Summary ==="
echo "To speed up TensorRT initialization, we should persist:"
echo "- /root/.paddleocr (already done via Docker volume)"
echo "- /root/.cache/paddle (if it contains engine files)"
echo "- Any .engine, .trt, or .plan files found"