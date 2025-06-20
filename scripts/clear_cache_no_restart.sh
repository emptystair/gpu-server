#!/bin/bash
# Clear OCR cache without restarting the server
# This avoids the long TensorRT initialization time

echo "Clearing OCR cache (without server restart)..."
echo "============================================"

# Clear Redis and disk cache only, skip memory cache
python /home/ryanb/Projects/gpu-server0.1/scripts/clear_ocr_cache.py --redis --disk --no-restart

echo "Done! Server remains running."