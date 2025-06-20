#!/bin/bash
# Quick cache clear script for OCR server

echo "Quick OCR Cache Clear"
echo "===================="

# Clear Redis cache from container
echo -e "\n1. Clearing Redis cache..."
docker exec gpu-ocr-server python3 -c "
import redis
r = redis.Redis(host='ocr-redis', port=6379, db=0)
keys = 0
for key in r.scan_iter(match='*'):
    r.delete(key)
    keys += 1
print(f'Deleted {keys} Redis keys')
" 2>/dev/null || echo "Failed to clear Redis from container"

# Try from host as well
if command -v redis-cli &> /dev/null; then
    redis-cli --scan --pattern "*ocr*" | xargs -r redis-cli del 2>/dev/null
    echo "Cleared Redis cache from host"
fi

# Clear disk cache in container
echo -e "\n2. Clearing disk cache..."
docker exec gpu-ocr-server sh -c "rm -rf /app/cache/* /app/cache/.cache_metadata.json 2>/dev/null" && \
    echo "Cleared container disk cache" || \
    echo "Failed to clear container disk cache"

# Clear local cache
if [ -d "./cache" ]; then
    rm -rf ./cache/*
    echo "Cleared local disk cache"
fi

# Restart service to clear memory cache
echo -e "\n3. Restarting service to clear memory cache..."
docker compose restart gpu-ocr-server

# Wait for service to be ready
echo -e "\nWaiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health | grep -q "healthy"; then
        echo "Service is ready!"
        break
    fi
    sleep 1
done

echo -e "\nCache clearing complete!"