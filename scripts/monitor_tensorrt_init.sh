#!/bin/bash
# Monitor TensorRT initialization progress

echo "=== Monitoring TensorRT Initialization ==="
echo "Started at: $(date)"
echo

# Function to check server status
check_status() {
    # Check if server is responding
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✓ Server is ready and responding!"
        return 0
    fi
    return 1
}

# Monitor loop
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    echo -n "[$elapsed seconds] "
    
    # Check container status
    container_status=$(docker ps --filter "name=gpu-ocr-server" --format "{{.Status}}")
    echo -n "Container: $container_status | "
    
    # Check GPU memory usage
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo -n "GPU Memory: ${gpu_mem}MB | "
    
    # Check for engine files
    engine_count=$(docker exec gpu-ocr-server find /root/.paddleocr -name "*.engine" 2>/dev/null | wc -l)
    echo -n "Engine files: $engine_count | "
    
    # Check if server is ready
    if check_status; then
        echo
        echo "=== Initialization Complete ==="
        echo "Total time: $elapsed seconds"
        
        # Show engine files if any
        echo
        echo "Engine files created:"
        docker exec gpu-ocr-server find /root/.paddleocr -name "*.engine" -exec ls -lh {} \; 2>/dev/null
        
        # Show shape files
        echo
        echo "Shape files:"
        docker exec gpu-ocr-server find /root/.paddleocr -name "*shape*.txt" -exec ls -lh {} \; 2>/dev/null
        
        break
    else
        echo "Initializing..."
    fi
    
    # Check every 10 seconds
    sleep 10
done

echo
echo "=== Testing Server ==="
# Test with a simple health check
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool