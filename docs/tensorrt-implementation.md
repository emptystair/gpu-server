# TensorRT Implementation for GPU OCR Server

## Overview
This document describes the TensorRT implementation for the GPU OCR Server to accelerate OCR processing on the RTX 4090.

## What We've Implemented

### 1. Docker Integration
- **Updated Dockerfile**: Added TensorRT installation from NVIDIA's PyPI index
- **Environment Variables**: Configured TensorRT flags for optimal performance
- **Verification Step**: Added TensorRT version check during build

### 2. Configuration Updates
- **Enabled TensorRT by default**: Set `use_tensorrt: bool = True` in OCRConfig
- **Environment variable support**: `USE_TENSORRT=true` in Docker
- **FP16 precision mode**: Optimized for RTX 4090's tensor cores

### 3. Existing TensorRT Infrastructure
The codebase already includes comprehensive TensorRT support in `src/models/tensorrt_optimizer.py`:

- **Model Conversion Pipeline**: PaddlePaddle → ONNX → TensorRT
- **FP16 Optimization**: Leverages RTX 4090's tensor cores
- **Dynamic Batching**: Supports batch sizes from 1 to 50
- **Model Caching**: Avoids repeated optimization
- **Performance Validation**: Ensures accuracy is maintained

### 4. Scripts and Tools
- **run_docker_tensorrt.sh**: Build and run Docker container with TensorRT
- **test_tensorrt_performance.py**: Compare performance with/without TensorRT

## Expected Performance Improvements

Based on the TensorRT optimizer implementation, you can expect:
- **3-5x speedup** for inference
- **FP16 precision** with minimal accuracy loss (<1%)
- **Better GPU utilization** through optimized kernels
- **Reduced memory usage** through kernel fusion

## How to Use

### 1. Build and Run with TensorRT
```bash
# Build and run the Docker container
./run_docker_tensorrt.sh

# Or use docker-compose
docker-compose up --build
```

### 2. Verify TensorRT is Active
Check the logs for TensorRT initialization:
```bash
docker logs gpu-ocr-server | grep -i tensorrt
```

### 3. Test Performance
Run the performance tests:
```bash
# Basic TensorRT test
python tests/performance_tests/test_tensorrt_performance.py

# Real document test
python tests/performance_tests/test_real_tensorrt.py

# Full batch test (96 PDFs)
python tests/performance_tests/test_tensorrt_final.py
```

## Configuration Options

### Environment Variables
- `USE_TENSORRT=true`: Enable/disable TensorRT
- `FLAGS_tensorrt_precision_mode=FP16`: Set precision (FP16/FP32/INT8)
- `FLAGS_tensorrt_workspace_size=4096`: TensorRT workspace size in MB
- `FLAGS_tensorrt_max_batch_size=50`: Maximum batch size

### In Code
```python
config = Config(
    ocr=OCRConfig(
        use_tensorrt=True,
        tensorrt_precision="FP16"
    )
)
```

## Troubleshooting

### TensorRT Not Found
If you see "TensorRT not available" in logs:
1. Ensure you're using the updated Dockerfile
2. Check CUDA compatibility (requires CUDA 12.x)
3. Verify the container has GPU access

### Performance Not Improved
If performance doesn't improve:
1. Check if TensorRT optimization completed (first run takes longer)
2. Verify GPU memory is available (needs ~4GB for optimization)
3. Check batch sizes - larger batches benefit more from TensorRT

### Model Conversion Fails
If PaddlePaddle to TensorRT conversion fails:
1. Ensure paddle2onnx is installed
2. Check model paths are correct
3. Verify ONNX conversion succeeds before TensorRT

## Next Steps

### Further Optimizations
1. **INT8 Quantization**: Even faster inference with calibration
2. **Custom TensorRT Plugins**: For PaddleOCR-specific operations
3. **Multi-GPU Support**: Scale across multiple RTX 4090s
4. **Profile-Guided Optimization**: Tune for specific workloads

### Monitoring
- GPU utilization should increase with TensorRT
- Memory usage may spike during optimization
- First inference is slower (optimization time)
- Subsequent runs use cached engines