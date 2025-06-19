#!/bin/bash

# Install TensorRT for CUDA 12.2 on Ubuntu
# This script installs TensorRT for immediate testing outside Docker

echo "Installing TensorRT for CUDA 12.2..."

# Check CUDA version
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d'.' -f1,2)
echo "Detected CUDA version: $cuda_version"

if [[ "$cuda_version" != "12.2" ]]; then
    echo "Warning: This script is optimized for CUDA 12.2, but found $cuda_version"
fi

# Install TensorRT via pip (simplest method for testing)
echo "Installing TensorRT Python packages..."
pip install nvidia-tensorrt==8.6.1.post1

# Alternative: Install specific TensorRT components
pip install nvidia-cuda-runtime-cu12==12.2.128
pip install nvidia-cudnn-cu12==8.9.7.29
pip install nvidia-tensorrt-bindings==8.6.1.post1
pip install nvidia-tensorrt-libs==8.6.1.post1

# Verify installation
echo "Verifying TensorRT installation..."
python -c "
try:
    import tensorrt as trt
    print(f'TensorRT version: {trt.__version__}')
    print(f'CUDA version supported: {trt.get_cuda_version()}')
    print(f'Supported plugins: {trt.get_plugin_registry().plugin_creator_list}')
    print('TensorRT installation successful!')
except Exception as e:
    print(f'TensorRT installation failed: {e}')
"

echo "Done!"