# Multi-stage build for GPU OCR Server with PaddleOCR and TensorRT
# Optimized for RTX 4090 with CUDA 12.2

# Stage 1: TensorRT and CUDA base
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS tensorrt-base

# Stage 2: Main application with TensorRT
# Using TensorRT container which includes CUDA, cuDNN, and TensorRT
FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Install system dependencies (Python already included in TensorRT image)
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    # Image processing libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libpango-1.0-0 \
    libcairo2 \
    # PDF processing
    libmupdf-dev \
    mupdf-tools \
    poppler-utils \
    # File type detection
    libmagic1 \
    # Performance monitoring
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Python is already configured in TensorRT image

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# RTX 4090 specific optimizations
ENV NVIDIA_DISABLE_REQUIRE=true
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

WORKDIR /app

# Install Python dependencies in layers for better caching
# Layer 1: NumPy (base for many packages)
RUN pip install --no-cache-dir numpy==1.24.3

# Layer 2: PaddlePaddle GPU for CUDA 12.x
# Using specific version compatible with CUDA 12.2
RUN pip install --no-cache-dir paddlepaddle-gpu==2.5.2 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Layer 3: TensorRT is already included in the base image
# Just verify it's available
RUN python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# Layer 4: PaddleOCR and related packages
RUN pip install --no-cache-dir \
    paddleocr==2.7.0.3 \
    paddle2onnx==1.0.6 \
    onnx==1.14.1 \
    onnxruntime-gpu==1.16.3

# Layer 5: Image processing libraries
RUN pip install --no-cache-dir \
    Pillow==10.1.0 \
    opencv-python==4.8.1.78 \
    PyMuPDF==1.23.8 \
    pdf2image==1.16.3 \
    scikit-image==0.22.0

# Layer 6: FastAPI and web framework
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    python-magic==0.4.27

# Layer 7: Monitoring and utilities (skip pycuda for now)
RUN pip install --no-cache-dir \
    nvidia-ml-py==12.535.133 \
    diskcache==5.6.3 \
    redis==5.0.1 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    httpx==0.25.2

# Layer 8: Test dependencies (for comprehensive testing)
RUN pip install --no-cache-dir \
    aiohttp==3.9.1 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1

# Create necessary directories
RUN mkdir -p /app/cache /app/model_cache /app/logs /app/uploads

# Create CUDNN symlinks for PaddlePaddle compatibility
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.8

# Set environment variables for PaddleOCR
ENV FLAGS_use_tf32=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy application code
COPY . /app

# Verify TensorRT and CUDA after all installations
RUN python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}'); import paddle; paddle.utils.run_check()"

# Download PaddleOCR models during build (optional, for faster startup)
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False); print('Models downloaded')" || true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Ensure cuDNN libraries are found
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]