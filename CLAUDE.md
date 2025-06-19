# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Server 0.1 is a GPU-accelerated OCR (Optical Character Recognition) server designed to provide high-performance text recognition services from images and PDFs using NVIDIA GPU acceleration with TensorRT optimization.

## Technology Stack

- **Language**: Python 3.10
- **OCR Engine**: PaddleOCR 2.7.0.3 with PaddlePaddle GPU 2.6.1
- **GPU Acceleration**: NVIDIA TensorRT 8.6.1
- **Web Framework**: FastAPI 0.104.1
- **Containerization**: Docker & Docker Compose
- **Base Image**: `nvcr.io/nvidia/tensorrt:23.12-py3`

## Project Structure

```
.
├── src/
│   ├── api/               # API endpoints, routes, schemas, middleware
│   ├── models/            # OCR models, TensorRT optimization, result formatting
│   ├── utils/             # PDF/image processing, caching utilities
│   ├── main.py            # Application entry point
│   ├── ocr_service.py     # Core OCR service implementation
│   ├── gpu_monitor.py     # GPU resource monitoring
│   └── config.py          # Configuration management
├── tests/
│   ├── performance_tests/ # TensorRT performance testing scripts
│   └── testpdfs/          # Test PDF documents
├── archive/               # Archived Docker and script files
├── docker-compose.yml     # Active Docker Compose configuration
├── Dockerfile             # Active Dockerfile with TensorRT
└── requirements.txt       # Python dependencies
```

## Development Commands

### Docker (Recommended)
```bash
# Start services
docker-compose up -d

# View logs
docker logs gpu-ocr-server -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose build
```

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install PaddlePaddle GPU first
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install other dependencies
pip install -r requirements.txt

# Run server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Run unit tests
python tests/run_tests.py

# Run performance tests
python tests/performance_tests/test_single_pdf.py
python tests/performance_tests/test_secondary_subset.py

# Save OCR results
python tests/performance_tests/save_ocr_results.py
```

## Key Implementation Areas

1. **OCR Service** (`src/ocr_service.py`): Core service coordinating OCR operations with TensorRT
2. **TensorRT Optimization** (`src/models/paddle_ocr.py`): Enabled via `use_tensorrt=True`
3. **API Layer** (`src/api/`): 13 RESTful endpoints for OCR requests
4. **Processing Utilities** (`src/utils/`): PDF/image processing, multi-tier caching
5. **Configuration** (`src/config.py`): Environment-based configuration with TensorRT settings

## Performance Metrics

With TensorRT enabled on RTX 4090:
- **Baseline**: 10.5 docs/min, 53.6 pages/min
- **TensorRT**: 39.1 docs/min, 160.4 pages/min (3.7x improvement)
- **Confidence**: 96.8% average maintained

## Current Status

### ✅ Completed:
- TensorRT integration and optimization
- All 13 API endpoints
- Health check fix (cache_manager reference)
- Multi-tier caching (memory → Redis → disk)
- GPU monitoring and metrics
- Async job processing
- Performance testing suite
- Docker configuration with TensorRT base image

### 🔧 Configuration:
- TensorRT FP16 precision mode
- 16GB GPU memory allocation
- Dynamic batch sizing
- cuDNN optimization enabled

## Important Notes

1. **TensorRT**: Requires Docker container for proper setup
2. **GPU Memory**: Configured for 16GB (RTX 4090 has 24GB)
3. **Caching**: Redis required for distributed caching
4. **Health Checks**: Fixed in routes.py (cache_manager access)

## Environment Variables

Key settings in docker-compose.yml:
```yaml
- USE_TENSORRT=true
- TENSORRT_PRECISION=FP16
- MAX_BATCH_SIZE=50
- GPU_MEMORY_BUFFER_MB=500
- DEFAULT_DPI=120
```

## API Features

- **Languages**: 10+ including English, Chinese, Japanese, Korean
- **Output Formats**: JSON, text, markdown, HTML
- **Processing Strategies**: Speed, balanced, accuracy
- **Batch Processing**: Up to 50 documents
- **Async Jobs**: Background processing with status tracking