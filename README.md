# GPU OCR Server

A high-performance OCR server optimized for NVIDIA RTX 4090, using PaddleOCR with TensorRT acceleration.

## Features

- **GPU Acceleration**: Optimized for RTX 4090 with 24GB VRAM
- **High Performance**: 3-6 pages/second PDF processing at 120 DPI
- **Multiple Format Support**: PDF, PNG, JPEG, BMP, TIFF
- **Batch Processing**: Efficient batch operations with configurable sizes
- **Real-time Monitoring**: GPU memory, temperature, and utilization tracking
- **Image Enhancement**: Automatic quality detection and OCR optimization
- **REST API**: FastAPI-based endpoints for easy integration

## Current Implementation Status

###  Completed Components

1. **Configuration System** (`src/config.py`)
   - Dataclass-based configuration
   - Environment variable support
   - RTX 4090 optimizations (FP16, 120 DPI default)

2. **API Schemas** (`src/api/schemas.py`)
   - Pydantic models for requests/responses
   - ProcessingStrategy enum (speed/accuracy/balanced)
   - Comprehensive validation

3. **GPU Monitoring** (`src/gpu_monitor.py`)
   - Real-time GPU metrics via pynvml
   - Memory, temperature, utilization tracking
   - Background monitoring thread

4. **Image Processing** (`src/utils/image_processor.py`)
   - Quality detection (blur, noise, contrast, skew)
   - Auto-enhancement for OCR
   - DPI normalization
   - Batch processing support

5. **PDF Processing** (`src/utils/pdf_processor.py`)
   - PyMuPDF-based conversion
   - Page range selection
   - Metadata extraction
   - Tested with 96 real documents

6. **Result Formatting** (`src/models/result_formatter.py`)
   - Multiple output formats (JSON, text, structured)
   - Confidence filtering
   - Layout preservation

### ✅ Additional Completed Components

7. **Cache Management** (`src/utils/cache_manager.py`)
   - Multi-tier caching (memory → Redis → disk)
   - Distributed cache support
   - AWS Batch compatibility

8. **OCR Service** (`src/ocr_service.py`)
   - Clean API using ProcessingRequest/ProcessingResult
   - Multi-strategy processing (SPEED, BALANCED, ACCURACY)
   - Dynamic batch sizing based on GPU memory
   - Result caching and lifecycle management

9. **REST API Layer** (`src/api/`)
   - **13 endpoints** fully implemented
   - Complete middleware stack
   - Async job processing
   - Health checks and metrics

10. **PaddleOCR Integration** (`src/models/paddle_ocr.py`)
    - RTX 4090 optimized settings
    - TensorRT acceleration support
    - Multi-language OCR (10+ languages)

### 🚧 Pending/Partial Components

- TensorRT Optimizer (placeholder implemented)
- Main Application (basic structure exists)
- Docker configuration (Dockerfile exists, docker-compose pending)

## Performance Metrics

Based on testing with RTX 4090:

| Metric | Value |
|--------|-------|
| PDF Processing | 3-6 pages/second at 120 DPI |
| Memory per Page | 3.85 MB at 120 DPI |
| GPU Capacity | ~6,200 pages in VRAM |
| Success Rate | 100% (96 test documents) |

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/emptystair/gpu-server.git
cd gpu-server

# Build and run with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Note: For PaddlePaddle GPU, use:
pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Run the server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Quick API Test

```bash
# Process a single image
curl -X POST http://localhost:8000/api/v1/ocr/process \
  -F "file=@sample.jpg" \
  -F "language=en" \
  -F "output_format=json"

# Check GPU status
curl http://localhost:8000/api/v1/gpu/status

# Submit async job
curl -X POST http://localhost:8000/api/v1/jobs/submit \
  -F "file=@large_document.pdf" \
  -F "strategy=accuracy"
```

## API Endpoints (Implemented)

### OCR Processing
- `POST /api/v1/ocr/process` - Process single document (image or PDF)
- `POST /api/v1/ocr/process-batch` - Batch process multiple documents
- `POST /api/v1/ocr/image` - Process uploaded image
- `POST /api/v1/ocr/url` - Process image from URL
- `POST /api/v1/ocr/pdf` - Process PDF with page range support

### Async Jobs
- `POST /api/v1/jobs/submit` - Submit async OCR job
- `GET /api/v1/jobs/{job_id}/status` - Check job status
- `GET /api/v1/jobs/{job_id}/result` - Get job results

### System Monitoring
- `GET /api/v1/health` - Health check
- `GET /api/v1/ready` - Readiness probe
- `GET /api/v1/gpu/status` - GPU metrics
- `GET /api/v1/stats` - Processing statistics
- `GET /api/v1/metrics` - Prometheus metrics

See [API Documentation](docs/api_endpoints.md) for detailed endpoint information.

## Configuration

Key environment variables:

```bash
DEFAULT_DPI=120              # OCR processing DPI
MAX_BATCH_SIZE=50           # Maximum batch size
TENSORRT_PRECISION=FP16     # TensorRT precision mode
GPU_MEMORY_BUFFER_MB=500    # Reserved GPU memory
```

## Testing

Comprehensive test suite available:

```bash
# Run all tests
cd tests
python test_pdf_processor.py
python test_image_processor.py
python test_gpu_monitor.py
```

See `tests/test_results_summary.md` for detailed test results.

## Requirements

- NVIDIA GPU with 8GB+ VRAM (optimized for RTX 4090)
- CUDA 12.2+
- Python 3.10+
- Docker (optional but recommended)

## License

[Add your license here]

## Contributing

[Add contribution guidelines]