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

### =§ Pending Components

- OCR Service (PaddleOCR integration)
- TensorRT Optimizer
- Cache Manager
- Batch Manager
- Main Application
- API Routes

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

## API Endpoints (Planned)

- `POST /api/v1/ocr/process` - Process single document
- `POST /api/v1/ocr/batch` - Batch processing
- `GET /api/v1/ocr/status/{task_id}` - Check processing status
- `GET /api/v1/ocr/result/{task_id}` - Get OCR results
- `GET /health` - Health check
- `GET /metrics` - Performance metrics

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