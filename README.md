# GPU OCR Server

A high-performance OCR server optimized for NVIDIA RTX 4090, using PaddleOCR with TensorRT acceleration.

## Features

- **TensorRT Acceleration**: 3-4x performance improvement with NVIDIA TensorRT
- **GPU Acceleration**: Optimized for RTX 4090 with 24GB VRAM
- **High Performance**: 39 docs/min, 160 pages/min with TensorRT (3.7x improvement)
- **Multiple Format Support**: PDF, PNG, JPEG, BMP, TIFF, WEBP
- **Batch Processing**: Efficient batch operations with configurable sizes
- **Real-time Monitoring**: GPU memory, temperature, and utilization tracking
- **Image Enhancement**: Automatic quality detection and OCR optimization
- **REST API**: FastAPI-based endpoints for easy integration
- **Multi-language Support**: 10+ languages including English, Chinese, Japanese, Korean, etc.

## Performance Metrics

Based on testing with RTX 4090 and TensorRT:

| Configuration | Documents/min | Pages/min | Improvement |
|--------------|---------------|-----------|-------------|
| Baseline (CPU) | 10.5 | 53.6 | - |
| GPU (no TensorRT) | 10.5 | 53.6 | 1x |
| GPU + TensorRT | 39.1 | 160.4 | 3.7x |
| GPU + TensorRT (cached) | 146.8 | 815.0 | 14x |

- **OCR Confidence**: 96.8% average
- **Processing Speed**: 1.9s average per document (uncached)
- **GPU Utilization**: 18-28% during processing

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/gpu-server0.1.git
cd gpu-server0.1

# Start services with Docker Compose
docker-compose up -d

# Check health
curl http://localhost:8000/api/v1/health

# View logs
docker logs gpu-ocr-server
```

### Docker Configuration

The project uses:
- **Base Image**: `nvcr.io/nvidia/tensorrt:23.12-py3` (includes CUDA, cuDNN, TensorRT)
- **CUDA**: 12.3 (with 12.2 compatibility mode)
- **TensorRT**: 8.6.1
- **Python**: 3.10

### Environment Variables

Key configuration in `docker-compose.yml`:
```yaml
- NVIDIA_VISIBLE_DEVICES=0
- USE_TENSORRT=true
- TENSORRT_PRECISION=FP16
- MAX_BATCH_SIZE=50
- GPU_MEMORY_BUFFER_MB=500
- DEFAULT_DPI=120
```

## API Endpoints

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

## API Examples

### Process a PDF
```bash
curl -X POST http://localhost:8000/api/v1/ocr/process \
  -F "file=@document.pdf" \
  -F "language=en" \
  -F "strategy=speed" \
  -F "output_format=json"
```

### Check GPU Status
```bash
curl http://localhost:8000/api/v1/gpu/status
```

### Submit Async Job
```bash
# Submit job
curl -X POST http://localhost:8000/api/v1/jobs/submit \
  -F "file=@large_document.pdf" \
  -F "strategy=accuracy" \
  -F "language=en"

# Check status (use job_id from response)
curl http://localhost:8000/api/v1/jobs/{job_id}/status
```

## Project Structure

```
.
├── src/
│   ├── api/               # API endpoints and middleware
│   ├── models/            # OCR models and TensorRT optimization
│   ├── utils/             # Image/PDF processing, caching
│   ├── main.py            # FastAPI application
│   ├── ocr_service.py     # Core OCR service
│   └── config.py          # Configuration management
├── tests/                 # Test suite
│   └── performance_tests/ # TensorRT performance tests
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Docker image with TensorRT
└── requirements.txt       # Python dependencies
```

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PaddlePaddle GPU first
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install other dependencies
pip install -r requirements.txt

# Run the server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Note**: For TensorRT support, use the Docker container as TensorRT installation is complex.

## Testing

### Run Performance Tests
```bash
# Test with sample PDFs
python tests/performance_tests/test_single_pdf.py

# Test batch processing
python tests/performance_tests/test_secondary_subset.py

# Save OCR results
python tests/performance_tests/save_ocr_results.py
```

### Run Unit Tests
```bash
python tests/run_tests.py
```

## Requirements

- **Hardware**:
  - NVIDIA GPU with 8GB+ VRAM (optimized for RTX 4090)
  - CUDA-capable GPU (Compute Capability 7.0+)
  
- **Software**:
  - Docker and Docker Compose
  - NVIDIA Container Toolkit
  - CUDA 12.0+ (for local development)
  - Python 3.10+

## Troubleshooting

### Container Won't Start
- Check GPU availability: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.2.2-base nvidia-smi`
- Check logs: `docker logs gpu-ocr-server`

### Low Performance
- Ensure TensorRT is enabled: Check `USE_TENSORRT=true` in environment
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Check for caching: Results may be cached for repeated requests

### Out of Memory
- Reduce batch size: Lower `MAX_BATCH_SIZE`
- Reduce GPU memory allocation: Lower `gpu_mem` in configuration
- Process fewer pages at once

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.