# GPU OCR Server - Comprehensive Testing Guide

This guide explains how to run comprehensive tests on the GPU OCR Server using the PDF files in the `testpdfs` folder.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured
- GPU with CUDA support
- Test PDFs located in `/home/ryanb/Projects/gpu-server0.1/tests/testpdfs`

## Quick Start

Run all tests with Docker:

```bash
./run_docker_tests.sh --build --test-type all
```

## Test Types

### 1. Setup Verification
Verify all dependencies and environment:
```bash
./run_docker_tests.sh --test-type verify
```

### 2. Direct OCR Service Tests
Test the OCR service directly (without API):
```bash
./run_docker_tests.sh --test-type direct --max-files 10
```

This tests:
- Different processing strategies (SPEED, BALANCED, ACCURACY)
- GPU utilization and memory management
- Batch processing optimization
- Performance metrics

### 3. API Endpoint Tests
Test through REST API endpoints:
```bash
./run_docker_tests.sh --test-type api --max-files 10
```

This tests:
- Synchronous processing (`/api/v1/ocr/pdf`)
- Asynchronous job processing (`/api/v1/jobs/submit`)
- Batch processing (`/api/v1/ocr/process-batch`)
- Concurrent request handling
- Error handling and recovery

### 4. All Tests
Run complete test suite:
```bash
./run_docker_tests.sh --test-type all
```

## Docker Commands

### Build the test image:
```bash
docker-compose -f docker-compose.test.yml build
```

### Start the OCR server:
```bash
docker-compose -f docker-compose.test.yml up ocr-server
```

### Run direct tests:
```bash
docker-compose -f docker-compose.test.yml run direct-test
```

### Run API tests with custom parameters:
```bash
docker-compose -f docker-compose.test.yml run test-runner \
  python tests/test_api_pdf_processing.py --max-files 20
```

### View logs:
```bash
docker-compose -f docker-compose.test.yml logs -f ocr-server
```

### Clean up:
```bash
docker-compose -f docker-compose.test.yml down -v
```

## Test Scripts

### 1. `test_pdf_comprehensive.py`
Direct OCR service testing with detailed metrics:
- Processes PDFs through OCRService directly
- Tests all processing strategies
- Measures performance metrics
- Outputs: `test_results_comprehensive.json`

### 2. `test_api_pdf_processing.py`
API endpoint testing:
- Tests REST API endpoints
- Includes sync/async processing
- Batch processing tests
- Performance stress tests
- Outputs: `test_api_results.json`

### 3. `verify_test_setup.py`
Environment verification:
- Checks all dependencies
- Verifies GPU availability
- Validates PDF test folder
- Tests OCR service imports

## Command Line Options

### run_docker_tests.sh
```
Options:
  --build           Build Docker image before running tests
  --test-type TYPE  Type of test: all, direct, api, verify (default: all)
  --max-files N     Maximum number of files to test
  --keep-running    Keep containers running after tests
```

### test_pdf_comprehensive.py
```
Options:
  --pdf-folder PATH  Folder containing PDF files
  --max-files N      Maximum number of files to test
  --log-level LEVEL  Logging level (DEBUG, INFO, WARNING, ERROR)
```

### test_api_pdf_processing.py
```
Options:
  --server URL       Server URL (default: http://localhost:8000)
  --pdf-folder PATH  Folder containing PDF files
  --max-files N      Maximum number of files to test
  --log-level LEVEL  Logging level
```

## Test Results

Results are saved in JSON format in the `test-results` directory:

1. **test_results_comprehensive.json**
   - Processing times per strategy
   - Confidence scores
   - GPU utilization metrics
   - Pages per second rates

2. **test_api_results.json**
   - API response times
   - Success/failure rates
   - Concurrent request handling
   - Batch processing performance

## Expected Performance (RTX 4090)

Based on configuration:
- **SPEED mode**: 100-120 pages/minute
- **BALANCED mode**: 80-100 pages/minute
- **ACCURACY mode**: 50-70 pages/minute

## Troubleshooting

### Docker build fails
```bash
# Clean rebuild
docker-compose -f docker-compose.test.yml build --no-cache
```

### GPU not detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

### Server not starting
```bash
# Check logs
docker-compose -f docker-compose.test.yml logs ocr-server

# Check health
curl http://localhost:8000/api/v1/health
```

### Out of memory errors
- Reduce batch size with `--max-files` option
- Check GPU memory: `nvidia-smi`
- Adjust `GPU_MEMORY_BUFFER_MB` in docker-compose.test.yml

## Monitoring During Tests

Watch GPU utilization:
```bash
watch -n 1 nvidia-smi
```

Monitor Docker logs:
```bash
docker-compose -f docker-compose.test.yml logs -f
```

Check API metrics:
```bash
curl http://localhost:8000/api/v1/gpu/status
curl http://localhost:8000/api/v1/stats
```

## Next Steps

After running tests:
1. Review JSON result files in `test-results/`
2. Check logs for any errors or warnings
3. Adjust configuration based on performance metrics
4. Run production deployment with optimized settings