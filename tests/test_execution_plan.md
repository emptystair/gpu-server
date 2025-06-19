# Test Execution Plan for GPU OCR Server

## Overview
This document provides a structured approach to testing all components of the GPU OCR Server, ensuring reliability, performance, and correctness.

## Test Environment Setup

### 1. Install Dependencies
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Install development dependencies
pip install -e .
```

### 2. Configure Test Environment
```bash
# Set test environment variables
export ENVIRONMENT=test
export CUDA_VISIBLE_DEVICES=-1  # Disable GPU for unit tests
export LOG_LEVEL=DEBUG
```

### 3. Prepare Test Data
```bash
# Create test directories
mkdir -p tests/data/images tests/data/pdfs tests/data/outputs

# Download sample test documents (if needed)
python tests/scripts/download_test_data.py
```

## Test Execution Phases

### Phase 1: Unit Tests (Daily)
Quick, isolated tests that don't require external services.

```bash
# Run all unit tests
python tests/run_tests.py unit

# Run specific component tests
pytest tests/test_cache_manager.py -v
pytest tests/test_image_processor.py -v
pytest tests/test_pdf_processor.py -v
```

**Expected Duration**: 5-10 minutes
**Coverage Target**: >90%

### Phase 2: Integration Tests (Daily)
Tests that verify component interactions.

```bash
# Run integration tests
python tests/run_tests.py integration

# Specific integration tests
pytest tests/test_ocr_pipeline_integration.py -v
pytest tests/test_api_routes.py -v
```

**Expected Duration**: 15-20 minutes
**Coverage Target**: >80%

### Phase 3: GPU Tests (Before Release)
Tests requiring actual GPU hardware.

```bash
# Enable GPU for these tests
export CUDA_VISIBLE_DEVICES=0

# Run GPU-specific tests
python tests/run_tests.py gpu

# GPU monitor tests
pytest tests/test_gpu_monitor_complete.py -m gpu -v

# Performance with GPU
pytest tests/test_ocr_service.py -m gpu -v
```

**Expected Duration**: 10-15 minutes
**Requirements**: NVIDIA GPU with CUDA

### Phase 4: Performance Tests (Weekly)
Benchmark and stress tests.

```bash
# Run performance benchmarks
python tests/run_tests.py performance

# Specific performance tests
pytest tests/test_pdf_dpi_performance.py -v
pytest tests/test_api_performance.py -v --benchmark-only
```

**Expected Duration**: 30-45 minutes
**Metrics Tracked**:
- Processing speed (pages/minute)
- Memory usage
- GPU utilization
- Response times

### Phase 5: End-to-End Tests (Before Release)
Complete system tests with all components.

```bash
# Start all services
docker-compose -f docker-compose.test.yml up -d

# Run E2E tests
python tests/run_tests.py e2e

# API acceptance tests
pytest tests/test_e2e_workflows.py -v
```

**Expected Duration**: 20-30 minutes
**Coverage**: All critical user workflows

## Continuous Integration Pipeline

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: |
          pip install -r requirements-test.txt
          python tests/run_tests.py unit

  integration-tests:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: |
          pip install -r requirements-test.txt
          python tests/run_tests.py integration

  gpu-tests:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v3
      - name: Run GPU tests
        run: |
          pip install -r requirements-test.txt
          python tests/run_tests.py gpu
```

## Test Data Management

### Test Document Types
1. **Images**
   - Small (< 1MB): Quick unit tests
   - Medium (1-10MB): Integration tests
   - Large (> 10MB): Performance tests

2. **PDFs**
   - Single page: Basic functionality
   - Multi-page (10-20 pages): Standard tests
   - Large (100+ pages): Stress tests

3. **Edge Cases**
   - Corrupted files
   - Empty files
   - Unsupported formats
   - Very low quality scans

### Test Data Location
```
tests/data/
├── images/
│   ├── simple_text.png
│   ├── complex_layout.jpg
│   ├── handwritten.tiff
│   └── low_quality.jpg
├── pdfs/
│   ├── single_page.pdf
│   ├── multi_page.pdf
│   └── scanned_document.pdf
└── edge_cases/
    ├── corrupted.pdf
    ├── empty.jpg
    └── huge_file.tiff
```

## Debugging Failed Tests

### 1. Run Failed Tests Only
```bash
# Re-run only failed tests
python tests/run_tests.py --failed
```

### 2. Increase Verbosity
```bash
# Very verbose output
pytest tests/test_ocr_service.py::test_specific_function -vvv
```

### 3. Debug Mode
```bash
# Drop into debugger on failure
pytest tests/test_api_routes.py --pdb
```

### 4. Check Logs
```bash
# View test logs
tail -f tests/logs/test_*.log
```

## Performance Benchmarks

### Baseline Metrics (RTX 4090)
- Single page OCR: < 500ms
- 10-page PDF: < 5s
- Batch (10 images): < 10s
- API latency: < 100ms
- GPU memory usage: < 20GB
- Throughput: 50+ pages/minute

### Performance Regression Tests
```bash
# Run benchmarks and compare with baseline
pytest tests/performance/ --benchmark-compare=baseline.json
```

## Test Reporting

### 1. Coverage Report
```bash
# Generate HTML coverage report
python tests/run_tests.py all
open htmlcov/index.html
```

### 2. Test Results Summary
```bash
# Generate JUnit XML report
pytest --junitxml=test-results.xml

# Generate Allure report
pytest --alluredir=allure-results
allure serve allure-results
```

### 3. Performance Report
```bash
# Generate performance comparison
pytest-benchmark compare baseline.json latest.json
```

## Troubleshooting Common Issues

### GPU Not Available
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Redis Connection Failed
```bash
# Start Redis for tests
docker run -d -p 6379:6379 redis:7-alpine
```

### Test Timeouts
```bash
# Increase timeout for slow tests
pytest --timeout=300 tests/test_slow.py
```

### Memory Issues
```bash
# Run tests with memory profiling
pytest --memprof tests/test_memory_intensive.py
```

## Pre-Release Checklist

- [ ] All unit tests passing (>90% coverage)
- [ ] All integration tests passing
- [ ] GPU tests passing on target hardware
- [ ] Performance benchmarks meet targets
- [ ] E2E tests cover all user workflows
- [ ] No security vulnerabilities (bandit scan)
- [ ] Documentation tests passing
- [ ] Load tests show stable performance
- [ ] Memory leak tests clean
- [ ] Error scenarios properly handled

## Next Steps

1. **Set up CI/CD pipeline** with automated test execution
2. **Create test data repository** with realistic documents
3. **Implement load testing** with Locust
4. **Add mutation testing** for better coverage
5. **Set up test environment** in cloud for GPU tests