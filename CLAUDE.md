# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Server 0.1 is a GPU-accelerated OCR (Optical Character Recognition) server designed to provide high-performance text recognition services from images and PDFs using NVIDIA GPU acceleration.

## Technology Stack

- **Language**: Python
- **OCR Engine**: PaddleOCR
- **GPU Acceleration**: TensorRT (NVIDIA inference optimization)
- **Web Framework**: FastAPI/Flask (based on api structure)
- **Containerization**: Docker & Docker Compose

## Project Structure

```
src/
├── api/               # API endpoints, routes, schemas, middleware
├── models/            # OCR models, TensorRT optimization, result formatting
├── utils/             # PDF/image processing, caching utilities
├── main.py            # Application entry point
├── ocr_service.py     # Core OCR service implementation
├── gpu_monitor.py     # GPU resource monitoring
└── config.py          # Configuration management
```

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-test.txt
```

### Running the Server
```bash
# Development mode
python src/main.py

# Production mode with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Docker (when configured)
docker-compose up
```

### Testing
```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py --no-cov  # Without coverage

# Run specific test file
pytest tests/test_api_routes.py -v
```

## Key Implementation Areas

1. **OCR Service** (`src/ocr_service.py`): Core service coordinating OCR operations
2. **GPU Optimization** (`src/models/tensorrt_optimizer.py`): TensorRT model optimization for GPU inference
3. **API Layer** (`src/api/`): RESTful endpoints for OCR requests
   - **Schemas** (`src/api/schemas.py`): Comprehensive Pydantic models for request/response validation
   - Supports multiple languages (10+), various output formats (JSON, text, markdown, HTML)
   - Includes image preprocessing options, batch processing, and async job management
   - See `src/api/CLAUDE.MD` for detailed API documentation
4. **Processing Utilities** (`src/utils/`): PDF-to-image conversion, image preprocessing, result caching

## Current Status

The project has been significantly developed with the following components implemented:

### Completed Components:
- **API Layer** (`src/api/`): 
  - All 13 REST endpoints implemented in `routes.py`
  - Complete middleware stack (logging, rate limiting, metrics, error handling)
  - Pydantic v2 schemas with proper validation
- **OCR Service** (`src/ocr_service.py`): 
  - Accepts `ProcessingRequest` objects for clean API
  - Returns `ProcessingResult` with structured data
  - Includes initialization, readiness checks, and shutdown methods
- **GPU Monitoring** (`src/gpu_monitor.py`): 
  - Real-time GPU metrics (memory, utilization, temperature)
  - Synchronous API for performance metrics
- **Models** (`src/models/`):
  - PaddleOCR wrapper with `is_ready()` method
  - TensorRT optimization placeholder
  - Result formatting utilities

### Implementation Highlights:
- Fixed circular import issues between modules
- Updated to Pydantic v2 (pattern instead of regex)
- Proper error handling and resource cleanup
- Async job processing with in-memory storage (upgrade to Redis for production)
- Comprehensive health and readiness checks

## API Features

The schemas define support for:
- **Multiple OCR operations**: Single image, URL-based, PDF, and batch processing
- **10 languages**: English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic
- **4 output formats**: JSON (structured), plain text, Markdown, HTML
- **Advanced features**: Image preprocessing, confidence thresholds, layout preservation, async processing
- **Monitoring**: Health checks, GPU metrics, job status tracking