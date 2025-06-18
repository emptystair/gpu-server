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

Since the project is in initial setup phase, specific commands are not yet defined. When implementing:

1. **Dependencies**: Use `requirements.txt` for Python package management
2. **Running the server**: Likely `python src/main.py` or through Docker
3. **Testing**: Test files are in `tests/` directory, framework TBD
4. **Docker**: Use `docker-compose up` once docker-compose.yml is configured

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

The project structure has been created with the following components implemented:
- **API Schemas** (`src/api/schemas.py`): Complete Pydantic models for all request/response types
- All other implementation files are ready for development following the established modular architecture.

## API Features

The schemas define support for:
- **Multiple OCR operations**: Single image, URL-based, PDF, and batch processing
- **10 languages**: English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic
- **4 output formats**: JSON (structured), plain text, Markdown, HTML
- **Advanced features**: Image preprocessing, confidence thresholds, layout preservation, async processing
- **Monitoring**: Health checks, GPU metrics, job status tracking