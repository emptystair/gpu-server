# API Endpoints Documentation

This document provides detailed information about all implemented API endpoints in the GPU OCR Server.

## Base URL

All endpoints are prefixed with `/api/v1`

## Docker Configuration

The API runs in a Docker container using the NVIDIA TensorRT base image (`nvcr.io/nvidia/tensorrt:23.12-py3`) which includes:
- CUDA 12.2
- cuDNN 8
- TensorRT optimizations
- Python 3.10

## Authentication

Currently, no authentication is required. In production, consider implementing API keys or JWT tokens.

## Project Structure

The API implementation follows a modular structure:
```
src/
├── api/
│   ├── routes.py          # All 13 API endpoints
│   ├── schemas.py         # Pydantic v2 models
│   ├── middleware.py      # Logging, rate limiting, metrics
│   └── __init__.py        # API app initialization
├── models/
│   ├── paddle_ocr.py      # OCR engine wrapper
│   └── tensorrt_optimizer.py # GPU acceleration
└── utils/
    └── cache_manager.py   # Multi-tier caching
```

Archived components are stored in the `archive/` folder for reference.

## Endpoints

### 1. Process Single Document

**Endpoint:** `POST /api/v1/ocr/process`

**Description:** General-purpose endpoint for processing any document type (image or PDF). This endpoint leverages TensorRT optimization for RTX 4090 GPUs when available, providing 3-5x speedup over baseline PaddleOCR.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Parameters:**
- `file` (required): The document file to process
- `strategy` (optional): Processing strategy - `speed`, `balanced`, or `accuracy` (default: `balanced`)
- `dpi` (optional): DPI for PDF extraction (default: 150)
- `confidence_threshold` (optional): Minimum confidence threshold (default: 0.5)
- `language` (optional): OCR language (default: `en`)
- `output_format` (optional): Output format - `json`, `text`, `markdown`, or `html` (default: `json`)

**Response:** `OCRResponse` schema

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/ocr/process \
  -F "file=@document.pdf" \
  -F "strategy=accuracy" \
  -F "language=en"
```

### 2. Batch Process Documents

**Endpoint:** `POST /api/v1/ocr/process-batch`

**Description:** Process multiple documents in a single request.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Parameters:**
- `files` (required): Multiple document files (max 100)
- `strategy` (optional): Processing strategy
- `dpi` (optional): DPI for PDF extraction
- `confidence_threshold` (optional): Minimum confidence threshold
- `language` (optional): OCR language
- `parallel_processing` (optional): Process files in parallel (default: true)

**Response:** `BatchOCRResponse` schema

### 3. Process Image

**Endpoint:** `POST /api/v1/ocr/image`

**Description:** Specialized endpoint for image processing with full schema validation.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
- `file` (required): Image file
- Request body follows `OCRImageRequest` schema

**Response:** `OCRResponse` schema

### 4. Process Image from URL

**Endpoint:** `POST /api/v1/ocr/url`

**Description:** Download and process an image from a URL.

**Request:**
- Method: `POST`
- Content-Type: `application/json`

**Body:** `OCRUrlRequest` schema
```json
{
  "image_url": "https://example.com/document.jpg",
  "language": "en",
  "output_format": "json",
  "confidence_threshold": 0.5
}
```

**Response:** `OCRResponse` schema

### 5. Process PDF

**Endpoint:** `POST /api/v1/ocr/pdf`

**Description:** Process PDF files with page range selection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
- `file` (required): PDF file
- Request body follows `OCRPDFRequest` schema

**Response:** `OCRResponse` schema

### 6. Submit Async Job

**Endpoint:** `POST /api/v1/jobs/submit`

**Description:** Submit a document for asynchronous processing.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Form Parameters:**
- `file` (optional): Document file to process
- `image_url` (optional): URL of image to process (provide either file or image_url)
- `language` (optional): OCR language
- `output_format` (optional): Output format
- `strategy` (optional): Processing strategy
- `dpi` (optional): DPI for PDF extraction
- `confidence_threshold` (optional): Minimum confidence threshold

**Response:** `AsyncJobResponse` schema

**Example Response:**
```json
{
  "job_id": "job_550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-01-20T10:30:00Z",
  "estimated_completion": "2024-01-20T10:31:00Z",
  "status_url": "/api/v1/jobs/job_550e8400/status",
  "result_url": "/api/v1/jobs/job_550e8400/result"
}
```

### 7. Get Job Status

**Endpoint:** `GET /api/v1/jobs/{job_id}/status`

**Description:** Check the status of an async job.

**Response:** `JobStatusResponse` schema

### 8. Get Job Result

**Endpoint:** `GET /api/v1/jobs/{job_id}/result`

**Description:** Retrieve the results of a completed async job.

**Response Codes:**
- `200`: Job completed successfully (returns OCR results)
- `202`: Job still processing
- `404`: Job not found
- `500`: Job failed

### 9. GPU Status

**Endpoint:** `GET /api/v1/gpu/status`

**Description:** Get current GPU status and utilization metrics. When TensorRT is enabled, this endpoint also reports TensorRT engine status and optimization metrics.

**Response:** `GPUStatusResponse` schema

**Example Response:**
```json
{
  "available": true,
  "device_count": 1,
  "devices": [{
    "device_id": 0,
    "device_name": "NVIDIA GeForce RTX 4090",
    "memory_used_mb": 2048,
    "memory_total_mb": 24576,
    "temperature_celsius": 65.0,
    "utilization_percent": 45.2,
    "power_draw_watts": 250.5
  }]
}
```

### 10. Health Check

**Endpoint:** `GET /api/v1/health`

**Description:** Basic health check endpoint.

**Response:** `HealthResponse` schema

### 11. Readiness Check

**Endpoint:** `GET /api/v1/ready`

**Description:** Kubernetes readiness probe endpoint.

**Response:** `ReadinessResponse` schema

### 12. Statistics

**Endpoint:** `GET /api/v1/stats`

**Description:** Get processing statistics and performance metrics.

**Response:** `StatsResponse` schema

### 13. Metrics

**Endpoint:** `GET /api/v1/metrics`

**Description:** Prometheus-compatible metrics endpoint.

**Response:** Plain text metrics in Prometheus format

## Error Handling

All endpoints return consistent error responses using the `ErrorResponse` schema:

```json
{
  "error": "ValidationError",
  "message": "Invalid image format",
  "details": {
    "supported_formats": ["jpg", "png", "bmp"]
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Rate Limiting

Rate limiting is implemented using a token bucket algorithm:
- Default: 60 requests per minute
- Burst capacity: 10 requests
- Headers returned: `X-RateLimit-Limit`, `Retry-After`

## Request Tracking

All requests are assigned a unique ID available in:
- Request context: `request.state.request_id`
- Response header: `X-Request-ID`
- Log entries for correlation

## Performance with TensorRT

When running with TensorRT optimization on RTX 4090:

| Endpoint | Without TensorRT | With TensorRT | Speedup |
|----------|------------------|---------------|---------|
| `/ocr/process` (single image) | ~50ms | ~15ms | 3.3x |
| `/ocr/process-batch` (16 images) | ~800ms | ~150ms | 5.3x |
| `/ocr/pdf` (10 pages) | ~2000ms | ~500ms | 4x |

## Dependencies

Key dependencies from the Dockerfile:
- **PaddlePaddle GPU**: 2.5.2 (CUDA 12.x compatible)
- **PaddleOCR**: 2.7.0.3
- **TensorRT**: Included in base image
- **FastAPI**: 0.104.1
- **Pydantic**: 2.5.0
- **ONNX**: 1.14.1 (for TensorRT conversion)
- **PyMuPDF**: 1.23.8 (PDF processing)