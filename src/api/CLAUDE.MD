# API Module Documentation

This module contains the API layer for the GPU-accelerated OCR server, including endpoints, request/response schemas, and middleware.

## API Schemas (schemas.py)

### Enums

#### OCRLanguage
Supported languages for OCR processing:
- `en` - English
- `ch` - Chinese
- `japan` - Japanese
- `korean` - Korean
- `fr` - French
- `german` - German
- `es` - Spanish
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic

#### OutputFormat
Available output formats:
- `json` - Structured JSON with regions and metadata
- `text` - Plain text output
- `markdown` - Markdown formatted text
- `html` - HTML formatted text

#### ProcessingStatus
Processing status values:
- `pending` - Job queued
- `processing` - Currently processing
- `completed` - Successfully completed
- `failed` - Processing failed

### Request Models

#### ImageProcessingOptions
Optional preprocessing settings for images:
```json
{
  "enhance_contrast": false,
  "denoise": false,
  "deskew": false,
  "remove_background": false,
  "binarize": false,
  "resize_factor": 1.0  // 0.1-4.0
}
```

#### OCRImageRequest
For processing uploaded images:
```json
{
  "language": "en",
  "enable_angle_classification": true,
  "output_format": "json",
  "preprocessing": {...},
  "confidence_threshold": 0.5,
  "preserve_layout": false
}
```

#### OCRUrlRequest
For processing images from URLs:
```json
{
  "image_url": "https://example.com/document.jpg",
  "language": "en",
  "enable_angle_classification": true,
  "output_format": "json",
  "preprocessing": {...},
  "confidence_threshold": 0.5,
  "preserve_layout": false
}
```

#### OCRPDFRequest
For processing PDF files:
```json
{
  "language": "en",
  "output_format": "json",
  "preprocessing": {...},
  "confidence_threshold": 0.5,
  "page_range": "1-5,7,9-12",  // Optional
  "merge_pages": true
}
```

#### BatchOCRRequest
For batch processing multiple images:
```json
{
  "image_urls": [
    "https://example.com/doc1.jpg",
    "https://example.com/doc2.jpg"
  ],
  "language": "en",
  "output_format": "json",
  "preprocessing": {...},
  "confidence_threshold": 0.5,
  "parallel_processing": true
}
```
Note: Maximum 100 images per batch.

### Response Models

#### TextRegion
Detected text region with location:
```json
{
  "text": "Hello World",
  "confidence": 0.95,
  "bbox": {
    "x1": 10,
    "y1": 10,
    "x2": 100,
    "y2": 30
  },
  "polygon": [[10,10], [100,10], [100,30], [10,30]]  // Optional
}
```

#### OCRResponse
Standard OCR response:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "timestamp": "2024-01-20T10:30:00Z",
  "processing_time": 0.245,
  "text": "Extracted text content",  // For text/markdown/html formats
  "regions": [...],  // For JSON format
  "pages": [...],    // For PDF processing
  "total_regions": 15,
  "average_confidence": 0.92,
  "language_detected": "en",
  "gpu_info": {
    "gpu_id": 0,
    "utilization": 45.2
  }
}
```

#### BatchOCRResponse
Batch processing response:
```json
{
  "request_id": "batch_550e8400",
  "status": "completed",
  "timestamp": "2024-01-20T10:30:00Z",
  "total_images": 10,
  "successful": 9,
  "failed": 1,
  "results": [...],  // Array of OCRResponse or error objects
  "total_processing_time": 2.34
}
```

#### AsyncJobResponse
For asynchronous job submission:
```json
{
  "job_id": "job_550e8400",
  "status": "pending",
  "created_at": "2024-01-20T10:30:00Z",
  "estimated_completion": "2024-01-20T10:31:00Z",
  "status_url": "/api/v1/jobs/job_550e8400/status",
  "result_url": "/api/v1/jobs/job_550e8400/result"
}
```

#### ErrorResponse
Error response format:
```json
{
  "error": "ValidationError",
  "message": "Invalid image format",
  "detail": {
    "supported_formats": ["jpg", "png", "bmp"]
  },
  "request_id": "550e8400",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

#### HealthCheckResponse
Health check endpoint response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 3090",
    "memory_total": 24576,
    "memory_used": 2048
  },
  "models_loaded": true,
  "cache_enabled": true
}
```

#### GPUMetrics
GPU utilization metrics:
```json
{
  "gpu_id": 0,
  "name": "NVIDIA GeForce RTX 3090",
  "memory_used": 2048,
  "memory_total": 24576,
  "memory_percent": 8.3,
  "gpu_utilization": 45.2,
  "temperature": 65.0,
  "power_draw": 250.5
}
```

## Expected API Endpoints

Based on the schema definitions, the following endpoints should be implemented:

### Core OCR Endpoints
- `POST /api/v1/ocr/image` - Process uploaded image
- `POST /api/v1/ocr/url` - Process image from URL
- `POST /api/v1/ocr/pdf` - Process PDF file
- `POST /api/v1/ocr/batch` - Batch process multiple images

### Async Job Management
- `POST /api/v1/jobs/submit` - Submit async OCR job
- `GET /api/v1/jobs/{job_id}/status` - Check job status
- `GET /api/v1/jobs/{job_id}/result` - Retrieve job results

### System Endpoints
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics/gpu` - GPU metrics

## Implementation Notes

1. All endpoints should validate requests using the Pydantic schemas
2. File uploads should support multipart/form-data
3. Implement proper error handling with ErrorResponse
4. Use request_id for tracking and debugging
5. Consider implementing rate limiting for batch endpoints
6. GPU metrics should be collected in real-time
7. Support both synchronous and asynchronous processing modes

## Middleware Requirements

- Request ID generation and tracking
- Request/response logging
- Performance monitoring
- Error handling and formatting
- CORS configuration for web clients
- Authentication/authorization (if required)
- Rate limiting for resource protection