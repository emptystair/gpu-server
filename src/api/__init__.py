"""
API Module

Provides REST API endpoints, request/response schemas, and middleware
for the GPU OCR Server.
"""

# from .routes import router, lifespan  # Commented to avoid circular import during tests
# from .middleware import setup_middleware, get_request_id  # Commented to avoid circular import during tests
from .schemas import (
    # Enums
    OCRLanguage,
    OutputFormat,
    ProcessingStatus,
    ProcessingStrategy,
    
    # Request schemas
    OCRRequest,
    BatchOCRRequest,
    ImageProcessingOptions,
    OCRImageRequest,
    OCRUrlRequest,
    OCRPDFRequest,
    
    # Response schemas
    OCRResponse,
    BatchOCRResponse,
    ErrorResponse,
    HealthCheckResponse,
    GPUMetrics,
    GPUStatusResponse,
    StatsResponse,
    HealthResponse,
    ReadinessResponse,
    
    # Data models
    BoundingBox,
    TextRegion,
    WordResponse,
    PageResponse,
    MemoryStatus,
    UtilizationStatus,
    ServiceStatus
)

__all__ = [
    # Router and lifecycle
    # "router",
    # "lifespan",
    
    # Middleware
    # "setup_middleware",
    # "get_request_id",
    
    # Enums
    "OCRLanguage",
    "OutputFormat", 
    "ProcessingStatus",
    "ProcessingStrategy",
    
    # Request schemas
    "OCRRequest",
    "BatchOCRRequest",
    "ImageProcessingOptions",
    "OCRImageRequest",
    "OCRUrlRequest",
    "OCRPDFRequest",
    
    # Response schemas
    "OCRResponse",
    "BatchOCRResponse",
    "ErrorResponse",
    "HealthCheckResponse",
    "GPUMetrics",
    "GPUStatusResponse",
    "StatsResponse",
    "HealthResponse",
    "ReadinessResponse",
    
    # Data models
    "BoundingBox",
    "TextRegion",
    "WordResponse",
    "PageResponse",
    "MemoryStatus",
    "UtilizationStatus",
    "ServiceStatus"
]