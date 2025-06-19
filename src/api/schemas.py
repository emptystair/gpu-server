from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, validator, HttpUrl
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging


class OCRLanguage(str, Enum):
    """Supported OCR languages"""
    ENGLISH = "en"
    CHINESE = "ch"
    JAPANESE = "japan"
    KOREAN = "korean"
    FRENCH = "fr"
    GERMAN = "german"
    SPANISH = "es"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"


class OutputFormat(str, Enum):
    """Output format options"""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class ProcessingStatus(str, Enum):
    """Processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStrategy(str, Enum):
    """Processing strategy options"""
    SPEED = "speed"
    ACCURACY = "accuracy"
    BALANCED = "balanced"


# Request Schemas
class OCRRequest(BaseModel):
    """Request schema for OCR processing"""
    strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    dpi: Optional[int] = Field(None, ge=72, le=600)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    output_format: str = Field("json", pattern="^(json|text|xml)$")
    
    @validator('dpi')
    def validate_dpi(cls, v):
        """Ensure DPI is reasonable for OCR"""
        if v and v < 120:
            logging.warning(f"Low DPI {v} may affect OCR quality")
        return v


class BatchOCRRequest(BaseModel):
    """Request schema for batch processing"""
    strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    max_concurrent: int = Field(5, ge=1, le=10)
    fail_fast: bool = False


# Keep existing preprocessing options for compatibility
class ImageProcessingOptions(BaseModel):
    """Image preprocessing options"""
    enhance_contrast: bool = Field(False, description="Apply contrast enhancement")
    denoise: bool = Field(False, description="Apply denoising filter")
    deskew: bool = Field(False, description="Automatically deskew image")
    remove_background: bool = Field(False, description="Remove background")
    binarize: bool = Field(False, description="Convert to binary (black/white)")
    resize_factor: Optional[float] = Field(None, ge=0.1, le=4.0, description="Resize factor (0.1-4.0)")


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v


class TextRegion(BaseModel):
    """Detected text region with coordinates and confidence"""
    text: str = Field(..., description="Detected text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    polygon: Optional[List[List[int]]] = Field(None, description="Polygon points for non-rectangular regions")


class OCRRequestBase(BaseModel):
    """Base OCR request parameters"""
    language: OCRLanguage = Field(OCRLanguage.ENGLISH, description="OCR language")
    enable_angle_classification: bool = Field(True, description="Enable text angle classification")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="Output format")
    preprocessing: Optional[ImageProcessingOptions] = Field(None, description="Image preprocessing options")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    preserve_layout: bool = Field(False, description="Preserve original document layout")
    
    class Config:
        json_schema_extra = {
            "example": {
                "language": "en",
                "enable_angle_classification": True,
                "output_format": "json",
                "confidence_threshold": 0.5
            }
        }


class OCRImageRequest(OCRRequestBase):
    """Request for OCR on uploaded image"""
    pass


class OCRUrlRequest(OCRRequestBase):
    """Request for OCR on image from URL"""
    image_url: HttpUrl = Field(..., description="URL of the image to process")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/document.jpg",
                "language": "en",
                "output_format": "json"
            }
        }


class OCRPDFRequest(BaseModel):
    """Request for OCR on PDF file"""
    language: OCRLanguage = Field(OCRLanguage.ENGLISH, description="OCR language")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="Output format")
    preprocessing: Optional[ImageProcessingOptions] = Field(None, description="Image preprocessing options")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    page_range: Optional[str] = Field(None, description="Page range (e.g., '1-5,7,9-12')")
    merge_pages: bool = Field(True, description="Merge all pages into single result")
    
    @validator('page_range')
    def validate_page_range(cls, v):
        if v is None:
            return v
        try:
            # Validate page range format
            parts = v.split(',')
            for part in parts:
                if '-' in part:
                    start, end = part.split('-')
                    if int(start) > int(end):
                        raise ValueError(f"Invalid range: {part}")
                else:
                    int(part)  # Validate single page number
            return v
        except Exception:
            raise ValueError("Invalid page range format. Use format like '1-5,7,9-12'")


class BatchOCRRequest(BaseModel):
    """Request for batch OCR processing"""
    image_urls: List[HttpUrl] = Field(..., description="List of image URLs to process")
    language: OCRLanguage = Field(OCRLanguage.ENGLISH, description="OCR language")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="Output format")
    preprocessing: Optional[ImageProcessingOptions] = Field(None, description="Image preprocessing options")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    parallel_processing: bool = Field(True, description="Process images in parallel")
    
    @validator('image_urls')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 images")
        return v


# Response Schemas
class WordResponse(BaseModel):
    """Schema for word-level results"""
    text: str
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)


class PageResponse(BaseModel):
    """Schema for individual page results"""
    page_number: int = Field(..., ge=1)
    text: str
    words: List[WordResponse]
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: List[str] = []


class OCRResponse(BaseModel):
    """Response schema for OCR results"""
    document_id: str
    status: str = Field(..., pattern="^(success|partial|failed)$")
    pages: List[PageResponse]
    total_pages: int
    processing_time_ms: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any]


class BatchOCRResponse(BaseModel):
    """Response schema for batch results"""
    batch_id: str
    documents: List[OCRResponse]
    total_documents: int
    successful: int
    failed: int
    total_processing_time_ms: float


# GPU Status Schemas
class MemoryStatus(BaseModel):
    """GPU memory information"""
    total_mb: int
    used_mb: int
    free_mb: int
    reserved_mb: int
    
    @property
    def utilization_percent(self) -> float:
        return (self.used_mb / self.total_mb) * 100


class UtilizationStatus(BaseModel):
    """GPU utilization metrics"""
    compute_percent: float = Field(..., ge=0.0, le=100.0)
    memory_percent: float = Field(..., ge=0.0, le=100.0)
    encoder_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    decoder_percent: Optional[float] = Field(None, ge=0.0, le=100.0)


class GPUStatusResponse(BaseModel):
    """GPU status information"""
    device_id: int
    device_name: str
    memory: MemoryStatus
    utilization: UtilizationStatus
    temperature_celsius: float
    power_draw_watts: float


class StatsResponse(BaseModel):
    """Processing statistics"""
    uptime_seconds: float
    total_requests: int
    total_documents: int
    total_pages: int
    average_pages_per_second: float
    average_confidence: float
    errors_last_hour: int
    current_queue_size: int


class ServiceStatus(BaseModel):
    """Individual service status"""
    status: str = Field(..., pattern="^(up|down|degraded)$")
    message: Optional[str]
    last_check: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    services: Dict[str, ServiceStatus]


class ReadinessResponse(BaseModel):
    """Readiness check response"""
    ready: bool
    models_loaded: bool
    gpu_available: bool
    message: str
    

class ErrorResponse(BaseModel):
    """Standard error response"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]
    request_id: str
    timestamp: datetime


class GPUMetrics(BaseModel):
    """GPU metrics"""
    gpu_id: int = Field(..., description="GPU device ID")
    name: str = Field(..., description="GPU name")
    memory_used: int = Field(..., description="Memory used (MB)")
    memory_total: int = Field(..., description="Total memory (MB)")
    memory_percent: float = Field(..., description="Memory usage percentage")
    gpu_utilization: float = Field(..., description="GPU utilization percentage")
    temperature: float = Field(..., description="GPU temperature (Celsius)")
    power_draw: float = Field(..., description="Power draw (Watts)")
    

class AsyncJobResponse(BaseModel):
    """Response for async job submission"""
    job_id: str = Field(..., description="Unique job ID")
    status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="Job status")
    created_at: datetime = Field(..., description="Job creation time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    status_url: str = Field(..., description="URL to check job status")
    result_url: Optional[str] = Field(None, description="URL to retrieve results when ready")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "created_at": "2024-01-20T10:30:00Z",
                "status_url": "/api/v1/jobs/job_550e8400-e29b-41d4-a716-446655440000/status"
            }
        }


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job ID")
    status: ProcessingStatus = Field(..., description="Current status")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    result: Optional[Union[OCRResponse, BatchOCRResponse]] = Field(None, description="Results if completed")
    error: Optional[ErrorResponse] = Field(None, description="Error if failed")


# Data Structures
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    file_info: 'FileInfo'


@dataclass
class FileInfo:
    filename: str
    size_bytes: int
    mime_type: str
    extension: str


@dataclass
class ServiceHealth:
    model_loaded: bool
    memory_available: bool
    queue_status: str
    last_inference: Optional[datetime]


@dataclass
class GPUHealth:
    gpu_available: bool
    temperature_ok: bool
    memory_ok: bool
    utilization: float


@dataclass
class TokenBucket:
    capacity: int
    tokens: float
    last_refill: datetime
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        # Implementation would go here
        pass
    
    def refill(self):
        """Refill bucket based on time passed"""
        # Implementation would go here
        pass


# Additional response schemas
class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str
    timestamp: datetime


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: Optional[str] = None
    gpu_available: bool
    gpu_info: Optional[Dict[str, Any]] = None
    models_loaded: bool
    cache_enabled: bool


class HealthResponse(BaseModel):
    """Health endpoint response"""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    services: Dict[str, ServiceStatus]


class ReadinessResponse(BaseModel):
    """Readiness probe response"""
    ready: bool
    models_loaded: bool
    gpu_available: bool
    message: str