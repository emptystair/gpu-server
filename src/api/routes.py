"""
API Routes Module

Implements all REST API endpoints for the GPU OCR Server with comprehensive
file validation, error handling, and monitoring capabilities.
"""

import time
import asyncio
import logging
import hashlib
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import magic
from contextlib import asynccontextmanager
import httpx
from urllib.parse import urlparse
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse

from ..config import Config, load_config
from ..ocr_service import OCRService, ProcessingRequest, ProcessingResult
from ..gpu_monitor import GPUMonitor
from ..utils.cache_manager import CacheManager
from .schemas import (
    OCRLanguage, OutputFormat, ProcessingStrategy, ProcessingStatus,
    ImageProcessingOptions, OCRImageRequest, OCRUrlRequest, OCRPDFRequest,
    BatchOCRRequest, TextRegion, BoundingBox, OCRResponse as OCRResponseSchema,
    BatchOCRResponse, HealthCheckResponse, GPUMetrics, ErrorResponse,
    GPUStatusResponse, MemoryStatus, UtilizationStatus, StatsResponse,
    ServiceStatus, HealthResponse, ReadinessResponse, PageResponse,
    WordResponse, AsyncJobResponse, JobStatusResponse
)
from .middleware import get_request_id

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp'}
ALLOWED_PDF_TYPE = 'application/pdf'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.pdf'}

# Note: Services are initialized in main.py's lifespan manager
# and accessed via request.app.state in the dependency functions
# Global instances (no longer used - kept for reference)
# ocr_service: Optional[OCRService] = None
# gpu_monitor: Optional[GPUMonitor] = None
# cache_manager: Optional[CacheManager] = None
# config: Optional[Config] = None

# Statistics tracking
stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "total_documents": 0,
    "total_pages": 0,
    "total_errors": 0,
    "processing_times": [],
    "confidence_scores": [],
    "error_times": []
}

# Async jobs storage (in production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}


# Note: This lifespan manager is not used - main.py has its own
# @asynccontextmanager
# async def lifespan(app):
#     """Application lifespan manager"""
#     # Initialization is handled in main.py
#     pass


# Dependency injection
async def get_ocr_service(request: Request) -> OCRService:
    """Get OCR service instance"""
    ocr_service = getattr(request.app.state, 'ocr_service', None)
    if not ocr_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service not initialized"
        )
    return ocr_service


async def get_gpu_monitor(request: Request) -> GPUMonitor:
    """Get GPU monitor instance"""
    gpu_monitor = getattr(request.app.state, 'gpu_monitor', None)
    if not gpu_monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU monitor not initialized"
        )
    return gpu_monitor


async def get_cache_manager(request: Request) -> CacheManager:
    """Get cache manager instance"""
    cache_manager = getattr(request.app.state, 'cache_manager', None)
    if not cache_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager not initialized"
        )
    return cache_manager


# Create router
router = APIRouter(prefix="/api/v1", tags=["ocr"])


# File validation utilities
def validate_file_type(file_content: bytes, filename: str) -> str:
    """Validate file type using magic bytes"""
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(file_content)
        
        # Check if file type is allowed
        if file_type in ALLOWED_IMAGE_TYPES or file_type == ALLOWED_PDF_TYPE:
            return file_type
        
        # Check file extension as fallback
        ext = Path(filename).suffix.lower()
        if ext in ALLOWED_EXTENSIONS:
            if ext == '.pdf':
                return ALLOWED_PDF_TYPE
            else:
                return 'image/jpeg'  # Default for images
        
        raise ValueError(f"Unsupported file type: {file_type}")
        
    except Exception as e:
        logger.error(f"File type validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {str(e)}"
        )


def validate_file_size(file_size: int):
    """Validate file size"""
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"
        )


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary location"""
    try:
        # Create temporary file
        suffix = Path(upload_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Stream file content
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = 0
            
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                    
                total_size += len(chunk)
                validate_file_size(total_size)
                
                tmp_file.write(chunk)
            
            # Validate file type
            tmp_file.seek(0)
            first_chunk = tmp_file.read(8192)  # Read first 8KB for magic bytes
            validate_file_type(first_chunk, upload_file.filename)
            
        return tmp_path
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()
        raise


def cleanup_temp_file(file_path: Path):
    """Clean up temporary file"""
    try:
        if file_path and file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")


# OCR Processing Endpoints
@router.post("/ocr/process", response_model=OCRResponseSchema)
async def process_single_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    strategy: ProcessingStrategy = Form(ProcessingStrategy.BALANCED),
    dpi: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(0.5),
    language: OCRLanguage = Form(OCRLanguage.ENGLISH),
    output_format: OutputFormat = Form(OutputFormat.JSON),
    ocr_service: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Process a single document (image or PDF) for OCR.
    
    Accepts multipart/form-data with file upload and processing parameters.
    """
    request_id = get_request_id()
    start_time = time.time()
    tmp_path = None
    
    # Update statistics
    stats["total_requests"] += 1
    stats["total_documents"] += 1
    
    try:
        # Save uploaded file
        tmp_path = await save_upload_file(file)
        
        # Generate cache key
        with open(tmp_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        cache_key = cache.generate_key(
            document_hash=file_hash,
            strategy=strategy.value,
            dpi=dpi,
            language=language.value
        )
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for document {file.filename}")
            # Add cleanup task
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            
            # Update cached result with current request info
            cached_result["request_id"] = request_id
            cached_result["cache_hit"] = True
            return JSONResponse(content=cached_result)
        
        # Create processing request
        proc_request = ProcessingRequest(
            document_path=str(tmp_path),
            strategy=strategy,
            language=language.value,
            dpi=dpi or 150,
            enable_gpu_optimization=True,
            batch_size=1,
            confidence_threshold=confidence_threshold
        )
        
        # Store GPU utilization in request state
        monitor = getattr(request.app.state, 'gpu_monitor', None)
        if monitor:
            gpu_util = monitor.get_gpu_utilization()
            request.state.gpu_utilization = gpu_util.compute_percent
        else:
            request.state.gpu_utilization = 0
        
        # Process document
        result = await ocr_service.process_document(proc_request)
        
        # Format response
        processing_time = (time.time() - start_time) * 1000
        
        # Update statistics
        stats["total_pages"] += len(result.pages)
        stats["processing_times"].append(processing_time)
        if len(stats["processing_times"]) > 1000:
            stats["processing_times"] = stats["processing_times"][-1000:]
        
        # Build response
        response_data = {
            "request_id": request_id,
            "status": ProcessingStatus.COMPLETED.value,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "total_regions": sum(len(page.words) for page in result.pages),
            "average_confidence": result.average_confidence,
            "language_detected": language.value,
            # "gpu_info": {
            #     "gpu_id": 0,  # Default GPU ID
            #     "utilization": gpu_util.compute_percent
            # },
            "cache_hit": False
        }
        
        # Add content based on output format
        if output_format == OutputFormat.JSON:
            response_data["regions"] = []
            response_data["pages"] = []
            
            for page in result.pages:
                page_data = {
                    "page_number": page.page_number,
                    "text": page.text,
                    "regions": [
                        {
                            "text": word.text,
                            "confidence": word.confidence,
                            "bbox": {
                                "x1": word.bbox[0],
                                "y1": word.bbox[1],
                                "x2": word.bbox[2],
                                "y2": word.bbox[3]
                            }
                        }
                        for word in page.words
                    ]
                }
                response_data["pages"].append(page_data)
                response_data["regions"].extend(page_data["regions"])
                
        elif output_format == OutputFormat.TEXT:
            response_data["text"] = "\n\n".join(page.text for page in result.pages)
            
        elif output_format == OutputFormat.MARKDOWN:
            response_data["text"] = "\n\n---\n\n".join(
                f"## Page {page.page_number}\n\n{page.text}" 
                for page in result.pages
            )
            
        elif output_format == OutputFormat.HTML:
            html_pages = []
            for page in result.pages:
                html_pages.append(
                    f'<div class="page" data-page="{page.page_number}">\n'
                    f'<h2>Page {page.page_number}</h2>\n'
                    f'<div class="content">{page.text}</div>\n'
                    f'</div>'
                )
            response_data["text"] = "\n".join(html_pages)
        
        # Cache the result
        cache.set(cache_key, response_data, ttl=3600)
        
        # Add cleanup task
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        # Update confidence scores
        stats["confidence_scores"].append(result.average_confidence)
        if len(stats["confidence_scores"]) > 1000:
            stats["confidence_scores"] = stats["confidence_scores"][-1000:]
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        stats["total_errors"] += 1
        stats["error_times"].append(time.time())
        raise
    except Exception as e:
        stats["total_errors"] += 1
        stats["error_times"].append(time.time())
        logger.error(f"Document processing failed: {e}", exc_info=True)
        
        # Clean up file if exists
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.post("/ocr/process-batch", response_model=BatchOCRResponse)
async def process_batch_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    strategy: ProcessingStrategy = Form(ProcessingStrategy.BALANCED),
    dpi: Optional[int] = Form(None),
    confidence_threshold: Optional[float] = Form(0.5),
    language: OCRLanguage = Form(OCRLanguage.ENGLISH),
    parallel_processing: bool = Form(True),
    ocr_service: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Process multiple documents in batch.
    
    Supports concurrent processing with partial failure handling.
    """
    request_id = get_request_id()
    batch_id = f"batch_{request_id}"
    start_time = time.time()
    tmp_paths = []
    
    # Validate batch size
    if len(files) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100 files"
        )
    
    # Update statistics
    stats["total_requests"] += 1
    
    try:
        # Process each file
        results = []
        successful = 0
        failed = 0
        
        async def process_file(file: UploadFile, index: int) -> Dict[str, Any]:
            """Process individual file in batch"""
            tmp_path = None
            try:
                # Save file
                tmp_path = await save_upload_file(file)
                tmp_paths.append(tmp_path)
                
                # Generate cache key
                with open(tmp_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                cache_key = cache.generate_key(
                    document_hash=file_hash,
                    strategy=strategy.value,
                    dpi=dpi,
                    language=language.value
                )
                
                # Check cache
                cached_result = cache.get(cache_key)
                if cached_result:
                    cached_result["cache_hit"] = True
                    cached_result["file_name"] = file.filename
                    return cached_result
                
                # Process
                proc_request = ProcessingRequest(
                    document_path=str(tmp_path),
                    strategy=strategy,
                    language=language.value,
                    dpi=dpi or 150,
                    enable_gpu_optimization=True,
                    batch_size=1,
                    confidence_threshold=confidence_threshold
                )
                
                result = await ocr_service.process_document(proc_request)
                
                # Format result
                doc_result = {
                    "file_name": file.filename,
                    "status": ProcessingStatus.COMPLETED.value,
                    "pages": len(result.pages),
                    "text": "\n\n".join(page.text for page in result.pages),
                    "confidence": result.average_confidence,
                    "processing_time": result.processing_time_ms,
                    "cache_hit": False
                }
                
                # Cache result
                cache.set(cache_key, doc_result, ttl=3600)
                
                return doc_result
                
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                return {
                    "file_name": file.filename,
                    "status": ProcessingStatus.FAILED.value,
                    "error": str(e)
                }
        
        # Process files
        if parallel_processing:
            # Concurrent processing
            tasks = [
                process_file(file, i) 
                for i, file in enumerate(files)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing
            results = []
            for i, file in enumerate(files):
                result = await process_file(file, i)
                results.append(result)
        
        # Count successes and failures
        for result in results:
            if isinstance(result, dict):
                if result.get("status") == ProcessingStatus.COMPLETED.value:
                    successful += 1
                    stats["total_documents"] += 1
                    stats["total_pages"] += result.get("pages", 0)
                else:
                    failed += 1
                    stats["total_errors"] += 1
            else:
                failed += 1
                stats["total_errors"] += 1
        
        # Clean up all temp files
        for tmp_path in tmp_paths:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        # Build response
        total_processing_time = (time.time() - start_time) * 1000
        
        return {
            "request_id": batch_id,
            "status": ProcessingStatus.COMPLETED.value,
            "timestamp": datetime.utcnow().isoformat(),
            "total_images": len(files),
            "successful": successful,
            "failed": failed,
            "results": results,
            "total_processing_time": total_processing_time
        }
        
    except Exception as e:
        # Clean up files on error
        for tmp_path in tmp_paths:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.post("/ocr/image", response_model=OCRResponseSchema)
async def process_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr_request: OCRImageRequest = Depends(),
    ocr_service: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Process a single image file for OCR.
    
    This endpoint is specifically for image files with full request body validation.
    """
    request_id = get_request_id()
    start_time = time.time()
    tmp_path = None
    
    # Update statistics
    stats["total_requests"] += 1
    stats["total_documents"] += 1
    
    try:
        # Validate file is an image
        tmp_path = await save_upload_file(file)
        with open(tmp_path, 'rb') as f:
            file_content = f.read(8192)
        file_type = validate_file_type(file_content, file.filename)
        
        if file_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be an image. Got: {file_type}"
            )
        
        # Generate cache key
        with open(tmp_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Generate cache key similar to OCR service
        cache_key = f"{file_hash}_{ProcessingStrategy.BALANCED.value}_{ocr_request.language.value}"
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            cached_result["request_id"] = request_id
            cached_result["cache_hit"] = True
            return JSONResponse(content=cached_result)
        
        # Process image
        proc_request = ProcessingRequest(
            document_path=str(tmp_path),
            strategy=ProcessingStrategy.BALANCED,
            language=ocr_request.language.value,
            dpi=150,
            enable_gpu_optimization=True,
            batch_size=1,
            confidence_threshold=ocr_request.confidence_threshold,
            enable_angle_classification=ocr_request.enable_angle_classification,
            preprocessing_options=ocr_request.preprocessing
        )
        
        result = await ocr_service.process_document(proc_request)
        
        # Format response based on output format
        processing_time = (time.time() - start_time) * 1000
        response_data = _format_ocr_response(
            result, request_id, processing_time, 
            ocr_request.output_format, ocr_request.language.value,
            preserve_layout=ocr_request.preserve_layout
        )
        
        # Cache result
        cache.set(cache_key, response_data, ttl=3600)
        
        # Cleanup
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        stats["total_errors"] += 1
        raise
    except Exception as e:
        stats["total_errors"] += 1
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {str(e)}"
        )


@router.post("/ocr/url", response_model=OCRResponseSchema)
async def process_image_url(
    request: Request,
    background_tasks: BackgroundTasks,
    url_request: OCRUrlRequest,
    ocr_service: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Process an image from URL for OCR.
    
    Downloads the image and processes it.
    """
    request_id = get_request_id()
    start_time = time.time()
    tmp_path = None
    
    # Update statistics
    stats["total_requests"] += 1
    stats["total_documents"] += 1
    
    try:
        # Validate URL
        parsed_url = urlparse(url_request.image_url)
        if not parsed_url.scheme in ['http', 'https']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only HTTP(S) URLs are supported"
            )
        
        # Download image
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url_request.image_url)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').split(';')[0].strip()
            if content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"URL must point to an image. Got: {content_type}"
                )
            
            # Save to temp file
            suffix = Path(parsed_url.path).suffix or '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(response.content)
        
        # Generate cache key
        file_hash = hashlib.sha256(response.content).hexdigest()
        cache_key = cache.generate_key(
            document_hash=file_hash,
            strategy=url_request.strategy.value if url_request.strategy else ProcessingStrategy.BALANCED.value,
            dpi=150,
            language=url_request.language.value
        )
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            cached_result["request_id"] = request_id
            cached_result["cache_hit"] = True
            return JSONResponse(content=cached_result)
        
        # Process image
        proc_request = ProcessingRequest(
            document_path=str(tmp_path),
            strategy=url_request.strategy or ProcessingStrategy.BALANCED,
            language=url_request.language.value,
            dpi=150,
            enable_gpu_optimization=True,
            batch_size=1,
            confidence_threshold=url_request.confidence_threshold,
            enable_angle_classification=url_request.enable_angle_classification,
            preprocessing_options=url_request.preprocessing
        )
        
        result = await ocr_service.process_document(proc_request)
        
        # Format response
        processing_time = (time.time() - start_time) * 1000
        response_data = _format_ocr_response(
            result, request_id, processing_time,
            url_request.output_format, url_request.language.value,
            preserve_layout=url_request.preserve_layout
        )
        
        # Cache result
        cache.set(cache_key, response_data, ttl=3600)
        
        # Cleanup
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        return JSONResponse(content=response_data)
        
    except httpx.HTTPError as e:
        stats["total_errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download image: {str(e)}"
        )
    except HTTPException:
        stats["total_errors"] += 1
        raise
    except Exception as e:
        stats["total_errors"] += 1
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"URL processing failed: {str(e)}"
        )


@router.post("/ocr/pdf", response_model=OCRResponseSchema)
async def process_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pdf_request: OCRPDFRequest = Depends(),
    ocr_service: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """
    Process a PDF file for OCR.
    
    Supports page range selection and merged output.
    """
    request_id = get_request_id()
    start_time = time.time()
    tmp_path = None
    
    # Update statistics
    stats["total_requests"] += 1
    stats["total_documents"] += 1
    
    try:
        # Validate file is PDF
        tmp_path = await save_upload_file(file)
        with open(tmp_path, 'rb') as f:
            file_content = f.read(8192)
        file_type = validate_file_type(file_content, file.filename)
        
        if file_type != ALLOWED_PDF_TYPE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File must be a PDF. Got: {file_type}"
            )
        
        # Generate cache key
        with open(tmp_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        cache_key = cache.generate_key(
            document_hash=file_hash,
            strategy=ProcessingStrategy.BALANCED.value,
            dpi=150,
            language=pdf_request.language.value,
            page_range=pdf_request.page_range
        )
        
        # Check cache
        cached_result = cache.get(cache_key)
        if cached_result:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
            cached_result["request_id"] = request_id
            cached_result["cache_hit"] = True
            return JSONResponse(content=cached_result)
        
        # Process PDF
        proc_request = ProcessingRequest(
            document_path=str(tmp_path),
            strategy=ProcessingStrategy.BALANCED,
            language=pdf_request.language.value,
            dpi=150,
            enable_gpu_optimization=True,
            batch_size=1,
            confidence_threshold=pdf_request.confidence_threshold,
            preprocessing_options=pdf_request.preprocessing,
            page_range=pdf_request.page_range
        )
        
        result = await ocr_service.process_document(proc_request)
        
        # Format response
        processing_time = (time.time() - start_time) * 1000
        stats["total_pages"] += len(result.pages)
        
        response_data = _format_ocr_response(
            result, request_id, processing_time,
            pdf_request.output_format, pdf_request.language.value,
            merge_pages=pdf_request.merge_pages
        )
        
        # Cache result
        cache.set(cache_key, response_data, ttl=3600)
        
        # Cleanup
        background_tasks.add_task(cleanup_temp_file, tmp_path)
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        stats["total_errors"] += 1
        raise
    except Exception as e:
        stats["total_errors"] += 1
        if tmp_path:
            background_tasks.add_task(cleanup_temp_file, tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF processing failed: {str(e)}"
        )


# Async Job Management Endpoints
@router.post("/jobs/submit", response_model=AsyncJobResponse)
async def submit_async_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    language: OCRLanguage = Form(OCRLanguage.ENGLISH),
    output_format: OutputFormat = Form(OutputFormat.JSON),
    strategy: ProcessingStrategy = Form(ProcessingStrategy.BALANCED),
    dpi: Optional[int] = Form(None),
    confidence_threshold: float = Form(0.5),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    Submit an OCR job for asynchronous processing.
    
    Returns immediately with a job ID for status tracking.
    """
    request_id = get_request_id()
    
    # Validate input
    if not file and not image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either file or image_url must be provided"
        )
    
    if file and image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of file or image_url should be provided"
        )
    
    # Generate job ID
    job_id = f"job_{uuid.uuid4()}"
    
    # Create job entry
    job_data = {
        "job_id": job_id,
        "status": ProcessingStatus.PENDING,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "request_id": request_id,
        "parameters": {
            "language": language.value,
            "output_format": output_format.value,
            "strategy": strategy.value,
            "dpi": dpi or 150,
            "confidence_threshold": confidence_threshold
        }
    }
    
    # Store job
    jobs[job_id] = job_data
    
    # Process job in background
    background_tasks.add_task(
        _process_async_job,
        job_id,
        file,
        image_url,
        job_data["parameters"],
        ocr_service
    )
    
    # Calculate estimated completion (simple estimate based on file size)
    estimated_seconds = 10  # Default estimate
    if file:
        file_size_mb = file.size / (1024 * 1024) if hasattr(file, 'size') else 1
        estimated_seconds = max(5, int(file_size_mb * 5))  # 5 seconds per MB
    
    return AsyncJobResponse(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        created_at=job_data["created_at"],
        estimated_completion=job_data["created_at"] + timedelta(seconds=estimated_seconds),
        status_url=f"/api/v1/jobs/{job_id}/status",
        result_url=f"/api/v1/jobs/{job_id}/result"
    )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of an async OCR job.
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    # Calculate progress (simplified)
    progress = None
    if job["status"] == ProcessingStatus.PROCESSING:
        elapsed = (datetime.utcnow() - job["created_at"]).total_seconds()
        progress = min(90.0, elapsed * 10)  # Rough estimate
    elif job["status"] == ProcessingStatus.COMPLETED:
        progress = 100.0
    elif job["status"] == ProcessingStatus.FAILED:
        progress = 0.0
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress,
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error")
    )


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the results of a completed async OCR job.
    
    Returns 404 if job not found, 202 if still processing, or results if completed.
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    if job["status"] == ProcessingStatus.PENDING:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Job is pending",
                "status_url": f"/api/v1/jobs/{job_id}/status"
            }
        )
    
    if job["status"] == ProcessingStatus.PROCESSING:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Job is still processing",
                "status_url": f"/api/v1/jobs/{job_id}/status",
                "progress": job.get("progress", 0)
            }
        )
    
    if job["status"] == ProcessingStatus.FAILED:
        error = job.get("error", {"message": "Processing failed"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.get("message", "Processing failed")
        )
    
    # Return completed result
    return job.get("result", {})


# Helper functions
def _format_ocr_response(
    result: ProcessingResult,
    request_id: str,
    processing_time: float,
    output_format: OutputFormat,
    language: str,
    preserve_layout: bool = False,
    merge_pages: bool = True
) -> Dict[str, Any]:
    """Format OCR result based on output format"""
    
    response_data = {
        "request_id": request_id,
        "status": ProcessingStatus.COMPLETED.value,
        "timestamp": datetime.utcnow().isoformat(),
        "processing_time": processing_time,
        "total_regions": sum(len(page.words) for page in result.pages),
        "average_confidence": result.average_confidence,
        "language_detected": language,
        "cache_hit": False
    }
    
    if output_format == OutputFormat.JSON:
        response_data["regions"] = []
        response_data["pages"] = []
        
        for page in result.pages:
            page_data = {
                "page_number": page.page_number,
                "text": page.text,
                "regions": [
                    {
                        "text": word.text,
                        "confidence": word.confidence,
                        "bbox": {
                            "x1": word.bbox[0],
                            "y1": word.bbox[1],
                            "x2": word.bbox[2],
                            "y2": word.bbox[3]
                        }
                    }
                    for word in page.words
                ]
            }
            response_data["pages"].append(page_data)
            response_data["regions"].extend(page_data["regions"])
            
    elif output_format == OutputFormat.TEXT:
        if merge_pages:
            response_data["text"] = "\n\n".join(page.text for page in result.pages)
        else:
            response_data["pages"] = [
                {"page_number": page.page_number, "text": page.text}
                for page in result.pages
            ]
            
    elif output_format == OutputFormat.MARKDOWN:
        if merge_pages:
            response_data["text"] = "\n\n---\n\n".join(
                f"## Page {page.page_number}\n\n{page.text}"
                for page in result.pages
            )
        else:
            response_data["pages"] = [
                {
                    "page_number": page.page_number,
                    "text": f"## Page {page.page_number}\n\n{page.text}"
                }
                for page in result.pages
            ]
            
    elif output_format == OutputFormat.HTML:
        if merge_pages:
            html_pages = []
            for page in result.pages:
                html_pages.append(
                    f'<div class="page" data-page="{page.page_number}">\n'
                    f'<h2>Page {page.page_number}</h2>\n'
                    f'<div class="content">{page.text}</div>\n'
                    f'</div>'
                )
            response_data["text"] = "\n".join(html_pages)
        else:
            response_data["pages"] = [
                {
                    "page_number": page.page_number,
                    "text": f'<div class="page" data-page="{page.page_number}">\n'
                           f'<h2>Page {page.page_number}</h2>\n'
                           f'<div class="content">{page.text}</div>\n'
                           f'</div>'
                }
                for page in result.pages
            ]
    
    # Update statistics
    stats["confidence_scores"].append(result.average_confidence)
    if len(stats["confidence_scores"]) > 1000:
        stats["confidence_scores"] = stats["confidence_scores"][-1000:]
    
    return response_data


async def _process_async_job(
    job_id: str,
    file: Optional[UploadFile],
    image_url: Optional[str],
    parameters: Dict[str, Any],
    ocr_service: OCRService
):
    """Process async OCR job in background"""
    tmp_path = None
    
    try:
        # Update job status
        jobs[job_id]["status"] = ProcessingStatus.PROCESSING
        jobs[job_id]["updated_at"] = datetime.utcnow()
        
        # Process based on input type
        if file:
            # Save uploaded file
            tmp_path = await save_upload_file(file)
            document_path = str(tmp_path)
        else:
            # Download from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Save to temp file
                suffix = Path(urlparse(image_url).path).suffix or '.jpg'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    tmp_file.write(response.content)
                document_path = str(tmp_path)
        
        # Create processing request
        proc_request = ProcessingRequest(
            document_path=document_path,
            strategy=ProcessingStrategy(parameters["strategy"]),
            language=parameters["language"],
            dpi=parameters["dpi"],
            enable_gpu_optimization=True,
            batch_size=1,
            confidence_threshold=parameters["confidence_threshold"]
        )
        
        # Process document
        result = await ocr_service.process_document(proc_request)
        
        # Format result
        response_data = _format_ocr_response(
            result,
            jobs[job_id]["request_id"],
            result.processing_time_ms,
            OutputFormat(parameters["output_format"]),
            parameters["language"]
        )
        
        # Update job with result
        jobs[job_id]["status"] = ProcessingStatus.COMPLETED
        jobs[job_id]["completed_at"] = datetime.utcnow()
        jobs[job_id]["updated_at"] = datetime.utcnow()
        jobs[job_id]["result"] = response_data
        
        # Update statistics
        stats["total_documents"] += 1
        stats["total_pages"] += len(result.pages)
        
    except Exception as e:
        logger.error(f"Async job {job_id} failed: {e}", exc_info=True)
        
        # Update job with error
        jobs[job_id]["status"] = ProcessingStatus.FAILED
        jobs[job_id]["updated_at"] = datetime.utcnow()
        jobs[job_id]["error"] = {
            "error": type(e).__name__,
            "message": str(e),
            "request_id": jobs[job_id]["request_id"],
            "timestamp": datetime.utcnow()
        }
        
        stats["total_errors"] += 1
    
    finally:
        # Cleanup temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


# GPU Monitoring Endpoints
@router.get("/gpu/status", response_model=GPUStatusResponse)
async def get_gpu_status(
    monitor: GPUMonitor = Depends(get_gpu_monitor)
):
    """Get current GPU status and utilization metrics"""
    try:
        # Get various GPU metrics
        memory_info = monitor.get_available_memory()
        gpu_util = monitor.get_gpu_utilization()
        device_info = monitor.get_device_info()
        
        return GPUStatusResponse(
            device_id=device_info.get("device_id", 0),
            device_name=device_info.get("name", "No GPU"),
            memory=MemoryStatus(
                total_mb=memory_info.total_mb,
                used_mb=memory_info.used_mb,
                free_mb=memory_info.free_mb,
                reserved_mb=0  # Not tracked separately in current implementation
            ),
            utilization=UtilizationStatus(
                compute_percent=gpu_util.compute_percent,
                memory_percent=gpu_util.memory_percent,
                encoder_percent=None,
                decoder_percent=None
            ),
            temperature_celsius=gpu_util.temperature_celsius,
            power_draw_watts=gpu_util.power_draw_watts
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU monitoring unavailable"
        )


# Statistics Endpoint
@router.get("/stats", response_model=StatsResponse)
async def get_statistics(
    ocr: OCRService = Depends(get_ocr_service),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get processing statistics"""
    try:
        # Calculate metrics
        uptime = time.time() - stats["start_time"]
        
        # Average processing speed
        avg_pages_per_second = 0
        if stats["processing_times"]:
            total_time_seconds = sum(stats["processing_times"]) / 1000
            if total_time_seconds > 0:
                avg_pages_per_second = stats["total_pages"] / total_time_seconds
        
        # Average confidence
        avg_confidence = 0
        if stats["confidence_scores"]:
            avg_confidence = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
        
        # Errors in last hour
        current_time = time.time()
        errors_last_hour = sum(
            1 for t in stats["error_times"] 
            if current_time - t < 3600
        )
        
        # Get cache stats
        cache_stats = cache.get_stats()
        
        return {
            "uptime_seconds": uptime,
            "total_requests": stats["total_requests"],
            "total_documents": stats["total_documents"],
            "total_pages": stats["total_pages"],
            "average_pages_per_second": avg_pages_per_second,
            "average_confidence": avg_confidence,
            "errors_last_hour": errors_last_hour,
            "current_queue_size": ocr.get_queue_size() if hasattr(ocr, 'get_queue_size') else 0,
            "cache_hit_rate": cache_stats.get("overall_hit_rate", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


# Health Check Endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic health check endpoint"""
    try:
        # Check services
        services = {}
        
        # OCR Service
        service = getattr(request.app.state, 'ocr_service', None)
        if service and service.initialized:
            services["ocr"] = ServiceStatus(
                status="up",
                message="OCR service operational",
                last_check=datetime.utcnow()
            )
        else:
            services["ocr"] = ServiceStatus(
                status="down",
                message="OCR service not initialized",
                last_check=datetime.utcnow()
            )
        
        # GPU Monitor
        monitor = getattr(request.app.state, 'gpu_monitor', None)
        if monitor:
            try:
                device_info = monitor.get_device_info()
                if device_info.get("available"):
                    services["gpu"] = ServiceStatus(
                        status="up",
                        message="GPU monitoring operational",
                        last_check=datetime.utcnow()
                    )
                else:
                    services["gpu"] = ServiceStatus(
                        status="degraded",
                        message="GPU not available",
                        last_check=datetime.utcnow()
                    )
            except:
                services["gpu"] = ServiceStatus(
                    status="degraded",
                    message="GPU monitoring unavailable",
                    last_check=datetime.utcnow()
                )
        
        # Cache
        cache_manager = getattr(request.app.state, 'cache_manager', None)
        if cache_manager:
            services["cache"] = ServiceStatus(
                status="up",
                message="Cache service operational",
                last_check=datetime.utcnow()
            )
        
        # Determine overall status
        statuses = [s.status for s in services.values()]
        if all(s == "up" for s in statuses):
            overall_status = "healthy"
        elif any(s == "down" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "version": getattr(getattr(request.app.state, 'config', None), 'version', "1.0.0"),
            "services": services
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": "unknown",
            "services": {}
        }


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    request: Request,
    ocr: OCRService = Depends(get_ocr_service)
):
    """
    Detailed readiness check for Kubernetes.
    
    Checks if the service is ready to handle requests.
    """
    try:
        # Check if models are loaded
        models_loaded = ocr.initialized and ocr.is_ready()
        
        # Check GPU availability
        gpu_available = False
        monitor = getattr(request.app.state, 'gpu_monitor', None)
        if monitor:
            try:
                device_info = monitor.get_device_info()
                gpu_available = device_info.get("available", False)
            except:
                pass
        else:
            # GPU monitor not available, assume GPU is available if OCR is using it
            gpu_available = True
        
        # Check if initialization is in progress
        initializing = ocr.is_initializing() if hasattr(ocr, 'is_initializing') else False
        
        # Determine readiness
        ready = models_loaded and gpu_available and not initializing
        
        message = "Service is ready"
        if not ready:
            reasons = []
            if not models_loaded:
                reasons.append("Models not loaded")
            if not gpu_available:
                reasons.append("GPU not available")
            if initializing:
                reasons.append("Initialization in progress")
            message = f"Service not ready: {', '.join(reasons)}"
        
        return {
            "ready": ready,
            "models_loaded": models_loaded,
            "gpu_available": gpu_available,
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "models_loaded": False,
            "gpu_available": False,
            "message": f"Readiness check failed: {str(e)}"
        }


# Metrics endpoint for Prometheus (optional)
@router.get("/metrics")
async def get_metrics(request: Request):
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    # Request metrics
    metrics.append(f'ocr_requests_total {stats["total_requests"]}')
    metrics.append(f'ocr_documents_total {stats["total_documents"]}')
    metrics.append(f'ocr_pages_total {stats["total_pages"]}')
    metrics.append(f'ocr_errors_total {stats["total_errors"]}')
    
    # Processing time histogram
    if stats["processing_times"]:
        for percentile in [50, 90, 95, 99]:
            sorted_times = sorted(stats["processing_times"])
            index = int(len(sorted_times) * (percentile / 100))
            value = sorted_times[min(index, len(sorted_times) - 1)]
            metrics.append(f'ocr_processing_time_ms{{quantile="{percentile/100}"}} {value}')
    
    # GPU metrics
    try:
        monitor = getattr(request.app.state, 'gpu_monitor', None)
        if monitor:
            gpu_util = monitor.get_gpu_utilization()
            memory_info = monitor.get_available_memory()
            metrics.append(f'gpu_utilization_percent {gpu_util.compute_percent}')
            metrics.append(f'gpu_memory_used_mb {memory_info.used_mb}')
            metrics.append(f'gpu_temperature_celsius {gpu_util.temperature_celsius}')
    except:
        pass
    
    return "\n".join(metrics)