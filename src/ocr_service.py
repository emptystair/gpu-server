"""
OCR Service Module

Main orchestration service for document OCR processing. Coordinates between
PDF/image processing, GPU monitoring, batch optimization, and result formatting.
"""

import asyncio
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import hashlib
import json
import magic

from .config import Config, OCRConfig
from .gpu_monitor import GPUMonitor, MemoryInfo
from .models.paddle_ocr import PaddleOCRWrapper, OCROutput
from .models.tensorrt_optimizer import TensorRTOptimizer
from .models.result_formatter import ResultFormatter, PageMetadata, PageResult as FormattedPageResult
from .utils.pdf_processor import PDFProcessor
from .utils.image_processor import ImageProcessor
from .utils.cache_manager import CacheManager
from .api.schemas import ProcessingStrategy, BoundingBox

logger = logging.getLogger(__name__)


# Data structures specific to OCR service
@dataclass
class WordResult:
    """Individual word detection result"""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


@dataclass
class PageResult:
    """OCR results for a single page"""
    page_number: int
    text: str
    words: List[WordResult]
    confidence: float
    processing_time_ms: float
    image_size: Tuple[int, int]  # width, height


@dataclass
class OCRResult:
    """Complete OCR processing results"""
    pages: List[PageResult]
    total_pages: int
    processing_time_ms: float
    confidence_score: float
    metadata: Dict[str, Any]
    strategy_used: ProcessingStrategy
    batch_sizes_used: List[int]


@dataclass
class ProcessingStats:
    """OCR service performance statistics"""
    total_documents_processed: int = 0
    total_pages_processed: int = 0
    average_pages_per_second: float = 0.0
    errors_count: int = 0
    cache_hit_rate: float = 0.0
    total_processing_time_ms: float = 0.0
    average_batch_size: float = 0.0
    gpu_utilization_average: float = 0.0


@dataclass
class ProcessingRequest:
    """Request for document processing"""
    document_path: str
    strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    language: str = "en"
    dpi: int = 150
    enable_gpu_optimization: bool = True
    batch_size: Optional[int] = None
    confidence_threshold: float = 0.5
    enable_angle_classification: bool = True
    preprocessing_options: Optional[Dict[str, Any]] = None
    page_range: Optional[str] = None  # For PDFs: "1-5,7,9-12"


@dataclass 
class ProcessingResult:
    """Result of document processing"""
    pages: List[PageResult]
    average_confidence: float
    processing_time_ms: float
    strategy_used: ProcessingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    average_batch_size: float = 0.0
    gpu_utilization_average: float = 0.0
    cache_hit_rate: float = 0.0
    total_processing_time_ms: float = 0.0
    errors_count: int = 0


class OCRService:
    """
    Main OCR orchestration service.
    
    Handles document processing pipeline from input to structured output,
    with GPU optimization, batch processing, and error recovery.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the OCR orchestration service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.ocr_config = config.ocr
        
        # Initialize components
        self.gpu_monitor = GPUMonitor(device_id=config.gpu.device_id)
        self.paddle_ocr = PaddleOCRWrapper(config=config.ocr)
        self.tensorrt_optimizer = TensorRTOptimizer(precision=config.ocr.tensorrt_precision)
        self.pdf_processor = PDFProcessor(config=config.ocr)
        self.image_processor = ImageProcessor()
        self.result_formatter = ResultFormatter(format_config=None)  # Use default config
        self.cache_manager = CacheManager(config=config.cache)
        
        # Processing statistics
        self.stats = ProcessingStats()
        self._processing_times: List[float] = []
        self._batch_sizes: List[int] = []
        self.is_initialized = False
        self._initializing = False
        
        # Processing strategy configurations
        self.strategy_configs = {
            ProcessingStrategy.SPEED: {
                "dpi": 120,
                "batch_multiplier": 1.5,
                "enhance_image": False,
                "use_tensorrt": True
            },
            ProcessingStrategy.ACCURACY: {
                "dpi": 200,
                "batch_multiplier": 0.7,
                "enhance_image": True,
                "use_tensorrt": True
            },
            ProcessingStrategy.BALANCED: {
                "dpi": 150,
                "batch_multiplier": 1.0,
                "enhance_image": True,
                "use_tensorrt": True
            }
        }
        
        logger.info(f"OCR Service initialized with config: {config.ocr}")
    
    async def initialize(self):
        """
        Initialize OCR models and warm up GPU.
        
        Performs model loading, TensorRT optimization, and GPU warmup.
        """
        try:
            logger.info("Initializing OCR Service...")
            self._initializing = True
            
            # Start GPU monitoring
            self.gpu_monitor.start_monitoring()
            
            # Initialize cache
            self.cache_manager.initialize()
            
            # Initialize PaddleOCR with TensorRT if available
            use_tensorrt = self.ocr_config.use_tensorrt
            self.paddle_ocr.initialize_model(use_tensorrt=use_tensorrt)
            
            # Attempt TensorRT optimization for even better performance
            if use_tensorrt:
                await self._optimize_models_with_tensorrt()
            
            # Warm up GPU with dummy inference
            await self._warmup_gpu()
            
            self.is_initialized = True
            self._initializing = False
            logger.info("OCR Service initialization complete")
            
        except Exception as e:
            self._initializing = False
            logger.error(f"Failed to initialize OCR Service: {e}")
            raise RuntimeError(f"OCR Service initialization failed: {e}")
    
    async def process_document(
        self,
        request: ProcessingRequest
    ) -> ProcessingResult:
        """
        Main entry point for document processing.
        
        Args:
            request: ProcessingRequest with document path and options
            
        Returns:
            ProcessingResult with structured output
        """
        if not self.is_initialized:
            raise RuntimeError("OCR Service not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            # Read file bytes
            with open(request.document_path, 'rb') as f:
                file_bytes = f.read()
            
            # Determine file type
            mime = magic.Magic(mime=True)
            file_type = mime.from_buffer(file_bytes[:8192])
            
            # Generate cache key
            cache_key = self._generate_cache_key(file_bytes, request.strategy)
            
            # Check cache
            cached_result = await self._check_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for document: {cache_key}")
                self.stats.cache_hit_rate = (self.stats.cache_hit_rate * self.stats.total_documents_processed + 1) / (self.stats.total_documents_processed + 1)
                # Convert OCRResult to ProcessingResult
                return ProcessingResult(
                    pages=cached_result.pages,
                    average_confidence=cached_result.confidence_score,
                    processing_time_ms=cached_result.processing_time_ms,
                    strategy_used=cached_result.strategy_used,
                    metadata=cached_result.metadata
                )
            
            # Get strategy configuration
            strategy_config = self.strategy_configs[request.strategy]
            dpi = request.dpi or strategy_config['dpi']
            
            # Prepare document (extract pages or load image)
            logger.info(f"Processing document with strategy: {request.strategy.value}")
            images = await self._prepare_document(file_bytes, file_type, dpi)
            
            if not images:
                raise ValueError("No images extracted from document")
            
            logger.info(f"Extracted {len(images)} pages from document")
            
            # Determine optimal batch size
            batch_size = self._determine_batch_size(
                page_count=len(images),
                image_dimensions=(images[0].shape[1], images[0].shape[0]),
                strategy=request.strategy
            )
            
            # Process in batches
            page_results = await self._process_batches(
                images=images,
                batch_size=batch_size,
                enhance=strategy_config['enhance_image']
            )
            
            # Format results
            ocr_result = self._format_results(
                page_results=page_results,
                processing_strategy=request.strategy,
                start_time=start_time
            )
            
            # Cache results
            await self._cache_result(cache_key, ocr_result)
            
            # Update statistics
            self._update_statistics(ocr_result)
            
            # Convert OCRResult to ProcessingResult
            return ProcessingResult(
                pages=ocr_result.pages,
                average_confidence=ocr_result.confidence_score,
                processing_time_ms=ocr_result.processing_time_ms,
                strategy_used=ocr_result.strategy_used,
                metadata=ocr_result.metadata
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self.stats.errors_count += 1
            raise
    
    async def _prepare_document(
        self,
        file_bytes: bytes,
        file_type: str,
        dpi: int = 150
    ) -> List[np.ndarray]:
        """
        Convert document to processable images.
        
        Args:
            file_bytes: Document content
            file_type: MIME type
            dpi: DPI for PDF extraction
            
        Returns:
            List of images as numpy arrays
        """
        images = []
        
        if file_type == 'application/pdf':
            # Extract pages from PDF
            images = self.pdf_processor.extract_pages(
                pdf_bytes=file_bytes,
                dpi=dpi
            )
        elif file_type.startswith('image/'):
            # Load single image
            image = self.image_processor.bytes_to_image(file_bytes)
            if image is not None:
                images = [image]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return images
    
    def _determine_batch_size(
        self,
        page_count: int,
        image_dimensions: Tuple[int, int],
        strategy: ProcessingStrategy
    ) -> int:
        """
        Calculate optimal batch size based on GPU resources.
        
        Args:
            page_count: Number of pages to process
            image_dimensions: Width and height of images
            strategy: Processing strategy
            
        Returns:
            Optimal batch size
        """
        try:
            # Get available GPU memory
            memory_info = self.gpu_monitor.get_available_memory()
            available_mb = memory_info.free - self.ocr_config.gpu_memory_buffer_mb
            
            # Estimate memory per image
            width, height = image_dimensions
            memory_per_image_mb = self._estimate_memory_requirement(1, (width, height))
            
            # Calculate base batch size
            base_batch_size = int(available_mb / memory_per_image_mb)
            
            # Apply strategy multiplier
            strategy_config = self.strategy_configs[strategy]
            batch_size = int(base_batch_size * strategy_config['batch_multiplier'])
            
            # Apply limits
            batch_size = max(self.ocr_config.min_batch_size, batch_size)
            batch_size = min(self.ocr_config.max_batch_size, batch_size)
            batch_size = min(batch_size, page_count)
            
            logger.info(f"Determined batch size: {batch_size} "
                       f"(available memory: {available_mb}MB, "
                       f"per image: {memory_per_image_mb}MB)")
            
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to determine optimal batch size: {e}")
            # Fall back to conservative batch size
            return min(8, page_count)
    
    async def _process_batches(
        self,
        images: List[np.ndarray],
        batch_size: int,
        enhance: bool = True
    ) -> List[PageResult]:
        """
        Process images in optimized batches.
        
        Args:
            images: List of images to process
            batch_size: Initial batch size
            enhance: Whether to enhance images
            
        Returns:
            List of page results
        """
        page_results = []
        current_batch_size = batch_size
        batch_sizes_used = []
        
        # Process images with optional enhancement
        if enhance:
            logger.info("Enhancing images for better OCR quality")
            enhanced_images = []
            for img in images:
                enhanced = self.image_processor.enhance_for_ocr(img)
                enhanced_images.append(enhanced)
            images = enhanced_images
        
        # Process in batches with adaptive sizing
        i = 0
        while i < len(images):
            batch_start = time.time()
            
            # Check memory pressure and adjust batch size
            if self.gpu_monitor.check_memory_pressure():
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"GPU memory pressure detected, reducing batch size to {current_batch_size}")
            
            # Get current batch
            batch_end = min(i + current_batch_size, len(images))
            batch = images[i:batch_end]
            batch_sizes_used.append(len(batch))
            
            try:
                # Process batch through PaddleOCR
                logger.info(f"Processing batch {i//current_batch_size + 1}: "
                           f"pages {i+1}-{batch_end} (batch size: {len(batch)})")
                
                ocr_outputs = self.paddle_ocr.process_batch(batch, batch_size=len(batch))
                
                # Convert OCR outputs to page results
                for idx, (img, ocr_output) in enumerate(zip(batch, ocr_outputs)):
                    page_num = i + idx + 1
                    page_result = self._convert_ocr_output(
                        ocr_output=ocr_output,
                        page_number=page_num,
                        image_shape=img.shape,
                        processing_time_ms=(time.time() - batch_start) * 1000 / len(batch)
                    )
                    page_results.append(page_result)
                
                # Successfully processed, try increasing batch size
                if current_batch_size < batch_size and not self.gpu_monitor.check_memory_pressure():
                    current_batch_size = min(current_batch_size * 2, batch_size)
                
                i = batch_end
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                
                # Retry with smaller batch size
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"Retrying with smaller batch size: {current_batch_size}")
                    continue
                else:
                    # Single image processing failed, skip this image
                    logger.error(f"Failed to process page {i+1}, skipping")
                    i += 1
        
        # Store batch sizes for statistics
        self._batch_sizes.extend(batch_sizes_used)
        
        return page_results
    
    def _convert_ocr_output(
        self,
        ocr_output: OCROutput,
        page_number: int,
        image_shape: Tuple[int, ...],
        processing_time_ms: float
    ) -> PageResult:
        """Convert PaddleOCR output to PageResult format"""
        words = []
        all_text = []
        total_confidence = 0.0
        
        for box, text, confidence in zip(ocr_output.boxes, ocr_output.texts, ocr_output.confidences):
            # Convert box coordinates to bbox
            coords = box.coordinates
            x_coords = [pt[0] for pt in coords]
            y_coords = [pt[1] for pt in coords]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            word = WordResult(
                text=text,
                bbox=bbox,
                confidence=confidence
            )
            words.append(word)
            all_text.append(text)
            total_confidence += confidence
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(words) if words else 0.0
        
        return PageResult(
            page_number=page_number,
            text=" ".join(all_text),
            words=words,
            confidence=avg_confidence,
            processing_time_ms=processing_time_ms,
            image_size=(image_shape[1], image_shape[0])  # width, height
        )
    
    def _format_results(
        self,
        page_results: List[PageResult],
        processing_strategy: ProcessingStrategy,
        start_time: float
    ) -> OCRResult:
        """
        Format raw OCR results into structured output.
        
        Args:
            page_results: List of page results
            processing_strategy: Strategy used
            start_time: Processing start time
            
        Returns:
            Formatted OCR result
        """
        # Calculate overall confidence
        total_confidence = sum(p.confidence for p in page_results)
        avg_confidence = total_confidence / len(page_results) if page_results else 0.0
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build metadata
        metadata = {
            "strategy": processing_strategy.value,
            "pages_processed": len(page_results),
            "average_confidence": avg_confidence,
            "processing_stages": {
                "extraction": sum(p.processing_time_ms for p in page_results) * 0.2,
                "ocr": sum(p.processing_time_ms for p in page_results) * 0.7,
                "formatting": sum(p.processing_time_ms for p in page_results) * 0.1
            },
            "gpu_info": {
                "device_info": self.gpu_monitor.get_device_info() if hasattr(self.gpu_monitor, 'get_device_info') else {},
                "memory_used_mb": self.gpu_monitor.get_available_memory().used_mb
            }
        }
        
        return OCRResult(
            pages=page_results,
            total_pages=len(page_results),
            processing_time_ms=processing_time_ms,
            confidence_score=avg_confidence,
            metadata=metadata,
            strategy_used=processing_strategy,
            batch_sizes_used=list(set(self._batch_sizes[-len(page_results):]))
        )
    
    def _estimate_memory_requirement(
        self,
        batch_size: int,
        image_dimensions: Tuple[int, int]
    ) -> int:
        """
        Estimate GPU memory needed for batch.
        
        Args:
            batch_size: Number of images in batch
            image_dimensions: Width and height of images
            
        Returns:
            Estimated memory in MB
        """
        width, height = image_dimensions
        
        # Base memory calculation: batch_size * width * height * channels * bytes_per_pixel
        image_memory = batch_size * width * height * 3 * 4 / (1024 * 1024)
        
        # Add overhead for model operations (detection + recognition)
        # Detection model overhead
        det_overhead = 200  # MB base
        det_overhead += batch_size * 50  # MB per image
        
        # Recognition model overhead
        rec_overhead = 150  # MB base
        rec_overhead += batch_size * 30  # MB per image
        
        # Total estimate with safety factor
        total_mb = (image_memory + det_overhead + rec_overhead) * 1.2
        
        return int(total_mb)
    
    async def _warmup_gpu(self):
        """Pre-allocate GPU resources with dummy inference."""
        try:
            logger.info("Warming up GPU with dummy inference...")
            
            # Create dummy images of different sizes
            dummy_sizes = [(640, 640), (960, 960), (1280, 720)]
            
            for size in dummy_sizes:
                dummy_image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
                
                # Run inference
                _ = self.paddle_ocr.process_batch([dummy_image], batch_size=1)
            
            logger.info("GPU warmup complete")
            
        except Exception as e:
            logger.warning(f"GPU warmup failed: {e}")
    
    async def _optimize_models_with_tensorrt(self):
        """Attempt to optimize models with TensorRT"""
        try:
            logger.info("Attempting TensorRT optimization...")
            
            # Optimize detection model if path is available
            if self.ocr_config.paddle_det_model_dir:
                det_result = self.tensorrt_optimizer.optimize_model(
                    paddle_model_path=self.ocr_config.paddle_det_model_dir,
                    output_path="/app/model_cache/tensorrt/det_optimized.engine",
                    model_type="detection"
                )
                if det_result:
                    logger.info(f"Detection model optimized: {det_result.speedup_factor:.2f}x speedup")
            
            # Optimize recognition model if path is available
            if self.ocr_config.paddle_rec_model_dir:
                rec_result = self.tensorrt_optimizer.optimize_model(
                    paddle_model_path=self.ocr_config.paddle_rec_model_dir,
                    output_path="/app/model_cache/tensorrt/rec_optimized.engine",
                    model_type="recognition"
                )
                if rec_result:
                    logger.info(f"Recognition model optimized: {rec_result.speedup_factor:.2f}x speedup")
                    
        except Exception as e:
            logger.warning(f"TensorRT optimization failed (will use standard inference): {e}")
    
    def get_processing_stats(self) -> ProcessingStats:
        """
        Return current processing statistics.
        
        Returns:
            Processing statistics
        """
        # Calculate averages
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            self.stats.average_pages_per_second = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        if self._batch_sizes:
            self.stats.average_batch_size = sum(self._batch_sizes) / len(self._batch_sizes)
        
        # Get GPU utilization
        try:
            gpu_util = self.gpu_monitor.get_gpu_utilization()
            self.stats.gpu_utilization_average = gpu_util.compute_percent
        except:
            pass
        
        return self.stats
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("Cleaning up OCR Service...")
            
            # Clean up PaddleOCR
            self.paddle_ocr.cleanup()
            
            # Stop GPU monitoring
            self.gpu_monitor.stop_monitoring()
            
            # Clear cache
            self.cache_manager.cleanup()
            
            logger.info("OCR Service cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _generate_cache_key(self, file_bytes: bytes, strategy: ProcessingStrategy) -> str:
        """Generate cache key for document"""
        content_hash = hashlib.md5(file_bytes).hexdigest()
        return f"{content_hash}_{strategy.value}"
    
    async def _check_cache(self, cache_key: str) -> Optional[OCRResult]:
        """Check cache for existing result"""
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            try:
                # Reconstruct OCRResult from cached data
                return self._deserialize_ocr_result(cached_data)
            except Exception as e:
                logger.warning(f"Failed to deserialize cached result: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: OCRResult):
        """Cache OCR result"""
        try:
            serialized = self._serialize_ocr_result(result)
            self.cache_manager.set(cache_key, serialized)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _serialize_ocr_result(self, result: OCRResult) -> Dict[str, Any]:
        """Serialize OCRResult for caching"""
        return {
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "words": [
                        {
                            "text": w.text,
                            "bbox": w.bbox,
                            "confidence": w.confidence
                        }
                        for w in p.words
                    ],
                    "confidence": p.confidence,
                    "processing_time_ms": p.processing_time_ms,
                    "image_size": p.image_size
                }
                for p in result.pages
            ],
            "total_pages": result.total_pages,
            "processing_time_ms": result.processing_time_ms,
            "confidence_score": result.confidence_score,
            "metadata": result.metadata,
            "strategy_used": result.strategy_used.value,
            "batch_sizes_used": result.batch_sizes_used
        }
    
    def _deserialize_ocr_result(self, data: Dict[str, Any]) -> OCRResult:
        """Deserialize OCRResult from cache"""
        pages = []
        for p_data in data["pages"]:
            words = [
                WordResult(
                    text=w["text"],
                    bbox=tuple(w["bbox"]),
                    confidence=w["confidence"]
                )
                for w in p_data["words"]
            ]
            
            page = PageResult(
                page_number=p_data["page_number"],
                text=p_data["text"],
                words=words,
                confidence=p_data["confidence"],
                processing_time_ms=p_data["processing_time_ms"],
                image_size=tuple(p_data["image_size"])
            )
            pages.append(page)
        
        return OCRResult(
            pages=pages,
            total_pages=data["total_pages"],
            processing_time_ms=data["processing_time_ms"],
            confidence_score=data["confidence_score"],
            metadata=data["metadata"],
            strategy_used=ProcessingStrategy(data["strategy_used"]),
            batch_sizes_used=data["batch_sizes_used"]
        )
    
    def _update_statistics(self, result: OCRResult):
        """Update processing statistics"""
        self.stats.total_documents_processed += 1
        self.stats.total_pages_processed += result.total_pages
        self.stats.total_processing_time_ms += result.processing_time_ms
        
        # Track processing times per page
        if result.pages:
            avg_page_time = result.processing_time_ms / len(result.pages)
            self._processing_times.append(avg_page_time)
            
            # Keep only last 100 times for rolling average
            if len(self._processing_times) > 100:
                self._processing_times = self._processing_times[-100:]
    
    @property
    def initialized(self) -> bool:
        """Check if service is initialized"""
        return self.is_initialized
    
    def is_ready(self) -> bool:
        """Check if service is ready to process documents"""
        return self.is_initialized and self.paddle_ocr.is_ready()
    
    def is_initializing(self) -> bool:
        """Check if service is currently initializing"""
        return self._initializing
    
    def get_queue_size(self) -> int:
        """Get current processing queue size"""
        # In this implementation, we don't have a queue
        # This is a placeholder for future queue implementation
        return 0
    
    async def shutdown(self):
        """Shutdown the OCR service"""
        logger.info("Shutting down OCR service...")
        # Cleanup resources if needed
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
        self.is_initialized = False