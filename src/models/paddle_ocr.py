"""
PaddleOCR Wrapper Module

Provides a high-performance wrapper around PaddleOCR optimized for RTX 4090.
Handles batch processing, GPU memory management, and optional TensorRT acceleration.
"""

import os
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    paddle = None
    PaddleOCR = None

from ..config import OCRConfig
from ..gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    """Text detection bounding box"""
    coordinates: List[Tuple[int, int]]  # Polygon points
    confidence: float


@dataclass
class TextRecognition:
    """Text recognition result"""
    text: str
    confidence: float


@dataclass
class OCROutput:
    """Complete OCR output for an image"""
    boxes: List[TextBox]
    texts: List[str]
    confidences: List[float]


@dataclass
class ModelInfo:
    """Model status and metadata"""
    model_version: str
    optimization_enabled: bool
    current_batch_size: int
    total_inferences: int
    average_latency_ms: float


class PaddleOCRWrapper:
    """
    Wrapper for PaddleOCR with RTX 4090 optimizations.
    
    Features:
    - Batch processing with dynamic sizing
    - GPU memory management
    - Optional TensorRT acceleration
    - Error recovery and logging
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize PaddleOCR with TensorRT backend support.
        
        Args:
            config: OCR configuration with model paths and settings
        """
        self.config = config
        self.ocr = None
        self.gpu_monitor = GPUMonitor()
        
        # Performance tracking
        self.total_inferences = 0
        self.total_latency_ms = 0.0
        self.model_version = "en_PP-OCRv4"
        
        # RTX 4090 optimizations
        self._configure_paddle_flags()
        
        if not PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle is not installed. OCR functionality will be limited.")
    
    def _configure_paddle_flags(self):
        """Configure PaddlePaddle flags for RTX 4090 optimization"""
        if paddle is None:
            return
            
        # Set environment variables for flags that can't be set at runtime
        import os
        os.environ['FLAGS_use_tf32'] = '1'
        
        # Set other flags that can be set at runtime
        try:
            paddle.set_flags({
                'FLAGS_cudnn_exhaustive_search': True,
                'FLAGS_conv_workspace_size_limit': 512,  # MB
                'FLAGS_allocator_strategy': 'auto_growth'
            })
        except ValueError as e:
            logger.warning(f"Could not set some PaddlePaddle flags: {e}")
        
        # Set device (default to GPU 0)
        try:
            paddle.set_device('gpu:0')
        except Exception as e:
            logger.warning(f"Could not set GPU device, using CPU: {e}")
            paddle.set_device('cpu')
        
        logger.info("Configured PaddlePaddle for RTX 4090 optimization")
    
    def initialize_model(self, use_tensorrt: bool = True):
        """
        Load and configure OCR models.
        
        Args:
            use_tensorrt: Whether to enable TensorRT optimization
        """
        if not PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle not available - skipping model initialization")
            return
            
        try:
            # Model configuration for PaddleOCR
            ocr_config = {
                'use_angle_cls': self.config.use_angle_cls,
                'lang': self.config.lang,
                'use_gpu': True,  # Enable GPU for RTX 4090
                'gpu_id': 0,  # Default to GPU 0
                'gpu_mem': 16000,  # Use 16GB of 24GB available on RTX 4090
                'use_tensorrt': use_tensorrt and self.config.use_tensorrt,  # Enable TensorRT if configured
                'precision': self.config.tensorrt_precision.lower(),
                'max_batch_size': self.config.max_batch_size,
                'min_subgraph_size': 10,
                'tensorrt_workspace_size': 1 << 30,  # 1GB workspace
                'tensorrt_use_static_engine': True,  # Use static engines for caching
                'use_space_char': self.config.use_space_char,
                'drop_score': 0.5,
                'det_db_thresh': 0.3,  # Optimized for accuracy
                'det_db_box_thresh': 0.6,
                'det_db_unclip_ratio': 1.5,
                'det_east_score_thresh': 0.8,
                'det_east_cover_thresh': 0.1,
                'det_east_nms_thresh': 0.2,
                'rec_batch_num': 6,  # Optimized for RTX 4090
                'max_text_length': 25,
                'rec_image_shape': "3, 48, 320",
                'cls_image_shape': "3, 48, 192",
                'enable_mkldnn': False,
                'cpu_threads': 10,
                'show_log': logger.level <= logging.DEBUG
            }
            
            # Add custom model paths if provided
            if self.config.paddle_det_model_dir:
                ocr_config['det_model_dir'] = self.config.paddle_det_model_dir
            if self.config.paddle_rec_model_dir:
                ocr_config['rec_model_dir'] = self.config.paddle_rec_model_dir
            if self.config.paddle_cls_model_dir:
                ocr_config['cls_model_dir'] = self.config.paddle_cls_model_dir
            
            # Initialize PaddleOCR
            logger.info(f"Initializing PaddleOCR with TensorRT: {use_tensorrt}")
            logger.info("Starting PaddleOCR initialization...")
            
            # Check if engine cache directory exists
            engine_cache_dir = "/root/.paddleocr/tensorrt_engines"
            if os.path.exists(engine_cache_dir):
                engine_files = list(Path(engine_cache_dir).glob("*.engine"))
                if engine_files:
                    logger.info(f"Found {len(engine_files)} cached TensorRT engines")
                else:
                    logger.info("TensorRT engine cache directory exists but is empty")
            
            init_start_time = time.time()
            self.ocr = PaddleOCR(**ocr_config)
            init_time = time.time() - init_start_time
            logger.info(f"PaddleOCR initialization completed in {init_time:.2f} seconds")
            
            # Warm up the model
            self._warmup_model()
            
            logger.info("PaddleOCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _warmup_model(self):
        """Warm up the model with realistic data for optimal TensorRT shape collection"""
        if not self.ocr:
            return
            
        try:
            # Check if we're using TensorRT and need proper warmup
            if self.config.use_tensorrt:
                # Check if TensorRT shape files already exist
                shape_files_exist = self._check_tensorrt_shape_files()
                if shape_files_exist:
                    logger.info("TensorRT shape files already exist, skipping intensive warmup")
                    # Do a minimal warmup just to load models
                    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    _ = self.ocr.ocr(dummy_img, cls=self.config.use_angle_cls)
                    return
                
                # Try to use a real PDF for warmup if available
                warmup_pdf_path = "/app/tests/startup/DOC734S3110.pdf"
                if os.path.exists(warmup_pdf_path):
                    logger.info("Using real PDF for TensorRT warmup to collect accurate shapes")
                    logger.info("This is a one-time process that may take 2-3 minutes...")
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(warmup_pdf_path)
                        
                        # Process only first page with fewer DPI settings for faster warmup
                        warmup_count = 0
                        for page_num in range(min(1, len(doc))):  # Only first page
                            page = doc[page_num]
                            # Reduced DPI settings for faster warmup
                            for dpi in [120, 150]:  # Removed 200 DPI
                                start_time = time.time()
                                mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                                pix = page.get_pixmap(matrix=mat)
                                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                
                                # Convert RGBA to RGB if needed
                                if img.shape[2] == 4:
                                    img = img[:, :, :3]
                                
                                # Add slight noise to avoid exact cache hits
                                noise = np.random.randint(0, 5, img.shape, dtype=np.uint8)
                                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                                
                                logger.info(f"Warmup {warmup_count+1}: Processing image {img.shape} at {dpi} DPI")
                                result = self.ocr.ocr(img, cls=self.config.use_angle_cls)
                                
                                elapsed = time.time() - start_time
                                detections = len(result[0]) if result and result[0] else 0
                                logger.info(f"Warmup {warmup_count+1} completed in {elapsed:.2f}s with {detections} detections")
                                warmup_count += 1
                        
                        doc.close()
                        logger.info("TensorRT warmup with real document completed")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to use real PDF for warmup: {e}, falling back to synthetic images")
            
            # Fallback to synthetic images with realistic sizes
            sizes = [(1280, 720), (1920, 1080), (960, 1280)]  # Common document sizes
            logger.info(f"Running {self.config.warmup_iterations} warmup iterations with synthetic images")
            
            for i in range(self.config.warmup_iterations):
                size = sizes[i % len(sizes)]
                # Create image with text-like patterns
                dummy_img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
                # Add some black rectangles to simulate text
                for j in range(10):
                    y = int(j * size[1] / 10)
                    cv2.rectangle(dummy_img, (50, y), (size[0]-50, y+20), (0, 0, 0), -1)
                
                _ = self.ocr.ocr(dummy_img, cls=self.config.use_angle_cls)
                
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def process_batch(
        self,
        images: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[OCROutput]:
        """
        Process image batch through OCR pipeline.
        
        Args:
            images: List of images as numpy arrays
            batch_size: Optional batch size override
            
        Returns:
            List of OCR outputs for each image
        """
        if not PADDLE_AVAILABLE:
            logger.warning("PaddlePaddle not available - returning empty results")
            return [OCROutput(boxes=[], texts=[], confidences=[]) for _ in images]
            
        if not self.ocr:
            raise RuntimeError("Model not initialized. Call initialize_model() first")
        
        if not images:
            return []
        
        # Use configured batch size if not specified
        if batch_size is None:
            batch_size = self._determine_batch_size(len(images))
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _determine_batch_size(self, num_images: int) -> int:
        """Determine optimal batch size based on GPU memory"""
        try:
            # Get available GPU memory
            memory_info = self.gpu_monitor.get_available_memory()
            available_mb = memory_info.free_mb
            
            # Estimate memory per image (conservative estimate)
            memory_per_image_mb = 150  # Approximate for OCR processing
            
            # Calculate max batch size based on memory
            max_batch_from_memory = int(available_mb * 0.8 / memory_per_image_mb)
            
            # Apply limits
            batch_size = min(
                max_batch_from_memory,
                self.config.max_batch_size,
                num_images
            )
            
            # Ensure at least batch size of 1
            batch_size = max(1, batch_size)
            
            logger.debug(f"Determined batch size: {batch_size} (available memory: {available_mb}MB)")
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to determine dynamic batch size: {e}")
            return min(self.config.max_batch_size, num_images)
    
    def _process_single_batch(self, images: List[np.ndarray]) -> List[OCROutput]:
        """Process a single batch of images"""
        start_time = time.time()
        results = []
        
        try:
            # Process each image in the batch
            for img in images:
                try:
                    # Run OCR
                    ocr_result = self.ocr.ocr(img, cls=self.config.use_angle_cls)
                    
                    # Parse results
                    ocr_output = self._parse_ocr_result(ocr_result)
                    results.append(ocr_output)
                    
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    # Return empty result for failed image
                    results.append(OCROutput(boxes=[], texts=[], confidences=[]))
            
            # Update performance metrics
            batch_time_ms = (time.time() - start_time) * 1000
            self.total_inferences += len(images)
            self.total_latency_ms += batch_time_ms
            
            logger.info(f"Processed batch of {len(images)} images in {batch_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return empty results for entire batch
            results = [OCROutput(boxes=[], texts=[], confidences=[]) for _ in images]
        
        return results
    
    def _parse_ocr_result(self, ocr_result: List) -> OCROutput:
        """Parse PaddleOCR result into OCROutput format"""
        boxes = []
        texts = []
        confidences = []
        
        if not ocr_result or not ocr_result[0]:
            return OCROutput(boxes=[], texts=[], confidences=[])
        
        # PaddleOCR returns results for each page (we process single images)
        page_result = ocr_result[0]
        
        if page_result is None:
            return OCROutput(boxes=[], texts=[], confidences=[])
        
        for line in page_result:
            if line is None:
                continue
                
            # Extract box coordinates
            box_points = line[0]
            text_info = line[1]
            
            # Create TextBox
            box = TextBox(
                coordinates=[(int(pt[0]), int(pt[1])) for pt in box_points],
                confidence=float(text_info[1])
            )
            boxes.append(box)
            
            # Extract text and confidence
            texts.append(text_info[0])
            confidences.append(float(text_info[1]))
        
        return OCROutput(boxes=boxes, texts=texts, confidences=confidences)
    
    def cleanup(self):
        """Release model resources and clear GPU memory"""
        try:
            if self.ocr:
                # PaddleOCR doesn't have explicit cleanup, but we can clear paddle cache
                if paddle:
                    paddle.device.cuda.empty_cache()
                
                self.ocr = None
                logger.info("PaddleOCR resources cleaned up")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_model_info(self) -> ModelInfo:
        """Return model metadata and status"""
        avg_latency = 0.0
        if self.total_inferences > 0:
            avg_latency = self.total_latency_ms / self.total_inferences
        
        return ModelInfo(
            model_version=self.model_version,
            optimization_enabled=self.config.use_tensorrt,
            current_batch_size=self.config.max_batch_size,
            total_inferences=self.total_inferences,
            average_latency_ms=avg_latency
        )
    
    def _load_detection_model(self):
        """Load text detection model (DB++)"""
        # This is handled internally by PaddleOCR
        # Keeping method for API compatibility
        pass
    
    def _load_recognition_model(self):
        """Load text recognition model (CRNN)"""
        # This is handled internally by PaddleOCR
        # Keeping method for API compatibility
        pass
    
    def _configure_tensorrt(self):
        """Configure TensorRT optimization settings"""
        # TensorRT configuration is handled in initialize_model
        # Keeping method for API compatibility
        pass
    
    def _preprocess_batch(self, images: List[np.ndarray]) -> Any:
        """Prepare images for model input"""
        # PaddleOCR handles preprocessing internally
        # Keeping method for API compatibility
        return images
    
    def _detect_text_regions(self, batch_tensor: Any) -> List[List[TextBox]]:
        """Run text detection model"""
        # Detection is part of the unified OCR pipeline in PaddleOCR
        # Keeping method for API compatibility
        return []
    
    def _recognize_text(self, image_crops: List[np.ndarray]) -> List[TextRecognition]:
        """Run text recognition on detected regions"""
        # Recognition is part of the unified OCR pipeline in PaddleOCR
        # Keeping method for API compatibility
        return []
    
    def _postprocess_results(
        self,
        detections: List[List[TextBox]],
        recognitions: List[TextRecognition]
    ) -> List[OCROutput]:
        """Combine detection and recognition results"""
        # Post-processing is handled in _parse_ocr_result
        # Keeping method for API compatibility
        return []
    
    def _verify_model_integrity(self, model_path: str) -> bool:
        """Verify model files are valid"""
        # Basic file existence check
        if not os.path.exists(model_path):
            return False
        
        # Check for required model files
        required_files = ['inference.pdmodel', 'inference.pdiparams']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        
        return True
    
    def _check_tensorrt_shape_files(self) -> bool:
        """Check if TensorRT shape files already exist"""
        shape_files = [
            "/root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/det_trt_dynamic_shape.txt",
            "/root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/rec_trt_dynamic_shape.txt",
            "/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/cls_trt_dynamic_shape.txt"
        ]
        
        # Check if all files exist and have content
        for shape_file in shape_files:
            if not os.path.exists(shape_file):
                return False
            try:
                # Check if file has content (not empty)
                if os.path.getsize(shape_file) == 0:
                    return False
            except:
                return False
        
        return True
    
    def is_ready(self) -> bool:
        """Check if OCR model is ready for inference"""
        if not PADDLE_AVAILABLE:
            return False
        return self.ocr is not None