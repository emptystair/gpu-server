# CLAUDE.MD - /src/utils/

## Component Purpose
Provides utility functions for document processing, image manipulation, and caching. These components handle the preprocessing of various input formats and optimization of resource usage through intelligent caching strategies.

## Classes and Functions to Implement

### File: pdf_processor.py

```python
class PDFProcessor:
    def __init__(self, config: PDFConfig):
        """Purpose: Initialize PDF processing with configuration
        Dependencies: PyMuPDF (fitz), pdf2image, PIL
        Priority: CORE
        """
    
    def extract_pages(
        self,
        pdf_bytes: bytes,
        dpi: int = 120,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """Purpose: Convert PDF pages to numpy arrays
        Calls:
            - _load_pdf()
            - _render_page()
            - _convert_to_array()
        Called by: ocr_service._prepare_document()
        Priority: CORE
        """
    
    def _load_pdf(self, pdf_bytes: bytes) -> fitz.Document:
        """Purpose: Load PDF from bytes
        Calls:
            - fitz.open() with stream
            - _validate_pdf()
        Called by: extract_pages()
        Priority: CORE
        """
    
    def _render_page(
        self,
        page: fitz.Page,
        dpi: int
    ) -> PIL.Image:
        """Purpose: Render PDF page to image
        Calls:
            - page.get_pixmap()
            - Convert to PIL Image
        Called by: extract_pages()
        Priority: CORE
        """
    
    def _convert_to_array(self, image: PIL.Image) -> np.ndarray:
        """Purpose: Convert PIL image to numpy array
        Calls:
            - np.array()
            - Ensure correct dtype/shape
        Called by: extract_pages()
        Priority: CORE
        """
    
    def _validate_pdf(self, doc: fitz.Document) -> ValidationResult:
        """Purpose: Check PDF validity and properties
        Calls:
            - Check page count
            - Check encryption
            - Detect scanned vs text
        Called by: _load_pdf()
        Priority: CORE
        """
    
    def get_pdf_metadata(self, pdf_bytes: bytes) -> PDFMetadata:
        """Purpose: Extract PDF metadata
        Calls:
            - Load PDF
            - Extract properties
        Called by: API for info endpoints
        Priority: MONITORING
        """
    
    def extract_images_from_pdf(
        self,
        pdf_bytes: bytes
    ) -> List[EmbeddedImage]:
        """Purpose: Extract embedded images from PDF
        Calls:
            - Iterate through page images
            - Extract image data
        Called by: Advanced processing
        Priority: OPTIMIZATION
        """
    
    def optimize_for_ocr(
        self,
        page_image: np.ndarray
    ) -> np.ndarray:
        """Purpose: Preprocess image for better OCR
        Calls:
            - image_processor.enhance_for_ocr()
        Called by: extract_pages()
        Priority: OPTIMIZATION
        """
    
    def split_pdf(
        self,
        pdf_bytes: bytes,
        chunks: int
    ) -> List[bytes]:
        """Purpose: Split PDF into smaller chunks
        Calls:
            - Calculate split points
            - Create sub-PDFs
        Called by: Batch processing
        Priority: OPTIMIZATION
        """
```

### File: image_processor.py

```python
class ImageProcessor:
    def __init__(self, config: ImageConfig):
        """Purpose: Initialize image processing utilities
        Dependencies: OpenCV, PIL, scikit-image
        Priority: CORE
        """
    
    def prepare_image(
        self,
        image_bytes: bytes,
        target_dpi: int = 120
    ) -> np.ndarray:
        """Purpose: Prepare single image for OCR
        Calls:
            - _load_image()
            - _normalize_dpi()
            - _convert_color_space()
        Called by: ocr_service._prepare_document()
        Priority: CORE
        """
    
    def enhance_for_ocr(
        self,
        image: np.ndarray,
        enhancement_level: str = "auto"
    ) -> np.ndarray:
        """Purpose: Apply OCR-specific enhancements
        Calls:
            - _detect_quality_issues()
            - _apply_denoising()
            - _enhance_contrast()
            - _correct_skew()
        Called by: pdf_processor.optimize_for_ocr()
        Priority: OPTIMIZATION
        """
    
    def _load_image(self, image_bytes: bytes) -> PIL.Image:
        """Purpose: Load image from bytes
        Calls:
            - PIL.Image.open()
            - _validate_image()
        Called by: prepare_image()
        Priority: CORE
        """
    
    def _normalize_dpi(
        self,
        image: PIL.Image,
        target_dpi: int
    ) -> PIL.Image:
        """Purpose: Adjust image DPI
        Calls:
            - Calculate scaling factor
            - Resize if needed
        Called by: prepare_image()
        Priority: CORE
        """
    
    def _convert_color_space(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Purpose: Convert to grayscale for OCR
        Calls:
            - cv2.cvtColor()
            - Preserve color if needed
        Called by: prepare_image()
        Priority: CORE
        """
    
    def _detect_quality_issues(
        self,
        image: np.ndarray
    ) -> QualityReport:
        """Purpose: Analyze image quality
        Calls:
            - Check blur level
            - Detect noise
            - Check contrast
        Called by: enhance_for_ocr()
        Priority: OPTIMIZATION
        """
    
    def _apply_denoising(
        self,
        image: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """Purpose: Remove noise from image
        Calls:
            - cv2.fastNlMeansDenoising()
            - Adaptive filtering
        Called by: enhance_for_ocr()
        Priority: OPTIMIZATION
        """
    
    def _enhance_contrast(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Purpose: Improve image contrast
        Calls:
            - CLAHE algorithm
            - Histogram equalization
        Called by: enhance_for_ocr()
        Priority: OPTIMIZATION
        """
    
    def _correct_skew(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Purpose: Detect and correct image skew
        Calls:
            - Hough line detection
            - Rotation correction
        Called by: enhance_for_ocr()
        Priority: OPTIMIZATION
        """
    
    def batch_resize(
        self,
        images: List[np.ndarray],
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Purpose: Resize batch for uniform processing
        Calls:
            - cv2.resize() per image
            - Maintain aspect ratio
        Called by: Batch processing
        Priority: OPTIMIZATION
        """
    
    def detect_orientation(
        self,
        image: np.ndarray
    ) -> int:
        """Purpose: Detect image rotation
        Calls:
            - Text orientation detection
            - Return rotation angle
        Called by: Pre-processing
        Priority: OPTIMIZATION
        """
```

### File: cache_manager.py

```python
class CacheManager:
    def __init__(self, config: CacheConfig):
        """Purpose: Initialize caching system
        Dependencies: diskcache, redis (optional)
        Priority: OPTIMIZATION
        """
    
    def initialize_cache(self):
        """Purpose: Set up cache directories and connections
        Calls:
            - Create cache directories
            - Connect to Redis if configured
            - Load persistent cache
        Called by: main.startup_event()
        Priority: OPTIMIZATION
        """
    
    async def get_cached_result(
        self,
        document_hash: str
    ) -> Optional[OCRResult]:
        """Purpose: Retrieve cached OCR result
        Calls:
            - Check memory cache
            - Check disk cache
            - Check Redis cache
        Called by: ocr_service.process_document()
        Priority: OPTIMIZATION
        """
    
    async def cache_result(
        self,
        document_hash: str,
        result: OCRResult,
        ttl_seconds: int = 3600
    ):
        """Purpose: Store OCR result in cache
        Calls:
            - Store in memory cache
            - Store in disk cache
            - Store in Redis if available
        Called by: ocr_service.process_document()
        Priority: OPTIMIZATION
        """
    
    def get_model_cache_path(
        self,
        model_name: str,
        version: str
    ) -> Path:
        """Purpose: Get path for cached model files
        Calls:
            - Generate cache key
            - Return path
        Called by: Model loading functions
        Priority: OPTIMIZATION
        """
    
    def cache_tensorrt_engine(
        self,
        engine_bytes: bytes,
        config_hash: str
    ) -> Path:
        """Purpose: Cache compiled TensorRT engine
        Calls:
            - Save engine to disk
            - Update cache index
        Called by: tensorrt_optimizer.optimize_model()
        Priority: OPTIMIZATION
        """
    
    def get_tensorrt_engine(
        self,
        config_hash: str
    ) -> Optional[bytes]:
        """Purpose: Retrieve cached TensorRT engine
        Calls:
            - Check cache existence
            - Load from disk
        Called by: tensorrt_optimizer.optimize_model()
        Priority: OPTIMIZATION
        """
    
    def _generate_document_hash(
        self,
        file_bytes: bytes,
        processing_params: Dict
    ) -> str:
        """Purpose: Generate unique hash for document
        Calls:
            - hashlib functions
            - Include parameters
        Called by: get_cached_result(), cache_result()
        Priority: OPTIMIZATION
        """
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Purpose: Clear cache contents
        Calls:
            - Clear memory cache
            - Clear disk cache
            - Clear Redis cache
        Called by: main.shutdown_event(), maintenance
        Priority: OPTIMIZATION
        """
    
    def get_cache_stats(self) -> CacheStats:
        """Purpose: Return cache performance statistics
        Calls:
            - Calculate hit rates
            - Get size info
        Called by: Monitoring endpoints
        Priority: MONITORING
        """
    
    def evict_old_entries(self, max_age_seconds: int):
        """Purpose: Remove old cache entries
        Calls:
            - Scan cache entries
            - Delete expired items
        Called by: Background maintenance
        Priority: OPTIMIZATION
        """
```

## Data Structures

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
from enum import Enum

@dataclass
class PDFConfig:
    max_pages: int = 1000
    default_dpi: int = 120
    render_timeout_seconds: int = 30
    max_file_size_mb: int = 100

@dataclass
class ImageConfig:
    supported_formats: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "tiff", "bmp"])
    max_dimension: int = 4096
    enhancement_enabled: bool = True
    
@dataclass
class CacheConfig:
    cache_dir: Path = Path("./cache")
    memory_cache_size_mb: int = 512
    disk_cache_size_gb: int = 10
    redis_url: Optional[str] = None
    default_ttl_seconds: int = 3600

@dataclass
class PDFMetadata:
    page_count: int
    title: Optional[str]
    author: Optional[str]
    creation_date: Optional[datetime]
    is_encrypted: bool
    is_scanned: bool
    file_size_bytes: int

@dataclass
class EmbeddedImage:
    page_number: int
    image_index: int
    image_data: bytes
    width: int
    height: int
    format: str

@dataclass
class QualityReport:
    blur_score: float  # 0-1, higher is blurrier
    noise_level: float  # 0-1, higher is noisier
    contrast_score: float  # 0-1, higher is better
    skew_angle: float  # degrees
    recommended_enhancements: List[str]

@dataclass
class CacheStats:
    memory_hits: int
    memory_misses: int
    disk_hits: int
    disk_misses: int
    total_size_mb: float
    entry_count: int
    hit_rate: float
    
class EnhancementLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    AUTO = "auto"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    properties: Dict[str, Any]
```

## Cross-Component Dependencies

**Imports from:**
- `../config` - Configuration classes
- `../models/result_formatter` - For cache serialization

**External libraries:**
- PyMuPDF==1.23.8 (fitz)
- pdf2image==1.16.3
- Pillow==10.1.0
- opencv-python==4.8.1.78
- scikit-image==0.22.0
- numpy==1.24.3
- diskcache==5.6.3
- redis==5.0.1 (optional)

**Calls into:**
- File system for cache storage
- Image processing libraries
- PDF rendering engines

**Called by:**
- `ocr_service.py` - Document preparation
- Model components - For preprocessing
- API layer - For direct utilities

## Implementation Notes

1. **PDF Processing:**
   - Use PyMuPDF for speed
   - Render at exact DPI needed
   - Handle encrypted PDFs gracefully
   - Detect scanned vs text PDFs

2. **Image Enhancement:**
   - Auto-detect quality issues
   - Apply minimal processing by default
   - Preserve original for fallback
   - Use GPU acceleration where possible

3. **Caching Strategy:**
   - Three-tier cache (memory, disk, Redis)
   - Hash includes processing parameters
   - LRU eviction for memory cache
   - Background cleanup for disk cache

4. **Memory Management:**
   - Stream large PDFs page by page
   - Release image memory after processing
   - Use memory mapping for large files
   - Monitor cache size limits

5. **Performance Tips:**
   - Pre-compile regex patterns
   - Reuse image buffers
   - Batch similar operations
   - Parallelize independent tasks

## Error Handling

- **PDF Errors**: Log corruption, attempt repair, skip bad pages
- **Image Errors**: Validate format, handle truncated files
- **Cache Errors**: Fall through to processing, log failures
- **Memory Errors**: Clear caches, reduce batch size
- **Enhancement Errors**: Return original image, log issue