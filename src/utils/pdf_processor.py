"""
PDF Processing Module

Handles PDF to image conversion for OCR processing, optimized for GPU acceleration.
Uses PyMuPDF for fast rendering and integrates with ImageProcessor for OCR optimization.
"""

import io
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from ..config import Config
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


@dataclass
class PDFConfig:
    """Configuration for PDF processing"""
    max_pages: int = 1000
    default_dpi: int = 120
    render_timeout_seconds: int = 30
    max_file_size_mb: int = 100


@dataclass
class PDFMetadata:
    """PDF document metadata"""
    page_count: int
    title: Optional[str]
    author: Optional[str]
    creation_date: Optional[datetime]
    is_encrypted: bool
    is_scanned: bool
    file_size_bytes: int


@dataclass
class EmbeddedImage:
    """Embedded image information"""
    page_number: int
    image_index: int
    image_data: bytes
    width: int
    height: int
    format: str


@dataclass
class ValidationResult:
    """PDF validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    properties: Dict[str, Any]


class PDFProcessor:
    """
    Handles PDF processing for OCR, including rendering pages to images
    and extracting embedded content.
    """
    
    def __init__(self, config: Optional[PDFConfig] = None):
        """
        Purpose: Initialize PDF processing with configuration
        Dependencies: PyMuPDF (fitz), PIL
        Priority: CORE
        """
        self.config = config or PDFConfig()
        self.image_processor = ImageProcessor()
        logger.info(f"PDFProcessor initialized with DPI: {self.config.default_dpi}")
    
    def extract_pages(
        self,
        pdf_bytes: bytes,
        dpi: int = 120,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Purpose: Convert PDF pages to numpy arrays
        Calls:
            - _load_pdf()
            - _render_page()
            - _convert_to_array()
        Called by: ocr_service._prepare_document()
        Priority: CORE
        """
        # Load and validate PDF
        doc = self._load_pdf(pdf_bytes)
        
        # Determine page range
        start_page = 0
        end_page = len(doc)
        
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(len(doc), page_range[1])
        
        # Check page limit
        num_pages = end_page - start_page
        if num_pages > self.config.max_pages:
            logger.warning(f"Page count {num_pages} exceeds limit {self.config.max_pages}")
            end_page = start_page + self.config.max_pages
        
        # Extract pages
        page_images = []
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                
                # Render page to image
                pil_image = self._render_page(page, dpi or self.config.default_dpi)
                
                # Convert to numpy array
                np_array = self._convert_to_array(pil_image)
                
                # Optimize for OCR
                optimized = self.optimize_for_ocr(np_array)
                
                page_images.append(optimized)
                
                logger.debug(f"Processed page {page_num + 1}/{len(doc)}")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                # Continue with other pages
        
        doc.close()
        return page_images
    
    def _load_pdf(self, pdf_bytes: bytes) -> fitz.Document:
        """
        Purpose: Load PDF from bytes
        Calls:
            - fitz.open() with stream
            - _validate_pdf()
        Called by: extract_pages()
        Priority: CORE
        """
        # Check file size
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(f"PDF size {file_size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB")
        
        # Load PDF from bytes
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise ValueError(f"Invalid PDF file: {e}")
        
        # Validate PDF
        validation = self._validate_pdf(doc)
        if not validation.is_valid:
            doc.close()
            raise ValueError(f"PDF validation failed: {', '.join(validation.errors)}")
        
        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(f"PDF warning: {warning}")
        
        return doc
    
    def _render_page(self, page: fitz.Page, dpi: int) -> Image.Image:
        """
        Purpose: Render PDF page to image
        Calls:
            - page.get_pixmap()
            - Convert to PIL Image
        Called by: extract_pages()
        Priority: CORE
        """
        # Calculate zoom factor for desired DPI
        # PDF default is 72 DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def _convert_to_array(self, image: Image.Image) -> np.ndarray:
        """
        Purpose: Convert PIL image to numpy array
        Calls:
            - np.array()
            - Ensure correct dtype/shape
        Called by: extract_pages()
        Priority: CORE
        """
        # Convert to numpy array
        np_array = np.array(image)
        
        # Ensure correct dtype
        if np_array.dtype != np.uint8:
            np_array = np_array.astype(np.uint8)
        
        # Ensure we have the right shape
        if len(np_array.shape) == 2:
            # Already grayscale
            pass
        elif len(np_array.shape) == 3:
            # Color image - will be converted by image processor if needed
            pass
        else:
            raise ValueError(f"Unexpected image shape: {np_array.shape}")
        
        return np_array
    
    def _validate_pdf(self, doc: fitz.Document) -> ValidationResult:
        """
        Purpose: Check PDF validity and properties
        Calls:
            - Check page count
            - Check encryption
            - Detect scanned vs text
        Called by: _load_pdf()
        Priority: CORE
        """
        errors = []
        warnings = []
        properties = {}
        
        # Check if encrypted
        if doc.is_encrypted:
            if not doc.authenticate(""):
                errors.append("PDF is password protected")
            else:
                warnings.append("PDF was encrypted but opened with empty password")
        
        # Check page count
        page_count = len(doc)
        properties['page_count'] = page_count
        
        if page_count == 0:
            errors.append("PDF has no pages")
        elif page_count > self.config.max_pages:
            warnings.append(f"PDF has {page_count} pages, exceeds limit of {self.config.max_pages}")
        
        # Detect if scanned (no extractable text)
        is_scanned = True
        for page_num in range(min(5, page_count)):  # Check first 5 pages
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                is_scanned = False
                break
        
        properties['is_scanned'] = is_scanned
        
        # Check for forms
        if doc.is_form_pdf:
            warnings.append("PDF contains form fields")
            properties['has_forms'] = True
        
        # Check for digital signatures (if supported)
        if hasattr(doc, 'is_signed') and doc.is_signed:
            warnings.append("PDF is digitally signed")
            properties['is_signed'] = True
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            properties=properties
        )
    
    def get_pdf_metadata(self, pdf_bytes: bytes) -> PDFMetadata:
        """
        Purpose: Extract PDF metadata
        Calls:
            - Load PDF
            - Extract properties
        Called by: API for info endpoints
        Priority: MONITORING
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract metadata
        metadata = doc.metadata
        
        # Parse creation date
        creation_date = None
        if metadata.get('creationDate'):
            try:
                # PyMuPDF returns dates in PDF format
                date_str = metadata['creationDate']
                if date_str.startswith('D:'):
                    date_str = date_str[2:]
                # Basic parsing - might need enhancement for all PDF date formats
                creation_date = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
            except:
                pass
        
        # Detect if scanned
        is_scanned = True
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            if page.get_text().strip():
                is_scanned = False
                break
        
        result = PDFMetadata(
            page_count=len(doc),
            title=metadata.get('title'),
            author=metadata.get('author'),
            creation_date=creation_date,
            is_encrypted=doc.is_encrypted,
            is_scanned=is_scanned,
            file_size_bytes=len(pdf_bytes)
        )
        
        doc.close()
        return result
    
    def extract_images_from_pdf(self, pdf_bytes: bytes) -> List[EmbeddedImage]:
        """
        Purpose: Extract embedded images from PDF
        Calls:
            - Iterate through page images
            - Extract image data
        Called by: Advanced processing
        Priority: OPTIMIZATION
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        embedded_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to bytes
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        fmt = "PNG"
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                        fmt = "PNG"
                    
                    embedded_images.append(EmbeddedImage(
                        page_number=page_num + 1,
                        image_index=img_index,
                        image_data=img_data,
                        width=pix.width,
                        height=pix.height,
                        format=fmt
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        doc.close()
        return embedded_images
    
    def optimize_for_ocr(self, page_image: np.ndarray) -> np.ndarray:
        """
        Purpose: Preprocess image for better OCR
        Calls:
            - image_processor.enhance_for_ocr()
        Called by: extract_pages()
        Priority: OPTIMIZATION
        """
        # Use the image processor's enhancement
        return self.image_processor.enhance_for_ocr(page_image, enhancement_level="auto")
    
    def split_pdf(self, pdf_bytes: bytes, chunks: int) -> List[bytes]:
        """
        Purpose: Split PDF into smaller chunks
        Calls:
            - Calculate split points
            - Create sub-PDFs
        Called by: Batch processing
        Priority: OPTIMIZATION
        """
        if chunks <= 0:
            raise ValueError("Number of chunks must be positive")
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        if chunks >= total_pages:
            # Each page becomes its own chunk
            chunks = total_pages
        
        # Calculate pages per chunk
        pages_per_chunk = total_pages // chunks
        remainder = total_pages % chunks
        
        # Split PDF
        pdf_chunks = []
        start_page = 0
        
        for i in range(chunks):
            # Calculate end page for this chunk
            end_page = start_page + pages_per_chunk
            if i < remainder:
                end_page += 1
            
            # Create new PDF with selected pages
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
            
            # Save to bytes
            pdf_data = new_doc.tobytes()
            pdf_chunks.append(pdf_data)
            
            new_doc.close()
            start_page = end_page
        
        doc.close()
        return pdf_chunks