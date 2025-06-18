import io
import logging
import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math
from skimage import filters, morphology, transform
from skimage.metrics import structural_similarity as ssim
import warnings

from src.config import OCRConfig, load_config

# Suppress PIL decompression bomb warnings for large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class EnhancementLevel(str, Enum):
    """Enhancement level options"""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    AUTO = "auto"


@dataclass
class QualityReport:
    """Image quality analysis report"""
    blur_score: float  # 0-1, higher is blurrier
    noise_level: float  # 0-1, higher is noisier
    contrast_score: float  # 0-1, higher is better
    skew_angle: float  # degrees
    recommended_enhancements: List[str]


@dataclass
class ValidationResult:
    """Image validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    properties: Dict[str, Any]


class ImageProcessor:
    """Process and enhance images for OCR"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """Purpose: Initialize image processing utilities
        Dependencies: OpenCV, PIL, scikit-image
        Priority: CORE
        """
        self.config = config or load_config().ocr
        self.logger = logging.getLogger(__name__)
        
        # Image size constraints
        self.max_dimension = self.config.max_image_size
        self.supported_formats = set(self.config.supported_image_formats)
        
        # Quality thresholds
        self.blur_threshold = 0.3
        self.noise_threshold = 0.2
        self.contrast_threshold = 0.4
        self.skew_threshold = 2.0  # degrees
        
        # Enhancement parameters
        self.enhancement_params = {
            EnhancementLevel.LIGHT: {
                'denoise_strength': 3,
                'contrast_clip_limit': 2.0,
                'contrast_grid_size': (4, 4)
            },
            EnhancementLevel.MODERATE: {
                'denoise_strength': 7,
                'contrast_clip_limit': 3.0,
                'contrast_grid_size': (8, 8)
            },
            EnhancementLevel.AGGRESSIVE: {
                'denoise_strength': 11,
                'contrast_clip_limit': 4.0,
                'contrast_grid_size': (16, 16)
            }
        }
    
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
        try:
            # Load image from bytes
            pil_image = self._load_image(image_bytes)
            
            # Normalize DPI
            pil_image = self._normalize_dpi(pil_image, target_dpi)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Convert color space to grayscale
            image_array = self._convert_color_space(image_array)
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error preparing image: {e}")
            raise
    
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
        # Make a copy to avoid modifying original
        enhanced = image.copy()
        
        # Convert string to enum
        if isinstance(enhancement_level, str):
            enhancement_level = EnhancementLevel(enhancement_level.lower())
        
        # Detect quality issues if auto mode
        if enhancement_level == EnhancementLevel.AUTO:
            quality_report = self._detect_quality_issues(enhanced)
            
            # Apply recommended enhancements
            if quality_report.skew_angle > self.skew_threshold:
                enhanced = self._correct_skew(enhanced)
            
            if quality_report.noise_level > self.noise_threshold:
                enhanced = self._apply_denoising(enhanced, quality_report.noise_level)
            
            if quality_report.contrast_score < self.contrast_threshold:
                enhanced = self._enhance_contrast(enhanced)
            
            if quality_report.blur_score > self.blur_threshold:
                # Apply sharpening for blurry images
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        elif enhancement_level != EnhancementLevel.NONE:
            # Apply fixed enhancement level
            params = self.enhancement_params[enhancement_level]
            
            # Denoise
            enhanced = self._apply_denoising(enhanced, 0.5)
            
            # Enhance contrast
            enhanced = self._enhance_contrast(enhanced)
            
            # Correct skew
            enhanced = self._correct_skew(enhanced)
        
        return enhanced
    
    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Purpose: Load image from bytes
        Calls:
            - PIL.Image.open()
            - _validate_image()
        Called by: prepare_image()
        Priority: CORE
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L', 'RGBA'):
                image = image.convert('RGB')
            
            # Validate image
            validation = self._validate_image(image)
            if not validation.is_valid:
                raise ValueError(f"Invalid image: {', '.join(validation.errors)}")
            
            # Log warnings if any
            for warning in validation.warnings:
                self.logger.warning(warning)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            raise
    
    def _validate_image(self, image: Image.Image) -> ValidationResult:
        """Validate image properties"""
        errors = []
        warnings = []
        properties = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height
        }
        
        # Check format
        if image.format and image.format.lower() not in self.supported_formats:
            errors.append(f"Unsupported format: {image.format}")
        
        # Check dimensions
        if image.width > self.max_dimension or image.height > self.max_dimension:
            errors.append(f"Image too large: {image.width}x{image.height} (max: {self.max_dimension})")
        
        # Check if image is too small
        if image.width < 50 or image.height < 50:
            warnings.append(f"Image very small: {image.width}x{image.height}")
        
        # Check aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 20 or aspect_ratio < 0.05:
            warnings.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            properties=properties
        )
    
    def _normalize_dpi(
        self,
        image: Image.Image,
        target_dpi: int
    ) -> Image.Image:
        """Purpose: Adjust image DPI
        Calls:
            - Calculate scaling factor
            - Resize if needed
        Called by: prepare_image()
        Priority: CORE
        """
        # Get current DPI
        current_dpi = image.info.get('dpi', (72, 72))
        if isinstance(current_dpi, (int, float)):
            current_dpi = (current_dpi, current_dpi)
        
        # Use first value if DPI is different for x and y
        current_dpi_value = current_dpi[0] if current_dpi[0] else 72
        
        # Calculate scaling factor
        scale_factor = target_dpi / current_dpi_value
        
        # Only resize if scale factor is significantly different from 1
        if abs(scale_factor - 1.0) > 0.1:
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            # Use high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.logger.debug(f"Resized image from {image.width}x{image.height} "
                            f"(DPI: {current_dpi_value}) to {new_width}x{new_height} "
                            f"(DPI: {target_dpi})")
        
        # Set DPI info
        image.info['dpi'] = (target_dpi, target_dpi)
        
        return image
    
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
        # If already grayscale, return as is
        if len(image.shape) == 2:
            return image
        
        # Convert to grayscale using luminosity method
        # This gives better results for text than simple averaging
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                # Remove alpha channel first
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray
        
        return image
    
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
        recommended_enhancements = []
        
        # Detect blur using Laplacian variance
        blur_score = self._calculate_blur_score(image)
        if blur_score > self.blur_threshold:
            recommended_enhancements.append("sharpen")
        
        # Detect noise level
        noise_level = self._calculate_noise_level(image)
        if noise_level > self.noise_threshold:
            recommended_enhancements.append("denoise")
        
        # Check contrast
        contrast_score = self._calculate_contrast_score(image)
        if contrast_score < self.contrast_threshold:
            recommended_enhancements.append("enhance_contrast")
        
        # Detect skew angle
        skew_angle = self._detect_skew_angle(image)
        if abs(skew_angle) > self.skew_threshold:
            recommended_enhancements.append("deskew")
        
        return QualityReport(
            blur_score=blur_score,
            noise_level=noise_level,
            contrast_score=contrast_score,
            skew_angle=skew_angle,
            recommended_enhancements=recommended_enhancements
        )
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score (0-1, higher is blurrier)"""
        # Use Laplacian variance method
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (inverse relationship)
        # Lower variance means more blur
        blur_score = 1.0 / (1.0 + variance / 100.0)
        
        return blur_score
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate noise level (0-1, higher is noisier)"""
        # Estimate noise using difference between image and denoised version
        denoised = cv2.fastNlMeansDenoising(image, h=3)
        noise = np.abs(image.astype(float) - denoised.astype(float))
        
        # Calculate noise level as ratio of noise to signal
        noise_level = np.mean(noise) / (np.mean(image) + 1e-6)
        
        # Clip to 0-1 range
        return np.clip(noise_level, 0, 1)
    
    def _calculate_contrast_score(self, image: np.ndarray) -> float:
        """Calculate contrast score (0-1, higher is better)"""
        # Calculate RMS contrast
        img_float = image.astype(float) / 255.0
        contrast = img_float.std()
        
        # Normalize to 0-1 range
        return np.clip(contrast * 4, 0, 1)  # Scale factor for typical text images
    
    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect text skew angle in degrees"""
        # Use Hough transform to detect lines
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return 0.0
        
        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 <= angle <= 45:  # Only consider reasonable angles
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Return median angle
        return float(np.median(angles))
    
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
        # Adjust denoising strength based on noise level
        h = int(3 + noise_level * 20)  # 3-23 range
        h = min(h, 21)  # Cap at reasonable level
        
        # Apply fast non-local means denoising
        denoised = cv2.fastNlMeansDenoising(image, h=h)
        
        # For high noise, apply additional morphological filtering
        if noise_level > 0.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        return denoised
    
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
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # For very low contrast images, apply additional global equalization
        if self._calculate_contrast_score(image) < 0.2:
            enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
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
        # Detect skew angle
        angle = self._detect_skew_angle(image)
        
        # Only correct if angle is significant
        if abs(angle) < 0.5:
            return image
        
        # Get image center
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new bounds
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation with white background
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)
        
        self.logger.debug(f"Corrected skew by {angle:.1f} degrees")
        
        return rotated
    
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
        resized_images = []
        target_width, target_height = target_size
        
        for image in images:
            h, w = image.shape[:2]
            
            # Calculate scaling factor to maintain aspect ratio
            scale = min(target_width / w, target_height / h)
            
            # Don't upscale images
            if scale > 1.0:
                scale = 1.0
            
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            
            # Pad to target size if needed
            if new_width < target_width or new_height < target_height:
                # Create padded image with white background
                padded = np.full((target_height, target_width), 255, dtype=np.uint8)
                
                # Center the resized image
                y_offset = (target_height - new_height) // 2
                x_offset = (target_width - new_width) // 2
                
                padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                resized = padded
            
            resized_images.append(resized)
        
        return resized_images
    
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
        # Try to detect text orientation using multiple methods
        
        # Method 1: Use Tesseract OSD if available
        # This would require pytesseract, skipping for now
        
        # Method 2: Use projection profiles
        angles_to_try = [0, 90, 180, 270]
        best_score = -1
        best_angle = 0
        
        for angle in angles_to_try:
            # Rotate image
            if angle == 0:
                rotated = image
            else:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                if angle == 90 or angle == 270:
                    rotated = cv2.warpAffine(image, M, (h, w))
                else:
                    rotated = cv2.warpAffine(image, M, (w, h))
            
            # Calculate horizontal projection variance
            # Text should have high variance in horizontal projection
            projection = np.sum(rotated, axis=1)
            score = np.var(projection)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        return best_angle


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create processor
    processor = ImageProcessor()
    
    # Test with a sample image
    print("Image processor initialized")
    print(f"Supported formats: {processor.supported_formats}")
    print(f"Max dimension: {processor.max_dimension}")
    
    # Create a test image
    test_image = np.ones((100, 200), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test OCR", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    
    # Test quality detection
    quality = processor._detect_quality_issues(test_image)
    print(f"\nQuality report:")
    print(f"  Blur score: {quality.blur_score:.3f}")
    print(f"  Noise level: {quality.noise_level:.3f}")
    print(f"  Contrast score: {quality.contrast_score:.3f}")
    print(f"  Skew angle: {quality.skew_angle:.1f}°")
    print(f"  Recommendations: {quality.recommended_enhancements}")
    
    # Test enhancement
    enhanced = processor.enhance_for_ocr(test_image, "auto")
    print(f"\nEnhancement applied. Shape: {enhanced.shape}")