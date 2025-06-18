import numpy as np
import cv2
from PIL import Image
import io
import logging
from src.utils.image_processor import ImageProcessor, EnhancementLevel, QualityReport

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("=== Image Processor Tests ===\n")

# Initialize processor
processor = ImageProcessor()

# Test 1: Create a test image with various quality issues
print("Test 1: Creating test images with quality issues")

# Create a clean text image
clean_image = np.ones((200, 400), dtype=np.uint8) * 255
cv2.putText(clean_image, "Clean Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 2)

# Create a noisy image
noisy_image = clean_image.copy()
noise = np.random.normal(0, 25, noisy_image.shape).astype(np.uint8)
noisy_image = cv2.add(noisy_image, noise)

# Create a blurry image
blurry_image = cv2.GaussianBlur(clean_image, (15, 15), 0)

# Create a low contrast image
low_contrast = (clean_image.astype(float) * 0.3 + 127).astype(np.uint8)

# Create a skewed image
rows, cols = clean_image.shape
angle = 5  # 5 degree skew
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
skewed_image = cv2.warpAffine(clean_image, M, (cols, rows), borderValue=255)

# Test quality detection on each image
test_images = [
    ("Clean", clean_image),
    ("Noisy", noisy_image),
    ("Blurry", blurry_image),
    ("Low Contrast", low_contrast),
    ("Skewed", skewed_image)
]

for name, image in test_images:
    quality = processor._detect_quality_issues(image)
    print(f"\n{name} Image Quality:")
    print(f"  Blur: {quality.blur_score:.3f}")
    print(f"  Noise: {quality.noise_level:.3f}")
    print(f"  Contrast: {quality.contrast_score:.3f}")
    print(f"  Skew: {quality.skew_angle:.1f}°")
    print(f"  Recommendations: {quality.recommended_enhancements}")

# Test 2: Image preparation from bytes
print("\n\nTest 2: Image preparation from bytes")

# Convert numpy array to PIL Image and then to bytes
pil_image = Image.fromarray(clean_image)
img_buffer = io.BytesIO()
pil_image.save(img_buffer, format='PNG')
img_bytes = img_buffer.getvalue()

# Prepare image
prepared = processor.prepare_image(img_bytes, target_dpi=120)
print(f"Prepared image shape: {prepared.shape}")
print(f"Prepared image dtype: {prepared.dtype}")

# Test 3: Enhancement levels
print("\n\nTest 3: Testing different enhancement levels")

for level in [EnhancementLevel.NONE, EnhancementLevel.LIGHT, 
              EnhancementLevel.MODERATE, EnhancementLevel.AGGRESSIVE, 
              EnhancementLevel.AUTO]:
    enhanced = processor.enhance_for_ocr(noisy_image, level.value)
    quality_after = processor._detect_quality_issues(enhanced)
    print(f"\nEnhancement level: {level.value}")
    print(f"  Noise before: {processor._detect_quality_issues(noisy_image).noise_level:.3f}")
    print(f"  Noise after: {quality_after.noise_level:.3f}")

# Test 4: Batch resize
print("\n\nTest 4: Batch resize functionality")

batch = [clean_image, noisy_image, blurry_image]
target_size = (300, 300)
resized_batch = processor.batch_resize(batch, target_size)

print(f"Original shapes: {[img.shape for img in batch]}")
print(f"Resized shapes: {[img.shape for img in resized_batch]}")

# Test 5: Orientation detection
print("\n\nTest 5: Orientation detection")

# Create rotated versions
rotated_90 = cv2.rotate(clean_image, cv2.ROTATE_90_CLOCKWISE)
rotated_180 = cv2.rotate(clean_image, cv2.ROTATE_180)
rotated_270 = cv2.rotate(clean_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

orientations = [
    ("Original", clean_image),
    ("90° CW", rotated_90),
    ("180°", rotated_180),
    ("270° CW", rotated_270)
]

for name, img in orientations:
    angle = processor.detect_orientation(img)
    print(f"{name}: detected rotation = {angle}°")

# Test 6: Large image handling
print("\n\nTest 6: Large image validation")

# Create a large image
large_image = Image.new('RGB', (10000, 10000), color='white')
validation = processor._validate_image(large_image)
print(f"Large image valid: {validation.is_valid}")
print(f"Errors: {validation.errors}")
print(f"Warnings: {validation.warnings}")

# Test 7: Error handling
print("\n\nTest 7: Error handling")

try:
    # Try to load invalid image data
    processor.prepare_image(b"not an image")
except Exception as e:
    print(f"Expected error caught: {type(e).__name__}: {e}")

print("\n\nAll tests completed!")