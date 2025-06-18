import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
from src.utils.image_processor import ImageProcessor

print("=== Real Image Processing Test ===\n")

# Create a realistic document image
width, height = 800, 600
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)

# Add some text to simulate a document
texts = [
    "GPU OCR Server Test Document",
    "",
    "This is a test document created to verify the image",
    "processing capabilities of our OCR preprocessing pipeline.",
    "",
    "Features tested:",
    "- Image loading and validation",
    "- DPI normalization to 120 DPI",
    "- Color space conversion to grayscale",
    "- Quality issue detection",
    "- Automatic enhancement",
    "",
    "The image processor should handle various formats",
    "including PNG, JPEG, BMP, TIFF, and WebP.",
    "",
    "Processing optimizations for RTX 4090:",
    "- Maximum image size: 8192 pixels",
    "- Batch processing support",
    "- GPU-accelerated operations (when available)"
]

# Draw text on image
y_offset = 50
for text in texts:
    draw.text((50, y_offset), text, fill='black')
    y_offset += 30

# Add some noise and reduce quality to test enhancement
# Convert to numpy array
img_array = np.array(image)

# Add gaussian noise
noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
noisy_img = cv2.add(img_array, noise)

# Reduce contrast slightly
low_contrast = (noisy_img.astype(float) * 0.8 + 25).astype(np.uint8)

# Add slight rotation (2 degrees)
rows, cols = low_contrast.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 2, 1)
rotated = cv2.warpAffine(low_contrast, M, (cols, rows), borderValue=(255, 255, 255))

# Convert back to PIL Image
test_image = Image.fromarray(rotated)

# Save as different formats
formats = ['PNG', 'JPEG', 'BMP']
processor = ImageProcessor()

for fmt in formats:
    print(f"\nTesting {fmt} format:")
    
    # Save to bytes
    buffer = io.BytesIO()
    test_image.save(buffer, format=fmt, quality=85 if fmt == 'JPEG' else None)
    img_bytes = buffer.getvalue()
    
    print(f"  Image size: {len(img_bytes)} bytes")
    
    # Process image
    try:
        # Prepare image
        prepared = processor.prepare_image(img_bytes, target_dpi=120)
        print(f"  Prepared shape: {prepared.shape}")
        
        # Detect quality issues
        quality = processor._detect_quality_issues(prepared)
        print(f"  Quality analysis:")
        print(f"    - Blur score: {quality.blur_score:.3f}")
        print(f"    - Noise level: {quality.noise_level:.3f}")
        print(f"    - Contrast: {quality.contrast_score:.3f}")
        print(f"    - Skew angle: {quality.skew_angle:.1f}°")
        
        # Apply auto enhancement
        enhanced = processor.enhance_for_ocr(prepared, "auto")
        quality_after = processor._detect_quality_issues(enhanced)
        print(f"  After enhancement:")
        print(f"    - Blur score: {quality_after.blur_score:.3f}")
        print(f"    - Noise level: {quality_after.noise_level:.3f}")
        print(f"    - Contrast: {quality_after.contrast_score:.3f}")
        print(f"    - Skew angle: {quality_after.skew_angle:.1f}°")
        
        # Save sample output
        if fmt == 'PNG':
            cv2.imwrite('test_original.png', prepared)
            cv2.imwrite('test_enhanced.png', enhanced)
            print(f"\n  Sample images saved: test_original.png, test_enhanced.png")
            
    except Exception as e:
        print(f"  Error: {e}")

print("\n\nTest completed!")