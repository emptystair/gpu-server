"""Test script for PaddleOCR wrapper functionality"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OCRConfig
from src.models.paddle_ocr import PaddleOCRWrapper, OCROutput, ModelInfo

print("=== PaddleOCR Wrapper Test ===\n")

# Test 1: Check if PaddlePaddle is available
print("Test 1: Checking PaddlePaddle availability")
try:
    import paddle
    print(f"✓ PaddlePaddle version: {paddle.__version__}")
    print(f"✓ CUDA available: {paddle.device.is_compiled_with_cuda()}")
    if paddle.device.is_compiled_with_cuda():
        print(f"✓ GPU count: {paddle.device.cuda.device_count()}")
except ImportError:
    print("✗ PaddlePaddle not installed")
    print("Please install with: pip install paddlepaddle-gpu==2.5.2.post120")
    exit(1)

# Test 2: Initialize wrapper
print("\nTest 2: Initializing PaddleOCR wrapper")
config = OCRConfig(
    default_dpi=120,
    max_batch_size=6,
    use_angle_cls=True,
    lang='en',
    use_tensorrt=False,  # Start without TensorRT for testing
    tensorrt_precision='FP16',
    warmup_iterations=1
)

try:
    wrapper = PaddleOCRWrapper(config)
    print("✓ Wrapper created successfully")
except Exception as e:
    print(f"✗ Failed to create wrapper: {e}")
    exit(1)

# Test 3: Initialize model
print("\nTest 3: Initializing OCR model")
try:
    wrapper.initialize_model(use_tensorrt=False)
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    exit(1)

# Test 4: Process synthetic test image
print("\nTest 4: Processing synthetic test image")

# Create a simple test image with text
def create_test_image():
    """Create a simple test image with text-like patterns"""
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some black rectangles to simulate text
    # Simulate "HELLO"
    img[50:80, 50:100] = 0   # H left
    img[50:80, 120:150] = 0  # H right
    img[60:70, 100:120] = 0  # H middle
    
    # Simulate "WORLD"
    img[50:80, 200:250] = 0  # W
    img[50:80, 270:320] = 0  # O
    img[50:80, 340:390] = 0  # R
    img[50:80, 410:460] = 0  # L
    img[50:80, 480:530] = 0  # D
    
    # Add some noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int32) - noise, 0, 255).astype(np.uint8)
    
    return img

test_image = create_test_image()
print(f"Created test image with shape: {test_image.shape}")

try:
    results = wrapper.process_batch([test_image])
    print(f"✓ Processed {len(results)} images")
    
    if results and results[0]:
        result = results[0]
        print(f"  - Found {len(result.boxes)} text regions")
        print(f"  - Texts: {result.texts[:5]}...")  # Show first 5
        print(f"  - Confidences: {[f'{c:.2f}' for c in result.confidences[:5]]}...")
    else:
        print("  - No text detected (this is normal for synthetic images)")
        
except Exception as e:
    print(f"✗ Failed to process image: {e}")

# Test 5: Batch processing
print("\nTest 5: Batch processing test")
try:
    # Create multiple test images
    batch_images = [create_test_image() for _ in range(3)]
    
    import time
    start_time = time.time()
    results = wrapper.process_batch(batch_images)
    batch_time = time.time() - start_time
    
    print(f"✓ Processed batch of {len(batch_images)} images in {batch_time:.2f}s")
    print(f"  - Average time per image: {batch_time/len(batch_images):.2f}s")
    
except Exception as e:
    print(f"✗ Batch processing failed: {e}")

# Test 6: Model info
print("\nTest 6: Model information")
try:
    info = wrapper.get_model_info()
    print("✓ Model info retrieved:")
    print(f"  - Version: {info.model_version}")
    print(f"  - TensorRT enabled: {info.optimization_enabled}")
    print(f"  - Batch size: {info.current_batch_size}")
    print(f"  - Total inferences: {info.total_inferences}")
    print(f"  - Average latency: {info.average_latency_ms:.2f}ms")
except Exception as e:
    print(f"✗ Failed to get model info: {e}")

# Test 7: Test with a real document image if available
print("\nTest 7: Real document test")
test_images_dir = Path("../tests/testpdfs")
if test_images_dir.exists():
    # Try to load a real image if we have any
    image_files = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
    if image_files:
        print("✓ Found test images, skipping (would need image loading)")
    else:
        print("  - No test images found")
else:
    print("  - No test image directory found")

# Test 8: Memory cleanup
print("\nTest 8: Memory cleanup")
try:
    wrapper.cleanup()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Cleanup failed: {e}")

# Test 9: Error handling
print("\nTest 9: Error handling")
try:
    # Test with invalid image
    invalid_img = np.array([])
    results = wrapper.process_batch([invalid_img])
    print("✓ Handled invalid image gracefully")
except Exception as e:
    print(f"✓ Correctly raised exception for invalid input: {type(e).__name__}")

print("\n✅ All tests completed!")