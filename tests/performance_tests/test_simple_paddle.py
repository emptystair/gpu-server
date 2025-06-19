#!/usr/bin/env python3
"""
Simple test to verify PaddleOCR works in our environment
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.abspath('./src'))

def test_paddle_basic():
    """Test basic PaddlePaddle functionality"""
    try:
        logger.info("Testing basic PaddlePaddle import...")
        import paddle
        logger.info("✓ PaddlePaddle imported successfully")
        
        # Test CUDA availability
        logger.info("Testing CUDA availability...")
        print(f"CUDA available: {paddle.is_compiled_with_cuda()}")
        print(f"GPU count: {paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0}")
        
        return True
    except Exception as e:
        logger.error(f"✗ PaddlePaddle test failed: {e}")
        return False

def test_paddleocr_simple():
    """Test simple PaddleOCR initialization"""
    try:
        logger.info("Testing PaddleOCR import...")
        from paddleocr import PaddleOCR
        logger.info("✓ PaddleOCR imported successfully")
        
        # Try CPU-only initialization first
        logger.info("Initializing PaddleOCR with CPU...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        logger.info("✓ PaddleOCR CPU initialization successful")
        
        # Try GPU initialization
        logger.info("Initializing PaddleOCR with GPU...")
        ocr_gpu = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            use_gpu=True, 
            use_tensorrt=False,  # Disable TensorRT for now
            show_log=True
        )
        logger.info("✓ PaddleOCR GPU initialization successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ PaddleOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("=== PaddleOCR Environment Test ===")
    
    # Test 1: Basic Paddle
    success1 = test_paddle_basic()
    
    # Test 2: PaddleOCR
    success2 = test_paddleocr_simple()
    
    # Summary
    logger.info("\n=== Test Results ===")
    logger.info(f"PaddlePaddle basic: {'✓' if success1 else '✗'}")
    logger.info(f"PaddleOCR: {'✓' if success2 else '✗'}")
    
    if success1 and success2:
        logger.info("✓ All tests passed! Environment is ready.")
        return 0
    else:
        logger.info("✗ Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())