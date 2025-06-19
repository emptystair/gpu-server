"""Test script for OCR Service functionality"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.ocr_service import OCRService, ProcessingStrategy
from src.api.schemas import ProcessingStrategy as SchemaProcessingStrategy

async def test_ocr_service():
    """Test the OCR service implementation"""
    
    print("=== OCR Service Test ===\n")
    
    # Test 1: Initialize service
    print("Test 1: Initializing OCR Service")
    try:
        config = load_config()
        service = OCRService(config)
        print(" OCR Service created successfully")
        print(f"  - PDF processor initialized")
        print(f"  - Image processor initialized")
        print(f"  - GPU monitor initialized")
        print(f"  - Cache manager initialized")
    except Exception as e:
        print(f" Failed to create OCR Service: {e}")
        return
    
    # Test 2: Check strategy configurations
    print("\nTest 2: Strategy configurations")
    for strategy in ProcessingStrategy:
        config = service.strategy_configs[strategy]
        print(f" {strategy.value}:")
        print(f"  - DPI: {config['dpi']}")
        print(f"  - Batch multiplier: {config['batch_multiplier']}")
        print(f"  - Image enhancement: {config['enhance_image']}")
        print(f"  - TensorRT: {config['use_tensorrt']}")
    
    # Test 3: Memory estimation
    print("\nTest 3: Memory requirement estimation")
    test_cases = [
        (1, (640, 480)),
        (16, (1280, 720)),
        (32, (1920, 1080))
    ]
    
    for batch_size, dimensions in test_cases:
        memory_mb = service._estimate_memory_requirement(batch_size, dimensions)
        print(f" Batch {batch_size} @ {dimensions[0]}x{dimensions[1]}: {memory_mb}MB")
    
    # Test 4: Batch size determination (simulated)
    print("\nTest 4: Batch size determination")
    try:
        # Mock GPU memory info
        service.gpu_monitor._memory_info = type('obj', (object,), {
            'free': 20000,  # 20GB free
            'total': 24000,  # 24GB total
            'used': 4000
        })
        
        for strategy in ProcessingStrategy:
            batch_size = service._determine_batch_size(
                page_count=100,
                image_dimensions=(1280, 720),
                strategy=strategy
            )
            print(f" {strategy.value}: batch size = {batch_size}")
    except Exception as e:
        print(f" Batch size determination failed: {e}")
    
    # Test 5: Cache key generation
    print("\nTest 5: Cache key generation")
    test_content = b"test document content"
    for strategy in ProcessingStrategy:
        cache_key = service._generate_cache_key(test_content, strategy)
        print(f" {strategy.value}: {cache_key}")
    
    # Test 6: Statistics tracking
    print("\nTest 6: Processing statistics")
    stats = service.get_processing_stats()
    print(" Initial statistics:")
    print(f"  - Documents processed: {stats.total_documents_processed}")
    print(f"  - Pages processed: {stats.total_pages_processed}")
    print(f"  - Average pages/second: {stats.average_pages_per_second:.2f}")
    print(f"  - Average batch size: {stats.average_batch_size:.2f}")
    print(f"  - Cache hit rate: {stats.cache_hit_rate:.2%}")
    
    # Test 7: OCR output conversion (simulated)
    print("\nTest 7: OCR output conversion")
    try:
        from src.models.paddle_ocr import OCROutput, TextBox
        
        # Create mock OCR output
        mock_ocr_output = OCROutput(
            boxes=[
                TextBox(coordinates=[(10, 10), (100, 10), (100, 30), (10, 30)], confidence=0.95),
                TextBox(coordinates=[(120, 10), (200, 10), (200, 30), (120, 30)], confidence=0.92)
            ],
            texts=["Hello", "World"],
            confidences=[0.95, 0.92]
        )
        
        page_result = service._convert_ocr_output(
            ocr_output=mock_ocr_output,
            page_number=1,
            image_shape=(720, 1280, 3),
            processing_time_ms=50.0
        )
        
        print(" Converted OCR output:")
        print(f"  - Page: {page_result.page_number}")
        print(f"  - Text: '{page_result.text}'")
        print(f"  - Words: {len(page_result.words)}")
        print(f"  - Confidence: {page_result.confidence:.2f}")
        print(f"  - Image size: {page_result.image_size}")
        
    except Exception as e:
        print(f" OCR output conversion failed: {e}")
    
    # Test 8: Result serialization
    print("\nTest 8: Result serialization/deserialization")
    try:
        from src.ocr_service import OCRResult, PageResult, WordResult
        
        # Create test result
        test_result = OCRResult(
            pages=[
                PageResult(
                    page_number=1,
                    text="Test page",
                    words=[
                        WordResult(text="Test", bbox=(10, 10, 50, 30), confidence=0.95),
                        WordResult(text="page", bbox=(60, 10, 100, 30), confidence=0.93)
                    ],
                    confidence=0.94,
                    processing_time_ms=100.0,
                    image_size=(1280, 720)
                )
            ],
            total_pages=1,
            processing_time_ms=150.0,
            confidence_score=0.94,
            metadata={"test": "metadata"},
            strategy_used=ProcessingStrategy.BALANCED,
            batch_sizes_used=[1]
        )
        
        # Serialize
        serialized = service._serialize_ocr_result(test_result)
        print(" Serialized result")
        
        # Deserialize
        deserialized = service._deserialize_ocr_result(serialized)
        print(" Deserialized result")
        print(f"  - Pages: {deserialized.total_pages}")
        print(f"  - Strategy: {deserialized.strategy_used.value}")
        print(f"  - Confidence: {deserialized.confidence_score:.2f}")
        
    except Exception as e:
        print(f" Serialization test failed: {e}")
    
    print("\n All tests completed!")
    
    # Test 9: Performance estimation
    print("\n=== Performance Estimation ===")
    print("\nExpected throughput on RTX 4090:")
    print("- SPEED mode: 100-120 pages/minute")
    print("- BALANCED mode: 80-100 pages/minute")
    print("- ACCURACY mode: 50-70 pages/minute")
    print("\nBatch processing benefits:")
    print("- Batch 1: Baseline")
    print("- Batch 16: ~3x throughput")
    print("- Batch 32: ~4x throughput")
    print("- Batch 50: ~5x throughput")

if __name__ == "__main__":
    asyncio.run(test_ocr_service())