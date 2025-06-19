"""
Example: Using the OCR Service

This example demonstrates how to use the OCR service to process documents
with different strategies and options.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.ocr_service import OCRService, ProcessingStrategy


async def process_pdf_example():
    """Example of processing a PDF document"""
    
    print("=== PDF Processing Example ===\n")
    
    # Initialize service
    config = load_config()
    service = OCRService(config)
    
    # Initialize models and GPU
    print("Initializing OCR service...")
    await service.initialize()
    print("✓ Service initialized\n")
    
    # Simulate PDF processing
    print("Processing PDF with different strategies:\n")
    
    # Example PDF bytes (in real usage, read from file)
    # pdf_bytes = Path("document.pdf").read_bytes()
    pdf_bytes = b"PDF mock content"  # Mock for example
    
    # Process with SPEED strategy
    print("1. SPEED Strategy (120 DPI, larger batches)")
    try:
        result = await service.process_document(
            file_bytes=pdf_bytes,
            file_type="application/pdf",
            processing_strategy=ProcessingStrategy.SPEED
        )
        print(f"   - Pages processed: {result.total_pages}")
        print(f"   - Processing time: {result.processing_time_ms:.0f}ms")
        print(f"   - Confidence: {result.confidence_score:.2%}")
        print(f"   - Batch sizes used: {result.batch_sizes_used}\n")
    except Exception as e:
        print(f"   ✗ Processing failed: {e}\n")
    
    # Process with ACCURACY strategy
    print("2. ACCURACY Strategy (200 DPI, image enhancement)")
    try:
        result = await service.process_document(
            file_bytes=pdf_bytes,
            file_type="application/pdf",
            processing_strategy=ProcessingStrategy.ACCURACY
        )
        print(f"   - Pages processed: {result.total_pages}")
        print(f"   - Processing time: {result.processing_time_ms:.0f}ms")
        print(f"   - Confidence: {result.confidence_score:.2%}")
        print(f"   - Enhanced images: Yes\n")
    except Exception as e:
        print(f"   ✗ Processing failed: {e}\n")
    
    # Process with custom DPI
    print("3. Custom DPI Override")
    try:
        result = await service.process_document(
            file_bytes=pdf_bytes,
            file_type="application/pdf",
            processing_strategy=ProcessingStrategy.BALANCED,
            dpi=300  # Override default DPI
        )
        print(f"   - Custom DPI: 300")
        print(f"   - Processing time: {result.processing_time_ms:.0f}ms\n")
    except Exception as e:
        print(f"   ✗ Processing failed: {e}\n")
    
    # Clean up
    await service.cleanup()


async def process_image_example():
    """Example of processing image files"""
    
    print("=== Image Processing Example ===\n")
    
    # Initialize service
    config = load_config()
    service = OCRService(config)
    await service.initialize()
    
    # Example image bytes (in real usage, read from file)
    # image_bytes = Path("document.jpg").read_bytes()
    image_bytes = b"JPEG mock content"  # Mock for example
    
    print("Processing single image:")
    try:
        result = await service.process_document(
            file_bytes=image_bytes,
            file_type="image/jpeg",
            processing_strategy=ProcessingStrategy.BALANCED
        )
        
        # Access results
        if result.pages:
            page = result.pages[0]
            print(f"✓ Text extracted: {len(page.text)} characters")
            print(f"✓ Words detected: {len(page.words)}")
            print(f"✓ Confidence: {page.confidence:.2%}")
            
            # Show first few words
            if page.words:
                print("\nFirst 5 words:")
                for i, word in enumerate(page.words[:5]):
                    print(f"  {i+1}. '{word.text}' (conf: {word.confidence:.2f})")
    except Exception as e:
        print(f"✗ Processing failed: {e}")
    
    await service.cleanup()


async def batch_processing_example():
    """Example of batch processing optimization"""
    
    print("=== Batch Processing Example ===\n")
    
    # Initialize service
    config = load_config()
    service = OCRService(config)
    await service.initialize()
    
    # Check processing statistics
    print("Initial statistics:")
    stats = service.get_processing_stats()
    print(f"- Documents: {stats.total_documents_processed}")
    print(f"- Pages: {stats.total_pages_processed}")
    print(f"- Avg pages/sec: {stats.average_pages_per_second:.2f}\n")
    
    # Process multiple documents
    documents = [
        (b"PDF 1", "application/pdf"),
        (b"PDF 2", "application/pdf"),
        (b"Image 1", "image/jpeg"),
    ]
    
    for i, (content, mime_type) in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}...")
        try:
            result = await service.process_document(
                file_bytes=content,
                file_type=mime_type,
                processing_strategy=ProcessingStrategy.SPEED
            )
            print(f"  ✓ Completed in {result.processing_time_ms:.0f}ms")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Final statistics
    print("\nFinal statistics:")
    stats = service.get_processing_stats()
    print(f"- Documents: {stats.total_documents_processed}")
    print(f"- Pages: {stats.total_pages_processed}")
    print(f"- Avg pages/sec: {stats.average_pages_per_second:.2f}")
    print(f"- Avg batch size: {stats.average_batch_size:.1f}")
    print(f"- Cache hit rate: {stats.cache_hit_rate:.2%}")
    print(f"- GPU utilization: {stats.gpu_utilization_average:.1f}%")
    
    await service.cleanup()


async def main():
    """Run all examples"""
    
    print("GPU OCR Service Examples\n")
    print("=" * 50 + "\n")
    
    # Note: In real usage, only one of these would run at a time
    # as they each initialize and cleanup the service
    
    try:
        # PDF processing example
        # await process_pdf_example()
        
        # Image processing example
        # await process_image_example()
        
        # Batch processing example
        # await batch_processing_example()
        
        print("Examples would process real documents in production.")
        print("\nKey features demonstrated:")
        print("- Multiple processing strategies (SPEED, ACCURACY, BALANCED)")
        print("- Dynamic batch sizing based on GPU memory")
        print("- Result caching for repeated documents")
        print("- Comprehensive statistics tracking")
        print("- Error handling and recovery")
        
    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())