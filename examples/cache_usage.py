"""
Example: Using the distributed cache manager

This example demonstrates cache usage in different deployment scenarios:
1. Local GPU server with Celery workers
2. AWS Batch distributed processing
"""

import os
import time
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils.cache_manager import CacheManager
from src.api.schemas import OCRResult, PageResult, TextRegion, BoundingBox


def example_local_gpu_server():
    """Example: Local GPU server with Redis shared cache"""
    
    print("=== Local GPU Server Example ===\n")
    
    # Configuration for local deployment
    config = load_config()
    
    # Initialize cache manager
    cache_manager = CacheManager(config.cache)
    
    # Simulate OCR result
    ocr_result = OCRResult(
        pages=[
            PageResult(
                page_number=1,
                text="Sample extracted text from page 1",
                words=[],
                lines=[],
                confidence=0.95,
                processing_time_ms=150.5
            )
        ],
        total_pages=1,
        processing_time_ms=200.0,
        document_hash="abc123",
        metadata={"source": "example.pdf"}
    )
    
    # Generate cache key based on document and processing parameters
    cache_key = cache_manager.generate_key(
        document_hash="abc123",
        dpi=150,
        strategy="balanced",
        language="en"
    )
    
    print(f"1. Storing OCR result in cache")
    print(f"   Key: {cache_key[:16]}...")
    
    # Store result
    success = cache_manager.set(cache_key, ocr_result, ttl=3600)
    print(f"   Stored: {'✓' if success else '✗'}")
    
    # Retrieve result
    print(f"\n2. Retrieving from cache")
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        print(f"   ✓ Retrieved successfully")
        print(f"   - Pages: {cached_result.total_pages}")
        print(f"   - Processing time: {cached_result.processing_time_ms}ms")
    
    # Check statistics
    stats = cache_manager.get_stats()
    print(f"\n3. Cache statistics")
    print(f"   - Hit rate: {stats['overall_hit_rate']:.2%}")
    print(f"   - Memory tier: {stats['tiers']['memory']['size_mb']:.2f}MB")
    print(f"   - Redis tier: {'Connected' if stats['tiers']['redis']['enabled'] else 'Not available'}")


def example_celery_worker():
    """Example: Celery worker with optimized cache"""
    
    print("\n=== Celery Worker Example ===\n")
    
    # Simulate Celery worker environment
    os.environ['CELERY_WORKER_ID'] = 'worker-gpu-001'
    
    config = load_config()
    cache_manager = CacheManager(config.cache)
    
    print(f"Worker Configuration:")
    print(f"- Worker ID: {cache_manager.worker_id}")
    print(f"- Memory cache: 128MB (worker optimized)")
    print(f"- Redis: Primary shared cache")
    print(f"- Disk: Local fallback cache")
    
    # Simulate batch processing
    print(f"\nProcessing document batch:")
    
    documents = [
        {"id": "doc1", "pages": 10},
        {"id": "doc2", "pages": 5},
        {"id": "doc3", "pages": 15}
    ]
    
    for doc in documents:
        # Check cache first
        cache_key = cache_manager.generate_key(
            document_id=doc['id'],
            worker=cache_manager.worker_id
        )
        
        cached = cache_manager.get(cache_key)
        if cached:
            print(f"✓ {doc['id']}: Cache hit - skipping processing")
        else:
            print(f"✗ {doc['id']}: Cache miss - processing {doc['pages']} pages")
            
            # Simulate processing
            result = {"processed": True, "pages": doc['pages']}
            cache_manager.set(cache_key, result, ttl=7200)
    
    # Clean up
    del os.environ['CELERY_WORKER_ID']


def example_aws_batch():
    """Example: AWS Batch with S3 persistent cache"""
    
    print("\n=== AWS Batch Example ===\n")
    
    # Simulate AWS Batch environment
    os.environ['AWS_BATCH_JOB_ID'] = 'ocr-batch-job-12345'
    os.environ['S3_CACHE_BUCKET'] = 'my-ocr-cache-bucket'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    
    config = load_config()
    config.cache.s3_cache_bucket = 'my-ocr-cache-bucket'
    
    # Note: This will fail without actual AWS credentials
    # In production, use IAM roles for authentication
    try:
        cache_manager = CacheManager(config.cache)
        
        print(f"AWS Batch Configuration:")
        print(f"- Job ID: {cache_manager.worker_id}")
        print(f"- Memory cache: 128MB (worker optimized)")
        print(f"- Redis: Primary shared cache (if available)")
        print(f"- S3 bucket: {config.cache.s3_cache_bucket}")
        
        # Example: Processing large document set
        print(f"\nProcessing large document batch:")
        
        # In AWS Batch, you might process a subset of documents
        # assigned to this particular job
        job_documents = get_job_documents()  # Hypothetical function
        
        for doc_s3_key in job_documents:
            # Generate cache key including S3 location
            cache_key = cache_manager.generate_key(
                s3_key=doc_s3_key,
                processing_version="v2.1"
            )
            
            # Check if already processed
            if cache_manager.get(cache_key):
                print(f"✓ {doc_s3_key}: Already processed")
                continue
            
            # Process document (simplified)
            print(f"⚡ {doc_s3_key}: Processing...")
            result = process_document_from_s3(doc_s3_key)
            
            # Cache with longer TTL for S3
            cache_manager.set(cache_key, result, ttl=86400)  # 24 hours
            
    except Exception as e:
        print(f"Note: AWS example requires valid AWS credentials: {e}")
    
    # Clean up
    del os.environ['AWS_BATCH_JOB_ID']
    if 'S3_CACHE_BUCKET' in os.environ:
        del os.environ['S3_CACHE_BUCKET']
    if 'AWS_DEFAULT_REGION' in os.environ:
        del os.environ['AWS_DEFAULT_REGION']


def example_cache_patterns():
    """Example: Common caching patterns"""
    
    print("\n=== Caching Patterns ===\n")
    
    config = load_config()
    cache_manager = CacheManager(config.cache)
    
    # Pattern 1: Document fingerprinting
    print("1. Document fingerprinting pattern:")
    
    import hashlib
    document_content = b"PDF binary content here..."
    document_hash = hashlib.sha256(document_content).hexdigest()
    
    cache_key = cache_manager.generate_key(
        doc_hash=document_hash,
        version="2.0"
    )
    print(f"   Cache key from content hash: {cache_key[:16]}...")
    
    # Pattern 2: Partial results caching
    print("\n2. Partial results pattern:")
    
    # Cache individual pages for resume capability
    for page_num in range(1, 6):
        page_key = cache_manager.generate_key(
            doc_hash=document_hash,
            page=page_num
        )
        page_result = {"page": page_num, "text": f"Page {page_num} content"}
        cache_manager.set(page_key, page_result, ttl=1800)
    print("   ✓ Cached 5 individual pages")
    
    # Pattern 3: Cache invalidation
    print("\n3. Cache invalidation pattern:")
    
    # Invalidate all pages for a document
    invalidated = 0
    for page_num in range(1, 6):
        page_key = cache_manager.generate_key(
            doc_hash=document_hash,
            page=page_num
        )
        if cache_manager.delete(page_key):
            invalidated += 1
    print(f"   ✓ Invalidated {invalidated} cached pages")
    
    # Pattern 4: Tiered TTL strategy
    print("\n4. Tiered TTL strategy:")
    
    # Hot data - short TTL
    hot_key = cache_manager.generate_key(recent=True)
    cache_manager.set(hot_key, {"status": "processing"}, ttl=300)  # 5 min
    
    # Warm data - medium TTL
    warm_key = cache_manager.generate_key(recent=False)
    cache_manager.set(warm_key, {"status": "completed"}, ttl=3600)  # 1 hour
    
    # Cold data - long TTL (persistent cache)
    cold_key = cache_manager.generate_key(archive=True)
    cache_manager.set(cold_key, {"status": "archived"}, ttl=86400)  # 24 hours
    
    print("   ✓ Set tiered TTLs: hot(5m), warm(1h), cold(24h)")


def get_job_documents():
    """Mock function to simulate getting documents for batch job"""
    return [
        "s3://input-bucket/batch1/doc1.pdf",
        "s3://input-bucket/batch1/doc2.pdf",
        "s3://input-bucket/batch1/doc3.pdf"
    ]


def process_document_from_s3(s3_key):
    """Mock function to simulate document processing"""
    return {
        "s3_key": s3_key,
        "pages": 10,
        "status": "completed",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    print("Distributed Cache Manager Examples\n")
    print("=" * 50)
    
    # Run examples
    example_local_gpu_server()
    example_celery_worker()
    example_aws_batch()
    example_cache_patterns()
    
    print("\n" + "=" * 50)
    print("\nKey Takeaways:")
    print("1. Cache manager adapts to deployment environment")
    print("2. Memory cache is always available (LRU, thread-safe)")
    print("3. Redis provides shared cache between workers")
    print("4. S3/disk provides persistent cache for results")
    print("5. Automatic promotion between cache tiers")
    print("6. Graceful degradation if backends unavailable")