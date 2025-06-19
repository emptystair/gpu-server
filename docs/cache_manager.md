# Distributed Cache Manager Documentation

## Overview

The Cache Manager provides a multi-tier caching system designed to work in both local GPU server environments with Celery workers and distributed AWS Batch processing. It implements a three-tier cache hierarchy: Memory → Redis → S3/Disk.

## Architecture

```
┌─────────────────────┐
│   Memory Cache      │  ← Per-worker LRU (128MB workers, 512MB main)
├─────────────────────┤
│    Redis Cache      │  ← Shared between all workers (primary)
├─────────────────────┤
│  S3/Disk Cache      │  ← Persistent storage (S3 for AWS, disk for local)
└─────────────────────┘
```

## Features

- **Environment Detection**: Automatically detects AWS Batch vs local deployment
- **Multi-tier Caching**: Three-level cache with automatic promotion
- **Worker Optimization**: Smaller memory caches for distributed workers
- **Thread-safe Operations**: Safe for concurrent access
- **Graceful Degradation**: Falls back if cache backends unavailable
- **Cache Statistics**: Comprehensive monitoring and hit rates

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=secretpassword

# Cache Sizes
MEMORY_CACHE_SIZE_MB=512      # Main process
DISK_CACHE_SIZE_GB=10.0
DISK_CACHE_PATH=./cache

# AWS S3 Configuration (for AWS Batch)
S3_CACHE_BUCKET=my-ocr-cache
S3_CACHE_REGION=us-east-1

# Worker Identification
CELERY_WORKER_ID=worker-001    # For Celery workers
AWS_BATCH_JOB_ID=job-123       # Set by AWS Batch
```

### Python Configuration

```python
from src.config import CacheConfig

config = CacheConfig(
    enable_cache=True,
    cache_ttl=3600,  # 1 hour default
    redis_host="localhost",
    redis_port=6379,
    memory_cache_size_mb=512,
    disk_cache_size_gb=10.0,
    disk_cache_path="./cache",
    s3_cache_bucket="my-ocr-cache",
    s3_cache_region="us-east-1"
)
```

## Usage Examples

### Basic Operations

```python
from src.utils.cache_manager import CacheManager
from src.config import load_config

# Initialize
config = load_config()
cache = CacheManager(config.cache)

# Store value
key = cache.generate_key("document.pdf", dpi=150)
value = {"pages": 10, "text": "extracted text"}
cache.set(key, value, ttl=3600)

# Retrieve value
result = cache.get(key)

# Delete value
cache.delete(key)

# Check statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['overall_hit_rate']:.2%}")
```

### Key Generation

```python
# Generate consistent cache keys
key = cache.generate_key(
    document_hash="abc123",
    dpi=150,
    strategy="balanced",
    language="en"
)

# Keys are deterministic - same inputs = same key
key1 = cache.generate_key("doc.pdf", version=1)
key2 = cache.generate_key("doc.pdf", version=1)
assert key1 == key2
```

### Environment-Specific Usage

#### Local GPU Server
```python
# Main process has larger memory cache
# Redis is primary shared cache
# Disk provides persistent fallback

cache = CacheManager(config)
# Memory: 512MB, Redis: Shared, Disk: Local
```

#### Celery Worker
```python
os.environ['CELERY_WORKER_ID'] = 'worker-001'
cache = CacheManager(config)
# Memory: 128MB (optimized), Redis: Shared, Disk: Worker-specific
```

#### AWS Batch
```python
os.environ['AWS_BATCH_JOB_ID'] = 'job-123'
os.environ['S3_CACHE_BUCKET'] = 'my-cache'
cache = CacheManager(config)
# Memory: 128MB, Redis: If available, S3: Persistent
```

## Cache Tiers

### 1. Memory Cache (L1)
- **Size**: 128MB (workers) or 512MB (main process)
- **Scope**: Per-process
- **Eviction**: LRU (Least Recently Used)
- **Thread-safe**: Yes
- **TTL Support**: Yes

### 2. Redis Cache (L2)
- **Size**: Configured by Redis server
- **Scope**: Shared across all workers
- **Persistence**: Optional (Redis persistence)
- **High Availability**: Redis clustering supported
- **TTL Support**: Native Redis expiration

### 3. Persistent Cache (L3)
- **Local Mode**: Disk-based cache
  - Location: `./cache/{worker_id}/`
  - Size limit: Configurable (default 10GB)
  - Sharding: First 2 chars of key
- **AWS Mode**: S3-based cache
  - Bucket: Configured via environment
  - Prefix: `ocr-cache/{job_id}/`
  - Storage class: STANDARD_IA

## Cache Promotion

When a cache miss occurs at a higher tier, the value is fetched from lower tiers and promoted:

```
Request → Memory (miss) → Redis (miss) → S3/Disk (hit)
              ↑               ↑
              └───────────────┴─── Value promoted
```

## Performance Considerations

### Batch Processing
```python
# Pre-warm cache for batch
for doc in documents:
    key = cache.generate_key(doc.id)
    if not cache.get(key):
        # Process and cache
        result = process_document(doc)
        cache.set(key, result)
```

### Memory Management
```python
# Monitor cache size
stats = cache.get_stats()
if stats['tiers']['memory']['size_mb'] > 400:
    # Memory cache approaching limit
    # Consider clearing old entries
```

### TTL Strategies
```python
# Hot data - short TTL
cache.set(hot_key, data, ttl=300)      # 5 minutes

# Warm data - medium TTL  
cache.set(warm_key, data, ttl=3600)    # 1 hour

# Cold data - long TTL
cache.set(cold_key, data, ttl=86400)   # 24 hours
```

## Monitoring

### Cache Statistics
```python
stats = cache.get_stats()

# Overall metrics
print(f"Total requests: {stats['total_requests']}")
print(f"Overall hit rate: {stats['overall_hit_rate']:.2%}")

# Per-tier metrics
for tier, info in stats['tiers'].items():
    print(f"{tier}: {info['size_mb']:.2f}MB")

# Hit rates by tier
for tier, rate in stats['hit_rates'].items():
    print(f"{tier} hit rate: {rate:.2%}")
```

### Health Checks
```python
# Check cache backend availability
if cache.redis_cache:
    print("Redis: Connected")
else:
    print("Redis: Not available")

if cache.persistent_cache:
    print(f"Persistent: {type(cache.persistent_cache).__name__}")
```

## Best Practices

1. **Key Design**: Include all parameters that affect output
   ```python
   key = cache.generate_key(
       document_hash=hash,
       dpi=dpi,
       strategy=strategy,
       model_version=version
   )
   ```

2. **TTL Selection**: Balance freshness vs performance
   - Processing results: 1-24 hours
   - Temporary data: 5-60 minutes
   - Model outputs: 24-72 hours

3. **Error Handling**: Cache operations should never break processing
   ```python
   try:
       cached = cache.get(key)
   except Exception as e:
       logger.warning(f"Cache get failed: {e}")
       cached = None
   ```

4. **Batch Operations**: Use cache to avoid reprocessing
   ```python
   results = []
   for item in batch:
       key = cache.generate_key(item.id)
       result = cache.get(key)
       if not result:
           result = process_item(item)
           cache.set(key, result)
       results.append(result)
   ```

## Troubleshooting

### Redis Connection Issues
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Check Redis memory
redis-cli info memory
```

### S3 Access Issues
```bash
# Verify S3 access
aws s3 ls s3://my-cache-bucket/ocr-cache/

# Check IAM permissions
aws sts get-caller-identity
```

### Cache Size Issues
```python
# Clear specific tier
if cache.memory_cache:
    cache.memory_cache.clear()

# Clear all tiers
cleared = cache.clear()
print(f"Cleared {cleared} entries")
```

## Performance Tuning

### Memory Cache Size
- Workers: 128MB (prevents memory pressure)
- Main process: 512MB (better hit rate)
- Adjust based on available RAM

### Redis Configuration
```
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1  # Persistence
```

### S3 Optimization
- Use STANDARD_IA for cost savings
- Enable S3 Transfer Acceleration for large files
- Consider S3 lifecycle policies for old cache entries