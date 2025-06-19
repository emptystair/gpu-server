"""
Cache Manager Module

Multi-tier caching system supporting both local GPU server with Celery workers
and AWS Batch distributed processing. Implements memory -> Redis -> S3/disk hierarchy.
"""

import os
import json
import time
import hashlib
import logging
import pickle
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import lru_cache
from abc import ABC, abstractmethod
import threading
from collections import OrderedDict

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception

try:
    import boto3
    from botocore.exceptions import BotoCore3Error, ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None
    BotoCore3Error = Exception
    ClientError = Exception

from ..config import CacheConfig
# from ..api.schemas import OCRResult, PageResult  # Commented to avoid circular import

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    memory_hits: int = 0
    memory_misses: int = 0
    redis_hits: int = 0
    redis_misses: int = 0
    persistent_hits: int = 0
    persistent_misses: int = 0
    total_requests: int = 0
    total_size_mb: float = 0.0
    entry_count: int = 0
    
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        total_hits = self.memory_hits + self.redis_hits + self.persistent_hits
        return total_hits / self.total_requests
    
    @property
    def tier_hit_rates(self) -> Dict[str, float]:
        """Calculate hit rates per tier"""
        rates = {}
        
        # Memory tier
        memory_requests = self.memory_hits + self.memory_misses
        rates['memory'] = self.memory_hits / memory_requests if memory_requests > 0 else 0.0
        
        # Redis tier (only counts if memory missed)
        redis_requests = self.redis_hits + self.redis_misses
        rates['redis'] = self.redis_hits / redis_requests if redis_requests > 0 else 0.0
        
        # Persistent tier (only counts if both memory and redis missed)
        persistent_requests = self.persistent_hits + self.persistent_misses
        rates['persistent'] = self.persistent_hits / persistent_requests if persistent_requests > 0 else 0.0
        
        return rates


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Store value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    def get_size(self) -> float:
        """Get total size in MB"""
        pass


class MemoryCacheBackend(CacheBackend):
    """Thread-safe in-memory LRU cache backend"""
    
    def __init__(self, max_size_mb: int = 128):
        self.max_size_mb = max_size_mb
        self.cache = OrderedDict()
        self.size_bytes = 0
        self.lock = threading.RLock()
        self.ttl_map = {}
        logger.info(f"Initialized memory cache with {max_size_mb}MB limit")
    
    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            # Check if key exists and not expired
            if key in self.cache:
                if self._is_expired(key):
                    self._remove(key)
                    return None
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                value_size = len(value)
                
                # Remove old value if exists
                if key in self.cache:
                    self._remove(key)
                
                # Check if value fits in cache
                if value_size > self.max_size_mb * 1024 * 1024:
                    logger.warning(f"Value too large for memory cache: {value_size / 1024 / 1024:.2f}MB")
                    return False
                
                # Evict entries if necessary
                while (self.size_bytes + value_size) > (self.max_size_mb * 1024 * 1024) and self.cache:
                    self._evict_lru()
                
                # Add new entry
                self.cache[key] = value
                self.size_bytes += value_size
                
                # Set TTL if provided
                if ttl:
                    self.ttl_map[key] = time.time() + ttl
                
                return True
                
            except Exception as e:
                logger.error(f"Memory cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                self._remove(key)
                return True
            return False
    
    def clear(self) -> int:
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.ttl_map.clear()
            self.size_bytes = 0
            return count
    
    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                if self._is_expired(key):
                    self._remove(key)
                    return False
                return True
            return False
    
    def get_size(self) -> float:
        with self.lock:
            return self.size_bytes / 1024 / 1024
    
    def _remove(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            value = self.cache[key]
            self.size_bytes -= len(value)
            del self.cache[key]
            self.ttl_map.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            self.size_bytes -= len(self.cache.get(key, b''))
            self.ttl_map.pop(key, None)
    
    def _is_expired(self, key: str) -> bool:
        """Check if key has expired"""
        if key in self.ttl_map:
            return time.time() > self.ttl_map[key]
        return False


class RedisCacheBackend(CacheBackend):
    """Redis cache backend for distributed caching"""
    
    def __init__(self, redis_url: str, db: int = 0, key_prefix: str = "ocr_cache:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install redis package.")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Handle binary data
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.client.get(self._make_key(key))
        except RedisError as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        try:
            return bool(self.client.set(self._make_key(key), value, ex=ttl))
        except RedisError as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(self._make_key(key)))
        except RedisError as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> int:
        try:
            # Use SCAN to find all keys with our prefix
            keys = []
            cursor = 0
            while True:
                cursor, batch = self.client.scan(
                    cursor, 
                    match=f"{self.key_prefix}*",
                    count=1000
                )
                keys.extend(batch)
                if cursor == 0:
                    break
            
            if keys:
                return self.client.delete(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Redis clear error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(self._make_key(key)))
        except RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def get_size(self) -> float:
        """Estimate total size of cache entries"""
        try:
            # Get memory usage info
            info = self.client.info('memory')
            used_memory_mb = info.get('used_memory', 0) / 1024 / 1024
            
            # Rough estimate based on key count
            cursor = 0
            total_size = 0
            sample_count = 0
            
            while sample_count < 100:  # Sample first 100 keys
                cursor, keys = self.client.scan(
                    cursor,
                    match=f"{self.key_prefix}*",
                    count=10
                )
                
                for key in keys:
                    size = self.client.memory_usage(key) or 0
                    total_size += size
                    sample_count += 1
                
                if cursor == 0 or sample_count >= 100:
                    break
            
            # Extrapolate based on sample
            if sample_count > 0:
                avg_size = total_size / sample_count
                total_keys = sum(1 for _ in self.client.scan_iter(f"{self.key_prefix}*"))
                estimated_size_mb = (avg_size * total_keys) / 1024 / 1024
                return estimated_size_mb
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Redis size calculation error: {e}")
            return 0.0


class DiskCacheBackend(CacheBackend):
    """Local disk cache backend"""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to temp directory
            import tempfile
            self.cache_dir = Path(tempfile.mkdtemp(prefix="ocr_cache_"))
            logger.warning(f"Permission denied for {cache_dir}, using temp directory: {self.cache_dir}")
        self.metadata_file = self.cache_dir / ".cache_metadata.json"
        self.metadata = self._load_metadata()
        self.lock = threading.Lock()
        logger.info(f"Initialized disk cache at {cache_dir} with {max_size_gb}GB limit")
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Use first 2 chars for directory sharding
        shard = key[:2]
        shard_dir = self.cache_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            try:
                # Check metadata
                if key not in self.metadata:
                    return None
                
                # Check expiration
                meta = self.metadata[key]
                if meta.get('expires') and time.time() > meta['expires']:
                    self.delete(key)
                    return None
                
                # Read file
                file_path = self._get_file_path(key)
                if file_path.exists():
                    return file_path.read_bytes()
                else:
                    # File missing, clean up metadata
                    del self.metadata[key]
                    self._save_metadata()
                    return None
                    
            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
                return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                value_size = len(value)
                
                # Check size limit
                if value_size > self.max_size_bytes:
                    logger.warning(f"Value too large for disk cache: {value_size / 1024 / 1024:.2f}MB")
                    return False
                
                # Evict entries if necessary
                current_size = self._get_total_size()
                while (current_size + value_size) > self.max_size_bytes and self.metadata:
                    self._evict_oldest()
                    current_size = self._get_total_size()
                
                # Write file
                file_path = self._get_file_path(key)
                file_path.write_bytes(value)
                
                # Update metadata
                self.metadata[key] = {
                    'size': value_size,
                    'created': time.time(),
                    'expires': time.time() + ttl if ttl else None
                }
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Disk cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            try:
                if key in self.metadata:
                    # Delete file
                    file_path = self._get_file_path(key)
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Update metadata
                    del self.metadata[key]
                    self._save_metadata()
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Disk cache delete error: {e}")
                return False
    
    def clear(self) -> int:
        with self.lock:
            count = len(self.metadata)
            
            # Delete all cache files
            for key in list(self.metadata.keys()):
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            return count
    
    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.metadata:
                # Check expiration
                meta = self.metadata[key]
                if meta.get('expires') and time.time() > meta['expires']:
                    self.delete(key)
                    return False
                
                # Check file exists
                return self._get_file_path(key).exists()
            return False
    
    def get_size(self) -> float:
        """Get total cache size in MB"""
        with self.lock:
            return self._get_total_size() / 1024 / 1024
    
    def _get_total_size(self) -> int:
        """Get total size in bytes"""
        return sum(meta['size'] for meta in self.metadata.values())
    
    def _evict_oldest(self):
        """Evict oldest entry"""
        if not self.metadata:
            return
        
        # Find oldest entry
        oldest_key = min(self.metadata.keys(), 
                        key=lambda k: self.metadata[k]['created'])
        self.delete(oldest_key)


class S3CacheBackend(CacheBackend):
    """AWS S3 cache backend for distributed persistent storage"""
    
    def __init__(self, bucket: str, prefix: str = "ocr-cache/", 
                 region: Optional[str] = None):
        if not S3_AVAILABLE:
            raise ImportError("boto3 not available. Install boto3 package.")
        
        self.bucket = bucket
        self.prefix = prefix
        self.region = region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.region
        )
        
        # Verify bucket access
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"Connected to S3 bucket: {self.bucket}")
        except ClientError as e:
            logger.error(f"Failed to access S3 bucket {self.bucket}: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Create S3 object key"""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[bytes]:
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=self._make_key(key)
            )
            
            # Check expiration
            metadata = response.get('Metadata', {})
            expires = metadata.get('expires')
            if expires and float(expires) < time.time():
                self.delete(key)
                return None
            
            return response['Body'].read()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.error(f"S3 get error: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        try:
            metadata = {}
            if ttl:
                metadata['expires'] = str(time.time() + ttl)
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=self._make_key(key),
                Body=value,
                Metadata=metadata,
                StorageClass='STANDARD_IA'  # Infrequent access for cost savings
            )
            return True
            
        except ClientError as e:
            logger.error(f"S3 set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=self._make_key(key)
            )
            return True
            
        except ClientError as e:
            logger.error(f"S3 delete error: {e}")
            return False
    
    def clear(self) -> int:
        try:
            # List and delete all objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            count = 0
            objects_to_delete = []
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects_to_delete.append({'Key': obj['Key']})
                        count += 1
                        
                        # Delete in batches of 1000
                        if len(objects_to_delete) >= 1000:
                            self.s3_client.delete_objects(
                                Bucket=self.bucket,
                                Delete={'Objects': objects_to_delete}
                            )
                            objects_to_delete = []
            
            # Delete remaining objects
            if objects_to_delete:
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={'Objects': objects_to_delete}
                )
            
            return count
            
        except ClientError as e:
            logger.error(f"S3 clear error: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket,
                Key=self._make_key(key)
            )
            
            # Check expiration
            metadata = response.get('Metadata', {})
            expires = metadata.get('expires')
            if expires and float(expires) < time.time():
                self.delete(key)
                return False
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"S3 exists error: {e}")
            return False
    
    def get_size(self) -> float:
        """Get approximate total size in MB"""
        try:
            # Use CloudWatch metrics for accurate size
            # For now, estimate based on object count
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            total_size = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj.get('Size', 0)
            
            return total_size / 1024 / 1024
            
        except Exception as e:
            logger.error(f"S3 size calculation error: {e}")
            return 0.0


class CacheManager:
    """
    Multi-tier cache manager for distributed OCR processing.
    
    Supports both local GPU server with Celery workers and AWS Batch environments.
    Implements memory � Redis � S3/disk cache hierarchy.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager with environment detection.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.stats = CacheStats()
        
        # Detect environment
        self.is_aws_batch = os.environ.get('AWS_BATCH_JOB_ID') is not None
        self.is_celery_worker = os.environ.get('CELERY_WORKER_ID') is not None
        self.worker_id = os.environ.get('CELERY_WORKER_ID', 
                                       os.environ.get('AWS_BATCH_JOB_ID', 'main'))
        
        # Configure cache sizes based on environment
        if self.is_celery_worker or self.is_aws_batch:
            # Smaller memory cache for workers
            memory_cache_mb = 128
        else:
            # Larger memory cache for main process
            memory_cache_mb = config.memory_cache_size_mb or 512
        
        # Initialize cache tiers
        self.memory_cache = None
        self.redis_cache = None
        self.persistent_cache = None
        
        # Always initialize memory cache
        self.memory_cache = MemoryCacheBackend(max_size_mb=memory_cache_mb)
        
        # Initialize Redis if available (primary shared cache)
        redis_url = os.environ.get('REDIS_URL', 
                                  f"redis://{config.redis_host}:{config.redis_port}/{config.redis_db}")
        if config.redis_password:
            redis_url = redis_url.replace('redis://', f'redis://:{config.redis_password}@')
        
        try:
            self.redis_cache = RedisCacheBackend(
                redis_url=redis_url,
                key_prefix=f"ocr_cache:{self.worker_id}:"
            )
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis cache not available: {e}")
            self.redis_cache = None
        
        # Initialize persistent cache based on environment
        if self.is_aws_batch:
            # Use S3 for AWS Batch
            s3_bucket = os.environ.get('S3_CACHE_BUCKET', config.s3_cache_bucket)
            if s3_bucket and S3_AVAILABLE:
                try:
                    self.persistent_cache = S3CacheBackend(
                        bucket=s3_bucket,
                        prefix=f"ocr-cache/{self.worker_id}/"
                    )
                    logger.info(f"S3 cache initialized: {s3_bucket}")
                except Exception as e:
                    logger.warning(f"S3 cache not available: {e}")
                    self.persistent_cache = None
            else:
                logger.warning("S3 bucket not configured for AWS Batch")
        else:
            # Use local disk for non-AWS environments
            cache_dir = Path(config.disk_cache_path or "./cache")
            self.persistent_cache = DiskCacheBackend(
                cache_dir=cache_dir / self.worker_id,
                max_size_gb=config.disk_cache_size_gb or 10.0
            )
            logger.info(f"Disk cache initialized: {cache_dir}")
        
        self.default_ttl = config.cache_ttl
        
        logger.info(f"Cache manager initialized - Environment: "
                   f"{'AWS Batch' if self.is_aws_batch else 'Local'}, "
                   f"Worker: {self.worker_id}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache, checking all tiers.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        self.stats.total_requests += 1
        
        # Check memory cache
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                self.stats.memory_hits += 1
                return self._deserialize(value)
            self.stats.memory_misses += 1
        
        # Check Redis cache
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                self.stats.redis_hits += 1
                # Promote to memory cache
                if self.memory_cache:
                    self.memory_cache.set(key, value, ttl=self.default_ttl)
                return self._deserialize(value)
            self.stats.redis_misses += 1
        
        # Check persistent cache
        if self.persistent_cache:
            value = self.persistent_cache.get(key)
            if value is not None:
                self.stats.persistent_hits += 1
                # Promote to faster tiers
                if self.redis_cache:
                    self.redis_cache.set(key, value, ttl=self.default_ttl)
                if self.memory_cache:
                    self.memory_cache.set(key, value, ttl=self.default_ttl)
                return self._deserialize(value)
            self.stats.persistent_misses += 1
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache across all available tiers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        ttl = ttl or self.default_ttl
        
        try:
            # Serialize value
            serialized = self._serialize(value)
            
            # Set in all available tiers
            success = False
            
            # Memory cache (always try)
            if self.memory_cache:
                if self.memory_cache.set(key, serialized, ttl=ttl):
                    success = True
            
            # Redis cache (primary persistent)
            if self.redis_cache:
                if self.redis_cache.set(key, serialized, ttl=ttl):
                    success = True
            
            # Persistent cache (backup)
            if self.persistent_cache:
                # Use longer TTL for persistent storage
                persistent_ttl = ttl * 2 if ttl else None
                if self.persistent_cache.set(key, serialized, ttl=persistent_ttl):
                    success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from all cache tiers.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        success = False
        
        if self.memory_cache and self.memory_cache.delete(key):
            success = True
        
        if self.redis_cache and self.redis_cache.delete(key):
            success = True
        
        if self.persistent_cache and self.persistent_cache.delete(key):
            success = True
        
        return success
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Total number of entries cleared
        """
        total = 0
        
        if self.memory_cache:
            total += self.memory_cache.clear()
        
        if self.redis_cache:
            total += self.redis_cache.clear()
        
        if self.persistent_cache:
            total += self.persistent_cache.clear()
        
        # Reset stats
        self.stats = CacheStats()
        
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        stats_dict = asdict(self.stats)
        
        # Add tier-specific info
        stats_dict['tiers'] = {
            'memory': {
                'enabled': self.memory_cache is not None,
                'size_mb': self.memory_cache.get_size() if self.memory_cache else 0
            },
            'redis': {
                'enabled': self.redis_cache is not None,
                'size_mb': self.redis_cache.get_size() if self.redis_cache else 0
            },
            'persistent': {
                'enabled': self.persistent_cache is not None,
                'type': 'S3' if self.is_aws_batch else 'disk',
                'size_mb': self.persistent_cache.get_size() if self.persistent_cache else 0
            }
        }
        
        # Add environment info
        stats_dict['environment'] = {
            'is_aws_batch': self.is_aws_batch,
            'is_celery_worker': self.is_celery_worker,
            'worker_id': self.worker_id
        }
        
        # Calculate total size
        total_size = sum(tier['size_mb'] for tier in stats_dict['tiers'].values())
        self.stats.total_size_mb = total_size
        stats_dict['total_size_mb'] = total_size
        
        # Add hit rates
        stats_dict['hit_rates'] = self.stats.tier_hit_rates
        stats_dict['overall_hit_rate'] = self.stats.overall_hit_rate
        
        return stats_dict
    
    def initialize(self):
        """Initialize cache manager (for compatibility)"""
        # Already initialized in __init__
        pass
    
    def cleanup(self):
        """Clean up cache resources"""
        # Redis and S3 connections are managed by the backends
        pass
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)
    
    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)
        
        # Generate hash
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Key pattern (supports * wildcard)
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Note: This is a simplified implementation
        # In production, you'd want more sophisticated pattern matching
        
        if self.redis_cache and hasattr(self.redis_cache.client, 'scan_iter'):
            # Redis supports pattern matching
            for key in self.redis_cache.client.scan_iter(
                match=f"{self.redis_cache.key_prefix}{pattern}"
            ):
                if self.redis_cache.delete(key.decode()):
                    count += 1
        
        return count