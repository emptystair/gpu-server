#!/usr/bin/env python3
"""
Clear all OCR cache from Redis and disk
"""

import redis
import shutil
from pathlib import Path

def clear_redis_cache():
    """Clear Redis cache"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        # Get all OCR cache keys
        keys_deleted = 0
        
        # Clear keys with different patterns
        patterns = [
            "ocr_cache:*",
            "cache:*", 
            "*ocr*",
        ]
        
        for pattern in patterns:
            for key in r.scan_iter(match=pattern):
                r.delete(key)
                keys_deleted += 1
        
        # Also flush the entire database if needed (careful!)
        # r.flushdb()
        
        print(f"Deleted {keys_deleted} Redis cache entries")
        
    except Exception as e:
        print(f"Error clearing Redis cache: {e}")


def clear_disk_cache():
    """Clear disk cache"""
    cache_paths = [
        Path("./cache"),
        Path("/app/cache"),
        Path("/tmp/ocr_cache_*"),
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists() and cache_path.is_dir():
            try:
                shutil.rmtree(cache_path)
                cache_path.mkdir(exist_ok=True)
                print(f"Cleared disk cache at {cache_path}")
            except Exception as e:
                print(f"Error clearing disk cache at {cache_path}: {e}")


def main():
    print("Clearing all OCR caches...")
    clear_redis_cache()
    clear_disk_cache()
    print("Cache clearing complete!")


if __name__ == "__main__":
    main()