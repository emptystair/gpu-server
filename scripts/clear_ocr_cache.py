#!/usr/bin/env python3
"""
Clear OCR Cache - Comprehensive cache clearing utility

This script clears all OCR cache tiers:
- Memory cache (by restarting the service)
- Redis cache (both host and container)
- Disk cache (in container and local)

Usage:
    python clear_ocr_cache.py [options]
    
Options:
    --all          Clear all cache tiers (default)
    --redis        Clear only Redis cache
    --disk         Clear only disk cache
    --memory       Clear only memory cache (restart service)
    --stats        Show cache statistics before clearing
    --no-restart   Don't restart the service
"""

import sys
import os
import time
import argparse
import subprocess
import shutil
from pathlib import Path
import json

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis module not available. Install with: pip install redis")


class CacheCleaner:
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        self.container_name = 'gpu-ocr-server'
        self.stats = {
            'redis_keys_deleted': 0,
            'disk_files_deleted': 0,
            'disk_space_freed_mb': 0,
            'errors': []
        }
    
    def get_redis_stats(self):
        """Get Redis cache statistics"""
        if not REDIS_AVAILABLE:
            return None
        
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db)
            
            # Count OCR cache keys
            patterns = ["ocr_cache:*", "cache:*", "*ocr*"]
            total_keys = 0
            total_memory = 0
            
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = r.scan(cursor, match=pattern, count=100)
                    total_keys += len(keys)
                    
                    # Sample memory usage
                    for key in keys[:10]:  # Sample first 10
                        try:
                            memory = r.memory_usage(key) or 0
                            total_memory += memory
                        except:
                            pass
                    
                    if cursor == 0:
                        break
            
            # Get overall Redis info
            info = r.info('memory')
            
            return {
                'ocr_cache_keys': total_keys,
                'estimated_ocr_memory_mb': total_memory / 1024 / 1024,
                'total_redis_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                'redis_connected': True
            }
            
        except Exception as e:
            return {
                'redis_connected': False,
                'error': str(e)
            }
    
    def clear_redis_cache(self, from_container=False):
        """Clear Redis cache"""
        print("\n" + "="*50)
        print("CLEARING REDIS CACHE")
        print("="*50)
        
        if from_container:
            # Clear from within Docker container
            try:
                # Use docker exec to run Redis commands
                cmd = [
                    'docker', 'exec', self.container_name,
                    'python3', '-c',
                    '''
import redis
r = redis.Redis(host='ocr-redis', port=6379, db=0)
keys_deleted = 0
patterns = ["ocr_cache:*", "cache:*", "*ocr*"]
for pattern in patterns:
    for key in r.scan_iter(match=pattern):
        r.delete(key)
        keys_deleted += 1
print(f"Deleted {keys_deleted} keys from container Redis")
'''
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Container Redis: {result.stdout.strip()}")
                else:
                    print(f"Container Redis error: {result.stderr}")
                    self.stats['errors'].append(f"Container Redis: {result.stderr}")
                    
            except Exception as e:
                print(f"Error clearing container Redis: {e}")
                self.stats['errors'].append(f"Container Redis: {str(e)}")
        
        # Clear from host
        if REDIS_AVAILABLE:
            try:
                r = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db)
                
                patterns = ["ocr_cache:*", "cache:*", "*ocr*"]
                keys_deleted = 0
                
                for pattern in patterns:
                    cursor = 0
                    while True:
                        cursor, keys = r.scan(cursor, match=pattern, count=1000)
                        if keys:
                            keys_deleted += r.delete(*keys)
                        if cursor == 0:
                            break
                
                self.stats['redis_keys_deleted'] = keys_deleted
                print(f"Host Redis: Deleted {keys_deleted} cache keys")
                
                # Optional: Clear entire database (commented out for safety)
                # if input("\nClear entire Redis database? (y/N): ").lower() == 'y':
                #     r.flushdb()
                #     print("Entire Redis database cleared")
                    
            except Exception as e:
                print(f"Error clearing host Redis: {e}")
                self.stats['errors'].append(f"Host Redis: {str(e)}")
        else:
            print("Redis module not available on host")
    
    def clear_disk_cache(self):
        """Clear disk cache in container and locally"""
        print("\n" + "="*50)
        print("CLEARING DISK CACHE")
        print("="*50)
        
        # Clear container disk cache
        try:
            # Check disk usage before
            cmd_before = ['docker', 'exec', self.container_name, 'du', '-sh', '/app/cache']
            result_before = subprocess.run(cmd_before, capture_output=True, text=True)
            size_before = result_before.stdout.strip() if result_before.returncode == 0 else "Unknown"
            
            # Clear cache directory
            cmd_clear = ['docker', 'exec', self.container_name, 'rm', '-rf', '/app/cache/*']
            result = subprocess.run(cmd_clear, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Container disk cache: Cleared /app/cache (was {size_before})")
                
                # Also clear cache metadata
                cmd_meta = ['docker', 'exec', self.container_name, 'rm', '-f', '/app/cache/.cache_metadata.json']
                subprocess.run(cmd_meta, capture_output=True, text=True)
            else:
                print(f"Container disk cache error: {result.stderr}")
                self.stats['errors'].append(f"Container disk: {result.stderr}")
                
        except Exception as e:
            print(f"Error clearing container disk cache: {e}")
            self.stats['errors'].append(f"Container disk: {str(e)}")
        
        # Clear local disk cache
        local_cache_paths = [
            Path("./cache"),
            Path("/tmp/ocr_cache_*"),
        ]
        
        for cache_path in local_cache_paths:
            if cache_path.exists() and cache_path.is_dir():
                try:
                    # Calculate size before deletion
                    size_before = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                    size_mb = size_before / 1024 / 1024
                    
                    # Count files
                    file_count = sum(1 for f in cache_path.rglob('*') if f.is_file())
                    
                    shutil.rmtree(cache_path)
                    cache_path.mkdir(exist_ok=True)
                    
                    self.stats['disk_files_deleted'] += file_count
                    self.stats['disk_space_freed_mb'] += size_mb
                    
                    print(f"Local disk cache: Cleared {cache_path} ({file_count} files, {size_mb:.1f} MB)")
                except Exception as e:
                    print(f"Error clearing local disk cache at {cache_path}: {e}")
                    self.stats['errors'].append(f"Local disk {cache_path}: {str(e)}")
    
    def clear_memory_cache(self, restart_service=True):
        """Clear memory cache by restarting the service"""
        print("\n" + "="*50)
        print("CLEARING MEMORY CACHE")
        print("="*50)
        
        if not restart_service:
            print("Skipping service restart (--no-restart flag)")
            return
        
        try:
            print("Restarting OCR service to clear memory cache...")
            
            # Stop container
            cmd_stop = ['docker', 'compose', 'stop', 'ocr-server']
            result = subprocess.run(cmd_stop, capture_output=True, text=True)
            if result.returncode != 0:
                # Try alternative service name
                cmd_stop = ['docker', 'compose', 'stop', 'gpu-ocr-server']
                result = subprocess.run(cmd_stop, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Service stopped")
                time.sleep(2)
                
                # Start container
                cmd_start = ['docker', 'compose', 'up', '-d']
                result = subprocess.run(cmd_start, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("Service restarted successfully")
                    print("Waiting for service to be ready...")
                    time.sleep(10)
                    
                    # Check health
                    cmd_health = ['curl', '-s', 'http://localhost:8000/api/v1/health']
                    for i in range(30):  # Wait up to 30 seconds
                        result = subprocess.run(cmd_health, capture_output=True, text=True)
                        if result.returncode == 0 and 'healthy' in result.stdout:
                            print("Service is healthy and ready")
                            break
                        time.sleep(1)
                else:
                    print(f"Error starting service: {result.stderr}")
                    self.stats['errors'].append(f"Service restart: {result.stderr}")
            else:
                print(f"Error stopping service: {result.stderr}")
                self.stats['errors'].append(f"Service stop: {result.stderr}")
                
        except Exception as e:
            print(f"Error restarting service: {e}")
            self.stats['errors'].append(f"Service restart: {str(e)}")
    
    def show_stats(self):
        """Show cache statistics"""
        print("\n" + "="*50)
        print("CACHE STATISTICS")
        print("="*50)
        
        redis_stats = self.get_redis_stats()
        if redis_stats and redis_stats.get('redis_connected'):
            print(f"Redis:")
            print(f"  OCR cache keys: {redis_stats['ocr_cache_keys']}")
            print(f"  Estimated OCR memory: {redis_stats['estimated_ocr_memory_mb']:.1f} MB")
            print(f"  Total Redis memory: {redis_stats['total_redis_memory_mb']:.1f} MB")
        else:
            print("Redis: Not connected")
        
        # Try to get disk cache stats from container
        try:
            cmd = ['docker', 'exec', self.container_name, 'du', '-sh', '/app/cache']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"\nContainer disk cache: {result.stdout.strip()}")
        except:
            pass
    
    def print_summary(self):
        """Print clearing summary"""
        print("\n" + "="*50)
        print("CACHE CLEARING SUMMARY")
        print("="*50)
        
        print(f"Redis keys deleted: {self.stats['redis_keys_deleted']}")
        print(f"Disk files deleted: {self.stats['disk_files_deleted']}")
        print(f"Disk space freed: {self.stats['disk_space_freed_mb']:.1f} MB")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                print(f"  - {error}")
        else:
            print("\nAll caches cleared successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Clear OCR cache from all tiers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--all', action='store_true', default=True,
                        help='Clear all cache tiers (default)')
    parser.add_argument('--redis', action='store_true',
                        help='Clear only Redis cache')
    parser.add_argument('--disk', action='store_true',
                        help='Clear only disk cache')
    parser.add_argument('--memory', action='store_true',
                        help='Clear only memory cache (restart service)')
    parser.add_argument('--stats', action='store_true',
                        help='Show cache statistics before clearing')
    parser.add_argument('--no-restart', action='store_true',
                        help="Don't restart the service")
    
    args = parser.parse_args()
    
    # If specific caches are selected, don't clear all
    if args.redis or args.disk or args.memory:
        args.all = False
    
    cleaner = CacheCleaner()
    
    print("OCR Cache Cleaner")
    print("=" * 50)
    
    # Show stats if requested
    if args.stats:
        cleaner.show_stats()
    
    # Clear caches
    if args.all or args.redis:
        cleaner.clear_redis_cache(from_container=True)
    
    if args.all or args.disk:
        cleaner.clear_disk_cache()
    
    if args.all or args.memory:
        cleaner.clear_memory_cache(restart_service=not args.no_restart)
    
    # Print summary
    cleaner.print_summary()


if __name__ == "__main__":
    main()