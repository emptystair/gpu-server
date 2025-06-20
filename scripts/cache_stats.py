#!/usr/bin/env python3
"""
Display OCR cache statistics across all tiers
"""

import os
import subprocess
import json
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis module not available")


def get_redis_stats():
    """Get Redis cache statistics"""
    stats = {
        'host': {'connected': False, 'keys': 0, 'memory_mb': 0},
        'container': {'connected': False, 'keys': 0, 'memory_mb': 0}
    }
    
    # Host Redis stats
    if REDIS_AVAILABLE:
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            
            # Count OCR keys
            ocr_keys = 0
            for key in r.scan_iter(match='*ocr*'):
                ocr_keys += 1
            
            info = r.info('memory')
            stats['host'] = {
                'connected': True,
                'keys': ocr_keys,
                'memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                'total_keys': r.dbsize()
            }
        except:
            pass
    
    # Container Redis stats
    try:
        cmd = ['docker', 'exec', 'gpu-ocr-server', 'python3', '-c', '''
import redis
import json
r = redis.Redis(host="ocr-redis", port=6379, db=0)
ocr_keys = sum(1 for _ in r.scan_iter(match="*ocr*"))
info = r.info("memory")
print(json.dumps({
    "keys": ocr_keys,
    "memory_mb": info.get("used_memory", 0) / 1024 / 1024,
    "total_keys": r.dbsize()
}))
''']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            container_stats = json.loads(result.stdout)
            container_stats['connected'] = True
            stats['container'] = container_stats
    except:
        pass
    
    return stats


def get_disk_cache_stats():
    """Get disk cache statistics"""
    stats = {
        'container': {'path': '/app/cache', 'size_mb': 0, 'files': 0},
        'local': []
    }
    
    # Container disk cache
    try:
        # Get size
        cmd_size = ['docker', 'exec', 'gpu-ocr-server', 'du', '-sm', '/app/cache']
        result = subprocess.run(cmd_size, capture_output=True, text=True)
        if result.returncode == 0:
            size_mb = int(result.stdout.split()[0])
            stats['container']['size_mb'] = size_mb
        
        # Get file count
        cmd_count = ['docker', 'exec', 'gpu-ocr-server', 'find', '/app/cache', '-type', 'f', '-name', '*.cache']
        result = subprocess.run(cmd_count, capture_output=True, text=True)
        if result.returncode == 0:
            stats['container']['files'] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        pass
    
    # Local disk caches
    local_paths = [Path('./cache'), Path('/tmp').glob('ocr_cache_*')]
    for path in local_paths:
        if isinstance(path, Path) and path.exists():
            paths = [path]
        else:
            paths = list(path)
        
        for p in paths:
            if p.exists() and p.is_dir():
                files = list(p.rglob('*.cache'))
                size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
                stats['local'].append({
                    'path': str(p),
                    'files': len(files),
                    'size_mb': size_mb
                })
    
    return stats


def get_api_cache_stats():
    """Get cache stats from API endpoint"""
    try:
        import requests
        response = requests.get('http://localhost:8000/api/v1/stats', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def main():
    print("OCR Cache Statistics")
    print("=" * 60)
    
    # Redis stats
    redis_stats = get_redis_stats()
    print("\nREDIS CACHE:")
    print("-" * 40)
    
    if redis_stats['host']['connected']:
        print(f"Host Redis:")
        print(f"  OCR Keys: {redis_stats['host']['keys']}")
        print(f"  Total Keys: {redis_stats['host']['total_keys']}")
        print(f"  Memory Used: {redis_stats['host']['memory_mb']:.1f} MB")
    else:
        print("Host Redis: Not connected")
    
    if redis_stats['container']['connected']:
        print(f"\nContainer Redis:")
        print(f"  OCR Keys: {redis_stats['container']['keys']}")
        print(f"  Total Keys: {redis_stats['container']['total_keys']}")
        print(f"  Memory Used: {redis_stats['container']['memory_mb']:.1f} MB")
    else:
        print("\nContainer Redis: Not connected")
    
    # Disk cache stats
    disk_stats = get_disk_cache_stats()
    print("\n\nDISK CACHE:")
    print("-" * 40)
    
    print(f"Container ({disk_stats['container']['path']}):")
    print(f"  Files: {disk_stats['container']['files']}")
    print(f"  Size: {disk_stats['container']['size_mb']:.1f} MB")
    
    if disk_stats['local']:
        print("\nLocal:")
        for cache in disk_stats['local']:
            print(f"  {cache['path']}:")
            print(f"    Files: {cache['files']}")
            print(f"    Size: {cache['size_mb']:.1f} MB")
    
    # API stats
    api_stats = get_api_cache_stats()
    if api_stats:
        print("\n\nAPI STATISTICS:")
        print("-" * 40)
        if 'cache_stats' in api_stats:
            cache = api_stats['cache_stats']
            print(f"  Hit Rate: {cache.get('hit_rate', 0):.1%}")
            print(f"  Total Requests: {cache.get('total_requests', 0)}")
    
    # Summary
    total_redis_keys = redis_stats['host']['keys'] + redis_stats['container']['keys']
    total_disk_files = disk_stats['container']['files'] + sum(c['files'] for c in disk_stats['local'])
    total_disk_mb = disk_stats['container']['size_mb'] + sum(c['size_mb'] for c in disk_stats['local'])
    
    print("\n\nSUMMARY:")
    print("-" * 40)
    print(f"Total Redis OCR Keys: {total_redis_keys}")
    print(f"Total Disk Cache Files: {total_disk_files}")
    print(f"Total Disk Cache Size: {total_disk_mb:.1f} MB")


if __name__ == "__main__":
    main()