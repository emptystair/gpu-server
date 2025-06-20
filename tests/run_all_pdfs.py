#!/usr/bin/env python3
"""
Run all PDFs in testpdfs directory through OCR service and save results.
Clears cache before running to ensure fresh processing.
"""

import os
import sys
import json
import time
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import httpx
from tqdm import tqdm

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TESTPDFS_DIR = Path(__file__).parent / "testpdfs"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)


async def clear_cache():
    """Clear all cache tiers via Redis"""
    print("Clearing cache...")
    try:
        import redis
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Clear all OCR cache keys
        keys_deleted = 0
        for key in r.scan_iter(match="ocr_cache:*"):
            r.delete(key)
            keys_deleted += 1
        
        print(f"Deleted {keys_deleted} Redis cache entries")
        
        # Also clear disk cache if accessible
        disk_cache_path = Path("/app/cache")
        if disk_cache_path.exists():
            import shutil
            shutil.rmtree(disk_cache_path)
            disk_cache_path.mkdir()
            print("Cleared disk cache")
            
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")
        print("Continuing anyway...")


async def get_server_status(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Check server health and GPU status"""
    try:
        # Health check
        health_resp = await client.get(f"{API_BASE_URL}/health")
        health_resp.raise_for_status()
        
        # GPU status
        gpu_resp = await client.get(f"{API_BASE_URL}/gpu/status")
        gpu_resp.raise_for_status()
        
        return {
            "health": health_resp.json(),
            "gpu": gpu_resp.json()
        }
    except Exception as e:
        print(f"Error checking server status: {e}")
        return None


async def process_pdf(client: httpx.AsyncClient, pdf_path: Path) -> Dict[str, Any]:
    """Process a single PDF through OCR"""
    start_time = time.time()
    
    try:
        # Read file
        with open(pdf_path, 'rb') as f:
            file_content = f.read()
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Prepare multipart form data
        files = {
            'file': (pdf_path.name, file_content, 'application/pdf')
        }
        data = {
            'language': 'en',
            'output_format': 'json',
            'confidence_threshold': '0.5',
            'merge_pages': 'true'
        }
        
        # Send request with longer timeout for large PDFs
        response = await client.post(
            f"{API_BASE_URL}/ocr/pdf",
            files=files,
            data=data,
            timeout=300.0  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "filename": pdf_path.name,
                "file_size_mb": len(file_content) / (1024 * 1024),
                "file_hash": file_hash,
                "processing_time": processing_time,
                "ocr_time": result.get("processing_time", 0) / 1000.0,  # Convert ms to seconds
                "pages": len(result.get("pages", [])),
                "regions": result.get("total_regions", 0),
                "confidence": result.get("average_confidence", 0),
                "cache_hit": result.get("cache_hit", False),
                "text_length": len(result.get("pages", [{}])[0].get("text", "")) if result.get("pages") else 0,
                "result": result
            }
        else:
            return {
                "status": "error",
                "filename": pdf_path.name,
                "file_size_mb": len(file_content) / (1024 * 1024),
                "file_hash": file_hash,
                "processing_time": time.time() - start_time,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "filename": pdf_path.name,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }


async def run_batch_processing(pdf_files: List[Path], concurrent_limit: int = 2):
    """Process PDFs with controlled concurrency"""
    results = []
    
    async with httpx.AsyncClient() as client:
        # Check server status first
        print("Checking server status...")
        status = await get_server_status(client)
        if status:
            print(f"Server healthy: {status['health']['status']}")
            print(f"GPU: {status['gpu']['device_name']} - {status['gpu']['memory']['free_mb']}MB free")
        else:
            print("Warning: Could not get server status")
        
        print(f"\nProcessing {len(pdf_files)} PDFs with concurrency limit of {concurrent_limit}...")
        
        # Process in batches
        for i in range(0, len(pdf_files), concurrent_limit):
            batch = pdf_files[i:i + concurrent_limit]
            batch_tasks = [process_pdf(client, pdf) for pdf in batch]
            
            # Process batch
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Progress update
            completed = min(i + concurrent_limit, len(pdf_files))
            print(f"Progress: {completed}/{len(pdf_files)} PDFs processed")
            
            # Show latest result summary
            for result in batch_results:
                if result['status'] == 'success':
                    print(f"  ✓ {result['filename']}: {result['pages']} pages, "
                          f"{result['ocr_time']:.2f}s OCR time, "
                          f"{result['confidence']:.1%} confidence"
                          f"{' (cached)' if result['cache_hit'] else ''}")
                else:
                    print(f"  ✗ {result['filename']}: {result['error']}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze processing results"""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    if not successful:
        return {"error": "No successful results to analyze"}
    
    # Calculate statistics
    total_pages = sum(r['pages'] for r in successful)
    total_regions = sum(r['regions'] for r in successful)
    ocr_times = [r['ocr_time'] for r in successful if not r['cache_hit']]
    file_sizes = [r['file_size_mb'] for r in successful]
    confidences = [r['confidence'] for r in successful]
    
    analysis = {
        "summary": {
            "total_pdfs": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "cache_hits": sum(1 for r in successful if r['cache_hit']),
            "total_pages": total_pages,
            "total_regions": total_regions,
            "total_file_size_mb": sum(file_sizes),
        },
        "performance": {
            "avg_ocr_time_per_pdf": sum(ocr_times) / len(ocr_times) if ocr_times else 0,
            "avg_ocr_time_per_page": sum(ocr_times) / total_pages if total_pages > 0 else 0,
            "total_processing_time": sum(r['processing_time'] for r in results),
            "pages_per_second": total_pages / sum(ocr_times) if ocr_times else 0,
        },
        "quality": {
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        },
        "file_stats": {
            "avg_file_size_mb": sum(file_sizes) / len(file_sizes),
            "largest_file_mb": max(file_sizes),
            "smallest_file_mb": min(file_sizes),
        }
    }
    
    # Add per-file details
    analysis["files"] = [
        {
            "filename": r['filename'],
            "pages": r['pages'],
            "regions": r['regions'],
            "confidence": r['confidence'],
            "ocr_time": r['ocr_time'],
            "cache_hit": r['cache_hit'],
            "file_size_mb": r['file_size_mb']
        }
        for r in successful
    ]
    
    if failed:
        analysis["errors"] = [
            {
                "filename": r['filename'],
                "error": r['error']
            }
            for r in failed
        ]
    
    return analysis


async def main():
    """Main execution function"""
    print("GPU OCR Server - Batch PDF Processing")
    print("=" * 50)
    
    # Get list of PDFs
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {TESTPDFS_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {TESTPDFS_DIR}")
    
    # Clear cache
    await clear_cache()
    
    # Process PDFs
    start_time = time.time()
    results = await run_batch_processing(pdf_files, concurrent_limit=2)
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results)
    
    # Save detailed results
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "processing_time": total_time,
            "testpdfs_dir": str(TESTPDFS_DIR),
            "api_url": API_BASE_URL
        },
        "analysis": analysis,
        "detailed_results": results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total PDFs processed: {analysis['summary']['total_pdfs']}")
    print(f"Successful: {analysis['summary']['successful']}")
    print(f"Failed: {analysis['summary']['failed']}")
    print(f"Cache hits: {analysis['summary']['cache_hits']}")
    print(f"Total pages: {analysis['summary']['total_pages']}")
    print(f"Total regions: {analysis['summary']['total_regions']}")
    print(f"\nPerformance:")
    print(f"  Average OCR time per PDF: {analysis['performance']['avg_ocr_time_per_pdf']:.2f}s")
    print(f"  Average OCR time per page: {analysis['performance']['avg_ocr_time_per_page']:.3f}s")
    print(f"  Pages per second: {analysis['performance']['pages_per_second']:.1f}")
    print(f"\nQuality:")
    print(f"  Average confidence: {analysis['quality']['avg_confidence']:.1%}")
    print(f"  Confidence range: {analysis['quality']['min_confidence']:.1%} - {analysis['quality']['max_confidence']:.1%}")
    
    if 'errors' in analysis:
        print(f"\nErrors encountered:")
        for error in analysis['errors']:
            print(f"  - {error['filename']}: {error['error']}")


if __name__ == "__main__":
    asyncio.run(main())