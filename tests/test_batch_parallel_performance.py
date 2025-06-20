#!/usr/bin/env python3
"""
Test batch and parallel processing performance for GPU OCR server
"""

import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configuration
SERVER_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
TEST_PDFS_DIR = Path(__file__).parent / "testpdfs"
OUTPUT_DIR = Path(__file__).parent / "performance_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test configurations
BATCH_SIZES = [1, 5, 10, 20, 30, 50]
CONCURRENT_REQUESTS = [1, 2, 4, 8, 16]
DPI_SETTINGS = [120]  # Focus on single DPI for now
MAX_PDFS_PER_TEST = 50  # Limit for testing


async def process_single_pdf(session: aiohttp.ClientSession, pdf_path: Path, dpi: int = 120) -> Dict[str, Any]:
    """Process a single PDF file"""
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=pdf_path.name, content_type='application/pdf')
            data.add_field('dpi', str(dpi))
            
            async with session.post(f"{SERVER_URL}{API_PREFIX}/ocr/pdf", data=data) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "file": pdf_path.name,
                    "status": result.get("status"),
                    "processing_time": result.get("processing_time", 0),
                    "total_time": (end_time - start_time) * 1000,
                    "pages": result.get("pages", 0),
                    "confidence": result.get("average_confidence", 0)
                }
    except Exception as e:
        return {
            "file": pdf_path.name,
            "status": "error",
            "error": str(e),
            "total_time": (time.time() - start_time) * 1000
        }


async def process_batch_pdf(session: aiohttp.ClientSession, pdf_paths: List[Path], dpi: int = 120) -> Dict[str, Any]:
    """Process multiple PDFs in a batch request"""
    start_time = time.time()
    
    try:
        data = aiohttp.FormData()
        
        # Add all files to the batch
        for pdf_path in pdf_paths:
            with open(pdf_path, 'rb') as f:
                data.add_field('files', f, filename=pdf_path.name, content_type='application/pdf')
        
        data.add_field('dpi', str(dpi))
        data.add_field('parallel_processing', 'true')
        
        async with session.post(f"{SERVER_URL}{API_PREFIX}/ocr/batch", data=data) as response:
            result = await response.json()
            end_time = time.time()
            
            return {
                "batch_size": len(pdf_paths),
                "status": result.get("status"),
                "successful": result.get("successful", 0),
                "failed": result.get("failed", 0),
                "total_processing_time": result.get("total_processing_time", 0),
                "total_time": (end_time - start_time) * 1000,
                "results": result.get("results", [])
            }
    except Exception as e:
        return {
            "batch_size": len(pdf_paths),
            "status": "error",
            "error": str(e),
            "total_time": (time.time() - start_time) * 1000
        }


async def test_concurrent_single_requests(pdf_paths: List[Path], concurrent_count: int, dpi: int = 120) -> Dict[str, Any]:
    """Test concurrent single PDF requests"""
    print(f"\nTesting concurrent single requests (concurrency={concurrent_count})...")
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create tasks for concurrent processing
        tasks = []
        for i in range(0, len(pdf_paths), concurrent_count):
            batch = pdf_paths[i:i+concurrent_count]
            batch_tasks = [process_single_pdf(session, pdf, dpi) for pdf in batch]
            tasks.extend(batch_tasks)
        
        # Process with limited concurrency
        results = []
        for i in range(0, len(tasks), concurrent_count):
            batch_results = await asyncio.gather(*tasks[i:i+concurrent_count])
            results.extend(batch_results)
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        successful = [r for r in results if r.get("status") == "completed"]
        processing_times = [r["processing_time"] for r in successful if "processing_time" in r]
        total_times = [r["total_time"] for r in results if "total_time" in r]
        
        return {
            "test_type": "concurrent_single",
            "concurrency": concurrent_count,
            "total_pdfs": len(pdf_paths),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_test_time": total_time,
            "avg_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "median_processing_time": statistics.median(processing_times) if processing_times else 0,
            "avg_total_time": statistics.mean(total_times) if total_times else 0,
            "throughput_pdfs_per_second": len(successful) / (total_time / 1000) if total_time > 0 else 0,
            "results": results
        }


async def test_batch_processing(pdf_paths: List[Path], batch_size: int, dpi: int = 120) -> Dict[str, Any]:
    """Test batch processing"""
    print(f"\nTesting batch processing (batch_size={batch_size})...")
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        results = []
        
        # Process PDFs in batches
        for i in range(0, len(pdf_paths), batch_size):
            batch = pdf_paths[i:i+batch_size]
            result = await process_batch_pdf(session, batch, dpi)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        total_successful = sum(r.get("successful", 0) for r in results)
        total_failed = sum(r.get("failed", 0) for r in results)
        processing_times = [r["total_processing_time"] for r in results if "total_processing_time" in r]
        
        return {
            "test_type": "batch",
            "batch_size": batch_size,
            "total_pdfs": len(pdf_paths),
            "total_batches": len(results),
            "successful": total_successful,
            "failed": total_failed,
            "total_test_time": total_time,
            "avg_batch_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "throughput_pdfs_per_second": total_successful / (total_time / 1000) if total_time > 0 else 0,
            "batch_results": results
        }


async def run_comprehensive_tests():
    """Run comprehensive performance tests"""
    print("Starting comprehensive batch and parallel processing tests...")
    
    # Get test PDFs
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))[:MAX_PDFS_PER_TEST]
    print(f"Found {len(pdf_files)} PDF files for testing")
    
    if not pdf_files:
        print("No PDF files found in testpdfs directory!")
        return
    
    all_results = {
        "test_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": SERVER_URL,
            "total_pdfs": len(pdf_files),
            "cpu_count": multiprocessing.cpu_count()
        },
        "concurrent_single_tests": [],
        "batch_tests": [],
        "comparison": {}
    }
    
    # Test 1: Baseline - Sequential single requests
    print("\n=== Testing baseline (sequential single requests) ===")
    baseline = await test_concurrent_single_requests(pdf_files[:10], concurrent_count=1)
    all_results["baseline"] = baseline
    
    # Test 2: Concurrent single requests with different concurrency levels
    print("\n=== Testing concurrent single requests ===")
    for concurrent_count in CONCURRENT_REQUESTS:
        if concurrent_count == 1:
            continue  # Skip, already done as baseline
        result = await test_concurrent_single_requests(pdf_files, concurrent_count)
        all_results["concurrent_single_tests"].append(result)
        
        # Save intermediate results
        with open(OUTPUT_DIR / f"concurrent_{concurrent_count}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Test 3: Batch processing with different batch sizes
    print("\n=== Testing batch processing ===")
    for batch_size in BATCH_SIZES:
        result = await test_batch_processing(pdf_files, batch_size)
        all_results["batch_tests"].append(result)
        
        # Save intermediate results
        with open(OUTPUT_DIR / f"batch_{batch_size}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Generate comparison summary
    print("\n=== Generating comparison summary ===")
    
    # Find best configurations
    best_concurrent = max(all_results["concurrent_single_tests"], 
                         key=lambda x: x["throughput_pdfs_per_second"]) if all_results["concurrent_single_tests"] else None
    best_batch = max(all_results["batch_tests"], 
                    key=lambda x: x["throughput_pdfs_per_second"]) if all_results["batch_tests"] else None
    
    all_results["comparison"] = {
        "baseline_throughput": baseline["throughput_pdfs_per_second"],
        "best_concurrent": {
            "concurrency": best_concurrent["concurrency"] if best_concurrent else None,
            "throughput": best_concurrent["throughput_pdfs_per_second"] if best_concurrent else 0,
            "improvement": (best_concurrent["throughput_pdfs_per_second"] / baseline["throughput_pdfs_per_second"] - 1) * 100 if best_concurrent and baseline["throughput_pdfs_per_second"] > 0 else 0
        },
        "best_batch": {
            "batch_size": best_batch["batch_size"] if best_batch else None,
            "throughput": best_batch["throughput_pdfs_per_second"] if best_batch else 0,
            "improvement": (best_batch["throughput_pdfs_per_second"] / baseline["throughput_pdfs_per_second"] - 1) * 100 if best_batch and baseline["throughput_pdfs_per_second"] > 0 else 0
        }
    }
    
    # Save complete results
    with open(OUTPUT_DIR / "complete_performance_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    print(f"Baseline throughput: {baseline['throughput_pdfs_per_second']:.2f} PDFs/second")
    
    if best_concurrent:
        print(f"\nBest concurrent configuration:")
        print(f"  Concurrency: {best_concurrent['concurrency']}")
        print(f"  Throughput: {best_concurrent['throughput_pdfs_per_second']:.2f} PDFs/second")
        print(f"  Improvement: {all_results['comparison']['best_concurrent']['improvement']:.1f}%")
    
    if best_batch:
        print(f"\nBest batch configuration:")
        print(f"  Batch size: {best_batch['batch_size']}")
        print(f"  Throughput: {best_batch['throughput_pdfs_per_second']:.2f} PDFs/second")
        print(f"  Improvement: {all_results['comparison']['best_batch']['improvement']:.1f}%")
    
    print("\nDetailed results saved to:", OUTPUT_DIR)


async def test_mixed_workload():
    """Test mixed workload with both batch and concurrent requests"""
    print("\n=== Testing mixed workload (batch + concurrent) ===")
    
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))[:MAX_PDFS_PER_TEST]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create mixed tasks
        tasks = []
        
        # Add some batch requests
        for i in range(0, 20, 10):
            batch = pdf_files[i:i+10]
            tasks.append(process_batch_pdf(session, batch))
        
        # Add some concurrent single requests
        for pdf in pdf_files[20:30]:
            tasks.append(process_single_pdf(session, pdf))
        
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"Mixed workload completed in {total_time:.2f}ms")
        
        return {
            "test_type": "mixed",
            "total_time": total_time,
            "results": results
        }


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_comprehensive_tests())
    
    # Optional: Run mixed workload test
    # asyncio.run(test_mixed_workload())