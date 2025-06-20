#!/usr/bin/env python3
"""
Test optimized batch processing with higher concurrency and better batching
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
import httpx
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TESTPDFS_DIR = Path(__file__).parent / "testpdfs"
RESULTS_DIR = Path(__file__).parent / "results"

async def test_batch_endpoint(pdf_files: List[Path], batch_size: int = 10):
    """Test the batch processing endpoint"""
    print(f"\nTesting batch endpoint with {batch_size} PDFs per request...")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        start_time = time.time()
        
        # Prepare batch request
        files = []
        for pdf_path in pdf_files[:batch_size]:
            with open(pdf_path, 'rb') as f:
                content = f.read()
            files.append(('files', (pdf_path.name, content, 'application/pdf')))
        
        # Send batch request
        response = await client.post(
            f"{API_BASE_URL}/ocr/process-batch",
            files=files,
            data={
                'strategy': 'speed',
                'language': 'en',
                'parallel_processing': 'true',
                'confidence_threshold': '0.5'
            }
        )
        
        batch_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch processed in {batch_time:.2f}s")
            print(f"Successful: {result['successful']}, Failed: {result['failed']}")
            print(f"Total processing time: {result['total_processing_time_ms']/1000:.2f}s")
            
            # Calculate throughput
            total_pages = sum(doc.get('pages', 0) for doc in result['documents'])
            pages_per_second = total_pages / batch_time if batch_time > 0 else 0
            print(f"Throughput: {pages_per_second:.2f} pages/second")
            
            return result
        else:
            print(f"Batch request failed: {response.status_code}")
            print(response.text[:500])
            return None


async def test_concurrent_single_requests(pdf_files: List[Path], concurrency: int = 10):
    """Test concurrent single PDF requests"""
    print(f"\nTesting {concurrency} concurrent single PDF requests...")
    
    async def process_single_pdf(client: httpx.AsyncClient, pdf_path: Path) -> Dict[str, Any]:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        files = {'file': (pdf_path.name, content, 'application/pdf')}
        data = {
            'language': 'en',
            'output_format': 'json',
            'confidence_threshold': '0.5',
            'merge_pages': 'true'
        }
        
        start = time.time()
        try:
            response = await client.post(
                f"{API_BASE_URL}/ocr/pdf",
                files=files,
                data=data,
                timeout=300.0
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'filename': pdf_path.name,
                    'elapsed': elapsed,
                    'data': response.json()
                }
            else:
                return {
                    'status': 'error',
                    'filename': pdf_path.name,
                    'elapsed': elapsed,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'error',
                'filename': pdf_path.name,
                'elapsed': time.time() - start,
                'error': str(e)
            }
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        # Process PDFs concurrently
        tasks = [process_single_pdf(client, pdf) for pdf in pdf_files[:concurrency]]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['status'] == 'success']
        total_pages = sum(len(r['data'].get('pages', [])) for r in successful)
        
        print(f"Completed in {total_time:.2f}s")
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Total pages: {total_pages}")
        print(f"Throughput: {total_pages / total_time:.2f} pages/second")
        
        return results


async def test_memory_usage():
    """Monitor GPU memory usage during processing"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/gpu/status")
        if response.status_code == 200:
            gpu_status = response.json()
            print(f"\nGPU Memory Status:")
            print(f"  Total: {gpu_status['memory']['total_mb']} MB")
            print(f"  Used: {gpu_status['memory']['used_mb']} MB")
            print(f"  Free: {gpu_status['memory']['free_mb']} MB")
            print(f"  Reserved: {gpu_status['memory']['reserved_mb']} MB")
            return gpu_status
        return None


async def optimize_batch_size(pdf_files: List[Path]):
    """Find optimal batch size based on GPU memory"""
    print("\nDetermining optimal batch configuration...")
    
    # Get initial GPU status
    gpu_status = await test_memory_usage()
    if not gpu_status:
        print("Could not get GPU status")
        return
    
    free_memory = gpu_status['memory']['free_mb']
    total_memory = gpu_status['memory']['total_mb']
    
    # Estimate memory per page (from our analysis: ~150MB per image in batch)
    memory_per_page_mb = 150
    
    # Calculate theoretical max pages in memory (80% of free memory)
    max_pages_in_memory = int(free_memory * 0.8 / memory_per_page_mb)
    
    print(f"\nOptimal batch configuration:")
    print(f"  Available memory: {free_memory} MB")
    print(f"  Estimated memory per page: {memory_per_page_mb} MB")
    print(f"  Max pages in single batch: {max_pages_in_memory}")
    print(f"  Recommended concurrent PDFs: {min(max_pages_in_memory // 5, 20)}")
    
    return max_pages_in_memory


async def run_performance_comparison():
    """Compare different processing strategies"""
    # Get list of PDFs
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))[:20]  # Use first 20 for testing
    
    if not pdf_files:
        print(f"No PDF files found in {TESTPDFS_DIR}")
        return
    
    print(f"Running performance comparison with {len(pdf_files)} PDFs")
    
    # Initial GPU status
    await test_memory_usage()
    
    results = {}
    
    # Test 1: Sequential processing (baseline)
    print("\n" + "="*60)
    print("TEST 1: Sequential Processing (2 concurrent)")
    start = time.time()
    sequential_results = await test_concurrent_single_requests(pdf_files, concurrency=2)
    results['sequential_2'] = {
        'time': time.time() - start,
        'results': sequential_results
    }
    
    await asyncio.sleep(5)  # Let GPU cool down
    
    # Test 2: Higher concurrency
    print("\n" + "="*60)
    print("TEST 2: Higher Concurrency (10 concurrent)")
    start = time.time()
    concurrent_results = await test_concurrent_single_requests(pdf_files, concurrency=10)
    results['concurrent_10'] = {
        'time': time.time() - start,
        'results': concurrent_results
    }
    
    await asyncio.sleep(5)
    
    # Test 3: Batch endpoint
    print("\n" + "="*60)
    print("TEST 3: Batch Endpoint (10 PDFs per batch)")
    start = time.time()
    batch_results = await test_batch_endpoint(pdf_files, batch_size=10)
    results['batch_10'] = {
        'time': time.time() - start,
        'results': batch_results
    }
    
    # Save comparison results
    comparison_file = RESULTS_DIR / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {comparison_file}")
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    for test_name, test_data in results.items():
        print(f"\n{test_name}:")
        print(f"  Total time: {test_data['time']:.2f}s")
        
        if test_name.startswith('batch'):
            if test_data['results']:
                total_pages = sum(doc.get('pages', 0) for doc in test_data['results']['documents'])
                pages_per_second = total_pages / test_data['time'] if test_data['time'] > 0 else 0
                print(f"  Pages per second: {pages_per_second:.2f}")
        else:
            successful = [r for r in test_data['results'] if r['status'] == 'success']
            total_pages = sum(len(r['data'].get('pages', [])) for r in successful)
            pages_per_second = total_pages / test_data['time'] if test_data['time'] > 0 else 0
            print(f"  Pages per second: {pages_per_second:.2f}")


async def main():
    """Main test function"""
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Find optimal batch size
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))
    await optimize_batch_size(pdf_files)
    
    # Run performance comparison
    await run_performance_comparison()


if __name__ == "__main__":
    asyncio.run(main())