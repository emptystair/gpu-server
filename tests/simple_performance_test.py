#!/usr/bin/env python3
"""
Simple performance test for batch vs sequential processing
"""

import time
import requests
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

SERVER_URL = "http://localhost:8000/api/v1"
TEST_PDFS = list(Path("testpdfs").glob("*.pdf"))[:20]  # Test with 20 PDFs

def process_single_pdf(pdf_path, dpi=120):
    """Process a single PDF"""
    start = time.time()
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {'dpi': dpi}
        
        try:
            response = requests.post(f"{SERVER_URL}/ocr/pdf", files=files, data=data, timeout=60)
            result = response.json()
            
            return {
                'file': pdf_path.name,
                'status': result.get('status'),
                'processing_time': result.get('processing_time', 0),
                'total_time': (time.time() - start) * 1000,
                'pages': result.get('pages', 0)
            }
        except Exception as e:
            return {
                'file': pdf_path.name,
                'status': 'error',
                'error': str(e),
                'total_time': (time.time() - start) * 1000
            }


def test_sequential():
    """Test sequential processing"""
    print("\n=== Testing Sequential Processing ===")
    start = time.time()
    results = []
    
    for pdf in TEST_PDFS:
        print(f"Processing {pdf.name}...")
        result = process_single_pdf(pdf)
        results.append(result)
        if result['status'] == 'completed':
            print(f"  ✓ {result['processing_time']:.0f}ms")
        else:
            print(f"  ✗ {result.get('error', 'Failed')}")
    
    total_time = (time.time() - start) * 1000
    successful = [r for r in results if r['status'] == 'completed']
    processing_times = [r['processing_time'] for r in successful]
    
    print(f"\nSequential Results:")
    print(f"  Total PDFs: {len(TEST_PDFS)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total time: {total_time:.0f}ms")
    print(f"  Throughput: {len(successful) / (total_time / 1000):.2f} PDFs/sec")
    if processing_times:
        print(f"  Avg processing time: {statistics.mean(processing_times):.0f}ms")
    
    return results, total_time


def test_parallel(num_workers=4):
    """Test parallel processing"""
    print(f"\n=== Testing Parallel Processing (workers={num_workers}) ===")
    start = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in TEST_PDFS}
        
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            result = future.result()
            results.append(result)
            
            if result['status'] == 'completed':
                print(f"  ✓ {pdf.name}: {result['processing_time']:.0f}ms")
            else:
                print(f"  ✗ {pdf.name}: {result.get('error', 'Failed')}")
    
    total_time = (time.time() - start) * 1000
    successful = [r for r in results if r['status'] == 'completed']
    processing_times = [r['processing_time'] for r in successful]
    
    print(f"\nParallel Results (workers={num_workers}):")
    print(f"  Total PDFs: {len(TEST_PDFS)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total time: {total_time:.0f}ms")
    print(f"  Throughput: {len(successful) / (total_time / 1000):.2f} PDFs/sec")
    if processing_times:
        print(f"  Avg processing time: {statistics.mean(processing_times):.0f}ms")
    
    return results, total_time


def test_batch(batch_size=10):
    """Test batch processing"""
    print(f"\n=== Testing Batch Processing (batch_size={batch_size}) ===")
    start = time.time()
    all_results = []
    
    for i in range(0, len(TEST_PDFS), batch_size):
        batch = TEST_PDFS[i:i+batch_size]
        
        files = []
        for pdf in batch:
            files.append(('files', (pdf.name, open(pdf, 'rb'), 'application/pdf')))
        
        data = {
            'dpi': 120,
            'parallel_processing': 'true'
        }
        
        try:
            response = requests.post(f"{SERVER_URL}/ocr/batch", files=files, data=data, timeout=300)
            result = response.json()
            
            print(f"Batch {i//batch_size + 1}: {result.get('successful', 0)}/{len(batch)} successful")
            
            if 'results' in result:
                all_results.extend(result['results'])
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
        
        # Close files
        for _, file_tuple in files:
            file_tuple[1].close()
    
    total_time = (time.time() - start) * 1000
    successful = [r for r in all_results if r.get('status') == 'completed']
    
    print(f"\nBatch Results (batch_size={batch_size}):")
    print(f"  Total PDFs: {len(TEST_PDFS)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Total time: {total_time:.0f}ms")
    print(f"  Throughput: {len(successful) / (total_time / 1000):.2f} PDFs/sec")
    
    return all_results, total_time


def main():
    """Run all tests"""
    print(f"Testing with {len(TEST_PDFS)} PDFs")
    
    # Check server health first
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("Server is not healthy!")
            return
    except:
        print("Cannot connect to server!")
        return
    
    results = {}
    
    # Test 1: Sequential
    seq_results, seq_time = test_sequential()
    results['sequential'] = {
        'time': seq_time,
        'throughput': len([r for r in seq_results if r['status'] == 'completed']) / (seq_time / 1000)
    }
    
    # Test 2: Parallel with different worker counts
    for workers in [2, 4, 8]:
        par_results, par_time = test_parallel(workers)
        results[f'parallel_{workers}'] = {
            'time': par_time,
            'throughput': len([r for r in par_results if r['status'] == 'completed']) / (par_time / 1000)
        }
    
    # Test 3: Batch processing
    for batch_size in [5, 10, 20]:
        batch_results, batch_time = test_batch(batch_size)
        results[f'batch_{batch_size}'] = {
            'time': batch_time,
            'throughput': len([r for r in batch_results if r.get('status') == 'completed']) / (batch_time / 1000)
        }
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    baseline = results['sequential']['throughput']
    
    for test_name, test_results in results.items():
        improvement = ((test_results['throughput'] / baseline) - 1) * 100 if baseline > 0 else 0
        print(f"{test_name:20} {test_results['throughput']:6.2f} PDFs/sec ({improvement:+.0f}%)")
    
    # Save results
    with open('performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to performance_results.json")


if __name__ == "__main__":
    main()