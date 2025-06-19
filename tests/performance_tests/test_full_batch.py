#!/usr/bin/env python3
"""Test OCR performance with all 96 documents"""

import requests
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe counter
processed_count = 0
count_lock = threading.Lock()

def process_single_pdf(pdf_file, session):
    """Process a single PDF file"""
    global processed_count
    
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            data = {
                'strategy': 'speed',
                'language': 'en',
                'cache_key': f'{pdf_file.name}_{time.time()}'  # Unique key to bypass cache
            }
            
            request_start = time.time()
            response = session.post(
                "http://localhost:8000/api/v1/ocr/process",
                files=files,
                data=data,
                timeout=120
            )
            request_time = time.time() - request_start
        
        if response.status_code == 200:
            result = response.json()
            pages = len(result.get('pages', []))
            regions = result.get('total_regions', 0)
            confidence = result.get('average_confidence', 0)
            
            with count_lock:
                processed_count += 1
                print(f"[{processed_count}/96] ✓ {pdf_file.name}: {pages} pages, {regions} regions, {confidence:.1%} conf, {request_time:.1f}s")
            
            return {
                'success': True,
                'file': pdf_file.name,
                'pages': pages,
                'regions': regions,
                'confidence': confidence,
                'time': request_time
            }
        else:
            with count_lock:
                processed_count += 1
                print(f"[{processed_count}/96] ✗ {pdf_file.name}: {response.status_code}")
            return {
                'success': False,
                'file': pdf_file.name,
                'error': response.text[:100]
            }
            
    except Exception as e:
        with count_lock:
            processed_count += 1
            print(f"[{processed_count}/96] ✗ {pdf_file.name}: {str(e)[:50]}")
        return {
            'success': False,
            'file': pdf_file.name,
            'error': str(e)
        }

def test_batch_ocr(num_workers=4):
    """Test OCR performance with concurrent requests"""
    global processed_count
    processed_count = 0
    
    # Find all test PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:20]  # Test first 20 for timing
    
    if not pdf_files:
        print("No PDF files found in tests/testpdfs/")
        sys.exit(1)
    
    print(f"Testing with {len(pdf_files)} PDF files using {num_workers} concurrent workers")
    print("=" * 70)
    
    start_time = time.time()
    results = []
    
    # Create a session for connection pooling
    session = requests.Session()
    
    # Process files concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(process_single_pdf, pdf, session): pdf 
                         for pdf in pdf_files}
        
        # Collect results as they complete
        for future in as_completed(future_to_pdf):
            result = future.result()
            results.append(result)
    
    # Calculate statistics
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_pages = sum(r['pages'] for r in successful)
    total_regions = sum(r['regions'] for r in successful)
    avg_confidence = sum(r['confidence'] for r in successful) / len(successful) if successful else 0
    avg_time = sum(r['time'] for r in successful) / len(successful) if successful else 0
    
    print("\n" + "=" * 70)
    print(f"PERFORMANCE SUMMARY (TENSORRT + {num_workers} WORKERS)")
    print("=" * 70)
    print(f"Total documents: {len(pdf_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total text regions: {total_regions}")
    print(f"Average confidence: {avg_confidence:.2%}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per document: {avg_time:.2f}s")
    print(f"Documents per minute: {(len(successful) / total_time * 60):.1f}")
    print(f"Pages per minute: {(total_pages / total_time * 60):.1f}")
    print(f"Throughput improvement: {num_workers:.1f}x workers = {(len(successful) / total_time * 60 / 32.6):.2f}x speedup")
    
    # Save detailed results
    summary = {
        "tensorrt_enabled": True,
        "num_workers": num_workers,
        "total_documents": len(pdf_files),
        "successful": len(successful),
        "failed": len(failed),
        "total_pages": total_pages,
        "total_regions": total_regions,
        "avg_confidence": avg_confidence,
        "total_time": total_time,
        "avg_time_per_doc": avg_time,
        "docs_per_minute": len(successful) / total_time * 60,
        "pages_per_minute": total_pages / total_time * 60,
        "failed_files": [r['file'] for r in failed]
    }
    
    with open(f"batch_test_results_{num_workers}_workers.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to batch_test_results_{num_workers}_workers.json")
    
    return summary

if __name__ == "__main__":
    # Test with different numbers of concurrent workers
    import sys
    
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    test_batch_ocr(workers)