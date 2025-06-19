#!/usr/bin/env python3
"""Test OCR performance with TensorRT"""

import requests
import sys
import time
import json
from pathlib import Path

def test_batch_ocr():
    # Find test PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:10]  # Test first 10 PDFs
    
    if not pdf_files:
        print("No PDF files found in testpdfs/")
        sys.exit(1)
    
    print(f"Testing with {len(pdf_files)} PDF files")
    print("=" * 50)
    
    start_time = time.time()
    total_pages = 0
    total_regions = 0
    successful = 0
    failed = 0
    processing_times = []
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            with open(pdf_file, 'rb') as f:
                files = {'file': (pdf_file.name, f, 'application/pdf')}
                data = {
                    'strategy': 'speed',
                    'language': 'en'
                }
                
                request_start = time.time()
                response = requests.post(
                    "http://localhost:8000/api/v1/ocr/process",
                    files=files,
                    data=data,
                    timeout=120
                )
                request_time = time.time() - request_start
                processing_times.append(request_time)
            
            if response.status_code == 200:
                result = response.json()
                pages = len(result.get('pages', []))
                regions = result.get('total_regions', 0)
                confidence = result.get('average_confidence', 0)
                
                print(f"  ✓ Success: {pages} pages, {regions} regions, {confidence:.2%} confidence")
                print(f"  Processing time: {request_time:.2f}s")
                
                total_pages += pages
                total_regions += regions
                successful += 1
            else:
                print(f"  ✗ Failed: {response.status_code} - {response.text[:100]}")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY (WITH TENSORRT)")
    print("=" * 50)
    print(f"Total documents: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total pages processed: {total_pages}")
    print(f"Total text regions: {total_regions}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per document: {avg_time:.2f}s")
    print(f"Documents per minute: {(successful / total_time * 60):.1f}")
    print(f"Pages per minute: {(total_pages / total_time * 60):.1f}")
    
    # Save results
    results = {
        "tensorrt_enabled": True,
        "total_documents": len(pdf_files),
        "successful": successful,
        "failed": failed,
        "total_pages": total_pages,
        "total_regions": total_regions,
        "total_time": total_time,
        "avg_time_per_doc": avg_time,
        "docs_per_minute": successful / total_time * 60,
        "pages_per_minute": total_pages / total_time * 60
    }
    
    with open("tensorrt_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to tensorrt_test_results.json")
        
if __name__ == "__main__":
    test_batch_ocr()