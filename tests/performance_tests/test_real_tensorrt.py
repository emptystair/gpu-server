#!/usr/bin/env python3
"""Test real TensorRT performance by processing a single PDF multiple times"""

import requests
import time
import os
from pathlib import Path
import hashlib

def test_pdf(pdf_path, run_num):
    """Test a single PDF with unique request ID"""
    # Create unique request ID
    request_id = f"test_{run_num}_{time.time()}"
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'request_id': request_id
        }
        
        start = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/ocr/process",
            files=files,
            data=data,
            timeout=120
        )
        end = time.time()
    
    if response.status_code == 200:
        result = response.json()
        return {
            'success': True,
            'time': end - start,
            'pages': len(result.get('pages', [])),
            'regions': result.get('total_regions', 0),
            'confidence': result.get('average_confidence', 0)
        }
    else:
        return {'success': False, 'error': response.text}

def main():
    # Test with a few different PDFs to get real performance
    pdf_dir = Path("tests/testpdfs")
    test_files = [
        "DOCC111697069_019511-00182.pdf",  # 5 pages
        "DOCC111697142_019511-00460.pdf",  # 24 pages
        "DOCC111697253_019511-00962.pdf",  # 23 pages
        "DOCC111697356_019511-01733.pdf",  # 18 pages
        "DOCC111697124_019511-00359.pdf",  # 14 pages
    ]
    
    print("Testing real TensorRT performance (5 different PDFs, 3 runs each)")
    print("=" * 70)
    
    all_times = []
    all_pages = []
    
    for pdf_name in test_files:
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            print(f"Skipping {pdf_name} - not found")
            continue
            
        print(f"\nTesting {pdf_name}:")
        times = []
        
        for run in range(3):
            # Wait between runs to ensure no overlap
            if run > 0:
                time.sleep(2)
                
            result = test_pdf(pdf_path, run)
            
            if result['success']:
                times.append(result['time'])
                all_times.append(result['time'])
                all_pages.append(result['pages'])
                
                pages_per_sec = result['pages'] / result['time']
                print(f"  Run {run+1}: {result['pages']} pages in {result['time']:.2f}s = {pages_per_sec:.1f} pages/sec")
            else:
                print(f"  Run {run+1}: Failed - {result['error'][:50]}")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"  Average: {avg_time:.2f}s")
    
    if all_times:
        total_pages = sum(all_pages)
        total_time = sum(all_times)
        avg_pages_per_run = total_pages / len(all_times)
        avg_time_per_run = total_time / len(all_times)
        
        print("\n" + "=" * 70)
        print("SUMMARY - REAL TENSORRT PERFORMANCE")
        print("=" * 70)
        print(f"Total runs: {len(all_times)}")
        print(f"Total pages processed: {total_pages}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average pages per document: {avg_pages_per_run:.1f}")
        print(f"Average time per document: {avg_time_per_run:.2f}s")
        print(f"Average pages per second: {total_pages / total_time:.1f}")
        print(f"Projected documents per minute: {60 / avg_time_per_run:.1f}")
        print(f"Projected pages per minute: {total_pages / total_time * 60:.1f}")

if __name__ == "__main__":
    main()