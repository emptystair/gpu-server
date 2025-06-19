#!/usr/bin/env python3
"""Test real OCR performance without caching"""

import requests
import time
import os
from pathlib import Path

def test_single_pdf(pdf_path):
    """Test a single PDF and return timing"""
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en'
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
    # Test first 10 different PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:10]
    
    print("Testing OCR performance with TensorRT (no caching)")
    print("=" * 50)
    
    total_start = time.time()
    total_pages = 0
    total_regions = 0
    times = []
    
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n[{i}/10] Processing {pdf.name}...")
        result = test_single_pdf(pdf)
        
        if result['success']:
            times.append(result['time'])
            total_pages += result['pages']
            total_regions += result['regions']
            print(f"  ✓ {result['pages']} pages, {result['regions']} regions")
            print(f"  Time: {result['time']:.2f}s ({result['pages']/result['time']:.1f} pages/sec)")
            print(f"  Confidence: {result['confidence']:.1%}")
        else:
            print(f"  ✗ Failed: {result['error'][:100]}")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total documents: {len(pdf_files)}")
    print(f"Total pages: {total_pages}")
    print(f"Total regions: {total_regions}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per document: {sum(times)/len(times):.2f}s")
    print(f"Documents per minute: {len(pdf_files) / total_time * 60:.1f}")
    print(f"Pages per minute: {total_pages / total_time * 60:.1f}")

if __name__ == "__main__":
    main()