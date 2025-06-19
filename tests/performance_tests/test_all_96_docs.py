#!/usr/bin/env python3
"""Test all 96 PDFs with TensorRT - no caching"""

import requests
import time
import os
import hashlib
from pathlib import Path
from datetime import datetime

def clear_docker_cache():
    """Clear cache in Docker container"""
    os.system("docker exec gpu-ocr-server rm -rf /app/cache/* 2>/dev/null")
    print("Cache cleared in container")

def test_single_pdf(pdf_path, index, total):
    """Test a single PDF with unique parameters to avoid caching"""
    # Add timestamp to make each request unique
    timestamp = str(time.time())
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'request_id': f'{pdf_path.name}_{timestamp}'  # Unique ID
        }
        
        start = time.time()
        try:
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
                    'confidence': result.get('average_confidence', 0),
                    'file_size': pdf_path.stat().st_size
                }
            else:
                return {
                    'success': False,
                    'error': f"Status {response.status_code}: {response.text[:100]}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)[:100]
            }

def main():
    # Clear cache first
    clear_docker_cache()
    time.sleep(2)
    
    # Get all 96 PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Testing all {len(pdf_files)} PDFs with TensorRT (no caching)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_start = time.time()
    results = []
    
    # Process each PDF
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)...")
        result = test_single_pdf(pdf, i, len(pdf_files))
        
        if result['success']:
            results.append(result)
            pages_per_sec = result['pages'] / result['time'] if result['time'] > 0 else 0
            print(f"  ✓ {result['pages']} pages, {result['regions']} regions, {result['confidence']:.1%} conf")
            print(f"  Time: {result['time']:.2f}s ({pages_per_sec:.1f} pages/sec)")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    # Calculate statistics
    total_time = time.time() - total_start
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        total_pages = sum(r['pages'] for r in successful)
        total_regions = sum(r['regions'] for r in successful)
        total_size = sum(r['file_size'] for r in successful) / (1024 * 1024)  # MB
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        avg_time = sum(r['time'] for r in successful) / len(successful)
        
        # Group by page count
        page_groups = {}
        for r in successful:
            pages = r['pages']
            if pages not in page_groups:
                page_groups[pages] = []
            page_groups[pages].append(r['time'])
        
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY - TENSORRT ENABLED")
        print("=" * 70)
        print(f"Total documents: {len(pdf_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(pdf_files) - len(successful)}")
        print(f"Total file size: {total_size:.1f} MB")
        print(f"Total pages processed: {total_pages}")
        print(f"Total text regions: {total_regions:,}")
        print(f"Average confidence: {avg_confidence:.2%}")
        print(f"\nTiming:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per document: {avg_time:.2f}s")
        print(f"  Documents per minute: {len(successful) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        print(f"  MB per minute: {total_size / total_time * 60:.1f}")
        
        print(f"\nPerformance by document size:")
        for pages in sorted(page_groups.keys()):
            times = page_groups[pages]
            avg = sum(times) / len(times)
            print(f"  {pages:2d} page docs: {len(times):3d} files, avg {avg:.2f}s/doc")
        
        # Save detailed results
        with open('tensorrt_96_docs_results.txt', 'w') as f:
            f.write(f"TensorRT Performance Test - {datetime.now()}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total documents: {len(pdf_files)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Total pages: {total_pages}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Docs/min: {len(successful) / total_time * 60:.1f}\n")
            f.write(f"Pages/min: {total_pages / total_time * 60:.1f}\n")
        
        print(f"\nResults saved to tensorrt_96_docs_results.txt")
    else:
        print("\nNo successful results to report!")

if __name__ == "__main__":
    main()