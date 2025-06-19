#!/usr/bin/env python3
"""Final TensorRT performance test - fresh container, no cache"""

import requests
import time
import subprocess
from pathlib import Path
from datetime import datetime

def test_single_pdf(pdf_path):
    """Test a single PDF and measure time"""
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
    # Select diverse PDFs for testing
    pdf_dir = Path("tests/testpdfs")
    test_files = [
        ("DOCC111697087_019511-00294.pdf", 1),   # 1 page
        ("DOCC111697106_019511-00322.pdf", 2),   # 2 pages
        ("DOCC111697069_019511-00182.pdf", 5),   # 5 pages
        ("DOCC111697133_019511-00404.pdf", 7),   # 7 pages
        ("DOCC111697139_019511-00434.pdf", 11),  # 11 pages
        ("DOCC111697124_019511-00359.pdf", 14),  # 14 pages
        ("DOCC111697356_019511-01733.pdf", 18),  # 18 pages
        ("DOCC111697253_019511-00962.pdf", 23),  # 23 pages
        ("DOCC111697142_019511-00460.pdf", 24),  # 24 pages
    ]
    
    print("TensorRT Performance Test - Fresh Container")
    print("=" * 70)
    print("Testing with diverse document sizes (1-24 pages)")
    print("=" * 70)
    
    total_pages = 0
    total_time = 0
    total_regions = 0
    results_by_size = {}
    
    for pdf_name, expected_pages in test_files:
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            print(f"Skipping {pdf_name} - not found")
            continue
        
        print(f"\nProcessing {pdf_name} ({expected_pages} pages)...")
        
        # Process the PDF
        result = test_single_pdf(pdf_path)
        
        if result['success']:
            pages = result['pages']
            proc_time = result['time']
            regions = result['regions']
            confidence = result['confidence']
            
            total_pages += pages
            total_time += proc_time
            total_regions += regions
            
            pages_per_sec = pages / proc_time if proc_time > 0 else 0
            
            print(f"  ✓ Processed in {proc_time:.3f}s")
            print(f"  Pages: {pages}, Regions: {regions}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Speed: {pages_per_sec:.1f} pages/sec")
            
            if pages not in results_by_size:
                results_by_size[pages] = []
            results_by_size[pages].append(proc_time)
            
        else:
            print(f"  ✗ Failed: {result['error'][:100]}")
        
        # Brief pause between documents
        time.sleep(1)
    
    # Calculate final statistics
    if total_time > 0:
        print("\n" + "=" * 70)
        print("TENSORRT PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Total documents: {len([f for f in test_files if (pdf_dir / f[0]).exists()])}")
        print(f"Total pages: {total_pages}")
        print(f"Total regions: {total_regions:,}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"\nAverage performance:")
        print(f"  Pages per second: {total_pages / total_time:.1f}")
        print(f"  Documents per minute: {len(test_files) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        
        print(f"\nPerformance by document size:")
        for pages in sorted(results_by_size.keys()):
            times = results_by_size[pages]
            avg_time = sum(times) / len(times)
            pages_per_sec = pages / avg_time
            print(f"  {pages:2d} pages: {avg_time:.3f}s ({pages_per_sec:.1f} pages/sec)")
        
        # Performance comparison
        baseline_pages_per_min = 53.6
        current_pages_per_min = total_pages / total_time * 60
        improvement = current_pages_per_min / baseline_pages_per_min
        
        print(f"\nComparison with baseline:")
        print(f"  Baseline (no TensorRT): 53.6 pages/min")
        print(f"  Current (with TensorRT): {current_pages_per_min:.1f} pages/min")
        print(f"  Performance improvement: {improvement:.1f}x")

if __name__ == "__main__":
    main()