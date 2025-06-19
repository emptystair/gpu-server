#!/usr/bin/env python3
"""Test subset of secondary PDFs for accurate performance metrics"""

import requests
import time
from pathlib import Path
from datetime import datetime
import statistics

def test_pdf(pdf_path):
    """Test a single PDF"""
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'request_id': f'secondary_{pdf_path.stem}_{time.time()}'
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
                    'confidence': result.get('average_confidence', 0)
                }
            else:
                return {'success': False, 'error': response.status_code}
        except Exception as e:
            return {'success': False, 'error': str(e)[:50]}

def main():
    # Get 30 PDFs from secondary folder (skip first one that might be cached)
    pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs/secondary")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[1:31]  # Skip first, take next 30
    
    print(f"Testing {len(pdf_files)} secondary PDFs for TensorRT performance")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_start = time.time()
    results = []
    
    for i, pdf in enumerate(pdf_files, 1):
        print(f"[{i:2d}/30] {pdf.name:20s} ", end='', flush=True)
        
        result = test_pdf(pdf)
        
        if result['success']:
            results.append(result)
            pages = result['pages']
            proc_time = result['time']
            pages_per_sec = pages / proc_time if proc_time > 0 else 0
            print(f"✓ {pages:2d}p, {proc_time:5.2f}s, {pages_per_sec:5.1f}p/s, {result['confidence']:.0%} conf")
        else:
            print(f"✗ Error: {result['error']}")
        
        # Small pause between files
        if i < len(pdf_files):
            time.sleep(0.2)
    
    total_time = time.time() - total_start
    
    if results:
        # Calculate statistics
        total_pages = sum(r['pages'] for r in results)
        total_regions = sum(r['regions'] for r in results)
        processing_times = [r['time'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Remove outliers (very fast times < 0.5s that might be partial cache)
        real_times = [t for t in processing_times if t >= 0.5]
        
        print("\n" + "=" * 70)
        print("TENSORRT PERFORMANCE - SECONDARY PDFS (NO CACHE)")
        print("=" * 70)
        print(f"Documents tested: {len(pdf_files)}")
        print(f"Successful: {len(results)}")
        print(f"Total pages: {total_pages}")
        print(f"Total regions: {total_regions:,}")
        print(f"Average confidence: {statistics.mean(confidences):.1%}")
        
        print(f"\nTiming Analysis:")
        print(f"  Total elapsed time: {total_time:.1f}s")
        print(f"  All processing times:")
        print(f"    Mean: {statistics.mean(processing_times):.2f}s")
        print(f"    Median: {statistics.median(processing_times):.2f}s")
        print(f"    Min: {min(processing_times):.2f}s")
        print(f"    Max: {max(processing_times):.2f}s")
        
        if real_times:
            print(f"  Real processing times (≥0.5s, {len(real_times)} samples):")
            print(f"    Mean: {statistics.mean(real_times):.2f}s")
            print(f"    Median: {statistics.median(real_times):.2f}s")
        
        print(f"\nPerformance Metrics:")
        print(f"  Documents per minute: {len(results) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        
        print(f"\nComparison with baseline:")
        print(f"  Baseline: 10.5 docs/min, 53.6 pages/min")
        print(f"  Current: {len(results) / total_time * 60:.1f} docs/min, {total_pages / total_time * 60:.1f} pages/min")
        print(f"  Improvement: {(len(results) / total_time * 60) / 10.5:.1f}x docs, {(total_pages / total_time * 60) / 53.6:.1f}x pages")

if __name__ == "__main__":
    main()