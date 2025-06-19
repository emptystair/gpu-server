#!/usr/bin/env python3
"""Test secondary PDFs for accurate TensorRT performance without cache"""

import requests
import time
import os
from pathlib import Path
from datetime import datetime
import json
import statistics

def test_single_pdf(pdf_path, run_number):
    """Test a single PDF and return timing info"""
    # Unique request ID to prevent any caching
    request_id = f"secondary_test_{pdf_path.stem}_{run_number}_{time.time()}"
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'request_id': request_id
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
    # Get all PDFs from secondary folder
    pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs/secondary")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Testing {len(pdf_files)} new PDFs from secondary folder")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Warm up with one file
    if pdf_files:
        print("\nWarming up server...")
        test_single_pdf(pdf_files[0], 0)
        time.sleep(2)
    
    print("\nStarting performance test...\n")
    
    total_start = time.time()
    results = []
    
    # Process each PDF
    for i, pdf in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)...", end='', flush=True)
        
        result = test_single_pdf(pdf, i)
        
        if result['success']:
            results.append(result)
            pages = result['pages']
            proc_time = result['time']
            regions = result['regions']
            confidence = result['confidence']
            
            pages_per_sec = pages / proc_time if proc_time > 0 else 0
            print(f" ✓ {pages} pages in {proc_time:.3f}s ({pages_per_sec:.1f} pages/sec)")
        else:
            print(f" ✗ Failed: {result['error']}")
        
        # Small delay between files to ensure clean processing
        if i < len(pdf_files):
            time.sleep(0.5)
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    if results:
        successful = [r for r in results if r.get('success', False)]
        total_pages = sum(r['pages'] for r in successful)
        total_regions = sum(r['regions'] for r in successful)
        total_size_mb = sum(r['file_size'] for r in successful) / (1024 * 1024)
        
        processing_times = [r['time'] for r in successful]
        avg_time = statistics.mean(processing_times)
        median_time = statistics.median(processing_times)
        
        confidences = [r['confidence'] for r in successful]
        avg_confidence = statistics.mean(confidences) if confidences else 0
        
        # Group by page count
        page_groups = {}
        for r in successful:
            pages = r['pages']
            if pages not in page_groups:
                page_groups[pages] = []
            page_groups[pages].append(r['time'])
        
        print("\n" + "=" * 70)
        print("TENSORRT PERFORMANCE RESULTS - NEW PDFS (NO CACHE)")
        print("=" * 70)
        print(f"Total documents tested: {len(pdf_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(pdf_files) - len(successful)}")
        print(f"Total file size: {total_size_mb:.1f} MB")
        print(f"Total pages processed: {total_pages}")
        print(f"Total text regions: {total_regions:,}")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        print(f"\nTiming Statistics:")
        print(f"  Total elapsed time: {total_time:.2f}s")
        print(f"  Processing time (sum): {sum(processing_times):.2f}s")
        print(f"  Average time per document: {avg_time:.3f}s")
        print(f"  Median time per document: {median_time:.3f}s")
        print(f"  Min processing time: {min(processing_times):.3f}s")
        print(f"  Max processing time: {max(processing_times):.3f}s")
        
        print(f"\nPerformance Metrics:")
        print(f"  Documents per minute: {len(successful) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        print(f"  MB per minute: {total_size_mb / total_time * 60:.1f}")
        
        print(f"\nPerformance by document size:")
        for pages in sorted(page_groups.keys()):
            times = page_groups[pages]
            avg = statistics.mean(times)
            count = len(times)
            print(f"  {pages:3d} page docs: {count:3d} files, avg {avg:.3f}s/doc ({pages/avg:.1f} pages/sec)")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f'tensorrt_secondary_pdfs_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump({
                'summary': {
                    'test_type': 'secondary_pdfs_no_cache',
                    'tensorrt_enabled': True,
                    'total_documents': len(pdf_files),
                    'successful': len(successful),
                    'failed': len(pdf_files) - len(successful),
                    'total_pages': total_pages,
                    'total_regions': total_regions,
                    'total_size_mb': total_size_mb,
                    'average_confidence': avg_confidence,
                    'timing': {
                        'total_elapsed': total_time,
                        'sum_processing': sum(processing_times),
                        'avg_per_doc': avg_time,
                        'median_per_doc': median_time,
                        'min': min(processing_times),
                        'max': max(processing_times)
                    },
                    'performance': {
                        'docs_per_minute': len(successful) / total_time * 60,
                        'pages_per_minute': total_pages / total_time * 60,
                        'mb_per_minute': total_size_mb / total_time * 60
                    }
                },
                'page_groups': {str(k): {
                    'count': len(v), 
                    'avg_time': statistics.mean(v),
                    'times': v
                } for k, v in page_groups.items()},
                'individual_results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to {result_file}")
        
        # Comparison with baseline
        print(f"\nPerformance Comparison:")
        print(f"  Baseline (no TensorRT): 10.5 docs/min, 53.6 pages/min")
        print(f"  Current (with TensorRT): {len(successful) / total_time * 60:.1f} docs/min, {total_pages / total_time * 60:.1f} pages/min")
        print(f"  Document processing improvement: {(len(successful) / total_time * 60) / 10.5:.1f}x")
        print(f"  Page processing improvement: {(total_pages / total_time * 60) / 53.6:.1f}x")

if __name__ == "__main__":
    main()