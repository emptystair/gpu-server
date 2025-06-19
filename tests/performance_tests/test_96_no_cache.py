#!/usr/bin/env python3
"""Test all 96 PDFs with TensorRT - truly bypassing cache"""

import requests
import time
import os
from pathlib import Path
from datetime import datetime
import hashlib
import json

def test_single_pdf(pdf_path, index, total):
    """Test a single PDF with cache bypass"""
    # Create unique file hash for tracking
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'cache_bypass': 'true',  # Force bypass cache
            'request_id': f'{file_hash}_{time.time()}'  # Unique ID
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
                pages = len(result.get('pages', []))
                regions = result.get('total_regions', 0)
                confidence = result.get('average_confidence', 0)
                
                # Check if this was a cache hit (should not be)
                cache_hit = result.get('cache_hit', False)
                processing_time = end - start
                
                print(f"[{index}/{total}] {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
                print(f"  ✓ {pages} pages, {regions} regions, {confidence:.1%} conf")
                print(f"  Time: {processing_time:.2f}s ({pages / processing_time:.1f} pages/sec)")
                if cache_hit:
                    print(f"  ⚠️ WARNING: Cache hit despite bypass!")
                
                return {
                    'success': True,
                    'time': processing_time,
                    'pages': pages,
                    'regions': regions,
                    'confidence': confidence,
                    'file_size': pdf_path.stat().st_size,
                    'cache_hit': cache_hit
                }
            else:
                print(f"[{index}/{total}] {pdf_path.name} - Failed: {response.status_code}")
                return {
                    'success': False,
                    'error': f"Status {response.status_code}: {response.text[:100]}"
                }
        except Exception as e:
            print(f"[{index}/{total}] {pdf_path.name} - Error: {str(e)[:50]}")
            return {
                'success': False,
                'error': str(e)[:100]
            }

def main():
    # Get all 96 PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Testing all {len(pdf_files)} PDFs with TensorRT (cache bypass enabled)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Warm up the server with a dummy request
    print("\nWarming up server...")
    try:
        response = requests.get("http://localhost:8000/api/v1/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Server ready. Processed {stats.get('total_requests', 0)} requests so far.")
    except:
        pass
    
    print("\nStarting tests...\n")
    
    total_start = time.time()
    results = []
    
    # Process each PDF
    for i, pdf in enumerate(pdf_files, 1):
        result = test_single_pdf(pdf, i, len(pdf_files))
        results.append(result)
        
        # Small delay between requests to ensure no overlap
        if i < len(pdf_files):
            time.sleep(0.1)
    
    # Calculate statistics
    total_time = time.time() - total_start
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        total_pages = sum(r['pages'] for r in successful)
        total_regions = sum(r['regions'] for r in successful)
        total_size = sum(r['file_size'] for r in successful) / (1024 * 1024)  # MB
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        processing_times = [r['time'] for r in successful]
        avg_time = sum(processing_times) / len(processing_times)
        cache_hits = sum(1 for r in successful if r.get('cache_hit', False))
        
        # Group by page count
        page_groups = {}
        for r in successful:
            pages = r['pages']
            if pages not in page_groups:
                page_groups[pages] = []
            page_groups[pages].append(r['time'])
        
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY - TENSORRT ENABLED (NO CACHE)")
        print("=" * 70)
        print(f"Total documents: {len(pdf_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(pdf_files) - len(successful)}")
        print(f"Cache hits: {cache_hits} (should be 0)")
        print(f"Total file size: {total_size:.1f} MB")
        print(f"Total pages processed: {total_pages}")
        print(f"Total text regions: {total_regions:,}")
        print(f"Average confidence: {avg_confidence:.2%}")
        print(f"\nTiming:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processing time (sum): {sum(processing_times):.2f}s")
        print(f"  Average time per document: {avg_time:.2f}s")
        print(f"  Documents per minute: {len(successful) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        print(f"  MB per minute: {total_size / total_time * 60:.1f}")
        
        print(f"\nPerformance by document size:")
        for pages in sorted(page_groups.keys()):
            times = page_groups[pages]
            avg = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"  {pages:2d} page docs: {len(times):3d} files, avg {avg:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f'tensorrt_96_docs_no_cache_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump({
                'summary': {
                    'tensorrt_enabled': True,
                    'cache_bypass': True,
                    'total_documents': len(pdf_files),
                    'successful': len(successful),
                    'failed': len(pdf_files) - len(successful),
                    'cache_hits': cache_hits,
                    'total_pages': total_pages,
                    'total_regions': total_regions,
                    'total_size_mb': total_size,
                    'average_confidence': avg_confidence,
                    'total_time': total_time,
                    'processing_time_sum': sum(processing_times),
                    'average_time_per_doc': avg_time,
                    'docs_per_minute': len(successful) / total_time * 60,
                    'pages_per_minute': total_pages / total_time * 60,
                    'mb_per_minute': total_size / total_time * 60
                },
                'page_groups': {str(k): v for k, v in page_groups.items()},
                'individual_results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to {result_file}")
        
        # Compare with baseline
        baseline_docs_per_min = 10.5  # Without TensorRT
        current_docs_per_min = len(successful) / total_time * 60
        improvement = current_docs_per_min / baseline_docs_per_min
        print(f"\nPerformance improvement over baseline:")
        print(f"  Baseline: {baseline_docs_per_min:.1f} docs/min")
        print(f"  Current:  {current_docs_per_min:.1f} docs/min")
        print(f"  Improvement: {improvement:.1f}x")
        
    else:
        print("\nNo successful results to report!")

if __name__ == "__main__":
    main()