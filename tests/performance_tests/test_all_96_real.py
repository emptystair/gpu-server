#!/usr/bin/env python3
"""Test all 96 PDFs with real TensorRT processing - with timing verification"""

import requests
import time
import os
from pathlib import Path
from datetime import datetime
import json

def process_pdf_batch(pdf_files, batch_size=5):
    """Process PDFs in batches to get more accurate timing"""
    results = []
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        batch_start = time.time()
        
        print(f"\nBatch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size}")
        
        for pdf in batch:
            file_start = time.time()
            
            # Unique request ID
            request_id = f"batch_test_{pdf.name}_{time.time()}"
            
            with open(pdf, 'rb') as f:
                files = {'file': (pdf.name, f, 'application/pdf')}
                data = {
                    'strategy': 'speed',
                    'language': 'en',
                    'request_id': request_id
                }
                
                response = requests.post(
                    "http://localhost:8000/api/v1/ocr/process",
                    files=files,
                    data=data,
                    timeout=120
                )
            
            file_time = time.time() - file_start
            
            if response.status_code == 200:
                result = response.json()
                pages = len(result.get('pages', []))
                regions = result.get('total_regions', 0)
                confidence = result.get('average_confidence', 0)
                
                print(f"  ✓ {pdf.name}: {pages} pages in {file_time:.2f}s")
                
                results.append({
                    'file': pdf.name,
                    'success': True,
                    'pages': pages,
                    'regions': regions,
                    'confidence': confidence,
                    'time': file_time,
                    'size_kb': pdf.stat().st_size / 1024
                })
            else:
                print(f"  ✗ {pdf.name}: Failed - {response.status_code}")
                results.append({
                    'file': pdf.name,
                    'success': False,
                    'time': file_time
                })
        
        batch_time = time.time() - batch_start
        successful_in_batch = sum(1 for r in results[-len(batch):] if r['success'])
        pages_in_batch = sum(r.get('pages', 0) for r in results[-len(batch):] if r['success'])
        
        print(f"  Batch time: {batch_time:.2f}s for {successful_in_batch} docs, {pages_in_batch} pages")
        print(f"  Batch rate: {successful_in_batch/batch_time*60:.1f} docs/min, {pages_in_batch/batch_time*60:.1f} pages/min")
        
        # Pause between batches
        if i + batch_size < len(pdf_files):
            time.sleep(2)
    
    return results

def main():
    # Get all PDFs
    pdf_dir = Path("tests/testpdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"Testing all {len(pdf_files)} PDFs with TensorRT - Real Performance Test")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing in batches of 5 to ensure accurate timing")
    print("=" * 70)
    
    # Process all PDFs
    total_start = time.time()
    results = process_pdf_batch(pdf_files)
    total_time = time.time() - total_start
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    
    if successful:
        total_pages = sum(r['pages'] for r in successful)
        total_regions = sum(r['regions'] for r in successful)
        total_size_mb = sum(r['size_kb'] for r in successful) / 1024
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        
        # Remove outliers (very fast times that might be cached)
        processing_times = [r['time'] for r in successful]
        avg_time = sum(processing_times) / len(processing_times)
        
        # Filter out suspiciously fast times (< 10ms)
        real_times = [t for t in processing_times if t >= 0.01]
        real_avg_time = sum(real_times) / len(real_times) if real_times else avg_time
        
        # Group by page count
        page_groups = {}
        for r in successful:
            pages = r['pages']
            if pages not in page_groups:
                page_groups[pages] = {'times': [], 'count': 0}
            page_groups[pages]['times'].append(r['time'])
            page_groups[pages]['count'] += 1
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS - TENSORRT PERFORMANCE")
        print("=" * 70)
        print(f"Total documents: {len(pdf_files)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        print(f"Total file size: {total_size_mb:.1f} MB")
        print(f"Total pages processed: {total_pages}")
        print(f"Total text regions: {total_regions:,}")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        print(f"\nTiming Analysis:")
        print(f"  Total elapsed time: {total_time:.2f}s")
        print(f"  Sum of processing times: {sum(processing_times):.2f}s")
        print(f"  Average time per document: {real_avg_time:.3f}s")
        print(f"  Documents per minute: {len(successful) / total_time * 60:.1f}")
        print(f"  Pages per minute: {total_pages / total_time * 60:.1f}")
        print(f"  MB per minute: {total_size_mb / total_time * 60:.1f}")
        
        # Exclude very fast outliers
        fast_count = len([t for t in processing_times if t < 0.01])
        if fast_count > 0:
            print(f"\n  Note: {fast_count} documents processed in <10ms (possible cache hits)")
            real_docs = len(successful) - fast_count
            real_time = sum(real_times)
            if real_docs > 0:
                print(f"  Real performance (excluding <10ms): {real_docs} docs in {real_time:.1f}s")
                print(f"  Real rate: {real_docs / real_time * 60:.1f} docs/min")
        
        print(f"\nPerformance by document size:")
        for pages in sorted(page_groups.keys()):
            group = page_groups[pages]
            times = group['times']
            avg = sum(times) / len(times)
            # Filter real processing times
            real_times_group = [t for t in times if t >= 0.01]
            real_avg = sum(real_times_group) / len(real_times_group) if real_times_group else avg
            print(f"  {pages:2d} page docs: {group['count']:3d} files, avg {real_avg:.3f}s/doc")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f'tensorrt_all_96_real_{timestamp}.json'
        
        with open(result_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_documents': len(pdf_files),
                    'successful': len(successful),
                    'total_pages': total_pages,
                    'total_regions': total_regions,
                    'total_size_mb': total_size_mb,
                    'average_confidence': avg_confidence,
                    'total_time': total_time,
                    'avg_time_per_doc': real_avg_time,
                    'docs_per_minute': len(successful) / total_time * 60,
                    'pages_per_minute': total_pages / total_time * 60,
                    'fast_outliers': fast_count
                },
                'page_groups': {str(k): {'count': v['count'], 'avg_time': sum(v['times'])/len(v['times'])} 
                               for k, v in page_groups.items()},
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {result_file}")
        
        # Performance comparison
        print(f"\nPerformance Summary:")
        print(f"  Baseline (no TensorRT): 10.5 docs/min, 53.6 pages/min")
        print(f"  Current (with TensorRT): {len(successful) / total_time * 60:.1f} docs/min, {total_pages / total_time * 60:.1f} pages/min")
        print(f"  Improvement: {(len(successful) / total_time * 60) / 10.5:.1f}x faster")
        
    else:
        print("\nNo successful results!")

if __name__ == "__main__":
    main()