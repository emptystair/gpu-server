#!/usr/bin/env python3
"""Quick test of secondary PDFs to check performance"""

import requests
import time
from pathlib import Path
from datetime import datetime

def test_pdf(pdf_path):
    """Test a single PDF"""
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en'
        }
        
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/ocr/process",
                files=files,
                data=data,
                timeout=60
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
                    'file': pdf_path.name
                }
            else:
                return {'success': False, 'file': pdf_path.name, 'error': response.status_code}
        except Exception as e:
            return {'success': False, 'file': pdf_path.name, 'error': str(e)[:50]}

def main():
    # Get first 10 PDFs from secondary folder
    pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs/secondary")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))[:10]
    
    print(f"Quick test of {len(pdf_files)} secondary PDFs")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    total_start = time.time()
    results = []
    total_pages = 0
    
    for i, pdf in enumerate(pdf_files, 1):
        print(f"[{i:2d}/10] {pdf.name:20s} ", end='', flush=True)
        
        result = test_pdf(pdf)
        results.append(result)
        
        if result['success']:
            total_pages += result['pages']
            pages_per_sec = result['pages'] / result['time'] if result['time'] > 0 else 0
            print(f"✓ {result['pages']:2d} pages, {result['time']:5.2f}s, {pages_per_sec:6.1f} p/s")
        else:
            print(f"✗ Error: {result['error']}")
        
        # Pause between files
        time.sleep(0.5)
    
    total_time = time.time() - total_start
    successful = [r for r in results if r['success']]
    
    if successful:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Successful: {len(successful)}/{len(pdf_files)}")
        print(f"Total pages: {total_pages}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Processing times: {[f'{r["time"]:.2f}s' for r in successful]}")
        print(f"\nPerformance:")
        print(f"  Docs/min: {len(successful) / total_time * 60:.1f}")
        print(f"  Pages/min: {total_pages / total_time * 60:.1f}")
        print(f"\nVs baseline (10.5 docs/min, 53.6 pages/min):")
        print(f"  Document improvement: {(len(successful) / total_time * 60) / 10.5:.1f}x")
        print(f"  Page improvement: {(total_pages / total_time * 60) / 53.6:.1f}x")

if __name__ == "__main__":
    main()