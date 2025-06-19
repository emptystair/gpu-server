#!/usr/bin/env python3
"""Analyze OCR results and show different ways to access the data"""

import json
from pathlib import Path

def analyze_ocr_json(json_file):
    """Analyze OCR JSON results"""
    print(f"\nAnalyzing: {json_file}")
    print("=" * 70)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Basic metadata
    print(f"Request ID: {data.get('request_id', 'N/A')}")
    print(f"Status: {data.get('status', 'N/A')}")
    print(f"Processing time: {data.get('processing_time', 0):.2f}ms")
    print(f"Cache hit: {data.get('cache_hit', False)}")
    print(f"Total pages: {len(data.get('pages', []))}")
    print(f"Total regions: {data.get('total_regions', 0)}")
    print(f"Average confidence: {data.get('average_confidence', 0):.2%}")
    
    # Page analysis
    pages = data.get('pages', [])
    if pages:
        print(f"\nPage details:")
        for page in pages:
            page_num = page.get('page_number', 0)
            regions = page.get('regions', [])
            print(f"  Page {page_num}: {len(regions)} text regions")
            
            # Show first few regions
            if regions:
                print(f"    Sample text regions:")
                for i, region in enumerate(regions[:3]):
                    text = region.get('text', '').strip()
                    conf = region.get('confidence', 0)
                    bbox = region.get('bbox', {})
                    print(f"      [{i+1}] \"{text[:50]}...\" (conf: {conf:.2%})")
                    print(f"          Position: x1={bbox.get('x1', 0)}, y1={bbox.get('y1', 0)}")
    
    # Find low confidence regions
    print(f"\nLow confidence regions (< 90%):")
    low_conf_count = 0
    for page in pages:
        for region in page.get('regions', []):
            if region.get('confidence', 1) < 0.9:
                low_conf_count += 1
                text = region.get('text', '').strip()
                conf = region.get('confidence', 0)
                print(f"  - \"{text[:50]}...\" (conf: {conf:.2%})")
                if low_conf_count >= 5:
                    print(f"  ... and {sum(1 for p in pages for r in p.get('regions', []) if r.get('confidence', 1) < 0.9) - 5} more")
                    break
        if low_conf_count >= 5:
            break
    
    if low_conf_count == 0:
        print("  None found - all regions have high confidence!")
    
    return data

def main():
    # Analyze all JSON results in the ocr_results directory
    results_dir = Path("ocr_results")
    json_files = list(results_dir.glob("*_ocr_result.json"))
    
    if not json_files:
        print("No OCR result files found in ocr_results/")
        return
    
    print(f"Found {len(json_files)} OCR result files")
    
    for json_file in json_files:
        analyze_ocr_json(json_file)
    
    print("\n" + "=" * 70)
    print("You can access OCR results via the API in different formats:")
    print("  - output_format='json': Structured data with bounding boxes")
    print("  - output_format='text': Plain text extraction")
    print("  - output_format='markdown': Formatted markdown")
    print("  - output_format='html': HTML formatted output")

if __name__ == "__main__":
    main()