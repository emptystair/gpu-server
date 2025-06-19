#!/usr/bin/env python3
"""Process PDFs and save OCR results to files"""

import requests
import json
from pathlib import Path
from datetime import datetime

def process_and_save(pdf_path, output_dir):
    """Process a PDF and save results"""
    print(f"\nProcessing {pdf_path.name}...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'output_format': 'json'  # Can be: json, text, markdown, html
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/ocr/process",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Save JSON result
                json_file = output_dir / f"{pdf_path.stem}_ocr_result.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Extract and save text content
                text_file = output_dir / f"{pdf_path.stem}_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Results for: {pdf_path.name}\n")
                    f.write(f"Processed: {datetime.now()}\n")
                    f.write(f"Total pages: {len(result.get('pages', []))}\n")
                    f.write(f"Total regions: {result.get('total_regions', 0)}\n")
                    f.write(f"Average confidence: {result.get('average_confidence', 0):.2%}\n")
                    f.write("="*70 + "\n\n")
                    
                    # Write text by page
                    for page in result.get('pages', []):
                        page_num = page.get('page_number', 0)
                        f.write(f"\n--- Page {page_num} ---\n")
                        f.write(f"Regions: {len(page.get('regions', []))}\n\n")
                        
                        for region in page.get('regions', []):
                            text = region.get('text', '').strip()
                            confidence = region.get('confidence', 0)
                            if text:
                                f.write(f"{text}\n")
                                if confidence < 0.9:  # Flag low confidence
                                    f.write(f"  [Confidence: {confidence:.2%}]\n")
                        f.write("\n")
                
                # Also get markdown format
                data['output_format'] = 'markdown'
                response = requests.post(
                    "http://localhost:8000/api/v1/ocr/process",
                    files={'file': (pdf_path.name, open(pdf_path, 'rb'), 'application/pdf')},
                    data=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    md_result = response.json()
                    md_file = output_dir / f"{pdf_path.stem}_markdown.md"
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(md_result.get('text', ''))
                
                print(f"✓ Success! Results saved to:")
                print(f"  - {json_file}")
                print(f"  - {text_file}")
                print(f"  - {md_file}")
                
                # Show summary
                pages = len(result.get('pages', []))
                regions = result.get('total_regions', 0)
                confidence = result.get('average_confidence', 0)
                print(f"  Summary: {pages} pages, {regions} regions, {confidence:.1%} confidence")
                
                return True
                
            else:
                print(f"✗ Error: Status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            return False

def main():
    # Create output directory
    output_dir = Path("ocr_results")
    output_dir.mkdir(exist_ok=True)
    
    print("OCR Results Saver")
    print("=" * 70)
    
    # Process a few sample PDFs
    test_pdfs = [
        Path("tests/testpdfs/DOCC111697069_019511-00182.pdf"),  # 5 pages
        Path("tests/testpdfs/secondary/DOC717S208.pdf"),        # 3 pages
        Path("tests/testpdfs/secondary/DOC719S607.pdf"),        # 11 pages
    ]
    
    for pdf in test_pdfs:
        if pdf.exists():
            process_and_save(pdf, output_dir)
        else:
            print(f"\n✗ Not found: {pdf}")
    
    print(f"\n\nAll results saved to: {output_dir.absolute()}")
    print("\nYou can view the results in these formats:")
    print("  - *_ocr_result.json: Complete structured OCR data")
    print("  - *_text.txt: Plain text extraction")
    print("  - *_markdown.md: Formatted markdown output")

if __name__ == "__main__":
    main()