#!/usr/bin/env python3
"""Test a single PDF to check if server is working"""

import requests
import time
from pathlib import Path

def test_pdf(pdf_path):
    print(f"Testing {pdf_path.name}...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {'strategy': 'speed', 'language': 'en'}
        
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/ocr/process",
                files=files,
                data=data,
                timeout=30
            )
            end = time.time()
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                pages = len(result.get('pages', []))
                regions = result.get('total_regions', 0)
                print(f"Success! {pages} pages, {regions} regions in {end-start:.2f}s")
                return True
            else:
                print(f"Error: {response.text[:200]}")
                return False
        except Exception as e:
            print(f"Exception: {e}")
            return False

# Test with original PDF
original = Path("tests/testpdfs/DOCC111697069_019511-00182.pdf")
if test_pdf(original):
    print("\nOriginal PDF works. Testing secondary PDF...")
    secondary = Path("tests/testpdfs/secondary/DOC710S730.pdf")
    test_pdf(secondary)
else:
    print("Server issue with original PDFs too")