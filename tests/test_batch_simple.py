#!/usr/bin/env python3
"""
Simple batch processing test with detailed logging
"""

import asyncio
import time
from pathlib import Path
import httpx

API_BASE_URL = "http://localhost:8000/api/v1"
TESTPDFS_DIR = Path(__file__).parent / "testpdfs"

async def test_single_pdf():
    """Test processing a single PDF"""
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))[:1]
    if not pdf_files:
        print("No PDFs found")
        return
    
    pdf_path = pdf_files[0]
    print(f"\nTesting single PDF: {pdf_path.name}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        start = time.time()
        response = await client.post(
            f"{API_BASE_URL}/ocr/pdf",
            files={'file': (pdf_path.name, content, 'application/pdf')},
            data={
                'language': 'en',
                'output_format': 'json',
                'confidence_threshold': '0.5',
                'merge_pages': 'true'
            }
        )
        elapsed = time.time() - start
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Pages: {len(result.get('pages', []))}")
            print(f"Processing time: {result.get('processing_time', 0)/1000:.2f}s")
            print(f"Confidence: {result.get('average_confidence', 0):.1%}")
        else:
            print(f"Error: {response.text[:200]}")


async def test_concurrent_pdfs(count=5):
    """Test concurrent PDF processing"""
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))[:count]
    if not pdf_files:
        print("No PDFs found")
        return
    
    print(f"\nTesting {len(pdf_files)} concurrent PDFs...")
    
    async def process_pdf(client, pdf_path):
        with open(pdf_path, 'rb') as f:
            content = f.read()
        
        start = time.time()
        try:
            response = await client.post(
                f"{API_BASE_URL}/ocr/pdf",
                files={'file': (pdf_path.name, content, 'application/pdf')},
                data={
                    'language': 'en',
                    'output_format': 'json',
                    'confidence_threshold': '0.5',
                    'merge_pages': 'true'
                },
                timeout=120.0
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                pages = len(result.get('pages', []))
                print(f"✓ {pdf_path.name}: {pages} pages in {elapsed:.2f}s")
                return pages, elapsed
            else:
                print(f"✗ {pdf_path.name}: HTTP {response.status_code}")
                return 0, elapsed
        except Exception as e:
            print(f"✗ {pdf_path.name}: {str(e)}")
            return 0, time.time() - start
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        tasks = [process_pdf(client, pdf) for pdf in pdf_files]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        total_pages = sum(r[0] for r in results)
        
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Total pages: {total_pages}")
        print(f"Pages per second: {total_pages/total_time:.2f}")


async def test_batch_endpoint(count=5):
    """Test batch endpoint"""
    pdf_files = sorted(TESTPDFS_DIR.glob("*.pdf"))[:count]
    if not pdf_files:
        print("No PDFs found")
        return
    
    print(f"\nTesting batch endpoint with {len(pdf_files)} PDFs...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Prepare files
        files = []
        for pdf_path in pdf_files:
            with open(pdf_path, 'rb') as f:
                content = f.read()
            files.append(('files', (pdf_path.name, content, 'application/pdf')))
        
        start = time.time()
        response = await client.post(
            f"{API_BASE_URL}/ocr/process-batch",
            files=files,
            data={
                'strategy': 'speed',
                'language': 'en',
                'parallel_processing': 'true',
                'confidence_threshold': '0.5'
            }
        )
        elapsed = time.time() - start
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Successful: {result.get('successful', 0)}")
            print(f"Failed: {result.get('failed', 0)}")
            
            total_pages = sum(doc.get('pages', 0) for doc in result.get('documents', []))
            print(f"Total pages: {total_pages}")
            print(f"Pages per second: {total_pages/elapsed:.2f}")
        else:
            print(f"Error: {response.text[:500]}")


async def main():
    print("GPU OCR Batch Testing")
    print("=" * 50)
    
    # Test 1: Single PDF
    await test_single_pdf()
    
    await asyncio.sleep(2)
    
    # Test 2: 5 concurrent PDFs
    await test_concurrent_pdfs(5)
    
    await asyncio.sleep(2)
    
    # Test 3: Batch endpoint with 5 PDFs
    await test_batch_endpoint(5)


if __name__ == "__main__":
    asyncio.run(main())