#!/usr/bin/env python3
"""
Simple batch processing test for GPU OCR server
Tests batch sizes without complex concurrency to isolate batch performance
"""

import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Any

# Configuration
SERVER_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"
TEST_PDFS_DIR = Path(__file__).parent / "testpdfs"

# Test configurations - start small
BATCH_SIZES = [1, 2, 5, 10]
MAX_PDFS = 10  # Limit to avoid overwhelming the server


def wait_for_server(timeout: int = 60) -> bool:
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{SERVER_URL}{API_PREFIX}/health", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(2)
    
    return False


def test_single_pdf(pdf_path: Path, dpi: int = 120) -> Dict[str, Any]:
    """Test processing a single PDF"""
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            data = {'dpi': str(dpi)}
            
            response = requests.post(
                f"{SERVER_URL}{API_PREFIX}/ocr/pdf", 
                files=files, 
                data=data,
                timeout=120  # 2 minute timeout
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "file": pdf_path.name,
                    "status": "success",
                    "processing_time": result.get("processing_time", 0),
                    "total_time": (end_time - start_time) * 1000,
                    "pages": result.get("pages", 0),
                    "confidence": result.get("average_confidence", 0)
                }
            else:
                return {
                    "file": pdf_path.name,
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "total_time": (end_time - start_time) * 1000
                }
                
    except Exception as e:
        return {
            "file": pdf_path.name,
            "status": "error",
            "error": str(e),
            "total_time": (time.time() - start_time) * 1000
        }


def test_batch_processing(pdf_paths: List[Path], batch_size: int) -> Dict[str, Any]:
    """Test batch processing using the batch endpoint"""
    print(f"\nTesting batch size: {batch_size} with {len(pdf_paths)} PDFs")
    
    results = []
    start_time = time.time()
    
    # Process PDFs in batches
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i + batch_size]
        batch_start = time.time()
        
        # For now, process sequentially since batch endpoint might not be available
        batch_results = []
        for pdf_path in batch:
            result = test_single_pdf(pdf_path)
            batch_results.append(result)
            print(f"  Processed: {result['file']} - {result['status']}")
        
        batch_time = (time.time() - batch_start) * 1000
        results.append({
            "batch_size": len(batch),
            "batch_time": batch_time,
            "results": batch_results
        })
    
    total_time = time.time() - start_time
    successful = sum(1 for batch in results for r in batch["results"] if r["status"] == "success")
    
    return {
        "batch_size": batch_size,
        "total_pdfs": len(pdf_paths),
        "successful": successful,
        "failed": len(pdf_paths) - successful,
        "total_time": total_time,
        "throughput": successful / total_time if total_time > 0 else 0,
        "batches": results
    }


def main():
    """Run batch processing tests"""
    print("Starting batch processing tests...")
    
    # Wait for server
    if not wait_for_server():
        print("Server is not ready. Exiting.")
        return
    
    # Get test PDFs
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))[:MAX_PDFS]
    if not pdf_files:
        print("No PDF files found for testing!")
        return
    
    print(f"Found {len(pdf_files)} PDF files for testing")
    
    # Run tests for different batch sizes
    results = {
        "test_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": SERVER_URL,
            "total_pdfs": len(pdf_files)
        },
        "batch_tests": []
    }
    
    for batch_size in BATCH_SIZES:
        print(f"\n{'='*60}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*60}")
        
        test_result = test_batch_processing(pdf_files, batch_size)
        results["batch_tests"].append(test_result)
        
        print(f"\nBatch size {batch_size} results:")
        print(f"  Total time: {test_result['total_time']:.2f}s")
        print(f"  Successful: {test_result['successful']}/{test_result['total_pdfs']}")
        print(f"  Throughput: {test_result['throughput']:.2f} PDFs/second")
        
        # Small delay between tests
        time.sleep(2)
    
    # Save results
    output_file = Path(__file__).parent / "simple_batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test in results["batch_tests"]:
        print(f"Batch size {test['batch_size']:2d}: "
              f"{test['throughput']:6.2f} PDFs/sec "
              f"({test['successful']:2d}/{test['total_pdfs']:2d} successful)")


if __name__ == "__main__":
    main()