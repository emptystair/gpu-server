#!/usr/bin/env python3
"""
Verify test setup and dependencies for PDF OCR testing
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    dependencies = {
        'paddleocr': 'PaddleOCR',
        'paddlepaddle': 'PaddlePaddle',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pynvml': 'PyNVML (GPU monitoring)',
        'aiohttp': 'aiohttp (for API tests)',
        'pymupdf': 'PyMuPDF (PDF processing)'
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Missing")
            missing.append(module)
    
    return missing

def check_pdf_folder():
    """Check if PDF test folder exists and has files"""
    pdf_folder = Path('/home/ryanb/Projects/gpu-server0.1/tests/testpdfs')
    
    print(f"\nChecking PDF test folder: {pdf_folder}")
    
    if not pdf_folder.exists():
        print("  ✗ Folder does not exist")
        return False
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"  ✓ Found {len(pdf_files)} PDF files")
    
    if pdf_files:
        # Show first 5 files
        print("  Sample files:")
        for pdf in pdf_files[:5]:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"    - {pdf.name} ({size_mb:.2f} MB)")
    
    return len(pdf_files) > 0

def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        print(f"  ✓ Found {device_count} GPU(s)")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_mb = memory_info.total / (1024 * 1024)
            memory_used_mb = memory_info.used / (1024 * 1024)
            
            print(f"    GPU {i}: {name}")
            print(f"    Memory: {memory_used_mb:.0f}/{memory_total_mb:.0f} MB used")
            
        pynvml.nvmlShutdown()
        return True
        
    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")
        return False

def check_ocr_service():
    """Try to import OCR service"""
    print("\nChecking OCR service...")
    
    try:
        from src.ocr_service import OCRService
        from src.config import load_config
        print("  ✓ OCR service can be imported")
        
        # Try to load config
        config = load_config()
        print("  ✓ Configuration loaded")
        
        return True
    except Exception as e:
        print(f"  ✗ OCR service check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("=== GPU OCR Server Test Setup Verification ===\n")
    
    all_good = True
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        all_good = False
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
    
    # Check PDF folder
    if not check_pdf_folder():
        all_good = False
    
    # Check GPU
    if not check_gpu():
        all_good = False
        print("\nWarning: GPU not available. Tests may run slowly or fail.")
    
    # Check OCR service
    if not check_ocr_service():
        all_good = False
    
    # Summary
    print("\n=== Verification Summary ===")
    if all_good:
        print("✓ All checks passed! Ready to run tests.")
        print("\nTo run tests:")
        print("  1. Direct OCR tests: python tests/test_pdf_comprehensive.py")
        print("  2. API tests: python tests/test_api_pdf_processing.py")
        print("  3. All tests: ./tests/run_pdf_tests.sh --test-type all")
    else:
        print("✗ Some checks failed. Please fix the issues above before running tests.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())