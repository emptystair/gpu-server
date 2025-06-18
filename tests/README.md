# GPU OCR Server Tests

This directory contains all test files and test results for the GPU OCR Server project.

## Test Organization

### Test Scripts
- **Configuration Tests**: `test_gpu_monitor.py`, `test_gpu_monitor_advanced.py`
- **Image Processing Tests**: `test_image_processor.py`, `test_image_processor_real.py`
- **PDF Processing Tests**: `test_pdf_processor.py`, `test_pdf_real_documents.py`, `test_pdf_dpi_performance.py`, `test_pdf_verification.py`
- **API Tests**: `test_api.py`, `test_ocr_service.py` (placeholders)

### Test Data
- **Test PDFs**: `testpdfs/` - Contains 96 real scanned PDF documents for testing
- **Test Outputs**: 
  - `test_original.png` - Sample original image
  - `test_enhanced.png` - Sample enhanced image after processing
  - `test_pdf_page.png` - Sample PDF page extraction
  - `pdf_test_outputs/` - Sample PDF page extractions

### Test Results
- **Summary**: `test_results_summary.md` - Comprehensive test results and performance metrics
- **Quick Summary**: `test_pdf_summary.py` - Script to display test summary

## Running Tests

### Individual Component Tests
```bash
# Activate virtual environment
source ../venv/bin/activate

# Test GPU monitoring
python test_gpu_monitor.py

# Test image processing
python test_image_processor.py
python test_image_processor_real.py

# Test PDF processing
python test_pdf_processor.py
python test_pdf_real_documents.py
python test_pdf_dpi_performance.py
python test_pdf_verification.py
```

### Performance Tests
```bash
# DPI performance comparison
python test_pdf_dpi_performance.py

# Real document processing
python test_pdf_real_documents.py
```

## Test Results Summary

All implemented components have been thoroughly tested and are working correctly:

- ✅ Configuration system with RTX 4090 optimizations
- ✅ GPU monitoring with real-time metrics
- ✅ Image processing with OCR enhancement
- ✅ PDF processing with configurable DPI
- ✅ Error handling and edge cases
- ✅ Performance optimization for batch processing

See `test_results_summary.md` for detailed results and performance metrics.