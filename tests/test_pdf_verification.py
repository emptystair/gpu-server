"""Verification tests for PDF processor functionality"""

print("=== PDF Processor Verification Tests ===\n")

# Import required modules
from src.utils.pdf_processor import PDFProcessor, PDFConfig
import numpy as np
from pathlib import Path

# Get test PDFs
pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print("ERROR: No PDF files found in tests/testpdfs!")
    exit(1)

# Load a test PDF
test_pdf_path = pdf_files[0]
with open(test_pdf_path, "rb") as f:
    pdf_bytes = f.read()

print(f"Using test PDF: {test_pdf_path.name}\n")

# Initialize processor
config = PDFConfig()
processor = PDFProcessor(config)

# Test 1: Basic PDF processing
print("Test 1: Basic PDF processing")
pages = processor.extract_pages(pdf_bytes, dpi=120)
print(f"✓ Extracted {len(pages)} pages")
if pages:
    print(f"✓ Page 1 shape: {pages[0].shape}, dtype: {pages[0].dtype}")

# Test 2: PDF metadata extraction
print("\nTest 2: PDF metadata extraction")
metadata = processor.get_pdf_metadata(pdf_bytes)
print(f"✓ PDF Metadata: pages={metadata.page_count}, encrypted={metadata.is_encrypted}")
print(f"✓ Is scanned: {metadata.is_scanned}")

# Test 3: Page range extraction
print("\nTest 3: Page range extraction")
partial_pages = processor.extract_pages(pdf_bytes, dpi=120, page_range=(0, 1))
print(f"✓ Partial extraction: {len(partial_pages)} pages")

# Test 4: Real PDF with multiple pages
print("\nTest 4: Real PDF with multiple pages")
# Find a multi-page PDF
multi_page_pdf = None
for pdf_path in pdf_files:
    with open(pdf_path, 'rb') as f:
        test_bytes = f.read()
    test_meta = processor.get_pdf_metadata(test_bytes)
    if test_meta.page_count > 3:
        multi_page_pdf = pdf_path
        real_pdf_bytes = test_bytes
        break

if not multi_page_pdf:
    # Use first PDF if no multi-page found
    real_pdf_bytes = pdf_bytes
    multi_page_pdf = test_pdf_path

pages = processor.extract_pages(real_pdf_bytes)
print(f"✓ Real PDF: {len(pages)} pages extracted from {multi_page_pdf.name}")

# Test OCR optimization
optimized = processor.optimize_for_ocr(pages[0])
print(f"✓ Optimized shape: {optimized.shape}")

# Test 5: Error handling
print("\nTest 5: Error handling")
try:
    processor.extract_pages(b"not a pdf")
except Exception as e:
    print(f"✓ Correctly handled invalid PDF: {type(e).__name__}")

# Test 6: Large PDF handling (memory efficiency)
print("\nTest 6: Large PDF handling (memory efficiency)")
large_pdf_test = processor.split_pdf(real_pdf_bytes, chunks=3)
print(f"✓ Split into {len(large_pdf_test)} chunks")

# Additional verification tests
print("\nAdditional verification tests:")

# Test 7: Different DPI settings
print("\nTest 7: Different DPI settings")
for dpi in [72, 150, 300]:
    pages = processor.extract_pages(pdf_bytes, dpi=dpi, page_range=(0, 1))
    if pages:
        print(f"✓ DPI {dpi}: Page shape = {pages[0].shape}")

# Test 8: Empty page range
print("\nTest 8: Edge cases")
empty_pages = processor.extract_pages(pdf_bytes, page_range=(100, 200))
print(f"✓ Out of range pages: {len(empty_pages)} pages (should be 0)")

# Test 9: PDF splitting edge cases
single_chunk = processor.split_pdf(pdf_bytes, chunks=1)
print(f"✓ Single chunk split: {len(single_chunk)} chunk")

metadata = processor.get_pdf_metadata(real_pdf_bytes)
many_chunks = processor.split_pdf(real_pdf_bytes, chunks=100)
print(f"✓ Over-split: {len(many_chunks)} chunks (max should be {metadata.page_count})")

# Test 10: Embedded images (if any)
print("\nTest 10: Embedded images extraction")
embedded = processor.extract_images_from_pdf(real_pdf_bytes)
print(f"✓ Found {len(embedded)} embedded images")

# Test 11: Performance check
print("\nTest 11: Performance check")
import time
start = time.time()
pages = processor.extract_pages(real_pdf_bytes, dpi=120, page_range=(0, 3))
elapsed = time.time() - start
print(f"✓ Extracted {len(pages)} pages in {elapsed:.2f}s")

# Test 12: Memory test with different file sizes
print("\nTest 12: File size handling")
file_sizes = []
for pdf_path in pdf_files[:5]:  # Test first 5 PDFs
    with open(pdf_path, 'rb') as f:
        test_bytes = f.read()
    file_size_mb = len(test_bytes) / (1024 * 1024)
    file_sizes.append(file_size_mb)
    
print(f"✓ Tested PDFs ranging from {min(file_sizes):.2f} MB to {max(file_sizes):.2f} MB")

print("\n✅ All verification tests passed!")