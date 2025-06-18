"""Test PDF rendering performance at different DPIs"""

import time
from pathlib import Path
from src.utils.pdf_processor import PDFProcessor, PDFConfig

print("=== PDF DPI Performance Test ===\n")

# Initialize processor
config = PDFConfig(default_dpi=120, max_pages=1000)
processor = PDFProcessor(config)

# Load a real PDF for testing
pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in test directory!")
    exit(1)

# Pick a medium-sized PDF for testing
# Let's find one with multiple pages
test_pdf = None
for pdf_path in pdf_files:
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    metadata = processor.get_pdf_metadata(pdf_bytes)
    if metadata.page_count >= 5:  # Find a PDF with at least 5 pages
        test_pdf = pdf_path
        real_pdf_bytes = pdf_bytes
        break

if not test_pdf:
    # If no multi-page PDF found, use the first one
    test_pdf = pdf_files[0]
    with open(test_pdf, 'rb') as f:
        real_pdf_bytes = f.read()

# Get metadata
metadata = processor.get_pdf_metadata(real_pdf_bytes)
print(f"Test PDF: {test_pdf.name}")
print(f"Total pages: {metadata.page_count}")
print(f"File size: {len(real_pdf_bytes) / (1024*1024):.2f} MB")
print(f"Is scanned: {metadata.is_scanned}\n")

# Test rendering performance at different DPIs
print("Testing rendering performance at different DPIs:")
print("-" * 50)

dpis = [72, 120, 150, 200]
for dpi in dpis:
    start = time.time()
    pages = processor.extract_pages(real_pdf_bytes, dpi=dpi)
    elapsed = time.time() - start
    print(f"DPI {dpi}: {elapsed:.2f}s for {len(pages)} pages")
    if pages:
        print(f"  Page size: {pages[0].shape}")
        # Calculate memory usage for one page
        page_memory_mb = pages[0].nbytes / (1024 * 1024)
        total_memory_mb = page_memory_mb * len(pages)
        print(f"  Memory per page: {page_memory_mb:.2f} MB")
        print(f"  Total memory: {total_memory_mb:.2f} MB")
        print(f"  Pages/second: {len(pages)/elapsed:.2f}")
    print()

# Test with even higher DPIs for stress testing
print("\nStress test with higher DPIs (first page only):")
print("-" * 50)

high_dpis = [300, 400, 600]
for dpi in high_dpis:
    start = time.time()
    # Extract only first page to avoid memory issues
    pages = processor.extract_pages(real_pdf_bytes, dpi=dpi, page_range=(0, 1))
    elapsed = time.time() - start
    print(f"DPI {dpi}: {elapsed:.2f}s for {len(pages)} page(s)")
    if pages:
        print(f"  Page size: {pages[0].shape}")
        page_memory_mb = pages[0].nbytes / (1024 * 1024)
        print(f"  Memory per page: {page_memory_mb:.2f} MB")
    print()

# Performance comparison summary
print("\nPerformance Summary:")
print("-" * 50)
print("DPI scaling factors (relative to 72 DPI):")
for dpi in dpis:
    factor = dpi / 72
    print(f"  {dpi} DPI: {factor:.1f}x resolution, ~{factor**2:.1f}x memory")

print("\nRecommendations:")
print("- 72 DPI: Fast processing, low quality (draft mode)")
print("- 120 DPI: Good balance for most OCR tasks (recommended)")
print("- 150-200 DPI: Higher quality for detailed documents")
print("- 300+ DPI: Maximum quality but high memory usage")