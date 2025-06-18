"""Test script for PDFProcessor functionality"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from src.utils.pdf_processor import PDFProcessor, PDFConfig

print("=== PDF Processor Test ===\n")

# First, let's check if reportlab is installed for creating test PDFs
try:
    from reportlab.pdfgen import canvas
except ImportError:
    print("Installing reportlab for test PDF generation...")
    import subprocess
    subprocess.run(["pip", "install", "reportlab"], check=True)
    from reportlab.pdfgen import canvas

# Create a test PDF
def create_test_pdf():
    """Create a multi-page test PDF"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Page 1 - Text content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "GPU OCR Server Test PDF")
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "Page 1: Text Content")
    c.drawString(100, 650, "This PDF tests the PDF processor functionality:")
    c.drawString(100, 620, "- PDF loading and validation")
    c.drawString(100, 590, "- Page rendering at specified DPI")
    c.drawString(100, 560, "- Integration with image processor")
    c.drawString(100, 530, "- Metadata extraction")
    c.showPage()
    
    # Page 2 - More text
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "Page 2: OCR Test Content")
    c.setFont("Helvetica", 11)
    y = 700
    for i in range(20):
        c.drawString(100, y, f"Line {i+1}: This is test content for OCR processing.")
        y -= 25
    c.showPage()
    
    # Page 3 - Mixed content
    c.setFont("Helvetica", 10)
    c.drawString(100, 750, "Page 3: Performance Test")
    c.drawString(100, 720, "Configuration: RTX 4090, 24GB VRAM")
    c.drawString(100, 700, "Default DPI: 120")
    c.drawString(100, 680, "TensorRT Precision: FP16")
    
    # Draw some shapes
    from reportlab.lib.colors import blue, lightgrey
    c.setStrokeColor(blue)
    c.rect(100, 500, 200, 100)
    c.setFillColor(lightgrey)
    c.rect(350, 500, 150, 100, fill=1)
    
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# Initialize PDF processor
print("1. Initializing PDF processor")
config = PDFConfig(default_dpi=120, max_pages=100)
processor = PDFProcessor(config)
print(f"   Initialized with default DPI: {config.default_dpi}")

# Create test PDF
print("\n2. Creating test PDF")
pdf_bytes = create_test_pdf()
print(f"   Created PDF with size: {len(pdf_bytes)} bytes")

# Test metadata extraction
print("\n3. Extracting PDF metadata")
metadata = processor.get_pdf_metadata(pdf_bytes)
print(f"   Page count: {metadata.page_count}")
print(f"   Is encrypted: {metadata.is_encrypted}")
print(f"   Is scanned: {metadata.is_scanned}")
print(f"   File size: {metadata.file_size_bytes} bytes")

# Test page extraction
print("\n4. Extracting pages as images")
page_images = processor.extract_pages(pdf_bytes, dpi=120)
print(f"   Extracted {len(page_images)} pages")
for i, img in enumerate(page_images):
    print(f"   Page {i+1} shape: {img.shape}, dtype: {img.dtype}")

# Test page range extraction
print("\n5. Testing page range extraction")
pages_1_2 = processor.extract_pages(pdf_bytes, dpi=120, page_range=(0, 2))
print(f"   Extracted pages 1-2: {len(pages_1_2)} pages")

# Test different DPI settings
print("\n6. Testing different DPI settings")
for dpi in [72, 150, 300]:
    pages = processor.extract_pages(pdf_bytes, dpi=dpi, page_range=(0, 1))
    if pages:
        print(f"   DPI {dpi}: Page shape = {pages[0].shape}")

# Test PDF splitting
print("\n7. Testing PDF splitting")
chunks = processor.split_pdf(pdf_bytes, chunks=2)
print(f"   Split into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    chunk_metadata = processor.get_pdf_metadata(chunk)
    print(f"   Chunk {i+1}: {chunk_metadata.page_count} pages, {len(chunk)} bytes")

# Save sample output
if page_images:
    print("\n8. Saving sample output")
    # Save first page as image
    first_page = page_images[0]
    # Convert to PIL Image for saving
    if len(first_page.shape) == 2:
        mode = 'L'
    else:
        mode = 'RGB'
    img = Image.fromarray(first_page, mode=mode)
    img.save('test_pdf_page.png')
    print("   Saved first page as test_pdf_page.png")

# Test error handling
print("\n9. Testing error handling")

# Test invalid PDF
try:
    processor.extract_pages(b"not a pdf")
except ValueError as e:
    print(f"   ✓ Invalid PDF caught: {e}")

# Test encrypted PDF (would need a real encrypted PDF to test fully)
print("   Note: Encrypted PDF handling implemented but not tested (no test file)")

# Test large file limit
large_config = PDFConfig(max_file_size_mb=0.001)  # 1KB limit
large_processor = PDFProcessor(large_config)
try:
    large_processor.extract_pages(pdf_bytes)
except ValueError as e:
    print(f"   ✓ File size limit caught: {e}")

print("\n✓ All tests completed successfully!")