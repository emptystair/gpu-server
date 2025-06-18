"""Test PDFProcessor with real PDF documents"""

import os
import glob
import time
import statistics
from pathlib import Path
from src.utils.pdf_processor import PDFProcessor, PDFConfig
import numpy as np
from PIL import Image

print("=== Real PDF Document Processing Test ===\n")

# Initialize processor
config = PDFConfig(default_dpi=120, max_pages=1000)
processor = PDFProcessor(config)

# Get all PDF files
pdf_dir = Path("/home/ryanb/Projects/gpu-server0.1/tests/testpdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files to process\n")

# Process a sample of PDFs
sample_size = min(10, len(pdf_files))  # Process first 10 PDFs
processing_times = []
page_counts = []
file_sizes = []
scanned_count = 0
text_count = 0
errors = []

print(f"Processing {sample_size} sample PDFs...\n")

for i, pdf_path in enumerate(pdf_files[:sample_size]):
    print(f"{i+1}/{sample_size}: Processing {pdf_path.name}")
    
    try:
        # Read PDF bytes
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        file_sizes.append(file_size_mb)
        
        # Get metadata
        start_time = time.time()
        metadata = processor.get_pdf_metadata(pdf_bytes)
        
        print(f"  - Pages: {metadata.page_count}")
        print(f"  - Size: {file_size_mb:.2f} MB")
        print(f"  - Scanned: {metadata.is_scanned}")
        print(f"  - Encrypted: {metadata.is_encrypted}")
        
        page_counts.append(metadata.page_count)
        if metadata.is_scanned:
            scanned_count += 1
        else:
            text_count += 1
        
        # Extract first page as test
        if metadata.page_count > 0:
            pages = processor.extract_pages(pdf_bytes, dpi=120, page_range=(0, 1))
            if pages:
                print(f"  - First page shape: {pages[0].shape}")
                print(f"  - First page dtype: {pages[0].dtype}")
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        print(f"  - Processing time: {processing_time:.2f}s\n")
        
    except Exception as e:
        print(f"  - ERROR: {e}\n")
        errors.append((pdf_path.name, str(e)))

# Summary statistics
print("\n=== Processing Summary ===")
print(f"Total PDFs processed: {sample_size}")
print(f"Successful: {sample_size - len(errors)}")
print(f"Errors: {len(errors)}")

if page_counts:
    print(f"\nPage statistics:")
    print(f"  - Total pages: {sum(page_counts)}")
    print(f"  - Average pages per PDF: {statistics.mean(page_counts):.1f}")
    print(f"  - Min pages: {min(page_counts)}")
    print(f"  - Max pages: {max(page_counts)}")

if file_sizes:
    print(f"\nFile size statistics:")
    print(f"  - Average size: {statistics.mean(file_sizes):.2f} MB")
    print(f"  - Min size: {min(file_sizes):.2f} MB")
    print(f"  - Max size: {max(file_sizes):.2f} MB")

print(f"\nDocument types:")
print(f"  - Scanned PDFs: {scanned_count}")
print(f"  - Text PDFs: {text_count}")

if processing_times:
    print(f"\nPerformance:")
    print(f"  - Average processing time: {statistics.mean(processing_times):.2f}s")
    print(f"  - Total processing time: {sum(processing_times):.2f}s")

if errors:
    print(f"\nErrors encountered:")
    for filename, error in errors:
        print(f"  - {filename}: {error}")

# Test batch processing with multiple PDFs
print("\n=== Batch Processing Test ===")
if len(pdf_files) >= 3:
    print("Testing extraction of multiple PDFs in sequence...")
    batch_start = time.time()
    
    for pdf_path in pdf_files[:3]:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        pages = processor.extract_pages(pdf_bytes, dpi=120, page_range=(0, 1))
        print(f"  - {pdf_path.name}: Extracted {len(pages)} page(s)")
    
    batch_time = time.time() - batch_start
    print(f"  - Batch processing time: {batch_time:.2f}s")

# Test different DPI settings on a single PDF
if pdf_files:
    print("\n=== DPI Comparison Test ===")
    test_pdf = pdf_files[0]
    with open(test_pdf, 'rb') as f:
        pdf_bytes = f.read()
    
    print(f"Testing different DPI settings on {test_pdf.name}:")
    for dpi in [72, 120, 150, 300]:
        try:
            pages = processor.extract_pages(pdf_bytes, dpi=dpi, page_range=(0, 1))
            if pages:
                print(f"  - DPI {dpi}: Shape = {pages[0].shape}")
        except Exception as e:
            print(f"  - DPI {dpi}: Error - {e}")

# Save sample outputs
print("\n=== Saving Sample Outputs ===")
output_dir = Path("pdf_test_outputs")
output_dir.mkdir(exist_ok=True)

# Process and save first page of first 3 PDFs
for i, pdf_path in enumerate(pdf_files[:3]):
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        pages = processor.extract_pages(pdf_bytes, dpi=120, page_range=(0, 1))
        if pages:
            # Save as image
            page_array = pages[0]
            if len(page_array.shape) == 2:
                img = Image.fromarray(page_array, mode='L')
            else:
                img = Image.fromarray(page_array, mode='RGB')
            
            output_path = output_dir / f"sample_{i+1}_{pdf_path.stem}.png"
            img.save(output_path)
            print(f"  - Saved: {output_path.name}")
    except Exception as e:
        print(f"  - Error saving {pdf_path.name}: {e}")

print("\n✓ Real document testing completed!")