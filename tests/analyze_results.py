#!/usr/bin/env python3
"""
Analyze OCR results and generate detailed statistics and visualizations
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any

def load_latest_results() -> Dict[str, Any]:
    """Load the most recent results file"""
    results_dir = Path(__file__).parent / "results"
    result_files = sorted(results_dir.glob("ocr_results_*.json"), reverse=True)
    
    if not result_files:
        print("No results files found!")
        sys.exit(1)
    
    latest_file = result_files[0]
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def analyze_performance(data: Dict[str, Any]) -> None:
    """Analyze performance metrics"""
    results = [r for r in data['detailed_results'] if r['status'] == 'success']
    
    if not results:
        print("No successful results to analyze")
        return
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Processing times
    print("\nProcessing Time Statistics (seconds):")
    print(f"  Mean OCR time: {df['ocr_time'].mean():.2f}")
    print(f"  Median OCR time: {df['ocr_time'].median():.2f}")
    print(f"  Std deviation: {df['ocr_time'].std():.2f}")
    print(f"  Min OCR time: {df['ocr_time'].min():.2f}")
    print(f"  Max OCR time: {df['ocr_time'].max():.2f}")
    
    # Pages analysis
    print(f"\nPages Statistics:")
    print(f"  Total pages: {df['pages'].sum()}")
    print(f"  Mean pages per PDF: {df['pages'].mean():.1f}")
    print(f"  Max pages in single PDF: {df['pages'].max()}")
    
    # Time per page
    df['time_per_page'] = df['ocr_time'] / df['pages']
    print(f"\nTime per Page Statistics (seconds):")
    print(f"  Mean: {df['time_per_page'].mean():.3f}")
    print(f"  Median: {df['time_per_page'].median():.3f}")
    print(f"  Min: {df['time_per_page'].min():.3f}")
    print(f"  Max: {df['time_per_page'].max():.3f}")
    
    # Throughput
    total_pages = df['pages'].sum()
    total_time = df['ocr_time'].sum()
    print(f"\nThroughput:")
    print(f"  Pages per second: {total_pages / total_time:.2f}")
    print(f"  PDFs per minute: {len(df) / (total_time / 60):.2f}")
    
    # File size impact
    print(f"\nFile Size Analysis:")
    print(f"  Total data processed: {df['file_size_mb'].sum():.1f} MB")
    print(f"  Mean file size: {df['file_size_mb'].mean():.2f} MB")
    df['mb_per_second'] = df['file_size_mb'] / df['ocr_time']
    print(f"  Processing speed: {df['mb_per_second'].mean():.2f} MB/s")

def analyze_quality(data: Dict[str, Any]) -> None:
    """Analyze OCR quality metrics"""
    results = [r for r in data['detailed_results'] if r['status'] == 'success']
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("QUALITY ANALYSIS")
    print("="*60)
    
    # Confidence statistics
    print("\nConfidence Score Distribution:")
    print(f"  Mean: {df['confidence'].mean():.1%}")
    print(f"  Median: {df['confidence'].median():.1%}")
    print(f"  Std deviation: {df['confidence'].std():.1%}")
    print(f"  25th percentile: {df['confidence'].quantile(0.25):.1%}")
    print(f"  75th percentile: {df['confidence'].quantile(0.75):.1%}")
    
    # Low confidence files
    low_conf_threshold = 0.95
    low_conf_files = df[df['confidence'] < low_conf_threshold]
    if len(low_conf_files) > 0:
        print(f"\nFiles with confidence < {low_conf_threshold:.0%}: {len(low_conf_files)}")
        for _, row in low_conf_files.iterrows():
            print(f"  - {row['filename']}: {row['confidence']:.1%}")
    
    # Regions analysis
    print(f"\nText Regions Statistics:")
    print(f"  Total regions detected: {df['regions'].sum()}")
    print(f"  Mean regions per page: {(df['regions'] / df['pages']).mean():.1f}")
    print(f"  Mean regions per PDF: {df['regions'].mean():.1f}")

def analyze_by_file_characteristics(data: Dict[str, Any]) -> None:
    """Analyze performance by file characteristics"""
    results = [r for r in data['detailed_results'] if r['status'] == 'success']
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("ANALYSIS BY FILE CHARACTERISTICS")
    print("="*60)
    
    # Group by page count
    page_bins = [0, 1, 5, 10, 20, 100]
    page_labels = ['1 page', '2-5 pages', '6-10 pages', '11-20 pages', '20+ pages']
    df['page_group'] = pd.cut(df['pages'], bins=page_bins, labels=page_labels)
    
    print("\nPerformance by Page Count:")
    page_groups = df.groupby('page_group')
    for group, data in page_groups:
        if len(data) > 0:
            avg_time = data['ocr_time'].mean()
            avg_conf = data['confidence'].mean()
            print(f"  {group}: {len(data)} files, {avg_time:.2f}s avg time, {avg_conf:.1%} avg confidence")
    
    # Analyze processing efficiency
    df['pages_per_second'] = df['pages'] / df['ocr_time']
    
    print("\nProcessing Efficiency by Document Size:")
    for group, data in page_groups:
        if len(data) > 0:
            avg_pps = data['pages_per_second'].mean()
            print(f"  {group}: {avg_pps:.2f} pages/second")

def generate_summary_report(data: Dict[str, Any]) -> None:
    """Generate a summary report"""
    print("\n" + "="*60)
    print("TENSORRT PERFORMANCE SUMMARY")
    print("="*60)
    
    analysis = data['analysis']
    metadata = data['metadata']
    
    print(f"\nTest Run Information:")
    print(f"  Timestamp: {metadata['timestamp']}")
    print(f"  Total processing time: {metadata['processing_time']:.1f} seconds")
    
    print(f"\nOverall Results:")
    print(f"  Success rate: {analysis['summary']['successful'] / analysis['summary']['total_pdfs']:.1%}")
    print(f"  Cache hit rate: {analysis['summary']['cache_hits'] / analysis['summary']['total_pdfs']:.1%}")
    
    print(f"\nTensorRT Performance Metrics:")
    print(f"  Average OCR speed: {analysis['performance']['pages_per_second']:.2f} pages/second")
    print(f"  Average latency per page: {analysis['performance']['avg_ocr_time_per_page']*1000:.0f} ms")
    print(f"  Total pages processed: {analysis['summary']['total_pages']}")
    print(f"  Total regions detected: {analysis['summary']['total_regions']}")
    
    print(f"\nQuality Metrics:")
    print(f"  Average confidence: {analysis['quality']['avg_confidence']:.1%}")
    print(f"  Confidence range: {analysis['quality']['min_confidence']:.1%} - {analysis['quality']['max_confidence']:.1%}")
    
    # Calculate estimated daily capacity
    pages_per_second = analysis['performance']['pages_per_second']
    daily_pages = pages_per_second * 86400  # seconds in a day
    print(f"\nEstimated Processing Capacity (24/7):")
    print(f"  Pages per hour: {pages_per_second * 3600:,.0f}")
    print(f"  Pages per day: {daily_pages:,.0f}")
    print(f"  PDFs per day (avg 5.5 pages): {daily_pages / 5.5:,.0f}")

def save_csv_summary(data: Dict[str, Any]) -> None:
    """Save a CSV summary of the results"""
    results = [r for r in data['detailed_results'] if r['status'] == 'success']
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Add calculated fields
    df['time_per_page'] = df['ocr_time'] / df['pages']
    df['pages_per_second'] = df['pages'] / df['ocr_time']
    df['regions_per_page'] = df['regions'] / df['pages']
    
    # Save to CSV
    csv_file = Path(__file__).parent / "results" / "ocr_results_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nCSV summary saved to: {csv_file}")

def main():
    """Main analysis function"""
    # Load results
    data = load_latest_results()
    
    # Run analyses
    analyze_performance(data)
    analyze_quality(data)
    analyze_by_file_characteristics(data)
    generate_summary_report(data)
    save_csv_summary(data)
    
    # Check for errors
    errors = [r for r in data['detailed_results'] if r['status'] == 'error']
    if errors:
        print(f"\n{len(errors)} ERRORS ENCOUNTERED:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error['filename']}: {error['error']}")

if __name__ == "__main__":
    main()