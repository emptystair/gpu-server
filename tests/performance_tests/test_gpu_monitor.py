#!/usr/bin/env python3
"""Monitor GPU usage while processing PDFs to verify actual processing"""

import requests
import time
import os
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# Global for GPU monitoring
gpu_stats = []
monitoring = True

def monitor_gpu():
    """Monitor GPU usage in background"""
    global monitoring
    while monitoring:
        try:
            result = subprocess.run(['docker', 'exec', 'gpu-ocr-server', 'nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used = result.stdout.strip().split(', ')
                gpu_stats.append({
                    'time': time.time(),
                    'gpu_util': int(gpu_util),
                    'mem_used': int(mem_used)
                })
        except:
            pass
        time.sleep(0.1)  # Monitor every 100ms

def process_pdf(pdf_path):
    """Process a single PDF"""
    # Create unique request
    request_id = f"gpu_test_{time.time()}"
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {
            'strategy': 'speed',
            'language': 'en',
            'request_id': request_id
        }
        
        start = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/ocr/process",
            files=files,
            data=data,
            timeout=120
        )
        end = time.time()
    
    if response.status_code == 200:
        result = response.json()
        return {
            'success': True,
            'time': end - start,
            'pages': len(result.get('pages', [])),
            'file': pdf_path.name
        }
    else:
        return {'success': False, 'file': pdf_path.name}

def main():
    global monitoring
    
    # Get test PDFs
    pdf_dir = Path("tests/testpdfs")
    test_pdfs = sorted(pdf_dir.glob("*.pdf"))[:20]  # Test first 20
    
    print("GPU Usage Monitoring Test - Processing 20 PDFs")
    print("=" * 70)
    
    # Start GPU monitoring
    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()
    
    # Record baseline
    print("Recording baseline GPU usage (5 seconds)...")
    time.sleep(5)
    
    baseline_gpu = sum(s['gpu_util'] for s in gpu_stats[-10:]) / min(10, len(gpu_stats)) if gpu_stats else 0
    baseline_mem = sum(s['mem_used'] for s in gpu_stats[-10:]) / min(10, len(gpu_stats)) if gpu_stats else 0
    
    print(f"Baseline: GPU {baseline_gpu:.1f}%, Memory {baseline_mem:.0f} MB")
    print("\nProcessing PDFs...")
    
    # Process PDFs
    process_start = time.time()
    results = []
    
    for i, pdf in enumerate(test_pdfs, 1):
        print(f"\n[{i}/20] Processing {pdf.name}...", end='', flush=True)
        result = process_pdf(pdf)
        results.append(result)
        
        if result['success']:
            print(f" {result['pages']} pages in {result['time']:.2f}s")
        else:
            print(" Failed")
        
        # Brief pause between files
        time.sleep(0.5)
    
    process_end = time.time()
    
    # Stop monitoring
    print("\nProcessing complete. Recording final baseline...")
    time.sleep(3)
    monitoring = False
    monitor_thread.join()
    
    # Analyze results
    successful = [r for r in results if r['success']]
    total_pages = sum(r['pages'] for r in successful)
    total_time = process_end - process_start
    
    # Find peak GPU usage during processing
    process_stats = [s for s in gpu_stats if process_start <= s['time'] <= process_end]
    if process_stats:
        peak_gpu = max(s['gpu_util'] for s in process_stats)
        avg_gpu = sum(s['gpu_util'] for s in process_stats) / len(process_stats)
        peak_mem = max(s['mem_used'] for s in process_stats)
        avg_mem = sum(s['mem_used'] for s in process_stats) / len(process_stats)
    else:
        peak_gpu = avg_gpu = peak_mem = avg_mem = 0
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Documents processed: {len(successful)}/{len(test_pdfs)}")
    print(f"Total pages: {total_pages}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Docs/min: {len(successful) / total_time * 60:.1f}")
    print(f"Pages/min: {total_pages / total_time * 60:.1f}")
    
    print(f"\nGPU Usage During Processing:")
    print(f"  Peak GPU:    {peak_gpu}% (baseline: {baseline_gpu:.1f}%)")
    print(f"  Average GPU: {avg_gpu:.1f}% (baseline: {baseline_gpu:.1f}%)")
    print(f"  Peak Memory: {peak_mem} MB")
    print(f"  Avg Memory:  {avg_mem:.0f} MB")
    
    if peak_gpu > baseline_gpu + 5:
        print(f"\n✓ GPU usage increased by {peak_gpu - baseline_gpu:.0f}% - actual processing confirmed!")
    else:
        print(f"\n⚠️ No significant GPU usage increase detected - possible caching!")

if __name__ == "__main__":
    main()