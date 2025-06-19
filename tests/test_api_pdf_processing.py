#!/usr/bin/env python3
"""
API-based PDF testing script for GPU OCR Server.
Tests PDF processing through the REST API endpoints.
"""

import os
import time
import json
import asyncio
import aiohttp
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class APITestRunner:
    """Test PDFs through the API endpoints"""
    
    def __init__(self, base_url: str, pdf_folder: str):
        self.base_url = base_url.rstrip('/')
        self.pdf_folder = Path(pdf_folder)
        self.session = None
        self.results = []
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        
        # Check server health
        try:
            async with self.session.get(f"{self.base_url}/api/v1/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    self.logger.info(f"Server is {health['status']}")
                else:
                    raise Exception(f"Health check failed: {resp.status}")
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            raise
            
    async def test_sync_processing(self, pdf_path: Path, strategy: str = "balanced") -> Dict[str, Any]:
        """Test synchronous PDF processing"""
        start_time = time.time()
        result = {
            'filename': pdf_path.name,
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'strategy': strategy,
            'endpoint': '/api/v1/ocr/pdf',
            'status': 'pending',
            'error': None,
            'processing_time': 0
        }
        
        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('file', open(pdf_path, 'rb'), 
                         filename=pdf_path.name,
                         content_type='application/pdf')
            data.add_field('language', 'en')
            data.add_field('output_format', 'json')
            data.add_field('merge_pages', 'true')
            data.add_field('confidence_threshold', '0.5')
            
            # Send request
            self.logger.info(f"Processing {pdf_path.name} via {result['endpoint']}...")
            async with self.session.post(
                f"{self.base_url}{result['endpoint']}", 
                data=data
            ) as resp:
                result['status_code'] = resp.status
                
                if resp.status == 200:
                    response_data = await resp.json()
                    result['status'] = 'success'
                    result['request_id'] = response_data.get('request_id')
                    result['pages'] = len(response_data.get('pages', []))
                    result['confidence'] = response_data.get('average_confidence', 0)
                    result['processing_time_ms'] = response_data.get('processing_time', 0)
                    self.logger.info(f"✓ Processed {pdf_path.name}: {result['pages']} pages")
                else:
                    error_text = await resp.text()
                    result['status'] = 'failed'
                    result['error'] = f"HTTP {resp.status}: {error_text}"
                    self.logger.error(f"✗ Failed {pdf_path.name}: {result['error']}")
                    
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"✗ Exception processing {pdf_path.name}: {e}")
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    async def test_async_processing(self, pdf_path: Path, strategy: str = "balanced") -> Dict[str, Any]:
        """Test asynchronous PDF processing"""
        start_time = time.time()
        result = {
            'filename': pdf_path.name,
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'strategy': strategy,
            'endpoint': '/api/v1/jobs/submit',
            'status': 'pending',
            'error': None,
            'processing_time': 0
        }
        
        try:
            # Submit async job
            data = aiohttp.FormData()
            data.add_field('file', open(pdf_path, 'rb'), 
                         filename=pdf_path.name,
                         content_type='application/pdf')
            data.add_field('language', 'en')
            data.add_field('output_format', 'json')
            data.add_field('strategy', strategy)
            
            self.logger.info(f"Submitting async job for {pdf_path.name}...")
            async with self.session.post(
                f"{self.base_url}{result['endpoint']}", 
                data=data
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"Submit failed: HTTP {resp.status} - {error_text}")
                    
                job_info = await resp.json()
                job_id = job_info['job_id']
                result['job_id'] = job_id
                self.logger.info(f"Job submitted: {job_id}")
            
            # Poll for completion
            max_wait = 60  # seconds
            poll_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                async with self.session.get(
                    f"{self.base_url}/api/v1/jobs/{job_id}/status"
                ) as resp:
                    if resp.status == 200:
                        status_data = await resp.json()
                        job_status = status_data['status']
                        
                        if job_status == 'completed':
                            # Get results
                            async with self.session.get(
                                f"{self.base_url}/api/v1/jobs/{job_id}/result"
                            ) as result_resp:
                                if result_resp.status == 200:
                                    ocr_result = await result_resp.json()
                                    result['status'] = 'success'
                                    result['pages'] = len(ocr_result.get('pages', []))
                                    result['confidence'] = ocr_result.get('average_confidence', 0)
                                    self.logger.info(f"✓ Async processed {pdf_path.name}")
                                    break
                                    
                        elif job_status == 'failed':
                            result['status'] = 'failed'
                            result['error'] = status_data.get('error', 'Job failed')
                            self.logger.error(f"✗ Job failed for {pdf_path.name}")
                            break
                            
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
            if elapsed >= max_wait:
                result['status'] = 'timeout'
                result['error'] = f'Job did not complete within {max_wait}s'
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"✗ Exception in async processing: {e}")
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    async def test_batch_processing(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """Test batch processing endpoint"""
        self.logger.info(f"Testing batch processing with {len(pdf_files)} files...")
        start_time = time.time()
        
        result = {
            'endpoint': '/api/v1/ocr/process-batch',
            'file_count': len(pdf_files),
            'status': 'pending',
            'error': None,
            'processing_time': 0
        }
        
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            
            for pdf_path in pdf_files:
                data.add_field('files', open(pdf_path, 'rb'),
                             filename=pdf_path.name,
                             content_type='application/pdf')
            
            data.add_field('strategy', 'balanced')
            data.add_field('language', 'en')
            data.add_field('parallel_processing', 'true')
            
            async with self.session.post(
                f"{self.base_url}{result['endpoint']}", 
                data=data
            ) as resp:
                result['status_code'] = resp.status
                
                if resp.status == 200:
                    batch_result = await resp.json()
                    result['status'] = 'success'
                    result['successful'] = batch_result.get('successful', 0)
                    result['failed'] = batch_result.get('failed', 0)
                    result['total_processing_time'] = batch_result.get('total_processing_time', 0)
                    self.logger.info(f"✓ Batch processed: {result['successful']}/{result['file_count']} successful")
                else:
                    error_text = await resp.text()
                    result['status'] = 'failed'
                    result['error'] = f"HTTP {resp.status}: {error_text}"
                    
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.logger.error(f"✗ Batch processing failed: {e}")
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/gpu/status") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Failed to get GPU status: {e}")
        return {}
    
    async def run_comprehensive_tests(self, max_files: int = None):
        """Run comprehensive API tests"""
        # Get PDF files
        pdf_files = sorted(list(self.pdf_folder.glob("*.pdf")))
        if max_files:
            pdf_files = pdf_files[:max_files]
            
        self.logger.info(f"Found {len(pdf_files)} PDF files to test")
        
        all_results = {
            'test_start': datetime.now().isoformat(),
            'server_url': self.base_url,
            'pdf_count': len(pdf_files),
            'tests': {}
        }
        
        # Test 1: Synchronous processing
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Test 1: Synchronous PDF Processing")
        self.logger.info(f"{'='*60}")
        
        sync_results = []
        for i, pdf in enumerate(pdf_files[:5]):  # Test first 5 files
            result = await self.test_sync_processing(pdf)
            sync_results.append(result)
            
            # Check GPU status
            if i % 2 == 0:
                gpu_status = await self.get_gpu_status()
                if gpu_status and 'devices' in gpu_status:
                    device = gpu_status['devices'][0]
                    self.logger.info(f"GPU: {device['utilization_percent']:.1f}% util, "
                                   f"{device['memory_used_mb']}/{device['memory_total_mb']}MB")
        
        all_results['tests']['sync_processing'] = sync_results
        
        # Test 2: Asynchronous processing
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Test 2: Asynchronous Job Processing")
        self.logger.info(f"{'='*60}")
        
        async_results = []
        for pdf in pdf_files[5:8]:  # Test next 3 files
            result = await self.test_async_processing(pdf, strategy="accuracy")
            async_results.append(result)
            
        all_results['tests']['async_processing'] = async_results
        
        # Test 3: Batch processing
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Test 3: Batch Processing")
        self.logger.info(f"{'='*60}")
        
        batch_result = await self.test_batch_processing(pdf_files[8:13])  # Test 5 files
        all_results['tests']['batch_processing'] = batch_result
        
        # Test 4: Performance test - rapid submissions
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Test 4: Performance Test - Rapid Submissions")
        self.logger.info(f"{'='*60}")
        
        perf_start = time.time()
        perf_tasks = []
        
        # Submit 10 files concurrently
        for pdf in pdf_files[:10]:
            task = self.test_sync_processing(pdf, strategy="speed")
            perf_tasks.append(task)
            
        perf_results = await asyncio.gather(*perf_tasks, return_exceptions=True)
        perf_time = time.time() - perf_start
        
        successful_perf = [r for r in perf_results if isinstance(r, dict) and r.get('status') == 'success']
        all_results['tests']['performance_test'] = {
            'concurrent_requests': len(perf_tasks),
            'total_time': perf_time,
            'successful': len(successful_perf),
            'requests_per_second': len(perf_tasks) / perf_time
        }
        
        self.logger.info(f"Performance: {len(successful_perf)}/{len(perf_tasks)} successful, "
                       f"{all_results['tests']['performance_test']['requests_per_second']:.2f} req/s")
        
        # Get final statistics
        try:
            async with self.session.get(f"{self.base_url}/api/v1/stats") as resp:
                if resp.status == 200:
                    all_results['final_stats'] = await resp.json()
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
        
        # Save results
        results_file = Path('test_api_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"\nResults saved to {results_file}")
        
        # Print summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("API TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total PDFs tested: {len(pdf_files)}")
        self.logger.info(f"Sync processing: {len([r for r in sync_results if r['status'] == 'success'])}/{len(sync_results)} successful")
        self.logger.info(f"Async processing: {len([r for r in async_results if r['status'] == 'success'])}/{len(async_results)} successful")
        self.logger.info(f"Batch processing: {'Success' if batch_result['status'] == 'success' else 'Failed'}")
        self.logger.info(f"Performance test: {all_results['tests']['performance_test']['requests_per_second']:.2f} req/s")
        
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='API-based PDF OCR testing')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                       help='Server URL')
    parser.add_argument('--pdf-folder', type=str, 
                       default='/home/ryanb/Projects/gpu-server0.1/tests/testpdfs',
                       help='Folder containing PDF files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to test')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run tester
    tester = APITestRunner(args.server, args.pdf_folder)
    
    try:
        await tester.initialize()
        await tester.run_comprehensive_tests(args.max_files)
    except KeyboardInterrupt:
        logging.info("\nTest interrupted by user")
    except Exception as e:
        logging.error(f"Test failed: {e}", exc_info=True)
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())