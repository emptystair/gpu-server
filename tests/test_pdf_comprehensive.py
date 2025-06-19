#!/usr/bin/env python3
"""
Comprehensive PDF testing script for GPU OCR Server.
Tests multiple PDFs through the OCR service with different strategies and configurations.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse
import logging

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.ocr_service import OCRService, ProcessingRequest, ProcessingStrategy
from src.api.schemas import OCRLanguage


class PDFTestRunner:
    """Comprehensive PDF test runner"""
    
    def __init__(self, config, pdf_folder: str):
        self.config = config
        self.pdf_folder = Path(pdf_folder)
        self.service = OCRService(config)
        self.results = []
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize OCR service"""
        self.logger.info("Initializing OCR service...")
        await self.service.initialize()
        self.logger.info("OCR service initialized successfully")
        
    async def test_single_pdf(self, pdf_path: Path, strategy: ProcessingStrategy) -> Dict[str, Any]:
        """Test a single PDF file"""
        start_time = time.time()
        result = {
            'filename': pdf_path.name,
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'strategy': strategy.value,
            'status': 'pending',
            'error': None,
            'pages': 0,
            'processing_time': 0,
            'confidence': 0,
            'text_length': 0
        }
        
        try:
            # Create processing request
            request = ProcessingRequest(
                document_path=str(pdf_path),
                strategy=strategy,
                language=OCRLanguage.ENGLISH,
                dpi=150,
                confidence_threshold=0.5
            )
            
            # Process document
            self.logger.info(f"Processing {pdf_path.name} with {strategy.value} strategy...")
            ocr_result = await self.service.process_document(request)
            
            # Update result
            result['status'] = 'success'
            result['pages'] = len(ocr_result.pages)
            result['processing_time'] = time.time() - start_time
            result['confidence'] = ocr_result.average_confidence
            result['text_length'] = sum(len(page.text) for page in ocr_result.pages)
            result['strategy_used'] = ocr_result.strategy_used.value
            result['processing_time_ms'] = ocr_result.processing_time_ms
            
            self.logger.info(f"✓ Processed {pdf_path.name}: {result['pages']} pages, "
                           f"{result['confidence']:.2%} confidence, "
                           f"{result['processing_time']:.2f}s")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(f"✗ Failed to process {pdf_path.name}: {e}")
            
        return result
    
    async def test_batch(self, pdf_files: List[Path], strategy: ProcessingStrategy) -> List[Dict[str, Any]]:
        """Test a batch of PDFs"""
        batch_results = []
        
        for pdf_path in pdf_files:
            result = await self.test_single_pdf(pdf_path, strategy)
            batch_results.append(result)
            
            # Get GPU stats after each file
            stats = self.service.get_processing_stats()
            self.logger.info(f"GPU Stats - Memory: {stats.gpu_memory_used_mb:.0f}MB, "
                           f"Utilization: {stats.gpu_utilization_average:.1f}%")
            
            # Small delay between files
            await asyncio.sleep(0.5)
            
        return batch_results
    
    async def run_comprehensive_tests(self, max_files: int = None):
        """Run comprehensive tests on all PDFs"""
        # Get all PDF files
        pdf_files = sorted(list(self.pdf_folder.glob("*.pdf")))
        if max_files:
            pdf_files = pdf_files[:max_files]
            
        self.logger.info(f"Found {len(pdf_files)} PDF files to test")
        
        # Test each strategy
        strategies = [ProcessingStrategy.SPEED, ProcessingStrategy.BALANCED, ProcessingStrategy.ACCURACY]
        
        all_results = {
            'test_start': datetime.now().isoformat(),
            'pdf_count': len(pdf_files),
            'strategies': {},
            'summary': {}
        }
        
        for strategy in strategies:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing with {strategy.value.upper()} strategy")
            self.logger.info(f"{'='*60}")
            
            strategy_start = time.time()
            results = await self.test_batch(pdf_files, strategy)
            strategy_time = time.time() - strategy_start
            
            # Calculate statistics
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'failed']
            
            strategy_stats = {
                'total_files': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time': strategy_time,
                'avg_time_per_file': strategy_time / len(results) if results else 0,
                'total_pages': sum(r['pages'] for r in successful),
                'avg_confidence': sum(r['confidence'] for r in successful) / len(successful) if successful else 0,
                'pages_per_second': sum(r['pages'] for r in successful) / strategy_time if strategy_time > 0 else 0,
                'results': results
            }
            
            all_results['strategies'][strategy.value] = strategy_stats
            
            # Print summary
            self.logger.info(f"\n{strategy.value.upper()} Strategy Summary:")
            self.logger.info(f"  Files processed: {strategy_stats['successful']}/{strategy_stats['total_files']}")
            self.logger.info(f"  Total pages: {strategy_stats['total_pages']}")
            self.logger.info(f"  Average confidence: {strategy_stats['avg_confidence']:.2%}")
            self.logger.info(f"  Pages/second: {strategy_stats['pages_per_second']:.2f}")
            self.logger.info(f"  Avg time/file: {strategy_stats['avg_time_per_file']:.2f}s")
            
            if failed:
                self.logger.warning(f"  Failed files: {len(failed)}")
                for f in failed[:5]:  # Show first 5 failures
                    self.logger.warning(f"    - {f['filename']}: {f['error']}")
        
        # Overall summary
        all_results['test_end'] = datetime.now().isoformat()
        all_results['summary'] = {
            'total_test_time': sum(s['total_time'] for s in all_results['strategies'].values()),
            'best_speed_strategy': max(all_results['strategies'].items(), 
                                      key=lambda x: x[1]['pages_per_second'])[0],
            'best_accuracy_strategy': max(all_results['strategies'].items(), 
                                        key=lambda x: x[1]['avg_confidence'])[0]
        }
        
        # Save results
        results_file = Path('test_results_comprehensive.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        self.logger.info(f"\nResults saved to {results_file}")
        
        # Print final summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("COMPREHENSIVE TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total PDFs tested: {len(pdf_files)}")
        self.logger.info(f"Total test time: {all_results['summary']['total_test_time']:.2f}s")
        self.logger.info(f"Best speed: {all_results['summary']['best_speed_strategy']}")
        self.logger.info(f"Best accuracy: {all_results['summary']['best_accuracy_strategy']}")
        
        # GPU final stats
        final_stats = self.service.get_processing_stats()
        self.logger.info(f"\nFinal GPU Stats:")
        self.logger.info(f"  Total documents: {final_stats.total_documents_processed}")
        self.logger.info(f"  Total pages: {final_stats.total_pages_processed}")
        self.logger.info(f"  Cache hit rate: {final_stats.cache_hit_rate:.2%}")
        self.logger.info(f"  Avg GPU utilization: {final_stats.gpu_utilization_average:.1f}%")
        
    async def cleanup(self):
        """Clean up resources"""
        if self.service:
            await self.service.cleanup()


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Comprehensive PDF OCR testing')
    parser.add_argument('--pdf-folder', type=str, 
                       default='/home/ryanb/Projects/gpu-server0.1/tests/testpdfs',
                       help='Folder containing PDF files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to test (default: all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config()
    
    # Create and run tester
    tester = PDFTestRunner(config, args.pdf_folder)
    
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