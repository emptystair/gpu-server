"""
OCR Pipeline Integration Tests

Tests the complete OCR processing pipeline from file input to final output,
including caching, GPU optimization, and error handling.
"""

import pytest
import asyncio
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, AsyncMock
import io

from src.config import load_config
from src.api.schemas import ProcessingStrategy
from src.ocr_service import OCRService, ProcessingRequest, ProcessingResult
from src.gpu_monitor import GPUMonitor
from src.utils.cache_manager import CacheManager


class TestOCRPipeline:
    """Test complete OCR processing pipeline"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        config = load_config()
        config.environment = "test"
        config.ocr.warmup_iterations = 0  # Skip warmup in tests
        config.cache.cache_type = "memory"
        return config
    
    @pytest.fixture
    def test_images(self, temp_dir):
        """Create test images with text"""
        images = {}
        
        # Simple text image
        img1 = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img1)
        draw.text((50, 50), "Hello World\nThis is a test", fill='black')
        path1 = temp_dir / "simple.png"
        img1.save(path1)
        images['simple'] = path1
        
        # Complex layout image
        img2 = Image.new('RGB', (1200, 800), color='white')
        draw = ImageDraw.Draw(img2)
        # Title
        draw.text((100, 50), "Document Title", fill='black')
        # Multiple columns
        draw.text((100, 150), "Column 1 text\nMore content", fill='black')
        draw.text((600, 150), "Column 2 text\nAdditional info", fill='black')
        path2 = temp_dir / "complex.png"
        img2.save(path2)
        images['complex'] = path2
        
        # Low quality image
        img3 = Image.new('RGB', (400, 300), color='gray')
        draw = ImageDraw.Draw(img3)
        draw.text((50, 50), "Blurry text", fill='darkgray')
        path3 = temp_dir / "low_quality.jpg"
        img3.save(path3, quality=30)
        images['low_quality'] = path3
        
        return images
    
    @pytest.fixture
    def test_pdf(self, temp_dir):
        """Create test PDF"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
        except ImportError:
            pytest.skip("reportlab not installed")
        
        pdf_path = temp_dir / "test.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        
        # Page 1
        c.setFont("Helvetica", 16)
        c.drawString(100, 750, "Test PDF Document")
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "This is page 1 of the test PDF")
        c.drawString(100, 650, "It contains sample text for OCR testing")
        c.showPage()
        
        # Page 2
        c.drawString(100, 750, "Page 2 Content")
        c.drawString(100, 700, "More test content on the second page")
        c.showPage()
        
        c.save()
        return pdf_path
    
    @pytest.mark.integration
    async def test_single_image_pipeline(self, config, test_images, mock_gpu_monitor, mock_cache_manager):
        """Test complete pipeline for single image"""
        # Create OCR service
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            # Mock PaddleOCR
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            mock_ocr.process_batch = AsyncMock(return_value=[
                Mock(
                    boxes=[[10, 10, 100, 30]],
                    texts=["Hello World"],
                    scores=[0.95]
                ),
                Mock(
                    boxes=[[10, 40, 150, 60]],
                    texts=["This is a test"],
                    scores=[0.92]
                )
            ])
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            # Process image
            request = ProcessingRequest(
                document_path=str(test_images['simple']),
                strategy=ProcessingStrategy.BALANCED,
                language="en",
                dpi=150,
                enable_gpu_optimization=True
            )
            
            result = await service.process_document(request)
            
            # Verify result
            assert isinstance(result, ProcessingResult)
            assert len(result.pages) == 1
            assert result.pages[0].page_number == 1
            assert "Hello World" in result.pages[0].text
            assert "This is a test" in result.pages[0].text
            assert result.average_confidence > 0.9
    
    @pytest.mark.integration
    async def test_pdf_pipeline(self, config, test_pdf, mock_gpu_monitor, mock_cache_manager):
        """Test complete pipeline for PDF processing"""
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            # Mock OCR results for each page
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            mock_ocr.process_batch = AsyncMock(side_effect=[
                # Page 1 results
                [Mock(
                    boxes=[[100, 100, 300, 120]],
                    texts=["Test PDF Document"],
                    scores=[0.98]
                )],
                # Page 2 results
                [Mock(
                    boxes=[[100, 100, 250, 120]],
                    texts=["Page 2 Content"],
                    scores=[0.96]
                )]
            ])
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            # Process PDF
            request = ProcessingRequest(
                document_path=str(test_pdf),
                strategy=ProcessingStrategy.BALANCED,
                language="en",
                dpi=200
            )
            
            result = await service.process_document(request)
            
            # Verify result
            assert len(result.pages) == 2
            assert result.pages[0].text == "Test PDF Document"
            assert result.pages[1].text == "Page 2 Content"
            assert result.total_pages == 2
    
    @pytest.mark.integration
    async def test_caching_integration(self, config, test_images, mock_gpu_monitor):
        """Test caching behavior in pipeline"""
        # Use real cache manager
        cache = CacheManager(config.cache)
        
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            call_count = 0
            
            async def mock_process(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return [Mock(boxes=[], texts=["Cached result"], scores=[0.9])]
            
            mock_ocr.process_batch = mock_process
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            request = ProcessingRequest(
                document_path=str(test_images['simple']),
                strategy=ProcessingStrategy.SPEED,
                language="en"
            )
            
            # First request - should process
            result1 = await service.process_document(request)
            assert call_count == 1
            
            # Second request - should use cache
            result2 = await service.process_document(request)
            assert call_count == 1  # No additional calls
            
            # Results should be identical
            assert result1.pages[0].text == result2.pages[0].text
    
    @pytest.mark.integration
    async def test_strategy_differences(self, config, test_images, mock_gpu_monitor, mock_cache_manager):
        """Test different processing strategies"""
        strategies = [
            ProcessingStrategy.SPEED,
            ProcessingStrategy.BALANCED,
            ProcessingStrategy.ACCURACY
        ]
        
        results = {}
        
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            
            # Mock different results for different strategies
            def strategy_based_results(*args, **kwargs):
                # In real implementation, accuracy mode would have higher scores
                if service.current_strategy == ProcessingStrategy.ACCURACY:
                    return [Mock(boxes=[], texts=["High accuracy"], scores=[0.99])]
                elif service.current_strategy == ProcessingStrategy.SPEED:
                    return [Mock(boxes=[], texts=["Fast result"], scores=[0.85])]
                else:
                    return [Mock(boxes=[], texts=["Balanced"], scores=[0.92])]
            
            mock_ocr.process_batch = AsyncMock(side_effect=strategy_based_results)
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            for strategy in strategies:
                service.current_strategy = strategy  # For testing
                request = ProcessingRequest(
                    document_path=str(test_images['simple']),
                    strategy=strategy,
                    language="en"
                )
                
                result = await service.process_document(request)
                results[strategy] = result
            
            # Verify different strategies produce different results
            assert results[ProcessingStrategy.SPEED].pages[0].text == "Fast result"
            assert results[ProcessingStrategy.ACCURACY].pages[0].text == "High accuracy"
            assert results[ProcessingStrategy.BALANCED].pages[0].text == "Balanced"
    
    @pytest.mark.integration
    async def test_batch_processing(self, config, test_images, mock_gpu_monitor, mock_cache_manager):
        """Test batch processing optimization"""
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            
            # Track batch sizes
            batch_sizes = []
            
            async def track_batch_size(images, *args, **kwargs):
                batch_sizes.append(len(images))
                return [Mock(boxes=[], texts=[f"Result {i}"], scores=[0.9]) 
                       for i in range(len(images))]
            
            mock_ocr.process_batch = track_batch_size
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            # Create multi-page document
            images = [test_images['simple']] * 10
            
            # Process with batch optimization
            request = ProcessingRequest(
                document_path=str(images[0]),  # Not used in this test
                strategy=ProcessingStrategy.BALANCED,
                batch_size=5
            )
            
            # Mock the document preparation to return multiple images
            with patch.object(service, '_prepare_document', return_value=images):
                result = await service.process_document(request)
            
            # Should have processed in batches
            assert len(batch_sizes) > 0
            assert max(batch_sizes) <= 5  # Respects batch size limit
    
    @pytest.mark.integration
    async def test_error_recovery(self, config, test_images, mock_gpu_monitor, mock_cache_manager):
        """Test error handling and recovery"""
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            
            # Simulate intermittent failures
            call_count = 0
            
            async def flaky_process(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Temporary failure")
                return [Mock(boxes=[], texts=["Success"], scores=[0.9])]
            
            mock_ocr.process_batch = flaky_process
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            request = ProcessingRequest(
                document_path=str(test_images['simple']),
                strategy=ProcessingStrategy.BALANCED
            )
            
            # Should retry and succeed
            with patch.object(service, '_process_with_retry', 
                            side_effect=service._process_with_retry) as mock_retry:
                result = await service.process_document(request)
            
            assert result is not None
            assert call_count > 1  # Retried after failure
    
    @pytest.mark.integration
    async def test_memory_pressure_handling(self, config, test_images, mock_cache_manager):
        """Test handling of GPU memory pressure"""
        # Create GPU monitor that simulates memory pressure
        mock_gpu = Mock()
        mock_gpu.get_available_memory = AsyncMock(return_value=1000)  # Low memory
        mock_gpu.check_memory_pressure = AsyncMock(return_value=True)
        mock_gpu.get_current_status = AsyncMock()
        mock_gpu.initialize = AsyncMock()
        mock_gpu.shutdown = AsyncMock()
        
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            mock_ocr.process_batch = AsyncMock(
                return_value=[Mock(boxes=[], texts=["Result"], scores=[0.9])]
            )
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            # Process with large batch request
            request = ProcessingRequest(
                document_path=str(test_images['simple']),
                strategy=ProcessingStrategy.BALANCED,
                batch_size=50  # Large batch
            )
            
            # Should automatically reduce batch size
            with patch.object(service, '_determine_batch_size') as mock_batch:
                mock_batch.return_value = 5  # Reduced due to memory pressure
                
                result = await service.process_document(request)
                
                # Should still complete successfully
                assert result is not None
                mock_batch.assert_called()
    
    @pytest.mark.performance
    @pytest.mark.integration
    async def test_processing_performance(self, config, test_images, mock_gpu_monitor, mock_cache_manager):
        """Test processing performance metrics"""
        import time
        
        with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
            mock_ocr = Mock()
            mock_ocr.initialize = AsyncMock()
            
            # Simulate realistic processing time
            async def timed_process(images, *args, **kwargs):
                await asyncio.sleep(0.01 * len(images))  # 10ms per image
                return [Mock(boxes=[], texts=[f"Result {i}"], scores=[0.9]) 
                       for i in range(len(images))]
            
            mock_ocr.process_batch = timed_process
            mock_paddle.return_value = mock_ocr
            
            service = OCRService(config)
            await service.initialize()
            
            # Process multiple documents
            processing_times = []
            
            for img_name, img_path in test_images.items():
                request = ProcessingRequest(
                    document_path=str(img_path),
                    strategy=ProcessingStrategy.SPEED
                )
                
                start = time.time()
                result = await service.process_document(request)
                elapsed = time.time() - start
                
                processing_times.append(elapsed)
                
                # Basic performance assertions
                assert elapsed < 1.0  # Should be fast
                assert result.processing_time_ms > 0
            
            # Check consistency
            avg_time = sum(processing_times) / len(processing_times)
            print(f"Average processing time: {avg_time*1000:.2f}ms")