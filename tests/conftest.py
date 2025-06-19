"""
PyTest Configuration and Fixtures

Provides common test fixtures and configuration for all tests.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator, Dict, Any
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from PIL import Image
import io

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config, OCRConfig, ServerConfig, GPUConfig, CacheConfig, Environment
from src.gpu_monitor import GPUMonitor, GPUMetrics, MemoryInfo, GPUUtilization
from src.utils.cache_manager import CacheManager
from src.ocr_service import OCRService


# Configure pytest
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Config:
    """Provide test configuration"""
    return Config(
        ocr=OCRConfig(
            default_dpi=150,
            max_batch_size=10,
            gpu_memory_buffer_mb=200,
            warmup_iterations=1,
            model_cache_dir=str(Path(tempfile.gettempdir()) / "test_models")
        ),
        server=ServerConfig(
            host="127.0.0.1",
            port=8001,
            workers=1,
            enable_docs=True,
            log_level="DEBUG",
            upload_path=str(Path(tempfile.gettempdir()) / "test_uploads"),
            temp_path=str(Path(tempfile.gettempdir()) / "test_temp")
        ),
        gpu=GPUConfig(
            device_id=0,
            monitoring_interval_seconds=0.1,
            memory_threshold_percent=80.0,
            enable_monitoring=False  # Disable for tests
        ),
        cache=CacheConfig(
            enable_cache=True,
            cache_type="memory",
            memory_cache_size_mb=128,
            disk_cache_path=str(Path(tempfile.gettempdir()) / "test_cache")
        ),
        environment=Environment.DEVELOPMENT
    )


@pytest.fixture
def mock_gpu_monitor() -> Mock:
    """Provide mock GPU monitor"""
    monitor = Mock(spec=GPUMonitor)
    
    # Mock the synchronous methods that exist in GPUMonitor
    monitor.get_memory_info = Mock(return_value=MemoryInfo(
        total_mb=24576,
        used_mb=4096,
        free_mb=20480,
        utilization_percent=16.7
    ))
    
    monitor.get_gpu_utilization = Mock(return_value=GPUUtilization(
        compute_percent=25.0,
        memory_percent=16.7,
        temperature_celsius=45.0,
        power_draw_watts=150.0
    ))
    
    monitor.get_available_memory = Mock(return_value=MemoryInfo(
        total_mb=24576,
        used_mb=4096,
        free_mb=20480,
        utilization_percent=16.7
    ))
    
    monitor.check_memory_pressure = Mock(return_value=False)
    monitor.start_monitoring = Mock()
    monitor.stop_monitoring = Mock()
    monitor.is_initialized = True
    monitor.gpu_available = True
    
    return monitor


@pytest.fixture
def mock_cache_manager(test_config: Config) -> CacheManager:
    """Provide cache manager for testing"""
    # Use memory-only cache for tests
    test_config.cache.cache_type = "memory"
    test_config.cache.enable_cache = True
    cache = CacheManager(test_config.cache)
    yield cache
    # Cleanup
    cache.clear()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_image() -> bytes:
    """Create a test image"""
    # Create a simple test image with text
    img = Image.new('RGB', (800, 600), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Add some text
    text = "Test OCR Image\nThis is a test document\nFor GPU OCR Server"
    draw.multiline_text((50, 50), text, fill='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def test_pdf() -> bytes:
    """Create a test PDF"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        pytest.skip("reportlab not installed")
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Add content
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Test PDF Document")
    c.drawString(100, 700, "This is a test PDF for OCR processing")
    c.drawString(100, 650, "Page 1 of 2")
    c.showPage()
    
    # Add second page
    c.drawString(100, 750, "Page 2")
    c.drawString(100, 700, "More test content for OCR")
    c.showPage()
    
    c.save()
    return buffer.getvalue()


@pytest.fixture
def test_documents(temp_dir: Path) -> Dict[str, Path]:
    """Create various test documents"""
    docs = {}
    
    # Create test image
    img_path = temp_dir / "test_image.png"
    img = Image.new('RGB', (800, 600), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Test Image", fill='black')
    img.save(img_path)
    docs['image'] = img_path
    
    # Create test text file (for negative testing)
    txt_path = temp_dir / "test.txt"
    txt_path.write_text("This is not an image")
    docs['text'] = txt_path
    
    # Create empty file
    empty_path = temp_dir / "empty.jpg"
    empty_path.touch()
    docs['empty'] = empty_path
    
    return docs


@pytest_asyncio.fixture
async def mock_ocr_service(test_config: Config, mock_gpu_monitor: Mock, mock_cache_manager: CacheManager):
    """Provide mock OCR service"""
    with patch('src.ocr_service.PaddleOCRWrapper') as mock_paddle:
        # Mock PaddleOCR
        mock_instance = Mock()
        mock_instance.initialize = AsyncMock()
        mock_instance.process_batch = AsyncMock(return_value=[])
        mock_paddle.return_value = mock_instance
        
        # Create service
        service = OCRService(test_config, mock_gpu_monitor, mock_cache_manager)
        
        # Mock initialization
        service.initialized = True
        service.paddle_ocr = mock_instance
        
        yield service


@pytest.fixture
def mock_redis():
    """Provide mock Redis client"""
    redis_mock = Mock()
    redis_mock.get = Mock(return_value=None)
    redis_mock.set = Mock(return_value=True)
    redis_mock.delete = Mock(return_value=1)
    redis_mock.exists = Mock(return_value=False)
    redis_mock.expire = Mock(return_value=True)
    redis_mock.ping = Mock(return_value=True)
    redis_mock.pipeline = Mock(return_value=redis_mock)
    redis_mock.execute = Mock(return_value=[None, None])
    return redis_mock


@pytest.fixture
def api_client():
    """Provide test client for API testing"""
    from fastapi.testclient import TestClient
    from src.main import app
    
    # Override dependencies for testing
    app.state.config = test_config()
    app.state.ocr_service = Mock()
    app.state.gpu_monitor = mock_gpu_monitor()
    app.state.cache_manager = Mock()
    
    return TestClient(app)


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Slow tests")


# Test data generators
def generate_test_image(width: int = 800, height: int = 600, text: str = "Test") -> np.ndarray:
    """Generate test image with text"""
    img = Image.new('RGB', (width, height), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), text, fill='black')
    return np.array(img)


def generate_batch_images(count: int = 5) -> list[np.ndarray]:
    """Generate batch of test images"""
    return [
        generate_test_image(text=f"Test Image {i+1}")
        for i in range(count)
    ]


# Environment setup/teardown
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")  # Disable GPU for tests
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENABLE_CACHE", "false")  # Disable cache by default