# CLAUDE.MD - /src/

## Component Purpose
Top-level orchestration layer that coordinates all OCR processing components, manages the application lifecycle, and provides the main entry point for the FastAPI server.

## Classes and Functions to Implement

### File: main.py

```python
app: FastAPI
    """Global FastAPI application instance
    Configures middleware, routes, and startup/shutdown events
    """

async def startup_event():
    """Purpose: Initialize application resources on startup
    Calls: 
        - ocr_service.initialize()
        - gpu_monitor.start_monitoring()
        - cache_manager.initialize_cache()
    Called by: FastAPI startup
    Priority: CORE
    """

async def shutdown_event():
    """Purpose: Clean up resources on shutdown
    Calls:
        - ocr_service.cleanup()
        - gpu_monitor.stop_monitoring()
        - cache_manager.clear_cache()
    Called by: FastAPI shutdown
    Priority: CORE
    """

def create_app() -> FastAPI:
    """Purpose: Factory function to create configured FastAPI app
    Calls: 
        - api.routes.register_routes()
        - api.middleware.setup_middleware()
    Called by: Main execution
    Priority: CORE
    """
```

### File: ocr_service.py

```python
class OCRService:
    def __init__(self, config: OCRConfig):
        """Purpose: Initialize the OCR orchestration service
        Dependencies: models.paddle_ocr, utils.pdf_processor, gpu_monitor
        Initializes: PaddleOCR model, TensorRT optimizer, batch processor
        Priority: CORE
        """
    
    async def process_document(
        self, 
        file_bytes: bytes, 
        file_type: str,
        processing_strategy: ProcessingStrategy
    ) -> OCRResult:
        """Purpose: Main entry point for document processing
        Calls: 
            - _prepare_document()
            - _determine_batch_size()
            - _process_batches()
            - _format_results()
        Called by: api.routes.process_endpoint
        Priority: CORE
        """
    
    async def _prepare_document(
        self, 
        file_bytes: bytes, 
        file_type: str
    ) -> List[np.ndarray]:
        """Purpose: Convert document to processable images
        Calls:
            - pdf_processor.extract_pages() (if PDF)
            - image_processor.prepare_image() (if image)
        Called by: process_document()
        Priority: CORE
        """
    
    def _determine_batch_size(
        self, 
        page_count: int,
        image_dimensions: Tuple[int, int]
    ) -> int:
        """Purpose: Calculate optimal batch size based on GPU resources
        Calls:
            - gpu_monitor.get_available_memory()
            - _estimate_memory_requirement()
        Called by: process_document()
        Priority: OPTIMIZATION
        """
    
    async def _process_batches(
        self,
        images: List[np.ndarray],
        batch_size: int
    ) -> List[PageResult]:
        """Purpose: Process images in optimized batches
        Calls:
            - paddle_ocr.process_batch()
            - gpu_monitor.check_memory_pressure()
        Called by: process_document()
        Priority: CORE
        """
    
    def _format_results(
        self,
        page_results: List[PageResult]
    ) -> OCRResult:
        """Purpose: Format raw OCR results into structured output
        Calls:
            - result_formatter.format_page_results()
        Called by: process_document()
        Priority: CORE
        """
    
    def get_processing_stats(self) -> ProcessingStats:
        """Purpose: Return current processing statistics
        Calls: Internal metrics tracking
        Called by: api.routes.stats_endpoint
        Priority: MONITORING
        """
    
    async def initialize(self):
        """Purpose: Initialize OCR models and warm up GPU
        Calls:
            - paddle_ocr.initialize_model()
            - tensorrt_optimizer.optimize_model()
            - _warmup_gpu()
        Called by: main.startup_event
        Priority: CORE
        """
    
    async def cleanup(self):
        """Purpose: Clean up resources
        Calls:
            - paddle_ocr.cleanup()
            - Clear internal buffers
        Called by: main.shutdown_event
        Priority: CORE
        """
    
    def _estimate_memory_requirement(
        self,
        batch_size: int,
        image_dimensions: Tuple[int, int]
    ) -> int:
        """Purpose: Estimate GPU memory needed for batch
        Calls: None (calculation based on model params)
        Called by: _determine_batch_size()
        Priority: OPTIMIZATION
        """
    
    async def _warmup_gpu(self):
        """Purpose: Pre-allocate GPU resources with dummy inference
        Calls:
            - paddle_ocr.process_batch() with dummy data
        Called by: initialize()
        Priority: OPTIMIZATION
        """
```

### File: gpu_monitor.py

```python
class GPUMonitor:
    def __init__(self, device_id: int = 0):
        """Purpose: Initialize GPU monitoring with nvidia-ml-py
        Dependencies: nvidia-ml-py (pynvml)
        Priority: MONITORING
        """
    
    def start_monitoring(self):
        """Purpose: Start background GPU monitoring thread
        Calls:
            - pynvml.nvmlInit()
            - _monitoring_loop() in thread
        Called by: main.startup_event
        Priority: MONITORING
        """
    
    def stop_monitoring(self):
        """Purpose: Stop monitoring thread and cleanup
        Calls:
            - pynvml.nvmlShutdown()
        Called by: main.shutdown_event
        Priority: MONITORING
        """
    
    def get_available_memory(self) -> MemoryInfo:
        """Purpose: Get current GPU memory availability
        Calls:
            - pynvml.nvmlDeviceGetMemoryInfo()
        Called by: ocr_service._determine_batch_size()
        Priority: OPTIMIZATION
        """
    
    def get_gpu_utilization(self) -> GPUUtilization:
        """Purpose: Get current GPU compute utilization
        Calls:
            - pynvml.nvmlDeviceGetUtilizationRates()
        Called by: api.routes.gpu_status_endpoint
        Priority: MONITORING
        """
    
    def check_memory_pressure(self) -> bool:
        """Purpose: Check if GPU memory is under pressure
        Calls:
            - get_available_memory()
        Called by: ocr_service._process_batches()
        Priority: OPTIMIZATION
        """
    
    def get_temperature(self) -> float:
        """Purpose: Get GPU temperature for thermal monitoring
        Calls:
            - pynvml.nvmlDeviceGetTemperature()
        Called by: api.routes.gpu_status_endpoint
        Priority: MONITORING
        """
    
    def _monitoring_loop(self):
        """Purpose: Background thread for continuous monitoring
        Calls:
            - All get_* methods periodically
            - _update_metrics()
        Called by: start_monitoring() in thread
        Priority: MONITORING
        """
    
    def _update_metrics(self, metrics: GPUMetrics):
        """Purpose: Update internal metrics storage
        Calls: None (updates internal state)
        Called by: _monitoring_loop()
        Priority: MONITORING
        """
```

### File: config.py

```python
@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    default_dpi: int = 120
    max_batch_size: int = 50
    min_batch_size: int = 1
    gpu_memory_buffer_mb: int = 500
    tensorrt_precision: str = "FP16"
    model_cache_dir: str = "./model_cache"
    warmup_iterations: int = 3

@dataclass
class ServerConfig:
    """FastAPI server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 300

@dataclass
class GPUConfig:
    """GPU monitoring configuration"""
    device_id: int = 0
    monitoring_interval_seconds: float = 1.0
    memory_threshold_percent: float = 85.0

def load_config() -> Config:
    """Purpose: Load configuration from environment and files
    Calls:
        - os.environ.get()
        - Load from config.yaml if exists
    Called by: Various initialization functions
    Priority: CORE
    """

class Config:
    """Unified configuration container"""
    ocr: OCRConfig
    server: ServerConfig
    gpu: GPUConfig
```

## Data Structures

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

class ProcessingStrategy(Enum):
    SPEED = "speed"          # Optimize for throughput
    ACCURACY = "accuracy"    # Optimize for quality
    BALANCED = "balanced"    # Balance speed/quality

@dataclass
class OCRResult:
    pages: List[PageResult]
    total_pages: int
    processing_time_ms: float
    confidence_score: float
    metadata: Dict[str, any]

@dataclass
class PageResult:
    page_number: int
    text: str
    words: List[WordResult]
    confidence: float
    processing_time_ms: float

@dataclass
class WordResult:
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

@dataclass
class MemoryInfo:
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_percent: float

@dataclass
class GPUUtilization:
    compute_percent: float
    memory_percent: float
    temperature_celsius: float
    power_draw_watts: float

@dataclass
class ProcessingStats:
    total_documents_processed: int
    total_pages_processed: int
    average_pages_per_second: float
    average_batch_size: float
    gpu_utilization_average: float
```

## Cross-Component Dependencies

**Imports from:**
- `api/` - routes, schemas, middleware
- `models/` - paddle_ocr, tensorrt_optimizer, result_formatter
- `utils/` - pdf_processor, image_processor, cache_manager

**External libraries:**
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- numpy==1.24.3
- nvidia-ml-py==12.535.133
- python-multipart==0.0.6
- pydantic==2.5.0
- python-dotenv==1.0.0
- loguru==0.7.2

**Calls into:**
- All subdirectory components
- External GPU monitoring APIs

**Called by:**
- External HTTP clients
- Container orchestration (Docker/K8s)

## Implementation Notes

1. **Initialization Order:**
   - GPU Monitor must start before OCR Service
   - Model loading happens during startup event
   - TensorRT optimization occurs after model load

2. **Memory Management:**
   - Always reserve 500MB GPU buffer for system
   - Monitor memory pressure during batch processing
   - Dynamically adjust batch size if pressure detected

3. **Error Recovery:**
   - Implement exponential backoff for GPU errors
   - Fallback to CPU processing if GPU fails
   - Maintain request queue during high load

4. **Performance Targets:**
   - 50-120 pages/minute throughput
   - Sub-100ms latency for single page
   - 95% GPU utilization under load

## Error Handling

- **GPU Errors**: Log, attempt recovery, fallback to CPU
- **Memory Errors**: Reduce batch size, clear cache, retry
- **Model Errors**: Reload model, warm up, retry
- **Input Errors**: Validate early, return descriptive errors
- **Timeout Errors**: Cancel processing, clean up resources