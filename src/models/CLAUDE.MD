# CLAUDE.MD - /src/models/

## Component Purpose
Encapsulates all ML model operations including PaddleOCR text extraction, TensorRT optimization for GPU acceleration, and result formatting. This layer provides a clean interface to the OCR engine while handling all model-specific optimizations.

## Classes and Functions to Implement

### File: paddle_ocr.py

```python
class PaddleOCRWrapper:
    def __init__(self, config: ModelConfig):
        """Purpose: Initialize PaddleOCR with TensorRT backend
        Dependencies: paddleocr, paddlepaddle-gpu
        Initializes: Detection and recognition models
        Priority: CORE
        """
    
    def initialize_model(self, use_tensorrt: bool = True):
        """Purpose: Load and configure OCR models
        Calls:
            - _load_detection_model()
            - _load_recognition_model()
            - _configure_tensorrt() if use_tensorrt
        Called by: ocr_service.initialize()
        Priority: CORE
        """
    
    def process_batch(
        self,
        images: List[np.ndarray],
        batch_size: int
    ) -> List[OCROutput]:
        """Purpose: Process image batch through OCR pipeline
        Calls:
            - _preprocess_batch()
            - _detect_text_regions()
            - _recognize_text()
            - _postprocess_results()
        Called by: ocr_service._process_batches()
        Priority: CORE
        """
    
    def _load_detection_model(self):
        """Purpose: Load text detection model (DB++)
        Calls:
            - paddleocr.PPStructure() or custom loader
            - _verify_model_integrity()
        Called by: initialize_model()
        Priority: CORE
        """
    
    def _load_recognition_model(self):
        """Purpose: Load text recognition model (CRNN)
        Calls:
            - paddleocr model loading APIs
            - _verify_model_integrity()
        Called by: initialize_model()
        Priority: CORE
        """
    
    def _configure_tensorrt(self):
        """Purpose: Configure TensorRT optimization settings
        Calls:
            - tensorrt_optimizer.prepare_config()
        Called by: initialize_model()
        Priority: OPTIMIZATION
        """
    
    def _preprocess_batch(
        self,
        images: List[np.ndarray]
    ) -> paddle.Tensor:
        """Purpose: Prepare images for model input
        Calls:
            - Image normalization
            - Tensor conversion
            - Padding/resizing
        Called by: process_batch()
        Priority: CORE
        """
    
    def _detect_text_regions(
        self,
        batch_tensor: paddle.Tensor
    ) -> List[List[TextBox]]:
        """Purpose: Run text detection model
        Calls:
            - Detection model inference
            - NMS post-processing
        Called by: process_batch()
        Priority: CORE
        """
    
    def _recognize_text(
        self,
        image_crops: List[np.ndarray]
    ) -> List[TextRecognition]:
        """Purpose: Run text recognition on detected regions
        Calls:
            - Recognition model inference
            - CTC decoding
        Called by: process_batch()
        Priority: CORE
        """
    
    def _postprocess_results(
        self,
        detections: List[List[TextBox]],
        recognitions: List[TextRecognition]
    ) -> List[OCROutput]:
        """Purpose: Combine detection and recognition results
        Calls:
            - _merge_text_lines()
            - _calculate_confidence()
        Called by: process_batch()
        Priority: CORE
        """
    
    def _verify_model_integrity(self, model_path: str) -> bool:
        """Purpose: Verify model files are valid
        Calls:
            - Check file existence
            - Validate model structure
        Called by: _load_*_model()
        Priority: CORE
        """
    
    def cleanup(self):
        """Purpose: Release model resources
        Calls:
            - Clear GPU memory
            - Close paddle sessions
        Called by: ocr_service.cleanup()
        Priority: CORE
        """
    
    def get_model_info(self) -> ModelInfo:
        """Purpose: Return model metadata and status
        Calls: Internal state queries
        Called by: Monitoring endpoints
        Priority: MONITORING
        """
```

### File: tensorrt_optimizer.py

```python
class TensorRTOptimizer:
    def __init__(self, precision: str = "FP16"):
        """Purpose: Initialize TensorRT optimization engine
        Dependencies: tensorrt, paddle2onnx
        Priority: OPTIMIZATION
        """
    
    def optimize_model(
        self,
        paddle_model_path: str,
        output_path: str
    ) -> OptimizedModel:
        """Purpose: Convert PaddlePaddle model to TensorRT
        Calls:
            - _convert_to_onnx()
            - _build_tensorrt_engine()
            - _validate_optimization()
        Called by: ocr_service.initialize()
        Priority: OPTIMIZATION
        """
    
    def _convert_to_onnx(
        self,
        paddle_model_path: str
    ) -> str:
        """Purpose: Convert Paddle model to ONNX format
        Calls:
            - paddle2onnx.convert()
            - _validate_onnx_model()
        Called by: optimize_model()
        Priority: OPTIMIZATION
        """
    
    def _build_tensorrt_engine(
        self,
        onnx_path: str,
        config: TRTConfig
    ) -> trt.ICudaEngine:
        """Purpose: Build optimized TensorRT engine
        Calls:
            - trt.Builder operations
            - _configure_optimization_profile()
            - _set_precision_mode()
        Called by: optimize_model()
        Priority: OPTIMIZATION
        """
    
    def _configure_optimization_profile(
        self,
        builder: trt.Builder,
        expected_shapes: Dict[str, Shape]
    ):
        """Purpose: Set dynamic shape profiles for batching
        Calls:
            - builder.create_optimization_profile()
            - Set min/opt/max shapes
        Called by: _build_tensorrt_engine()
        Priority: OPTIMIZATION
        """
    
    def _set_precision_mode(
        self,
        config: trt.IBuilderConfig,
        precision: str
    ):
        """Purpose: Configure FP16/INT8 precision
        Calls:
            - config.set_flag() for precision
            - Enable tensor core usage
        Called by: _build_tensorrt_engine()
        Priority: OPTIMIZATION
        """
    
    def _validate_optimization(
        self,
        original_model: Any,
        optimized_engine: trt.ICudaEngine
    ) -> ValidationResult:
        """Purpose: Ensure optimization maintains accuracy
        Calls:
            - Run inference on test data
            - Compare outputs
        Called by: optimize_model()
        Priority: OPTIMIZATION
        """
    
    def prepare_config(self) -> TRTConfig:
        """Purpose: Prepare TensorRT configuration
        Calls: None (builds config object)
        Called by: paddle_ocr._configure_tensorrt()
        Priority: OPTIMIZATION
        """
    
    def benchmark_performance(
        self,
        engine: trt.ICudaEngine,
        test_batch_sizes: List[int]
    ) -> BenchmarkResults:
        """Purpose: Measure optimization speedup
        Calls:
            - Run timed inference loops
            - Calculate throughput metrics
        Called by: Testing/monitoring
        Priority: MONITORING
        """
```

### File: result_formatter.py

```python
class ResultFormatter:
    def __init__(self, format_config: FormatConfig):
        """Purpose: Initialize result formatting rules
        Dependencies: None (pure Python)
        Priority: CORE
        """
    
    def format_page_results(
        self,
        ocr_outputs: List[OCROutput],
        page_metadata: PageMetadata
    ) -> PageResult:
        """Purpose: Format raw OCR output into structured result
        Calls:
            - _merge_text_blocks()
            - _apply_reading_order()
            - _calculate_page_confidence()
        Called by: ocr_service._format_results()
        Priority: CORE
        """
    
    def _merge_text_blocks(
        self,
        text_regions: List[TextRegion]
    ) -> List[TextBlock]:
        """Purpose: Merge adjacent text regions into blocks
        Calls:
            - _calculate_proximity()
            - _check_alignment()
        Called by: format_page_results()
        Priority: CORE
        """
    
    def _apply_reading_order(
        self,
        text_blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Purpose: Sort text blocks in reading order
        Calls:
            - _detect_columns()
            - _sort_by_position()
        Called by: format_page_results()
        Priority: CORE
        """
    
    def _calculate_page_confidence(
        self,
        word_confidences: List[float]
    ) -> float:
        """Purpose: Calculate overall page confidence score
        Calls: Statistical calculations
        Called by: format_page_results()
        Priority: CORE
        """
    
    def _detect_columns(
        self,
        text_blocks: List[TextBlock]
    ) -> List[Column]:
        """Purpose: Detect multi-column layouts
        Calls:
            - Clustering algorithms
            - _validate_column_structure()
        Called by: _apply_reading_order()
        Priority: CORE
        """
    
    def format_for_export(
        self,
        page_results: List[PageResult],
        export_format: str
    ) -> Union[Dict, str]:
        """Purpose: Convert results to requested format
        Calls:
            - _to_json() if JSON
            - _to_plain_text() if TXT
            - _to_structured_data() if XML
        Called by: API response formatting
        Priority: CORE
        """
    
    def _calculate_proximity(
        self,
        region1: TextRegion,
        region2: TextRegion
    ) -> float:
        """Purpose: Calculate distance between text regions
        Calls: Geometric calculations
        Called by: _merge_text_blocks()
        Priority: CORE
        """
    
    def apply_confidence_threshold(
        self,
        results: PageResult,
        threshold: float
    ) -> PageResult:
        """Purpose: Filter results by confidence
        Calls: Filter operations
        Called by: Optional post-processing
        Priority: CORE
        """
```

## Data Structures

```python
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import tensorrt as trt

@dataclass
class ModelConfig:
    det_model_dir: str
    rec_model_dir: str
    use_angle_cls: bool = True
    use_tensorrt: bool = True
    tensorrt_precision: str = "FP16"
    max_batch_size: int = 50

@dataclass
class TextBox:
    coordinates: List[Tuple[int, int]]  # Polygon points
    confidence: float

@dataclass
class TextRecognition:
    text: str
    confidence: float
    
@dataclass
class OCROutput:
    boxes: List[TextBox]
    texts: List[str]
    confidences: List[float]
    
@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    
@dataclass
class TextBlock:
    regions: List[TextRegion]
    merged_text: str
    block_confidence: float
    reading_order: int

@dataclass
class OptimizedModel:
    engine_path: str
    original_accuracy: float
    optimized_accuracy: float
    speedup_factor: float
    
@dataclass
class TRTConfig:
    max_batch_size: int
    precision: str
    workspace_size_mb: int
    enable_tensor_core: bool
    
@dataclass
class Shape:
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]
    
@dataclass
class ValidationResult:
    accuracy_delta: float
    latency_improvement: float
    memory_usage_mb: int
    
@dataclass
class BenchmarkResults:
    batch_sizes: List[int]
    throughputs: List[float]  # pages/second
    latencies: List[float]    # ms/page
    gpu_utilizations: List[float]
    
@dataclass
class ModelInfo:
    model_version: str
    optimization_enabled: bool
    current_batch_size: int
    total_inferences: int
    average_latency_ms: float
```

## Cross-Component Dependencies

**Imports from:**
- `../config` - Configuration classes
- `../gpu_monitor` - For optimization decisions

**External libraries:**
- paddlepaddle-gpu==2.5.2.post120
- paddleocr==2.7.0.3
- onnx==1.14.1
- onnxruntime-gpu==1.16.3
- paddle2onnx==1.0.6
- tensorrt==8.6.1.6
- pycuda==2022.2.2

**Calls into:**
- TensorRT C++ API via Python bindings
- PaddlePaddle inference engine
- CUDA runtime for memory management

**Called by:**
- `ocr_service.py` - Main orchestration
- Testing/benchmarking scripts

## Implementation Notes

1. **Model Loading Strategy:**
   - Cache converted TensorRT engines
   - Lazy load models on first request
   - Keep models in GPU memory

2. **Batch Processing:**
   - Dynamic batching based on input sizes
   - Padding strategy for uniform dimensions
   - Stream-based processing for large batches

3. **TensorRT Optimization:**
   - Use FP16 by default for 2-3x speedup
   - INT8 requires calibration dataset
   - Profile common input shapes

4. **Memory Management:**
   - Pre-allocate GPU buffers
   - Reuse tensors across batches
   - Monitor memory fragmentation

5. **Error Recovery:**
   - Fallback to non-optimized model
   - Validate outputs against threshold
   - Automatic model reloading

## Error Handling

- **Model Loading Errors**: Log details, attempt download, fail gracefully
- **TensorRT Build Errors**: Fall back to standard inference
- **OOM Errors**: Reduce batch size, clear cache, retry
- **Inference Errors**: Validate inputs, log anomalies, skip bad images
- **Precision Errors**: Monitor accuracy degradation, alert if threshold exceeded