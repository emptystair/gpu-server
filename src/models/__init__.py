"""
Models package for GPU OCR Server

Provides OCR model wrappers, TensorRT optimization, and result formatting.
"""

from .paddle_ocr import (
    PaddleOCRWrapper,
    TextBox,
    TextRecognition,
    OCROutput,
    ModelInfo
)

from .tensorrt_optimizer import (
    TensorRTOptimizer,
    TRTConfig,
    Shape,
    OptimizedModel,
    ValidationResult,
    BenchmarkResults
)

from .result_formatter import (
    ResultFormatter,
    FormatConfig,
    TextRegion,
    TextBlock,
    PageMetadata,
    PageResult
)

__all__ = [
    # PaddleOCR
    'PaddleOCRWrapper',
    'TextBox',
    'TextRecognition', 
    'OCROutput',
    'ModelInfo',
    
    # TensorRT
    'TensorRTOptimizer',
    'TRTConfig',
    'Shape',
    'OptimizedModel',
    'ValidationResult',
    'BenchmarkResults',
    
    # Result Formatting
    'ResultFormatter',
    'FormatConfig',
    'TextRegion',
    'TextBlock',
    'PageMetadata',
    'PageResult'
]