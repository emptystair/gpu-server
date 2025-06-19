# TensorRT Optimization for PaddleOCR on RTX 4090

## Overview

The TensorRT optimizer provides significant performance improvements for PaddleOCR models on RTX 4090 GPUs, achieving 3-5x speedup through:

- FP16 precision with Tensor Core acceleration
- Dynamic batching support (1-50 images)
- TF32 operations for Ada Lovelace architecture
- Optimized memory allocation and caching

## Implementation Features

### 1. Model Conversion Pipeline
```
PaddlePaddle → ONNX → TensorRT Engine
```

- Converts PaddleOCR detection and recognition models separately
- Validates accuracy within 1% threshold
- Caches optimized engines to avoid re-conversion

### 2. RTX 4090 Optimizations

- **Tensor Cores**: Enabled for FP16 operations
- **TF32**: Enabled for additional speedup
- **Workspace**: 4GB allocation for complex models
- **Dynamic Shapes**: Supports variable input sizes

### 3. Dynamic Batching Configuration

#### Detection Model
- Min: 1×3×640×640
- Optimal: 16×3×960×960
- Max: 50×3×1920×1080

#### Recognition Model
- Min: 1×3×48×320
- Optimal: 16×3×48×320
- Max: 50×3×48×320

## Performance Expectations

### Single Image Inference
- Detection: ~50ms → ~15ms (3.3x speedup)
- Recognition: ~20ms → ~5ms (4x speedup)

### Batch Processing (16 images)
- Detection: ~800ms → ~150ms (5.3x speedup)
- Recognition: ~320ms → ~80ms (4x speedup)

### Memory Usage
- Base allocation: 500MB
- Per-image overhead: ~50MB
- Maximum (batch 50): ~3GB

## Usage Example

```python
from src.models.tensorrt_optimizer import TensorRTOptimizer

# Initialize optimizer
optimizer = TensorRTOptimizer(precision="FP16")

# Optimize detection model
det_result = optimizer.optimize_model(
    paddle_model_path="/path/to/det_model",
    output_path="/path/to/det.engine",
    model_type="detection"
)

# Optimize recognition model
rec_result = optimizer.optimize_model(
    paddle_model_path="/path/to/rec_model",
    output_path="/path/to/rec.engine",
    model_type="recognition"
)

# Check results
if det_result:
    print(f"Detection speedup: {det_result.speedup_factor}x")
    print(f"Accuracy retained: {det_result.optimized_accuracy * 100}%")
```

## Integration with PaddleOCR

When PaddleOCR is initialized with `use_tensorrt=True`, it automatically:

1. Checks for cached TensorRT engines
2. Falls back to standard inference if not available
3. Uses optimized engines for inference

```python
from paddleocr import PaddleOCR

# This will use TensorRT if available
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    use_tensorrt=True,
    precision='fp16'
)
```

## Fallback Behavior

The optimizer handles various failure scenarios gracefully:

- **TensorRT not installed**: Returns None, PaddleOCR uses standard inference
- **ONNX conversion fails**: Logs error and returns None
- **Validation fails**: Removes optimized engine and returns None
- **GPU OOM**: Reduces batch size automatically

## Cache Management

Optimized engines are cached based on:
- Model path
- Model type (detection/recognition)
- Precision setting
- Configuration hash

Cache location: `/app/model_cache/tensorrt/`

## Benchmarking

Use the benchmark method to measure performance:

```python
results = optimizer.benchmark_performance(
    engine_path="/path/to/engine",
    test_batch_sizes=[1, 8, 16, 32, 50]
)

for i, batch_size in enumerate(results.batch_sizes):
    print(f"Batch {batch_size}: {results.throughputs[i]:.1f} img/s")
```

## Notes

- FP8 precision is not available in CUDA 12.2
- INT8 requires calibration dataset (not implemented)
- TensorRT engines are GPU-specific (RTX 4090 only)
- Re-optimization needed for different GPU models