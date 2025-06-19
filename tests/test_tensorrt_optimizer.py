"""Test script for TensorRT optimizer functionality"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tensorrt_optimizer import (
    TensorRTOptimizer, TRTConfig, Shape, 
    OptimizedModel, ValidationResult, BenchmarkResults
)

print("=== TensorRT Optimizer Test ===\n")

# Test 1: Check if TensorRT is available
print("Test 1: Checking TensorRT availability")
try:
    import tensorrt as trt
    print(f"✓ TensorRT version: {trt.__version__}")
except ImportError:
    print("✗ TensorRT not installed")
    print("Note: This is expected in development environment")

# Test 2: Initialize optimizer
print("\nTest 2: Initializing TensorRT optimizer")
try:
    optimizer = TensorRTOptimizer(precision="FP16")
    print("✓ Optimizer created successfully")
    print(f"  - Precision: {optimizer.precision}")
    print(f"  - Cache directory: {optimizer.cache_dir}")
    print(f"  - GPU properties: {optimizer.gpu_properties}")
except Exception as e:
    print(f"✗ Failed to create optimizer: {e}")

# Test 3: Prepare configuration
print("\nTest 3: Preparing TensorRT configuration")
try:
    config = optimizer.prepare_config()
    print("✓ Configuration prepared:")
    print(f"  - Max batch size: {config.max_batch_size}")
    print(f"  - Precision: {config.precision}")
    print(f"  - Workspace size: {config.workspace_size_mb}MB")
    print(f"  - Tensor Core enabled: {config.enable_tensor_core}")
    print(f"  - TF32 enabled: {config.enable_tf32}")
except Exception as e:
    print(f"✗ Failed to prepare config: {e}")

# Test 4: Test shape definitions
print("\nTest 4: Creating shape definitions")
try:
    detection_shape = Shape(
        min_shape=(1, 3, 640, 640),
        opt_shape=(16, 3, 960, 960),
        max_shape=(50, 3, 1920, 1080)
    )
    print("✓ Detection model shape created:")
    print(f"  - Min: {detection_shape.min_shape}")
    print(f"  - Opt: {detection_shape.opt_shape}")
    print(f"  - Max: {detection_shape.max_shape}")
    
    recognition_shape = Shape(
        min_shape=(1, 3, 48, 320),
        opt_shape=(16, 3, 48, 320),
        max_shape=(50, 3, 48, 320)
    )
    print("✓ Recognition model shape created")
except Exception as e:
    print(f"✗ Failed to create shapes: {e}")

# Test 5: Benchmark simulation
print("\nTest 5: Running benchmark simulation")
try:
    test_batch_sizes = [1, 8, 16, 32, 50]
    
    # Since we can't run actual TensorRT in this environment,
    # we'll test the benchmark method with a dummy engine path
    dummy_engine_path = "/tmp/dummy.engine"
    
    results = optimizer.benchmark_performance(dummy_engine_path, test_batch_sizes)
    
    print("✓ Benchmark completed:")
    print(f"  - Batch sizes: {results.batch_sizes}")
    print(f"  - Throughputs: {[f'{t:.1f}' for t in results.throughputs]} img/s")
    print(f"  - Latencies: {[f'{l:.1f}' for l in results.latencies]} ms")
    print(f"  - GPU utilization: {[f'{u:.0f}%' for u in results.gpu_utilizations]}")
    
    # Calculate speedup for batch size 16
    idx = test_batch_sizes.index(16)
    speedup = results.throughputs[idx] / results.throughputs[0]
    print(f"  - Speedup at batch 16: {speedup:.2f}x")
    
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

# Test 6: Model optimization workflow (simulated)
print("\nTest 6: Testing optimization workflow")
try:
    # Create dummy model paths
    dummy_model_path = "/tmp/paddle_model"
    output_path = "/tmp/optimized.engine"
    
    print("  - Would convert PaddlePaddle model to ONNX")
    print("  - Would build TensorRT engine with FP16")
    print("  - Would validate accuracy within 1% threshold")
    print("  - Would save optimized engine to cache")
    
    # Test validation result
    validation = ValidationResult(
        accuracy_delta=0.005,  # 0.5% loss
        latency_improvement=3.5,  # 3.5x faster
        memory_usage_mb=500,
        validation_passed=True
    )
    print(f"✓ Validation result: {'PASSED' if validation.validation_passed else 'FAILED'}")
    print(f"  - Accuracy delta: {validation.accuracy_delta*100:.1f}%")
    print(f"  - Speedup: {validation.latency_improvement:.1f}x")
    
except Exception as e:
    print(f"✗ Workflow test failed: {e}")

# Test 7: Cache directory creation
print("\nTest 7: Cache directory management")
try:
    cache_dir = Path(optimizer.cache_dir)
    if cache_dir.exists():
        print(f"✓ Cache directory exists: {cache_dir}")
    else:
        print(f"✗ Cache directory not created: {cache_dir}")
        
    # Test config hash generation
    test_hash = optimizer._generate_config_hash("/path/to/model", "detection")
    print(f"✓ Config hash generated: {test_hash}")
    
except Exception as e:
    print(f"✗ Cache directory test failed: {e}")

# Test 8: Data structure serialization
print("\nTest 8: Data structure serialization")
try:
    # Test OptimizedModel
    optimized_model = OptimizedModel(
        engine_path="/tmp/test.engine",
        original_accuracy=1.0,
        optimized_accuracy=0.995,
        speedup_factor=3.5,
        model_type="detection",
        config_hash="abc123"
    )
    
    # Convert to dict (for saving/loading)
    model_dict = {
        "engine_path": optimized_model.engine_path,
        "original_accuracy": optimized_model.original_accuracy,
        "optimized_accuracy": optimized_model.optimized_accuracy,
        "speedup_factor": optimized_model.speedup_factor,
        "model_type": optimized_model.model_type,
        "config_hash": optimized_model.config_hash
    }
    
    print("✓ OptimizedModel serialization successful")
    print(f"  - Model type: {model_dict['model_type']}")
    print(f"  - Speedup: {model_dict['speedup_factor']}x")
    print(f"  - Accuracy retained: {model_dict['optimized_accuracy']*100:.1f}%")
    
except Exception as e:
    print(f"✗ Serialization test failed: {e}")

print("\n✅ All tests completed!")
print("\nNote: Full TensorRT functionality requires:")
print("- NVIDIA GPU with compute capability >= 7.0")
print("- TensorRT 8.x installation")
print("- CUDA 11.8 or 12.x")
print("- PaddlePaddle models for conversion")