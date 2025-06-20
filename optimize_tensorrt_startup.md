# Optimizing TensorRT Startup Time

## Current Situation
- TensorRT initialization takes ~8 minutes on first startup
- Shape files are persisted in `/root/.paddleocr` via Docker volume
- However, TensorRT still takes long to initialize on each restart

## Ways to Speed Up TensorRT Initialization

### 1. **Enable TensorRT Engine Caching**
TensorRT can serialize optimized engines to disk and reload them. We need to:

```python
# Add to paddle_ocr.py configuration
'use_static_optimization': True,  # Enable static graph optimization
'tensorrt_engine_file': '/root/.paddleocr/tensorrt_engines/',  # Engine cache directory
```

### 2. **Pre-build TensorRT Engines**
Create a startup script that builds engines once:

```bash
# Run once to build and cache engines
docker exec gpu-ocr-server python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, use_tensorrt=True)
# Process a sample to trigger engine building
import numpy as np
dummy_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
ocr.ocr(dummy_img, cls=True)
print('TensorRT engines built and cached')
"
```

### 3. **Mount TensorRT Cache Directory**
Add to docker-compose.yml:

```yaml
volumes:
  - tensorrt-engines:/root/.paddleocr/tensorrt_engines
  - paddle-cache:/tmp/paddle
```

### 4. **Use Fixed Input Shapes**
Instead of dynamic shapes, use fixed shapes for faster initialization:

```python
'tensorrt_use_static_engine': True,
'tensorrt_shape_info_filename': '/root/.paddleocr/shape_info.txt',
```

### 5. **Environment Variables for Caching**
Set these environment variables in docker-compose.yml:

```yaml
environment:
  - CUDA_CACHE_PATH=/root/.paddleocr/cuda_cache
  - FLAGS_use_tensorrt_engine_cache=1
  - FLAGS_tensorrt_engine_cache_dir=/root/.paddleocr/tensorrt_engines
```

### 6. **Reduce Warmup Iterations**
Modify the warmup to be minimal when engines are cached:

```python
def _warmup_model(self):
    if self._check_tensorrt_engines_exist():
        # Minimal warmup when engines are cached
        logger.info("TensorRT engines found, performing minimal warmup")
        dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        _ = self.ocr.ocr(dummy_img, cls=False)  # Skip angle classification
        return
```

### 7. **Use TensorRT Builder Cache**
Enable TensorRT's builder cache to reuse optimization results:

```yaml
environment:
  - FLAGS_trt_ibuilder_cache=/root/.paddleocr/trt_builder_cache
```

## Recommended Implementation

1. **Persist Additional Directories**:
   - `/tmp/paddle` - PaddlePaddle's temp cache
   - `/root/.cache` - General cache directory
   - Any TensorRT engine directories

2. **Create Engine Pre-build Script**:
   ```bash
   docker exec gpu-ocr-server python scripts/prebuild_tensorrt_engines.py
   ```

3. **Check Engine Cache on Startup**:
   ```python
   if os.path.exists('/root/.paddleocr/tensorrt_engines/'):
       logger.info("TensorRT engine cache found, fast startup enabled")
   ```

4. **Monitor Startup Time**:
   Add timing logs to track improvement:
   ```python
   start_time = time.time()
   self.ocr = PaddleOCR(**config)
   logger.info(f"PaddleOCR initialized in {time.time() - start_time:.2f} seconds")
   ```

## Expected Results
- First startup: 8 minutes (building engines)
- Subsequent startups: 30-60 seconds (loading cached engines)