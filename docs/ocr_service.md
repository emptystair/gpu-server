# OCR Service Documentation

## Overview

The OCR Service is the main orchestration layer for document processing in the GPU OCR Server. It coordinates between PDF/image processing, GPU monitoring, batch optimization, and result formatting to provide high-throughput OCR with RTX 4090 optimization.

## Features

- **Multi-format Support**: Process PDFs and images (JPEG, PNG, etc.)
- **Dynamic Batch Sizing**: Automatically adjusts batch size based on GPU memory
- **Processing Strategies**: SPEED, ACCURACY, and BALANCED modes
- **Result Caching**: Avoid reprocessing identical documents
- **GPU Optimization**: TensorRT acceleration and memory management
- **Error Recovery**: Graceful degradation with retry logic
- **Performance Tracking**: Comprehensive statistics and monitoring

## Architecture

```
OCRService
├── PDFProcessor       - Extract pages from PDFs
├── ImageProcessor     - Enhance images for OCR
├── GPUMonitor        - Track GPU resources
├── PaddleOCRWrapper  - Perform text detection/recognition
├── TensorRTOptimizer - Accelerate models
├── ResultFormatter   - Structure output
└── CacheManager      - Cache results
```

## Usage

### Basic Example

```python
from src.config import load_config
from src.ocr_service import OCRService, ProcessingRequest, ProcessingStrategy

# Initialize service
config = load_config()
service = OCRService(config)
await service.initialize()

# Create processing request
request = ProcessingRequest(
    document_path="/path/to/document.pdf",
    strategy=ProcessingStrategy.BALANCED,
    language="en",
    dpi=150,
    confidence_threshold=0.5
)

# Process document
result = await service.process_document(request)

# Access results
print(f"Average confidence: {result.average_confidence:.2%}")
print(f"Processing time: {result.processing_time_ms:.0f}ms")
print(f"Strategy used: {result.strategy_used.value}")

for page in result.pages:
    print(f"Page {page.page_number}: {page.text}")
    print(f"Confidence: {page.confidence:.2%}")
```

### API Changes (Updated)

The OCRService now uses a cleaner API design:

1. **Input**: Accepts `ProcessingRequest` objects instead of raw parameters
2. **Output**: Returns `ProcessingResult` objects with structured data
3. **File Reading**: Service handles file I/O internally from the provided path

#### ProcessingRequest Fields:
- `document_path`: Path to the document file
- `strategy`: Processing strategy (SPEED, BALANCED, ACCURACY)
- `language`: OCR language code
- `dpi`: DPI for PDF extraction
- `confidence_threshold`: Minimum confidence threshold
- `enable_angle_classification`: Enable text angle detection
- `preprocessing_options`: Image preprocessing settings
- `page_range`: Page selection for PDFs (e.g., "1-5,7,9-12")

#### ProcessingResult Fields:
- `pages`: List of PageResult objects
- `average_confidence`: Overall confidence score
- `processing_time_ms`: Total processing time
- `strategy_used`: Strategy that was applied
- `metadata`: Additional processing metadata

### Service Lifecycle

```python
# Check service status
if service.initialized:
    print("Service is initialized")

if service.is_ready():
    print("Service is ready for processing")

if service.is_initializing():
    print("Service is currently initializing")

# Get queue size (for future queue implementation)
queue_size = service.get_queue_size()

# Shutdown service
await service.shutdown()

# Clean up
await service.cleanup()
```

### Processing Strategies

#### SPEED Mode
- **DPI**: 120
- **Batch Size**: 1.5x larger
- **Image Enhancement**: Disabled
- **Use Case**: High-volume processing where speed is critical

#### ACCURACY Mode
- **DPI**: 200
- **Batch Size**: 0.7x smaller
- **Image Enhancement**: Enabled
- **Use Case**: Documents requiring high accuracy

#### BALANCED Mode
- **DPI**: 150
- **Batch Size**: Standard
- **Image Enhancement**: Enabled
- **Use Case**: General purpose processing

### Custom Configuration

```python
# Override DPI
result = await service.process_document(
    file_bytes=document,
    file_type="application/pdf",
    processing_strategy=ProcessingStrategy.BALANCED,
    dpi=300  # Custom DPI
)

# Additional options can be passed via kwargs
result = await service.process_document(
    file_bytes=document,
    file_type="image/jpeg",
    processing_strategy=ProcessingStrategy.ACCURACY,
    language="en",
    confidence_threshold=0.8
)
```

## Data Structures

### OCRResult
Complete processing results:
```python
@dataclass
class OCRResult:
    pages: List[PageResult]
    total_pages: int
    processing_time_ms: float
    confidence_score: float
    metadata: Dict[str, Any]
    strategy_used: ProcessingStrategy
    batch_sizes_used: List[int]
```

### PageResult
Individual page results:
```python
@dataclass
class PageResult:
    page_number: int
    text: str
    words: List[WordResult]
    confidence: float
    processing_time_ms: float
    image_size: Tuple[int, int]
```

### WordResult
Word-level detection:
```python
@dataclass
class WordResult:
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
```

## Performance Optimization

### Dynamic Batch Sizing

The service automatically determines optimal batch size based on:
1. Available GPU memory
2. Image dimensions
3. Processing strategy
4. Current GPU utilization

Formula:
```
available_memory = gpu_free_memory - buffer
memory_per_image = width * height * 3 * 4 + model_overhead
base_batch_size = available_memory / memory_per_image
final_batch_size = base_batch_size * strategy_multiplier
```

### Memory Management

- **Buffer**: 500MB reserved for system stability
- **Adaptive Sizing**: Reduces batch size on memory pressure
- **Estimation**: Accounts for model overhead (~350MB base + per-image)

### Caching

Results are cached based on:
- Document content hash (MD5)
- Processing strategy
- Cache key format: `{content_hash}_{strategy}`

## Statistics and Monitoring

Access processing statistics:
```python
stats = service.get_processing_stats()
print(f"Documents processed: {stats.total_documents_processed}")
print(f"Average pages/second: {stats.average_pages_per_second:.2f}")
print(f"GPU utilization: {stats.gpu_utilization_average:.1f}%")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
```

## Error Handling

The service implements several error recovery mechanisms:

1. **Batch Size Reduction**: On OOM errors, automatically reduces batch size
2. **Retry Logic**: Failed batches retry with smaller sizes
3. **Partial Results**: Returns processed pages even if some fail
4. **Graceful Degradation**: Falls back to single-image processing if needed

## Performance Expectations

On RTX 4090 with TensorRT optimization:

| Strategy | Pages/Minute | Batch Size | GPU Utilization |
|----------|--------------|------------|-----------------|
| SPEED    | 100-120      | 32-50      | 85-95%         |
| BALANCED | 80-100       | 16-32      | 80-90%         |
| ACCURACY | 50-70        | 8-16       | 70-80%         |

## Best Practices

1. **Initialize Once**: Initialize the service at application startup
2. **Reuse Service**: Don't create multiple instances
3. **Monitor Stats**: Track performance metrics for optimization
4. **Handle Errors**: Implement proper error handling for production
5. **Clean Up**: Always call cleanup() when shutting down

## Troubleshooting

### Low Throughput
- Check GPU utilization with statistics
- Verify TensorRT optimization is enabled
- Ensure adequate GPU memory available
- Consider using SPEED strategy

### High Error Rate
- Check input document quality
- Verify GPU memory isn't exhausted
- Review error logs for patterns
- Consider reducing max batch size

### Memory Issues
- Increase GPU memory buffer
- Reduce maximum batch size
- Enable more aggressive cleanup
- Monitor memory usage trends