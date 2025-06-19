
import os
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class OCRModel(str, Enum):
    PADDLE_OCR = "paddle_ocr"
    PADDLE_OCR_TENSORRT = "paddle_ocr_tensorrt"


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
    
    # Additional OCR settings
    use_angle_cls: bool = True
    lang: str = "en"
    use_tensorrt: bool = True  # Enable TensorRT for GPU acceleration
    use_space_char: bool = True
    paddle_det_model_dir: Optional[str] = None
    paddle_rec_model_dir: Optional[str] = None
    paddle_cls_model_dir: Optional[str] = None
    
    # Image processing
    max_image_size: int = 8192  # RTX 4090 optimized
    image_quality: int = 95
    supported_image_formats: List[str] = None
    
    # PDF processing
    pdf_max_pages: int = 200  # RTX 4090 optimized
    pdf_timeout: int = 300
    pdf_parallel_pages: int = 8
    max_file_size_mb: int = 100  # Maximum file size in MB
    max_pages: int = 200  # Maximum pages to process
    
    def __post_init__(self):
        if self.supported_image_formats is None:
            self.supported_image_formats = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]


@dataclass
class ServerConfig:
    """FastAPI server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 300
    
    # Additional server settings
    api_version: str = "v1"
    enable_docs: bool = True
    cors_origins: List[str] = None
    cors_allow_credentials: bool = False
    rate_limit_per_minute: int = 120  # RTX 4090 optimized
    
    # Storage
    upload_path: str = "./uploads"
    temp_path: str = "./temp"
    cleanup_interval_seconds: int = 3600
    cleanup_age_seconds: int = 86400
    
    # Security
    secret_key: str = "change-me-in-production"
    enable_api_key: bool = False
    api_key_header: str = "X-API-Key"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        self.api_prefix = f"/api/{self.api_version}"


@dataclass
class GPUConfig:
    """GPU monitoring configuration"""
    device_id: int = 0
    monitoring_interval_seconds: float = 1.0
    memory_threshold_percent: float = 85.0
    
    # RTX 4090 optimizations
    memory_fraction: float = 0.9  # Use up to 90% of 24GB VRAM
    enable_monitoring: bool = True
    tensorrt_workspace_size: int = 8589934592  # 8GB
    tensorrt_fp16_mode: bool = True
    tensorrt_int8_mode: bool = False
    tensorrt_max_batch_size: int = 32
    
    # Performance settings
    enable_mixed_precision: bool = True
    cudnn_benchmark: bool = True
    num_cpu_threads: int = 16
    
    # CUDA settings
    cuda_device_order: str = "PCI_BUS_ID"
    cuda_visible_devices: str = "0"


@dataclass
class CacheConfig:
    """Cache configuration"""
    enable_cache: bool = True
    cache_type: str = "memory"  # memory or redis
    cache_ttl: int = 3600
    cache_max_size: int = 5000
    
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Multi-tier cache settings
    memory_cache_size_mb: int = 512
    disk_cache_size_gb: float = 10.0
    disk_cache_path: str = "./cache"
    
    # AWS S3 cache settings
    s3_cache_bucket: Optional[str] = None
    s3_cache_region: Optional[str] = None


class Config:
    """Unified configuration container"""
    def __init__(self, 
                 ocr: OCRConfig = None,
                 server: ServerConfig = None,
                 gpu: GPUConfig = None,
                 cache: CacheConfig = None,
                 environment: Environment = None):
        self.ocr = ocr or OCRConfig()
        self.server = server or ServerConfig()
        self.gpu = gpu or GPUConfig()
        self.cache = cache or CacheConfig()
        self.environment = environment or Environment.DEVELOPMENT
        self.version = os.getenv("APP_VERSION", "1.0.0")
        
        # Set debug mode based on environment
        self.debug = self.environment == Environment.DEVELOPMENT
        
        # Update server settings based on environment
        if self.environment == Environment.PRODUCTION:
            self.server.enable_docs = False
            self.server.log_level = "INFO"
            self.server.log_format = "json"
        else:
            self.server.enable_docs = True
            self.server.log_level = "DEBUG"
            self.server.log_format = "text"
    
    def validate(self):
        """Validate configuration settings"""
        # Ensure required directories exist
        paths = [
            self.ocr.model_cache_dir,
            self.server.upload_path,
            self.server.temp_path,
            "./tensorrt_engines"  # TensorRT cache path
        ]
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Validate GPU settings
        if self.gpu.memory_fraction < 0 or self.gpu.memory_fraction > 1:
            raise ValueError("GPU memory_fraction must be between 0 and 1")
        
        if self.gpu.memory_threshold_percent < 0 or self.gpu.memory_threshold_percent > 100:
            raise ValueError("GPU memory_threshold_percent must be between 0 and 100")
        
        # Validate cache settings
        if self.cache.cache_type not in ["memory", "redis"]:
            raise ValueError("cache_type must be 'memory' or 'redis'")
        
        # Validate environment-specific settings
        if self.environment == Environment.PRODUCTION:
            if self.server.secret_key == "change-me-in-production":
                raise ValueError("secret_key must be changed in production")
            if self.debug:
                raise ValueError("debug must be False in production")
            
        # Validate server settings
        if self.server.max_request_size_mb > 1000:
            raise ValueError("max_request_size_mb cannot exceed 1000MB")
            
        # Validate OCR settings
        if self.ocr.default_dpi < 72 or self.ocr.default_dpi > 600:
            raise ValueError("default_dpi must be between 72 and 600")
    
    def get_model_config(self) -> dict:
        """Get OCR model configuration optimized for RTX 4090"""
        return {
            "use_angle_cls": self.ocr.use_angle_cls,
            "lang": self.ocr.lang,
            "use_gpu": True,
            "gpu_id": self.gpu.device_id,
            "use_space_char": self.ocr.use_space_char,
            "max_batch_size": self.ocr.max_batch_size,
            "use_tensorrt": self.ocr.use_tensorrt,
            "precision": "fp16" if self.gpu.tensorrt_fp16_mode else "fp32",
            "gpu_mem": self.gpu.memory_fraction,
            "enable_mkldnn": False,  # Disable CPU optimizations when using GPU
            "cpu_threads": self.gpu.num_cpu_threads,
            "det_model_dir": self.ocr.paddle_det_model_dir,
            "rec_model_dir": self.ocr.paddle_rec_model_dir,
            "cls_model_dir": self.ocr.paddle_cls_model_dir,
        }


def load_config() -> Config:
    """Purpose: Load configuration from environment and files
    Calls:
        - os.environ.get()
        - Load from config.yaml if exists
    Called by: Various initialization functions
    Priority: CORE
    """
    # Determine environment
    env_str = os.getenv("ENVIRONMENT", "development")
    environment = Environment(env_str.lower())
    
    # Load OCR configuration
    ocr_config = OCRConfig(
        default_dpi=int(os.getenv("PDF_DPI", "120")),
        max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "50")),
        min_batch_size=int(os.getenv("MIN_BATCH_SIZE", "1")),
        gpu_memory_buffer_mb=int(os.getenv("GPU_MEMORY_BUFFER_MB", "500")),
        tensorrt_precision=os.getenv("TENSORRT_PRECISION", "FP16"),
        model_cache_dir=os.getenv("MODEL_CACHE_DIR", "./model_cache"),
        warmup_iterations=int(os.getenv("WARMUP_ITERATIONS", "3")),
        use_angle_cls=os.getenv("PADDLE_USE_ANGLE_CLS", "true").lower() == "true",
        lang=os.getenv("PADDLE_LANG", "en"),
        use_tensorrt=os.getenv("USE_TENSORRT", "true").lower() == "true",
        use_space_char=os.getenv("PADDLE_USE_SPACE_CHAR", "true").lower() == "true",
        paddle_det_model_dir=os.getenv("PADDLE_DET_MODEL_DIR"),
        paddle_rec_model_dir=os.getenv("PADDLE_REC_MODEL_DIR"),
        paddle_cls_model_dir=os.getenv("PADDLE_CLS_MODEL_DIR"),
        max_image_size=int(os.getenv("MAX_IMAGE_SIZE", "8192")),
        image_quality=int(os.getenv("IMAGE_QUALITY", "95")),
        pdf_max_pages=int(os.getenv("PDF_MAX_PAGES", "200")),
        pdf_timeout=int(os.getenv("PDF_TIMEOUT", "300")),
        pdf_parallel_pages=int(os.getenv("PDF_PARALLEL_PAGES", "8")),
    )
    
    # Load server configuration
    server_config = ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        max_request_size_mb=int(os.getenv("MAX_REQUEST_SIZE_MB", "500")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT", "300")),
        api_version=os.getenv("API_VERSION", "v1"),
        enable_docs=environment != Environment.PRODUCTION,
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"],
        cors_allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")),
        upload_path=os.getenv("UPLOAD_PATH", "./uploads"),
        temp_path=os.getenv("TEMP_PATH", "./temp"),
        cleanup_interval_seconds=int(os.getenv("CLEANUP_INTERVAL", "3600")),
        cleanup_age_seconds=int(os.getenv("CLEANUP_AGE", "86400")),
        secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
        enable_api_key=os.getenv("ENABLE_API_KEY", "false").lower() == "true",
        api_key_header=os.getenv("API_KEY_HEADER", "X-API-Key"),
        log_level=os.getenv("LOG_LEVEL", "INFO" if environment == Environment.PRODUCTION else "DEBUG"),
        log_format=os.getenv("LOG_FORMAT", "json" if environment == Environment.PRODUCTION else "text"),
        log_file=os.getenv("LOG_FILE"),
    )
    
    # Load GPU configuration
    gpu_config = GPUConfig(
        device_id=int(os.getenv("GPU_DEVICE_ID", "0")),
        monitoring_interval_seconds=float(os.getenv("GPU_MONITOR_INTERVAL", "1.0")),
        memory_threshold_percent=float(os.getenv("GPU_MEMORY_THRESHOLD", "85.0")),
        memory_fraction=float(os.getenv("GPU_MEMORY_FRACTION", "0.9")),
        enable_monitoring=os.getenv("ENABLE_GPU_MONITORING", "true").lower() == "true",
        tensorrt_workspace_size=int(os.getenv("TENSORRT_WORKSPACE_SIZE", "8589934592")),
        tensorrt_fp16_mode=os.getenv("TENSORRT_FP16_MODE", "true").lower() == "true",
        tensorrt_int8_mode=os.getenv("TENSORRT_INT8_MODE", "false").lower() == "true",
        tensorrt_max_batch_size=int(os.getenv("TENSORRT_MAX_BATCH_SIZE", "32")),
        enable_mixed_precision=os.getenv("ENABLE_MIXED_PRECISION", "true").lower() == "true",
        cudnn_benchmark=os.getenv("CUDNN_BENCHMARK", "true").lower() == "true",
        num_cpu_threads=int(os.getenv("NUM_THREADS", "16")),
        cuda_device_order=os.getenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    )
    
    # Load cache configuration
    cache_config = CacheConfig(
        enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
        cache_type=os.getenv("CACHE_TYPE", "memory"),
        cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "5000")),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_password=os.getenv("REDIS_PASSWORD"),
        memory_cache_size_mb=int(os.getenv("MEMORY_CACHE_SIZE_MB", "512")),
        disk_cache_size_gb=float(os.getenv("DISK_CACHE_SIZE_GB", "10.0")),
        disk_cache_path=os.getenv("DISK_CACHE_PATH", "./cache"),
        s3_cache_bucket=os.getenv("S3_CACHE_BUCKET"),
        s3_cache_region=os.getenv("S3_CACHE_REGION"),
    )
    
    # Create unified config
    config = Config(
        ocr=ocr_config,
        server=server_config,
        gpu=gpu_config,
        cache=cache_config,
        environment=environment
    )
    
    # Validate configuration
    config.validate()
    
    return config


# Create global config instance
config = load_config()