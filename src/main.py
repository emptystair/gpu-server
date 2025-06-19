"""
GPU OCR Server Main Application

Production-ready FastAPI application that provides high-performance OCR processing
with GPU acceleration, comprehensive monitoring, and enterprise features.
"""

import os
import sys
import time
import signal
import asyncio
import logging
import tempfile
import shutil
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
import json
from datetime import datetime
import threading

# Third-party imports
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False
    uvloop = None

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

# Set up uvloop for better async performance if available
if UVLOOP_AVAILABLE:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Local imports
from .config import Config, Environment, load_config
from .ocr_service import OCRService
from .gpu_monitor import GPUMonitor
from .utils.cache_manager import CacheManager
from .api.routes import router
from .api.middleware import setup_middleware, get_request_id
from .api.schemas import ErrorResponse

# Configure structured logging
logger = logging.getLogger(__name__)


class GlobalState:
    """Container for global application state"""
    config: Optional[Config] = None
    ocr_service: Optional[OCRService] = None
    gpu_monitor: Optional[GPUMonitor] = None
    cache_manager: Optional[CacheManager] = None
    metrics_collector: Optional[Any] = None
    startup_complete: bool = False
    shutdown_in_progress: bool = False


# Global state instance
state = GlobalState()


def configure_logging(config: Config):
    """Configure application logging based on environment"""
    
    # Determine log format
    if config.server.log_format == "json":
        # JSON structured logging for production
        import structlog
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging to use structlog
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, config.server.log_level.upper())
        )
    else:
        # Standard text logging for development
        logging.basicConfig(
            level=getattr(logging, config.server.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Add file handler if configured
    if config.server.log_file:
        from logging.handlers import RotatingFileHandler
        
        # Create log directory if needed
        log_path = Path(config.server.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add rotating file handler
        file_handler = RotatingFileHandler(
            config.server.log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5
        )
        
        if config.server.log_format == "json":
            file_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        logging.getLogger().addHandler(file_handler)
    
    # Set log levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured for {config.environment.value} environment")


def validate_environment():
    """Validate runtime environment and dependencies"""
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    
    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - GPU acceleration will be disabled")
    except ImportError:
        logger.warning("PyTorch not installed - GPU features may be limited")
    
    # Check for required system libraries
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"Required system library not found: {e}")
    
    # Validate temporary directory permissions
    temp_test = Path(tempfile.gettempdir()) / f"ocr_test_{os.getpid()}"
    try:
        temp_test.touch()
        temp_test.unlink()
    except Exception as e:
        raise RuntimeError(f"Cannot write to temporary directory: {e}")


async def initialize_gpu_monitor(config: Config) -> GPUMonitor:
    """Initialize GPU monitoring with error handling"""
    try:
        logger.info("Initializing GPU monitor...")
        monitor = GPUMonitor(config.gpu.device_id)
        await monitor.initialize()
        
        # Get initial GPU status
        status = await monitor.get_current_status()
        logger.info(
            f"GPU initialized: {status.device_name} "
            f"({status.memory.total_mb}MB memory)"
        )
        
        return monitor
        
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitor: {e}", exc_info=True)
        if config.environment == Environment.PRODUCTION:
            # In production, GPU is required
            raise
        else:
            # In development, continue without GPU
            logger.warning("Continuing without GPU monitoring")
            return None


async def initialize_cache_manager(config: Config) -> CacheManager:
    """Initialize cache manager with fallback support"""
    try:
        logger.info("Initializing cache manager...")
        cache = CacheManager(config.cache)
        
        # Test cache connectivity
        test_key = f"startup_test_{os.getpid()}"
        cache.set(test_key, {"test": True}, ttl=60)
        result = cache.get(test_key)
        cache.delete(test_key)
        
        if result:
            logger.info("Cache manager initialized successfully")
        else:
            logger.warning("Cache test failed - using memory-only cache")
        
        return cache
        
    except Exception as e:
        logger.error(f"Failed to initialize cache manager: {e}")
        # Always return a cache manager, even if degraded
        config.cache.cache_type = "memory"
        return CacheManager(config.cache)


async def initialize_ocr_service(
    config: Config,
    gpu_monitor: Optional[GPUMonitor],
    cache_manager: CacheManager
) -> OCRService:
    """Initialize OCR service with model loading and warmup"""
    try:
        logger.info("Initializing OCR service...")
        
        # Create OCR service
        service = OCRService(config, gpu_monitor=gpu_monitor, cache_manager=cache_manager)
        
        # Initialize models
        logger.info("Loading OCR models...")
        logger.info("About to call service.initialize()...")
        await service.initialize()
        logger.info("service.initialize() completed successfully")
        
        # Warm up GPU
        if gpu_monitor and config.ocr.warmup_iterations > 0:
            logger.info(f"Warming up GPU with {config.ocr.warmup_iterations} iterations...")
            
            for i in range(config.ocr.warmup_iterations):
                await service._warmup_gpu()
                logger.debug(f"Warmup iteration {i+1}/{config.ocr.warmup_iterations} complete")
        
        logger.info("OCR service initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize OCR service: {e}", exc_info=True)
        raise


async def cleanup_temp_files(config: Config):
    """Clean up old temporary files"""
    try:
        temp_paths = [
            Path(config.server.temp_path),
            Path(config.server.upload_path)
        ]
        
        current_time = time.time()
        cleaned_count = 0
        
        for temp_path in temp_paths:
            if not temp_path.exists():
                continue
                
            for file_path in temp_path.iterdir():
                if file_path.is_file():
                    # Check file age
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > config.server.cleanup_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to clean up {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")


async def periodic_cleanup(config: Config):
    """Background task for periodic cleanup"""
    while not state.shutdown_in_progress:
        try:
            await asyncio.sleep(config.server.cleanup_interval_seconds)
            await cleanup_temp_files(config)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("Starting GPU OCR Server...")
    startup_start = time.time()
    
    try:
        # Load and validate configuration
        logger.info("Loading configuration...")
        state.config = load_config()
        state.config.validate()
        
        # Configure logging
        configure_logging(state.config)
        
        # Validate environment
        validate_environment()
        
        # Initialize components
        state.gpu_monitor = await initialize_gpu_monitor(state.config)
        state.cache_manager = await initialize_cache_manager(state.config)
        state.ocr_service = await initialize_ocr_service(
            state.config,
            state.gpu_monitor,
            state.cache_manager
        )
        
        # Start background tasks
        cleanup_task = asyncio.create_task(periodic_cleanup(state.config))
        
        # Store in app state for route access
        app.state.config = state.config
        app.state.ocr_service = state.ocr_service
        app.state.gpu_monitor = state.gpu_monitor
        app.state.cache_manager = state.cache_manager
        
        # Mark startup complete
        state.startup_complete = True
        startup_time = time.time() - startup_start
        
        logger.info(
            f"GPU OCR Server started successfully in {startup_time:.2f}s",
            extra={
                "environment": state.config.environment.value,
                "gpu_available": state.gpu_monitor is not None,
                "cache_type": state.config.cache.cache_type,
                "version": state.config.version if hasattr(state.config, 'version') else "1.0.0"
            }
        )
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        # Clean up any partially initialized resources
        if state.ocr_service:
            await state.ocr_service.shutdown()
        if state.gpu_monitor:
            await state.gpu_monitor.shutdown()
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down GPU OCR Server...")
        state.shutdown_in_progress = True
        shutdown_start = time.time()
        
        try:
            # Cancel background tasks
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            
            # Shutdown services
            if state.ocr_service:
                logger.info("Shutting down OCR service...")
                await state.ocr_service.shutdown()
            
            if state.gpu_monitor:
                logger.info("Shutting down GPU monitor...")
                await state.gpu_monitor.shutdown()
            
            # Final cleanup
            await cleanup_temp_files(state.config)
            
            shutdown_time = time.time() - shutdown_start
            logger.info(f"GPU OCR Server shutdown complete in {shutdown_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


def create_app() -> FastAPI:
    """Factory function to create configured FastAPI application"""
    
    # Load initial config for app creation
    config = load_config()
    
    # Create FastAPI instance
    app = FastAPI(
        title="GPU OCR Server",
        description="High-performance OCR service with GPU acceleration",
        version=getattr(config, 'version', '1.0.0'),
        docs_url="/docs" if config.server.enable_docs else None,
        redoc_url="/redoc" if config.server.enable_docs else None,
        openapi_url="/openapi.json" if config.server.enable_docs else None,
        lifespan=lifespan
    )
    
    # Configure middleware
    state.metrics_collector = setup_middleware(app, config)
    
    # Add trusted host middleware for production
    if config.environment == Environment.PRODUCTION:
        allowed_hosts = getattr(config.server, 'allowed_hosts', ["*"])
        if allowed_hosts != ["*"]:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
    
    # Include API routes
    app.include_router(router)
    
    # Add static file serving if configured
    static_path = getattr(config.server, 'static_path', None)
    if static_path and Path(static_path).exists():
        app.mount(
            "/static",
            StaticFiles(directory=static_path),
            name="static"
        )
    
    # Configure exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors"""
        request_id = get_request_id()
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "detail": exc.errors(),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        request_id = get_request_id()
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        request_id = get_request_id()
        logger.error(
            f"Unhandled exception for request {request_id}",
            exc_info=True,
            extra={'request_id': request_id}
        )
        
        # Don't expose internal errors in production
        if config.environment == Environment.PRODUCTION:
            message = "An internal error occurred"
            detail = {}
        else:
            message = str(exc)
            detail = {"type": type(exc).__name__}
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": message,
                "detail": detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # Add root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with service information"""
        return {
            "service": "GPU OCR Server",
            "version": getattr(config, 'version', '1.0.0'),
            "environment": config.environment.value,
            "status": "operational" if state.startup_complete else "starting",
            "endpoints": {
                "health": "/api/v1/health",
                "ready": "/api/v1/ready",
                "docs": "/docs" if config.server.enable_docs else None,
                "metrics": "/api/v1/metrics"
            }
        }
    
    # Add shutdown handler
    def handle_shutdown(signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        state.shutdown_in_progress = True
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    """Main execution block for running the server directly"""
    
    # Load configuration
    config = load_config()
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "src.main:app",
        "host": config.server.host,
        "port": config.server.port,
        "workers": config.server.workers,
        "loop": "uvloop" if UVLOOP_AVAILABLE else "asyncio",
        "access_log": False,  # Use middleware logging instead
        "log_config": None,  # Use custom logging config
        "server_header": False,  # Don't expose server header
        "date_header": True,
    }
    
    # Add SSL configuration if available
    ssl_keyfile = getattr(config.server, 'ssl_keyfile', None)
    ssl_certfile = getattr(config.server, 'ssl_certfile', None)
    
    if ssl_keyfile and ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile,
        })
    
    # Add reload for development
    if config.environment == Environment.DEVELOPMENT:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    
    # Set process title
    try:
        import setproctitle
        setproctitle.setproctitle("gpu-ocr-server")
    except ImportError:
        pass
    
    # Run server
    logger.info(
        f"Starting GPU OCR Server on {config.server.host}:{config.server.port} "
        f"({config.environment.value} mode)"
    )
    
    uvicorn.run(**uvicorn_config)