"""
Example: GPU OCR Server API

This example demonstrates how to set up and run the GPU OCR Server
with all middleware and routes configured.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.api import router, lifespan, setup_middleware


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Load configuration
    config = load_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="GPU OCR Server",
        description="High-performance OCR service with GPU acceleration",
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Setup middleware stack
    metrics_collector = setup_middleware(app, config)
    
    # Add trusted host middleware for security
    if config.server.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.server.allowed_hosts
        )
    
    # Include API routes
    app.include_router(router)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "GPU OCR Server",
            "version": config.version,
            "status": "operational",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    # Add metrics endpoint at app level
    @app.get("/api/v1/metrics/summary")
    async def metrics_summary():
        """Get middleware metrics summary"""
        metrics = await metrics_collector.get_metrics()
        return metrics
    
    return app


def main():
    """Run the API server"""
    
    # Create application
    app = create_app()
    
    # Load configuration for server settings
    config = load_config()
    
    # Configure uvicorn
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        loop="asyncio",
        log_level="info",
        access_log=True,
        use_colors=True,
        # SSL configuration (if needed)
        ssl_keyfile=config.server.ssl_keyfile if hasattr(config.server, 'ssl_keyfile') else None,
        ssl_certfile=config.server.ssl_certfile if hasattr(config.server, 'ssl_certfile') else None,
    )
    
    # Create and run server
    server = uvicorn.Server(uvicorn_config)
    server.run()


if __name__ == "__main__":
    main()