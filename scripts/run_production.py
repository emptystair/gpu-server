#!/usr/bin/env python3
"""
Production deployment script for GPU OCR Server

This script handles production deployment with proper process management,
monitoring, and graceful shutdown capabilities.
"""

import os
import sys
import time
import signal
import psutil
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, Environment


def check_system_requirements():
    """Verify system meets requirements for production deployment"""
    
    errors = []
    warnings = []
    
    # Check CPU cores
    cpu_count = psutil.cpu_count()
    if cpu_count < 8:
        warnings.append(f"Low CPU count ({cpu_count}). Recommended: 8+ cores")
    
    # Check RAM
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    if total_gb < 16:
        errors.append(f"Insufficient RAM ({total_gb:.1f}GB). Required: 16GB+")
    
    # Check available disk space
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    if free_gb < 50:
        warnings.append(f"Low disk space ({free_gb:.1f}GB free). Recommended: 50GB+")
    
    # Check GPU
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("No CUDA-capable GPU detected")
        else:
            # Check GPU memory
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            if gpu_memory_gb < 8:
                errors.append(f"Insufficient GPU memory ({gpu_memory_gb:.1f}GB). Required: 8GB+")
    except ImportError:
        errors.append("PyTorch not installed")
    
    # Check required directories
    required_dirs = [
        project_root / "model_cache",
        project_root / "tensorrt_engines",
        project_root / "logs",
        project_root / "cache",
        project_root / "uploads",
        project_root / "temp"
    ]
    
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directory {dir_path}: {e}")
    
    # Check port availability
    config = load_config()
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((config.server.host, config.server.port))
        sock.close()
    except OSError:
        errors.append(f"Port {config.server.port} is already in use")
    
    # Print results
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
        print()
    
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ❌ {error}")
        print("\nProduction deployment cannot proceed. Fix the above errors.")
        sys.exit(1)
    
    print("✅ System requirements check passed")


def set_process_limits():
    """Set process limits for production"""
    try:
        import resource
        
        # Set file descriptor limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
        
        # Set core dump size (0 to disable)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        
        print("✅ Process limits configured")
        
    except Exception as e:
        print(f"⚠️  Failed to set process limits: {e}")


def configure_production_environment():
    """Set production environment variables"""
    
    # Force production mode
    os.environ['ENVIRONMENT'] = 'production'
    
    # CUDA settings for optimal performance
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # TensorFlow/PyTorch settings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # OpenMP settings
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    
    # Malloc settings for better performance
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
    
    print("✅ Production environment configured")


def create_pid_file():
    """Create PID file for process management"""
    pid_file = project_root / "gpu-ocr-server.pid"
    
    # Check if already running
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if psutil.pid_exists(old_pid):
                print(f"❌ Server already running (PID: {old_pid})")
                sys.exit(1)
        except:
            pass
    
    # Write current PID
    pid_file.write_text(str(os.getpid()))
    
    # Register cleanup
    import atexit
    atexit.register(lambda: pid_file.unlink(missing_ok=True))
    
    print(f"✅ PID file created: {pid_file}")


def setup_monitoring():
    """Set up production monitoring"""
    
    # Configure Prometheus metrics if available
    try:
        from prometheus_client import start_http_server
        metrics_port = 9090
        start_http_server(metrics_port)
        print(f"✅ Prometheus metrics available on port {metrics_port}")
    except ImportError:
        print("ℹ️  Prometheus client not installed, metrics endpoint disabled")
    
    # Set up health check endpoint monitoring
    def monitor_health():
        import requests
        import time
        
        config = load_config()
        health_url = f"http://localhost:{config.server.port}/api/v1/health"
        
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                response = requests.get(health_url, timeout=5)
                if response.status_code != 200:
                    logging.error(f"Health check failed: {response.status_code}")
            except Exception as e:
                logging.error(f"Health check error: {e}")
    
    # Start health monitoring in background thread
    import threading
    health_thread = threading.Thread(target=monitor_health, daemon=True)
    health_thread.start()


def run_server():
    """Run the production server with uvicorn"""
    
    config = load_config()
    
    # Import uvicorn
    import uvicorn
    
    # Configure uvicorn for production
    uvicorn_config = uvicorn.Config(
        app="src.main:app",
        host=config.server.host,
        port=config.server.port,
        workers=1,  # Single worker for GPU (multi-worker doesn't share GPU well)
        loop="uvloop",
        log_level="info",
        access_log=False,  # Disable access log (using middleware)
        use_colors=False,  # No colors in production logs
        server_header=False,  # Don't expose server info
        date_header=True,
        # Timeouts
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        # Limits
        limit_concurrency=1000,
        limit_max_requests=10000,  # Restart worker after 10k requests
    )
    
    # Add SSL if configured
    if hasattr(config.server, 'ssl_keyfile') and hasattr(config.server, 'ssl_certfile'):
        uvicorn_config.ssl_keyfile = config.server.ssl_keyfile
        uvicorn_config.ssl_certfile = config.server.ssl_certfile
    
    # Create server
    server = uvicorn.Server(uvicorn_config)
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, initiating graceful shutdown...")
        server.should_exit = True
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run server
    print(f"\n🚀 GPU OCR Server starting in production mode")
    print(f"   Host: {config.server.host}")
    print(f"   Port: {config.server.port}")
    print(f"   Environment: {config.environment.value}")
    print(f"   Docs: {'Enabled' if config.server.enable_docs else 'Disabled'}")
    print(f"\n{'='*60}\n")
    
    server.run()


def main():
    """Main production deployment function"""
    
    print("GPU OCR Server - Production Deployment")
    print("="*60)
    
    # Check system requirements
    check_system_requirements()
    
    # Configure environment
    configure_production_environment()
    
    # Set process limits
    set_process_limits()
    
    # Create PID file
    create_pid_file()
    
    # Set up monitoring
    setup_monitoring()
    
    # Run server
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        logging.error("Server crashed", exc_info=True)
        sys.exit(1)
    
    print("\n✅ Server shutdown complete")


if __name__ == "__main__":
    main()