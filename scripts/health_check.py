#!/usr/bin/env python3
"""
Health check script for GPU OCR Server

Can be used for monitoring, load balancer health checks, or container orchestration.
"""

import sys
import time
import argparse
import requests
import json
from typing import Dict, Any, Optional


class HealthChecker:
    """Health check client for GPU OCR Server"""
    
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
    def check_health(self) -> Dict[str, Any]:
        """Basic health check"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_readiness(self) -> Dict[str, Any]:
        """Detailed readiness check"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/ready",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "ready": False,
                "error": str(e)
            }
    
    def check_gpu_status(self) -> Optional[Dict[str, Any]]:
        """Check GPU status"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/gpu/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
    
    def check_metrics(self) -> Optional[Dict[str, Any]]:
        """Get service metrics"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
    
    def perform_full_check(self, verbose: bool = False) -> bool:
        """Perform comprehensive health check"""
        
        all_healthy = True
        
        # 1. Basic health check
        health = self.check_health()
        health_status = health.get("status", "unknown")
        
        if verbose:
            print(f"Health Status: {health_status}")
            if health_status != "healthy":
                print(f"  Services: {json.dumps(health.get('services', {}), indent=2)}")
        
        if health_status != "healthy":
            all_healthy = False
        
        # 2. Readiness check
        readiness = self.check_readiness()
        is_ready = readiness.get("ready", False)
        
        if verbose:
            print(f"\nReadiness: {'Ready' if is_ready else 'Not Ready'}")
            print(f"  Models Loaded: {readiness.get('models_loaded', False)}")
            print(f"  GPU Available: {readiness.get('gpu_available', False)}")
            if not is_ready:
                print(f"  Message: {readiness.get('message', 'No message')}")
        
        if not is_ready:
            all_healthy = False
        
        # 3. GPU status (optional)
        gpu_status = self.check_gpu_status()
        if gpu_status and verbose:
            print(f"\nGPU Status:")
            print(f"  Device: {gpu_status.get('device_name', 'Unknown')}")
            print(f"  Memory: {gpu_status.get('memory', {}).get('used_mb', 0)}MB / "
                  f"{gpu_status.get('memory', {}).get('total_mb', 0)}MB")
            print(f"  Utilization: {gpu_status.get('utilization', {}).get('compute_percent', 0)}%")
            print(f"  Temperature: {gpu_status.get('temperature_celsius', 0)}°C")
        
        # 4. Service metrics (optional)
        metrics = self.check_metrics()
        if metrics and verbose:
            print(f"\nService Metrics:")
            print(f"  Uptime: {metrics.get('uptime_seconds', 0):.0f} seconds")
            print(f"  Total Requests: {metrics.get('total_requests', 0)}")
            print(f"  Total Documents: {metrics.get('total_documents', 0)}")
            print(f"  Average Speed: {metrics.get('average_pages_per_second', 0):.2f} pages/sec")
            print(f"  Queue Size: {metrics.get('current_queue_size', 0)}")
        
        return all_healthy


def main():
    """Main health check function"""
    
    parser = argparse.ArgumentParser(description="GPU OCR Server Health Check")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the OCR server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--check",
        choices=["health", "ready", "gpu", "full"],
        default="full",
        help="Type of check to perform (default: full)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--wait",
        type=int,
        help="Wait for service to be healthy (max seconds)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Check interval when waiting (default: 5 seconds)"
    )
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(args.url, args.timeout)
    
    # If waiting for healthy status
    if args.wait:
        start_time = time.time()
        while time.time() - start_time < args.wait:
            if args.check == "health":
                result = checker.check_health()
                if result.get("status") == "healthy":
                    print("✅ Service is healthy")
                    return 0
            elif args.check == "ready":
                result = checker.check_readiness()
                if result.get("ready"):
                    print("✅ Service is ready")
                    return 0
            else:
                if checker.perform_full_check(verbose=False):
                    print("✅ Service passed all checks")
                    return 0
            
            if args.verbose:
                print(f"⏳ Waiting... ({int(time.time() - start_time)}s elapsed)")
            
            time.sleep(args.interval)
        
        print("❌ Timeout waiting for service to be healthy")
        return 1
    
    # Perform single check
    if args.check == "health":
        result = checker.check_health()
        if args.verbose:
            print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "healthy" else 1
    
    elif args.check == "ready":
        result = checker.check_readiness()
        if args.verbose:
            print(json.dumps(result, indent=2))
        return 0 if result.get("ready") else 1
    
    elif args.check == "gpu":
        result = checker.check_gpu_status()
        if result:
            if args.verbose:
                print(json.dumps(result, indent=2))
            return 0
        else:
            print("❌ GPU status check failed")
            return 1
    
    else:  # full check
        all_healthy = checker.perform_full_check(verbose=args.verbose)
        if all_healthy:
            print("\n✅ All health checks passed")
            return 0
        else:
            print("\n❌ Some health checks failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())