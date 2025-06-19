"""
API Middleware Module

Provides request tracking, logging, rate limiting, metrics collection,
and error handling for the FastAPI application.
"""

import time
import uuid
import json
import logging
import traceback
from typing import Dict, Optional, Any, List, Callable, Set, Tuple
from datetime import datetime, timedelta
from contextvars import ContextVar
import asyncio
from collections import defaultdict
import ipaddress

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Send, Scope
from starlette.datastructures import Headers, MutableHeaders

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..config import Config
from .schemas import ErrorResponse

# Configure logging
logger = logging.getLogger(__name__)

# Context variable for request ID
request_id_context: ContextVar[str] = ContextVar('request_id', default='')


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID to all requests.
    
    Generates UUID for each request and makes it available throughout
    the request lifecycle via context variable and response headers.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate or get request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        # Store in context variable
        request_id_context.set(request_id)
        
        # Add to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers['X-Request-ID'] = request_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests and responses with structured format.
    
    Features:
    - JSON structured logging
    - Request/response size tracking
    - Duration measurement
    - Sensitive data masking
    """
    
    SENSITIVE_HEADERS = {'authorization', 'x-api-key', 'cookie', 'set-cookie'}
    SENSITIVE_FIELDS = {'password', 'token', 'secret', 'api_key'}
    
    def __init__(self, app: ASGIApp, log_response_body: bool = False):
        super().__init__(app)
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request info
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request
        request_body_size = int(request.headers.get('content-length', 0))
        await self._log_request(request, request_id, request_body_size)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Get response size (estimate)
        response_body_size = int(response.headers.get('content-length', 0))
        
        # Log response
        await self._log_response(
            request, response, request_id, 
            duration_ms, response_body_size
        )
        
        return response
    
    async def _log_request(self, request: Request, request_id: str, body_size: int):
        """Log incoming request"""
        log_data = {
            'event': 'request',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'headers': self._mask_headers(dict(request.headers)),
            'client_ip': request.client.host if request.client else None,
            'body_size': body_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(json.dumps(log_data))
    
    async def _log_response(
        self, 
        request: Request, 
        response: Response,
        request_id: str,
        duration_ms: float,
        body_size: int
    ):
        """Log outgoing response"""
        log_data = {
            'event': 'response',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'duration_ms': round(duration_ms, 2),
            'body_size': body_size,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log level based on status code
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
    
    def _mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive headers"""
        masked = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                masked[key] = '***MASKED***'
            else:
                masked[key] = value
        return masked


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling and formatting.
    
    Catches all unhandled exceptions and formats them as ErrorResponse.
    """
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # FastAPI HTTP exceptions
            return await self._handle_http_exception(request, e)
            
        except Exception as e:
            # Unhandled exceptions
            return await self._handle_exception(request, e)
    
    async def _handle_http_exception(self, request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        error_response = ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            detail=getattr(exc, 'detail', {}),
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    async def _handle_exception(self, request: Request, exc: Exception):
        """Handle unhandled exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log full stack trace
        logger.error(
            f"Unhandled exception for request {request_id}",
            exc_info=True,
            extra={'request_id': request_id}
        )
        
        # Prepare error response
        if self.debug:
            # Include stack trace in debug mode
            detail = {
                'exception': str(exc),
                'traceback': traceback.format_exc().split('\n')
            }
            message = str(exc)
        else:
            # Generic message in production
            detail = {}
            message = "An internal server error occurred"
        
        error_response = ErrorResponse(
            error='InternalServerError',
            message=message,
            detail=detail,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )


class RateLimiter:
    """
    Token bucket rate limiter with Redis support for distributed systems.
    """
    
    def __init__(
        self,
        rate: int = 60,  # Tokens per minute
        burst: int = 10,  # Burst capacity
        redis_client: Optional[Any] = None
    ):
        self.rate = rate
        self.burst = burst
        self.redis_client = redis_client
        self.local_buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'tokens': burst, 'last_update': time.time()}
        )
    
    async def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed: bool, retry_after: int seconds)
        """
        if self.redis_client and REDIS_AVAILABLE:
            return await self._check_redis(key)
        else:
            return await self._check_local(key)
    
    async def _check_local(self, key: str) -> Tuple[bool, int]:
        """Check rate limit using local memory"""
        bucket = self.local_buckets[key]
        now = time.time()
        
        # Refill tokens
        time_passed = now - bucket['last_update']
        tokens_to_add = time_passed * (self.rate / 60.0)
        bucket['tokens'] = min(self.burst, bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now
        
        # Check if request allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, 0
        else:
            # Calculate retry after
            tokens_needed = 1 - bucket['tokens']
            retry_after = int(tokens_needed / (self.rate / 60.0))
            return False, retry_after
    
    async def _check_redis(self, key: str) -> Tuple[bool, int]:
        """Check rate limit using Redis"""
        try:
            # Use Redis token bucket implementation
            bucket_key = f"rate_limit:{key}"
            tokens_key = f"{bucket_key}:tokens"
            update_key = f"{bucket_key}:update"
            
            pipe = self.redis_client.pipeline()
            now = time.time()
            
            # Get current state
            pipe.get(tokens_key)
            pipe.get(update_key)
            tokens, last_update = pipe.execute()
            
            tokens = float(tokens or self.burst)
            last_update = float(last_update or now)
            
            # Refill tokens
            time_passed = now - last_update
            tokens_to_add = time_passed * (self.rate / 60.0)
            tokens = min(self.burst, tokens + tokens_to_add)
            
            # Check if allowed
            if tokens >= 1:
                # Consume token
                pipe = self.redis_client.pipeline()
                pipe.set(tokens_key, tokens - 1, ex=300)  # 5 min expiry
                pipe.set(update_key, now, ex=300)
                pipe.execute()
                return True, 0
            else:
                # Update timestamp
                self.redis_client.set(update_key, now, ex=300)
                
                # Calculate retry after
                tokens_needed = 1 - tokens
                retry_after = int(tokens_needed / (self.rate / 60.0))
                return False, retry_after
                
        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            # Fallback to local
            return await self._check_local(key)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with Redis support.
    
    Features:
    - Token bucket algorithm
    - Per-IP rate limiting
    - Distributed rate limiting with Redis
    - Whitelist support for internal IPs
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate: int = 60,  # Requests per minute
        burst: int = 10,
        redis_client: Optional[Any] = None,
        whitelist_ips: Optional[Set[str]] = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate, burst, redis_client)
        self.whitelist_ips = whitelist_ips or set()
        
        # Add local network ranges to whitelist
        self.whitelist_networks = [
            ipaddress.ip_network('127.0.0.0/8'),
            ipaddress.ip_network('10.0.0.0/8'),
            ipaddress.ip_network('172.16.0.0/12'),
            ipaddress.ip_network('192.168.0.0/16'),
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check whitelist
        if self._is_whitelisted(client_ip):
            return await call_next(request)
        
        # Check rate limit
        allowed, retry_after = await self.rate_limiter.is_allowed(client_ip)
        
        if not allowed:
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            error_response = ErrorResponse(
                error='RateLimitExceeded',
                message='Too many requests',
                detail={
                    'rate': self.rate_limiter.rate,
                    'retry_after': retry_after
                },
                request_id=request_id,
                timestamp=datetime.utcnow()
            )
            
            return JSONResponse(
                status_code=429,
                content=error_response.dict(),
                headers={
                    'Retry-After': str(retry_after),
                    'X-RateLimit-Limit': str(self.rate_limiter.rate),
                    'X-Request-ID': request_id
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers['X-RateLimit-Limit'] = str(self.rate_limiter.rate)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, considering proxies"""
        # Check X-Forwarded-For header
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            # Get first IP in chain
            return forwarded.split(',')[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Use direct client IP
        if request.client:
            return request.client.host
        
        return 'unknown'
    
    def _is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        # Check exact match
        if ip in self.whitelist_ips:
            return True
        
        # Check network ranges
        try:
            ip_addr = ipaddress.ip_address(ip)
            for network in self.whitelist_networks:
                if ip_addr in network:
                    return True
        except ValueError:
            pass
        
        return False


class MetricsCollector:
    """
    Collect request metrics for monitoring.
    """
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.response_times = defaultdict(list)
        self.status_codes = defaultdict(int)
        self.request_sizes = []
        self.response_sizes = []
        self.gpu_utilizations = []
        self._lock = asyncio.Lock()
    
    async def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        request_size: int,
        response_size: int,
        gpu_utilization: Optional[float] = None
    ):
        """Record request metrics"""
        async with self._lock:
            # Request count
            key = f"{method}:{endpoint}"
            self.request_count[key] += 1
            
            # Response time
            self.response_times[key].append(duration_ms)
            # Keep only last 1000 samples per endpoint
            if len(self.response_times[key]) > 1000:
                self.response_times[key] = self.response_times[key][-1000:]
            
            # Status codes
            self.status_codes[status_code] += 1
            
            # Sizes
            self.request_sizes.append(request_size)
            self.response_sizes.append(response_size)
            
            # Keep only last 10000 samples
            if len(self.request_sizes) > 10000:
                self.request_sizes = self.request_sizes[-10000:]
                self.response_sizes = self.response_sizes[-10000:]
            
            # GPU utilization
            if gpu_utilization is not None:
                self.gpu_utilizations.append(gpu_utilization)
                if len(self.gpu_utilizations) > 1000:
                    self.gpu_utilizations = self.gpu_utilizations[-1000:]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        async with self._lock:
            metrics = {
                'request_count': dict(self.request_count),
                'status_codes': dict(self.status_codes),
                'response_times': {},
                'sizes': {
                    'avg_request': self._calculate_avg(self.request_sizes),
                    'avg_response': self._calculate_avg(self.response_sizes)
                },
                'gpu_utilization': {
                    'current': self.gpu_utilizations[-1] if self.gpu_utilizations else 0,
                    'avg': self._calculate_avg(self.gpu_utilizations)
                }
            }
            
            # Calculate percentiles for response times
            for endpoint, times in self.response_times.items():
                if times:
                    sorted_times = sorted(times)
                    metrics['response_times'][endpoint] = {
                        'p50': self._percentile(sorted_times, 50),
                        'p95': self._percentile(sorted_times, 95),
                        'p99': self._percentile(sorted_times, 99),
                        'avg': self._calculate_avg(times)
                    }
            
            return metrics
    
    def _calculate_avg(self, values: List[float]) -> float:
        """Calculate average of values"""
        return sum(values) / len(values) if values else 0
    
    def _percentile(self, sorted_values: List[float], p: int) -> float:
        """Calculate percentile"""
        if not sorted_values:
            return 0
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f < len(sorted_values) - 1 else f
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Collect metrics for all requests.
    """
    
    def __init__(self, app: ASGIApp, metrics_collector: MetricsCollector):
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        response_size = int(response.headers.get('content-length', 0))
        
        # Get GPU utilization if available
        gpu_utilization = None
        if hasattr(request.state, 'gpu_utilization'):
            gpu_utilization = request.state.gpu_utilization
        
        # Record metrics
        await self.metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            gpu_utilization=gpu_utilization
        )
        
        return response


def setup_middleware(app: FastAPI, config: Config) -> MetricsCollector:
    """
    Configure all middleware in the correct order.
    
    Order (outermost to innermost):
    1. RequestID
    2. ErrorHandling
    3. RequestLogging
    4. Metrics
    5. RateLimit
    6. CORS
    7. Compression
    
    Returns:
        MetricsCollector instance for accessing metrics
    """
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # 1. Request ID (outermost)
    app.add_middleware(RequestIDMiddleware)
    
    # 2. Error Handling
    app.add_middleware(
        ErrorHandlingMiddleware,
        debug=config.debug
    )
    
    # 3. Request Logging
    app.add_middleware(
        RequestLoggingMiddleware,
        log_response_body=config.debug
    )
    
    # 4. Metrics
    app.add_middleware(
        MetricsMiddleware,
        metrics_collector=metrics_collector
    )
    
    # 5. Rate Limiting
    # Initialize Redis client if available
    redis_client = None
    if REDIS_AVAILABLE and config.cache.redis_host:
        try:
            redis_client = redis.Redis(
                host=config.cache.redis_host,
                port=config.cache.redis_port,
                db=config.cache.redis_db,
                password=config.cache.redis_password,
                decode_responses=False
            )
            redis_client.ping()
            logger.info("Redis connected for rate limiting")
        except Exception as e:
            logger.warning(f"Redis connection failed for rate limiting: {e}")
            redis_client = None
    
    app.add_middleware(
        RateLimitMiddleware,
        rate=config.server.rate_limit_per_minute,
        burst=10,
        redis_client=redis_client,
        whitelist_ips=set()  # Add any specific IPs to whitelist
    )
    
    # 6. CORS
    configure_cors(app, config)
    
    # 7. Compression (innermost)
    configure_compression(app)
    
    logger.info("Middleware stack configured")
    
    return metrics_collector


def configure_cors(app: FastAPI, config: Config):
    """
    Configure CORS middleware.
    
    Args:
        app: FastAPI application instance
        config: Application configuration
    """
    # Default origins
    origins = config.server.cors_origins
    
    # Add localhost variations for development
    if config.debug:
        origins.extend([
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=config.server.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "Retry-After"],
        max_age=86400  # Cache preflight for 24 hours
    )
    
    logger.info(f"CORS configured for origins: {origins}")


def configure_compression(app: FastAPI):
    """
    Configure response compression.
    
    Args:
        app: FastAPI application instance
    """
    # Skip compression for these content types
    excluded_types = {
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'application/pdf',
        'application/zip',
        'application/gzip'
    }
    
    class ConditionalGZipMiddleware(GZipMiddleware):
        """GZip middleware that skips certain content types"""
        
        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                headers = Headers(scope=scope)
                
                # Check if we should skip compression
                if self._should_skip_compression(headers):
                    await self.app(scope, receive, send)
                    return
            
            await super().__call__(scope, receive, send)
        
        def _should_skip_compression(self, headers: Headers) -> bool:
            """Check if compression should be skipped"""
            # Check content type
            content_type = headers.get('content-type', '').split(';')[0].strip()
            if content_type in excluded_types:
                return True
            
            # Check if already compressed
            if headers.get('content-encoding'):
                return True
            
            return False
    
    # Add compression middleware with 1KB minimum size
    app.add_middleware(
        ConditionalGZipMiddleware,
        minimum_size=1024
    )
    
    logger.info("Response compression configured")


# Utility functions for middleware

def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_context.get()


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively mask sensitive data in dictionaries.
    
    Args:
        data: Dictionary to mask
        
    Returns:
        Dictionary with sensitive values masked
    """
    if not isinstance(data, dict):
        return data
    
    masked = {}
    sensitive_keys = {'password', 'token', 'secret', 'api_key', 'authorization'}
    
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            masked[key] = '***MASKED***'
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
        elif isinstance(value, list):
            masked[key] = [mask_sensitive_data(item) if isinstance(item, dict) else item for item in value]
        else:
            masked[key] = value
    
    return masked