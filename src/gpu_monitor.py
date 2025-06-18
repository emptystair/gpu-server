import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be disabled.")

from src.config import GPUConfig, load_config


@dataclass
class MemoryInfo:
    """GPU memory information"""
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_percent: float


@dataclass
class GPUUtilization:
    """GPU utilization metrics"""
    compute_percent: float
    memory_percent: float
    temperature_celsius: float
    power_draw_watts: float


@dataclass
class GPUMetrics:
    """Comprehensive GPU metrics for monitoring"""
    device_id: int
    device_name: str
    memory_info: MemoryInfo
    utilization: GPUUtilization
    timestamp: datetime
    driver_version: str
    cuda_version: str


class GPUMonitor:
    """Monitor GPU resources using nvidia-ml-py"""
    
    def __init__(self, device_id: int = 0):
        """Purpose: Initialize GPU monitoring with nvidia-ml-py
        Dependencies: nvidia-ml-py (pynvml)
        Priority: MONITORING
        """
        self.device_id = device_id
        self.device_handle = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.metrics_lock = threading.Lock()
        self.latest_metrics: Optional[GPUMetrics] = None
        self.is_initialized = False
        self.gpu_available = False
        
        # Load configuration
        config = load_config()
        self.gpu_config: GPUConfig = config.gpu
        
        # Override device_id if specified in config
        if self.gpu_config.device_id != device_id:
            self.device_id = self.gpu_config.device_id
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize NVML
        self._initialize_nvml()
    
    def _initialize_nvml(self):
        """Initialize NVIDIA Management Library"""
        if not PYNVML_AVAILABLE:
            self.logger.warning("pynvml not available. GPU monitoring disabled.")
            return
        
        try:
            pynvml.nvmlInit()
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self.gpu_available = True
            self.is_initialized = True
            
            # Log GPU information
            device_name = pynvml.nvmlDeviceGetName(self.device_handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode('utf-8')
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            self.logger.info(f"GPU monitoring initialized for device {self.device_id}: {device_name}")
            self.logger.info(f"NVIDIA Driver Version: {driver_version}")
            
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_available = False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing GPU monitoring: {e}")
            self.gpu_available = False
    
    def start_monitoring(self):
        """Purpose: Start background GPU monitoring thread
        Calls:
            - pynvml.nvmlInit()
            - _monitoring_loop() in thread
        Called by: main.startup_event
        Priority: MONITORING
        """
        if not self.gpu_available:
            self.logger.warning("GPU not available. Monitoring not started.")
            return
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread already running.")
            return
        
        # Clear stop event
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="GPUMonitorThread"
        )
        self.monitoring_thread.start()
        self.logger.info("GPU monitoring thread started.")
    
    def stop_monitoring(self):
        """Purpose: Stop monitoring thread and cleanup
        Calls:
            - pynvml.nvmlShutdown()
        Called by: main.shutdown_event
        Priority: MONITORING
        """
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
            if self.monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not stop gracefully.")
        
        # Shutdown NVML
        if self.is_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self.is_initialized = False
                self.logger.info("NVML shutdown complete.")
            except pynvml.NVMLError as e:
                self.logger.error(f"Error shutting down NVML: {e}")
    
    def get_available_memory(self) -> MemoryInfo:
        """Purpose: Get current GPU memory availability
        Calls:
            - pynvml.nvmlDeviceGetMemoryInfo()
        Called by: ocr_service._determine_batch_size()
        Priority: OPTIMIZATION
        """
        if not self.gpu_available:
            # Return mock data if GPU not available
            return MemoryInfo(
                total_mb=0,
                used_mb=0,
                free_mb=0,
                utilization_percent=0.0
            )
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
            total_mb = mem_info.total // (1024 * 1024)
            used_mb = mem_info.used // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)
            utilization_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
            
            return MemoryInfo(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                utilization_percent=utilization_percent
            )
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting memory info: {e}")
            return MemoryInfo(
                total_mb=0,
                used_mb=0,
                free_mb=0,
                utilization_percent=0.0
            )
    
    def get_gpu_utilization(self) -> GPUUtilization:
        """Purpose: Get current GPU compute utilization
        Calls:
            - pynvml.nvmlDeviceGetUtilizationRates()
        Called by: api.routes.gpu_status_endpoint
        Priority: MONITORING
        """
        if not self.gpu_available:
            # Return mock data if GPU not available
            return GPUUtilization(
                compute_percent=0.0,
                memory_percent=0.0,
                temperature_celsius=0.0,
                power_draw_watts=0.0
            )
        
        try:
            # Get utilization rates
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.device_handle)
            
            # Get temperature
            temperature = self.get_temperature()
            
            # Get power draw
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0  # Convert to watts
            except pynvml.NVMLError:
                power_draw = 0.0
            
            return GPUUtilization(
                compute_percent=float(util_rates.gpu),
                memory_percent=float(util_rates.memory),
                temperature_celsius=temperature,
                power_draw_watts=power_draw
            )
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting GPU utilization: {e}")
            return GPUUtilization(
                compute_percent=0.0,
                memory_percent=0.0,
                temperature_celsius=0.0,
                power_draw_watts=0.0
            )
    
    def check_memory_pressure(self) -> bool:
        """Purpose: Check if GPU memory is under pressure
        Calls:
            - get_available_memory()
        Called by: ocr_service._process_batches()
        Priority: OPTIMIZATION
        """
        memory_info = self.get_available_memory()
        
        # Check if memory utilization exceeds threshold
        return memory_info.utilization_percent > self.gpu_config.memory_threshold_percent
    
    def get_temperature(self) -> float:
        """Purpose: Get GPU temperature for thermal monitoring
        Calls:
            - pynvml.nvmlDeviceGetTemperature()
        Called by: api.routes.gpu_status_endpoint
        Priority: MONITORING
        """
        if not self.gpu_available:
            return 0.0
        
        try:
            # NVML_TEMPERATURE_GPU = 0
            temperature = pynvml.nvmlDeviceGetTemperature(self.device_handle, 0)
            return float(temperature)
        except pynvml.NVMLError as e:
            self.logger.error(f"Error getting temperature: {e}")
            return 0.0
    
    def _monitoring_loop(self):
        """Purpose: Background thread for continuous monitoring
        Calls:
            - All get_* methods periodically
            - _update_metrics()
        Called by: start_monitoring() in thread
        Priority: MONITORING
        """
        self.logger.info(f"GPU monitoring loop started with interval: {self.gpu_config.monitoring_interval_seconds}s")
        
        while not self.stop_event.is_set():
            try:
                # Collect all metrics
                memory_info = self.get_available_memory()
                utilization = self.get_gpu_utilization()
                
                # Get device info
                device_name = "Unknown"
                driver_version = "Unknown"
                cuda_version = "Unknown"
                
                if self.gpu_available:
                    try:
                        device_name = pynvml.nvmlDeviceGetName(self.device_handle)
                        if isinstance(device_name, bytes):
                            device_name = device_name.decode('utf-8')
                        driver_version = pynvml.nvmlSystemGetDriverVersion()
                        if isinstance(driver_version, bytes):
                            driver_version = driver_version.decode('utf-8')
                        cuda_version = str(pynvml.nvmlDeviceGetCudaComputeCapability(self.device_handle))
                    except pynvml.NVMLError:
                        pass
                
                # Create metrics object
                metrics = GPUMetrics(
                    device_id=self.device_id,
                    device_name=device_name,
                    memory_info=memory_info,
                    utilization=utilization,
                    timestamp=datetime.utcnow(),
                    driver_version=driver_version,
                    cuda_version=cuda_version
                )
                
                # Update metrics
                self._update_metrics(metrics)
                
                # Log if under memory pressure
                if self.check_memory_pressure():
                    self.logger.warning(
                        f"GPU memory pressure detected: {memory_info.utilization_percent:.1f}% "
                        f"(threshold: {self.gpu_config.memory_threshold_percent}%)"
                    )
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for monitoring interval
            self.stop_event.wait(self.gpu_config.monitoring_interval_seconds)
        
        self.logger.info("GPU monitoring loop stopped.")
    
    def _update_metrics(self, metrics: GPUMetrics):
        """Purpose: Update internal metrics storage
        Calls: None (updates internal state)
        Called by: _monitoring_loop()
        Priority: MONITORING
        """
        with self.metrics_lock:
            self.latest_metrics = metrics
            
            # Log high-level metrics periodically (every 10th update)
            if hasattr(self, '_update_count'):
                self._update_count += 1
            else:
                self._update_count = 1
            
            if self._update_count % 10 == 0:
                self.logger.debug(
                    f"GPU Stats - Compute: {metrics.utilization.compute_percent:.1f}%, "
                    f"Memory: {metrics.memory_info.utilization_percent:.1f}%, "
                    f"Temp: {metrics.utilization.temperature_celsius:.1f}°C, "
                    f"Power: {metrics.utilization.power_draw_watts:.1f}W"
                )
    
    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """Get the latest collected metrics"""
        with self.metrics_lock:
            return self.latest_metrics
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for use"""
        return self.gpu_available
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get static GPU device information"""
        if not self.gpu_available:
            return {
                "available": False,
                "error": "GPU not available or drivers not installed"
            }
        
        try:
            device_name = pynvml.nvmlDeviceGetName(self.device_handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode('utf-8')
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode('utf-8')
            
            return {
                "available": True,
                "device_id": self.device_id,
                "name": device_name,
                "driver_version": driver_version,
                "cuda_compute_capability": pynvml.nvmlDeviceGetCudaComputeCapability(self.device_handle),
                "total_memory_mb": pynvml.nvmlDeviceGetMemoryInfo(self.device_handle).total // (1024 * 1024)
            }
        except pynvml.NVMLError as e:
            return {
                "available": False,
                "error": f"Error getting device info: {str(e)}"
            }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create monitor
    monitor = GPUMonitor(device_id=0)
    
    # Check if GPU is available
    if monitor.is_gpu_available():
        print("GPU is available!")
        print(f"Device info: {monitor.get_device_info()}")
        
        # Get current metrics
        print(f"Memory info: {monitor.get_available_memory()}")
        print(f"GPU utilization: {monitor.get_gpu_utilization()}")
        print(f"Temperature: {monitor.get_temperature()}°C")
        print(f"Memory pressure: {monitor.check_memory_pressure()}")
        
        # Test monitoring thread
        monitor.start_monitoring()
        print("Monitoring started. Waiting 5 seconds...")
        time.sleep(5)
        
        # Get latest metrics
        latest = monitor.get_latest_metrics()
        if latest:
            print(f"Latest metrics timestamp: {latest.timestamp}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("Monitoring stopped.")
    else:
        print("GPU is not available.")