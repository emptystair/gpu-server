import time
import logging
from src.gpu_monitor import GPUMonitor, MemoryInfo, GPUUtilization, GPUMetrics

# Set up logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=== Advanced GPU Monitor Tests ===\n")

# Test 1: Check data structures
print("Test 1: Data Structure Validation")
mem_info = MemoryInfo(total_mb=24576, used_mb=8192, free_mb=16384, utilization_percent=33.3)
print(f"Created MemoryInfo: {mem_info}")

util_info = GPUUtilization(compute_percent=45.5, memory_percent=33.3, temperature_celsius=65.0, power_draw_watts=250.0)
print(f"Created GPUUtilization: {util_info}")
print()

# Test 2: Monitor lifecycle
print("Test 2: Monitor Lifecycle")
monitor = GPUMonitor(device_id=0)
print(f"Monitor created. GPU available: {monitor.is_gpu_available()}")
print(f"Device info: {monitor.get_device_info()}")
print()

# Test 3: Start monitoring and collect metrics
print("Test 3: Background Monitoring Thread")
monitor.start_monitoring()
print("Monitoring started. Waiting 3 seconds to collect metrics...")
time.sleep(3)

# Check if metrics were collected
latest = monitor.get_latest_metrics()
if latest:
    print(f"Latest metrics collected at: {latest.timestamp}")
    print(f"Device: {latest.device_name}")
    print(f"Memory utilization: {latest.memory_info.utilization_percent}%")
else:
    print("No metrics collected (expected without GPU)")
print()

# Test 4: Multiple start/stop cycles
print("Test 4: Multiple Start/Stop Cycles")
for i in range(3):
    print(f"Cycle {i + 1}: Starting monitor...")
    monitor.start_monitoring()
    time.sleep(1)
    monitor.stop_monitoring()
    print(f"Cycle {i + 1}: Stopped successfully")
print()

# Test 5: Configuration integration
print("Test 5: Configuration Integration")
from src.config import load_config
config = load_config()
print(f"GPU Config - Device ID: {config.gpu.device_id}")
print(f"GPU Config - Memory threshold: {config.gpu.memory_threshold_percent}%")
print(f"GPU Config - Monitoring interval: {config.gpu.monitoring_interval_seconds}s")

# Create monitor with config
monitor2 = GPUMonitor()  # Should use config device_id
print(f"Monitor created with config device ID: {monitor2.device_id}")
print()

# Test 6: Thread safety
print("Test 6: Thread Safety")
import threading

def query_monitor(monitor, thread_id):
    for _ in range(5):
        mem = monitor.get_available_memory()
        util = monitor.get_gpu_utilization()
        print(f"Thread {thread_id}: Memory={mem.utilization_percent}%, Compute={util.compute_percent}%")
        time.sleep(0.1)

monitor.start_monitoring()
threads = []
for i in range(3):
    t = threading.Thread(target=query_monitor, args=(monitor, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

monitor.stop_monitoring()
print("Thread safety test completed")
print()

# Test 7: Memory pressure simulation
print("Test 7: Memory Pressure Check")
# Since we don't have a real GPU, this will always return False
pressure = monitor.check_memory_pressure()
print(f"Memory pressure detected: {pressure}")
print(f"Threshold would be: {monitor.gpu_config.memory_threshold_percent}%")

print("\nAll advanced tests completed successfully!")