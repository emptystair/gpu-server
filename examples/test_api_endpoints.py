"""
Example: Test API Endpoints

This script demonstrates how to test all API endpoints of the GPU OCR Server.
"""

import asyncio
import aiohttp
import base64
from pathlib import Path
import json
import time
from typing import Dict, Any


class OCRAPIClient:
    """Client for testing OCR API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        
    async def health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/health") as response:
                return await response.json(), response.status
    
    async def readiness_check(self) -> Dict[str, Any]:
        """Test readiness endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/ready") as response:
                return await response.json(), response.status
    
    async def gpu_status(self) -> Dict[str, Any]:
        """Get GPU status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/gpu/status") as response:
                return await response.json(), response.status
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_url}/stats") as response:
                return await response.json(), response.status
    
    async def process_image(self, image_path: Path, **params) -> Dict[str, Any]:
        """Process single image"""
        async with aiohttp.ClientSession() as session:
            # Prepare multipart data
            data = aiohttp.FormData()
            
            # Add file
            with open(image_path, 'rb') as f:
                data.add_field('file',
                              f,
                              filename=image_path.name,
                              content_type='image/jpeg')
            
            # Add parameters
            data.add_field('strategy', params.get('strategy', 'balanced'))
            data.add_field('language', params.get('language', 'en'))
            data.add_field('output_format', params.get('output_format', 'json'))
            
            if 'dpi' in params:
                data.add_field('dpi', str(params['dpi']))
            if 'confidence_threshold' in params:
                data.add_field('confidence_threshold', str(params['confidence_threshold']))
            
            # Send request
            async with session.post(
                f"{self.api_url}/ocr/process",
                data=data
            ) as response:
                return await response.json(), response.status
    
    async def process_batch(self, image_paths: list[Path], **params) -> Dict[str, Any]:
        """Process batch of images"""
        async with aiohttp.ClientSession() as session:
            # Prepare multipart data
            data = aiohttp.FormData()
            
            # Add files
            for image_path in image_paths:
                with open(image_path, 'rb') as f:
                    data.add_field('files',
                                  f,
                                  filename=image_path.name,
                                  content_type='image/jpeg')
            
            # Add parameters
            data.add_field('strategy', params.get('strategy', 'balanced'))
            data.add_field('language', params.get('language', 'en'))
            data.add_field('parallel_processing', str(params.get('parallel_processing', True)))
            
            # Send request
            async with session.post(
                f"{self.api_url}/ocr/process-batch",
                data=data
            ) as response:
                return await response.json(), response.status


async def test_all_endpoints():
    """Test all API endpoints"""
    
    client = OCRAPIClient()
    
    print("=" * 60)
    print("GPU OCR Server API Test")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1. Testing Health Check...")
    try:
        result, status = await client.health_check()
        print(f"   Status: {status}")
        print(f"   Response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Readiness Check
    print("\n2. Testing Readiness Check...")
    try:
        result, status = await client.readiness_check()
        print(f"   Status: {status}")
        print(f"   Ready: {result.get('ready')}")
        print(f"   Models Loaded: {result.get('models_loaded')}")
        print(f"   GPU Available: {result.get('gpu_available')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. GPU Status
    print("\n3. Testing GPU Status...")
    try:
        result, status = await client.gpu_status()
        print(f"   Status: {status}")
        print(f"   Device: {result.get('device_name')}")
        print(f"   Memory: {result.get('memory', {}).get('used_mb')}MB / {result.get('memory', {}).get('total_mb')}MB")
        print(f"   Utilization: {result.get('utilization', {}).get('compute_percent')}%")
        print(f"   Temperature: {result.get('temperature_celsius')}°C")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Statistics
    print("\n4. Testing Statistics...")
    try:
        result, status = await client.get_stats()
        print(f"   Status: {status}")
        print(f"   Total Requests: {result.get('total_requests')}")
        print(f"   Total Documents: {result.get('total_documents')}")
        print(f"   Average Confidence: {result.get('average_confidence', 0):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Process Image (create a test image if needed)
    print("\n5. Testing Single Image Processing...")
    
    # Create a simple test image
    test_image = Path("test_image.jpg")
    if not test_image.exists():
        print("   Creating test image...")
        # You would need PIL or similar to create a test image
        # For now, we'll skip if no test image exists
        print("   Skipping - no test image available")
    else:
        try:
            start_time = time.time()
            result, status = await client.process_image(
                test_image,
                strategy="balanced",
                language="en",
                output_format="json"
            )
            processing_time = time.time() - start_time
            
            print(f"   Status: {status}")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Detected Regions: {result.get('total_regions', 0)}")
            print(f"   Average Confidence: {result.get('average_confidence', 0):.2f}")
            print(f"   Cache Hit: {result.get('cache_hit', False)}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # 6. Batch Processing
    print("\n6. Testing Batch Processing...")
    test_images = [Path(f"test_image_{i}.jpg") for i in range(3)]
    
    if all(img.exists() for img in test_images):
        try:
            start_time = time.time()
            result, status = await client.process_batch(
                test_images,
                strategy="speed",
                parallel_processing=True
            )
            processing_time = time.time() - start_time
            
            print(f"   Status: {status}")
            print(f"   Total Processing Time: {processing_time:.2f}s")
            print(f"   Total Images: {result.get('total_images', 0)}")
            print(f"   Successful: {result.get('successful', 0)}")
            print(f"   Failed: {result.get('failed', 0)}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("   Skipping - test images not available")
    
    print("\n" + "=" * 60)
    print("API Test Complete")
    print("=" * 60)


async def test_error_handling():
    """Test error handling scenarios"""
    
    client = OCRAPIClient()
    
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # 1. Invalid file type
    print("\n1. Testing Invalid File Type...")
    try:
        # Create a text file
        test_file = Path("test.txt")
        test_file.write_text("This is not an image")
        
        result, status = await client.process_image(test_file)
        print(f"   Status: {status}")
        print(f"   Error: {result.get('detail', 'No error')}")
        
        test_file.unlink()
    except Exception as e:
        print(f"   Expected error: {e}")
    
    # 2. File too large
    print("\n2. Testing File Size Limit...")
    # Would need to create a large file to test this
    print("   Skipping - requires large test file")
    
    # 3. Invalid parameters
    print("\n3. Testing Invalid Parameters...")
    if Path("test_image.jpg").exists():
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                with open("test_image.jpg", 'rb') as f:
                    data.add_field('file', f, filename='test.jpg')
                data.add_field('dpi', '1000')  # Invalid DPI
                
                async with session.post(
                    f"{client.api_url}/ocr/process",
                    data=data
                ) as response:
                    result = await response.json()
                    print(f"   Status: {response.status}")
                    print(f"   Error: {result.get('detail', 'No error')}")
        except Exception as e:
            print(f"   Error: {e}")


async def test_performance():
    """Test API performance"""
    
    client = OCRAPIClient()
    
    print("\n" + "=" * 60)
    print("Testing API Performance")
    print("=" * 60)
    
    # Test concurrent requests
    print("\n1. Testing Concurrent Requests...")
    
    async def make_request():
        try:
            result, status = await client.health_check()
            return status == 200
        except:
            return False
    
    # Make 10 concurrent requests
    start_time = time.time()
    tasks = [make_request() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    successful = sum(results)
    print(f"   Successful: {successful}/10")
    print(f"   Total Time: {duration:.2f}s")
    print(f"   Requests/sec: {10/duration:.2f}")


async def main():
    """Run all tests"""
    
    # Wait a bit for server to be ready
    print("Waiting for server to start...")
    await asyncio.sleep(2)
    
    # Run tests
    await test_all_endpoints()
    await test_error_handling()
    await test_performance()


if __name__ == "__main__":
    asyncio.run(main())