#!/usr/bin/env python3
"""
Example script demonstrating how to use the Image Segmentation API.
This script shows both file upload and base64 methods.
"""

import requests
import base64
import json
from PIL import Image
import numpy as np
import io

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test the model info endpoint."""
    print("Getting model information...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_file_upload(image_path):
    """Test image segmentation with file upload."""
    print(f"Testing image segmentation with file upload: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Confidence: {result['result']['confidence']}")
            print(f"Mask shape: {result['result']['mask_shape']}")
            print(f"Classes detected: {result['result']['classes_detected']}")
            print("Colored mask returned as base64 (truncated):", result['result']['mask_colored_base64'][:100] + "...")
        else:
            print(f"Error: {response.json()}")
        print()
    
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        print("Creating a test image instead...")
        create_test_image_and_predict()

def test_base64_upload():
    """Test image segmentation with base64 encoded image."""
    print("Testing image segmentation with base64 encoding...")
    
    # Create a simple test image
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, 'RGB')
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Send request
    data = {'image': image_base64}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(data), headers=headers)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Confidence: {result['result']['confidence']}")
        print(f"Mask shape: {result['result']['mask_shape']}")
        print(f"Classes detected: {result['result']['classes_detected']}")
        print("Colored mask returned as base64 (truncated):", result['result']['mask_colored_base64'][:100] + "...")
    else:
        print(f"Error: {response.json()}")
    print()

def create_test_image_and_predict():
    """Create a test image and predict on it."""
    print("Creating a test image...")
    
    # Create a simple test image (simulate street scene)
    image_array = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
    image = Image.fromarray(image_array, 'RGB')
    
    # Save temporarily
    test_path = "/tmp/test_segmentation_image.png"
    image.save(test_path)
    print(f"Test image saved to: {test_path}")
    
    # Test with file upload
    test_file_upload(test_path)

def main():
    """Main function to run all tests."""
    print("Image Segmentation API Test Script")
    print("=" * 40)
    print()
    
    # Test health check
    test_health_check()
    
    # Test model info
    test_model_info()
    
    # Test file upload (with a test image path - update this as needed)
    test_image_path = "sample_image.jpg"  # Change this to an actual image path
    test_file_upload(test_image_path)
    
    # Test base64 upload
    test_base64_upload()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()