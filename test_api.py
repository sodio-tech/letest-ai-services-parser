"""
Test script for AI CSV Parser Agent API
"""

import requests
import json
import time
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_file_upload():
    """Test file upload"""
    print("Testing file upload...")
    
    # Use sample CSV file
    sample_file = Path("data/samples/sample_test_cases.csv")
    if not sample_file.exists():
        print("Sample file not found, creating one...")
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text("""Test Title,Test Steps,Expected Outcome
Login Test,1. Navigate to login page 2. Enter valid username 3. Enter valid password 4. Click login button,User should be successfully logged in and redirected to dashboard
Registration Test,a. Go to registration page b. Fill in all required fields c. Click submit button d. Verify email confirmation,New user account should be created and confirmation email sent""")
    
    with open(sample_file, 'rb') as f:
        files = {'file': ('sample_test_cases.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        return response.json()['file_id']
    return None

def test_file_info(file_id):
    """Test file info endpoint"""
    print(f"Testing file info for {file_id}...")
    response = requests.get(f"{BASE_URL}/files/{file_id}/info")
    print(f"Status: {response.status_code}")
    
    try:
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {response.text}")
    print()

def test_processing(file_id):
    """Test CSV processing"""
    print(f"Testing CSV processing for {file_id}...")
    
    # Use query parameters instead of JSON body
    params = {
        "file_id": file_id,
        "output_format": "json",
        "include_metadata": "true",
        "sample_percentage": 0.2
    }
    
    response = requests.post(f"{BASE_URL}/process", params=params)
    print(f"Status: {response.status_code}")
    
    try:
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return response.json()['processing_id']
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        print(f"Raw response: {response.text}")
    
    return None

def test_processing_status(job_id):
    """Test processing status"""
    print(f"Testing processing status for {job_id}...")
    
    max_attempts = 30  # 30 seconds max
    for attempt in range(max_attempts):
        response = requests.get(f"{BASE_URL}/process/{job_id}/status")
        print(f"Attempt {attempt + 1}: Status {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}, Progress: {data.get('progress', 0)}%")
            
            if data['status'] in ['completed', 'failed']:
                return data
        else:
            print(f"Error: {response.text}")
        
        time.sleep(1)
    
    print("Processing timeout!")
    return None

def test_processing_result(job_id):
    """Test processing result"""
    print(f"Testing processing result for {job_id}...")
    response = requests.get(f"{BASE_URL}/process/{job_id}/result")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Total objects: {data.get('total_objects', 0)}")
        
        if data.get('data'):
            print("Sample output:")
            print(json.dumps(data['data'][0], indent=2))
    else:
        print(f"Error: {response.text}")
    print()

def test_configuration():
    """Test configuration endpoints"""
    print("Testing configuration...")
    
    # Get current config
    response = requests.get(f"{BASE_URL}/config")
    print(f"Get config status: {response.status_code}")
    if response.status_code == 200:
        print("Current configuration retrieved")
    
    # Update config
    update_payload = {
        "csv_parser": {
            "sample_percentage": 0.3
        }
    }
    
    response = requests.put(f"{BASE_URL}/config", json=update_payload)
    print(f"Update config status: {response.status_code}")
    if response.status_code == 200:
        print("Configuration updated")
    print()

def test_list_files():
    """Test list files endpoint"""
    print("Testing list files...")
    response = requests.get(f"{BASE_URL}/files")
    print(f"Status: {response.status_code}")
    print(f"Files: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print("AI CSV Parser Agent API Test Suite")
    print("=" * 40)
    
    # Test health
    test_health()
    
    # Test file upload
    file_id = test_file_upload()
    if not file_id:
        print("File upload failed, stopping tests")
        return
    
    # Test file info
    test_file_info(file_id)
    
    # Test processing
    job_id = test_processing(file_id)
    if not job_id:
        print("Processing start failed, stopping tests")
        return
    
    # Wait for processing to complete
    status = test_processing_status(job_id)
    if not status or status['status'] != 'completed':
        print("Processing failed or timeout, stopping tests")
        return
    
    # Test processing result
    test_processing_result(job_id)
    
    # Test configuration
    test_configuration()
    
    # Test list files
    test_list_files()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
