"""
Test script for the new single API endpoint
Tests the /process-csv endpoint that processes CSV in one request
"""

import requests
import json
import time
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_single_api():
    """Test the new single API endpoint"""
    print("🚀 Testing Single API Endpoint: /process-csv")
    print("=" * 50)
    
    # Create sample CSV file if it doesn't exist
    sample_file = Path("test_single_api.csv")
    if not sample_file.exists():
        print("📝 Creating sample CSV file...")
        sample_file.write_text("""Test Title,Test Steps,Expected Outcome
Login Test,1. Navigate to login page 2. Enter valid username 3. Enter valid password 4. Click login button,User should be successfully logged in and redirected to dashboard
Registration Test,a. Go to registration page b. Fill in all required fields c. Click submit button d. Verify email confirmation,New user account should be created and confirmation email sent
Password Reset,i. Click forgot password link ii. Enter registered email address iii. Click send reset link iv. Check email for reset instructions,Password reset email should be sent to the user
Product Search,Step 1: Open product catalog Step 2: Enter search term Step 3: Click search button Step 4: Verify results,Search should return relevant products matching the search term
Add to Cart,• Select a product • Click add to cart button • Verify cart icon updates • Check cart contents,Product should be added to cart and cart count should increase""")
        print(f"✅ Sample file created: {sample_file}")
    
    # Test the single API endpoint
    print("\n📤 Testing single API endpoint...")
    
    try:
        with open(sample_file, 'rb') as f:
            files = {'file': ('test_single_api.csv', f, 'text/csv')}
            data = {
                'output_format': 'json',
                'include_metadata': 'true',
                'sample_percentage': 0.2
            }
            
            print("🔄 Sending request to /process-csv...")
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/api/v1/process-csv",
                files=files,
                data=data,
                timeout=60  # 60 second timeout
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"⏱️  Request completed in {total_time:.2f} seconds")
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success! Processing completed")
                print(f"📈 Processing Time: {result.get('processing_time', 'N/A')} seconds")
                print(f"📊 Total Objects: {result.get('total_objects', 0)}")
                print(f"🎯 Success: {result.get('success', False)}")
                print(f"💬 Message: {result.get('message', 'N/A')}")
                
                # Show pattern info
                if result.get('pattern_info'):
                    pattern_info = result['pattern_info']
                    print(f"\n🔍 Pattern Analysis:")
                    print(f"   Type: {pattern_info.get('pattern_type', 'N/A')}")
                    print(f"   Confidence: {pattern_info.get('confidence', 'N/A')}")
                
                # Show validation results
                if result.get('validation_results'):
                    validation = result['validation_results']
                    print(f"\n✅ Validation Results:")
                    print(f"   Valid Objects: {validation.get('valid_objects', 0)}")
                    print(f"   Invalid Objects: {validation.get('invalid_objects', 0)}")
                    print(f"   Errors: {len(validation.get('errors', []))}")
                    print(f"   Warnings: {len(validation.get('warnings', []))}")
                
                # Show sample output
                if result.get('data') and len(result['data']) > 0:
                    print(f"\n📄 Sample Output (First Object):")
                    sample_data = result['data'][0]
                    print(json.dumps(sample_data, indent=2))
                
                print("\n🎉 Single API test completed successfully!")
                
            else:
                print("❌ Request failed!")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                
    except requests.exceptions.Timeout:
        print("⏰ Request timed out! The file might be too large for single processing.")
        print("💡 Try using the multi-step API for larger files.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        # Clean up test file
        if sample_file.exists():
            sample_file.unlink()
            print(f"🧹 Cleaned up test file: {sample_file}")

def test_single_api_with_large_file():
    """Test single API with a larger file to see size limits"""
    print("\n" + "=" * 50)
    print("🧪 Testing Single API with Size Limits")
    print("=" * 50)
    
    # Create a larger CSV file (simulate > 10MB)
    large_file = Path("test_large_file.csv")
    print("📝 Creating large CSV file...")
    
    # Generate large CSV content
    csv_content = "Test Title,Test Steps,Expected Outcome\n"
    for i in range(1000):  # 1000 rows
        csv_content += f"Test {i},1. Step one 2. Step two 3. Step three,Expected result {i}\n"
    
    large_file.write_text(csv_content)
    file_size = large_file.stat().st_size
    print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
    
    try:
        with open(large_file, 'rb') as f:
            files = {'file': ('test_large_file.csv', f, 'text/csv')}
            
            print("🔄 Testing with large file...")
            response = requests.post(
                f"{BASE_URL}/api/v1/process-csv",
                files=files,
                timeout=30
            )
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        # Clean up
        if large_file.exists():
            large_file.unlink()
            print(f"🧹 Cleaned up large test file: {large_file}")

def main():
    """Run all tests"""
    print("🧪 Single API Endpoint Test Suite")
    print("=" * 60)
    
    # Test 1: Normal single API
    test_single_api()
    
    # Test 2: Size limit testing
    test_single_api_with_large_file()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n💡 Usage Examples:")
    print("   curl -X POST 'http://localhost:8000/api/v1/process-csv' \\")
    print("        -F 'file=@your_file.csv' \\")
    print("        -F 'include_metadata=true'")
    print("\n📚 API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
