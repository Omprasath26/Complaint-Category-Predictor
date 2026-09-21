import requests
import time

def test_server():
    print("Testing Complaint Category Predictor Server...")
    print("=" * 60)
    
    # Wait for server to start
    print("Waiting for server to be ready...")
    time.sleep(3)
    
    # Test 1: Check if root serves HTML
    print("\n1. Testing root endpoint (should serve HTML):")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"   Status Code: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
        
        # Check if it's HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            print("   [SUCCESS] Serving HTML content")
            # Check if it contains expected HTML elements
            content = response.text.lower()
            if '<html' in content and '<body' in content:
                print("   [SUCCESS] Valid HTML structure detected")
            else:
                print("   [WARNING] HTML structure may be incomplete")
        else:
            print("   [WARNING] Not serving HTML (might be JSON API response)")
            print(f"   Response preview: {response.text[:100]}...")
            
    except requests.exceptions.ConnectionError:
        print("   [ERROR] Cannot connect to server at http://localhost:8000/")
        print("   Make sure the server is running with: python app.py")
        return False
    except Exception as e:
        print(f"   [ERROR] Unexpected error: {e}")
        return False
    
    # Test 2: Check API endpoints
    print("\n2. Testing API endpoints:")
    
    # Health endpoint
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   /health: Status={data['status']}, Model Loaded={data['model_loaded']}")
        else:
            print(f"   /health: Error - Status {response.status_code}")
    except Exception as e:
        print(f"   /health: Error - {e}")
    
    # Categories endpoint
    try:
        response = requests.get("http://localhost:8000/categories", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   /categories: {len(data['categories'])} categories available")
        else:
            print(f"   /categories: Error - Status {response.status_code}")
    except Exception as e:
        print(f"   /categories: Error - {e}")
    
    # Test 3: Test prediction
    print("\n3. Testing prediction with example complaint:")
    test_complaint = "My payment was deducted but my order was not confirmed."
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"complaint": test_complaint},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   Complaint: '{test_complaint[:40]}...'")
            print(f"   Category: {data['category']}")
            print(f"   Priority: {data['priority']}")
            print(f"   Confidence: {data['confidence']:.1%}")
            
            if data['category'] == 'Payment' and data['priority'] == 'High':
                print("   [SUCCESS] Prediction matches expected output!")
            else:
                print("   [WARNING] Prediction doesn't match expected output")
        else:
            print(f"   [ERROR] Prediction failed - Status {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   [ERROR] Prediction error: {e}")
    
    # Test 4: Check API docs
    print("\n4. Checking API documentation:")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("   /docs: API documentation is available")
        else:
            print(f"   /docs: Status {response.status_code}")
    except Exception as e:
        print(f"   /docs: Error - {e}")
    
    print("\n" + "=" * 60)
    print("Server Test Complete!")
    print("\nAccess Points:")
    print("  • Frontend: http://localhost:8000/")
    print("  • API Docs: http://localhost:8000/docs")
    print("  • API Info: http://localhost:8000/api")
    print("  • Health:   http://localhost:8000/health")
    
    return True

if __name__ == "__main__":
    test_server()