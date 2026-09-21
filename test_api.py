import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing Complaint Category Predictor API...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
        else:
            print(f"   Error: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   Error: Cannot connect to server. Make sure the server is running.")
        return False
    
    # Test 2: Get categories
    print("\n2. Testing categories endpoint...")
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            categories_data = response.json()
            print(f"   Available categories: {', '.join(categories_data['categories'])}")
            print(f"   Total categories: {categories_data['count']}")
        else:
            print(f"   Error: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   Error: Cannot connect to server.")
        return False
    
    # Test 3: Test prediction with example from problem statement
    print("\n3. Testing prediction endpoint...")
    test_complaint = "My payment was deducted but my order was not confirmed."
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"complaint": test_complaint},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"   Complaint: {test_complaint}")
            print(f"   Predicted Category: {prediction_data['category']}")
            print(f"   Priority: {prediction_data['priority']}")
            print(f"   Confidence: {prediction_data['confidence']:.2%}")
            
            # Check if it matches expected output
            if prediction_data['category'] == 'Payment' and prediction_data['priority'] == 'High':
                print("   [PASS] Test PASSED: Matches expected output (Payment, High)")
            else:
                print("   [FAIL] Test FAILED: Does not match expected output")
        else:
            print(f"   Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("   Error: Cannot connect to server.")
        return False
    
    # Test 4: Test additional examples
    print("\n4. Testing additional examples...")
    test_cases = [
        ("I cannot login to my account.", "Login", "High"),
        ("My order has not been delivered yet.", "Delivery", "Medium"),
        ("Website is not loading properly.", "Technical", "Medium"),
        ("Customer service is not responding.", "Other", "Low")
    ]
    
    for complaint, expected_category, expected_priority in test_cases:
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"complaint": complaint},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                category_match = prediction_data['category'] == expected_category
                priority_match = prediction_data['priority'] == expected_priority
                
                status = "[PASS]" if category_match and priority_match else "[FAIL]"
                print(f"   {status} '{complaint[:30]}...' -> {prediction_data['category']} ({prediction_data['priority']})")
            else:
                print(f"   [ERROR] Error for '{complaint[:30]}...': Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   Error: Cannot connect to server.")
            break
    
    # Test 5: Test empty complaint
    print("\n5. Testing error handling (empty complaint)...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"complaint": ""},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print("   [PASS] Correctly rejected empty complaint")
        else:
            print(f"   [FAIL] Expected 400, got {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   Error: Cannot connect to server.")
    
    print("\n" + "=" * 50)
    print("API testing completed!")
    return True

if __name__ == "__main__":
    # Wait a moment for server to start
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    test_api()