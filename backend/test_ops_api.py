"""Test the ops API endpoint"""
import requests

BASE_URL = "http://localhost:8000"

# Login as ops_admin
print("1. Logging in as ops_admin...")
login_response = requests.post(f"{BASE_URL}/login", json={
    "username": "ops_admin",
    "password": "123456"
})

if login_response.status_code != 200:
    print(f"   Login failed: {login_response.status_code}")
    print(f"   Response: {login_response.text}")
    exit(1)

login_data = login_response.json()
token = login_data.get("access_token")
role = login_data.get("role")
print(f"   Login successful! Role returned: {role}")

# Test /ops/consultations endpoint
print("\n2. Testing GET /ops/consultations...")
headers = {"Authorization": f"Bearer {token}"}
ops_response = requests.get(f"{BASE_URL}/ops/consultations", headers=headers)

print(f"   Status: {ops_response.status_code}")
if ops_response.status_code == 200:
    data = ops_response.json()
    print(f"   Stats: {data.get('stats')}")
    print(f"   Consultations count: {len(data.get('consultations', []))}")
    for c in data.get('consultations', []):
        print(f"     - ID={c['id']}, patient={c['patient']['full_name']}, status={c['status']}")
else:
    print(f"   Error: {ops_response.text}")

# Test /ops/consultations?status=pending_assignment
print("\n3. Testing GET /ops/consultations?status=pending_assignment...")
ops_response2 = requests.get(f"{BASE_URL}/ops/consultations?status=pending_assignment", headers=headers)

print(f"   Status: {ops_response2.status_code}")
if ops_response2.status_code == 200:
    data = ops_response2.json()
    print(f"   Consultations count: {len(data.get('consultations', []))}")
    for c in data.get('consultations', []):
        print(f"     - ID={c['id']}, patient={c['patient']['full_name']}, status={c['status']}")
else:
    print(f"   Error: {ops_response2.text}")
