"""Test the available dermatologists endpoint"""
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
print(f"   Login successful!")

# Test available dermatologists endpoint
print("\n2. Testing GET /ops/dermatologists/available...")
headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(f"{BASE_URL}/ops/dermatologists/available", headers=headers)

print(f"   Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"   Total dermatologists: {data.get('total')}")
    for d in data.get('dermatologists', []):
        print(f"     - ID={d['id']}, name={d['full_name']}, credentials={d['credentials']}")
else:
    print(f"   Error: {resp.text}")
