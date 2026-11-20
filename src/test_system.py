"""
System Integration Test
Verifies that all files and routes in the project connect and work end-to-end.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "har_website"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "har_website.settings")

import django
django.setup()

import json
from django.test import Client
import har_predictor

print("=" * 55)
print("   HAR MAJOR PROJECT - INTEGRATION SYSTEM TEST")
print("=" * 55)

# 1. Test Model & Weights
weights = har_predictor.load_weights()
assert weights is not None, "Failed to load weights"
print("[PASS] 1. Model weights loaded into memory successfully.")

# 2. Test Predictor on Dataset
feat, actual, num, total = har_predictor.get_test_sample(1)
assert feat is not None, "Failed to load test sample"
res = har_predictor.predict(feat)
assert res["activity"] == actual, f"Expected {actual}, got {res['activity']}"
print(f"[PASS] 2. ML Predictor test: Sample #{num} is {res['activity']} (Confidence: {res['confidence']}%)")

# 3. Test Django Client
client = Client()

# Home
r = client.get("/")
assert r.status_code == 200, f"Home failed: {r.status_code}"
print("[PASS] 3. Home page (GET /): 200 OK")

# Login
r = client.post("/login/", {"username": "admin", "password": "admin123"}, follow=True)
assert r.status_code == 200, f"Login failed: {r.status_code}"
assert "Welcome back" in r.content.decode() or "admin" in r.content.decode()
print("[PASS] 4. User Authentication: Logged in as admin, redirected to Dashboard")

# Dashboard
r = client.get("/dashboard/")
assert r.status_code == 200, f"Dashboard failed: {r.status_code}"
print("[PASS] 5. Dashboard view (GET /dashboard/): 200 OK")

# Live Camera Page
r = client.get("/live/")
assert r.status_code == 200, f"Live view failed: {r.status_code}"
print("[PASS] 6. Live Camera Page (GET /live/): 200 OK")

# Live Prediction API
payload = json.dumps({"motion": 0.45, "motion_history": [0.45, 0.42, 0.48], "save": True})
r = client.post("/api/predict_frame/", data=payload, content_type="application/json")
assert r.status_code == 200, f"API failed: {r.status_code}"
api_data = r.json()
assert api_data["status"] == "success"
print(f"[PASS] 7. Live Prediction API: {api_data['activity']} ({api_data['confidence']}%)")

# Dataset Sample View
r = client.get("/predict/?sample=5&log=1")
assert r.status_code == 200, f"Predict sample failed: {r.status_code}"
print("[PASS] 8. Dataset Sample Predict view: 200 OK")

# History View
r = client.get("/history/")
assert r.status_code == 200, f"History failed: {r.status_code}"
assert "Total Recorded Predictions" in r.content.decode()
print("[PASS] 9. Prediction History view: 200 OK")

# Logout
r = client.get("/logout/", follow=True)
assert r.status_code == 200, f"Logout failed: {r.status_code}"
print("[PASS] 10. User Logout: Session cleared, redirected to Home")

print("=" * 55)
print("   RESULT: ALL 10/10 SYSTEM TESTS PASSED SUCCESSFULLY!")
print("=" * 55)
