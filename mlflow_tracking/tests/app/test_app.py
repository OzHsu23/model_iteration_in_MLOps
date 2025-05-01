# tests/test_app.py

import pytest
from fastapp.testclient import TestClient
from training.app.app import app

client = TestClient(app)

def test_start_retrain_and_check_status(test_settings_dict):
    """Test starting a retrain job and checking its status."""

    # Step 1: Call /start_retrain
    response = client.post("/start_retrain", json=test_settings_dict)
    assert response.status_code == 200
    json_data = response.json()
    assert "job_id" in json_data
    job_id = json_data["job_id"]

    # Step 2: Call /retrain_status
    status_response = client.get("/retrain_status", params={"job_id": job_id})
    assert status_response.status_code == 200
    status_json = status_response.json()
    assert status_json["job_id"] == job_id
    assert status_json["status"] in ("pending", "running", "completed", "failed")