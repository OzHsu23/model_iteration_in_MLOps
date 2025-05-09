import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import json
from fastapi.testclient import TestClient
from app.app import app

@pytest.fixture(scope="session")
def client():
    """Reusable FastAPI test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session")
def test_settings_dict():
    """Load reusable mock settings from test_data/test_setting.json"""
    with open("tests/test_data/test_setting.json", "r") as f:
        return json.load(f)
