# training/tests/conftest.py
import json
import pytest
import cv2
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.app import app
from training.config import Config
from training.trainers.classification_settings import ClassificationSettings

@pytest.fixture(scope="session")
def config():
    """Fixture to load config object."""
    return Config("./tests/test_data/test_setting.json")

@pytest.fixture(scope="session")
def classification_settings(config):
    """Fixture to load classification settings object."""
    return ClassificationSettings.from_config(config)

@pytest.fixture
def test_settings_dict():
    """Fixture to load the JSON dict from test_setting.json."""
    with open("./tests/test_data/test_setting.json", "r") as f:
        data = json.load(f)
    return data

@pytest.fixture(scope="module")
def client():
    """Reusable FastAPI test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def retrain_job_id(client, test_settings_dict):
    """Fixture to submit retrain and return job_id."""
    response = client.post("/start_retrain", json=test_settings_dict)
    assert response.status_code == 200
    return response.json()["job_id"]

@pytest.fixture(scope="module")
def valid_classification_settings(tmp_path_factory):
    """Create a mock classification dataset with images and CSVs."""
    
    base = tmp_path_factory.mktemp("valid_cls_data")

    # Create train.csv and val.csv
    df = pd.DataFrame({
        "filename": ["img_1.jpg", "img_2.jpg"],
        "label": ["0", "1"]  # note: string labels
    })
    df.to_csv(base / "train.csv", index=False)
    df.to_csv(base / "val.csv", index=False)

    # Create dummy images
    for fname in ["img_1.jpg", "img_2.jpg"]:
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(base / fname), img)

    class Dummy:
        data_dir = str(base)
        img_size = 32
        train_csv = "train.csv"
        val_csv = "val.csv"

    return Dummy()