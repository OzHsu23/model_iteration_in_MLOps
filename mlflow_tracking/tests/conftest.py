# training/tests/conftest.py
import json
import pytest
from training.config import Config
from training.trainers.classification_settings import ClassificationSettings

@pytest.fixture(scope="session")
def config():
    """Fixture to load config object."""
    return Config("./tests/test_settings.json")

@pytest.fixture(scope="session")
def classification_settings(config):
    """Fixture to load classification settings object."""
    return ClassificationSettings.from_config(config)

@pytest.fixture
def test_settings_dict():
    """Fixture to load the JSON dict from test_settings.json."""
    with open("./tests/test_settings.json", "r") as f:
        data = json.load(f)
    return data