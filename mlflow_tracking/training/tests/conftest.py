# training/tests/conftest.py

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