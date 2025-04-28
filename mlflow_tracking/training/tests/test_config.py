#  training/tests/test_config.py
import pytest
from training.config import Config

def test_config_loading():
    """Test if the config file can be loaded and parsed correctly."""
    config = Config("./tests/test_settings.json")
    assert config.task_type == "classification"
    assert config.get_common_param("img_size") == 256
    assert config.get_task_model_param("model_name") == "efficientnet_b1"

