#  training/tests/test_config.py
import pytest
from training.config import Config

def test_config_core_values(config):
    """Verify core values loaded from real config file."""
    assert config.task_type == "classification"
    assert config.experiment_name == "efficientnet_classification_exp"
    assert config.get_common_param("img_size") == 256
    assert config.get_task_data_param("train_csv") == "train.csv"
    assert config.get_task_model_param("model_name") == "efficientnet_b1"
    assert config.get_task_training_param("batch_size") == 4
    assert config.get_mlflow_param("tracking_uri") == "/mlflow_tracking/mlruns"


def test_config_info_is_json_string(config):
    """Check that .info() returns valid JSON string"""
    info = config.info()
    assert isinstance(info, str)
    assert "efficientnet_classification_exp" in info

