# tests/test_train_settings.py

import pytest
import json
from training.api.schemas import TrainSettings



def test_train_settings_validation(test_settings_dict):
    """Test that TrainSettings correctly validates test_settings.json content."""
    settings = TrainSettings(**test_settings_dict)
    
    # Check some key attributes
    assert settings.task_type == "classification"
    assert settings.experiment_name == "efficientnet_classification_exp"
    assert settings.common.img_size == 256
    assert settings.task.model.model_name == "efficientnet_b1"
    assert settings.task.training.num_epochs == 2
    assert settings.mlflow.tracking_uri == "/mlflow_tracking/mlruns"
