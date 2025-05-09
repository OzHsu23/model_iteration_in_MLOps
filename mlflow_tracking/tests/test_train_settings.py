# tests/test_train_settings.py

def test_top_level_keys_exist(test_settings_dict):
    """Ensure all required top-level keys exist and are correct type"""
    assert isinstance(test_settings_dict.get("task_type"), str)
    assert isinstance(test_settings_dict.get("experiment_name"), str)
    assert isinstance(test_settings_dict.get("common"), dict)
    assert isinstance(test_settings_dict.get("task"), dict)
    assert isinstance(test_settings_dict.get("mlflow"), dict)


def test_common_block_types(test_settings_dict):
    """Validate types in 'common' block"""
    common = test_settings_dict["common"]
    assert isinstance(common.get("data_dir"), str)
    assert isinstance(common.get("img_size"), int)
    assert isinstance(common.get("random_seed"), int)


def test_task_data_block_types(test_settings_dict):
    """Validate types in 'task.data' block"""
    data = test_settings_dict["task"]["data"]
    assert isinstance(data.get("train_csv"), str)
    assert isinstance(data.get("val_csv"), str)


def test_task_model_block_types(test_settings_dict):
    """Validate types in 'task.model' block"""
    model = test_settings_dict["task"]["model"]
    assert isinstance(model.get("model_name"), str)
    assert isinstance(model.get("num_classes"), int)
    assert isinstance(model.get("pretrained"), bool)
    # weight_path 可為 None 或 str
    assert model.get("weight_path") is None or isinstance(model.get("weight_path"), str)


def test_task_training_block_types(test_settings_dict):
    """Validate types in 'task.training' block"""
    training = test_settings_dict["task"]["training"]
    assert isinstance(training.get("batch_size"), int)
    assert isinstance(training.get("num_epochs"), int)
    assert isinstance(training.get("learning_rate"), float)
    assert training["batch_size"] > 0
    assert 0.0 < training["learning_rate"] < 1.0


def test_mlflow_block_types(test_settings_dict):
    """Validate types in 'mlflow' block"""
    mlflow = test_settings_dict["mlflow"]
    assert isinstance(mlflow.get("tracking_uri"), str)
