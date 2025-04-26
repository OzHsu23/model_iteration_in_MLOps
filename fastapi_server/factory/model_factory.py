# factory/model_factory.py

from models.torch_wrapper import TorchModelWrapper
from models.torchvision_wrapper import TorchVisionWrapper
from fastapi import HTTPException
import os
import yaml

import os
import yaml
from fastapi import HTTPException
from models.torch_wrapper import TorchModelWrapper
from models.torchvision_wrapper import TorchVisionWrapper

def load_model_config_yaml(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise HTTPException(404, f"{path} not found")

class ModelFactory:
    @staticmethod
    def create_model(setting):
        model_setting = setting.get("model", {})
        wrapper_type = model_setting.get("wrapper", "torchvision")
        yaml_path = model_setting.get("yaml_path", None)

        if yaml_path is None:
            raise ValueError("yaml_path must be specified under model in setting.json")

        model_config = load_model_config_yaml(yaml_path)

        if wrapper_type == "torch":
            return TorchModelWrapper(config_path=yaml_path)
        elif wrapper_type == "torchvision":
            return TorchVisionWrapper(config_path=yaml_path)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
