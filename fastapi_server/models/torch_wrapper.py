'''
Purpose: torch saved .pt model implementation with config loading
Path: fastapi_server/models/torch_wrapper.py
'''

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
import os
from .base import BasePreprocessor, BaseModelWrapper

# ========== Config Loader ==========
def load_model_config_yaml(path="torch_model_config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"{path} not found")

# ========== Preprocessor ==========
class TorchPreprocessor(BasePreprocessor):
    def __init__(self, mean_std=None):
        if mean_std:
            mean = mean_std.get("mean", [0.485, 0.456, 0.406])
            std = mean_std.get("std", [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def preprocess(self, image: Image.Image):
        return self.transform(image).unsqueeze(0)  # shape [1, C, H, W]

# ========== Model Wrapper ==========
class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, config_path="./models/torch_model_config.yaml"):
        config = load_model_config_yaml(config_path)
        model_config = config.get("MODEL", {})

        model_path = model_config.get("PATH", None)
        if model_path is None:
            raise ValueError("No model PATH specified in config.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=device)
        if hasattr(self.model, 'eval'):
            self.model.eval()

        mean_std = model_config.get("PREPROCESS", None)
        pre = TorchPreprocessor(mean_std)
        super().__init__(pre)

        self.class_names = model_config.get("CLASSES", [str(i) for i in range(1000)])

    def predict(self, image: Image.Image):
        tensor = self.preprocessor.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = self.class_names[idx.item()]
        return label, conf.item()