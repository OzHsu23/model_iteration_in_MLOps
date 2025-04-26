'''
Purpose: torchvision implementation with config loading
Path: fastapi_server/models/tv_wrapper.py
'''

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import yaml
import os
from .base import BasePreprocessor, BaseModelWrapper

# ========== Config Loader ==========
def load_model_config_yaml(path="tv_model_config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"{path} not found")

# ========== Preprocessor ==========
class TorchVisionPreprocessor(BasePreprocessor):
    def __init__(self, weights_meta):
        # Use mean and std from weights meta if available, otherwise use default values
        if weights_meta:
            mean = weights_meta.get("mean", [0.485, 0.456, 0.406])
            std = weights_meta.get("std", [0.229, 0.224, 0.225])
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
        return self.transform(image).unsqueeze(0)  # Return tensor with shape [1, C, H, W]

# ========== Model Wrapper ==========
class TorchVisionWrapper(BaseModelWrapper):
    def __init__(self, config_path="./models/tv_model_config.yaml"):
        config = load_model_config_yaml(config_path)
        model_config = config.get("MODEL", {})

        model_name = model_config.get("NAME", "resnet50")
        pretrained_name = model_config.get("PRETRAINED", None)

        # Get model function from torchvision.models
        model_fn = getattr(models, model_name)

        # Load weights if specified
        if pretrained_name:
            weights_enum = getattr(models, pretrained_name)
            weights = weights_enum.IMAGENET1K_V2
            self.model = model_fn(weights=weights)
            weights_meta = weights.meta
        else:
            self.model = model_fn(weights=None)
            weights_meta = None

        pre = TorchVisionPreprocessor(weights_meta)
        super().__init__(pre)

        self.model.eval()
        self.class_names = weights_meta["categories"] if weights_meta else [str(i) for i in range(1000)]

    def predict(self, image: Image.Image):
        tensor = self.preprocessor.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = self.class_names[idx.item()]
        return label, conf.item()