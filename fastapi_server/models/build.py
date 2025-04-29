# models/build.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "efficientnet_b1", pretrained: bool = False):
        """
        Build a simple classifier based on the model_name.
        Currently supports only efficientnet_b1.
        """
        super(SimpleClassifier, self).__init__()

        if model_name != "efficientnet_b1":
            raise ValueError(f"Unsupported model_name: {model_name}, only 'efficientnet_b1' is supported.")
        
        self.backbone = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        return probs

def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Build and return a model instance ready to load state_dict.
    """
    model = SimpleClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False   # 推論時通常是 False，載自己的weight
    )
    return model
