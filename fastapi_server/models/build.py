# models/build.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "efficientnet_b1", pretrained: bool = False):
        super(SimpleClassifier, self).__init__()

        if model_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b1(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == "regnet_y":
            weights = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.regnet_y_3_2gf(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                f"Supported: efficientnet_b1, efficientnet_v2_s, regnet_y"
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
        pretrained=False  
    )
    return model

