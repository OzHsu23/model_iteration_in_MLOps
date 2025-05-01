# training/model_zoo/classifier_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes, model_name, pretrained, weight_path, input_size, device):
        """
        Args:
            num_classes (int): Number of output classes.
            model_name (str): Backbone model name.
            pretrained (bool): Whether to use pretrained ImageNet weights.
            weight_path (str): Optional. Path to custom weights to load.
            input_size (tuple): Expected input image size (height, width).
            device (torch.device): Device to run the model on.
        """
        super(SimpleClassifier, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.device = device

        # Select backbone
        if model_name == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b1(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'regnet_y':
            weights = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.regnet_y_3_2gf(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )

        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported: efficientnet_b1, efficientnet_v2_s, regnet_y")

        # Optional: load custom weights
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location=device)
            self.load_state_dict(state_dict)

        self.to(device)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            probs (Tensor): Softmax probabilities per class.
        """
        logits = self.backbone(x)
        probs = F.softmax(logits, dim=1)
        return probs

