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
        self.input_size = input_size  # Save expected input size for external reference
        self.device = device
        # Load backbone model based on model_name
        if model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}, default is 'efficientnet_b1")

        # If a custom weight path is provided, load weights
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.load_state_dict(state_dict)
        

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            logits (Tensor): Raw output logits.
            probs (Tensor): Softmax probabilities per class.
        """
        logits = self.backbone(x)           # Get raw logits from backbone
        probs = F.softmax(logits, dim=1)    # Compute softmax probabilities
        return probs
