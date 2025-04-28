# training/tests/test_dataset.py

import pytest
import torch
from training.model_zoo.classifier_model import SimpleClassifier

def test_model_forward(classification_settings):
    """Test if the model can perform a forward pass without errors."""
    model = SimpleClassifier(
        num_classes=classification_settings.num_classes,
        model_name=classification_settings.model_name,
        pretrained=classification_settings.pretrained,
        weight_path=classification_settings.weight_path,
        input_size=classification_settings.input_size,
        device=classification_settings.device
    )

    dummy_input = torch.randn(1, 3, classification_settings.img_size, classification_settings.img_size)
    probs = model(dummy_input)

    assert probs.shape[0] == 1
    assert probs.shape[1] == classification_settings.num_classes
