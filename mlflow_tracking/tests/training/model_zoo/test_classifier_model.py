# training/tests/test_classifier_model.py

import torch
import pytest
from training.model_zoo.classifier_model import SimpleClassifier


def test_model_forward_pass(classification_settings):
    """Test forward pass for SimpleClassifier with dummy input."""
    model = SimpleClassifier(
        num_classes=classification_settings.num_classes,
        model_name=classification_settings.model_name,
        pretrained=False,
        weight_path=None,
        input_size=(classification_settings.img_size, classification_settings.img_size),
        device=classification_settings.device
    )

    dummy_input = torch.randn(2, 3, classification_settings.img_size, classification_settings.img_size).to(classification_settings.device)
    output = model(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, classification_settings.num_classes)

    expected = torch.ones(2, device=output.device)
    assert torch.allclose(output.sum(dim=1), expected, atol=1e-5)


def test_model_invalid_name(classification_settings):
    """Test unsupported model name raises ValueError."""
    with pytest.raises(ValueError):
        SimpleClassifier(
            num_classes=classification_settings.num_classes,
            model_name="invalid_model",
            pretrained=False,
            weight_path=None,
            input_size=(classification_settings.img_size, classification_settings.img_size),
            device=classification_settings.device
        )


