# training/tests/test_classification_trainer.py

import pytest
import torch
from training.trainers.classification_trainer import ClassificationTrainer


def test_trainer_train_cycle(valid_classification_settings):
    """Test one training cycle with mocked classification settings."""
    from training.trainers.classification_settings import ClassificationSettings

    # Inject required fields to mock object
    settings = ClassificationSettings(
        data_dir=valid_classification_settings.data_dir,
        img_size=valid_classification_settings.img_size,
        input_size=valid_classification_settings.img_size,
        seed=42,
        train_csv=valid_classification_settings.train_csv,
        val_csv=valid_classification_settings.val_csv,
        model_name="efficientnet_b1",
        num_classes=2,
        pretrained=False,
        weight_path=None,
        batch_size=2,
        num_epochs=1,
        learning_rate=0.001,
        experiment_name="mock_exp",
        tracking_uri=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        job_id="trainer_test"
    )

    trainer = ClassificationTrainer(settings)
    trainer.train()
    metrics = trainer.get_metrics()
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["finished"] is True


def test_trainer_validate(valid_classification_settings):
    from training.trainers.classification_settings import ClassificationSettings

    settings = ClassificationSettings(
        data_dir=valid_classification_settings.data_dir,
        img_size=valid_classification_settings.img_size,
        input_size=valid_classification_settings.img_size,
        seed=42,
        train_csv=valid_classification_settings.train_csv,
        val_csv=valid_classification_settings.val_csv,
        model_name="efficientnet_b1",
        num_classes=2,
        pretrained=False,
        weight_path=None,
        batch_size=2,
        num_epochs=1,
        learning_rate=0.001,
        experiment_name="mock_exp",
        tracking_uri=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        job_id="trainer_test"
    )

    trainer = ClassificationTrainer(settings)
    acc = trainer.validate()
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
