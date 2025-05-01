# training/tests/test_dataset.py

import pytest
import torch
from training.trainers.classification_trainer import ClassificationTrainer

def test_trainer_one_step(classification_settings):
    """Test if the trainer can process one batch without errors."""
    trainer = ClassificationTrainer(classification_settings)
    trainer.model.train()

    images, labels = next(iter(trainer.train_loader))
    images, labels = images.to(trainer.device), labels.to(trainer.device)

    probs = trainer.model(images)
    loss = trainer.criterion(probs, labels)

    assert loss.item() > 0