# training/tests/test_dataset.py

import pytest
from training.datasets.classification_dataset import ClassificationDataset
from training.utils.preprocessing import TrainAugmentation

def test_dataset_loading(classification_settings):
    """Test if the dataset can load images and labels correctly."""
    dataset = ClassificationDataset(
        split='train',
        settings=classification_settings,
        transform=TrainAugmentation(classification_settings)
    )
    assert len(dataset) > 0

    image, label = dataset[0]
    assert image.shape[0] == 3  # C
    assert image.shape[1] == classification_settings.img_size  # H
    assert image.shape[2] == classification_settings.img_size  # W