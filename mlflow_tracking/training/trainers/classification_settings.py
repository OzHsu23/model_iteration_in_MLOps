# trainers/classification_settings.py

from dataclasses import dataclass
from typing import Optional, List
import torch

@dataclass
class ClassificationSettings:
    # Common settings
    data_dir: str
    img_size: int
    seed: int
    
    # Data settings
    train_csv: str
    val_csv: str

    # Model settings
    model_name: str
    num_classes: int
    pretrained: bool
    weight_path: Optional[str]
    input_size: List[int]

    # Training settings
    batch_size: int
    num_epochs: int
    learning_rate: float

    # MLflow
    experiment_name: str
    tracking_uri: Optional[str]

    device: torch.device

    @classmethod
    def from_config(cls, config):
        img_size = config.get_common_param("img_size", 256)
        return cls(
            data_dir=config.get_common_param("data_dir"),
            img_size=img_size,
            seed=config.get_common_param("seed", 42),
            train_csv=config.get_task_data_param("train_csv"),
            val_csv=config.get_task_data_param("val_csv"),
            model_name=config.get_task_model_param("model_name", "efficientnet_b1"),
            num_classes=config.get_task_model_param("num_classes", 2),
            pretrained=config.get_task_model_param("pretrained", True),
            weight_path=config.get_task_model_param("weight_path", None),
            input_size=[img_size, img_size],
            batch_size=config.get_task_training_param("batch_size", 8),
            num_epochs=config.get_task_training_param("num_epochs", 10),
            learning_rate=config.get_task_training_param("learning_rate", 3e-4),
            experiment_name=config.get("experiment_name", "Classification_Experiment"),
            tracking_uri=config.get_mlflow_param("tracking_uri"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )