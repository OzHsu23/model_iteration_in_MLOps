# training/api/schemas.py

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict





############## # FastAPI Schemas for Settings ###########

# Define a schema for incoming settings
class ApiSettings(BaseModel):
    task_type: str
    experiment_name: str
    common: dict
    classification: dict


########### # Settings for different tasks ###########

# Common settings shared across different tasks
class CommonSettings(BaseModel):
    data_dir: str
    img_size: int = 256
    random_seed: int = 42

    model_config = ConfigDict(extra="forbid")  # Forbid unexpected fields

# Data-related settings for classification
class DataSettings(BaseModel):
    train_csv: str
    val_csv: str

    model_config = ConfigDict(extra="forbid")

# Model-related settings for classification
class ModelSettings(BaseModel):
    model_name: str
    num_classes: int
    pretrained: bool
    weight_path: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

# Training-related settings for classification
class TrainingSettings(BaseModel):
    batch_size: int
    num_epochs: int
    learning_rate: float

    model_config = ConfigDict(extra="forbid")

# Full settings specifically for classification tasks
class ClassificationTaskSettings(BaseModel):
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings

    model_config = ConfigDict(extra="forbid")

# MLflow tracking settings
class MLflowSettings(BaseModel):
    tracking_uri: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

# Main train settings schema
class TrainSettings(BaseModel):
    task_type: Literal["classification"]  # Currently only supports "classification"
    experiment_name: str
    common: CommonSettings
    task: ClassificationTaskSettings     # Generalized "task" field for task-specific configs
    mlflow: Optional[MLflowSettings]

    model_config = ConfigDict(extra="forbid")

