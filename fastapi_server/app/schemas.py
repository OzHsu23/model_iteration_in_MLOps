# app/schemas.py

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, validator
from datetime import datetime

############## FastAPI Schemas for Settings ##############

# ==========================
# Model settings
# ==========================
class ModelSettings(BaseModel):
    model_type: Literal["torch_mlflow", "torch_pure", "tf_mlflow", "tf_pure"]
    run_id: Optional[str] = ""
    tracking_uri: Optional[str] = ""
    local_path: Optional[str] = ""
    yaml_path: Optional[str] = None
    model_name: str                     
    num_classes: int                    
    class_names: Optional[list[str]] = None
    img_size: Optional[int] = 256       

    model_config = ConfigDict(extra="forbid")

# ==========================
# Preprocessing settings
# ==========================
class PreprocessingSettings(BaseModel):
    resize: Optional[int] = 256

    model_config = ConfigDict(extra="forbid")

# ==========================
# Postprocessing settings
# ==========================
class PostprocessingSettings(BaseModel):
    type: Optional[str] = "default"

    model_config = ConfigDict(extra="forbid")

# ==========================
# Server settings
# ==========================
class ServerSettings(BaseModel):
    port: int = 8000

    model_config = ConfigDict(extra="forbid")

# ==========================
# Logging settings
# ==========================
class LoggingSettings(BaseModel):
    log_path: str = "logs/inference_logs.csv"
    enable: bool = True

    model_config = ConfigDict(extra="forbid")

# ==========================
# Full application settings
# ==========================
class AppSettings(BaseModel):
    task_type: Literal["classification", "detection", "segmentation"]   
    model: ModelSettings
    preprocessing: Optional[PreprocessingSettings] = PreprocessingSettings()
    postprocessing: Optional[PostprocessingSettings] = PostprocessingSettings()
    server: ServerSettings
    logging: LoggingSettings

    model_config = ConfigDict(extra="forbid")

############## Input Validation for API ##############
class ImageInput(BaseModel):
    filename: str = Field(..., description="Uploaded file name")

    @validator("filename")
    def validate_image_extension(cls, v: str) -> str:
        allowed_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        if not v.lower().endswith(allowed_ext):
            raise ValueError(f"Invalid image format. Supported formats: {allowed_ext}")
        return v

############## Log Record Schema ##############
class LogRecord(BaseModel):
    inference_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_version: Optional[str] = "unknown"
    framework: Optional[str] = "unknown"
    model_format: Optional[str] = "unknown"
    filename: str
    pred: str
    confidence: float
    label: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
