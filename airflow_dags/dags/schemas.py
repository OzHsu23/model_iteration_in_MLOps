# dags/schemas.py

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator


############## DAG Settings Schema ##############

# ==========================
# Production Line Simulation Settings
# ==========================
class ProductionLineConfig(BaseModel):
    inference_server_api: str
    inference_image_dir: str = "data/inference_images"
    repeat: int = 3  # Number of times to repeat image prediction simulation

    model_config = ConfigDict(extra="forbid")

    @field_validator("inference_server_api")
    @classmethod
    def check_url(cls, v):
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("inference_server_api must be a valid HTTP or HTTPS URL")
        return v

    @field_validator("inference_image_dir")
    @classmethod
    def ensure_dir_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"inference_image_dir not found: {v}")
        return v

    @field_validator("repeat")
    @classmethod
    def validate_repeat(cls, v):
        if v <= 0:
            raise ValueError("repeat must be a positive integer")
        return v



# ==========================
# Monitor Accuracy Settings
# ==========================
class MonitorYieldConfig(BaseModel):
    log_path: Optional[Path] = None
    flag_path: str = "resources/flags/need_retrain.flag"
    recent_window: int = 100
    yield_threshold: float = 0.7
    yield_drop_tolerance: float = 0.05
    
    monitor_delay_sec: int = 60

    model_config = ConfigDict(extra="forbid")

    @field_validator("flag_path")
    @classmethod
    def ensure_flag_dir_exists(cls, v):
        flag_dir = Path(v).parent
        if not flag_dir.exists():
            flag_dir.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("yield_threshold", "yield_drop_tolerance")
    @classmethod
    def check_range_0_1(cls, v, info):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v

    @field_validator("monitor_delay_sec")
    @classmethod
    def check_positive_delay(cls, v):
        if v < 0:
            raise ValueError("monitor_delay_sec must be non-negative")
        return v


# ==========================
# Prepare Training Data Settings
# ==========================
class PrepareTrainingDataConfig(BaseModel):
    inference_image_dir: str = "data/inference_images"
    selected_sample_dir: str = "data/selected_recent_samples"
    training_template_path: str = "configs/training_template.json"
    final_training_zip: str = "data/train_package.zip"
    merge_with_previous: Optional[str] = None
    b1_inference_api: str = "http://127.0.0.1:8011/batch_predict"
    b2_inference_api: str = "http://127.0.0.1:8012/batch_predict"
    confidence_threshold: float = 0.6
    val_ratio: float = 0.05

    model_config = ConfigDict(extra="forbid")

    @field_validator("confidence_threshold", "val_ratio")
    @classmethod
    def check_range_0_1(cls, v, info):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} must be between 0.0 and 1.0")
        return v

    @field_validator("inference_image_dir", "selected_sample_dir")
    @classmethod
    def ensure_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("training_template_path")
    @classmethod
    def check_template_exists(cls, v):
        if not Path(v).is_file():
            raise ValueError(f"training_template_path not found: {v}")
        return v


# ==========================
# Retrain Settings
# ==========================
class RetrainConfig(BaseModel):
    retrain_zip_path: str = "data/selected_recent_samples.zip"
    retrain_server_api: str = "http://127.0.0.1:8020"
    max_wait_sec: int = 600
    poll_interval_sec: int = 10

    model_config = ConfigDict(extra="forbid")

    @field_validator("retrain_zip_path")
    @classmethod
    def ensure_zip_parent(cls, v):
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("retrain_server_api")
    @classmethod
    def check_url(cls, v):
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("retrain_server_api must be a valid URL")
        return v

    @field_validator("max_wait_sec", "poll_interval_sec")
    @classmethod
    def check_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v


# ==========================
# Deploy Settings
# ==========================
class DeployConfig(BaseModel):
    inference_server_api: str
    job_id_to_deploy: str = ""

    model_config = ConfigDict(extra="forbid")


# ==========================
# Full Nested Config
# ==========================
class FullConfig(BaseModel):
    production_line: ProductionLineConfig 
    monitor: MonitorYieldConfig
    prepare_training_data: PrepareTrainingDataConfig
    retrain: RetrainConfig
    deploy: DeployConfig

    model_config = ConfigDict(extra="forbid")
