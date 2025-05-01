from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, validator
from pathlib import Path

############## DAG Settings Schema ##############

# ==========================
# Monitor Accuracy Settings
# ==========================
class MonitorYieldConfig(BaseModel):
    log_path: str = "resources/inference_logs.csv"
    flag_path: str = "resources/flags/need_retrain.flag"
    recent_window: int = 100
    threshold: Optional[float] = None
    delta: float = 0.05

    model_config = ConfigDict(extra="forbid")

    @validator("log_path")
    def check_log_path_exists(cls, v):
        if not Path(v).is_file():
            raise ValueError(f"Log file does not exist: {v}")
        return v

    @validator("flag_path")
    def ensure_flag_dir_exists(cls, v):
        flag_dir = Path(v).parent
        if not flag_dir.exists():
            flag_dir.mkdir(parents=True, exist_ok=True)
        return v

    @validator("threshold")
    def check_threshold_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v

    @validator("delta")
    def check_delta_positive(cls, v):
        if v < 0.0:
            raise ValueError("delta must be non-negative")
        return v


# ==========================
# Sample Extraction Settings
# ==========================
class SampleExtractionConfig(BaseModel):
    inference_image_dir: str = "data/inference_images"
    selected_sample_dir: str = "data/selected_recent_samples"
    selected_sample_zip: str = "data/selected_recent_samples.zip"

    model_config = ConfigDict(extra="forbid")

    @validator("inference_image_dir")
    def check_image_dir(cls, v):
        if not Path(v).is_dir():
            raise ValueError(f"inference_image_dir not found: {v}")
        return v

    @validator("selected_sample_dir")
    def ensure_sample_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @validator("selected_sample_zip")
    def ensure_zip_parent(cls, v):
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v


# ==========================
# Sample Selection Settings
# ==========================
class SampleSelectionConfig(BaseModel):
    strategy: Literal["agree_and_confident", "low_confidence"] = "agree_and_confident"
    confidence_threshold: float = 0.6
    min_selected_samples: int = 10
    merge_with_old: bool = True
    b1_api_url: str = "http://127.0.0.1:8011/batch_predict"
    b2_api_url: str = "http://127.0.0.1:8012/batch_predict"

    model_config = ConfigDict(extra="forbid")

    @validator("confidence_threshold")
    def check_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


# ==========================
# Output Settings
# ==========================
class OutputConfig(BaseModel):
    train_csv_path: str = "data/retrain/train.csv"
    val_csv_path: str = "data/retrain/val.csv"
    val_ratio: float = 0.2

    model_config = ConfigDict(extra="forbid")

    @validator("val_ratio")
    def check_val_ratio(cls, v):
        if not (0.0 < v < 1.0):
            raise ValueError("val_ratio must be between 0 and 1")
        return v


# ==========================
# Full Nested Config
# ==========================
class FullConfig(BaseModel):
    monitor: MonitorYieldConfig
    extraction: SampleExtractionConfig
    selection: SampleSelectionConfig
    output: OutputConfig

    model_config = ConfigDict(extra="forbid")
