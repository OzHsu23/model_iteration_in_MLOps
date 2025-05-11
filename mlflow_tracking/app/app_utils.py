# app/app_utils.py


import os
import json
import shutil
import zipfile
from datetime import datetime

import mlflow

from training.config import Config
from training.trainers.classification_settings import ClassificationSettings
from training.trainers.classification_trainer import ClassificationTrainer
from training.trainers.detection_trainer import DetectionTrainer
from training.trainers.segmentation_trainer import SegmentationTrainer
from training.utils.common import set_seed


from app.globals import (
    job_status,
    job_metrics,
    job_model_paths,
    job_progress,
    job_registry_path,
    save_job_status,
    PACKAGE_DIR
)

# ========== Load persisted job state ==========
def load_job_status():
    job_registry_path = "job_cache.json"
    if os.path.exists(job_registry_path):
        with open(job_registry_path, "r") as f:
            data = json.load(f)
            job_status.update(data.get("job_status", {}))
            job_metrics.update(data.get("job_metrics", {}))
            job_model_paths.update(data.get("job_model_paths", {}))
            job_progress.update(data.get("job_progress", {}))
    else:
        save_job_status()

# ========== Trainer Factory ==========
def get_trainer(config: Config):
    set_seed(config.get_common_param("random_seed", 42))
    if config.task_type == "classification":
        settings = ClassificationSettings.from_config(config)
        return ClassificationTrainer(settings)
    elif config.task_type == "detection":
        return DetectionTrainer(config)
    elif config.task_type == "segmentation":
        return SegmentationTrainer(config)
    else:
        raise ValueError(f"Unsupported task type: {config.task_type}")

# ========== Retrain from config JSON ==========
def run_config_retrain(setting_dict, job_id):
    try:
        job_status[job_id] = "running"
        save_job_status()

        # Inject job_id into config
        setting_dict["job_id"] = job_id
        config = Config(setting_dict)
        trainer = get_trainer(config)

        with mlflow.start_run(run_name=config.experiment_name):
            mlflow.log_param("job_id", job_id)
            trainer.train()
            job_metrics[job_id] = trainer.get_metrics()
            zip_path = save_model_package(trainer, job_id)
            job_model_paths[job_id] = zip_path
            job_status[job_id] = "completed"
    except Exception as e:
        job_status[job_id] = "failed"
        job_metrics[job_id] = {"error": str(e)}
    finally:
        save_job_status()

# ========== Retrain from uploaded ZIP ==========
def run_zip_retrain(zip_path, job_id):
    try:
        job_status[job_id] = "running"
        save_job_status()

        # Unpack ZIP contents
        work_dir = f"{PACKAGE_DIR}/job_{job_id}"
        os.makedirs(work_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(work_dir)

        # Load setting.json as dict and inject runtime info
        config_path = os.path.join(work_dir, "setting.json")
        with open(config_path, "r") as f:
            setting_dict = json.load(f)

        setting_dict["job_id"] = job_id
        setting_dict["common"]["data_dir"] = os.path.join(work_dir, 'images')
        
        print("[DEBUG] setting_dict:", setting_dict)

        # Create Config object and trainer
        config = Config(setting_dict)
        
        
        
        trainer = get_trainer(config)

        # Run training
        with mlflow.start_run(run_name=config.experiment_name):
            mlflow.log_param("job_id", job_id)
            trainer.train()
            job_metrics[job_id] = trainer.get_metrics()
            zip_path = save_model_package(trainer, job_id, extra_files=[config_path])
            job_model_paths[job_id] = zip_path
            job_status[job_id] = "completed"
    except Exception as e:
        job_status[job_id] = "failed"
        job_metrics[job_id] = {"error": str(e)}
    finally:
        save_job_status()

# ========== Save model artifacts ==========
def save_model_package(trainer, job_id, extra_files=[]):
    model_dir = f"{PACKAGE_DIR}/{job_id}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.pth")
    trainer.save_model(model_path)

    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.get_metrics(), f, indent=2)

    # Save inference setting
    inference_setting = convert_training_to_inference_setting(trainer.settings, job_id)
    with open(os.path.join(model_dir, "inference_setting.json"), "w") as f:
        json.dump(inference_setting, f, indent=2)

    # Zip everything
    zip_path = shutil.make_archive(model_dir, "zip", model_dir)
    return zip_path

def convert_training_to_inference_setting(settings, job_id: str):
    """
    Convert ClassificationSettings to inference setting JSON-compatible dict.
    """
    inference_setting = {
        "task_type": "classification",
        "model": {
            "model_type": "torch_mlflow",
            "run_id": "",  
            "tracking_uri": settings.tracking_uri or "",
            "local_path": f"./weights/{settings.model_name}/model_state_dict.pth",
            "yaml_path": "",
            "model_name": settings.model_name,
            "num_classes": settings.num_classes,
            "class_names": None, 
            "img_size": settings.img_size
        },
        "preprocessing": {
            "resize": settings.img_size
        },
        "postprocessing": {
            "type": "default"
        },
        "server": {
            "port": 8000
        },
        "logging": {
            "log_path": f"logs/inference_logs_{job_id}.csv",
            "enable": True
        }
    }
    return inference_setting