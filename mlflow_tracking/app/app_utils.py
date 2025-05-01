# app/app_utils.py


import os
import json
import shutil
import zipfile
from datetime import datetime
from training.config import Config
from training.trainers.classification_settings import ClassificationSettings
from training.trainers.classification_trainer import ClassificationTrainer
from training.trainers.detection_trainer import DetectionTrainer
from training.trainers.segmentation_trainer import SegmentationTrainer
from training.utils.common import set_seed

import mlflow

# ========== Job Info ==========
job_status = {}
job_metrics = {}
job_model_paths = {}
job_registry_path = "job_cache.json"

# ========== Persistence ==========
def save_job_status():
    with open(job_registry_path, "w") as f:
        json.dump({
            "job_status": job_status,
            "job_metrics": job_metrics,
            "job_model_paths": job_model_paths
        }, f, indent=2)

def load_job_status():
    if os.path.exists(job_registry_path):
        with open(job_registry_path, "r") as f:
            data = json.load(f)
            job_status.update(data.get("job_status", {}))
            job_metrics.update(data.get("job_metrics", {}))
            job_model_paths.update(data.get("job_model_paths", {}))

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

# ========== Retrain Logic ==========
def run_config_retrain(setting_dict, job_id):
    try:
        job_status[job_id] = "running"
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

def run_zip_retrain(zip_path, job_id):
    try:
        job_status[job_id] = "running"
        work_dir = f"/tmp/job_{job_id}"
        os.makedirs(work_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(work_dir)

        config_path = os.path.join(work_dir, "setting.json")
        config = Config(config_path)
        config.common['data_dir'] = os.path.join(work_dir, 'images')
        
        trainer = get_trainer(config)

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

# ========== Model Package ==========
def save_model_package(trainer, job_id, extra_files=[]):
    model_dir = f"/saved_models/{job_id}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(model_dir, "model.pth")
    trainer.save_model(model_path)

    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer.get_metrics(), f, indent=2)

    # Save extras
    for path in extra_files:
        if os.path.exists(path):
            shutil.copy(path, model_dir)

    # Zip the model dir
    zip_path = f"{model_dir}.zip"
    shutil.make_archive(model_dir, 'zip', model_dir)
    return zip_path
