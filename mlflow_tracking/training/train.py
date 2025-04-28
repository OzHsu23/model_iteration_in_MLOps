import argparse
import os

import sys
import psutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
from training.config import Config  
from training.trainers.classification_settings import ClassificationSettings
from training.trainers.classification_trainer import ClassificationTrainer
from training.trainers.detection_trainer import DetectionTrainer
from training.trainers.segmentation_trainer import SegmentationTrainer

from training.utils.common import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Training entry point")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    return parser.parse_args()

def get_trainer(config):
    # Set random seed for reproducibility
    random_seed = config.get_common_param("random_seed", 42) 
    set_seed(random_seed)
    
    # Init the trainer by task type
    task_type = config.task_type
    if task_type == "classification":
        settings = ClassificationSettings.from_config(config)
        return ClassificationTrainer(settings)
    elif task_type == "detection":
        return DetectionTrainer(config)
    elif task_type == "segmentation":
        return SegmentationTrainer(config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

def main():
    args = parse_args()

    # Load config
    config = Config(args.config)

    # Initialize the corresponding trainer
    trainer = get_trainer(config)

    # Start MLflow tracking
    with mlflow.start_run(run_name=config.experiment_name):
        mlflow.log_param("config_path", args.config)
        trainer.train()

if __name__ == "__main__":
    main()
