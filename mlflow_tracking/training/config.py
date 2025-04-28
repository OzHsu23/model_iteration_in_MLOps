# training/config.py
import json
import os

import json

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        self.config_data = config_data

        # Task type (classification, segmentation, etc.)
        self.task_type = config_data.get('task_type')
        if self.task_type is None:
            raise ValueError("Task type must be specified in the config file.")

        # Experiment name
        self.experiment_name = config_data.get('experiment_name', f"{self.task_type}_default_exp")

        # Sections
        self.common = config_data.get('common', {})
        self.task_specific = config_data.get(self.task_type, {})
        self.mlflow = config_data.get('mlflow', {})

    def get(self, key, default=None):
        return self.config_data.get(key, default)

    def get_common_param(self, key, default=None):
        return self.common.get(key, default)

    def get_task_data(self):
        return self.task_specific.get('data', {})

    def get_task_model(self):
        return self.task_specific.get('model', {})

    def get_task_training(self):
        return self.task_specific.get('training', {})

    def get_task_data_param(self, key, default=None):
        return self.get_task_data().get(key, default)

    def get_task_model_param(self, key, default=None):
        return self.get_task_model().get(key, default)

    def get_task_training_param(self, key, default=None):
        return self.get_task_training().get(key, default)

    def get_mlflow_param(self, key, default=None):
        return self.mlflow.get(key, default)

    def info(self):
        return json.dumps(self.config_data, indent=4)

