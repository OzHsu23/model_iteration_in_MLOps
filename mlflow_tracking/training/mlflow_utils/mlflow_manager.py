
# training/mlflow_utils/mlflow_manager.py
import mlflow
import mlflow.pytorch
import torch
import platform
import subprocess
from mlflow.models import infer_signature

import cv2
import numpy as np
import torchvision

class MLflowManager:
    @staticmethod
    def safe_start_run(experiment_name=None, run_name=None, nested=False):
        """Safely start a new MLflow run, ending any existing one."""
        if mlflow.active_run() is not None:
            print('Oz Ending existing run...')
            mlflow.end_run()
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, nested=nested)

    @staticmethod
    def safe_end_run():
        """Safely end the current MLflow run if active."""
        if mlflow.active_run() is not None:
            mlflow.end_run()

    @staticmethod
    def log_environment():
        """Log environment information such as Python, PyTorch version, and Git commit."""
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("cuda_device", torch.cuda.get_device_name(0))
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
            mlflow.log_param("git_commit", sha)
        except Exception:
            mlflow.log_param("git_commit", "Not Available")

    @staticmethod
    def log_config(config_dict):
        """Log all config parameters from a dictionary."""
        for key, value in config_dict.items():
            mlflow.log_param(key, value)

    @staticmethod
    def log_params_from_dict(params_dict):
        """Log multiple parameters from a dictionary."""
        for key, value in params_dict.items():
            mlflow.log_param(key, value)

    @staticmethod
    def log_metrics_from_dict(metrics_dict, step=None):
        """Log multiple metrics from a dictionary."""
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value, step=step)

    @staticmethod
    def log_model(model, artifact_path="model", input_example=None):
        """Log a PyTorch model artifact to MLflow with optional input example and environment info."""
        
        signature = None
        if input_example is not None:
            # If input example is provided, infer signature
            input_tensor = torch.from_numpy(input_example).float().to(model.device)
            
            with torch.no_grad():
                model.eval()
                output = model(input_tensor)
            
            signature = infer_signature(input_tensor.cpu().numpy(), output.detach().cpu().numpy())

        pip_requirements=[
            f"torch=={torch.__version__}",
            f"torchvision=={torchvision.__version__}",
            f"mlflow=={mlflow.__version__}",
            f"opencv-python={cv2.__version__}",
            f"numpy=={np.__version__}"
        ]
        
        
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            input_example=input_example,
            signature=signature,
            pip_requirements=pip_requirements,
            conda_env=None,
        )
