# models/loader.py 

import os
import mlflow
import torch
from fastapi import HTTPException
from app.schemas import ModelSettings
from models.build import build_model

# Try import TensorFlow
try:
    import tensorflow as tf
except ImportError:
    tf = None

# ==========================
#  Utility Functions
# ==========================

def check_mlflow_run_exists(run_id: str, tracking_uri: str) -> bool:
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        return run is not None
    except Exception:
        return False

# ==========================
#  Load functions for different model types
# ==========================

def load_torch_mlflow_model(model_cfg: ModelSettings):
    """
    Build model structure, then only load weights from MLflow artifacts.
    """
    model = build_model(
        model_name=model_cfg.model_name,
        num_classes=model_cfg.num_classes
    )
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=model_cfg.run_id, artifact_path="model", tracking_uri=model_cfg.tracking_uri)
        # Assume weights are stored as "state_dict.pth" or similar inside the artifact
        weight_file = os.path.join(local_path, "state_dict.pth")
        state_dict = torch.load(weight_file, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load weights from MLflow artifacts: {e}")

def load_torch_pure_model(model_cfg: ModelSettings):
    if not model_cfg.local_path or not os.path.exists(model_cfg.local_path):
        raise RuntimeError(f"Local model path invalid: {model_cfg.local_path}")

    model = build_model(
        model_name=model_cfg.model_name,
        num_classes=model_cfg.num_classes
    )
    state_dict = torch.load(model_cfg.local_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_tf_mlflow_model(model_cfg: ModelSettings):
    if tf is None:
        raise RuntimeError("TensorFlow not available in environment.")
    # Not implemented yet, will be supported in the future if needed
    raise NotImplementedError("TF MLflow loading not yet supported.")

def load_tf_pure_model(model_cfg: ModelSettings):
    if tf is None:
        raise RuntimeError("TensorFlow not available in environment.")
    # Not implemented yet, will be supported in the future if needed
    raise NotImplementedError("TF pure model loading not yet supported.")

# ==========================
#  Dispatch loader
# ==========================

def load_model_by_type(model_cfg: ModelSettings):
    """Load model based on the specified model_type in ModelSettings."""
    
    model_type = model_cfg.model_type

    loaders = {
        "torch_mlflow": (load_torch_mlflow_model, load_torch_pure_model),
        "torch_pure": (load_torch_pure_model, None),
        "tf_mlflow": (load_tf_mlflow_model, load_tf_pure_model),
        "tf_pure": (load_tf_pure_model, None),
    }

    if model_type not in loaders:
        raise HTTPException(400, f"Unsupported model_type: {model_type}")

    primary_loader, fallback_loader = loaders[model_type]

    try:
        return primary_loader(model_cfg)
    except Exception as e:
        if fallback_loader:
            print(f"[Warning] {model_type} load failed: {e}, fallback to local.")
            return fallback_loader(model_cfg)
        else:
            raise
