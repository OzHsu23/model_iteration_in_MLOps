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
        model_uri = f"runs:/{model_cfg.run_id}/model"
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
    model_type = model_cfg.model_type

    if model_type == "torch_mlflow":
        try:
            return load_torch_mlflow_model(model_cfg)
        except Exception as e:
            print(f"[Warning] MLflow load failed: {e}, fallback to local.")
            return load_torch_pure_model(model_cfg)

    elif model_type == "torch_pure":
        return load_torch_pure_model(model_cfg)

    elif model_type == "tf_mlflow":
        try:
            return load_tf_mlflow_model(model_cfg)
        except Exception as e:
            print(f"[Warning] TF MLflow load failed: {e}, fallback to local.")
            return load_tf_pure_model(model_cfg)

    elif model_type == "tf_pure":
        return load_tf_pure_model(model_cfg)

    else:
        raise HTTPException(400, f"Unsupported model_type: {model_type}")
