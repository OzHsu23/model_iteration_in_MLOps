# app/app_utils.py

import io
import os
import json
import yaml
import shutil
import zipfile
import pandas as pd
from uuid import uuid4
from datetime import datetime
from PIL import Image
from fastapi import HTTPException

from app.schemas import LogRecord

# ---------- Config Loading Functions ----------
def load_setting_json(path="setting.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(404, f"{path} not found")

# ---------- Image Extract Functions ----------
def extract_images_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Provided folder_path is not a valid directory.")
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            images.append(os.path.join(folder_path, filename))
    return images

def extract_images_from_zip(contents, temp_dir="temp_images"):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
        zip_ref.extractall(temp_dir)

    images = []
    for root, _, files in os.walk(temp_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                images.append(os.path.join(root, filename))
    return images

# ---------- Log Saving Utility ----------
def save_log_records(log_records: list, log_path: str):
    """
    Save a list of log records (dicts) to a CSV file.
    """
    df = pd.DataFrame(log_records)
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)

def create_log_record(model, filename: str, prediction_result: dict) -> LogRecord:
    """
    Create a LogRecord instance from model and prediction results.
    """
    return LogRecord(
        inference_id=str(uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        model_version=getattr(model, "model_version", "unknown"),
        framework=getattr(model, "framework", "unknown"),
        model_format=getattr(model, "model_format", "unknown"),
        filename=filename,
        pred=prediction_result.get("label"),
        confidence=prediction_result.get("confidence"),
        label=None
    )
