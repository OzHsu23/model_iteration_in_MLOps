# app/app_utils.py

import io
import os
import json
import yaml
import shutil
import zipfile
import tempfile
import pandas as pd
from PIL import Image
from uuid import uuid4
from datetime import datetime
from typing import Optional, Tuple, List
from fastapi import HTTPException, Request, UploadFile

from app.schemas import LogRecord
from factory.model_factory import ModelFactory
from app.schemas import AppSettings
from app.globals import SETTING_PATH, PACKAGE_DIR


# ---------- Config Loading Functions ----------
def load_setting_json(path):
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

def extract_images_from_zip(contents):
    """
    Extract image files from an uploaded zip file to a temporary directory.
    Returns a list of image paths and the temporary directory path.
    The directory will be automatically deleted when the caller exits the context.
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="batch_zip_")
    with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
        zip_ref.extractall(temp_dir.name)

    images = []
    for root, _, files in os.walk(temp_dir.name):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                images.append(os.path.join(root, filename))

    return images, temp_dir  # Caller must keep temp_dir open to prevent cleanup

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

# ---------- Prediction Pipeline ----------
def resolve_image_inputs(
    folder_path: Optional[str],
    zip_file: Optional[UploadFile]
) -> Tuple[List[str], Optional[tempfile.TemporaryDirectory]]:
    """
    Handle either folder path or uploaded zip file.
    Returns a tuple of image paths and optional TemporaryDirectory (must be cleaned up by caller).
    """
    if folder_path:
        try:
            images = extract_images_from_folder(folder_path)
            return images, None
        except ValueError as e:
            raise HTTPException(400, str(e))

    if zip_file:
        if not zip_file.filename.endswith(".zip"):
            raise HTTPException(400, "Uploaded file must be a .zip archive.")
        contents = zip_file.file.read()
        return extract_images_from_zip(contents)

    raise HTTPException(400, "Either folder_path or zip_file must be provided.")

def predict_images(image_paths: list, model, log_path: str) -> list:
    """
    Predict a list of image files using the given model and record logs.
    Returns the prediction results.
    """
    results = []
    log_records = []

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            file_bytes = f.read()
        prediction_result = model.predict(file_bytes)

        results.append({
            "filename": os.path.basename(img_path),
            "prediction": prediction_result
        })

        log_record = create_log_record(model, os.path.basename(img_path), prediction_result)
        log_records.append(log_record.model_dump())

    save_log_records(log_records, log_path)
    return results

# ---------- Export the current deployed model as zip ----------
def export_current_model_zip() -> str:
    settings_dict = load_setting_json(SETTING_PATH)
    settings = AppSettings(**settings_dict)

    model_path = settings.model.local_path
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path not found: {model_path}")

    model_dir = os.path.join(PACKAGE_DIR, "model")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    shutil.copy(model_path, os.path.join(model_dir, os.path.basename(model_path)))

    with open(os.path.join(model_dir, "inference_setting.json"), "w") as f:
        json.dump(settings_dict, f, indent=2)

    zip_base = model_dir
    shutil.make_archive(zip_base, 'zip', model_dir)
    zip_path = zip_base + ".zip"

    shutil.rmtree(model_dir)
    return zip_path

# ---------- Evaluate a given model on a test set ----------
def evaluate_model_zip(model_zip_path: str, image_path: str, metric: str) -> float:
    # Unzip model package
    model_dir = os.path.join(PACKAGE_DIR, "model")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Load model settings
    setting_path = os.path.join(model_dir, "inference_setting.json")
    if not os.path.exists(setting_path):
        raise RuntimeError("Missing inference_setting.json in model package")

    settings_dict = load_setting_json(setting_path)
    settings = AppSettings(**settings_dict)
    model_wrapper = ModelFactory.create_model(settings)

    # Handle image input (zip or folder)
    if zipfile.is_zipfile(image_path):
        extract_dir = tempfile.mkdtemp(prefix="eval_images_")
        with zipfile.ZipFile(image_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        image_path = extract_dir

    # Load test.csv
    csv_path = os.path.join(image_path, "test.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Missing test.csv under {image_path}")

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise RuntimeError("test.csv must contain 'filename' and 'label' columns")

    if metric == "accuracy":
        correct = 0
        total = 0
        for _, row in df.iterrows():
            img_file = os.path.join(image_path, row["filename"])
            if not os.path.exists(img_file):
                continue
            with open(img_file, "rb") as f:
                result = model_wrapper.predict(f.read())
            print('filename:', row["filename"], 'result:', result, 'label:', row["label"])
            if result.get("label") == str(row["label"]):
                correct += 1
            total += 1
        print('correct:', correct, 'total:', total)
        return correct / total if total > 0 else 0.0

    else:
        raise RuntimeError(f"Unsupported metric: {metric}, only 'accuracy' is supported.")

# ---------- Read deploy status ----------
def get_current_deploy_status(request: Request) -> dict:
    model = request.app.state.model
    meta = getattr(request.app.state, "model_meta", {})
    return {
        "model_version": getattr(model, "model_version", "unknown"),
        "framework": getattr(model, "framework", "unknown"),
        "format": getattr(model, "model_format", "unknown"),
        "loaded_from": meta.get("loaded_from", "n/a"),
        "deployment_time": meta.get("deployment_time", "n/a"),
    }
