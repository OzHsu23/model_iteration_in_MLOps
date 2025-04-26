import io
import os
import json
import yaml
import shutil
import zipfile
import pandas as pd
from PIL import Image
from fastapi import HTTPException

# ---------- Config Loading Functions ----------
def load_setting_json(path="setting.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(404, f"{path} not found")

# ----------- Flow Function -----------
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

def batch_predict_images(image_paths, wrapper, log_path="inference_logs.csv"):
    results = []
    for img_path in image_paths:
        with Image.open(img_path).convert("RGB") as image:
            label, confidence = wrapper.predict(image)
            results.append({
                "filename": os.path.basename(img_path),
                "pred": label,
                "confidence": confidence,
                "label": None
            })

    df = pd.DataFrame(results)
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)

    return results