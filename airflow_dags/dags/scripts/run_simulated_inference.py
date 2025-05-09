# scripts/run_simulated_inference.py

import os
import time
import requests
from schemas import FullConfig


def run_simulated_inference(config_path: str):
    """
    Simulate production line inference using settings from FullConfig.
    This module is triggered by the Airflow DAG.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Use FullConfig to load and validate config
    config = FullConfig.parse_file(config_path)
    production_cfg = config.production_line

    inference_url = production_cfg.inference_server_api
    image_dir = production_cfg.inference_image_dir
    repeat_times = production_cfg.repeat

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"[INFO] Running simulated inference for {repeat_times} rounds, {len(image_files)} images each.")

    for r in range(repeat_times):
        print(f"[INFO] Round {r + 1}/{repeat_times}")
        for fname in image_files:
            full_path = os.path.join(image_dir, fname)
            with open(full_path, "rb") as f:
                files = {"file": (fname, f, "image/jpeg")}
                try:
                    resp = requests.post(inference_url, files=files)
                    if resp.status_code == 200:
                        print(f"[OK] {fname} -> {resp.json()}")
                    else:
                        print(f"[ERROR] {fname} failed: {resp.status_code} - {resp.text}")
                except Exception as e:
                    print(f"[EXCEPTION] {fname}: {e}")
            time.sleep(0.05)  # Optional pacing between requests

    print("[INFO] Simulated inference complete.")
