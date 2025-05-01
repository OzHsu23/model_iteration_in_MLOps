import os
import shutil
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from schemas import FullConfig


def extract_samples_and_prepare_training_data(config: FullConfig, debug: bool = False):
    log_df = pd.read_csv(config.monitor.log_path).dropna(subset=["filename", "pred", "confidence"])
    recent_df = log_df.tail(config.monitor.recent_window)

    if recent_df.empty:
        raise ValueError("[ERROR] No recent valid inference records found.")

    # Clean sample dir
    if os.path.exists(config.extraction.selected_sample_dir):
        shutil.rmtree(config.extraction.selected_sample_dir)
    os.makedirs(config.extraction.selected_sample_dir, exist_ok=True)

    # Copy images
    for _, row in recent_df.iterrows():
        src = os.path.join(config.extraction.inference_image_dir, row["filename"])
        dst = os.path.join(config.extraction.selected_sample_dir, os.path.basename(row["filename"]))
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # Zip
    with zipfile.ZipFile(config.extraction.selected_sample_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(config.extraction.selected_sample_dir):
            fpath = os.path.join(config.extraction.selected_sample_dir, file)
            zipf.write(fpath, arcname=file)

    # Call B1/B2 APIs
    def call_batch_predict(api_url):
        with open(config.extraction.selected_sample_zip, "rb") as f:
            files = {"zip_file": (os.path.basename(f.name), f, "application/zip")}
            r = requests.post(api_url, files=files)
        if r.status_code != 200:
            raise RuntimeError(f"Error calling {api_url}: {r.text}")
        return {res["filename"]: res["prediction"] for res in r.json()["results"]}

    b1 = call_batch_predict(config.selection.b1_api_url)
    b2 = call_batch_predict(config.selection.b2_api_url)

    # Select consistent & confident
    selected = []
    for _, row in recent_df.iterrows():
        fname = row["filename"]
        b1_result = b1.get(fname)
        b2_result = b2.get(fname)
        if not b1_result or not b2_result:
            continue
        if (
            b1_result["label"] == b2_result["label"] and
            b1_result["confidence"] >= config.selection.confidence_threshold and
            b2_result["confidence"] >= config.selection.confidence_threshold
        ):
            selected.append({"filename": fname, "label": b1_result["label"]})

    if not selected:
        raise ValueError("[WARN] No confident-agreeing samples found.")

    df = pd.DataFrame(selected)
    train_df, val_df = train_test_split(
        df, test_size=config.output.val_ratio, random_state=42, stratify=df["label"]
    )

    os.makedirs(os.path.dirname(config.output.train_csv_path), exist_ok=True)
    train_df.to_csv(config.output.train_csv_path, index=False)
    val_df.to_csv(config.output.val_csv_path, index=False)

    print(f"[INFO] Saved {len(train_df)} training samples to {config.output.train_csv_path}")
    print(f"[INFO] Saved {len(val_df)} validation samples to {config.output.val_csv_path}")
