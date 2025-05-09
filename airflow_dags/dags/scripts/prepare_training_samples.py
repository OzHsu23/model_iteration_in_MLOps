import os
import shutil
import zipfile
import json
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from schemas import FullConfig

def call_batch_predict(api_url, zip_path):
    with open(zip_path, "rb") as f:
        files = {"zip_file": (os.path.basename(f.name), f, "application/zip")}
        r = requests.post(api_url, files=files)
    if r.status_code != 200:
        raise RuntimeError(f"Error calling {api_url}: {r.text}")
    return {res["filename"]: res["prediction"] for res in r.json()["results"]}


def select_confident_agree_samples(log_df, b1_result, b2_result, threshold):
    selected = []
    for _, row in log_df.iterrows():
        fname = row["filename"]
        b1_pred = b1_result.get(fname)
        b2_pred = b2_result.get(fname)
        if not b1_pred or not b2_pred:
            continue
        if b1_pred["label"] == b2_pred["label"] and b1_pred["confidence"] >= threshold and b2_pred["confidence"] >= threshold:
            selected.append({"filename": fname, "label": b1_pred["label"]})
    return pd.DataFrame(selected)


def extract_samples_and_prepare_training_data(config: FullConfig, debug: bool = False):
    prepare_cfg = config.prepare_training_data

    # Load recent inference log entries
    log_df = pd.read_csv(config.monitor.log_path).dropna(subset=["filename", "pred", "confidence"])
    recent_df = log_df.tail(config.monitor.recent_window)
    if recent_df.empty:
        raise ValueError("[ERROR] No recent valid inference records found.")

    # Step 1: Clear and create sample directory
    sample_dir = prepare_cfg.selected_sample_dir
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    image_dir = os.path.join(sample_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Step 2: Zip recent inference images to prepare for B1/B2 prediction
    temp_zip = os.path.join(sample_dir, "selected_samples.zip")
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for _, row in recent_df.iterrows():
            src = os.path.join(prepare_cfg.inference_image_dir, row["filename"])
            if os.path.exists(src):
                zipf.write(src, arcname=row["filename"])

    # Step 3: Call external prediction APIs (B1 and B2)
    b1 = call_batch_predict(prepare_cfg.b1_inference_api, temp_zip)
    b2 = call_batch_predict(prepare_cfg.b2_inference_api, temp_zip)

    # Step 4: Select samples where B1 and B2 agree and both are confident
    selected_df = select_confident_agree_samples(recent_df, b1, b2, prepare_cfg.confidence_threshold)
    if selected_df.empty:
        raise ValueError("[WARN] No confident-agreeing samples found.")

    # Step 5: Copy selected images into the sample directory
    for _, row in selected_df.iterrows():
        src = os.path.join(prepare_cfg.inference_image_dir, row["filename"])
        dst = os.path.join(image_dir, row["filename"])
        shutil.copyfile(src, dst)

    # Step 6: Split into train/val sets and save as CSV
    train_df, val_df = train_test_split(
        selected_df, test_size=prepare_cfg.val_ratio, stratify=selected_df["label"], random_state=42
    )
    train_df.to_csv(os.path.join(image_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(image_dir, "val.csv"), index=False)

    # Step 7: Generate setting.json based on training template
    with open(prepare_cfg.training_template_path, "r") as f:
        setting_dict = json.load(f)
    setting_dict["common"]["data_dir"] = "images"
    setting_path = os.path.join(sample_dir, "setting.json")
    with open(setting_path, "w") as f:
        json.dump(setting_dict, f, indent=2)

    # Step 8: Merge with previous training set if configured
    if prepare_cfg.merge_with_previous:
        prev = prepare_cfg.merge_with_previous
        for csv_name in ["train.csv", "val.csv"]:
            old_csv = os.path.join(prev, "images", csv_name)
            new_csv = os.path.join(image_dir, csv_name)
            if os.path.exists(old_csv):
                df_old = pd.read_csv(old_csv)
                df_new = pd.read_csv(new_csv)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.to_csv(new_csv, index=False)

        prev_img_dir = os.path.join(prev, "images")
        if os.path.exists(prev_img_dir):
            for f in os.listdir(prev_img_dir):
                shutil.copy(os.path.join(prev_img_dir, f), os.path.join(image_dir, f))

    # Step 9: Zip the full training folder (including setting.json and images/)
    zip_path = config.retrain.retrain_zip_path
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(sample_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, sample_dir)
                zipf.write(full_path, arcname=arcname)

    print(f"[INFO] Training package saved to {zip_path}")
