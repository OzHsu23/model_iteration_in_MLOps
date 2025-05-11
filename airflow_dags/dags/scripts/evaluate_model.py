# scripts/evaluate_model.py (temporary version)

import os
import requests
import json
from contextlib import nullcontext

from schemas import FullConfig


def evaluate_model_and_decide(config: FullConfig):
    eval_cfg = config.evaluate_before_deploy
    deploy_cfg = config.deploy

    image_data = eval_cfg.image_dir_or_zip
    metric     = eval_cfg.metric
    old_zip    = "/tmp/old_model.zip"
    new_zip    = "/tmp/new_model.zip"

    # --- Step 1: Download both models ---
    print(f"[INFO] Downloading old model from {eval_cfg.old_model_api} ...")
    r = requests.get(eval_cfg.old_model_api); r.raise_for_status()
    with open(old_zip, "wb") as f: f.write(r.content)

    print(f"[INFO] Downloading new model from {eval_cfg.new_model_api} ...")
    r = requests.get(eval_cfg.new_model_api, params={"job_id": deploy_cfg.job_id_to_deploy})
    r.raise_for_status()
    with open(new_zip, "wb") as f: f.write(r.content)

    # --- Step 2: Evaluate old model ---
    print("[INFO] Evaluating old model ...")
    old_score = call_evaluate_api(eval_cfg.eval_inference_api, old_zip, image_data, metric)

    # --- Step 3: Deploy new model zip ---
    print("[INFO] Deploying new model to inference server ...")
    deploy_api = eval_cfg.eval_inference_api.replace("evaluate", "deploy_model_zip")
    print("[INFO] Deploying new model to inference server ...")
    with open(new_zip, "rb") as f:
        files = {"file": (f"{deploy_cfg.job_id_to_deploy}.zip", f, "application/zip")}
        data = {"job_id": deploy_cfg.job_id_to_deploy}
        deploy_resp = requests.post(deploy_api, files=files, data=data)

    if deploy_resp.status_code == 200:
        print(f"[SUCCESS] Model deployed successfully: {deploy_resp.json()}")
    else:
        print(f"[ERROR] Deployment failed: {deploy_resp.status_code} - {deploy_resp.text}")
        deploy_resp.raise_for_status()

    # --- Step 4: Evaluate new model ---
    print("[INFO] Evaluating new model ...")
    new_score = call_evaluate_api(eval_cfg.eval_inference_api, new_zip, image_data, metric)

    # --- Step 5: Decision logic ---
    delta = new_score - old_score
    print(f"[RESULT] Old: {old_score:.4f}, New: {new_score:.4f}, Δ = {delta:.4f}")
    passed = delta >= eval_cfg.min_improvement
    print(f"[INFO] Evaluation {'PASSED' if passed else 'FAILED'} (need Δ≥{eval_cfg.min_improvement})")

    # --- Step 6: Save result ---
    result_dict = {
        "deploy": passed,
        "old_score": old_score,
        "new_score": new_score,
        "delta": delta,
        "threshold": eval_cfg.min_improvement
    }
    with open(eval_cfg.result_flag_path, "w") as fw:
        json.dump(result_dict, fw)
    print(f"[INFO] Written evaluation result to {eval_cfg.result_flag_path}")



def call_evaluate_api(api_url: str, model_path: str, image_path: str, metric: str) -> float:
    """
    Upload model_zip + image_zip and return the evaluated score.
    """
    with open(model_path, "rb") as m, open(image_path, "rb") as i:
        resp = requests.post(
            api_url,
            data={"metric": metric},
            files={"model_zip": m, "image_zip": i}
        )
    resp.raise_for_status()
    return resp.json().get("score", 0.0)
