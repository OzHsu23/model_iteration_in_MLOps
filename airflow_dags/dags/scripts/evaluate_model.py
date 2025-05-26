# scripts/evaluate_model.py (temporary version)

import requests
import json

from schemas import FullConfig


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


def evaluate_model_and_decide(config: FullConfig):
    """
    Evaluate the old and new models, deploy the new model if it passes the evaluation,
    and decide whether to deploy based on the evaluation scores.
    """
    eval_cfg = config.evaluate_before_deploy
    deploy_cfg = config.deploy

    old_zip = "/tmp/old_model.zip"
    new_zip = "/tmp/new_model.zip"

    download_model(eval_cfg.old_model_api, old_zip, label="old")
    download_model(eval_cfg.new_model_api, new_zip, params={"job_id": deploy_cfg.job_id_to_deploy}, label="new")

    old_score = evaluate_model(eval_cfg.eval_inference_api, old_zip, eval_cfg.image_dir_or_zip, eval_cfg.metric, "old")

    deploy_model(eval_cfg.eval_inference_api, new_zip, deploy_cfg.job_id_to_deploy)

    new_score = evaluate_model(eval_cfg.eval_inference_api, new_zip, eval_cfg.image_dir_or_zip, eval_cfg.metric, "new")

    result = decide_and_report(
        old_score,
        new_score,
        eval_cfg.min_improvement,
        eval_cfg.result_flag_path
    )

    return result


def download_model(url, save_path, params=None, label="model"):
    print(f"[INFO] Downloading {label} model from {url} ...")
    r = requests.get(url, params=params)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"[INFO] Saved {label} model to {save_path}")


def evaluate_model(eval_api, model_zip, image_data, metric, label="model"):
    print(f"[INFO] Evaluating {label} model ...")
    score = call_evaluate_api(eval_api, model_zip, image_data, metric)
    print(f"[INFO] {label.capitalize()} model score: {score:.4f}")
    return score


def deploy_model(eval_api, model_zip_path, job_id):
    deploy_api = eval_api.replace("evaluate", "deploy_model_zip")
    print("[INFO] Deploying new model to inference server ...")

    with open(model_zip_path, "rb") as f:
        files = {"file": (f"{job_id}.zip", f, "application/zip")}
        data = {"job_id": job_id}
        resp = requests.post(deploy_api, files=files, data=data)

    if resp.status_code == 200:
        print(f"[SUCCESS] Model deployed successfully: {resp.json()}")
    else:
        print(f"[ERROR] Deployment failed: {resp.status_code} - {resp.text}")
        resp.raise_for_status()


def decide_and_report(old_score, new_score, threshold, result_path):
    delta = new_score - old_score
    passed = delta >= threshold

    print(f"[RESULT] Old: {old_score:.4f}, New: {new_score:.4f}, Δ = {delta:.4f}")
    print(f"[INFO] Evaluation {'PASSED' if passed else 'FAILED'} (need Δ≥{threshold})")

    result = {
        "deploy": passed,
        "old_score": old_score,
        "new_score": new_score,
        "delta": delta,
        "threshold": threshold
    }

    with open(result_path, "w") as fw:
        json.dump(result, fw)
    print(f"[INFO] Written evaluation result to {result_path}")
    return result




# def evaluate_model_and_decide(config: FullConfig):
#     eval_cfg = config.evaluate_before_deploy
#     deploy_cfg = config.deploy

#     image_data = eval_cfg.image_dir_or_zip
#     metric     = eval_cfg.metric
#     old_zip    = "/tmp/old_model.zip"
#     new_zip    = "/tmp/new_model.zip"

#     # --- Step 1: Download both models ---
#     print(f"[INFO] Downloading old model from {eval_cfg.old_model_api} ...")
#     r = requests.get(eval_cfg.old_model_api)
#     r.raise_for_status()
#     with open(old_zip, "wb") as f:
#         f.write(r.content)

#     print(f"[INFO] Downloading new model from {eval_cfg.new_model_api} ...")
#     r = requests.get(eval_cfg.new_model_api, params={"job_id": deploy_cfg.job_id_to_deploy})
#     r.raise_for_status()
#     with open(new_zip, "wb") as f: 
#         f.write(r.content)

#     # --- Step 2: Evaluate old model ---
#     print("[INFO] Evaluating old model ...")
#     old_score = call_evaluate_api(eval_cfg.eval_inference_api, old_zip, image_data, metric)

#     # --- Step 3: Deploy new model zip ---
#     print("[INFO] Deploying new model to inference server ...")
#     deploy_api = eval_cfg.eval_inference_api.replace("evaluate", "deploy_model_zip")
#     print("[INFO] Deploying new model to inference server ...")
#     with open(new_zip, "rb") as f:
#         files = {"file": (f"{deploy_cfg.job_id_to_deploy}.zip", f, "application/zip")}
#         data = {"job_id": deploy_cfg.job_id_to_deploy}
#         deploy_resp = requests.post(deploy_api, files=files, data=data)

#     if deploy_resp.status_code == 200:
#         print(f"[SUCCESS] Model deployed successfully: {deploy_resp.json()}")
#     else:
#         print(f"[ERROR] Deployment failed: {deploy_resp.status_code} - {deploy_resp.text}")
#         deploy_resp.raise_for_status()

#     # --- Step 4: Evaluate new model ---
#     print("[INFO] Evaluating new model ...")
#     new_score = call_evaluate_api(eval_cfg.eval_inference_api, new_zip, image_data, metric)

#     # --- Step 5: Decision logic ---
#     delta = new_score - old_score
#     print(f"[RESULT] Old: {old_score:.4f}, New: {new_score:.4f}, Δ = {delta:.4f}")
#     passed = delta >= eval_cfg.min_improvement
#     print(f"[INFO] Evaluation {'PASSED' if passed else 'FAILED'} (need Δ≥{eval_cfg.min_improvement})")

#     # --- Step 6: Save result ---
#     result_dict = {
#         "deploy": passed,
#         "old_score": old_score,
#         "new_score": new_score,
#         "delta": delta,
#         "threshold": eval_cfg.min_improvement
#     }
#     with open(eval_cfg.result_flag_path, "w") as fw:
#         json.dump(result_dict, fw)
#     print(f"[INFO] Written evaluation result to {eval_cfg.result_flag_path}")




