# dags/scripts/deploy_new_model.py

import requests
import os
from schemas import FullConfig

def deploy_model(config_path: str):
    """
    Deploy the retrained model zip to inference server using job_id.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = FullConfig.parse_file(config_path)
    deploy_cfg = config.deploy
    retrain_cfg = config.retrain

    job_id = deploy_cfg.job_id_to_deploy
    if not job_id:
        raise ValueError("Missing job_id_to_deploy in config")

    # Step 1: Download model zip from retrain server
    download_url = f"{retrain_cfg.retrain_server_api.rstrip('/')}/download_model?job_id={job_id}"
    zip_path = f"/tmp/{job_id}.zip"
    print(f"[INFO] Downloading model ZIP from {download_url}")
    resp = requests.get(download_url)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(resp.content)

    # Step 2: Upload model zip to inference server
    inference_server_api = deploy_cfg.inference_server_api
    with open(zip_path, "rb") as f:
        files = {"file": (f"{job_id}.zip", f, "application/zip")}
        data = {"job_id": job_id}
        deploy_resp = requests.post(inference_server_api, files=files, data=data)

    if deploy_resp.status_code == 200:
        print(f"[SUCCESS] Model deployed successfully: {deploy_resp.json()}")
    else:
        print(f"[ERROR] Deployment failed: {deploy_resp.status_code} - {deploy_resp.text}")
        deploy_resp.raise_for_status()
