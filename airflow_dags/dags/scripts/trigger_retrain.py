# dags/scripts/trigger_retrain.py

import time
import requests
import json
from schemas import FullConfig

def start_retrain_and_wait(config_path: str) -> str:
    """
    Start retraining by uploading ZIP, monitor job until completion,
    and record job_id to config for later deployment.
    """
    # Load full config from file
    config = FullConfig.parse_file(config_path)
    retrain_cfg = config.retrain

    zip_path = retrain_cfg.retrain_zip_path
    base_url = retrain_cfg.retrain_server_api.rstrip("/")
    retrain_url = f"{base_url}/start_retrain"

    # === Step 1: Upload ZIP and start retraining ===
    with open(zip_path, "rb") as f:
        files = {"zip_file": (zip_path, f, "application/zip")}
        response = requests.post(retrain_url, files=files)
    response.raise_for_status()

    job_id = response.json()["job_id"]
    print(f"[INFO] Started retrain job: {job_id}")

    # === Step 2: Poll retrain progress until done ===
    timeout = retrain_cfg.max_wait_sec
    interval = retrain_cfg.poll_interval_sec
    elapsed = 0

    while elapsed < timeout:
        try:
            progress_url = f"{base_url}/retrain_progress?job_id={job_id}"
            progress_resp = requests.get(progress_url)
            if progress_resp.status_code == 200:
                print(f"[PROGRESS] {progress_resp.json()}")
        except Exception as e:
            print(f"[WARN] Progress query failed: {e}")

        # Check retrain status
        status_url = f"{base_url}/retrain_status?job_id={job_id}"
        status_resp = requests.get(status_url).json()
        if status_resp["status"] == "completed":
            print(f"[INFO] Retrain job completed: {job_id}")
            update_config_job_id(config_path, job_id)
            return job_id
        elif status_resp["status"] == "failed":
            raise Exception(f"[ERROR] Retrain failed: {status_resp.get('error')}")

        time.sleep(interval)
        elapsed += interval

    raise TimeoutError("Retrain did not complete within the allowed time.")


def update_config_job_id(config_path: str, job_id: str):
    """
    Update deploy.job_id_to_deploy in config JSON file.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg.setdefault("deploy", {})["job_id_to_deploy"] = job_id

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[INFO] Updated config with job_id_to_deploy: {job_id}")
