# training/api/job_manager.py

import os
import json
import subprocess

JOB_STATUS_FILE = "job_status.json"
job_status = {}

# Load previous job status from file
def load_job_status():
    global job_status
    if os.path.exists(JOB_STATUS_FILE):
        with open(JOB_STATUS_FILE, "r") as f:
            job_status = json.load(f)

# Save current job status to file
def save_job_status():
    with open(JOB_STATUS_FILE, "w") as f:
        json.dump(job_status, f)

# Run retrain job in the background
def run_retrain_job(settings: dict, job_id: str):
    try:
        temp_config_path = f"temp_settings_{job_id}.json"
        with open(temp_config_path, "w") as f:
            json.dump(settings, f)

        # Update job status to in_progress
        job_status[job_id] = "in_progress"

        subprocess.run(
            ["python", "training/train.py", "--config", temp_config_path],
            check=True
        )

        # Update job status to success
        job_status[job_id] = "success"
        save_job_status()

    except subprocess.CalledProcessError:
        # Update job status to failed
        job_status[job_id] = "failed"
        save_job_status()

    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
