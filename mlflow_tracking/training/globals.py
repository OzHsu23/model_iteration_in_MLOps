# training/globals.py

import json

# Global dictionaries to store job-related state
job_progress = {}        # Tracks current epoch, loss, accuracy, etc.
job_status = {}          # Tracks status: "pending", "running", "completed", "failed"
job_metrics = {}         # Final metrics or error message for the job
job_model_paths = {}     # File paths to saved models (zip)

# Path to persist all job states
job_registry_path = "job_cache.json"

# Persist all global job states to disk
def save_job_status():
    with open(job_registry_path, "w") as f:
        json.dump({
            "job_status": job_status,
            "job_metrics": job_metrics,
            "job_model_paths": job_model_paths,
            "job_progress": job_progress
        }, f, indent=2)

