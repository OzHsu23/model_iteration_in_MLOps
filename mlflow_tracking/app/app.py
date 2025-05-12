#app/app.py

import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError
from typing import Optional



from app.schemas import TrainSettings
from app.app_utils import (
    load_job_status, save_job_status,
    run_config_retrain, run_zip_retrain
)

from app.globals import (
    job_status,
    job_progress,
    job_model_paths,
    job_metrics,
    PACKAGE_DIR
)

app = FastAPI()

load_job_status()


@app.post("/start_retrain")
async def start_retrain(request: Request, background_tasks: BackgroundTasks, zip_file: Optional[UploadFile] = File(None)):
    
    job_id = str(uuid.uuid4())
    job_status[job_id] = "pending"
    save_job_status()

    try:
        if zip_file:
            # Handle retrain from ZIP file
            zip_path = f"{PACKAGE_DIR}/{job_id}.zip"
            with open(zip_path, "wb") as f:
                f.write(await zip_file.read())
            background_tasks.add_task(run_zip_retrain, zip_path, job_id)

            return {
                "job_id": job_id,
                "status": "submitted",
                "source": "zip",
                "message": "Zip-based retrain job submitted. Use /retrain_status to check progress."
            }

        # Handle retrain from JSON config
        body = await request.json()
        background_tasks.add_task(run_config_retrain, body, job_id)

        return {
            "job_id": job_id,
            "status": "submitted",
            "source": "config",
            "message": "Config-based retrain job submitted. Use /retrain_status to check progress."
        }

    except Exception as e:
        # Fail-safe: log error and return consistent structure
        job_status[job_id] = "failed"
        job_metrics[job_id] = {"error": str(e)}
        save_job_status()

        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "message": "Failed to parse input or submit retrain task."
        }


@app.get("/retrain_status")
def retrain_status(job_id: str):
    status = job_status.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Invalid job_id")
    result = {"job_id": job_id, "status": status}
    if status == "failed":
        result["error"] = job_metrics.get(job_id, {}).get("error", "Unknown error")
    return result

@app.get("/retrain_progress")
def retrain_progress(job_id: str):
    """
    Return live training progress for a given job_id.
    """
    if job_id not in job_progress:
        raise HTTPException(status_code=404, detail="No progress found for this job")

    return {
        "job_id": job_id,
        "status": job_status.get(job_id, "unknown"),
        "progress": job_progress[job_id]
    }

@app.get("/retrain_metrics")
def retrain_metrics(job_id: str):
    metrics = job_metrics.get(job_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="No metrics found for this job")
    return metrics


@app.get("/download_model")
def download_model(job_id: str):
    model_zip = job_model_paths.get(job_id)
    if model_zip is None or not os.path.exists(model_zip):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_zip, media_type="application/zip", filename=f"model_{job_id}.zip")

@app.get("/latest_job_id")
def get_latest_job_id():
    if not job_status:
        raise HTTPException(status_code=404, detail="No retrain jobs found.")
    latest_id = list(job_status.keys())[-1]
    return {
        "job_id": latest_id,
        "status": job_status[latest_id],
        "error": job_metrics.get(latest_id, {}).get("error")
    }