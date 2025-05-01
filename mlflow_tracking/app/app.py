#app/app.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError
from typing import Optional

from app.schemas import TrainSettings
from app.app_utils import (
    job_status, job_metrics, job_model_paths,
    load_job_status, run_config_retrain, run_zip_retrain
)

app = FastAPI()

load_job_status()

from fastapi import Request

@app.post("/start_retrain")
async def start_retrain(
    request: Request,
    background_tasks: BackgroundTasks,
    zip_file: Optional[UploadFile] = File(None)
):
    job_id = str(uuid.uuid4())
    job_status[job_id] = "pending"

    if zip_file:
        zip_path = f"/tmp/{job_id}.zip"
        with open(zip_path, "wb") as f:
            f.write(await zip_file.read())
        background_tasks.add_task(run_zip_retrain, zip_path, job_id)
        return {"job_id": job_id}

    try:
        body = await request.json()
        settings = TrainSettings(**body)
        background_tasks.add_task(run_config_retrain, settings.dict(), job_id)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Must provide either valid JSON config or zip_file")


@app.get("/retrain_status")
def retrain_status(job_id: str):
    status = job_status.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Invalid job_id")
    return {"job_id": job_id, "status": status}


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
