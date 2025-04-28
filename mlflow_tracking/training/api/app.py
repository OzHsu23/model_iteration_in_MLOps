# training/api/app.py

from fastapi import FastAPI, BackgroundTasks
from training.api.schemas import TrainSettings
from training.api.job_manager import run_retrain_job, job_status, load_job_status
import uuid

app = FastAPI()

# Load previous jobs when server starts
load_job_status()

# API to start retrain
@app.post("/start_retrain")
async def start_retrain(settings: TrainSettings, background_tasks: BackgroundTasks):
    # Generate a new job_id
    job_id = str(uuid.uuid4())
    job_status[job_id] = "pending"

    # Add retrain task to background
    background_tasks.add_task(run_retrain_job, settings.dict(), job_id)

    return {"job_id": job_id}

# API to check retrain status
@app.get("/retrain_status")
async def retrain_status(job_id: str):
    # Lookup job status
    status = job_status.get(job_id)
    if status is None:
        return {"error": "Invalid job_id"}
    return {"job_id": job_id, "status": status}
