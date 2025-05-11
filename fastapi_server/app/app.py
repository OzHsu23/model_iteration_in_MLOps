# app/app.py

import io
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import zipfile
import argparse
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional


from app.app_utils import (
    extract_images_from_folder,
    extract_images_from_zip,
    resolve_image_inputs,
    predict_images,
    load_setting_json,
    save_log_records,
    create_log_record,
    export_current_model_zip,
    evaluate_model_zip,
    get_current_deploy_status
)
from app.schemas import AppSettings
from factory.model_factory import ModelFactory
from app.globals import SETTING_PATH




# Check if setting file exists
if not os.path.exists(SETTING_PATH):
    raise FileNotFoundError(f"Setting file not found: {SETTING_PATH}")


# Initialize FastAPI app
app = FastAPI(title="Inference Server", version="1.0.0")

# Load settings and initialize model
settings_dict = load_setting_json(path=SETTING_PATH)
settings = AppSettings(**settings_dict)
model_wrapper = ModelFactory.create_model(settings)
app.state.model = model_wrapper
app.state.model_meta = {
    "loaded_from": settings.model.local_path,
    "deployment_time": datetime.utcnow().isoformat(),
}

# ========== Routes ==========

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict a single uploaded image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image files are supported.")
    
    contents = await file.read()
    prediction_result = app.state.model.predict(contents)

    log_record = create_log_record(app.state.model, file.filename, prediction_result)
    save_log_records([log_record.model_dump()], settings.logging.log_path)

    return prediction_result

@app.post("/batch_predict")
async def batch_predict(
    folder_path: str = Form(None),
    zip_file: UploadFile = File(None)
):
    """
    Predict multiple images from a folder path or uploaded zip file.
    Supports server-side folders or zip upload.
    """
    images, temp_dir = resolve_image_inputs(folder_path, zip_file)

    if not images:
        raise HTTPException(400, "No images found to predict.")

    try:
        results = predict_images(images, app.state.model, settings.logging.log_path)
        return JSONResponse(content={"results": results})
    finally:
        if temp_dir:
            temp_dir.cleanup()

@app.post("/reload_model")
async def reload_model():
    """
    Reload the model without restarting the server.
    """
    try:
        setting_path = app.state.model_meta.get("setting_path", SETTING_PATH)
        settings_dict = load_setting_json(setting_path)
        new_settings = AppSettings(**settings_dict)
        model_wrapper = ModelFactory.create_model(new_settings)
        app.state.model = model_wrapper
        return {"status": "Model reloaded successfully."}
    except Exception as e:
        raise HTTPException(500, f"Failed to reload model: {str(e)}")

@app.get("/server_health")
async def server_health():
    """
    Check if the server is alive.
    """
    return {"status": "ok"}

@app.get("/model_info")
async def model_info():
    """
    Get the current model's basic information.
    """
    model = app.state.model
    return {
        "framework": getattr(model, "framework", "unknown"),
        "model_version": getattr(model, "model_version", "unknown"),
        "model_format": getattr(model, "model_format", "unknown"),
        "input_size": getattr(model.preprocessor, "resize", "unknown")
    }

@app.get("/server_version")
async def server_version():
    """
    Get the server version information.
    """
    return {
        "server_version": app.version,
        "server_title": app.title,
        "deployment_time": datetime.utcnow().isoformat()
    }

@app.post("/deploy_model_zip") 
async def deploy_model_zip(
        file: UploadFile = File(...),
        job_id: str = Form(...)
    ):
    """
    Upload a model zip file and reload model using specified job_id.
    """
    if file is None or not hasattr(file, "filename") or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Uploaded file must be a .zip archive.")

    # Extract to a unique directory
    extract_dir = f"./weights/deployed_model_{job_id}"
    os.makedirs(extract_dir, exist_ok=True)

    contents = await file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "Failed to extract zip file. File may be corrupted.")

    setting_path = os.path.join(extract_dir, "inference_setting.json")
    if not os.path.exists(setting_path):
        raise HTTPException(400, "Missing inference_setting.json in the zip package")

    # Load settings and initialize model
    settings_dict = load_setting_json(setting_path)
    new_settings = AppSettings(**settings_dict)
    model_wrapper = ModelFactory.create_model(new_settings)

    # Update FastAPI app state
    app.state.model = model_wrapper
    app.state.model_meta = {
        "job_id": job_id,
        "loaded_from": extract_dir,
        "setting_path": setting_path,
        "deployment_time": datetime.utcnow().isoformat(),
        "model_version": getattr(model_wrapper, "model_version", "unknown")
    }

    return {
        "status": "Deployment successful",
        "job_id": job_id,
        "model_path": extract_dir,
        "deployment_time": app.state.model_meta["deployment_time"]
    }

@app.get("/get_logs")
def get_logs():
    """
    Serve the current inference log file as defined in the deployed model's settings.
    """
    setting_path = app.state.model_meta.get("setting_path", SETTING_PATH)
    settings_dict = load_setting_json(setting_path)
    settings = AppSettings(**settings_dict)

    log_path = settings.logging.log_path
    print(f"[INFO] Returning log file: {log_path}")

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail=f"Log file not found: {log_path}")

    return FileResponse(log_path, media_type="text/csv", filename=os.path.basename(log_path))


@app.get("/export_model")
async def export_model():
    """
    Export the currently deployed model as a zip file.
    """
    zip_path = export_current_model_zip()
    return FileResponse(zip_path, media_type="application/zip", filename="deployed_model.zip")


@app.post("/evaluate")
async def evaluate_model(
    model_zip: UploadFile = File(...),
    image_zip: UploadFile = File(...),
    metric: str = Form("accuracy")
): 
    """
    Evaluate a model using uploaded zip files.
    """
    if model_zip is None or not hasattr(model_zip, "filename") or not model_zip.filename.endswith(".zip"):
        raise HTTPException(400, "Uploaded model file must be a .zip archive.")
    if image_zip is None or not hasattr(image_zip, "filename") or not image_zip.filename.endswith(".zip"):
        raise HTTPException(400, "Uploaded image file must be a .zip archive.")
    # save uploaded model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as m:
        m.write(await model_zip.read())
        model_path = m.name

    # save uploaded images
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as i:
        i.write(await image_zip.read())
        image_path = i.name

    # run evaluation, log on failure
    try:
        score = evaluate_model_zip(model_path, image_path, metric)
    except Exception as e:
        print(f"evaluate_model_zip failed with args: "
                f"model={model_path}, data={image_path}, metric={metric}",
                exc_info=True)
        raise HTTPException(500, f"Evaluation error: {e}")

    return {"score": score, "metric": metric}

@app.get("/deploy_status") 
async def deploy_status(request: Request):
    """
    Return current model deployment info.
    """
    return get_current_deploy_status(request)