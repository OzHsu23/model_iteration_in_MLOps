# app/app.py


import os
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.app_utils import (
    extract_images_from_folder,
    extract_images_from_zip,
    load_setting_json,
    save_log_records,
    create_log_record
)
from app.schemas import AppSettings
from factory.model_factory import ModelFactory

# Initialize FastAPI app
app = FastAPI(title="Inference Server", version="1.0.0")

# Load settings and initialize model
settings_dict = load_setting_json()
settings = AppSettings(**settings_dict)
model_wrapper = ModelFactory.create_model(settings)
app.state.model = model_wrapper

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
    """
    images = []

    if folder_path:
        try:
            images = extract_images_from_folder(folder_path)
        except ValueError as e:
            raise HTTPException(400, str(e))

    elif zip_file:
        if not zip_file.filename.endswith(".zip"):
            raise HTTPException(400, "Uploaded file must be a .zip archive.")
        contents = await zip_file.read()
        images = extract_images_from_zip(contents)

    else:
        raise HTTPException(400, "Either folder_path or zip_file must be provided.")

    if not images:
        raise HTTPException(400, "No images found to predict.")

    results = []
    log_records = []

    for img_path in images:
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        prediction_result = app.state.model.predict(file_bytes)

        results.append({
            "filename": os.path.basename(img_path),
            "prediction": prediction_result
        })

        log_record = create_log_record(app.state.model, os.path.basename(img_path), prediction_result)
        log_records.append(log_record.model_dump())

    save_log_records(log_records, settings.logging.log_path)

    return JSONResponse(content={"results": results})

@app.post("/reload_model")
async def reload_model():
    """
    Reload the model without restarting the server.
    """
    try:
        settings_dict = load_setting_json()
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
