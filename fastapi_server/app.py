
from PIL import Image
import io, os, pandas as pd

from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from app_utils import *
from factory.model_factory import ModelFactory

app = FastAPI()

# ---------- App Initialization ----------
setting = load_setting_json()
wrapper = ModelFactory.create_model(setting)

# ---------- Routes ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Only image files supported.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence = wrapper.predict(image)

    # Logging
    log = {
        "filename": file.filename,
        "pred": label,
        "confidence": confidence,
        "label": None
    }
    df = pd.DataFrame([log])
    path = "inference_logs.csv"
    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

    return {"prediction": label, "confidence": confidence}

@app.post("/batch_predict")
async def batch_predict(
    folder_path: str = Form(None),
    zip_file: UploadFile = File(None)
):
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

    results = batch_predict_images(images, wrapper)
    return JSONResponse(content={"results": results})