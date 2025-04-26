import io
import os
from app import app
from fastapi.testclient import TestClient



client = TestClient(app)

def test_predict_valid_image():
    with open("tests/test_data/test_image.jpg", "rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")}
        )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()

def test_predict_invalid_file_type():
    with open("tests/test_data/test_file.txt", "rb") as text_file:
        response = client.post(
            "/predict",
            files={"file": ("test_file.txt", text_file, "text/plain")}
        )
    assert response.status_code == 415
    assert response.json()["detail"] == "Only image files supported."

def test_batch_predict_with_folder_path():
    response = client.post(
        "/batch_predict",
        data={"folder_path": "tests/test_data/test_images_folder"}
    )
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)

def test_batch_predict_with_zip_file():
    with open("tests/test_data/test_images.zip", "rb") as zip_file:
        response = client.post(
            "/batch_predict",
            files={"zip_file": ("test_images.zip", zip_file, "application/zip")}
        )
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)

def test_batch_predict_no_input():
    response = client.post("/batch_predict")
    assert response.status_code == 400
    assert response.json()["detail"] == "Either folder_path or zip_file must be provided."

def test_batch_predict_empty_folder():
    response = client.post(
        "/batch_predict",
        data={"folder_path": "tests/empty_folder"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Provided folder_path is not a valid directory."
    def test_predict_invalid_image_file():
        with open("tests/invalid_image.jpg", "rb") as invalid_image_file:
            response = client.post(
                "/predict",
                files={"file": ("invalid_image.jpg", invalid_image_file, "image/jpeg")}
            )
        assert response.status_code == 422  # Assuming invalid image raises a validation error

    def test_predict_logging():
        log_file = "inference_logs.csv"
        if os.path.exists(log_file):
            os.remove(log_file)

        with open("tests/test_image.jpg", "rb") as image_file:
            response = client.post(
                "/predict",
                files={"file": ("test_image.jpg", image_file, "image/jpeg")}
            )
        assert response.status_code == 200

        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            logs = f.readlines()
        assert len(logs) > 0  # Ensure at least one log entry exists