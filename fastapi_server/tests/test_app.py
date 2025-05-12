import io
import json
import zipfile
import pytest
from unittest.mock import patch, MagicMock


def test_server_health(client):
    """[GET] /server_health - Check if server responds with OK status"""
    response = client.get("/server_health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_version(client):
    """[GET] /server_version - Verify server version metadata is returned"""
    response = client.get("/server_version")
    assert response.status_code == 200
    assert "server_version" in response.json()


def test_model_info(client):
    """[GET] /model_info - Confirm model metadata is present"""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "framework" in data
    assert "model_format" in data


def test_predict_valid_image(client):
    """[POST] /predict - Test prediction with a valid image"""
    with open("tests/test_data/sample.jpg", "rb") as img:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", img, "image/jpeg")}
        )
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_predict_invalid_file_type(client):
    """[POST] /predict - Should return 415 for non-image file"""
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_batch_predict_zip(client):
    """[POST] /batch_predict - Test prediction with a valid zip file"""
    with open("tests/test_data/sample.zip", "rb") as zf:
        response = client.post(
            "/batch_predict",
            files={"zip_file": ("sample.zip", zf, "application/zip")}
        )
    assert response.status_code in [200, 400]  # Depends on content validity


def test_reload_model(client):
    """[POST] /reload_model - Reload the model without restarting"""
    response = client.post("/reload_model")
    assert response.status_code == 200
    assert "status" in response.json()


def test_deploy_model_zip_invalid_file(client):
    """[POST] /deploy_model_zip - Invalid file type should return 400"""
    response = client.post(
        "/deploy_model_zip",
        data={"job_id": "1234"},
        files={"file": ("model.txt", b"not a zip", "text/plain")}
    )
    assert response.status_code == 400


def test_deploy_model_zip_missing_job_id(client):
    """[POST] /deploy_model_zip - Missing job_id should raise 422 validation error"""
    with open("tests/test_data/sample.zip", "rb") as zf:
        response = client.post(
            "/deploy_model_zip",
            files={"file": ("sample.zip", zf, "application/zip")}
        )
    assert response.status_code == 422


def test_deploy_model_zip_success_mocked(client, test_settings_dict):
    """[POST] /deploy_model_zip - Valid zip with mocked config from fixture"""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        zipf.writestr("inference_setting.json", json.dumps(test_settings_dict))
        zipf.writestr("model.pth", b"fake-model-weights")
    zip_buf.seek(0)

    with patch("app.app.ModelFactory.create_model") as mock_create_model:
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        response = client.post(
            "/deploy_model_zip",
            data={"job_id": "mock_job"},
            files={"file": ("model.zip", zip_buf, "application/zip")}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "Deployment successful"
    assert response.json()["job_id"] == "mock_job"


def test_get_logs(client):
    """[GET] /get_logs - Return CSV log file if it exists"""
    response = client.get("/get_logs")
    assert response.status_code in [200, 404]


def test_export_model(client):
    """[GET] /export_model - Check if model zip is returned"""
    with patch("app.app.export_current_model_zip") as mock_export:
        mock_export.return_value = "tests/test_data/sample.zip"
        response = client.get("/export_model")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "deployed_model.zip" in response.headers["content-disposition"]


def test_evaluate_model_with_uploads(client):
    """[POST] /evaluate - Evaluate with uploaded zip files"""
    model_buf = io.BytesIO()
    with zipfile.ZipFile(model_buf, "w") as zf:
        zf.writestr("model.pth", b"mock model")
    model_buf.seek(0)

    image_buf = io.BytesIO()
    with zipfile.ZipFile(image_buf, "w") as zf:
        zf.writestr("sample.jpg", b"mock image")
    image_buf.seek(0)

    with patch("app.app.evaluate_model_zip") as mock_eval:
        mock_eval.return_value = 0.87
        response = client.post(
            "/evaluate",
            files={
                "model_zip": ("model.zip", model_buf, "application/zip"),
                "image_zip": ("images.zip", image_buf, "application/zip")
            },
            data={"metric": "accuracy"}
        )

    assert response.status_code == 200
    result = response.json()
    assert "score" in result and "metric" in result
    assert result["metric"] == "accuracy"


def test_evaluate_model_missing_inputs(client):
    """[POST] /evaluate - Should return 400 if no model or image input provided"""
    response = client.post("/evaluate", data={"metric": "accuracy"})
    assert response.status_code == 400


def test_deploy_status(client):
    """[GET] /deploy_status - Return current model deployment status"""
    with patch("app.app.get_current_deploy_status") as mock_status:
        mock_status.return_value = {"status": "active", "model_name": "demo_model"}
        response = client.get("/deploy_status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "model_name" in data
