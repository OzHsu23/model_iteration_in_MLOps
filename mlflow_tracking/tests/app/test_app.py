import io
import json
import zipfile
import pytest


def test_start_retrain_api(client, test_settings_dict):
    """[POST] /start_retrain - config + zip modes"""
    # Config mode
    res1 = client.post("/start_retrain", json=test_settings_dict)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "submitted"
    assert data1["source"] == "config"
    assert "job_id" in data1

    # Zip mode
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zipf:
        zipf.writestr("inference_setting.json", json.dumps(test_settings_dict))
        zipf.writestr("model.pth", b"fake-model")
    zip_buf.seek(0)

    res2 = client.post(
        "/start_retrain",
        files={"zip_file": ("mock_model.zip", zip_buf, "application/zip")}
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "submitted"
    assert data2["source"] == "zip"
    assert "job_id" in data2


@pytest.fixture
def retrain_job_id(client, test_settings_dict):
    """Fixture: Submit retrain job and return job_id."""
    response = client.post("/start_retrain", json=test_settings_dict)
    return response.json()["job_id"]


def test_retrain_status_api(client, retrain_job_id):
    """[GET] /retrain_status - valid & invalid cases"""
    from training import globals as g
    g.job_status[retrain_job_id] = "running"

    valid = client.get(f"/retrain_status?job_id={retrain_job_id}")
    assert valid.status_code == 200
    assert valid.json()["status"] == "running"

    invalid = client.get("/retrain_status?job_id=nonexistent")
    assert invalid.status_code == 404


def test_retrain_progress_api(client, retrain_job_id):
    """[GET] /retrain_progress - valid & invalid cases"""
    from training import globals as g
    g.job_status[retrain_job_id] = "running"
    g.job_progress[retrain_job_id] = {"epoch": 2, "total": 10}

    valid = client.get(f"/retrain_progress?job_id={retrain_job_id}")
    assert valid.status_code == 200
    assert valid.json()["progress"]["epoch"] == 2

    invalid = client.get("/retrain_progress?job_id=unknown")
    assert invalid.status_code == 404


def test_retrain_metrics_api(client, retrain_job_id):
    """[GET] /retrain_metrics - valid & invalid cases"""
    from training import globals as g
    g.job_status[retrain_job_id] = "completed"
    g.job_metrics[retrain_job_id] = {"accuracy": 0.98, "loss": 0.05}

    valid = client.get(f"/retrain_metrics?job_id={retrain_job_id}")
    assert valid.status_code == 200
    assert "accuracy" in valid.json()

    invalid = client.get("/retrain_metrics?job_id=nope")
    assert invalid.status_code == 404


def test_download_model_api(client, retrain_job_id):
    """[GET] /download_model - valid & invalid cases"""
    from training import globals as g
    g.job_status[retrain_job_id] = "completed"
    model_path = f"/tmp/model_{retrain_job_id}.zip"
    with open(model_path, "wb") as f:
        f.write(b"fake zip")
    g.job_model_paths[retrain_job_id] = model_path

    valid = client.get(f"/download_model?job_id={retrain_job_id}")
    assert valid.status_code == 200
    assert valid.headers["content-type"] == "application/zip"

    invalid = client.get("/download_model?job_id=invalid")
    assert invalid.status_code == 404


def test_latest_job_id_api(client, retrain_job_id):
    """[GET] /latest_job_id - valid & empty cases"""
    valid = client.get("/latest_job_id")
    assert valid.status_code == 200
    assert valid.json()["job_id"] == retrain_job_id

    from training import globals as g
    backup = g.job_status.copy()
    g.job_status.clear()

    empty = client.get("/latest_job_id")
    assert empty.status_code == 404

    g.job_status.update(backup)


def test_full_retrain_lifecycle(client, test_settings_dict):
    """Full retrain flow from submission to download and metrics (integration style)."""
    response = client.post("/start_retrain", json=test_settings_dict)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    from training import globals as g
    g.job_status[job_id] = "running"
    g.job_progress[job_id] = {"epoch": 3, "total": 10}

    progress = client.get(f"/retrain_progress?job_id={job_id}")
    assert progress.status_code == 200
    assert progress.json()["progress"]["epoch"] == 3

    g.job_status[job_id] = "completed"
    g.job_metrics[job_id] = {"accuracy": 0.99, "loss": 0.01}
    model_path = f"/tmp/model_{job_id}.zip"
    with open(model_path, "wb") as f:
        f.write(b"fake")
    g.job_model_paths[job_id] = model_path

    metrics = client.get(f"/retrain_metrics?job_id={job_id}")
    assert metrics.status_code == 200
    assert metrics.json()["accuracy"] > 0.9

    download = client.get(f"/download_model?job_id={job_id}")
    assert download.status_code == 200
