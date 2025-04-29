def test_server_health(client):
    """Test if server health endpoint is reachable"""
    response = client.get("/server_health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_server_version(client):
    """Test if server version returns expected keys"""
    response = client.get("/server_version")
    assert response.status_code == 200
    assert "server_version" in response.json()

def test_model_info(client):
    """Test if model_info returns expected metadata"""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "framework" in data
    assert "model_format" in data

def test_predict_valid_image(client):
    """Test predict route with valid image"""
    with open("tests/test_data/sample.jpg", "rb") as img:
        response = client.post("/predict", files={"file": ("sample.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_predict_invalid_file_type(client):
    """Test predict route with non-image file"""
    response = client.post("/predict", files={"file": ("sample.txt", b"hello", "text/plain")})
    assert response.status_code == 415

def test_batch_predict_zip(client):
    """Test batch_predict with a zip file (mocked content)"""
    with open("tests/test_data/sample.zip", "rb") as zf:
        response = client.post("/batch_predict", files={"zip_file": ("sample.zip", zf, "application/zip")})
    assert response.status_code in [200, 400]  # depends on mocked content

def test_reload_model(client):
    """Test model reload endpoint"""
    response = client.post("/reload_model")
    assert response.status_code == 200
    assert "status" in response.json()
