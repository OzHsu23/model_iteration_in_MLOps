FROM python:3.11-slim

WORKDIR /mlflow_tracking

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch and torchvision CPU versions
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
COPY ./requirements.txt /mlflow_tracking/requirements.txt
RUN pip install --no-cache-dir -r /mlflow_tracking/requirements.txt

# Copy source code
COPY . /mlflow_tracking



# CMD ["/bin/bash"]

CMD ["python",  "run_app.py"]