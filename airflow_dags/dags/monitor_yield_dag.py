# dags/monitor_yield_dags.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta

from dags.globals import DEFAULT_MONITORING_CONFIG
from scripts.check_yield import check_yield
from schemas import FullConfig
import json
import requests

# Default DAG arguments
default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

def fetch_log_from_fastapi(config: FullConfig) -> str:
    """
    Download the inference log file from the inference server defined in config.
    Save to config.monitor.log_path if provided, otherwise /tmp/inference_log.csv.
    """
    url = config.production_line.inference_server_api.rstrip("/").replace("/predict", "/get_logs")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch log from inference server: {e}")

    out_path = str(config.monitor.log_path or "/tmp/inference_log.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"[INFO] Downloaded log to: {out_path}")
    return out_path

def check_yield_wrapper(**context):
    config_path = context["params"]["config_path"]

    with open(config_path, "r") as f:
        raw_config = json.load(f)
    full_config = FullConfig(**raw_config)

    # Fetch log from FastAPI, and override in-place
    downloaded_log_path = fetch_log_from_fastapi(full_config)
    full_config.monitor.log_path = downloaded_log_path

    check_yield(full_config)

# Define DAG
with DAG(
    dag_id="monitor_yield_dag",
    description="Monitor yield from inference logs and trigger retrain flag if needed",
    default_args=default_args,
    start_date=datetime(2025, 4, 30),
    schedule_interval="@daily",
    catchup=False,
    tags=["monitoring", "manual_trigger"],
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string")
    }
) as dag:
    monitor_yield_task = PythonOperator(
        task_id="check_inference_yield",
        python_callable=check_yield_wrapper
    )
    
    monitor_yield_task