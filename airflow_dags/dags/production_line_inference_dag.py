# dags/production_line_inference_dag.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import json

from schemas import FullConfig
from scripts.run_simulated_inference import run_simulated_inference
from dags.globals import DEFAULT_MONITORING_CONFIG




def simulate_inference_task(**context):
    try:
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", context["params"]["config_path"])
        )
        print(f"[DEBUG] Using config path: {config_path}")
        run_simulated_inference(config_path)
    except Exception as e:
        print(f"[ERROR] Failed in simulate_inference_task: {e}")
        raise

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="production_line_inference_dag",
    description="Simulate production line inference for monitoring",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval=None,
    catchup=False,
    tags=["simulation", "monitoring"],
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string"),
    }
) as dag:


    simulate_task = PythonOperator(
        task_id="run_simulated_inference",
        python_callable=simulate_inference_task,
        provide_context=True
    )
    
    
    simulate_task
