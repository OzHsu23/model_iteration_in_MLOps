# dags/evaluate_model_before_deploy_dag.py

import os
import sys
import json
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dags.globals import DEFAULT_MONITORING_CONFIG
from schemas import FullConfig
from scripts.evaluate_model import evaluate_model_and_decide

# Default DAG arguments
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def evaluate_wrapper(**context):
    config_path = context["params"]["config_path"]

    with open(config_path, "r") as f:
        raw_config = json.load(f)
    full_config = FullConfig(**raw_config)

    evaluate_model_and_decide(full_config)

# Define DAG
with DAG(
    dag_id="evaluate_model_before_deploy_dag",
    description="Evaluate new model on test set before deployment and decide if deployment should proceed",
    default_args=default_args,
    start_date=datetime(2025, 4, 30),
    schedule_interval=None,
    catchup=False,
    tags=["evaluation", "manual_trigger"],
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string")
    }
) as dag:
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model_and_decide",
        python_callable=evaluate_wrapper
    )

    evaluate_model_task