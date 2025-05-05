# dags/prepare_training_data_dag.py

import os
import sys
import json
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param

from schemas import FullConfig
from scripts.prepare_training_samples import extract_samples_and_prepare_training_data
from dags.globals import DEFAULT_MONITORING_CONFIG

default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="prepare_training_data_dag",
    description="Prepare retraining data from recent inference logs and B1/B2 results",
    default_args=default_args,
    start_date=datetime(2025, 4, 30),
    schedule_interval="@daily",
    catchup=False,
    tags=["data_preparation"],
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string"),
    }
) as dag:

    def prepare_training_data(**context):
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", context["params"]["config_path"]))
        with open(config_path, "r") as f:
            raw_config = json.load(f)
            config = FullConfig(**raw_config)

        if os.path.exists(config.monitor.flag_path):
            print("[INFO] Retrain flag detected. Proceeding with data preparation.")
            extract_samples_and_prepare_training_data(config)
        else:
            print("[INFO] No retrain flag found. Skipping.")

    prepare_data_task = PythonOperator(
        task_id="extract_and_prepare_samples",
        python_callable=prepare_training_data,
        provide_context=True
    )
    
    prepare_data_task
