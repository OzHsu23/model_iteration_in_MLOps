# dags/deploy_model_dag.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta

from scripts.deploy_new_model import deploy_model
from dags.globals import DEFAULT_MONITORING_CONFIG

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="deploy_model_dag",
    description="Deploy latest retrained model to inference server",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval=None,
    catchup=False,
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string"),
    },
) as dag:

    deploy_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
        op_args=["{{ params.config_path }}"],
    )


