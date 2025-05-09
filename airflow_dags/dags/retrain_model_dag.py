# dags/retrain_model_dag.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta

from dags.globals import DEFAULT_MONITORING_CONFIG
from scripts import trigger_retrain


def trigger_with_param(**context):
    config_path = context["params"]["config_path"]
    return trigger_retrain.start_retrain_and_wait(config_path)


# DAG default arguments
default_args = {
    "owner": "airflow",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id="retrain_model_dag",
    description="Trigger retraining of the model using the retrain server",
    default_args=default_args,
    start_date=datetime(2025, 5, 1),
    schedule_interval=None,
    catchup=False,
    tags=["retraining"],
    params={
        "config_path": Param(DEFAULT_MONITORING_CONFIG, type="string"),
    }
) as dag:

    retrain_task = PythonOperator(
        task_id="trigger_retrain_and_wait",
        python_callable=trigger_with_param,
        provide_context=True,
    )
    
    retrain_task
