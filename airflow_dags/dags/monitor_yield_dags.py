# dags/monitor_yield_dags.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta


# Default DAG arguments
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Define DAG
with DAG(
    dag_id="monitor_yield_dag",
    description="Monitor accuracy from inference logs and trigger retrain flag if needed",
    default_args=default_args,
    start_date=datetime(2025, 4, 30),
    schedule_interval="@daily",   # Run once a day
    catchup=False,
    tags=["monitoring", "manual_trigger"],
    params={  # Accept override via UI
        "log_path": Param("resources/inference_logs.csv", type="string"),
        "config_path": Param("configs/monitoring_config.json", type="string"),
        "flag_path": Param("resources/flags/need_retrain.flag", type="string"),
    }
) as dag:

    # Task: Check accuracy
    def check_yield_wrapper(**context):
        from scripts.check_yield import check_yield

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        def resolve_path(p): return os.path.join(base_dir, p) if not os.path.isabs(p) else p

        log_path = resolve_path(context["params"]["log_path"])
        config_path = resolve_path(context["params"]["config_path"])
        flag_path = resolve_path(context["params"]["flag_path"])

        check_yield(log_path=log_path, config_path=config_path, flag_path=flag_path)

    # Create the Python task
    monitor_yield_task = PythonOperator(
        task_id="check_inference_yield",
        python_callable=check_yield_wrapper
    )

