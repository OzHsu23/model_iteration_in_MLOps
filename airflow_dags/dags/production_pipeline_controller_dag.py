# production_pipeline_controller_dag.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.sensors.time_delta import TimeDeltaSensor


from dags.globals import DEFAULT_MONITORING_CONFIG
from schemas import FullConfig


def check_retrain_flag(**kwargs):
    """
    Check if the retrain flag file exists.
    Return the task_id of the next task based on the flag.
    """
    config_path = kwargs["params"]["config_path"]
    with open(config_path, "r") as f:
        config = json.load(f)
    flag_path = config["monitor"]["flag_path"]
    if os.path.exists(flag_path):
        return "prepare_training_data"
    else:
        return "not_retrain"

def check_deploy_flag(**kwargs):
    """
    Check if the deploy flag file exists (e.g., set by evaluation DAG).
    Return the task_id of the next task: deploy or skip.
    """
    config_path = kwargs["params"]["config_path"]
    with open(config_path, "r") as f:
        config = json.load(f)
    result_flag_path = config["evaluate_before_deploy"]["result_flag_path"]
    with open(result_flag_path, "r") as f:
        result = json.load(f)
    isdeploy = result["deploy"]
    if isdeploy:
        return "deploy_new_model"
    else:
        return "skip_deploy"

def get_monitor_delay_from_config(config_path):
    with open(config_path, "r") as f:
        full_config = FullConfig(**json.load(f))
    return full_config.monitor.monitor_delay_sec

def delay_monitoring(**context):
    config_path = context["params"]["config_path"]
    delay_sec = get_monitor_delay_from_config(config_path)
    print(f"[INFO] Delaying monitor start by {delay_sec} seconds...")
    time.sleep(delay_sec)

def skip_message():
    print("No retrain needed this round.")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="model_iteration_in_MLOps",
    description="Controller DAG to simulate production loop: inference → monitor → retrain → deploy",
    start_date=datetime(2025, 5, 5),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    params={"config_path": Param(DEFAULT_MONITORING_CONFIG, type="string")},
    tags=["controller", "production", "loop"]
) as dag:

    # Step 1: Simulate inference and write logs
    t1_inference = TriggerDagRunOperator(
        task_id="production_line_inference",
        trigger_dag_id="production_line_inference_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )
    
    t_delay_monitor = PythonOperator(
        task_id="wait_before_monitoring",
        python_callable=delay_monitoring,
        provide_context=True
    )

    # Step 2: Monitor yield from logs
    t2_monitor = TriggerDagRunOperator(
        task_id="monitor_yield",
        trigger_dag_id="monitor_yield_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )

    # Step 3: Branch - check if retrain flag is present
    t3_check_flag = BranchPythonOperator(
        task_id="should_retrain",
        python_callable=check_retrain_flag,
        provide_context=True
    )

    # Step 4: Prepare training data from recent samples
    t4_prepare = TriggerDagRunOperator(
        task_id="prepare_training_data",
        trigger_dag_id="prepare_training_data_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )

    # Step 5: Retrain model and log with MLflow
    t5_retrain = TriggerDagRunOperator(
        task_id="retrain_model",
        trigger_dag_id="retrain_model_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )
    
    # Step 5.5: Evaluate new model performace
    t5_5_evaluate = TriggerDagRunOperator(
        task_id="evaluate_model_performance",
        trigger_dag_id="evaluate_model_before_deploy_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )
    
    t5_6_check_deploy_flag = BranchPythonOperator(
        task_id="should_deploy",
        python_callable=check_deploy_flag,
        provide_context=True
    )

    # Step 6: Deploy the new model to FastAPI
    t6_deploy = TriggerDagRunOperator(
        task_id="deploy_new_model",
        trigger_dag_id="deploy_model_dag",
        conf={"config_path": "{{ params.config_path }}"},
        wait_for_completion=True
    )

    t_skip_deploy = PythonOperator(
        task_id="skip_deploy",
        python_callable=lambda: print("Evaluation not improved. Skip deploy.")
    )


    t_end = PythonOperator(
        task_id="skip_retrain",
        python_callable=skip_message
    )

    # Define dependencies (control flow)
    t1_inference
    t_delay_monitor >> t2_monitor >> t3_check_flag 
    
    # retrain → evaluate → check deploy flag
    t3_check_flag >> t4_prepare >> t5_retrain >> t5_5_evaluate >> t5_6_check_deploy_flag
    
    # branch: deploy or skip
    t5_6_check_deploy_flag >> t_skip_deploy
    t5_6_check_deploy_flag >> t6_deploy
    
    
    # not retrain → end cycle
    t3_check_flag >> t_end
