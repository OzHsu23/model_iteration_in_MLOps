from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# 定義一個簡單的 function
def hello_world():
    print("Hello, world from Airflow!")

# 預設參數
default_args = {
    'owner': 'user',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# 定義 DAG
with DAG(
    dag_id='hello_world_dag',
    default_args=default_args,
    description='A simple Hello World DAG',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=hello_world,
    )
