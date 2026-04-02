from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from investing_engine.pipeline.engine import InvestingEngine


def run_pipeline() -> None:
    engine = InvestingEngine()
    engine.run_once()


with DAG(
    dag_id="investing_engine_prototype",
    start_date=datetime(2026, 3, 30),
    schedule="@daily",
    catchup=False,
    tags=["investing", "prototype"],
) as dag:
    PythonOperator(task_id="run_engine", python_callable=run_pipeline)
