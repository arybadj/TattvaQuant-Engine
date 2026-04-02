from __future__ import annotations

from pathlib import Path


def test_project_structure_contains_required_directories() -> None:
    required_paths = [
        Path("src/data"),
        Path("src/features"),
        Path("src/models"),
        Path("src/fusion"),
        Path("src/uncertainty"),
        Path("src/rl"),
        Path("src/execution"),
        Path("src/feedback"),
        Path("src/api"),
        Path("tests"),
        Path("notebooks"),
        Path("configs"),
        Path("docker"),
        Path("airflow"),
        Path("mlflow"),
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required path: {path}"


def test_project_structure_contains_required_wrapper_files() -> None:
    required_files = [
        Path("src/models/fund_model.py"),
        Path("src/fusion/regime_classifier.py"),
        Path("src/uncertainty/shift_detector.py"),
        Path("src/execution/cost_model.py"),
        Path("src/feedback/metrics.py"),
        Path("src/rl/agent.py"),
        Path("src/api/app.py"),
    ]
    for path in required_files:
        assert path.exists(), f"Missing required file: {path}"
