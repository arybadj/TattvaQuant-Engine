"""Prototype build-sequence manifest and phase-gate validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BuildPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_id: int
    name: str
    duration: str
    goal: str
    required_modules: list[str] = Field(default_factory=list)
    gate_rule: str
    required_backtest_metric: str = "positive_sharpe"


class BuildSequence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    non_negotiable_rule: str
    phases: list[BuildPhase]


class PhaseGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_id: int
    phase_name: str
    tests_passed: bool
    sharpe_positive: bool
    may_advance: bool
    blocking_reasons: list[str] = Field(default_factory=list)


def load_build_sequence(path: Path = Path("configs/build_sequence.yaml")) -> BuildSequence:
    import json

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
    except Exception:
        payload = json.loads(text)
    return BuildSequence.model_validate(payload)


def validate_phase_gate(
    phase: BuildPhase,
    *,
    tests_passed: bool,
    walk_forward_sharpe: float,
) -> PhaseGateResult:
    blocking_reasons: list[str] = []
    if not tests_passed:
        blocking_reasons.append("pytest suite did not pass")
    if walk_forward_sharpe <= 0.0:
        blocking_reasons.append("walk-forward Sharpe ratio is not positive")
    return PhaseGateResult(
        phase_id=phase.phase_id,
        phase_name=phase.name,
        tests_passed=tests_passed,
        sharpe_positive=walk_forward_sharpe > 0.0,
        may_advance=not blocking_reasons,
        blocking_reasons=blocking_reasons,
    )
