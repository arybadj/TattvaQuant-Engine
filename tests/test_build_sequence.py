from __future__ import annotations

from src.project.build_sequence import load_build_sequence, validate_phase_gate


def test_build_sequence_has_seven_ordered_phases() -> None:
    sequence = load_build_sequence()
    assert len(sequence.phases) == 7
    assert [phase.phase_id for phase in sequence.phases] == [1, 2, 3, 4, 5, 6, 7]


def test_phase_gate_blocks_when_tests_fail() -> None:
    sequence = load_build_sequence()
    result = validate_phase_gate(sequence.phases[0], tests_passed=False, walk_forward_sharpe=0.8)
    assert result.may_advance is False
    assert "pytest suite did not pass" in result.blocking_reasons


def test_phase_gate_blocks_when_sharpe_is_not_positive() -> None:
    sequence = load_build_sequence()
    result = validate_phase_gate(sequence.phases[0], tests_passed=True, walk_forward_sharpe=0.0)
    assert result.may_advance is False
    assert "walk-forward Sharpe ratio is not positive" in result.blocking_reasons


def test_phase_gate_allows_advance_only_when_both_conditions_pass() -> None:
    sequence = load_build_sequence()
    result = validate_phase_gate(sequence.phases[0], tests_passed=True, walk_forward_sharpe=0.35)
    assert result.may_advance is True
