"""Regime-aware fusion engine."""

from src.fusion.fusion_engine import (
    AttentionFusion,
    FusedSignal,
    FusionEngine,
    FusionOutput,
    LambdaController,
    LambdaSet,
    LambdaWeights,
    RegimeClassification,
    RegimeClassifier,
    RegimeState,
)

__all__ = [
    "AttentionFusion",
    "FusedSignal",
    "FusionEngine",
    "FusionOutput",
    "LambdaSet",
    "LambdaController",
    "LambdaWeights",
    "RegimeState",
    "RegimeClassification",
    "RegimeClassifier",
]
