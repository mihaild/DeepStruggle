"""Evaluation harnesses that measure agent behaviour rather than win rate."""

from .positions import PositionBuilder, build_action_round_position
from .behavioral_suite import (
    Assertion,
    BehavioralTest,
    NeverArgmax,
    Prefer,
    SuiteResult,
    TestResult,
    run_suite,
)

__all__ = [
    "Assertion",
    "BehavioralTest",
    "NeverArgmax",
    "Prefer",
    "PositionBuilder",
    "SuiteResult",
    "TestResult",
    "build_action_round_position",
    "run_suite",
]
