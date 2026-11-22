"""Backward-compatibility shim for environment wrappers and action encoders."""
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from ai.rewards.reward_calculator import (
    RewardCalculator,
    ZeroSumTerminalReward,
    ShapedZeroSumReward,
    UsefulActionsReward,
)

__all__ = [
    "ActionEncoder",
    "TsVectorizedEnv",
    "RewardCalculator",
    "ZeroSumTerminalReward",
    "ShapedZeroSumReward",
    "UsefulActionsReward",
]
