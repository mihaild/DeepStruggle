"""Backward-compatibility shim for environment wrappers and action encoders."""
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv, TsSingleEnv
from ai.rewards.reward_calculator import (
    RewardCalculator,
    ZeroSumTerminalReward,
    BlunderAwareRewardCalculator,
    ShapedZeroSumReward,
    UsefulActionsReward,
)

__all__ = [
    "ActionEncoder",
    "TsVectorizedEnv",
    "TsSingleEnv",
    "RewardCalculator",
    "ZeroSumTerminalReward",
    "BlunderAwareRewardCalculator",
    "ShapedZeroSumReward",
    "UsefulActionsReward",
]
