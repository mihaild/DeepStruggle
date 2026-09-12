"""Perspective-aligned reward strategies and blunder-aware credit assignment."""
from .reward_calculator import (
    RewardCalculator,
    ZeroSumTerminalReward,
    BlunderAwareRewardCalculator,
    ShapedZeroSumReward,
    UsefulActionsReward,
)

__all__ = [
    "RewardCalculator",
    "ZeroSumTerminalReward",
    "BlunderAwareRewardCalculator",
    "ShapedZeroSumReward",
    "UsefulActionsReward",
]
