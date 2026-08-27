"""Perspective-aligned reward strategies and blunder-aware credit assignment."""
from .reward_calculator import (
    RewardCalculator,
    ZeroSumTerminalReward,
    BlunderAwareRewardCalculator,
    ShapedZeroSumReward,
)

__all__ = [
    "RewardCalculator",
    "ZeroSumTerminalReward",
    "BlunderAwareRewardCalculator",
    "ShapedZeroSumReward",
]
