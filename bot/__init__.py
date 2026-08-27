"""Twilight Struggle Bot Clients and Interactive Players."""

from bot.base_bot import BaseBot
from bot.random_bot import RandomBot
from bot.heuristic_bot import HeuristicBot

try:
    from bot.neural_bot import NeuralBot
except ImportError:
    NeuralBot = None

__all__ = ["BaseBot", "RandomBot", "HeuristicBot", "NeuralBot"]
