"""Shared evaluation, simulation, and analytics engine for Twilight Struggle AI tools."""

from tools.lib.player_agent import (
    PlayerAgent,
    NeuralAgent,
    HeuristicAgent,
    RandomAgent,
    load_agent,
    resolve_device,
)
from tools.lib.batch_tournament import BatchMatchRunner, compute_mle_elo
from tools.lib.tournament_evaluator import TournamentEvaluator, classify_game_ending_reason
from tools.lib.self_play import generate_self_play_replay
from tools.lib.scoring_formatter import format_regional_scoring_breakdown
from tools.lib.checkpoint_utils import inspect_checkpoint, discover_checkpoints

__all__ = [
    "PlayerAgent",
    "NeuralAgent",
    "HeuristicAgent",
    "RandomAgent",
    "load_agent",
    "resolve_device",
    "BatchMatchRunner",
    "compute_mle_elo",
    "TournamentEvaluator",
    "classify_game_ending_reason",
    "generate_self_play_replay",
    "format_regional_scoring_breakdown",
    "inspect_checkpoint",
    "discover_checkpoints",
]
