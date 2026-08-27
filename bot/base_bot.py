"""Abstract BaseBot Interface for Twilight Struggle AI Agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import sys

try:
    import ts_engine as ts
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine as ts


class BaseBot(ABC):
    """Abstract base class for all Twilight Struggle bot clients and agents."""

    def __init__(self, role: str, name: Optional[str] = None):
        self.role: str = role.upper()  # "US" or "USSR"
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Selects an action dictionary according to the WebSocket / REST protocol.

        Args:
            state: Serialized GameStateDict representation of the game.
            legal_actions: Dict containing decision_type, valid_ids, allow_early_stop.

        Returns:
            Dict representing micro-action or None if no valid action is possible.
        """
        ...

    def select_flat_action(self, state: ts.GameState, player: ts.Player) -> int:
        """Selects a 212-dim flat action index directly against the C++ engine.

        Subclasses may override this for vectorized or high-speed flat inference.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement select_flat_action directly.")

    def reset(self) -> None:
        """Resets any internal bot state (memory, search tree, rollout cache)."""
        pass
