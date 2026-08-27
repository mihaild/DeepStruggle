"""Abstract Reward Calculator Interface & Concrete Strategies for Zero-Sum Games."""

from typing import Protocol, Optional, List, Dict, Any
import numpy as np
import torch
import ts_engine as ts


class RewardCalculator(Protocol):
    """Abstract interface for perspective-aligned reward calculation and zero-sum credit assignment."""

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        """Computes per-environment step rewards from the perspective of the acting player."""
        ...

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        """Computes zero-sum perspective transition sign (+1 if same player, -1 if control switches to opponent)."""
        ...


class ZeroSumTerminalReward:
    """Pure zero-sum algebraic terminal reward strategy.

    - Non-terminal steps: reward = 0.0
    - Terminal step: reward = terminal_outcome * acting_player
      (+1.0 if acting player won, -1.0 if acting player lost, 0.0 for draw).
    - Zero-sum transition sign: sign = curr_player * next_player.
    """

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        rewards = np.where(dones, terminal_utilities * acting_players, 0.0)
        return rewards.astype(np.float32)

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        return (curr_players * next_players).float()


class BlunderAwareRewardCalculator:
    """Blunder-Aware Reward Strategy with Spurious Reward Shielding and Credit Slicing.

    1. Rule 4.4 Held Scoring Card Blunder:
       - Blundering loser receives r = -1.0 on the blunder step.
       - Winner receives r = 0.0 (shielded from spurious +1.0 value inflation).
    2. Voluntary DEFCON Coup Suicide:
       - Blundering player receives r = -1.0.
       - Winner receives r = 0.0.
    3. Strategic Wins (Europe Control, Milestone VP, Final Scoring, Wargames, Event Traps):
       - Full zero-sum terminal outcome: Winner +1.0, Loser -1.0.
    """

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        N = len(acting_players)
        rewards = np.zeros(N, dtype=np.float32)

        for i in range(N):
            if not dones[i]:
                continue

            term_util = float(terminal_utilities[i])
            p_act = int(acting_players[i])

            if term_util == 0.0:
                rewards[i] = 0.0
                continue

            # Check if this was a blunder if state is available
            st = states[i] if states is not None and i < len(states) else None
            is_blunder = False
            loser_player = -1 if term_util > 0 else 1  # USSR is -1, US is 1

            if st is not None:
                if st.defcon <= 1:
                    # Unprovoked DEFCON suicide (direct coup/degradation) is a blunder.
                    # Provoked DEFCON suicide (event trap where opponent executed coup) is a strategic win!
                    if not st.has_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED):
                        is_blunder = True
                else:
                    # Held scoring blunder applies only if the LOSING player held a scoring card
                    if loser_player == 1:
                        is_blunder = any(
                            st.get_card_location(c) == ts.CardLocation.HAND_US
                            and ts.CardData.get_card_info(c).get("is_scoring")
                            for c in range(1, 111)
                        )
                    else:
                        is_blunder = any(
                            st.get_card_location(c) == ts.CardLocation.HAND_USSR
                            and ts.CardData.get_card_info(c).get("is_scoring")
                            for c in range(1, 111)
                        )
            if is_blunder:
                # Shield winner (r_winner = 0.0), heavily penalize loser (r_loser = -1.0)
                if p_act == loser_player:
                    rewards[i] = -1.0
                else:
                    rewards[i] = 0.0
            else:
                # Standard strategic win: Winner +1.0, Loser -1.0
                rewards[i] = term_util * p_act

        return rewards

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        return (curr_players * next_players).float()


class ShapedZeroSumReward:
    """Zero-sum terminal reward augmented with dense VP-delta reward shaping."""

    def __init__(self, vp_scale: float = 0.02):
        self.vp_scale = vp_scale

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        term_rewards = np.where(dones, terminal_utilities * acting_players, 0.0)
        delta_vp_us = (curr_victory_points - prev_victory_points).astype(np.float32)
        shaped_rewards = delta_vp_us * acting_players * self.vp_scale
        total_rewards = term_rewards + shaped_rewards
        return total_rewards.astype(np.float32)

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        return (curr_players * next_players).float()
