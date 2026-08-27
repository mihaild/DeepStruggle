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

            if st is not None:
                if st.defcon <= 1:
                    # Unprovoked DEFCON suicide is a blunder (acting_player == phasing_player).
                    # Provoked DEFCON suicide (event trap where opponent executed coup) is a strategic win (acting_player != phasing_player).
                    p_phasing = int(st.phasing_player)
                    is_provoked = (p_phasing != 0 and p_act != p_phasing)
                    if is_provoked:
                        # Strategic win via event trap: acting player provoked opponent DEFCON suicide (+1.0)
                        rewards[i] = term_util * p_act
                    else:
                        # Unprovoked DEFCON suicide: acting player dropped DEFCON to 1 on their own turn (-1.0)
                        rewards[i] = -1.0
                    continue
                else:
                    # Check Rule 4.4 held scoring cards at game end
                    held_scoring_us = any(
                        st.get_card_location(c) == ts.CardLocation.HAND_US
                        and ts.CardData.get_card_info(c).get("is_scoring")
                        for c in range(1, 111)
                    )
                    held_scoring_ussr = any(
                        st.get_card_location(c) == ts.CardLocation.HAND_USSR
                        and ts.CardData.get_card_info(c).get("is_scoring")
                        for c in range(1, 111)
                    )
                    loser_player = -1 if term_util > 0 else 1  # USSR is -1, US is 1
                    loser_held_scoring = held_scoring_us if loser_player == 1 else held_scoring_ussr

                    if loser_held_scoring:
                        # Game ended because of holding scoring cards (Rule 4.4).
                        # Give reward -1.0 to each side that holds scoring cards (in case both sides hold scorings).
                        # A side not holding scoring cards is shielded (0.0).
                        act_holds_scoring = held_scoring_us if p_act == 1 else held_scoring_ussr
                        if act_holds_scoring:
                            rewards[i] = -1.0
                        else:
                            rewards[i] = 0.0
                        continue

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


class UsefulActionsReward(BlunderAwareRewardCalculator):
    """Reward calculator encouraging usually useful actions via strategic potential shaping.

    Inherits endgame blunder-shielded terminal rewards from BlunderAwareRewardCalculator,
    and augments intermediate steps with the potential difference Delta Phi:
        R_step = Phi(s', p_act) - Phi(s, p_act)

    Potential Function Phi(s, p) consists of 4 normalized components:
    1. (vp in favor of current player) / 20, weight 0.5
    2. (battlegrounds controlled by player - battlegrounds controlled by opponent) / total_bgs, weight 0.3
    3. (sum across all unscored regions of (vp region would give) / (control_vp + num_bg)) / 6, weight 0.1
    4. (countries player has access to - countries opponent has access to) / 84, weight 0.1
    """

    needs_all_states: bool = True

    def __init__(self, potential_scale: float = 1.0):
        super().__init__()
        self.potential_scale: float = potential_scale
        self.prev_potentials: Optional[np.ndarray] = None

    def compute_potential(self, state: ts.GameState, player: int) -> float:
        """Compute the strategic potential Phi(s, p) in [-1.0, 1.0] for the given player."""
        p_enum = ts.Player.US if player == 1 else (ts.Player.USSR if player == -1 else ts.Player.NONE)
        return float(ts.Scoring.compute_useful_actions_potential(state, p_enum))

    def reset(self) -> None:
        """Reset internal potential tracking."""
        self.prev_potentials = None

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
        # 1. Terminal rewards from BlunderAwareRewardCalculator
        rewards = super().compute_step_rewards(
            acting_players=acting_players,
            dones=dones,
            terminal_utilities=terminal_utilities,
            prev_victory_points=prev_victory_points,
            curr_victory_points=curr_victory_points,
            states=states,
        )

        # 2. Intermediate potential delta shaping
        if states is not None and len(states) == N:
            if self.prev_potentials is None or len(self.prev_potentials) != N:
                self.prev_potentials = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    st = states[i]
                    if st is not None:
                        self.prev_potentials[i] = self.compute_potential(st, int(acting_players[i]))

            for i in range(N):
                st = states[i]
                if st is not None:
                    p_act = int(acting_players[i])
                    curr_pot = self.compute_potential(st, p_act)
                    if not dones[i]:
                        delta_pot = curr_pot - self.prev_potentials[i]
                        rewards[i] = float(delta_pot * self.potential_scale)
                    self.prev_potentials[i] = 0.0 if dones[i] else curr_pot

        return rewards.astype(np.float32)
