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

    Optionally scales the terminal magnitude by how late the game ended, via
    `decisiveness_turns` (K): a result on turn T is worth `1 - T/K` instead of 1.

    This exists because with gamma = 1 and a terminal-only reward the objective is
    *indifferent* to when you win. Taking a forced win now returns +1; declining it and
    winning three turns later also returns +1, so the policy gradient sees no difference
    between them and only risk separates the two. In practice a trained policy leaves a steady
    fraction of engine-verified forced wins on the table, at a rate that stops improving, with
    the misses concentrated where the critic is already optimistic. Perfect-information search
    does not fix it either, which is expected: search maximises the same indifferent objective.

    The scale applies to losses as well, so a self-inflicted defeat on turn 3 costs more than
    the same defeat on turn 9. That is deliberate -- a losing player should prolong the game
    rather than end it, and the sampled replays show turn-3 DEFCON-1 suicides.

    K is a slope, not a threshold: at K = 40 a turn-3 result is worth 0.925 and a turn-10
    result 0.75. Too steep a slope biases against legitimate build-to-final-scoring play, so
    the ending mix is the guardrail to watch.
    """

    def __init__(self, decisiveness_turns: float = 0.0) -> None:
        self.decisiveness_turns = float(decisiveness_turns)

    def terminal_scale(self, state: Optional[ts.GameState]) -> float:
        """Multiplier on the terminal magnitude; 1.0 when disabled or the turn is unknown."""
        if self.decisiveness_turns <= 0.0 or state is None:
            return 1.0
        turn = float(state.turn)
        return max(0.05, 1.0 - turn / self.decisiveness_turns)

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
            scale = self.terminal_scale(st)

            if st is not None:
                if st.defcon <= 1:
                    # Unprovoked DEFCON suicide is a blunder (acting_player == phasing_player).
                    # Provoked DEFCON suicide (event trap where opponent executed coup) is a strategic win (acting_player != phasing_player).
                    p_phasing = int(st.phasing_player)
                    is_provoked = (p_phasing != 0 and p_act != p_phasing)
                    if is_provoked:
                        # Strategic win via event trap: acting player provoked opponent DEFCON suicide (+1.0)
                        rewards[i] = term_util * p_act * scale
                    else:
                        # Unprovoked DEFCON suicide: acting player dropped DEFCON to 1 on their own turn (-1.0)
                        rewards[i] = -1.0 * scale
                    continue
                else:
                    # Check Rule 4.4 held scoring cards at game end (strictly after end of turn)
                    if ts.Engine.is_held_scoring_game_over(st):
                        p_enum = ts.Player.US if p_act == 1 else (ts.Player.USSR if p_act == -1 else ts.Player.NONE)
                        if ts.Engine.is_held_scoring_loss(st, p_enum):
                            rewards[i] = -1.0 * scale
                        else:
                            rewards[i] = 0.0
                        continue

            # Standard strategic win: Winner +1.0, Loser -1.0, scaled by decisiveness.
            rewards[i] = term_util * p_act * scale

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
    and augments intermediate steps with the potential difference Delta Phi defined
    strictly from the US perspective:
        Phi(s) in [-1.0, 1.0] from US perspective
        Delta_Phi_US = Phi(s') - Phi(s)
        R_step = p_act * Delta_Phi_US * potential_scale
    """

    needs_all_states: bool = True

    def __init__(self, potential_scale: float = 1.0):
        super().__init__()
        self.potential_scale: float = potential_scale
        self.prev_potentials: Optional[np.ndarray] = None

    def compute_potential(self, state: ts.GameState, player: Optional[int] = None) -> float:
        """Compute strategic potential Phi(s) in [-1.0, 1.0] strictly from US perspective."""
        return float(ts.Scoring.compute_useful_actions_potential(state, ts.Player.US))

    def reset(self) -> None:
        """Reset internal potential tracking."""
        self.prev_potentials = None

    def on_env_reset(self, env_idx: int, state: ts.GameState) -> None:
        """Called when a single environment resets to initialize its baseline potential to s_0."""
        if self.prev_potentials is not None and env_idx < len(self.prev_potentials):
            self.prev_potentials[env_idx] = self.compute_potential(state)

    def on_all_reset(self, states: List[ts.GameState]) -> None:
        """Called when all environments reset to initialize all baselines."""
        self.prev_potentials = np.array([self.compute_potential(st) for st in states], dtype=np.float32)

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

        # 2. Intermediate potential delta shaping defined strictly from US perspective
        if states is not None and len(states) == N:
            if self.prev_potentials is None or len(self.prev_potentials) != N:
                self.prev_potentials = np.zeros(N, dtype=np.float32)
                for i in range(N):
                    st = states[i]
                    if st is not None:
                        self.prev_potentials[i] = self.compute_potential(st)

            for i in range(N):
                st = states[i]
                if st is not None:
                    curr_pot = self.compute_potential(st)
                    if not dones[i]:
                        delta_us = curr_pot - self.prev_potentials[i]
                        p_act = int(acting_players[i])
                        rewards[i] = float(p_act * delta_us * self.potential_scale)
                        self.prev_potentials[i] = curr_pot
                    else:
                        self.prev_potentials[i] = curr_pot

        return rewards.astype(np.float32)
