"""Abstract Reward Calculator Interface & Concrete Strategies for Zero-Sum Games."""

from typing import Callable, Protocol, Optional, List, Dict, Any, Sequence
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
      (+1.0 if acting player won, -1.0 if acting player lost, 0.0 for draw), whatever the ending.
    - Zero-sum transition sign: sign = curr_player * next_player.

    Optionally scales the terminal magnitude by how late the game ended, via `decisiveness_turns`
    (K): a result on turn T is worth `max(0.05, 1 - T/K)` instead of 1 (0, the default, leaves it at 1).

    This exists because with gamma = 1 and a terminal-only reward the objective is *indifferent* to
    when you win: taking a forced win now returns +1, and so does declining it and winning three
    turns later. The scale applies to losses as well, so a self-inflicted defeat on turn 3 costs
    more than the same defeat on turn 9 -- a losing player should prolong the game. K is a slope,
    not a threshold: at K = 40 a turn-3 result is worth 0.925 and a turn-10 result 0.75. (Until
    2026-11-22 the scale lived on the blunder-aware scheme, which was removed: owner, "we tested
    that it is not needed".)
    """

    def __init__(self, decisiveness_turns: float = 0.0) -> None:
        self.decisiveness_turns = float(decisiveness_turns)

    def terminal_scale(self, state: Optional[ts.GameState]) -> float:
        """Multiplier on the terminal magnitude; 1.0 when disabled or the turn is unknown."""
        if self.decisiveness_turns <= 0.0 or state is None:
            return 1.0
        return max(0.05, 1.0 - float(state.turn) / self.decisiveness_turns)

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        rewards = np.where(dones, terminal_utilities * acting_players, 0.0).astype(np.float32)
        if self.decisiveness_turns > 0.0 and states is not None:
            for i in np.flatnonzero(dones):
                st = states[i] if i < len(states) else None
                rewards[i] *= self.terminal_scale(st)
        return rewards

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        return (curr_players * next_players).float()


#: The engine's ending names (ts.game_ending_reason) by the ending families --target-endings names.
ENDING_FAMILY: Dict[str, str] = {
    "20 VP": "20vp",
    "Europe Control": "europe_control",
    "DEFCON 1 (own decision)": "defcon",
    "DEFCON 1 (opponent decision)": "defcon",
    "wargames": "wargames",
    "final scoring": "final_scoring",
}
TARGET_ENDINGS = ("20vp", "europe_control", "defcon", "wargames", "final_scoring")


class EndingTargetedReward:
    """An exploiter's payoff for one way of winning (owner, 2026-11-18): from the learner's seat, +1 for a win by one
    of `targets` (ending families of ENDING_FAMILY), -1 for any loss, 0 for a win any other way and for a draw. Written
    zero-sum from the US's side (the fixed opponent does not train), so GAE's alternating signs hold as in the terminal
    scheme. Needs the learner's seat: an exploiter run, --opponent-lock-side.
    """

    def __init__(self, targets: Sequence[str], learner_side: int) -> None:
        bad = [t for t in targets if t not in TARGET_ENDINGS]
        if bad or not targets:
            raise ValueError(f"--target-endings must name some of {TARGET_ENDINGS}, got {list(targets)}")
        if learner_side not in (1, -1, 0):
            raise ValueError(f"learner_side must be 1, -1 or 0 (both seats, drawn per game), got {learner_side}")
        self.targets = frozenset(targets)
        self.learner_side = int(learner_side)
        self.sides_of: Optional[Callable[[], np.ndarray]] = None

    def attach_sides(self, sides_of: Callable[[], np.ndarray]) -> None:
        """Both seats (learner_side 0, owner 2026-11-20): `sides_of()` is each env's exploiter seat this game."""
        self.sides_of = sides_of

    def payoff(self, terminal_utility: float, state: Optional[ts.GameState], side: Optional[int] = None) -> float:
        """The learner's payoff for one finished game, the learner in seat `side` (default: the locked one)."""
        side = self.learner_side if side is None else int(side)
        winner = 1 if terminal_utility > 0 else -1 if terminal_utility < 0 else 0
        if winner == 0:
            return 0.0
        if winner != side:
            return -1.0
        if state is None:
            raise ValueError("an ending-targeted reward needs the finished game's state")
        return 1.0 if ENDING_FAMILY.get(str(ts.game_ending_reason(state))) in self.targets else 0.0

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        rewards = np.zeros(len(dones), dtype=np.float32)
        if self.learner_side == 0:
            if self.sides_of is None:
                raise RuntimeError("a both-seat ending-targeted reward reads each game's seat from the pool: attach_sides first")
            sides = np.asarray(self.sides_of(), dtype=np.int64)
        else:
            sides = np.full(len(dones), self.learner_side, dtype=np.int64)
        for i in np.flatnonzero(dones):
            st = states[int(i)] if states is not None else None
            u_us = self.payoff(float(terminal_utilities[i]), st, int(sides[i])) * int(sides[i])
            rewards[i] = u_us * float(acting_players[i])
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


class VPPotentialShaping:
    """Potential-based VP shaping on top of any reward calculator.

    Phi(s) = scale * VP(s) from the US side, and Phi(terminal) = 0. Each step adds
    Phi(s') - Phi(s) in the acting player's frame, so the terminal step adds -Phi(s), the VP
    lead the game ended on. With gamma = 1 the shaping over a game sums to -Phi(s_0), a
    constant (0 from a fresh deal), so it does not change which policy is optimal (Ng et al.
    1999). It only moves credit earlier: VP gained now is paid now and given back at the end.
    The critic learns V(s) - Phi(s) in the mover's frame instead of V(s).
    """

    def __init__(self, base: Any, scale: float) -> None:
        self.base = base
        self.scale = float(scale)

    def __getattr__(self, name: str) -> Any:
        # Reached only for names not found on the wrapper itself, e.g. `needs_all_states` or
        # `on_env_reset`, which the env looks up on the calculator.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def compute_step_rewards(
        self,
        acting_players: np.ndarray,
        dones: np.ndarray,
        terminal_utilities: np.ndarray,
        prev_victory_points: np.ndarray,
        curr_victory_points: np.ndarray,
        states: Optional[List[Optional[ts.GameState]]] = None,
    ) -> np.ndarray:
        rewards = self.base.compute_step_rewards(acting_players, dones, terminal_utilities,
                                                 prev_victory_points, curr_victory_points, states)
        prev = prev_victory_points.astype(np.float32)
        curr = np.where(dones, 0.0, curr_victory_points.astype(np.float32))
        shaping = self.scale * (curr - prev) * acting_players.astype(np.float32)
        return (rewards + shaping).astype(np.float32)

    def compute_zero_sum_sign(
        self,
        curr_players: torch.Tensor,
        next_players: torch.Tensor,
    ) -> torch.Tensor:
        return self.base.compute_zero_sum_sign(curr_players, next_players)


class UsefulActionsReward(ZeroSumTerminalReward):
    """Reward calculator encouraging usually useful actions via strategic potential shaping.

    Terminal steps take ZeroSumTerminalReward's payoff (winner +1, loser -1, scaled by
    `decisiveness_turns`; the blunder shielding it inherited before 2026-11-22 was removed),
    and intermediate steps the potential difference Delta Phi defined strictly from the US
    perspective:
        Phi(s) in [-1.0, 1.0] from US perspective
        Delta_Phi_US = Phi(s') - Phi(s)
        R_step = p_act * Delta_Phi_US * potential_scale
    """

    needs_all_states: bool = True

    def __init__(self, potential_scale: float = 1.0, decisiveness_turns: float = 0.0) -> None:
        super().__init__(decisiveness_turns=decisiveness_turns)
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
        # 1. Terminal rewards: the zero-sum outcome
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
