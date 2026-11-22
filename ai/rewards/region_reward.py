"""Per-region exploiters (owner, 2026-11-19; design with the plans session): an exploiter paid mostly for the VP it scores
from one region, during the game and at final scoring, with a small bonus for any win.

From the exploiter's side, per game, with ``x = clip(net region VP / scale, floor, cap)`` -- the net VP from every
scoring of the region, whoever plays the card, plus the region's share of final scoring if the game reaches it; ``scale``
is the region's 90th-percentile net in the main's own self-play in that seat, so an ordinary game has x near 0 and a
top-decile one x = 1:

* a win pays ``win_bonus + weight * x`` -- for Europe, a win by Europe Control at least 1 (owner, 2026-11-20);
* a loss pays ``-1 + loss_share * weight * x``; a draw ``loss_share * weight * x``.

Owner, 2026-11-19: "pay it a bit for this region when it loses, but mostly only when it wins. It should still prioritize
win over loss with region, but should heavily prioritize win with region over just win." The defaults (weight 0.8,
win_bonus 0.2, loss_share 0.2, x in [-1, 2]) make a top-decile win worth 1.0 against 0.2 for a plain one, and every win
worth more than every loss (worst win -0.6, best loss -0.68); the constructor refuses settings that break that order.
The ``loss_share`` part is paid densely, at the step each scoring happens (as the change in the clipped total), the rest
at the end of a won game, so the total is exactly the above. Written zero-sum from the US's side, as the other targeted rewards, so GAE's
alternating signs hold (the fixed opponent does not train).

The region's scorings are read exactly: the batch runner records, per env and card, the VP each Event moved during a
step, across all of that step's micro-steps (``set_capture_tracks`` / ``get_step_track_vp``, owner's approval of the
bindings change 2026-11-19), so a headline scoring, one played by a forced move, and either side's are all seen with
the VP the engine actually gave. Final scoring's share is read on the finished game's state (final scoring adds VP;
the board is the one it scored).

Europe Control (E9) reads ``EUROPE_CONTROL_VP`` = 100 wherever it is decided, an amount that clips to the cap or the floor
at every scale in use: in game the engine books it as the Europe scoring card's Event VP (``CardHandlers::recorded``);
at final scoring it goes through no Event, so ``final_scoring_vp_us`` books it here (audit V-8, 2026-11-22: it read 0, a
Europe Control win at final scoring paying ``control_win`` instead of the cap). Final scoring stops at Europe Control, so
no other region has a final share in that game. So on E9 a Europe exploiter's win by Europe Control pays
``win_bonus + weight * cap`` (1.8 at the defaults) and its loss by it ``-1 + loss_share * weight * floor`` (-1.16);
``control_win`` (1) binds only below a cap of 1.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import ts_engine as ts

#: Scoring card -> the region family it pays (Southeast Asia folds into Asia).
SCORING_CARDS: Dict[int, str] = {1: "asia", 2: "europe", 3: "middle_east", 37: "central_america", 38: "asia",
                                 79: "africa", 81: "south_america"}
REGIONS: Dict[str, "ts.Region"] = {
    "europe": ts.Region.EUROPE, "asia": ts.Region.ASIA, "middle_east": ts.Region.MIDDLE_EAST,
    "africa": ts.Region.AFRICA, "central_america": ts.Region.CENTRAL_AMERICA, "south_america": ts.Region.SOUTH_AMERICA}


#: Europe Control's VP as every engine record of it reads (``effect_bits::EUROPE_CONTROL_VP``, E9; not exported by the
#: bindings): the in-game record books it for the Europe scoring card, and final scoring's is booked here. A test holds it
#: to the engine's own in-game record (tests/training/test_region_reward.py).
EUROPE_CONTROL_VP = 100


def final_share_us(state: "ts.GameState", region: str) -> float:
    """The region's part of final scoring (+ US) on a finished game's board, when final scoring counted the regions."""
    return float(ts.Scoring.evaluate_region(state, REGIONS[region], True).net_delta)


def reached_final_scoring(state: "ts.GameState") -> bool:
    """Final scoring counted the regions (it ran and nobody controlled Europe)."""
    return str(ts.game_ending_reason(state)) == "final scoring"


def europe_control_at_final_scoring(state: "ts.GameState") -> bool:
    """The game was won by Europe Control at final scoring, not in game: final scoring runs only after the last turn's
    end incremented the turn past 10 (``finish_end_turn``), and a Europe scoring card's Event ends the game in its own
    turn, at most 10."""
    return str(ts.game_ending_reason(state)) == "Europe Control" and int(state.turn) > 10


def final_scoring_vp_us(state: "ts.GameState", region: str) -> float:
    """The region's VP (+ US) from final scoring on a finished game: its share when final scoring counted the regions,
    +/-``EUROPE_CONTROL_VP`` for Europe when final scoring ended at Europe Control (0 for every other region, which it
    never scored), 0 when the game ended before final scoring."""
    if europe_control_at_final_scoring(state):
        if region != "europe":
            return 0.0
        return float(EUROPE_CONTROL_VP if int(state.victory_points) > 0 else -EUROPE_CONTROL_VP)
    if reached_final_scoring(state):
        return final_share_us(state, region)
    return 0.0


class RegionTargetedReward:
    """The per-region exploiter's reward (module docstring). The trainer `attach`es the env's runner, whose per-step
    Event VP record `compute_step_rewards` reads (before the env resets finished games)."""

    terminal_only: bool = False   # the region part is paid densely, at each scoring

    def __init__(self, region: str, learner_side: int, scale: float, num_envs: int, weight: float = 0.8,
                 win_bonus: float = 0.2, loss_share: float = 0.2, floor: float = -1.0, cap: float = 2.0,
                 scale_ussr: Optional[float] = None, control_win: float = 1.0) -> None:
        """`learner_side` 1 / -1 locks the exploiter's seat; 0 plays both, each game's seat read from the opponent pool
        (`attach_sides`). `scale` is the US seat's scale (or the locked seat's), `scale_ussr` the USSR seat's when both
        are played (owner, 2026-11-19: one exploiter per region, both sides). `control_win`: a win by the region's own
        control ending (Europe Control, the only one) pays at least this (owner, 2026-11-20: "the full 1")."""
        if region not in REGIONS:
            raise ValueError(f"--target-region must be one of {sorted(REGIONS)}, got {region!r}")
        if learner_side not in (1, -1, 0):
            raise ValueError(f"learner_side must be 1, -1 or 0 (both seats), got {learner_side}")
        if learner_side == 0 and scale_ussr is None:
            raise ValueError("a both-seat region exploiter needs each seat's scale (--region-scale and --region-scale-ussr)")
        if not scale > 0.0 or not (scale_ussr is None or scale_ussr > 0.0):
            raise ValueError(f"--region-scale / --region-scale-ussr must be > 0, got {scale}, {scale_ussr}")
        if not 0.0 <= loss_share <= 1.0 or not floor < 0.0 < cap:
            raise ValueError(f"--region-loss-share must be in [0, 1] and --region-floor < 0 < --region-cap, got "
                             f"{loss_share}, {floor}, {cap}")
        if not win_bonus + weight * floor > -1.0 + loss_share * weight * cap:
            raise ValueError(f"the worst win ({win_bonus + weight * floor:+.2f}) must pay more than the best loss "
                             f"({-1.0 + loss_share * weight * cap:+.2f}): lower --region-weight, --region-cap or "
                             f"--region-loss-share, or raise --region-floor or --region-win-bonus")
        self.region, self.learner_side, self.scale = region, int(learner_side), float(scale)
        # Per seat (+1 US, -1 USSR): the locked seat's scale, or each seat's when both are played.
        self.scale_of = {1: float(scale), -1: float(scale if scale_ussr is None else scale_ussr)}
        if learner_side == -1:
            self.scale_of[-1] = float(scale)
        self.num_envs = int(num_envs)
        self.sides_of: Optional[Callable[[], np.ndarray]] = None
        self.weight, self.win_bonus = float(weight), float(win_bonus)
        self.loss_share, self.floor, self.cap = float(loss_share), float(floor), float(cap)
        self.control_win = float(control_win)
        self.control_wins = 0
        self.pending_us = np.zeros(num_envs, dtype=np.float64)     # this step's region scoring VP (+ US), per env
        self.scored_vp = np.zeros(num_envs, dtype=np.float64)      # the game's region VP so far, exploiter's side
        self.finished: Dict[int, List[float]] = {1: [], -1: []}    # region VP of finished games, exploiter's side, by seat
        self.scorings = 0
        self.cards = [c for c, r in SCORING_CARDS.items() if r == region]
        self.runner: Optional["ts.VectorizedBatchRunner"] = None

    def attach(self, runner: "ts.VectorizedBatchRunner") -> None:
        """Read `runner`'s per-step Event VP record from now on (turns its capture on)."""
        runner.set_capture_tracks(True)
        self.runner = runner

    def attach_sides(self, sides_of: Callable[[], np.ndarray]) -> None:
        """Both seats: `sides_of()` is each env's exploiter seat this game (the opponent pool's `learner_side`, redrawn
        only after the env's reward for a finished game is computed)."""
        self.sides_of = sides_of

    def _sides(self) -> np.ndarray:
        if self.learner_side != 0:
            return np.full(self.num_envs, self.learner_side, dtype=np.int64)
        if self.sides_of is None:
            raise RuntimeError("a both-seat region exploiter reads each game's seat from the opponent pool: attach_sides first")
        return np.asarray(self.sides_of(), dtype=np.int64)

    def _step_vp_us(self) -> np.ndarray:
        """This step's VP (+ US) from the region's scoring cards, per env."""
        if self.runner is None:
            raise RuntimeError("a region-targeted reward reads the runner's Event record: attach(env.runner) first")
        rec = np.asarray(self.runner.get_step_track_vp())
        vp = rec[:, self.cards].sum(axis=1).astype(np.float64)
        self.scorings += int((rec[:, self.cards] != 0).sum())
        return vp

    def _x(self, vp: float, side: int) -> float:
        """The clipped region term of `vp` net VP, exploiter's side, in seat `side`."""
        return float(np.clip(vp / self.scale_of[side], self.floor, self.cap))

    def _add(self, e: int, vp_us: float, side: int) -> float:
        """Add `vp_us` (+ US) to env `e`'s region total; the dense (loss_share) payment it earns, exploiter's side."""
        before = self._x(float(self.scored_vp[e]), side)
        self.scored_vp[e] += vp_us * side
        return self.loss_share * self.weight * (self._x(float(self.scored_vp[e]), side) - before)

    def compute_step_rewards(self, acting_players: np.ndarray, dones: np.ndarray, terminal_utilities: np.ndarray,
                             prev_victory_points: np.ndarray, curr_victory_points: np.ndarray,
                             states: Optional[List[Optional["ts.GameState"]]] = None) -> np.ndarray:
        u_learner = np.zeros(len(dones), dtype=np.float64)   # exploiter's side, this step
        sides = self._sides()
        if self.runner is not None:
            self.pending_us[:] = self._step_vp_us()
        for e in np.flatnonzero(self.pending_us != 0.0):
            u_learner[e] += self._add(int(e), float(self.pending_us[e]), int(sides[e]))
        for e in np.flatnonzero(dones):
            st = states[int(e)] if states is not None else None
            if st is None:
                raise ValueError("a region-targeted reward needs the finished game's state")
            side = int(sides[e])
            u = float(terminal_utilities[e])
            winner = 1 if u > 0 else -1 if u < 0 else 0
            final_us = final_scoring_vp_us(st, self.region)
            if final_us != 0.0:
                u_learner[e] += self._add(int(e), final_us, side)
            if winner == side:
                x = self._x(float(self.scored_vp[e]), side)
                pay = self.win_bonus + self.weight * x
                if self.region == "europe" and str(ts.game_ending_reason(st)) == "Europe Control":
                    # Owner, 2026-11-20: a win by controlling Europe pays at least the full 1 (the dense loss share of
                    # x already paid counts towards it).
                    pay = max(pay, self.control_win)
                    self.control_wins += 1
                u_learner[e] += pay - self.loss_share * self.weight * x
            elif winner == -side:
                u_learner[e] += -1.0
            self.finished[side].append(float(self.scored_vp[e]))
            self.scored_vp[e] = 0.0
        self.pending_us[:] = 0.0
        # Zero-sum from the US's side: u_US = u_learner * the exploiter's seat, the mover's reward u_US * mover.
        return (u_learner * sides * np.asarray(acting_players, dtype=np.float64)).astype(np.float32)

    def compute_zero_sum_sign(self, curr_players: torch.Tensor, next_players: torch.Tensor) -> torch.Tensor:
        return (curr_players * next_players).float()

    def metrics(self) -> Dict[str, float]:
        out = {"region/scorings": float(self.scorings)}   # scoring Events with a VP effect
        if self.region == "europe":
            out["region/control_wins"] = float(self.control_wins)
            self.control_wins = 0
        both = self.finished[1] + self.finished[-1]
        if both:
            out["region/vp_per_game"] = float(np.mean(both))
        for side, name in ((1, "us"), (-1, "ussr")):
            if self.finished[side] and self.learner_side == 0:
                out[f"region/vp_per_game_{name}"] = float(np.mean(self.finished[side]))
        self.finished = {1: [], -1: []}
        self.scorings = 0
        return out
