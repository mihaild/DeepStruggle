"""Per-region exploiters (owner, 2026-11-19; design with the plans session): an exploiter paid mostly for the VP it scores
from one region, during the game and at final scoring, with a small bonus for any win.

From the exploiter's side, per game:

* the region part: ``weight * (net VP from the region's scorings) / scale`` -- every scoring of the region, whoever plays
  the card, signed for the exploiter; plus the region's share of final scoring if the game reaches it. ``scale`` is the
  region's 90th-percentile net in the main's own self-play in that seat, so an ordinary game earns about 0 and a
  top-decile one about ``weight``;
* the outcome: ``win_bonus`` for any win, -1 for any loss, 0 for a draw.

``pay="always"`` pays the region part at the step it happens in every game (dense, signed); ``pay="won"`` holds it and
pays it at the end only if the exploiter won. Written zero-sum from the US's side, as the other targeted rewards, so GAE's
alternating signs hold (the fixed opponent does not train).

The region's scorings are read exactly: the batch runner records, per env and card, the VP each Event moved during a
step, across all of that step's micro-steps (``set_capture_tracks`` / ``get_step_track_vp``, owner's approval of the
bindings change 2026-11-19), so a headline scoring, one played by a forced move, and either side's are all seen with
the VP the engine actually gave. Final scoring's share is read on the finished game's state (final scoring adds VP;
the board is the one it scored).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import ts_engine as ts

#: Scoring card -> the region family it pays (Southeast Asia folds into Asia).
SCORING_CARDS: Dict[int, str] = {1: "asia", 2: "europe", 3: "middle_east", 37: "central_america", 38: "asia",
                                 79: "africa", 81: "south_america"}
REGIONS: Dict[str, "ts.Region"] = {
    "europe": ts.Region.EUROPE, "asia": ts.Region.ASIA, "middle_east": ts.Region.MIDDLE_EAST,
    "africa": ts.Region.AFRICA, "central_america": ts.Region.CENTRAL_AMERICA, "south_america": ts.Region.SOUTH_AMERICA}


def final_share_us(state: "ts.GameState", region: str) -> float:
    """The region's part of final scoring (+ US) on a finished game's board."""
    return float(ts.Scoring.evaluate_region(state, REGIONS[region], True).net_delta)


def reached_final_scoring(state: "ts.GameState") -> bool:
    return str(ts.game_ending_reason(state)) == "final scoring"


class RegionTargetedReward:
    """The per-region exploiter's reward (module docstring). The trainer `attach`es the env's runner, whose per-step
    Event VP record `compute_step_rewards` reads (before the env resets finished games)."""

    def __init__(self, region: str, learner_side: int, scale: float, num_envs: int, weight: float = 0.8,
                 win_bonus: float = 0.2, pay: str = "always") -> None:
        if region not in REGIONS:
            raise ValueError(f"--target-region must be one of {sorted(REGIONS)}, got {region!r}")
        if learner_side not in (1, -1):
            raise ValueError("--target-region needs the learner's seat locked (--opponent-lock-side us|ussr)")
        if not scale > 0.0:
            raise ValueError(f"--region-scale must be > 0, got {scale}")
        if pay not in ("always", "won"):
            raise ValueError(f"--region-pay must be always or won, got {pay!r}")
        self.region, self.learner_side, self.scale = region, int(learner_side), float(scale)
        self.weight, self.win_bonus, self.pay = float(weight), float(win_bonus), pay
        self.pending_us = np.zeros(num_envs, dtype=np.float64)     # this step's region scoring VP (+ US), per env
        self.held = np.zeros(num_envs, dtype=np.float64)           # pay="won": the region part so far, exploiter's side
        self.scored_vp = np.zeros(num_envs, dtype=np.float64)      # the game's region VP so far, exploiter's side
        self.finished: List[float] = []                            # region VP of finished games, exploiter's side
        self.scorings = 0
        self.cards = [c for c, r in SCORING_CARDS.items() if r == region]
        self.runner: Optional["ts.VectorizedBatchRunner"] = None

    def attach(self, runner: "ts.VectorizedBatchRunner") -> None:
        """Read `runner`'s per-step Event VP record from now on (turns its capture on)."""
        runner.set_capture_tracks(True)
        self.runner = runner

    def _step_vp_us(self) -> np.ndarray:
        """This step's VP (+ US) from the region's scoring cards, per env."""
        if self.runner is None:
            raise RuntimeError("a region-targeted reward reads the runner's Event record: attach(env.runner) first")
        rec = np.asarray(self.runner.get_step_track_vp())
        vp = rec[:, self.cards].sum(axis=1).astype(np.float64)
        self.scorings += int((rec[:, self.cards] != 0).sum())
        return vp

    def _part(self, vp_us: float) -> float:
        """The region part, exploiter's side, of `vp_us` VP (+ US)."""
        return self.weight * vp_us * self.learner_side / self.scale

    def compute_step_rewards(self, acting_players: np.ndarray, dones: np.ndarray, terminal_utilities: np.ndarray,
                             prev_victory_points: np.ndarray, curr_victory_points: np.ndarray,
                             states: Optional[List[Optional["ts.GameState"]]] = None) -> np.ndarray:
        u_learner = np.zeros(len(dones), dtype=np.float64)   # exploiter's side, this step
        if self.runner is not None:
            self.pending_us[:] = self._step_vp_us()
        for e in np.flatnonzero(self.pending_us != 0.0):
            part = self._part(float(self.pending_us[e]))
            self.scored_vp[e] += self.pending_us[e] * self.learner_side
            if self.pay == "always":
                u_learner[e] += part
            else:
                self.held[e] += part
        for e in np.flatnonzero(dones):
            st = states[int(e)] if states is not None else None
            if st is None:
                raise ValueError("a region-targeted reward needs the finished game's state")
            u = float(terminal_utilities[e])
            winner = 1 if u > 0 else -1 if u < 0 else 0
            final = final_share_us(st, self.region) if reached_final_scoring(st) else 0.0
            self.scored_vp[e] += final * self.learner_side
            region_part = self._part(final)
            if winner == self.learner_side:
                u_learner[e] += self.win_bonus + region_part + (self.held[e] if self.pay == "won" else 0.0)
            elif winner == -self.learner_side:
                u_learner[e] += -1.0 + (region_part if self.pay == "always" else 0.0)
            elif self.pay == "always":
                u_learner[e] += region_part
            self.finished.append(float(self.scored_vp[e]))
            self.held[e] = 0.0
            self.scored_vp[e] = 0.0
        self.pending_us[:] = 0.0
        # Zero-sum from the US's side: u_US = u_learner * learner_side, the mover's reward u_US * mover.
        return (u_learner * self.learner_side * np.asarray(acting_players, dtype=np.float64)).astype(np.float32)

    def compute_zero_sum_sign(self, curr_players: torch.Tensor, next_players: torch.Tensor) -> torch.Tensor:
        return (curr_players * next_players).float()

    def metrics(self) -> Dict[str, float]:
        out = {"region/scorings": float(self.scorings)}   # scoring Events with a VP effect
        if self.finished:
            out["region/vp_per_game"] = float(np.mean(self.finished))
        self.finished = []
        self.scorings = 0
        return out
