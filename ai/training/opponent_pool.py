"""Frozen past policies to play a share of the rollout environments against.

The problem this exists for: once one side finds a strategy the other has not answered, outcomes
stop depending on what the trailing side does. The critic then has nothing to discriminate, it
degenerates toward the base rate, and advantages vanish for *both* sides -- measured on E3-17-22
and reproduced independently on E3-15-22. Nothing that re-weights, filters or rescales the
existing signal escapes that, because there is no signal left to rescale.

Opponent *diversity* does escape it, and it is worth being precise about why, because opponent
*prioritisation* does not. Sampling opponents you lose to is pointless in pure self-play: the
current policy already faces the strongest opponent there is, namely itself. Diversity is the
different claim -- against an 80M snapshot the trailing side has games it can actually win, so the
terminal outcome separates its actions again and there is something for the advantage to be
non-zero about.

**`--opponent-pfsp` exists to test that paragraph, and may well refute the flag rather than the
paragraph.** Prioritised fictitious self-play weights the draw by how the learner is doing against
each snapshot. The argument above predicts it should *hurt*: the weak old snapshots that PFSP
deprioritises are exactly the ones supplying winnable games to the learner's trailing side. The
counter-argument is that a snapshot beaten 99% of the time supplies no signal to either side and
its games are wasted. Nobody has measured it, so the weighting defaults to `var` -- x(1-x), which
peaks on *evenly matched* opponents rather than on hard ones -- and `pfsp_uniform_mix` keeps a
floor of uniform probability under every pool member so that prioritisation cannot delete the
diversity the mechanism is built on. Watch `opp_pfsp_entropy`: if it collapses toward 0 the pool
has become all-recent and this is no longer testing prioritisation, it is testing a smaller pool.

**One opponent per rollout iteration.** The pool may hold many snapshots, but a single one is
sampled for each iteration and used by every mixed environment in it. That keeps the step to
exactly two forward passes -- learner and opponent, on complementary subsets of the batch, the
same total FLOPs as the single pass it replaces. Diversity accumulates across iterations instead
of within them. Sampling several per iteration would work too, at more kernel launches on smaller
batches.

**Which means a game is played against about five different snapshots.** An episode does not fit
in an iteration: measured on E3-20-28 the mean episode is 626.5 micro-actions against a
`buffer_size` of 128, so one game faces **4.89** opponents in sequence -- a swap roughly every 1.4
turns, and a random ~40% subset of a capacity-12 pool per game. The learner therefore never meets
a coherent adversary over a whole game; it meets a composite that exists as no policy. That is a
consequence of the two facts above sitting together, it was unremarked until 2026-09-16, and it is
why results have to be attributed by exposure rather than to "the" opponent (`on_episode_end`).
Whether it helps or hurts is untested: it costs coherence and buys within-episode diversity, and
diversity is the mechanism this docstring credits. See
`research/archive/E3_ladder/findings/pooling.md` §3b.

**A growing pool, kept spread across the run.** The first version took a fixed list of
checkpoints, which measured well (E3-19-22 held self-play imbalance to 9.0pp where its control
reached 71.7pp) but is not a usable recipe: it required eight snapshots of an already-finished
run. A real run has to pool against its own history as that history appears.

Eviction is by *spacing*, not by age. Dropping the oldest would let the pool become all-recent,
and then every opponent carries whatever strategy the run has converged on -- the trailing side
loses the winnable games that made the mechanism work in the first place, and the window quietly
closes. Instead the first and most recent snapshots are always kept and the interior point with
the smallest neighbouring gap is evicted, so the pool stays spread over the whole run.

**League members (P24).** Members added with `group="league"` come from *other* runs -- a main
exploiter's snapshots for the main agent, the main agent's newest snapshots for the exploiter. They
are kept apart from the run's own history: their step counts are another run's, so spacing them
against the run's own snapshots would mean nothing. The league group holds `league_capacity`
members and evicts the oldest-added first, because a league member is only as relevant as the
policy it was trained against is recent. With `league_frac` set, each iteration's opponent comes
from the league group with that probability, whatever the two groups' sizes. Otherwise every member
is drawn alike. The league group is left out of `state_dict`: the trainer re-scans its directories
on resume. A pool without league members runs exactly the code paths it ran before.

**A league member has a seat, and a league game keeps its member (P45, owner 2026-11-12).** Until then
the learner's seat was drawn 50/50 whoever the opponent was, so a seat-locked exploiter played its
untrained seat in half its games, and the league member was the iteration's opponent like any other,
so a game met about five opponents and no member played one from start to finish. Now a member's
seat is the one its run was locked to (`seats`: +1 US, -1 USSR, None either), read by the trainer
from that run's metadata, and league games are assigned per episode: when a mixed environment starts
a game the learner's seat is drawn as before, then with probability `league_frac` (1 for a league-
only pool; the league's share of members when `league_frac` is None) the game goes to a member of the
opposite seat, drawn alike among them, which plays it to the end (`env_member`). With no member for
that seat the game is an ordinary own-pool game. Own-pool games keep the one-opponent-per-iteration
draw, from the run's own members only, and consume the pool's random stream exactly as before; the
league's draws use a stream of their own, so a pool without league members, or with `league_frac`
0, plays the same sequence it did before this change. A member evicted mid-game hands its games to
the own pool for the rest of the episode (`opp_league_orphaned_games`).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.ts_env import model_obs_features
from bindings.ts_env import check_obs_width
from tools.lib.player_agent import NeuralAgent


#: P47: own-pool eviction rules (OpponentPool._eviction_victim).
EVICTION_MODES: Tuple[str, ...] = ("spread", "recent", "unpinned")


class OpponentPool:
    """A set of frozen policies, plus the per-environment assignment of who plays whom.

    `frac` of environments are *mixed*: the learner plays one side and a frozen snapshot the
    other. The rest are ordinary self-play, where the learner plays both sides.
    """

    def __init__(self, nets: Sequence[Any], num_envs: int, frac: float = 0.25,
                 seed: int = 0, lock_learner_side: Optional[int] = None,
                 capacity: int = 12, pfsp: bool = False, pfsp_weighting: str = "var",
                 pfsp_uniform_mix: float = 0.25, pfsp_prior: float = 4.0,
                 merged: Optional[Sequence[bool]] = None,
                 macro: Optional[Sequence[bool]] = None,
                 seat_balance: bool = False, seat_balance_max_frac: float = 0.8,
                 groups: Optional[Sequence[str]] = None, league_capacity: int = 0,
                 league_frac: Optional[float] = None, obs_features: Optional[int] = None,
                 seats: Optional[Sequence[Optional[int]]] = None, per_game_own: bool = False,
                 prior_window: int = 0, prior_turn_weight: Tuple[float, float] = (1.0, 1.0),
                 eviction: str = "spread", opponent_obs_features: Optional[int] = None) -> None:
        if not nets:
            raise ValueError("OpponentPool needs at least one frozen network")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        if pfsp_weighting not in ("var", "hard"):
            raise ValueError(f"pfsp_weighting must be 'var' or 'hard', got {pfsp_weighting!r}")
        if not 0.0 <= pfsp_uniform_mix < 1.0:
            raise ValueError(f"pfsp_uniform_mix must be in [0, 1), got {pfsp_uniform_mix}")
        if pfsp_prior <= 0.0:
            raise ValueError(f"pfsp_prior must be positive, got {pfsp_prior}")
        if league_frac is not None and not 0.0 <= league_frac <= 1.0:
            raise ValueError(f"league_frac must be in [0, 1], got {league_frac}")
        if prior_window < 0:
            raise ValueError(f"prior_window must be >= 0, got {prior_window}")
        if eviction not in EVICTION_MODES:
            raise ValueError(f"eviction must be one of {EVICTION_MODES}, got {eviction!r}")
        if prior_window and not per_game_own:
            raise ValueError("a critic prior per opponent (prior_window) needs one opponent per game (per_game_own)")
        if groups is not None:
            if len(groups) != len(nets):
                raise ValueError("groups must give one group per seed net")
            bad = sorted({g for g in groups if g not in ("self", "league")})
            if bad:
                raise ValueError(f"unknown pool group(s) {bad}; expected 'self' or 'league'")
            if "league" in groups and league_capacity < 1:
                raise ValueError("league members need league_capacity >= 1")
        self.nets = list(nets)
        # Rollouts use learner-view observations for both the policy and critic. Equal widths
        # alone do not imply equal semantics (HEADLINE_REVEAL changes existing slots).
        self.obs_features = model_obs_features(self.nets[0]) if obs_features is None else obs_features
        #: The feature set the frozen side is seated with. Normally the learner's. A fixed opponent may differ from
        #: the learner in rule bits only (ts.OBS_FEATURES_RULES: no width, no observation content), so its rows read
        #: the learner's observation unchanged while its seat plays its own rules -- as in a tournament. Only for a
        #: fixed opponent in every environment with the learner's side locked, so each seat's rules never change.
        self.opponent_obs_features = self.obs_features if opponent_obs_features is None else int(opponent_obs_features)
        if self.opponent_obs_features != self.obs_features:
            diff = self.opponent_obs_features ^ self.obs_features
            if diff & ~int(ts.OBS_FEATURES_RULES):
                raise ValueError(f"opponent observation features {self.opponent_obs_features:#x} differ from the "
                                 f"learner's {self.obs_features:#x} in observation bits, not only rule bits")
            if lock_learner_side is None or frac < 1.0:
                raise ValueError("an opponent with its own rule bits needs the learner's side locked and every "
                                 "environment mixed (--opponent-lock-side, --opponent-frac 1.0)")
        for net in self.nets:
            self._check_obs_features(net)
        self.capacity = capacity
        #: P47: which own-pool member goes when the pool is over capacity (EVICTION_MODES).
        self.eviction = eviction
        self.pfsp = bool(pfsp)
        self.pfsp_weighting = pfsp_weighting
        self.pfsp_uniform_mix = float(pfsp_uniform_mix)
        #: Beta prior on each opponent's win rate, in pseudo-games a side. Large enough that an
        #: opponent with two games does not dominate the draw on noise.
        self.pfsp_prior = float(pfsp_prior)
        #: A stable id per pool member. Indices shift under eviction, so results are keyed by id.
        self._next_id = 0
        self.ids: List[int] = []
        #: Weighted learner wins and games against each opponent id. Weighted, because an episode
        #: spans several iterations and therefore several opponents -- see `on_episode_end`.
        self.wins: Dict[int, float] = {}
        self.games: Dict[int, float] = {}
        #: Per environment, how many iterations of the current episode each opponent id played.
        self._exposure: List[Dict[int, float]] = [{} for _ in range(num_envs)]
        #: Step count each net was captured at, parallel to self.nets. Seeds get 0 so that a
        #: pool started from the initial policy keeps that policy as its earliest point.
        self.steps: List[int] = [0] * len(self.nets)
        #: Snapshot file each net was loaded from, parallel to self.nets; "" when it has none (the
        #: initial-policy seed). Recorded so a resume can find members written by an EARLIER leg
        #: of the lineage, which live in that leg's run directory, not the one being resumed.
        self.paths: List[str] = [""] * len(self.nets)
        #: P23 / E4.1: whether each member decides in the merged-influence view, parallel to
        #: self.nets. A member trained in the E4 view must keep being offered E4 masks, or it
        #: plays its untrained composed actions and becomes an artificially weak opponent.
        if merged is not None and len(merged) != len(self.nets):
            raise ValueError("merged must give one flag per seed net")
        self.merged: List[bool] = [bool(x) for x in merged] if merged is not None else [False] * len(self.nets)
        #: P42: whether each member decides in the macro view (the until-control slots), parallel to self.nets.
        if macro is not None and len(macro) != len(self.nets):
            raise ValueError("macro must give one flag per seed net")
        self.macro: List[bool] = [bool(x) for x in macro] if macro is not None else [False] * len(self.nets)
        if any(a and b for a, b in zip(self.merged, self.macro)):
            raise ValueError("a pool member cannot decide in both the merged and the macro view")
        #: "self" for the run's own history, "league" for another run's snapshots (P24).
        self.groups: List[str] = [str(g) for g in groups] if groups is not None else ["self"] * len(self.nets)
        self.league_capacity = int(league_capacity)
        #: Probability that an iteration's opponent is drawn from the league group, when both
        #: groups have members. None draws every member alike.
        self.league_frac: Optional[float] = None if league_frac is None else float(league_frac)
        #: P45: the seat each member was trained in (+1 US, -1 USSR), None when it may take either,
        #: parallel to self.nets. Only league members are ever seated by it.
        if seats is not None and len(seats) != len(self.nets):
            raise ValueError("seats must give one seat per seed net")
        self.seats: List[Optional[int]] = [self._check_seat(x) for x in seats] if seats is not None \
            else [None] * len(self.nets)
        #: The league's own random stream: per-episode league draws never touch `self.rng`, so the
        #: own pool's sequence is unchanged whatever the league does.
        self.league_rng = random.Random((int(seed) * 1_000_003) ^ 0x1EA6)
        #: P45's L6 (owner, 2026-11-14): own-pool members held for a whole game, drawn at the game's start from
        #: their own stream, as league members are; off, the own pool keeps one opponent per iteration.
        self.per_game_own = bool(per_game_own)
        self.game_rng = random.Random((int(seed) * 7_919) ^ 0x46B)
        self.own_orphaned_games = 0
        #: P46b: the learner's running score against each member, per seat (an exponential average over an
        #: effective window of `prior_window` games), and self-play's running US score; the critic's prior in a
        #: pool game is beta = logit(score against that member in that seat) - logit(self-play score in it),
        #: weighted by the turn (`prior_turn_weight`: the weight at turn 1 and at turn 10, linear between).
        self.prior_window = int(prior_window)
        self.prior_alpha = 2.0 / (self.prior_window + 1.0) if self.prior_window else 0.0
        self.prior_turn_weight = (float(prior_turn_weight[0]), float(prior_turn_weight[1]))
        self.prior_rate: Dict[tuple, float] = {}
        self.prior_sp_us = 0.5
        #: Per environment, the id of the league member playing its current game; -1 for an own-pool
        #: game (the iteration's opponent) or self-play.
        self.env_member = np.full(num_envs, -1, dtype=np.int64)
        #: League games completed, by (member id, the learner's seat); and games in which a member
        #: sat in a seat other than its own (must stay 0); and games handed back on eviction.
        self.league_seat_games: Dict[tuple, int] = {}
        self.league_wrong_seat_games = 0
        self.league_orphaned_games = 0
        for n in self.nets:
            n.eval()
            for p in n.parameters():
                p.requires_grad_(False)
        self.num_envs = num_envs
        self.frac = frac
        self.rng = random.Random(seed)
        for _ in self.nets:
            self._register()
        #: None to alternate per episode; +1 or -1 to pin the learner to one side throughout,
        #: which is what the single-frozen-opponent experiment wants.
        self.lock_learner_side = lock_learner_side

        #: Seat balancing (--seat-balance). A side collapse is a loop with no signal in it: the
        #: seat that falls behind loses almost every self-play game, its advantages flatten, the
        #: entropy bonus acts alone and it falls further behind. Under balancing, the further the
        #: self-play win rate drifts from even, the more games the WEAK seat plays -- as the learner,
        #: in mixed envs, against pool members it beats about half the time -- so it keeps getting
        #: winnable games and therefore a gradient. Off by default: everything below is inert.
        self.seat_balance = bool(seat_balance)
        self.seat_balance_max_frac = max(float(frac), float(seat_balance_max_frac))
        #: Moving average of the learner's self-play US win rate; 0.5 is even.
        self.sp_us = 0.5
        #: Learner wins and games against each opponent id, per learner side (+1 US, -1 USSR).
        self.side_wins: Dict[tuple, float] = {}
        self.side_games: Dict[tuple, float] = {}

        n_mixed = int(round(num_envs * frac))
        #: True where the learner faces a frozen opponent rather than itself.
        self.is_mixed = np.zeros(num_envs, dtype=bool)
        self.is_mixed[:n_mixed] = True
        #: Which side the learner plays in a mixed env (+1 US, -1 USSR); unused elsewhere.
        self.learner_side = np.ones(num_envs, dtype=np.int8)
        for i in range(num_envs):
            self.learner_side[i] = self._draw_side()
        for i in range(num_envs):
            self._assign_league(i)
        self.current: Any = self.nets[0]
        self.current_id: int = self.ids[0]
        self.current_merged: bool = self.merged[0]
        self.current_macro: bool = self.macro[0]

    @property
    def current_view(self) -> int:
        """The action view of the current opponent: 0 E4, 1 merged (E4.1), 2 macro (P42)."""
        return 1 if self.current_merged else 2 if self.current_macro else 0

    @staticmethod
    def _check_seat(seat: Optional[int]) -> Optional[int]:
        if seat not in (None, 1, -1):
            raise ValueError(f"a member's seat is +1 (US), -1 (USSR) or None, got {seat!r}")
        return None if seat is None else int(seat)

    def _index_of(self, oid: int) -> int:
        return self.ids.index(int(oid))

    def _view_at(self, j: int) -> int:
        return 1 if self.merged[j] else 2 if self.macro[j] else 0

    def view_of(self, env_index: int) -> int:
        """The action view of whoever plays the learner's opponent in one env: its league member's,
        or the iteration's opponent's."""
        oid = int(self.env_member[env_index])
        return self.current_view if oid < 0 else self._view_at(self._index_of(oid))

    def views_in_use(self) -> List[int]:
        """The distinct opponent action views across the mixed environments this iteration."""
        views = {self.current_view} if self.is_mixed.any() else set()
        for oid in np.unique(self.env_member[self.is_mixed & (self.env_member >= 0)]):
            views.add(self._view_at(self._index_of(int(oid))))
        return sorted(views)

    def opponent_rows(self, learner_np: np.ndarray) -> List[tuple]:
        """[(network, env indices)] for the rows the frozen side moves in: the iteration's opponent
        on its own-pool rows, then each league member on the rows of its own games."""
        rows = np.flatnonzero(~np.asarray(learner_np, dtype=bool))
        if not rows.size:
            return []
        member = self.env_member[rows]
        out: List[tuple] = []
        own = rows[member < 0]
        if own.size:
            out.append((self.current, own))
        for oid in np.unique(member[member >= 0]):
            out.append((self.nets[self._index_of(int(oid))], rows[member == oid]))
        return out

    def _check_obs_features(self, net: Any) -> None:
        actual = model_obs_features(net)
        if actual != self.opponent_obs_features:
            raise ValueError(f"opponent observation features {actual:#x} do not match "
                             f"learner/pool observation features {self.opponent_obs_features:#x}")

    def seat_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """(US, USSR) feature set of each environment's seats: the learner's, and on the frozen side the
        opponent's (which differs only in rule bits, and only under a locked side in every environment)."""
        us = np.full(self.num_envs, self.obs_features, dtype=np.uint32)
        ussr = us.copy()
        if self.opponent_obs_features != self.obs_features:
            frozen_us = self.is_mixed & (self.learner_side == -1)
            frozen_ussr = self.is_mixed & (self.learner_side == 1)
            us[frozen_us] = self.opponent_obs_features
            ussr[frozen_ussr] = self.opponent_obs_features
        return us, ussr

    def _register(self) -> int:
        """Give the newest net a stable id and zeroed statistics."""
        oid = self._next_id
        self._next_id += 1
        self.ids.append(oid)
        self.wins[oid] = 0.0
        self.games[oid] = 0.0
        return oid

    def win_rate(self, oid: int) -> float:
        """The learner's smoothed win rate against one opponent.

        Smoothed toward 0.5 with a Beta prior: an opponent with one game must not capture the
        draw because that game happened to be a win.
        """
        w = self.wins.get(oid, 0.0) + self.pfsp_prior
        n = self.games.get(oid, 0.0) + 2.0 * self.pfsp_prior
        return w / n

    # ---- seat balancing ------------------------------------------------------------------------

    def observe_selfplay(self, us_wins: float, games: float) -> None:
        """Fold one iteration's self-play results into the side-balance average."""
        if games <= 0:
            return
        alpha = min(1.0, games / 400.0)  # a few hundred games per iteration moves it quickly
        self.sp_us = (1.0 - alpha) * self.sp_us + alpha * (us_wins / games)

    def weak_side(self) -> int:
        """The seat the learner is currently losing with in self-play: +1 US, -1 USSR."""
        return 1 if self.sp_us < 0.5 else -1

    def pressure(self) -> float:
        """0 at an even self-play split, rising to 1 at 20/80 or worse."""
        return min(1.0, abs(self.sp_us - 0.5) / 0.3)

    def side_win_rate(self, oid: int, side: int) -> float:
        """The learner's smoothed win rate against one opponent while playing `side`."""
        w = self.side_wins.get((oid, side), 0.0) + self.pfsp_prior
        n = self.side_games.get((oid, side), 0.0) + 2.0 * self.pfsp_prior
        return w / n

    def _draw_mixed(self) -> bool:
        """Whether an env's next episode is learner-vs-pool, under seat balancing."""
        frac = self.frac + (self.seat_balance_max_frac - self.frac) * self.pressure()
        return self.rng.random() < frac

    def sampling_probs(self) -> List[float]:
        """The draw distribution over the pool, in `self.nets` order.

        Uniform unless `pfsp`. `var` weights x(1-x), peaking where the learner is evenly matched;
        `hard` weights (1-x)^2, peaking where the learner is losing. A uniform floor of
        `pfsp_uniform_mix` is mixed in either way, so no pool member can be driven to zero -- the
        pool's value is its spread, and a prioritiser that empties it has defeated the purpose.
        """
        m = len(self.nets)
        if m == 0:
            return []
        if self.seat_balance and self.pressure() > 0.0:
            # Opponents the weak seat beats about half the time: an easy opponent gives it wins
            # to learn from, a hard one losses, and x(1-x) prefers the ones where the outcome is
            # actually in doubt -- the games that carry a gradient. The uniform floor keeps every
            # member in play, as under PFSP.
            weak = self.weak_side()
            raw = [self.side_win_rate(oid, weak) * (1.0 - self.side_win_rate(oid, weak)) for oid in self.ids]
            total = sum(raw)
            u = self.pfsp_uniform_mix
            if total > 0.0:
                return [(1.0 - u) * (r / total) + u / m for r in raw]
            return [1.0 / m] * m
        if not self.pfsp:
            return [1.0 / m] * m
        raw: List[float] = []
        for oid in self.ids:
            x = self.win_rate(oid)
            raw.append(x * (1.0 - x) if self.pfsp_weighting == "var" else (1.0 - x) ** 2)
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / m] * m
        u = self.pfsp_uniform_mix
        return [(1.0 - u) * (r / total) + u / m for r in raw]

    def _draw_side(self) -> int:
        if self.lock_learner_side is not None:
            return int(self.lock_learner_side)
        if self.seat_balance:
            # Up to 90% of mixed episodes on the weak seat at full pressure; 50/50 when even.
            weak = self.weak_side()
            p_weak = 0.5 + 0.4 * self.pressure()
            return weak if self.rng.random() < p_weak else -weak
        return 1 if self.rng.random() < 0.5 else -1

    def add(self, net: Any, steps: int, merged: bool = False, path: str = "",
            group: str = "self", macro: bool = False, seat: Optional[int] = None) -> None:
        """Add a snapshot taken at `steps`, evicting to stay within capacity.

        The net is frozen in place. Callers pass a freshly loaded copy, not the live training
        model -- adding the model under training would give the learner an opponent whose
        weights move with it, which is self-play with extra steps.
        """
        if group not in ("self", "league"):
            raise ValueError(f"unknown pool group {group!r}; expected 'self' or 'league'")
        if group == "league" and self.league_capacity < 1:
            raise ValueError("league members need league_capacity >= 1")
        self._check_obs_features(net)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.nets.append(net)
        self.steps.append(int(steps))
        self.paths.append(str(path))
        self.merged.append(bool(merged))
        self.macro.append(bool(macro))
        self.groups.append(group)
        self.seats.append(self._check_seat(seat))
        self._register()

        while len(self._members("self")) > self.capacity:
            self._remove(self._eviction_victim())
        # Oldest-added first: ids are issued in arrival order. P45: `league_capacity` holds per seat, so a
        # new US exploiter generation evicts the oldest US member and never a USSR one; members without
        # a seat share one group, as every member did before.
        seat = self.seats[-1] if group == "league" else None
        while len([i for i in self._members("league") if self.seats[i] == seat]) > self.league_capacity:
            self._remove(min((i for i in self._members("league") if self.seats[i] == seat),
                             key=lambda i: self.ids[i]))

    def _eviction_victim(self) -> int:
        """The own-pool member to evict, by `self.eviction`:

        * 'spread' (every run before P47): keep the endpoints; drop the interior snapshot whose neighbours are closest
          together, the one carrying the least information about the run's span. The oldest member is never evicted.
        * 'recent': drop the oldest -- the pool is a window of the newest `capacity` members.
        * 'unpinned': 'spread' with the oldest a candidate too, scored by twice the gap to its one neighbour (the span
          its removal loses, on the scale of an interior member's two-sided gap); the newest is still kept.
        """
        order = sorted(self._members("self"), key=lambda i: self.steps[i])
        if self.eviction == "recent" or len(order) < 3:
            return order[0]
        victim = None
        best_gap = None
        if self.eviction == "unpinned":
            best_gap, victim = 2 * (self.steps[order[1]] - self.steps[order[0]]), order[0]
        for pos in range(1, len(order) - 1):
            i = order[pos]
            gap = self.steps[order[pos + 1]] - self.steps[order[pos - 1]]
            if best_gap is None or gap < best_gap:
                best_gap, victim = gap, i
        assert victim is not None
        return victim

    def _members(self, group: str) -> List[int]:
        """Indices, in `self.nets` order, of the members in one group."""
        return [i for i, g in enumerate(self.groups) if g == group]

    def _remove(self, victim: int) -> None:
        group = self.groups[victim]
        self.nets.pop(victim)
        self.steps.pop(victim)
        self.paths.pop(victim)
        self.merged.pop(victim)
        self.macro.pop(victim)
        self.groups.pop(victim)
        self.seats.pop(victim)
        dead = self.ids.pop(victim)
        # A member evicted mid-game: its games go to the iteration's opponent for the rest of the episode.
        orphaned = self.env_member == dead
        if orphaned.any():
            if group == "league":
                self.league_orphaned_games += int(orphaned.sum())
            else:
                self.own_orphaned_games += int(orphaned.sum())
            self.env_member[orphaned] = -1
        self.wins.pop(dead, None)
        self.games.pop(dead, None)
        for key in [k for k in self.prior_rate if k[0] == dead]:
            del self.prior_rate[key]
        # Exposure to an evicted opponent is dropped rather than reattributed: the outcome
        # would otherwise be credited to a snapshot that is no longer in the pool.
        for env in self._exposure:
            env.pop(dead, None)

    def start_iteration(self) -> None:
        """Pick the opponent this iteration's mixed environments will face, and record exposure.

        One opponent per iteration, as before; only the *distribution* changes under PFSP. An
        episode outlives an iteration -- roughly 300-600 micro-actions against 128 -- so a single
        game is played against three to five different snapshots in sequence. That is why results
        are attributed by exposure rather than to one opponent: crediting a whole episode to
        whichever snapshot happened to be current when it ended would be wrong most of the time.
        """
        probs = self.sampling_probs()
        league = self._members("league")
        own = self._members("self")
        if league and own:
            # P45: league games are assigned per episode (`_assign_league`), so the iteration's
            # opponent comes from the run's own members. The draw that used to choose between the
            # groups is still taken, so with --league-frac 0 the own pool's sequence is unchanged.
            if self.league_frac is not None:
                self.rng.random()
            if (self.pfsp or (self.seat_balance and self.pressure() > 0.0)) and probs:
                sub = [probs[i] for i in own]
                total = sum(sub)
                idx = own[self._weighted_index([q / total for q in sub])] if total > 0.0 \
                    else own[self.rng.randrange(len(own))]
            else:
                idx = own[self.rng.randrange(len(own))]
        elif (self.pfsp or (self.seat_balance and self.pressure() > 0.0)) and probs:
            idx = self._weighted_index(probs)
        else:
            idx = self.rng.randrange(len(self.nets))
        self.current = self.nets[idx]
        self.current_id = self.ids[idx]
        self.current_merged = self.merged[idx]
        self.current_macro = self.macro[idx]
        for i in range(self.num_envs):
            if self.is_mixed[i]:
                env = self._exposure[i]
                oid = int(self.env_member[i]) if self.env_member[i] >= 0 else self.current_id
                env[oid] = env.get(oid, 0.0) + 1.0

    def _assign_league(self, env_index: int) -> None:
        """P45: decide at a game's start whether a league member plays it, and which one.

        The learner's seat is already drawn; the member must sit opposite it. Draws use the
        league's own stream. No member for that seat: an ordinary own-pool game."""
        self.env_member[env_index] = -1
        if not self.is_mixed[env_index]:
            return
        if not self._draw_league_member(env_index) and self.per_game_own:
            self._assign_own(env_index)

    def _draw_league_member(self, env_index: int) -> bool:
        """The league draw of `_assign_league`, unchanged: True when a league member takes the game."""
        league = self._members("league")
        if not league:
            return False
        own = self._members("self")
        if not own:
            p = 1.0
        elif self.league_frac is not None:
            p = self.league_frac
        else:
            p = len(league) / len(self.nets)
        if self.league_rng.random() >= p:
            return False
        side = int(self.learner_side[env_index])
        fits = [j for j in league if self.seats[j] is None or self.seats[j] == -side]
        if not fits:
            return False
        j = fits[self.league_rng.randrange(len(fits))]
        self.env_member[env_index] = self.ids[j]
        return True

    def _assign_own(self, env_index: int) -> None:
        """L6: an own-pool member for the whole of this env's game, drawn from the game stream with the pool's
        own distribution (uniform, or PFSP / seat balancing restricted to the own members)."""
        own = self._members("self")
        if not own:
            return
        probs = self.sampling_probs()
        if (self.pfsp or (self.seat_balance and self.pressure() > 0.0)) and probs:
            sub = [probs[i] for i in own]
            total = sum(sub)
            if total > 0.0:
                r = self.game_rng.random() * total
                acc = 0.0
                for i, q in zip(own, sub):
                    acc += q
                    if r < acc:
                        self.env_member[env_index] = self.ids[i]
                        return
                self.env_member[env_index] = self.ids[own[-1]]
                return
        self.env_member[env_index] = self.ids[own[self.game_rng.randrange(len(own))]]

    # ---- P46b: the critic's prior per opponent and seat -------------------------------------------

    def _prior_self(self, side: int) -> float:
        return self.prior_sp_us if side == 1 else 1.0 - self.prior_sp_us

    def prior_rate_of(self, oid: int, side: int) -> float:
        """The learner's running score against member `oid` in `side`; a member without games starts at 50%:
        it is the learner's snapshot from moments ago (P46, 2026-11-14 -- the members' mean, the first rule, is far
        from even early in a run and took ~700 games to settle)."""
        key = (int(oid), int(side))
        if key not in self.prior_rate:
            self.prior_rate[key] = 0.5
        return self.prior_rate[key]

    def prior_beta(self, oid: int, side: int) -> float:
        """beta = logit(score against `oid` in `side`) - logit(self-play score in `side`), both clipped."""
        def lg(q: float) -> float:
            q = min(max(q, 0.01), 0.99)
            return math.log(q / (1.0 - q))
        return lg(self.prior_rate_of(oid, side)) - lg(self._prior_self(side))

    def prior_turn_w(self, turn: int) -> float:
        t = min(max(int(turn), 1), 10)
        w1, w10 = self.prior_turn_weight
        return w1 + (w10 - w1) * (t - 1) / 9.0

    def prior_shift(self, decision_players: np.ndarray, turns: np.ndarray) -> np.ndarray:
        """Per env, the log-odds shift on the critic's win probability at the current decision: +w(turn) beta
        where the learner decides in a pool game with a whole-game member, -w(turn) beta where the member
        decides (the critic reads from the decider's side), 0 in self-play and in games without a member."""
        out = np.zeros(self.num_envs, dtype=np.float32)
        if not self.prior_window:
            return out
        dp = np.asarray(decision_players, dtype=np.int8)
        for i in np.flatnonzero(self.is_mixed & (self.env_member >= 0)):
            side = int(self.learner_side[i])
            if dp[i] not in (side, -side):
                continue
            b = self.prior_turn_w(int(turns[i])) * self.prior_beta(int(self.env_member[i]), side)
            out[i] = b if dp[i] == side else -b
        return out

    def prior_terms(self, decision_players: np.ndarray, turns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per env, beta signed by the decider (+ learner, - member, 0 in self-play and games without a member)
        and the turn line w(turn): the two factors of `prior_shift`, for a gain the critic learns (P46, arms A/B)."""
        beta = np.zeros(self.num_envs, dtype=np.float32)
        wline = np.zeros(self.num_envs, dtype=np.float32)
        if not self.prior_window:
            return beta, wline
        dp = np.asarray(decision_players, dtype=np.int8)
        for i in np.flatnonzero(self.is_mixed & (self.env_member >= 0)):
            side = int(self.learner_side[i])
            if dp[i] not in (side, -side):
                continue
            b = self.prior_beta(int(self.env_member[i]), side)
            beta[i] = b if dp[i] == side else -b
            wline[i] = self.prior_turn_w(int(turns[i]))
        return beta, wline

    def prior_selfplay(self, victory_points: float) -> None:
        """Fold one self-play game's result into the running US score (draws count half)."""
        if not self.prior_window:
            return
        vp = float(victory_points)
        score = 0.5 if vp == 0.0 else (1.0 if vp > 0 else 0.0)
        self.prior_sp_us += self.prior_alpha * (score - self.prior_sp_us)

    def _weighted_index(self, probs: Sequence[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                return i
        return len(probs) - 1

    def on_episode_end(self, env_index: int,
                       victory_points: Optional[float] = None) -> None:
        """Record the result against whoever played it, then re-draw the learner's side.

        The runaway is bidirectional -- one arm ran away in the US's favour -- so an experiment
        that only ever trains the learner as one side would answer a narrower question than the
        one being asked.

        `victory_points` is the engine's signed VP, positive for the US. The learner won iff its
        sign matches the side the learner was playing; zero is a draw and counts as half. The
        result is split across the opponents this episode was actually played against, in
        proportion to how many iterations each was current for. Passing None records nothing,
        which is what a caller with no outcome to report should do.
        """
        member = int(self.env_member[env_index])
        if member >= 0 and member in self.games:
            side_now = int(self.learner_side[env_index])
            key = (member, side_now)
            self.league_seat_games[key] = self.league_seat_games.get(key, 0) + 1
            seat = self.seats[self._index_of(member)]
            if seat is not None and seat == side_now:
                self.league_wrong_seat_games += 1
        if victory_points is not None and self.is_mixed[env_index]:
            exposure = self._exposure[env_index]
            total = sum(exposure.values())
            if total > 0.0:
                side = int(self.learner_side[env_index])
                vp = float(victory_points)
                score = 0.5 if vp == 0.0 else (1.0 if (vp > 0) == (side == 1) else 0.0)
                for oid, share in exposure.items():
                    if oid not in self.games:      # evicted mid-episode
                        continue
                    w = share / total
                    self.games[oid] += w
                    self.wins[oid] += w * score
                    key = (oid, side)
                    self.side_games[key] = self.side_games.get(key, 0.0) + w
                    self.side_wins[key] = self.side_wins.get(key, 0.0) + w * score
        if (self.prior_window and victory_points is not None and self.is_mixed[env_index] and member >= 0
                and member in self.ids):
            # P46b: the whole game was against `member` (L6), so its result moves that member's running score.
            side = int(self.learner_side[env_index])
            vp = float(victory_points)
            score = 0.5 if vp == 0.0 else (1.0 if (vp > 0) == (side == 1) else 0.0)
            r = self.prior_rate_of(member, side)
            self.prior_rate[(member, side)] = r + self.prior_alpha * (score - r)
        self._exposure[env_index] = {}
        if self.seat_balance:
            # Mixed or self-play is re-drawn only at an episode boundary, so a game is never
            # half one and half the other. A newly mixed env starts its exposure to the current
            # opponent now rather than at the next iteration.
            self.is_mixed[env_index] = self._draw_mixed()
            if self.is_mixed[env_index]:
                self._exposure[env_index][self.current_id] = 1.0
        self.learner_side[env_index] = self._draw_side()
        self._assign_league(env_index)
        if self.seat_balance and self.env_member[env_index] >= 0:
            # A newly mixed env that went to a league member starts its exposure to that member.
            self._exposure[env_index] = {int(self.env_member[env_index]): 1.0}

    def state_dict(self) -> Dict[str, Any]:
        """Everything about the pool except the weights, which live in the snapshot files.

        Membership is recorded as the *step count* each member was captured at, because that is
        what identifies a snapshot on disk (`snapshot_<steps>steps.pt`) and it survives eviction
        reindexing. The per-opponent win and game counts come with it: without them a resumed
        PFSP draw starts blind and re-learns what the run already knew, and a uniform draw
        silently loses the record of which opponents were ever played.
        """
        # The run's own history only: league members belong to other runs, and the trainer
        # re-scans their directories on resume rather than restoring a stale membership. Without
        # league members this is the whole pool, exported exactly as before P24.
        steps, paths, merged, ids = list(self.steps), list(self.paths), list(self.merged), list(self.ids)
        wins, games = dict(self.wins), dict(self.games)
        side_wins, side_games = dict(self.side_wins), dict(self.side_games)
        if "league" in self.groups:
            own = self._members("self")
            keep = {self.ids[i] for i in own}
            steps = [self.steps[i] for i in own]
            paths = [self.paths[i] for i in own]
            merged = [self.merged[i] for i in own]
            ids = [self.ids[i] for i in own]
            wins = {k: v for k, v in wins.items() if k in keep}
            games = {k: v for k, v in games.items() if k in keep}
            side_wins = {k: v for k, v in side_wins.items() if k[0] in keep}
            side_games = {k: v for k, v in side_games.items() if k[0] in keep}
        return {
            "steps": steps,
            # Where each member's weights are, so a continuation into a new run directory keeps
            # the members earlier legs wrote. Absent from states written before it was added.
            "paths": paths,
            # Informational: on resume each member's view is re-derived from its run directory
            # (tools/lib/action_view.py), which cannot disagree with the snapshot it describes.
            "merged": merged,
            "ids": ids,
            "next_id": int(self._next_id),
            "wins": {int(k): float(v) for k, v in wins.items()},
            "games": {int(k): float(v) for k, v in games.items()},
            "rng_state": self.rng.getstate(),
            # Seat balancing's record, as [oid, side, value] rows (tuple keys do not serialise).
            "sp_us": float(self.sp_us),
            "side_wins": [[int(o), int(s), float(v)] for (o, s), v in side_wins.items()],
            "side_games": [[int(o), int(s), float(v)] for (o, s), v in side_games.items()],
            # P46b's running scores, as [oid, side, value] rows, and self-play's running US score.
            "prior_rate": [[int(o), int(s_), float(v)] for (o, s_), v in self.prior_rate.items() if o in set(ids)],
            "prior_sp_us": float(self.prior_sp_us),
        }

    def load_state_dict(self, blob: Dict[str, Any], loaded_steps: Sequence[int],
                        loaded_paths: Optional[Sequence[str]] = None) -> None:
        """Restore steps, ids and statistics onto a pool whose nets were just reloaded.

        `loaded_steps` is what the caller actually managed to load, in order, which may be a
        subset of what was saved if a snapshot has since been pruned. Statistics are carried over
        for the members that survived and dropped for the rest, so a partially recoverable pool
        keeps what it can rather than being silently zeroed or wrongly re-keyed.

        The steps themselves are restored too. They used not to be: the constructor records every
        seed at step 0, so after a resume the whole restored pool sat at 0. Eviction goes by
        spacing, and members at one step have zero spacing, so each new snapshot evicted a
        restored member until two were left -- the pool drained to its recent end -- and the next
        resume could not find the step-0 members on disk and dropped them.
        """
        if len(loaded_steps) != len(self.nets):
            raise ValueError(f"loaded_steps names {len(loaded_steps)} members but the pool holds "
                             f"{len(self.nets)} nets")
        if loaded_paths is not None and len(loaded_paths) != len(self.nets):
            raise ValueError(f"loaded_paths names {len(loaded_paths)} members but the pool holds "
                             f"{len(self.nets)} nets")
        self.steps = [int(st) for st in loaded_steps]
        self.paths = [str(q) for q in loaded_paths] if loaded_paths is not None else [""] * len(self.nets)
        self.groups = ["self"] * len(self.nets)
        self.seats = [None] * len(self.nets)
        self.env_member[:] = -1
        saved_steps = list(blob.get("steps", []))
        saved_ids = list(blob.get("ids", []))
        by_step = {int(st): int(oid) for st, oid in zip(saved_steps, saved_ids)}

        self.ids = []
        for st in loaded_steps:
            oid = by_step.get(int(st))
            if oid is None:
                oid = self._next_id
                self._next_id += 1
            self.ids.append(int(oid))
        self._next_id = max([int(blob.get("next_id", 0))] + [i + 1 for i in self.ids] + [0])

        keep = set(self.ids)
        self.wins = {int(k): float(v) for k, v in blob.get("wins", {}).items() if int(k) in keep}
        self.games = {int(k): float(v) for k, v in blob.get("games", {}).items() if int(k) in keep}
        for oid in self.ids:
            self.wins.setdefault(oid, 0.0)
            self.games.setdefault(oid, 0.0)
        self.sp_us = float(blob.get("sp_us", 0.5))
        self.side_wins = {(int(o), int(s)): float(v) for o, s, v in blob.get("side_wins", []) if int(o) in keep}
        self.side_games = {(int(o), int(s)): float(v) for o, s, v in blob.get("side_games", []) if int(o) in keep}
        rng = blob.get("rng_state")
        if rng is not None:
            try:
                self.rng.setstate(rng)
            except (TypeError, ValueError):
                pass
        self.prior_rate = {(int(o), int(s_)): float(v) for o, s_, v in blob.get("prior_rate", []) if int(o) in keep}
        self.prior_sp_us = float(blob.get("prior_sp_us", 0.5))
        if self.per_game_own:
            # Games in flight restart their opponent: the record of which member played them is not saved.
            for i in range(self.num_envs):
                self._assign_league(i)
        self.current = self.nets[0]
        self.current_id = self.ids[0]
        self.current_merged = self.merged[0]
        self.current_macro = self.macro[0]

    def learner_acts(self, decision_players: np.ndarray) -> np.ndarray:
        """Per environment: is the learner the one to move?

        True everywhere in self-play environments, and in mixed ones only when the side to move
        is the learner's.
        """
        dp = np.asarray(decision_players, dtype=np.int8)
        return ~self.is_mixed | (dp == self.learner_side)

    def stats(self) -> Dict[str, float]:
        """Pool diagnostics, including the ones that say whether the pool is doing anything.

        Until 2026-09-16 this returned only size, span and the mixed fraction -- that the pool
        exists and is used, never how the learner was faring against it. There was therefore no
        in-run instrument for whether pooling works, and the only live external opponents
        (HeuristicBot, RandomBot) saturate above 89% by 120M. `opp_win_rate_mean` and its spread
        are that missing instrument, and PFSP needs them anyway.
        """
        own_steps = [self.steps[i] for i in self._members("self")] or self.steps
        out: Dict[str, float] = {
            "opp_frac_mixed": float(self.is_mixed.mean()),
            "opp_pool_size": float(len(self.nets)),
            "opp_pool_span_m": float((max(own_steps) - min(own_steps)) / 1e6)
            if own_steps else 0.0,
        }
        league = [self.ids[i] for i in self._members("league")]
        if league:
            # P24: how the learner fares against the league, per seat, raw rather than smoothed --
            # the league driver decides resets on these, and a prior would hide an early lead.
            out["opp_league_size"] = float(len(league))
            out["opp_league_win_rate_mean"] = float(sum(self.win_rate(o) for o in league) / len(league))
            for side, name in ((1, "us"), (-1, "ussr")):
                g = sum(self.side_games.get((o, side), 0.0) for o in league)
                w = sum(self.side_wins.get((o, side), 0.0) for o in league)
                out[f"opp_league_games_{name}"] = float(g)
                out[f"opp_league_win_{name}"] = float(w / g) if g > 0.0 else 0.5
            # P45: each member's games by the learner's seat, and the games that broke seating.
            for o in league:
                for side, name in ((1, "us"), (-1, "ussr")):
                    out[f"opp_league_m{o}_games_learner_{name}"] = float(self.league_seat_games.get((o, side), 0))
            out["opp_league_frac_envs"] = float((self.env_member >= 0).mean())
            out["opp_league_wrong_seat_games"] = float(self.league_wrong_seat_games)
            out["opp_league_orphaned_games"] = float(self.league_orphaned_games)
        if self.per_game_own:
            mixed = self.is_mixed
            out["opp_per_game_frac"] = float((self.env_member[mixed] >= 0).mean()) if mixed.any() else 0.0
            out["opp_own_orphaned_games"] = float(self.own_orphaned_games)
        if self.prior_window:
            out["opp_prior_sp_us"] = float(self.prior_sp_us)
            betas = []
            for i in self._members("self"):
                oid = self.ids[i]
                for side, name in ((1, "us"), (-1, "ussr")):
                    if (oid, side) in self.prior_rate:
                        b = self.prior_beta(oid, side)
                        betas.append(abs(b))
                        out[f"opp_prior_m{oid}_score_{name}"] = float(self.prior_rate[(oid, side)])
                        out[f"opp_prior_m{oid}_beta_{name}"] = float(b)
            if betas:
                out["opp_prior_beta_abs_mean"] = float(sum(betas) / len(betas))
        rates = [self.win_rate(oid) for oid in self.ids]
        played = [self.games.get(oid, 0.0) for oid in self.ids]
        if rates:
            out["opp_win_rate_mean"] = float(sum(rates) / len(rates))
            out["opp_win_rate_min"] = float(min(rates))
            out["opp_win_rate_max"] = float(max(rates))
            out["opp_games_recorded"] = float(sum(played))
        probs = self.sampling_probs()
        if probs:
            # Entropy in nats, normalised by log(pool size): 1.0 is uniform, 0.0 is a single
            # opponent. The number to watch under PFSP -- a collapse means the prioritiser has
            # emptied the pool of the diversity the mechanism depends on.
            ent = -sum(q * math.log(q) for q in probs if q > 0.0)
            out["opp_pfsp_entropy"] = float(ent / math.log(len(probs))) if len(probs) > 1 else 1.0
            out["opp_pfsp_max_prob"] = float(max(probs))
        if self.seat_balance:
            out["opp_seat_sp_us"] = float(self.sp_us)
            out["opp_seat_pressure"] = float(self.pressure())
            out["opp_seat_weak_is_us"] = float(self.weak_side() == 1)
            mixed = self.is_mixed
            if mixed.any():
                out["opp_seat_learner_on_weak"] = float((self.learner_side[mixed] == self.weak_side()).mean())
        return out


def load_pool(paths: Sequence[str], device: Any) -> List[Any]:
    """Load frozen networks from checkpoint paths, refusing a layout mismatch rather than
    letting a retired-layout checkpoint read the wrong floats."""

    nets = []
    for path in paths:
        agent = NeuralAgent.from_checkpoint(path, device=device)
        check_obs_width(agent.model)
        nets.append(agent.model)
    return nets
