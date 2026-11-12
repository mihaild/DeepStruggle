"""P24 / P45: league members in the opponent pool, and the directory scan that finds them.

A league member is another run's snapshot -- a main exploiter's for the main agent, the main
agent's newest for the exploiter. It must not disturb the run's own history (spacing eviction, the
resume record), and a pool without any must behave exactly as before.

P45 (owner, 2026-11-12): a member has a seat, the one its run was locked to, and plays only there,
and a league game keeps one member from start to finish. The checks numbered below are P45's
(research/plans/P45_league.md, "The pool change and its checks").
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

import torch
import torch.nn as nn

from ai.training.generic_trainer import LEAGUE_SNAPSHOT_RE, league_member_seat, scan_league_snapshots
from ai.training.opponent_pool import OpponentPool


class _Tiny(nn.Module):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), k))

    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


class _Const(nn.Module):
    """A policy that always prefers one action: its logits are 0 except `action`, which is 10."""

    def __init__(self, action: int, width: int = 6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.action, self.width = action, width

    def forward(self, obs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.zeros(obs.shape[0], self.width)
        z[:, self.action] = 10.0
        return z, torch.zeros(obs.shape[0], 1), torch.zeros(obs.shape[0], 1)


def _end_all(p: OpponentPool, vp: float = 1.0) -> None:
    for i in range(p.num_envs):
        p.on_episode_end(i, victory_points=vp)


def _seated_pool(seats: List[Optional[int]], num_envs: int = 64, frac: float = 1.0,
                 league_frac: Optional[float] = 0.5, seed: int = 5) -> OpponentPool:
    """An own pool of three snapshots plus one league member per entry of `seats`."""
    p = OpponentPool([_Tiny(0.0)], num_envs=num_envs, frac=frac, seed=seed, capacity=8,
                     league_capacity=8, league_frac=league_frac)
    for n in (5, 10):
        p.add(_Tiny(float(n)), n * 1_000_000)
    for k, seat in enumerate(seats):
        p.add(_Tiny(-1.0 - k), 1 + k, group="league", seat=seat)
    _end_all(p)        # every env starts a game with the members present
    return p


def _main_pool(league_capacity: int = 2, league_frac: Any = None) -> OpponentPool:
    """A run's own pool of capacity 4, seeded at step 0, ready to take league members."""
    return OpponentPool([_Tiny(0.0)], num_envs=8, frac=0.5, seed=3, capacity=4,
                        league_capacity=league_capacity, league_frac=league_frac)


def test_league_members_do_not_count_against_the_runs_own_capacity() -> None:
    p = _main_pool()
    for n in (5, 10, 15, 20, 25):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 60_000_000, group="league")
    p.add(_Tiny(-2.0), 70_000_000, group="league")
    own = [p.steps[i] for i in p._members("self")]
    assert len(own) == 4 and own[0] == 0 and own[-1] == 25_000_000   # spacing kept the endpoints
    assert [p.steps[i] for i in p._members("league")] == [60_000_000, 70_000_000]


def test_the_league_evicts_its_oldest_added_member_whatever_its_steps() -> None:
    """Exploiter generations restart their step counter, so arrival order is what ages them."""
    p = _main_pool(league_capacity=2)
    p.add(_Tiny(1.0), 90_000_000, group="league")
    p.add(_Tiny(2.0), 70_000_000, group="league")
    p.add(_Tiny(3.0), 80_000_000, group="league")
    assert [p.steps[i] for i in p._members("league")] == [70_000_000, 80_000_000]


def _league_share(p: OpponentPool, rounds: int = 200) -> float:
    """The share of mixed games assigned to a league member, over `rounds` episode ends of every env."""
    hits = games = 0
    for _ in range(rounds):
        _end_all(p)
        mixed = p.is_mixed
        hits += int((p.env_member[mixed] >= 0).sum())
        games += int(mixed.sum())
    return hits / games


def test_league_frac_sets_the_leagues_share_of_games() -> None:
    p = _main_pool(league_capacity=1, league_frac=0.25)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 1, group="league")           # unseated: takes either seat
    assert 0.21 < _league_share(p) < 0.29


def test_without_league_frac_the_league_share_follows_its_size() -> None:
    p = _main_pool(league_capacity=1)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 1, group="league")
    assert 0.16 < _league_share(p) < 0.24          # 1 of 5 members


def test_the_iterations_opponent_comes_from_the_runs_own_members() -> None:
    p = _main_pool(league_capacity=1, league_frac=0.5)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 1, group="league")
    league_id = p.ids[p._members("league")[0]]
    for _ in range(500):
        p.start_iteration()
        assert p.current_id != league_id


# ---- P45's checks -------------------------------------------------------------------------------

def test_1_a_seat_locked_member_never_plays_its_other_seat() -> None:
    p = _seated_pool([1, -1])                   # a US-locked and a USSR-locked member
    seat = {p.ids[j]: int(p.seats[j] or 0) for j in p._members("league")}
    for _ in range(2000 // p.num_envs + 1):
        _end_all(p)
        for i in np.flatnonzero(p.env_member >= 0):
            assert int(p.learner_side[i]) == -seat[int(p.env_member[i])]
    assert p.league_wrong_seat_games == 0
    assert p.stats()["opp_league_wrong_seat_games"] == 0.0


def test_2_a_league_game_keeps_its_member_until_the_episode_ends() -> None:
    p = _seated_pool([1, -1])
    before = p.env_member.copy()
    for _ in range(20):                         # iterations inside one episode
        p.start_iteration()
        assert np.array_equal(p.env_member, before)
    changed = 0
    for _ in range(30):
        p.on_episode_end(0, victory_points=1.0)
        changed += int(p.env_member[0] != before[0])
    assert changed > 0                          # it changes at an episode end


def _old_sequence(seed: int, num_envs: int, n_nets: int, iters: int, league_frac: Optional[float],
                  n_own: int) -> List[Tuple[int, Tuple[int, ...]]]:
    """The (opponent index, learner seats) sequence of the pool before P45, re-derived from its code:
    sides drawn at construction and at every episode end with `rng.random() < 0.5`, the iteration's
    opponent with `randrange` (over every member; over the own members after a frac draw with a league)."""
    rng = random.Random(seed)
    sides = [1 if rng.random() < 0.5 else -1 for _ in range(num_envs)]
    out = []
    for _ in range(iters):
        if league_frac is not None and n_own < n_nets:
            r = rng.random()
            assert not r < league_frac           # league_frac 0: never the league
            idx = rng.randrange(n_own)
        else:
            idx = rng.randrange(n_nets)
        for i in range(num_envs):
            sides[i] = 1 if rng.random() < 0.5 else -1
        out.append((idx, tuple(sides)))
    return out


def _new_sequence(p: OpponentPool, iters: int) -> List[Tuple[int, Tuple[int, ...]]]:
    out = []
    for _ in range(iters):
        p.start_iteration()
        idx = p.ids.index(p.current_id)
        _end_all(p)
        out.append((idx, tuple(int(x) for x in p.learner_side)))
    return out


def test_3_own_pool_games_are_untouched() -> None:
    # No league members: the pool plays the sequence the code before the change played.
    p = OpponentPool([_Tiny(0.0), _Tiny(1.0), _Tiny(2.0)], num_envs=6, frac=1.0, seed=17, capacity=4)
    assert _new_sequence(p, 50) == _old_sequence(17, 6, 3, 50, None, 3)
    # --league-frac 0 with a league member present: the own pool's sequence is still the old one.
    q = OpponentPool([_Tiny(0.0), _Tiny(1.0), _Tiny(2.0)], num_envs=6, frac=1.0, seed=17, capacity=4,
                     league_capacity=2, league_frac=0.0)
    q.add(_Tiny(-1.0), 1, group="league", seat=1)
    got = _new_sequence(q, 50)
    want = _old_sequence(17, 6, 4, 50, 0.0, 3)
    assert got == want
    assert (q.env_member < 0).all()


def test_4_shares_and_seats() -> None:
    p = _seated_pool([1, 1, -1, -1], num_envs=256, league_frac=0.5)
    counts: dict = {}
    league_seats = {1: 0, -1: 0}
    games = league_games = 0
    for _ in range(80):
        _end_all(p)
        games += int(p.is_mixed.sum())
        for i in np.flatnonzero(p.env_member >= 0):
            league_games += 1
            oid = int(p.env_member[i])
            counts[oid] = counts.get(oid, 0) + 1
            league_seats[int(p.learner_side[i])] += 1
    assert 0.46 < league_games / games < 0.54                 # the league's share is --league-frac
    by_seat: dict = {1: [], -1: []}
    for j in p._members("league"):
        by_seat[int(p.seats[j] or 0)].append(counts[p.ids[j]])
    for seat in (1, -1):
        a, b = by_seat[seat]
        assert abs(a - b) / (a + b) < 0.08                      # members of one seat drawn alike
    assert abs(league_seats[1] - league_seats[-1]) / league_games < 0.06   # the learner's seats 50/50


def test_5_one_seat_filled() -> None:
    p = _seated_pool([1], num_envs=256, league_frac=1.0)    # only a US-locked member
    seen_league = 0
    for _ in range(20):
        _end_all(p)
        league = p.env_member >= 0
        assert (p.learner_side[league] == -1).all()           # only with the learner as the USSR
        assert (p.env_member[p.learner_side == 1] < 0).all()  # the rest fall back to the own pool
        seen_league += int(league.sum())
    assert seen_league > 0


def _run_dir(root: str, name: str, lock: Optional[str]) -> str:
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump({"opponent_lock_side": lock}, f)
    q = os.path.join(d, "snapshot_10000000steps.pt")
    open(q, "wb").close()
    return q


def test_6_the_seat_comes_from_the_members_run() -> None:
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as league:
        us = _run_dir(root, "us_run", "us")
        ussr = _run_dir(root, "ussr_run", "ussr")
        free = _run_dir(root, "free_run", None)
        os.symlink(us, os.path.join(league, "R108_snapshot_10000000steps.pt"))
        assert league_member_seat(os.path.join(league, "R108_snapshot_10000000steps.pt")) == 1
        assert league_member_seat(ussr) == -1
        assert league_member_seat(free) is None
        bare = os.path.join(root, "snapshot_20000000steps.pt")      # no metadata beside it
        open(bare, "wb").close()
        with pytest.raises(ValueError, match="cannot be known"):
            league_member_seat(bare)


def test_7_each_envs_opponent_rows_come_from_its_own_member() -> None:
    own = _Const(0)
    us_member, ussr_member = _Const(3), _Const(5)
    p = OpponentPool([own], num_envs=64, frac=1.0, seed=9, capacity=4, league_capacity=4, league_frac=0.7)
    p.add(us_member, 1, group="league", seat=1)
    p.add(ussr_member, 2, group="league", seat=-1)
    _end_all(p)
    p.start_iteration()
    learner = np.zeros(64, dtype=bool)                          # every row is the frozen side's move
    learner[::5] = True                                         # ... except some learner rows
    obs, masks = torch.zeros(64, 4), torch.ones(64, 6, dtype=torch.bool)
    logits = torch.full((64, 6), -1.0)
    for net, rows in p.opponent_rows(learner):
        idx = torch.from_numpy(rows)
        logits.index_copy_(0, idx, net(obs.index_select(0, idx), masks.index_select(0, idx))[0])
    act = logits.argmax(1).numpy()
    for i in range(64):
        if learner[i]:
            assert (logits[i] == -1.0).all()                    # learner rows untouched
            continue
        oid = int(p.env_member[i])
        want = 0 if oid < 0 else (3 if p.nets[p.ids.index(oid)] is us_member else 5)
        assert act[i] == want
    assert (p.env_member >= 0).any() and (p.env_member < 0).any()
    assert np.array_equal(p.learner_acts(np.ones(64, dtype=np.int8)), p.learner_side == 1)


def test_8_a_resume_carries_no_assignment() -> None:
    p = _seated_pool([1, -1])
    p.start_iteration()
    _end_all(p, 3.0)
    blob = p.state_dict()
    assert "env_member" not in blob
    q = OpponentPool([_Tiny(0.0), _Tiny(5.0), _Tiny(10.0)], num_envs=64, frac=1.0, seed=5, capacity=8,
                     league_capacity=8, league_frac=0.5)
    q.load_state_dict(blob, blob["steps"], blob["paths"])
    assert (q.env_member < 0).all() and q.seats == [None, None, None]
    assert q.state_dict()["wins"] == blob["wins"] and q.state_dict()["games"] == blob["games"]


def test_an_evicted_member_hands_its_games_to_the_own_pool() -> None:
    p = _seated_pool([1], num_envs=64, league_frac=1.0)
    p.league_capacity = 1
    first = p.ids[p._members("league")[0]]
    busy = int((p.env_member == first).sum())
    assert busy > 0
    p.add(_Tiny(-9.0), 99, group="league", seat=1)              # the same seat: evicts the first member
    assert p.league_orphaned_games == busy
    assert not (p.env_member == first).any()                    # its games went to the own pool
    assert p.stats()["opp_league_orphaned_games"] == float(busy)


def test_the_leagues_capacity_holds_per_seat() -> None:
    """Capacity four a seat (P45): a new generation of one seat's exploiter evicts that seat's oldest only."""
    p = _main_pool(league_capacity=2)
    for k, seat in enumerate((1, 1, -1, -1)):
        p.add(_Tiny(float(k)), k, group="league", seat=seat)
    first_us, first_ussr = (p.ids[j] for j in p._members("league")[::2])
    p.add(_Tiny(9.0), 9, group="league", seat=1)
    ids = [p.ids[j] for j in p._members("league")]
    assert first_us not in ids and first_ussr in ids
    assert sorted(p.seats[j] for j in p._members("league")) == [-1, -1, 1, 1]


def test_seats_are_validated() -> None:
    p = _main_pool()
    with pytest.raises(ValueError, match="seat"):
        p.add(_Tiny(1.0), 1, group="league", seat=2)


def test_the_resume_record_holds_the_runs_own_history_only() -> None:
    p = _main_pool()
    p.add(_Tiny(1.0), 5_000_000)
    p.add(_Tiny(-1.0), 60_000_000, group="league")
    blob = p.state_dict()
    assert blob["steps"] == [0, 5_000_000]
    assert len(blob["ids"]) == 2 and len(blob["paths"]) == 2


def test_league_stats_are_per_seat() -> None:
    p = OpponentPool([_Tiny(0.0)], num_envs=2, frac=1.0, seed=1, groups=["league"],
                     league_capacity=2)
    p.start_iteration()
    p.learner_side[:] = [1, -1]          # env 0: learner is US; env 1: learner is USSR
    p.on_episode_end(0, victory_points=5.0)    # US won: the learner (US) won
    p.on_episode_end(1, victory_points=5.0)    # US won: the learner (USSR) lost
    st = p.stats()
    assert st["opp_league_size"] == 1.0
    assert st["opp_league_win_us"] == 1.0 and st["opp_league_win_ussr"] == 0.0
    assert st["opp_league_games_us"] == 1.0 and st["opp_league_games_ussr"] == 1.0


def test_a_pool_without_league_members_reports_no_league_stats() -> None:
    st = _main_pool().stats()
    assert not any(k.startswith("opp_league") for k in st)


def test_the_scan_takes_snapshots_only_and_waits_for_a_file_to_settle() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = time.time() - 120
        for k, f in enumerate(("snapshot_10000000steps.pt", "E5-03-43-1_snapshot_20000000steps.pt",
                               "pool_15000000steps.pt", "resume_10000000steps.pt",
                               "snapshot_final.pt", "snapshot_0s.pt")):
            open(os.path.join(d, f), "wb").close()
            os.utime(os.path.join(d, f), (old + k, old + k))   # oldest first by mtime
        fresh = os.path.join(d, "snapshot_30000000steps.pt")
        open(fresh, "wb").close()             # just written: left for the next scan
        seen: set = set()
        got = [os.path.basename(q) for q in scan_league_snapshots([d], seen)]
        assert got == ["snapshot_10000000steps.pt", "E5-03-43-1_snapshot_20000000steps.pt"]
        assert scan_league_snapshots([d], seen, settle_seconds=0.0)[-1] == os.path.realpath(fresh)


def test_the_scan_takes_a_snapshot_published_under_two_names_once() -> None:
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as league:
        real = os.path.join(src, "snapshot_40000000steps.pt")
        open(real, "wb").close()
        os.symlink(real, os.path.join(league, "E5-03-43-1_snapshot_40000000steps.pt"))
        seen = {os.path.realpath(real)}
        assert scan_league_snapshots([league], seen, settle_seconds=0.0) == []
        assert LEAGUE_SNAPSHOT_RE.search("E5-03-43-1_snapshot_40000000steps.pt")
