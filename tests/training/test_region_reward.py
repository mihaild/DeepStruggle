"""The per-region exploiter's reward (ai/rewards/region_reward.py): scorings of its region paid by their VP, final scoring's
share, a small win bonus, -1 for a loss, written zero-sum from the US's side."""
from __future__ import annotations

import random
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import ts_engine as ts

from ai.rewards.region_reward import (EUROPE_CONTROL_VP, REGIONS, SCORING_CARDS, RegionTargetedReward,
                                      europe_control_at_final_scoring, final_scoring_vp_us, final_share_us,
                                      reached_final_scoring)
from bindings.ts_env import TsVectorizedEnv

E = ts.Engine
L = ts.CardLocation
HANDS = (L.HAND_US_KNOWN, L.HAND_US_UNKNOWN, L.HAND_USSR_KNOWN, L.HAND_USSR_UNKNOWN)
EVENT = int(ts.FLAT_RESOLUTION)          # the play mode "Event" on the phasing side's own card


def _finished(seed: int) -> Any:
    env = TsVectorizedEnv(num_envs=1, base_seed=seed, auto_reset=False)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    for _ in range(6000):
        st = env.runner.get_state(0)
        if ts.Engine.is_terminal(st):
            return st
        _, masks, _, _, _ = env.step(np.array([int(rng.choice(np.flatnonzero(np.asarray(masks)[0])))]))
    raise AssertionError("no finished game")


def _game(r: RegionTargetedReward, scorings_us: list[float], u: float, st: Any) -> float:
    """The learner's total reward over a game with these region scorings (+ US) and terminal utility `u`."""
    total = 0.0
    for vp in scorings_us:
        r.pending_us[:] = [vp]
        total += float(r.compute_step_rewards(np.array([r.learner_side]), np.array([False]), np.zeros(1), np.zeros(1),
                                              np.zeros(1))[0])
    total += float(r.compute_step_rewards(np.array([r.learner_side]), np.array([True]), np.array([u]), np.zeros(1),
                                          np.zeros(1), [st])[0])
    return total


def test_a_win_pays_the_region_in_full_a_loss_a_small_share() -> None:
    st = _finished(11)
    share = final_share_us(st, "asia") if reached_final_scoring(st) else 0.0
    for side in (1, -1):
        def make() -> RegionTargetedReward:
            return RegionTargetedReward("asia", learner_side=side, scale=10.0, num_envs=1)
        vp = 5.0 * side            # +5 for the learner from an Asia scoring
        x = float(np.clip((vp + share) * side / 10.0, -1.0, 2.0))
        assert _game(make(), [vp], float(side), st) == pytest.approx(0.2 + 0.8 * x, abs=1e-6)
        assert _game(make(), [vp], -float(side), st) == pytest.approx(-1.0 + 0.2 * 0.8 * x, abs=1e-6)
        assert _game(make(), [vp], 0.0, st) == pytest.approx(0.2 * 0.8 * x, abs=1e-6)


def test_the_loss_share_is_paid_at_the_scoring_and_the_term_is_clipped() -> None:
    r = RegionTargetedReward("asia", learner_side=-1, scale=10.0, num_envs=2)
    r.pending_us[:] = [-5.0, 0.0]          # env 0: Asia scored 5 for the USSR learner
    rew = r.compute_step_rewards(np.array([-1, 1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert rew[0] == pytest.approx(0.2 * 0.8 * 0.5) and rew[1] == 0.0      # the USSR mover's reward
    r.pending_us[:] = [-40.0, 0.0]         # past the cap (2): only up to it is paid
    rew = r.compute_step_rewards(np.array([-1, 1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    assert rew[0] == pytest.approx(0.2 * 0.8 * (2.0 - 0.5))


def test_every_win_pays_more_than_every_loss() -> None:
    st = _finished(11)
    worst_win = _game(RegionTargetedReward("africa", 1, 5.0, 1), [-100.0], 1.0, st)
    best_loss = _game(RegionTargetedReward("africa", 1, 5.0, 1), [100.0], -1.0, st)
    assert worst_win == pytest.approx(0.2 - 0.8) and best_loss == pytest.approx(-1.0 + 0.2 * 0.8 * 2.0)
    assert worst_win > best_loss
    with pytest.raises(ValueError):
        RegionTargetedReward("africa", 1, 5.0, 1, loss_share=0.5)    # best loss -0.2 > worst win -0.6


def test_an_attached_env_pays_each_scoring_of_the_region_the_engine_records_at_its_step() -> None:
    r = RegionTargetedReward("asia", learner_side=1, scale=10.0, num_envs=64, weight=0.8, win_bonus=0.2)
    env = TsVectorizedEnv(num_envs=64, base_seed=7, reward_calculator=r)
    r.attach(env.runner)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(3)
    paid = 0
    for _ in range(4000):
        movers = np.asarray(env.runner.get_decision_players(), dtype=np.float64)
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        _, masks, rew, dones, _ = env.step(acts)
        rec = np.asarray(env.runner.get_step_track_vp())
        asia = rec[:, [1, 38]].sum(axis=1)
        for e in np.flatnonzero(~np.asarray(dones, dtype=bool)):
            # The US learner's loss share of the scoring, as the mover's reward (within the clip).
            before, after = r.scored_vp[e] - asia[e], r.scored_vp[e]
            if -10.0 <= before <= 20.0 and -10.0 <= after <= 20.0:
                assert rew[e] == pytest.approx(0.2 * 0.8 * asia[e] / 10.0 * movers[e], abs=1e-6), (e, asia[e], rew[e])
            paid += int(asia[e] != 0)
    assert paid > 0, "no Asia scoring in 4,000 random steps of 64 games"


def test_it_needs_a_seat_or_both_scales_a_region_and_a_scale() -> None:
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 0, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("antarctica", 1, 10.0, 1)
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 1, 0.0, 1)


def test_both_seats_read_each_games_seat_and_its_scale() -> None:
    """One exploiter playing both seats (owner, 2026-11-19): env 0 is its US game, env 1 its USSR game; each is paid from
    its own seat with that seat's scale, and the record of who scored is the same Asia scoring (+5 for the US)."""
    r = RegionTargetedReward("asia", learner_side=0, scale=10.0, num_envs=2, scale_ussr=5.0)
    sides = np.array([1, -1], dtype=np.int8)
    r.attach_sides(lambda: sides)
    r.pending_us[:] = [5.0, 5.0]
    rew = r.compute_step_rewards(np.array([1, -1]), np.array([False, False]), np.zeros(2), np.zeros(2), np.zeros(2))
    # US exploiter: +0.16 * 0.5 as the US mover; USSR exploiter: -0.16 * 1.0 for it, as the USSR mover's reward.
    assert rew[0] == pytest.approx(0.2 * 0.8 * 0.5) and rew[1] == pytest.approx(-0.2 * 0.8 * 1.0)
    assert r.scored_vp.tolist() == [5.0, -5.0]


def test_both_seats_need_the_pool_and_both_scales() -> None:
    with pytest.raises(ValueError):
        RegionTargetedReward("asia", 0, 10.0, 1)
    r = RegionTargetedReward("asia", 0, 10.0, 1, scale_ussr=8.0)
    with pytest.raises(RuntimeError):
        r.compute_step_rewards(np.array([1]), np.array([False]), np.zeros(1), np.zeros(1), np.zeros(1))


def test_a_europe_control_win_pays_at_least_the_full_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner, 2026-11-20: the Europe exploiter is paid at least 1 for winning by Europe Control, whatever its VP. Random
    games almost never end that way, so a finished game is read as one (the ending reason is the only input)."""
    st = _finished(11)
    monkeypatch.setattr(ts, "game_ending_reason", lambda state: "Europe Control")
    for side in (1, -1):
        r = RegionTargetedReward("europe", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r, [-30.0 * side], float(side), st) == pytest.approx(1.0, abs=1e-6)   # x at the floor, then the win
        r = RegionTargetedReward("europe", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r, [30.0 * side], float(side), st) == pytest.approx(0.2 + 0.8 * 2.0, abs=1e-6)  # above 1 stays
        r2 = RegionTargetedReward("asia", learner_side=side, scale=10.0, num_envs=1)
        assert _game(r2, [-30.0 * side], float(side), st) == pytest.approx(0.2 - 0.8, abs=1e-6)    # only Europe's own



# ---- Real games to each ending (audit V-8, 2026-11-22): the reward read through an attached env, as in training. ----

def _card(name: str) -> int:
    return int(ts.CardData.get_card_by_name(name))


def _us_action_round(seed: int) -> ts.GameState:
    """A turn 1-3 position where the US picks its action round's card (seeded random play from the deal)."""
    for k in range(200):
        s = ts.GameState()
        E.init_game(s, seed + k)
        rng = random.Random(seed + k)
        for _ in range(20000):
            if E.is_terminal(s):
                break
            if (s.current_phase == ts.Phase.ACTION_ROUND and s.ctx_stack_depth == 0
                    and s.ctx().decision_type == ts.DecisionType.SELECT_CARD and s.ctx().decision_player == ts.Player.US
                    and s.phasing_player == ts.Player.US and 1 <= s.turn <= 3 and s.action_round <= 5):
                return s
            E.step_flat(s, rng.choice([i for i, v in enumerate(E.get_flat_action_mask(s)) if v]))
    raise AssertionError("no US action round reached")


def _set_region(s: ts.GameState, region: "ts.Region", us: int, ussr: int) -> None:
    for c in range(84):
        if ts.MapData.get_country_info(c)["region"] == region:
            s.set_country(c, us, ussr)


def _play(start: ts.GameState, reward: RegionTargetedReward, seed: int, first: Sequence[int] = (),
          side: Optional[int] = None) -> Tuple[ts.GameState, float, float]:
    """Plays `start` to its end in an attached env: the `first` flat actions, then seeded random moves. Returns the
    finished state, the exploiter's total reward over the game (in seat `side`, default the locked one) and the
    region's in-game VP (+ US) as the runner recorded it, read independently of the reward."""
    side = reward.learner_side if side is None else side
    env = TsVectorizedEnv(num_envs=1, base_seed=1, reward_calculator=reward, auto_reset=False)
    reward.attach(env.runner)
    if reward.learner_side == 0:
        reward.attach_sides(lambda: np.array([side], dtype=np.int8))
    env.reset_all()
    env.runner.set_state(0, start)
    env.runner.refresh_all()
    rng = np.random.default_rng(seed)
    mask = np.asarray(E.get_flat_action_mask(start))
    total, region_vp = 0.0, 0.0
    for n in range(5000):
        mover = int(env.runner.get_decision_players()[0])
        act = first[n] if n < len(first) else int(rng.choice(np.flatnonzero(mask)))
        _, masks, rew, dones, _ = env.step(np.array([act]))
        total += float(rew[0]) * mover * side          # the mover's reward, from the exploiter's seat
        region_vp += float(np.asarray(env.runner.get_step_track_vp())[0, reward.cards].sum())
        mask = np.asarray(masks)[0]
        if dones[0]:
            return env.runner.get_state(0), total, region_vp
    raise AssertionError("the game did not end")


def _expected(r: RegionTargetedReward, end: ts.GameState, region_vp_us: float, side: int) -> float:
    """What the module docstring says the exploiter is paid for this game: x = clip(net region VP / scale), a win
    win_bonus + weight * x (a Europe Control win at least control_win), a loss -1 + loss_share * weight * x, a draw
    loss_share * weight * x."""
    vp = (region_vp_us + final_scoring_vp_us(end, r.region)) * side
    x = float(np.clip(vp / r.scale_of[side], r.floor, r.cap))
    winner = int(np.sign(int(end.victory_points)))
    if winner == side:
        pay = r.win_bonus + r.weight * x
        if r.region == "europe" and str(ts.game_ending_reason(end)) == "Europe Control":
            pay = max(pay, r.control_win)
        return pay
    return (-1.0 if winner == -side else 0.0) + r.loss_share * r.weight * x


def _turn_ten(seed: int, europe_us: int) -> ts.GameState:
    """A position played on from turn 10 (the turn counter set from an early action round), every scoring card out of the
    hands, the US with `europe_us` Influence and the USSR none in every European country (0 leaves the board)."""
    s = _us_action_round(seed)
    s.turn = 10
    for c in SCORING_CARDS:
        if s.get_card_location(c) in HANDS:
            s.set_card_location(c, L.DISCARD_PILE)
    if europe_us:
        _set_region(s, ts.Region.EUROPE, europe_us, 0)
    return s


def _ending(make: Callable[[int], ts.GameState], want: Callable[[ts.GameState], bool], region: str,
            ) -> Tuple[ts.GameState, int]:
    """The first seed whose position, played out at random, ends as `want` says (deterministic: seeded engine and
    moves); the finished state and the seed."""
    for seed in range(40):
        end, _, _ = _play(make(seed), RegionTargetedReward(region, 1, 10.0, 1), seed)
        if want(end):
            return end, seed
    raise AssertionError("no seed reached the ending")


def test_the_engines_europe_control_record_is_the_constant_the_reward_books_at_final_scoring() -> None:
    s = _us_action_round(1600)
    _set_region(s, ts.Region.EUROPE, 20, 0)
    eu = _card("Europe Scoring")
    s.set_card_location(eu, L.HAND_US_UNKNOWN)
    end, _, europe_vp = _play(s, RegionTargetedReward("europe", 1, 10.0, 1), 0, first=[eu - 1])
    assert str(ts.game_ending_reason(end)) == "Europe Control" and int(end.turn) <= 10
    assert europe_vp == EUROPE_CONTROL_VP
    assert not europe_control_at_final_scoring(end) and final_scoring_vp_us(end, "europe") == 0.0


@pytest.mark.parametrize("region", ["europe", "asia"])
def test_an_in_game_europe_control_win_pays_the_cap_and_its_loss_the_floor(region: str) -> None:
    """The US plays Europe Scoring controlling Europe: booked +100, clipped to the cap -- the Europe exploiter's win
    pays 0.2 + 0.8 x 2 = 1.8 (above control_win), the USSR one's loss -1 + 0.16 x -1. Another region's exploiter is
    paid for its own region only (none scored here): a plain win 0.2, a plain loss -1."""
    eu = _card("Europe Scoring")
    for side in (1, -1):
        s = _us_action_round(1600)
        _set_region(s, ts.Region.EUROPE, 20, 0)
        s.set_card_location(eu, L.HAND_US_UNKNOWN)
        r = RegionTargetedReward(region, side, 10.0, 1)
        end, total, region_vp = _play(s, r, 0, first=[eu - 1])
        assert str(ts.game_ending_reason(end)) == "Europe Control"
        want = {("europe", 1): 1.8, ("europe", -1): -1.16, ("asia", 1): 0.2, ("asia", -1): -1.0}[(region, side)]
        assert total == pytest.approx(want, abs=1e-6)
        assert total == pytest.approx(_expected(r, end, region_vp, side), abs=1e-6)


def test_a_europe_control_win_at_final_scoring_pays_the_same_as_in_game() -> None:
    """Audit V-8: final scoring's Europe Control goes through no Event, so the in-game record has nothing; the reward
    books +/-100 itself (it read 0: the win paid control_win 1.0, the loss -1.0, and vp_per_game missed it)."""
    end, seed = _ending(lambda k: _turn_ten(1600 + 37 * k, 20),
                        lambda st: str(ts.game_ending_reason(st)) == "Europe Control", "europe")
    assert int(end.turn) == 11 and europe_control_at_final_scoring(end) and not reached_final_scoring(end)
    assert final_scoring_vp_us(end, "europe") == EUROPE_CONTROL_VP and final_scoring_vp_us(end, "asia") == 0.0
    for side, want in ((1, 1.8), (-1, -1.16)):
        r = RegionTargetedReward("europe", side, 10.0, 1)
        _, total, region_vp = _play(_turn_ten(1600 + 37 * seed, 20), r, seed)
        assert region_vp == 0.0      # no Event booked it
        assert total == pytest.approx(want, abs=1e-6)
        assert r.finished[side] == [EUROPE_CONTROL_VP * side]    # region/vp_per_game counts it
    # Both seats (learner_side 0), as the both-seat exploiters ran: the USSR seat with its own scale.
    r = RegionTargetedReward("europe", 0, 10.0, 1, scale_ussr=11.0)
    _, total, _ = _play(_turn_ten(1600 + 37 * seed, 20), r, seed, side=-1)
    assert total == pytest.approx(-1.16, abs=1e-6)
    # Another region's exploiter: final scoring stopped at Europe Control, so its region has no final share.
    r = RegionTargetedReward("asia", 1, 15.0, 1)
    _, total, region_vp = _play(_turn_ten(1600 + 37 * seed, 20), r, seed)
    assert total == pytest.approx(0.2 + 0.8 * float(np.clip(region_vp / 15.0, -1.0, 2.0)), abs=1e-6)


@pytest.mark.parametrize("region", ["europe", "asia", "middle_east", "south_america"])
def test_final_scoring_without_control_pays_the_regions_share(region: str) -> None:
    end, seed = _ending(lambda k: _turn_ten(1600 + 37 * k, 0),
                        lambda st: str(ts.game_ending_reason(st)) == "final scoring", region)
    assert int(end.turn) == 11 and reached_final_scoring(end) and not europe_control_at_final_scoring(end)
    assert final_scoring_vp_us(end, region) == final_share_us(end, region)
    for side in (1, -1):
        r = RegionTargetedReward(region, side, 4.0, 1)
        _, total, region_vp = _play(_turn_ten(1600 + 37 * seed, 0), r, seed)
        assert total == pytest.approx(_expected(r, end, region_vp, side), abs=1e-6)
        assert r.finished[side] == [pytest.approx((region_vp + final_share_us(end, region)) * side)]


def test_the_final_share_is_what_the_engines_final_scoring_adds() -> None:
    """final_share_us reads each region as Scoring::execute_final_scoring scores it (Shuttle Diplomacy ignored): on an
    unclamped total, the six shares plus the China Card are the whole change."""
    checked = 0
    for seed in range(1600, 1700, 7):
        s = _us_action_round(seed)
        s.victory_points = 0
        if E.is_terminal(s):
            continue
        shares = sum(final_share_us(s, r) for r in REGIONS)
        china = {ts.Player.US: 1, ts.Player.USSR: -1}.get(s.china_card_holder, 0)
        if abs(shares + china) >= 20:
            continue
        ts.Scoring.execute_final_scoring(s)
        assert int(s.victory_points) == shares + china
        checked += 1
    assert checked >= 5


def test_a_20_vp_win_pays_the_regions_in_game_scorings_only() -> None:
    """The US scores Asia from 15 VP and wins at 20: the Asia exploiter is paid for the VP the record booked (the
    track's move, capped by the win), the Europe exploiter a plain win; their USSR counterparts lose."""
    asia = _card("Asia Scoring")
    for region in ("asia", "europe"):
        for side in (1, -1):
            s = _us_action_round(1600)
            _set_region(s, ts.Region.ASIA, 10, 0)
            s.set_card_location(asia, L.HAND_US_UNKNOWN)
            s.victory_points = 15
            r = RegionTargetedReward(region, side, 10.0, 1)
            end, total, region_vp = _play(s, r, 0, first=[asia - 1])
            assert str(ts.game_ending_reason(end)) == "20 VP" and int(end.victory_points) == 20
            assert region_vp == (5.0 if region == "asia" else 0.0)
            want = (0.2 + 0.8 * 0.5 if side == 1 else -1.0 - 0.16 * 0.5) if region == "asia" else (0.2 if side == 1 else -1.0)
            assert total == pytest.approx(want, abs=1e-6)
            assert total == pytest.approx(_expected(r, end, region_vp, side), abs=1e-6)


def test_a_defcon_loss_pays_minus_one_plus_the_share_already_paid() -> None:
    """The US plays Duck and Cover's Event at DEFCON 2 and loses at DEFCON 1: no final scoring, so the region's in-game
    scorings alone count -- here Europe was scored earlier in the game for the US (+5 booked before the suicide)."""
    dc = _card("Duck and Cover")
    for side in (1, -1):
        s = _us_action_round(1600)
        s.set_card_location(dc, L.HAND_US_UNKNOWN)
        s.defcon = 2
        r = RegionTargetedReward("europe", side, 10.0, 1)
        r.scored_vp[0] = 5.0 * side          # an earlier Europe scoring of +5 for the US
        dense_before = 0.2 * 0.8 * 0.5 * side
        end, total, region_vp = _play(s, r, 0, first=[dc - 1, EVENT])
        assert str(ts.game_ending_reason(end)) == "DEFCON 1 (own decision)" and int(end.victory_points) < 0
        assert region_vp == 0.0 and final_scoring_vp_us(end, "europe") == 0.0
        # Total over the game = the dense share paid at the earlier scoring + what the end pays now.
        want = (-1.0 + 0.2 * 0.8 * 0.5) if side == 1 else (0.2 + 0.8 * -0.5)
        assert dense_before + total == pytest.approx(want, abs=1e-6)
