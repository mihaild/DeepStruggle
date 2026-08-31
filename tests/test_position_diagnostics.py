"""Position diagnostics: battleground occupancy, regional scores, salvageability.

These back the metrics that read the board rather than the result. Win rate plateaus while
the underlying play stays incoherent, so they are the ones expected to move first.
"""

import ts_engine as ts

from ai.eval.position_diagnostics import (
    BATTLEGROUNDS,
    empty_battlegrounds,
    is_salvageable,
    profile_self_play,
    region_score_sums,
    scalar_metrics,
)

WEST_GERMANY, EAST_GERMANY = 13, 14


def _fresh() -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    return st


def test_battleground_list_matches_the_engine() -> None:
    """29 battlegrounds, and every one is flagged as such by MapData."""
    assert len(BATTLEGROUNDS) == 29
    for cid in BATTLEGROUNDS:
        assert ts.MapData.get_country_info(cid)["battleground"]


def test_empty_battlegrounds_tracks_influence() -> None:
    """Placing influence removes a battleground from the empty set, and vice versa."""
    st = _fresh()
    empty = set(empty_battlegrounds(st))

    # East Germany is seeded with USSR influence by the opening setup.
    assert EAST_GERMANY not in empty

    target = next(cid for cid in BATTLEGROUNDS if cid in empty)
    c = st.get_country(target)
    st.set_country(target, 1, int(c.ussr_influence))
    assert target not in set(empty_battlegrounds(st))


def test_region_scores_agree_with_the_engines_net_delta() -> None:
    """us_score - ussr_score must equal what actually scoring the region would award.

    Guards the whole salvageability criterion: it is built on evaluate_region, so if that
    ever diverged from Scoring.score_region the filter would silently drift.
    """
    st = _fresh()
    # Give both sides something to score so the comparison is not trivially 0 - 0.
    st.set_country(WEST_GERMANY, 4, 0)
    st.set_country(EAST_GERMANY, 0, 4)

    for region in (ts.Region.EUROPE, ts.Region.ASIA, ts.Region.AFRICA):
        summary = ts.Scoring.evaluate_region(st, region, False)
        probe = st.clone()
        before = int(probe.victory_points)
        ts.Scoring.score_region(probe, region)
        applied = int(probe.victory_points) - before
        assert int(summary.us_score) - int(summary.ussr_score) == applied, (
            f"{region}: evaluate_region says {summary.us_score}-{summary.ussr_score} "
            f"but scoring it moved VP by {applied}"
        )


def test_region_score_sums_add_up_over_regions() -> None:
    st = _fresh()
    us, ussr = region_score_sums(st)
    assert us >= 0 and ussr >= 0
    from ai.eval.position_diagnostics import REGIONS
    assert us == sum(int(ts.Scoring.evaluate_region(st, r, False).us_score) for r in REGIONS)


def test_a_balanced_board_is_salvageable_even_when_both_sides_score_heavily() -> None:
    """One side worth 20 across Africa and Central America, the other 20 across South
    America and Asia, is a balanced board -- scoring it all would move the VP track by
    nothing. The bar is on the net, not on either side's absolute holdings."""
    from ai.eval.position_diagnostics import region_score_net, region_score_sums

    st = _fresh()
    us, ussr = region_score_sums(st)
    net = region_score_net(st)
    assert net == us - ussr
    # A large but symmetric board must not be rejected.
    assert is_salvageable(st, max_region_net=abs(net))


def test_salvageable_rejects_decided_and_terminal_positions() -> None:
    """A decided start contributes no gradient at all, so it must never enter the pool."""
    st = _fresh()
    assert is_salvageable(st), "the opening position should qualify"

    lopsided = _fresh()
    lopsided.victory_points = 15
    assert not is_salvageable(lopsided), "a 15 VP lead is past the |VP| <= 10 bar"

    over = _fresh()
    assert not is_salvageable(over, max_region_net=-1), "region-net bar must be enforced"

    done = _fresh()
    done.current_phase = ts.Phase.GAME_OVER
    assert not is_salvageable(done)


def test_profile_reports_per_turn_structure() -> None:
    """A short profile with a trivial policy still produces well-formed metrics."""
    import numpy as np
    from bindings.action_encoder import ActionEncoder

    def first_legal(state: ts.GameState, player: ts.Player):
        legal = np.flatnonzero(ActionEncoder.get_legal_mask(state))
        return int(legal[0]) if len(legal) else None

    profile = profile_self_play(first_legal, num_games=3, base_seed=77)
    assert profile["num_games"] == 3
    assert profile["per_turn"], "no turns recorded"
    first = profile["per_turn"][1]
    assert first["reached_frac"] == 1.0, "every game passes through turn 1"
    assert 0.0 <= first["salvageable_frac"] <= 1.0
    assert set(scalar_metrics(profile["per_turn"], 5.0)) == set(profile["scalars"])


def test_batched_profile_normalises_and_agrees_in_shape() -> None:
    """reached_frac is a fraction of episodes started, so it cannot exceed 1.

    The first version counted positions from envs still mid-game against a denominator of
    *completed* episodes, which pushed turn-1 "reached" above 160%. It also folded
    positions in as they happened, so stopping at N completions over-represented short
    games -- envs that end quickly finish first. Positions are now buffered per env and
    only counted when that episode ends.
    """
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from ai.eval.position_diagnostics import profile_self_play_batched

    profile = profile_self_play_batched(
        create_coldwar_net_v2("cpu"), num_envs=8, max_iters=4000)
    assert profile["num_games"] > 0, "no episode completed"

    per_turn = profile["per_turn"]
    assert per_turn[1]["reached_frac"] == 1.0, "every episode passes through turn 1"
    for turn, row in per_turn.items():
        assert 0.0 <= row["reached_frac"] <= 1.0, f"turn {turn}: {row['reached_frac']}"
        assert 0.0 <= row["salvageable_frac"] <= row["reached_frac"] + 1e-9
    turns = sorted(per_turn)
    fracs = [per_turn[t]["reached_frac"] for t in turns]
    assert fracs == sorted(fracs, reverse=True), "reachability must fall monotonically"


def test_batched_profile_samples_every_env_exactly_once() -> None:
    """The sample must be one episode per env, never "the first N episodes to finish".

    This is the invariant that was missing. The profiler used to stop once num_episodes
    episodes had completed, with num_episodes well below num_envs -- so the sample was the
    fastest N of num_envs games, and envs that auto-reset could be counted twice while slow
    envs were never counted at all. At N=30 over 128 envs that reported a mean final turn of
    3.30 and 0% of games reaching turn 9; draining the same policy unbiased gave 6.14 and
    21%. Every late-game metric was pinned to zero by construction, because games that run
    long are exactly the ones still in flight when the budget expires.

    num_games == num_envs is what makes that impossible to reintroduce: it can only hold if
    each env contributed its own single episode.
    """
    from ai.eval.position_diagnostics import profile_self_play_batched
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2

    num_envs = 12
    profile = profile_self_play_batched(
        create_coldwar_net_v2("cpu"), num_envs=num_envs, max_iters=20_000)

    assert profile["num_games"] == num_envs, (
        f"profiled {profile['num_games']} episodes across {num_envs} envs; the sample is "
        f"length-selected unless it is exactly one episode per env"
    )
    # reach[1] is the denominator, and every episode passes through turn 1.
    assert profile["per_turn"][1]["reached_frac"] == 1.0


def test_batched_profile_mean_turn_matches_an_independent_drain() -> None:
    """Cross-check the headline number against a drain written without the profiler.

    A stopping rule that truncates long games shows up here as a profiler mean well below
    the reference mean, which is exactly how the original bug presented (3.30 against 6.14).
    Both sides use the same deterministic policy and the same base seed, so they should
    agree exactly rather than merely closely.
    """
    import numpy as np
    import torch

    from ai.eval.position_diagnostics import profile_self_play_batched
    from bindings.ts_env import TsVectorizedEnv

    class FirstLegal(torch.nn.Module):
        """Deterministic: always take the lowest-indexed legal action."""

        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def sample_action(self, obs, mask, temperature: float = 1.0):
            actions = torch.argmax(mask.to(torch.int32), dim=1)
            z = torch.zeros(actions.shape[0])
            return actions, z, z, z, z

    num_envs, seed = 12, 4321
    profile = profile_self_play_batched(
        FirstLegal(), num_envs=num_envs, base_seed=seed, max_iters=20_000)

    # Reference: run the first episode of every env to completion, tracking its last turn.
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=seed)
    obs, masks, _ = env.reset_all()
    done_first = [False] * num_envs
    max_turn = [1] * num_envs
    for _ in range(20_000):
        if all(done_first):
            break
        for i in range(num_envs):
            if done_first[i]:
                continue
            st = env.runner.get_state(i)
            if not ts.Engine.is_terminal(st):
                max_turn[i] = max(max_turn[i], int(st.turn))
        actions = np.asarray(masks).argmax(axis=1)
        obs, masks, _, dones, _ = env.step(actions)
        for i, d in enumerate(dones):
            if d:
                done_first[i] = True

    reference = sum(max_turn) / len(max_turn)
    assert abs(profile["scalars"]["diag/mean_final_turn"] - reference) < 1e-6, (
        f"profiler reports mean final turn {profile['scalars']['diag/mean_final_turn']:.2f} "
        f"but an independent drain of the same policy and seeds gives {reference:.2f}; "
        f"a gap here means the sample is truncating long games"
    )
