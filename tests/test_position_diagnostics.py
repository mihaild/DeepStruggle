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
        create_coldwar_net_v2("cpu"), num_envs=8, num_episodes=8, max_iters=4000)
    assert profile["num_games"] > 0, "no episode completed"

    per_turn = profile["per_turn"]
    assert per_turn[1]["reached_frac"] == 1.0, "every episode passes through turn 1"
    for turn, row in per_turn.items():
        assert 0.0 <= row["reached_frac"] <= 1.0, f"turn {turn}: {row['reached_frac']}"
        assert 0.0 <= row["salvageable_frac"] <= row["reached_frac"] + 1e-9
    turns = sorted(per_turn)
    fracs = [per_turn[t]["reached_frac"] for t in turns]
    assert fracs == sorted(fracs, reverse=True), "reachability must fall monotonically"
