"""The setup probe measures placements, not noise.

Every number this probe reports is a difference between the board before the setup block and the
board after it, so the things that can silently break it are structural: the block being a
different length than assumed, the difference being taken against the wrong baseline, or the
policy never actually being consulted. A probe that got any of those wrong would still print a
plausible table, which is exactly how the last unchecked probe went wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts
from bindings.action_encoder import ActionEncoder

from ai.eval.setup_probe import (
    POLAND,
    SETUP_DECISIONS,
    SETUP_TARGETS,
    WEST_GERMANY,
    all_targets_met,
    measure,
    target_rates,
    uniform_policy,
)


def test_the_setup_block_is_exactly_fifteen_decisions() -> None:
    """6 USSR + 7 US + 2 US bonus. The probe steps a fixed count, so this is load-bearing."""
    runner = ts.VectorizedBatchRunner(2, 4242)
    rng = np.random.default_rng(0)
    seen = 0
    while runner.get_state(0).current_phase == ts.Phase.SETUP:
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) for m in masks],
                             auto_advance=False)
        seen += 1
        assert seen <= 40, "setup did not terminate"
    assert seen == SETUP_DECISIONS


def test_each_side_places_its_full_allotment() -> None:
    """The totals are fixed by the rules, so they are the arithmetic check on the difference."""
    m = measure(uniform_policy(0), num_games=32, batch_size=32)
    assert m.malformed == 0
    assert (m.ussr.influence.sum(axis=1) == 6).all(), m.ussr.influence.sum(axis=1)[:5]
    assert (m.us.influence.sum(axis=1) == 9).all(), m.us.influence.sum(axis=1)[:5]
    assert (m.ussr.influence >= 0).all() and (m.us.influence >= 0).all(), \
        "a negative placement means the baseline was taken after the block, not before"


def test_the_ussr_places_only_in_eastern_europe() -> None:
    """The engine's own mask decides this; if the probe read the wrong side it would not hold."""
    m = measure(uniform_policy(1), num_games=16, batch_size=16)
    placed = np.flatnonzero(m.ussr.influence.sum(axis=0))
    for cid in placed:
        assert ts.MapData.get_country_info(int(cid))["in_eastern_europe"], \
            f"USSR placed in {ts.MapData.get_country_info(int(cid))['name']}, not Eastern Europe"


def test_a_fixed_policy_is_distinguishable_from_a_random_one() -> None:
    """The point of the probe: two policies that open differently must not report the same.

    A probe that ignored the policy -- stepping the engine itself, say -- would give these two
    identical distributions, and every arm-to-arm comparison built on it would be worthless.
    """
    def always_poland(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        del obs
        out = []
        for row in masks:
            legal = np.flatnonzero(row)
            want = ActionEncoder.NODE_OFFSET + POLAND
            out.append(want if row[want] else legal[0])
        return np.array(out)

    fixed = measure(always_poland, num_games=32, batch_size=32)
    random = measure(uniform_policy(2), num_games=32, batch_size=32)

    assert fixed.ussr.rate_at_least(POLAND, 3)[0] == 1.0, \
        "a policy that always picks Poland must place 3 there every game"
    assert random.ussr.rate_at_least(POLAND, 3)[0] < 0.5
    assert fixed.ussr.entropy() < random.ussr.entropy()


def test_europe_scoring_is_read_before_any_placement() -> None:
    """The conditioning is only free if the hand is known at the setup node -- check it is."""
    runner = ts.VectorizedBatchRunner(8, 77)
    st = runner.get_state(0)
    assert st.current_phase == ts.Phase.SETUP
    holders = [ts.hand_holder(runner.get_state(i).get_card_location(2)) for i in range(8)]
    assert any(h != ts.Player.NONE for h in holders), \
        "Europe Scoring was in nobody's hand in eight fresh games; the deal has not happened yet"


def test_the_margin_is_read_from_the_finished_board_not_the_placement() -> None:
    """East Germany starts at 3 against stability 3: held, with nothing to spare.

    A probe that read the placement alone would score one influence there as "1" and miss that
    the country went from takeable-by-one to buffered. That is the whole reason the standard
    opening puts a point in, so the probe has to see it.
    """
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    assert int(st.get_country(14).ussr_influence) == 3
    assert ts.MapData.get_country_info(14)["stability"] == 3
    assert ts.Scoring.get_country_control(st, 14) == ts.Player.USSR, "East Germany moved"

    m = measure(uniform_policy(3), num_games=64, batch_size=64)
    placed = m.ussr.influence[:, 14].astype(np.int16)
    final = m.ussr.final[:, 14].astype(np.int16)
    assert (final == placed + 3).all(), "the finished board must include the starting influence"

    margin = m.ussr.margin()[:, 14]
    assert (margin == placed).all(), \
        "with the US at zero and stability 3, the margin in East Germany is what was placed"
    # A game that placed nothing there holds it at exactly control, and is counted as fragile.
    none_placed = placed == 0
    if none_placed.any():
        assert (m.ussr.fragile() > 0)[none_placed].all()


@pytest.mark.parametrize("target", SETUP_TARGETS, ids=lambda t: t.name)
def test_every_target_is_its_countrys_control_threshold(target) -> None:
    """Each bar is `stability`, which with the opponent at zero is exactly what control costs.

    If a country's stability ever changes, the bar has to move with it -- otherwise the probe
    goes on reporting a threshold that no longer means "controlled".
    """
    info = ts.MapData.get_country_info(target.cid)
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    opponent = (int(st.get_country(target.cid).ussr_influence) if target.side == "US"
                else int(st.get_country(target.cid).us_influence))
    assert opponent == 0, f"{target.name} does not start empty for the opponent"
    assert target.threshold == int(info["stability"]), (
        f"{target.name} has stability {info['stability']} but the target is "
        f"{target.threshold}; control costs the stability")


def test_the_composite_is_the_conjunction_of_the_targets() -> None:
    """"All four" must be per game, not the product of four independent rates."""
    m = measure(uniform_policy(4), num_games=64, batch_size=64)
    rate = all_targets_met(m)[0]

    expected = np.ones(m.games, dtype=bool)
    for t in SETUP_TARGETS:
        side = m.ussr if t.side == "USSR" else m.us
        expected &= (side.final[:, t.cid] >= t.threshold)
    assert rate == pytest.approx(expected.mean())

    for t, r, _lo, _hi in target_rates(m):
        assert rate <= r + 1e-9, f"the composite cannot exceed {t.name}'s own rate"
