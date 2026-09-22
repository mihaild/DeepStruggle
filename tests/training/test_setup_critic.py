"""The setup-critic probe compares two boards that differ only in the US setup.

Its whole claim rests on three structural facts: each opening is placed exactly as written, setup
has ended when the critic is read, and the deal is the same under both openings. Any of those
failing would still print a plausible table.
"""

from __future__ import annotations

import pytest
import ts_engine as ts

from ai.eval.setup_critic import after_setup, deal
from tools.lib.openings import OPENINGS

PAIR = ("ph_west_germany", "ph_no_west_germany")


def _influence(state: ts.GameState) -> dict:
    out = {}
    for cid in range(84):
        c = state.get_country(cid)
        if c.us_influence or c.ussr_influence:
            out[cid] = (int(c.us_influence), int(c.ussr_influence))
    return out


def _expected(opening: str) -> dict:
    """The fixed starting influence plus what the opening places."""
    base = _influence(_initial())
    for side, k in (("US", 0), ("USSR", 1)):
        for cid in OPENINGS[opening][side]:
            u, s = base.get(cid, (0, 0))
            base[cid] = (u + 1, s) if k == 0 else (u, s + 1)
    return base


def _initial() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 101)
    return s


@pytest.mark.parametrize("opening", PAIR)
def test_each_opening_is_placed_exactly_as_written(opening: str) -> None:
    state = after_setup(101, opening)
    assert _influence(state) == _expected(opening)
    assert state.current_phase != ts.Phase.SETUP


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_the_deal_does_not_depend_on_the_opening(seed: int) -> None:
    assert deal(after_setup(seed, PAIR[0])) == deal(after_setup(seed, PAIR[1]))


def test_different_seeds_are_different_deals() -> None:
    """Otherwise 'eight deals' would be one deal eight times."""
    assert deal(after_setup(101, PAIR[0])) != deal(after_setup(102, PAIR[0]))
