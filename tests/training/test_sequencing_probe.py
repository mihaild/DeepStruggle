"""The card-disposal probe, checked against the shape of the position it was written from.

A probe that does not catch the case that motivated it catches nothing. `experiments.md` §25 is
that case: the USSR holds UN Intervention and two US cards it must not play, at a space box
needing 3 Ops, and only one of the two can space itself. It spent the scarce exit on the card
that had its own.

The §25 hand was Tear Down this Wall and Grain Sales. **This file uses Duck and Cover in place of
Tear Down this Wall**, because `blunders.defcon_suicide_cards` -- the project's one definition of
"a card you must not play here" -- does not list Tear Down this Wall, and a test that asserted
otherwise would be asserting against a definition the probe does not use. Duck and Cover is 3 Ops
and US-associated, so the structure is identical: two banned US cards, one spaceable at this box
and one not. Whether Tear Down this Wall belongs in that list is a question about `blunders.py`,
noted where the probe is reported rather than settled here.
"""

from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.positions import PositionBuilder, card_action
from ai.eval.sequencing import (
    SPACE_ACTION,
    UN_INTERVENTION,
    DisposalCounts,
    is_spaceable,
    is_un_intervention_companion_node,
    measure,
    observe,
)

#: 3 Ops, US-associated, and on the DEFCON-suicide list: the spaceable half of the pair.
DUCK_AND_COVER = 4
#: 2 Ops, US-associated, and on the list: the half with no way out at this box.
GRAIN_SALES = 67

#: Box 5 needs 3 Ops, so a track at 4 is the §25 situation: the 3-Ops card can space and the
#: 2-Ops card cannot.
SPACE_TRACK_AT_FOUR = 4


def _the_section_25_hand() -> ts.GameState:
    """The §25 structure, with a foothold that makes the two cards actually dangerous.

    `defcon_suicide_cards` bans the "opponent gets Operations and coups" cards only when there is
    something to coup -- USSR influence in Africa or the Americas. Without it the position is not
    the one §25 describes, so the probe would be scored against a hand with nothing at stake.
    """
    nigeria = next(c for c in range(84)
                   if ts.MapData.get_country_info(c)["name"] == "Nigeria")
    return PositionBuilder(
        hand=[UN_INTERVENTION, DUCK_AND_COVER, GRAIN_SALES],
        side=ts.Player.USSR, turn=8, defcon=2, ussr_space=SPACE_TRACK_AT_FOUR,
        influence=[(nigeria, ts.Player.USSR, 2)],
    ).build()


def _at_the_companion_node(counts: DisposalCounts) -> tuple[ts.GameState, dict]:
    """Step the §25 hand to where UN Intervention asks which card to spend itself on.

    Driven exactly as `measure` drives it, including the carry: the spaceability answer is taken
    at the decision where UN Intervention is chosen, because at the companion node a card action
    means something else entirely.
    """
    carry: dict = {}
    state = _the_section_25_hand()
    observe(state, card_action(UN_INTERVENTION), counts, carry)
    assert carry.get("spaceable"), "spaceability should have been recorded at the card choice"

    ts.Engine.step_flat(state, card_action(UN_INTERVENTION))
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    ts.Engine.step_flat(state, 110 + int(ts.PlayMode.EVENT))
    assert is_un_intervention_companion_node(state)
    return state, carry


def test_the_position_reproduces_at_all() -> None:
    """Both cards must be DEFCON-suicide here, or the probe is measuring a different position."""
    from ai.eval.blunders import defcon_suicide_cards

    state = _the_section_25_hand()
    banned = defcon_suicide_cards(state, ts.Player.USSR)
    assert DUCK_AND_COVER in banned and GRAIN_SALES in banned, (
        f"both US cards must read as DEFCON-suicide for the position to be the one §25 "
        f"describes; banned here: {sorted(banned)}")


def test_spaceability_is_the_distinction() -> None:
    """3 Ops reaches box 5 and 2 Ops does not. This is the whole of the §25 error."""
    state = _the_section_25_hand()
    assert is_spaceable(state, DUCK_AND_COVER), "Duck and Cover must be spaceable here"
    assert not is_spaceable(state, GRAIN_SALES), "Grain Sales must not be spaceable here"


def test_spaceability_is_read_from_the_engine_not_from_printed_ops() -> None:
    """With the track one box lower, the 2-Ops card becomes spaceable and the answer flips.

    Printed Ops does not move; what the engine allows does. A probe that compared Ops against a
    constant would give the same answer in both positions.
    """
    nigeria = next(c for c in range(84)
                   if ts.MapData.get_country_info(c)["name"] == "Nigeria")
    low = PositionBuilder(
        hand=[UN_INTERVENTION, DUCK_AND_COVER, GRAIN_SALES],
        side=ts.Player.USSR, turn=8, defcon=2, ussr_space=0,
        influence=[(nigeria, ts.Player.USSR, 2)],
    ).build()
    assert is_spaceable(low, GRAIN_SALES), "box 1 needs 2 Ops, which Grain Sales has"


def test_the_section_25_mistake_is_counted() -> None:
    """Spending UN Intervention on the card that could have spaced itself."""
    counts = DisposalCounts()
    state, carry = _at_the_companion_node(counts)
    observe(state, card_action(DUCK_AND_COVER), counts, carry)

    assert counts.spent_on_spaceable.opportunities == 1
    assert counts.spent_on_spaceable.mistakes == 1, (
        "the §25 error must be counted: UN Intervention went to the card with another exit")
    # It is also, trivially, a use of the tool on *a* suicide card, so the looser measure counts
    # the opportunity and no mistake.
    assert counts.spent_elsewhere.opportunities == 1
    assert counts.spent_elsewhere.mistakes == 0


def test_the_correct_play_is_not_counted() -> None:
    """Spending it on the card with no other way out is the right answer and scores clean."""
    counts = DisposalCounts()
    state, carry = _at_the_companion_node(counts)
    observe(state, card_action(GRAIN_SALES), counts, carry)

    assert counts.spent_on_spaceable.opportunities == 1
    assert counts.spent_on_spaceable.mistakes == 0
    assert counts.spent_elsewhere.mistakes == 0


def test_playing_the_suicide_card_raw_is_counted() -> None:
    """The outcome the first two measures are upstream of."""
    state = _the_section_25_hand()
    ts.Engine.step_flat(state, card_action(GRAIN_SALES))
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    assert int(state.ctx().pending_op_card) == GRAIN_SALES

    counts = DisposalCounts()
    observe(state, 110 + int(ts.PlayMode.OPS), counts)
    assert counts.played_raw.opportunities == 1
    assert counts.played_raw.mistakes == 1


def test_spacing_the_suicide_card_is_the_exit_and_is_not_counted() -> None:
    state = _the_section_25_hand()
    ts.Engine.step_flat(state, card_action(DUCK_AND_COVER))
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE

    counts = DisposalCounts()
    observe(state, SPACE_ACTION, counts)
    assert counts.played_raw.opportunities == 1
    assert counts.played_raw.mistakes == 0, "spacing it is exactly the way out, not a mistake"


def test_nothing_is_counted_above_defcon_2() -> None:
    """'A card you must not play' is a DEFCON-2 statement; at DEFCON 3 there is no bar."""
    nigeria = next(c for c in range(84)
                   if ts.MapData.get_country_info(c)["name"] == "Nigeria")
    state = PositionBuilder(
        hand=[UN_INTERVENTION, DUCK_AND_COVER, GRAIN_SALES],
        side=ts.Player.USSR, turn=8, defcon=3, ussr_space=SPACE_TRACK_AT_FOUR,
        influence=[(nigeria, ts.Player.USSR, 2)],
    ).build()
    ts.Engine.step_flat(state, card_action(GRAIN_SALES))
    counts = DisposalCounts()
    observe(state, 110 + int(ts.PlayMode.OPS), counts)
    assert counts.played_raw.opportunities == 0


def test_the_probe_runs_over_real_games() -> None:
    """End to end on random play: it must not raise, and rates stay within their denominators."""
    rng = np.random.default_rng(0)

    def random_policy(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        del obs
        out: list[int] = []
        for row in masks:
            legal = np.flatnonzero(row)
            out.append(int(rng.choice(legal)) if legal.size else 0)
        return np.array(out)

    counts = measure(random_policy, num_games=8, batch_size=8, max_steps=2000)
    assert counts.games == 8
    for r in counts.all_rates():
        assert 0 <= r.mistakes <= r.opportunities
