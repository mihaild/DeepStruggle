"""A forced die belongs to the player who rolls it, whichever player that is.

The engine used to accept a die at the POINT_NODE that named a target, stash the pair in
`temp_cards[2]`/`[3]` *per side* -- [2] the US's, [3] the USSR's -- and then read them back at
the chance node *per actor*: `forced_us = (actor == US) ? first : second`. Those two conventions
agree for a US realignment and disagree for a USSR one, so a USSR realignment came out with its
two dice transposed. It went unnoticed because the only test exercising that path realigned as
the US, where the transposition is the identity.

The fix is not a corrected mapping. Forced dice are a test and replayer affordance rather than
game state, so nothing is stored: the ROLL_DIE action carries both, `primary_id` for the acting
player and `secondary_id` for the opponent, and that is the one place either is read. With no
stored pair there is no pair to transpose.

These tests check both directions. A one-sided test passes just as well against an engine that
ignores forced dice entirely, which is the other way to get this wrong.

Which of them is the negative control is worth stating, because it is not the obvious one. The
transposition only ever affected dice *stashed at the POINT_NODE*; dice passed on the ROLL_DIE
action were mapped correctly even before, which is why the replayer -- which passes them there
-- never saw it. So `test_the_acting_player_gets_the_die_they_were_given` passes against the old
engine too, and it is `test_a_target_choice_carries_no_die` that fails there: the old engine
honoured those dice, so the three pairs it is given produce three different boards instead of
one.
"""

from __future__ import annotations

from typing import Tuple

import pytest

import ts_engine as ts

INDIA = 33
DUCK_AND_COVER = 4


def _realign(realigner: ts.Player, actor_die: int, opponent_die: int) -> Tuple[int, int, int, int]:
    """Realign India as `realigner`, forcing both dice at the chance node.

    India starts even at 3/3, so the roll alone decides it. Returns
    (us_roll, ussr_roll, us_influence, ussr_influence) afterwards.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    state.current_phase = ts.Phase.ACTION_ROUND
    state.turn = 1
    state.action_round = 1
    state.phasing_player = realigner
    ctx = state.ctx()
    ctx.decision_player = realigner
    ctx.decision_type = ts.DecisionType.POINT_NODE
    ctx.op_mode = ts.OpMode.REALIGN
    ctx.remaining_steps = 1
    ctx.pending_ops_value = 1
    ctx.pending_op_card = DUCK_AND_COVER
    ctx.allow_early_stop = 1
    state.set_country(INDIA, 3, 3)

    assert ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, INDIA, 0, 0))
    assert state.ctx().pending_roll_type == ts.RollType.REALIGNMENT
    assert ts.Engine.step(
        state, ts.MicroAction(ts.DecisionType.ROLL_DIE, actor_die, opponent_die, 0))

    record = state.to_dict()["die_roll"]
    country = state.get_country(INDIA)
    return (int(record["roll1"]), int(record["roll2"]),
            int(country.us_influence), int(country.ussr_influence))


@pytest.mark.parametrize("realigner,name", [(ts.Player.US, "US"), (ts.Player.USSR, "USSR")])
def test_the_acting_player_gets_the_die_they_were_given(realigner: ts.Player, name: str) -> None:
    """primary_id is the actor's die, whether the actor is the US or the USSR."""
    us_roll, ussr_roll, us_inf, ussr_inf = _realign(realigner, actor_die=6, opponent_die=1)

    actor_roll = us_roll if realigner == ts.Player.US else ussr_roll
    other_roll = ussr_roll if realigner == ts.Player.US else us_roll
    assert (actor_roll, other_roll) == (6, 1), (
        f"{name} realigned with a forced 6 against a forced 1 and the engine rolled "
        f"us={us_roll} ussr={ussr_roll}; the dice reached the wrong sides"
    )

    # 6 against 1 on an even country: the actor wins by 5 and clears the opponent out.
    if realigner == ts.Player.US:
        assert (us_inf, ussr_inf) == (3, 0)
    else:
        assert (us_inf, ussr_inf) == (0, 3)


@pytest.mark.parametrize("realigner", [ts.Player.US, ts.Player.USSR])
def test_the_dice_are_not_simply_ignored(realigner: ts.Player) -> None:
    """Swapping the two dice must swap the outcome -- otherwise nothing is being forced."""
    _, _, us_win, ussr_win = _realign(realigner, actor_die=6, opponent_die=1)
    _, _, us_lose, ussr_lose = _realign(realigner, actor_die=1, opponent_die=6)
    assert (us_win, ussr_win) != (us_lose, ussr_lose), (
        "the same realignment produced the same board with the dice reversed, so the forced "
        "dice are reaching nothing"
    )


def test_a_target_choice_carries_no_die() -> None:
    """The POINT_NODE's spare fields are ignored; only the ROLL_DIE supplies a die.

    This is the property that makes a transposition unwritable: there is nowhere to stash a
    die between naming a target and rolling for it.
    """
    seen = set()
    for actor_die, opponent_die in ((6, 1), (1, 6), (3, 3)):
        state = ts.GameState()
        ts.Engine.init_game(state, 777)
        state.current_phase = ts.Phase.ACTION_ROUND
        state.turn = 1
        state.action_round = 1
        state.phasing_player = ts.Player.USSR
        ctx = state.ctx()
        ctx.decision_player = ts.Player.USSR
        ctx.decision_type = ts.DecisionType.POINT_NODE
        ctx.op_mode = ts.OpMode.REALIGN
        ctx.remaining_steps = 1
        ctx.pending_ops_value = 1
        ctx.pending_op_card = DUCK_AND_COVER
        ctx.allow_early_stop = 1
        state.set_country(INDIA, 3, 3)
        # dice offered where they used to be honoured -- on the target choice
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, INDIA,
                                             actor_die, opponent_die))
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        record = state.to_dict()["die_roll"]
        seen.add((int(record["roll1"]), int(record["roll2"])))

    assert len(seen) == 1, (
        f"the dice on the target choice changed the outcome ({sorted(seen)}); they must be "
        f"ignored, leaving the roll to the engine's own stream"
    )
