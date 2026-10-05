"""Play-mode names come from ts.Resolution, everywhere a SELECT_PLAY_MODE id is named.

P17 reordered the enum -- SPACE 2 -> 1, the single OPS became OPS_INFLUENCE / OPS_COUP /
OPS_REALIGN -- and three hand-written copies of the old `{0: EVENT, 1: OPS, 2: SPACE}` survived it:
match replays labelled every Influence play "Space Race", the self-play blunder check scored them
as space race plays, and the DEFCON-ending log named them SPACE.
"""

from __future__ import annotations

import ts_engine as ts

from ai.eval.blunders import _play_mode
from ai.eval.defcon_endings import _mode_name
from tools.play_match import format_action_description

R = ts.Resolution


def test_match_replay_names_every_play_mode() -> None:
    def name(r: "ts.Resolution") -> str:
        return format_action_description({"decision_type": int(ts.DecisionType.SELECT_PLAY_MODE),
                                          "primary_id": int(r)}, {})
    assert name(R.EVENT) == "Plays as Event"
    assert name(R.SPACE) == "Plays as Space Race"
    assert name(R.OPS_INFLUENCE) == "Plays as Operations (Influence)"
    assert name(R.OPS_COUP) == "Plays as Operations (Coup)"
    assert name(R.OPS_REALIGN) == "Plays as Operations (Realignment)"


def test_blunder_and_ending_modes_follow_the_enum() -> None:
    expect = {R.EVENT: "EVENT", R.SPACE: "SPACE", R.OPS_INFLUENCE: "OPS", R.OPS_COUP: "OPS",
              R.OPS_REALIGN: "OPS"}
    for r, mode in expect.items():
        assert _play_mode(int(r)) == mode
        assert _mode_name(int(r)) == mode
