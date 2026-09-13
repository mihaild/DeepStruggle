"""Guards for the two probes that read state back out of a running batch.

Both were wrong in the same family of ways on first write, and both failures produced numbers
that looked like results. These pin the two facts that make them correct.
"""
from __future__ import annotations

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.forced_setup import (NODE_OFFSET, SETUP_DECISIONS, US_OPENING, USSR_OPENING,
                                  WEST_GERMANY, _expand)
from ai.eval.held_scoring import ANY_HAND, HANDS, SCORING_CARDS
from tools.lib.openings import OPENINGS, acting_side, expand, scripted_setup_index


def test_scoring_card_set_is_the_seven() -> None:
    assert len(SCORING_CARDS) == 7
    names = {str(ts.CardData.get_card_name(c)) for c in SCORING_CARDS}
    assert "Europe Scoring" in names and "Asia Scoring" in names


def test_hand_locations_cover_the_known_half() -> None:
    """A hand is two CardLocations. Testing only the UNKNOWN half misses stranded cards."""
    for side in ("US", "USSR"):
        assert len(HANDS[side]) == 2
    assert ts.CardLocation.HAND_US_KNOWN in HANDS["US"]
    assert ts.CardLocation.HAND_USSR_KNOWN in HANDS["USSR"]
    assert set(ANY_HAND) == set(HANDS["US"]) | set(HANDS["USSR"])
    # hand_of() alone is the unknown half, which is precisely the bug being pinned.
    assert ts.hand_of(ts.Player.US) not in (ts.CardLocation.HAND_US_KNOWN,)


def test_get_state_is_a_live_view_and_clone_detaches() -> None:
    """The aliasing that made every held-scoring ending read back as turn 1."""
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=4, base_seed=11)
    _, masks, _ = env.reset_all()
    live = env.runner.get_state(0)
    snap = live.clone()
    before = (int(live.turn), int(live.action_round))

    for _ in range(40):
        acts = np.array([int(np.flatnonzero(np.asarray(m))[0]) for m in masks], dtype=np.int64)
        _, masks, _, _, _ = env.step(acts)

    after = (int(live.turn), int(live.action_round))
    assert after != before, "get_state stopped aliasing -- the clone() calls can be revisited"
    assert (int(snap.turn), int(snap.action_round)) == before, "clone() must detach"


def test_opening_scripts_have_the_right_point_counts() -> None:
    assert len(_expand(USSR_OPENING)) == 6, "USSR places 6 in Eastern Europe"
    assert len(_expand(US_OPENING)) == 9, "US places 7 in Western Europe plus 2 bonus"
    assert len(_expand(USSR_OPENING)) + len(_expand(US_OPENING)) == SETUP_DECISIONS
    assert _expand(US_OPENING).count(WEST_GERMANY) == 4


def test_setup_decision_player_is_not_the_phasing_player() -> None:
    """Keying the script off phasing_player sent every US placement down the USSR script."""
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=8, base_seed=23)
    _, masks, _ = env.reset_all()
    seen_decision, seen_phasing = set(), set()
    for _ in range(SETUP_DECISIONS):
        seen_decision.update(int(p) for p in env.runner.get_decision_players())
        seen_phasing.update(int(env.runner.get_state(i).phasing_player) for i in range(8))
        acts = np.array([int(np.flatnonzero(np.asarray(m))[0]) for m in masks], dtype=np.int64)
        _, masks, _, _, _ = env.step(acts)

    assert len(seen_decision) == 2, "both sides place during setup"
    assert seen_phasing != seen_decision, (
        "phasing_player no longer tracks the whole of setup -- re-check forced_setup's side key")


def test_forced_opening_is_legal_end_to_end() -> None:
    """Every scripted placement must be in the mask; off_script is a hard failure."""
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=8, base_seed=37)
    _, masks, _ = env.reset_all()
    cursor = {"US": [0] * 8, "USSR": [0] * 8}
    scripts = {"US": _expand(US_OPENING), "USSR": _expand(USSR_OPENING)}

    for _ in range(SETUP_DECISIONS):
        m = np.asarray(masks)
        players = env.runner.get_decision_players()
        acts = np.zeros(8, dtype=np.int64)
        for i in range(8):
            side = "US" if int(players[i]) == int(ts.Player.US) else "USSR"
            k = cursor[side][i]
            assert k < len(scripts[side]), f"{side} script exhausted -- side key is wrong"
            cand = NODE_OFFSET + scripts[side][k]
            assert m[i][cand], f"{side} placement {k} illegal at action {cand}"
            cursor[side][i] = k + 1
            acts[i] = cand
        _, masks, _, _, _ = env.step(acts)

    for i in range(8):
        st = env.runner.get_state(i)
        assert st.current_phase != ts.Phase.SETUP, "setup should be over after 15 decisions"
        assert int(st.get_country(WEST_GERMANY).us_influence) == 4


@pytest.mark.parametrize("side", ["US", "USSR"])
def test_openings_place_only_in_their_own_half(side: str) -> None:
    """A misdirected script would be caught by the mask, but name the intent too."""
    from ai.eval.forced_setup import EAST_GERMANY, IRAN, ITALY, POLAND, YUGOSLAVIA

    ussr_side = {EAST_GERMANY, POLAND, YUGOSLAVIA}
    us_side = {WEST_GERMANY, ITALY, IRAN}
    placed = set(_expand(USSR_OPENING if side == "USSR" else US_OPENING))
    assert placed == (ussr_side if side == "USSR" else us_side)


def test_probe_and_replay_generators_share_one_opening() -> None:
    """The probe's numbers and the replays must describe the same opening."""
    assert OPENINGS["human"]["US"] == expand(US_OPENING)
    assert OPENINGS["human"]["USSR"] == expand(USSR_OPENING)
    assert _expand is expand


def test_scripted_setup_index_drives_a_single_game_to_the_human_board() -> None:
    """End to end through the engine, the way the replay generators call it."""
    state = ts.GameState()
    ts.Engine.init_game(state, 4242)
    cursor = {"US": 0, "USSR": 0}

    for _ in range(SETUP_DECISIONS):
        assert state.current_phase == ts.Phase.SETUP
        idx = scripted_setup_index(state, acting_side(state), "human", cursor)
        assert idx is not None, "script ran out before setup ended"
        ts.Engine.step(state, ts.decode_flat_action(state, idx))

    assert state.current_phase != ts.Phase.SETUP
    assert cursor == {"US": 9, "USSR": 6}
    # The finished board, not the placements: two of these countries are not empty at the
    # start, so the scripted points are added to what is already there. East Germany starts
    # at 3 USSR and Iran at 1 US, which is why +1 and +2 read as 4 and 3.
    assert int(state.get_country(WEST_GERMANY).us_influence) == 4
    assert int(state.get_country(15).ussr_influence) == 4      # Poland, +4 from empty
    assert int(state.get_country(14).ussr_influence) == 4      # East Germany, 3 + 1
    assert int(state.get_country(18).ussr_influence) == 1      # Yugoslavia, +1 from empty
    assert int(state.get_country(10).us_influence) == 3        # Italy, +3 from empty
    assert int(state.get_country(25).us_influence) == 3        # Iran, 1 + 2


def test_scripted_setup_index_returns_none_outside_setup() -> None:
    """None means 'let the agent decide' -- it must never keep firing after setup."""
    state = ts.GameState()
    ts.Engine.init_game(state, 99)
    cursor = {"US": 0, "USSR": 0}
    for _ in range(SETUP_DECISIONS):
        idx = scripted_setup_index(state, acting_side(state), "human", cursor)
        assert idx is not None
        ts.Engine.step(state, ts.decode_flat_action(state, idx))
    assert scripted_setup_index(state, acting_side(state), "human", cursor) is None
