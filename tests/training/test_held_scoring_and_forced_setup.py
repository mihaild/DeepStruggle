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
