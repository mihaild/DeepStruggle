"""Every key the engine's display and save dicts write is declared in the TypedDicts (audit V-10, 2026-11-22).

`replay_schema.golden.json` is generated from the TypedDicts, so it cannot see the engine adding a key they lack --
`headline_first_owner` (E9, audit E-7) went into `state_json::display_state` and nowhere else. Two checks, so drift
fails whichever way it comes:

* at run time, the dicts of real positions (seeded random games through every phase, Chernobyl in effect, Grain Sales'
  play-mode choice) are walked key by key against the TypedDict of each nesting level;
* in the source, every literal key `state_json.cpp`'s `display_state` / `save` sets, including the ones written only
  in rare states, must be declared somewhere in that dict's family of TypedDicts.

Engine keys must be a subset of the typed ones, not equal: the TypedDicts also carry what Python callers attach
(`observation_b64`, `action_logs`) and keys older replays hold.
"""
from __future__ import annotations

import pathlib
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple, get_type_hints

import ts_engine as ts

from web.server import replay_types as rt

STATE_JSON = pathlib.Path(__file__).resolve().parents[2] / "bindings" / "state_json.cpp"

#: Display dict: each nesting level's TypedDict, by its path from the top (`[]` a list's elements, `*` a map's values).
DISPLAY: Dict[Tuple[str, ...], Any] = {
    (): rt.GameStateDict,
    ("die_roll",): rt.DieRollDict,
    ("event_tracks", "[]"): rt.EventTrackDict,
    ("china_card",): rt.ChinaCardDict,
    ("countries", "*"): rt.CountryStateDict,
    ("hands",): rt.HandsDict,
    ("hands", "US_cards", "[]"): rt.HandCardDict,
    ("hands", "USSR_cards", "[]"): rt.HandCardDict,
    ("decision_context",): rt.DecisionContextDict,
    ("legal_actions",): rt.LegalActionsDict,
}
#: Maps keyed by data (a side, a country name, a card id), not by field names.
DISPLAY_MAPS: Set[Tuple[str, ...]] = {("mil_ops",), ("space",), ("space_turns_used",), ("countries",), ("card_locations",),
                ("legal_actions", "valid_action_labels")}
SAVE: Dict[Tuple[str, ...], Any] = {
    (): rt.SaveStateDict,
    ("last_roll",): rt.SaveRollDict,
    ("last_event_tracks", "[]"): rt.SaveEventTrackDict,
    ("ctx_stack", "[]"): rt.SaveDecisionFrameDict,
}
#: The keys of DISPLAY_MAPS' maps that are literal in the source (the two sides).
SIDE_KEYS = {"US", "USSR"}


def _typed(td: Any) -> Set[str]:
    return set(get_type_hints(td))


def _untyped(d: Mapping[str, Any], family: Dict[Tuple[str, ...], Any], maps: Set[Tuple[str, ...]],
             path: Tuple[str, ...] = ()) -> List[str]:
    """`path.key` for every key of `d` (recursively) that its level's TypedDict does not declare."""
    out: List[str] = []
    if path in maps:
        for k, v in d.items():
            if isinstance(v, dict):
                out += _untyped(v, family, maps, path + ("*",))
        return out
    td = family.get(path)
    if td is None:
        return [".".join(path) + " (a dict no TypedDict describes)"]
    declared = _typed(td)
    for k, v in d.items():
        if k not in declared:
            out.append(".".join(path + (k,)))
        if isinstance(v, dict):
            out += _untyped(v, family, maps, path + (k,))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for item in v:
                out += _untyped(item, family, maps, path + (k, "[]"))
    return out


def _positions() -> Iterable[ts.GameState]:
    """Real positions across whole random games, plus the states that write the conditional keys."""
    for seed in (3, 17, 29, 41):
        s = ts.GameState()
        ts.Engine.init_game(s, seed)
        rng = random.Random(seed)
        for n in range(4000):
            if ts.Engine.is_terminal(s):
                yield s
                break
            if n % 25 == 0:
                yield ts.state_from_save_json(s.to_save_json())
            ts.Engine.step_flat(s, rng.choice([i for i, v in enumerate(ts.Engine.get_flat_action_mask(s)) if v]))
    # Chernobyl in effect: `chernobyl_region`.
    s = ts.GameState()
    ts.Engine.init_game(s, 5)
    s.set_flag(ts.EffectBits.CHERNOBYL_ACTIVE)
    yield s
    # Grain Sales' play-mode choice on the drawn card: `legal_actions.un_intervention`.
    yield _grain_sales_play_mode()


def _grain_sales_play_mode() -> ts.GameState:
    gs = int(ts.CardData.get_card_by_name("Grain Sales to Soviets"))
    for seed in range(200):
        s = ts.GameState()
        ts.Engine.init_game(s, seed)
        rng = random.Random(seed)
        for _ in range(3000):
            if ts.Engine.is_terminal(s):
                break
            ctx = s.ctx()
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and ctx.resolving_card == gs:
                return s
            if (s.current_phase == ts.Phase.ACTION_ROUND and s.ctx_stack_depth == 0
                    and ctx.decision_type == ts.DecisionType.SELECT_CARD and s.phasing_player == ts.Player.US
                    and ctx.decision_player == ts.Player.US):
                s.set_card_location(gs, ts.CardLocation.HAND_US_UNKNOWN)
                ts.Engine.step_flat(s, ts.FLAT_CARD + gs - 1)
                if s.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE:
                    ts.Engine.step_flat(s, ts.FLAT_RESOLUTION)    # Grain Sales' Event: the drawn card is offered
                continue
            ts.Engine.step_flat(s, rng.choice([i for i, v in enumerate(ts.Engine.get_flat_action_mask(s)) if v]))
    raise AssertionError("Grain Sales' play-mode choice not reached")


def test_every_key_of_real_display_and_save_dicts_is_typed() -> None:
    seen_display: Set[str] = set()
    for s in _positions():
        d = ts.state_to_dict(s)
        seen_display |= set(d) | set(d["legal_actions"])
        assert _untyped(d, DISPLAY, DISPLAY_MAPS) == [], "display keys the TypedDicts lack"
        assert _untyped(s.to_save_dict(), SAVE, set()) == [], "save keys the TypedDicts lack"
    # The conditional keys were really exercised.
    assert {"headline_first_owner", "chernobyl_region", "un_intervention"} <= seen_display


def _function_keys(name: str) -> Set[str]:
    """Every literal key `state_json.cpp`'s function `name` sets (`.set("key", ...)`)."""
    src = STATE_JSON.read_text(encoding="utf-8")
    start = src.index(f"Value {name}(const GameState& state) {{")
    body = src[start:src.index("\n}\n", start)]
    return set(re.findall(r'\.set\("([A-Za-z0-9_]+)"', body))


def test_every_key_the_source_writes_is_typed() -> None:
    """Also the keys only rare states write (P43's canonical fields), which no random game reaches."""
    display = set().union(*(_typed(td) for td in DISPLAY.values())) | SIDE_KEYS
    save = set().union(*(_typed(td) for td in SAVE.values()))
    missing_display = sorted(_function_keys("display_state") - display)
    missing_save = sorted(_function_keys("save") - save)
    assert not missing_display, f"display_state writes keys no display TypedDict declares: {missing_display}"
    assert not missing_save, f"save writes keys no save TypedDict declares: {missing_save}"
    assert {"headline_first_owner", "chernobyl_region", "un_intervention"} <= _function_keys("display_state")
    assert {"headline_second_owner", "canon_floor", "canon_prio"} <= _function_keys("save")
