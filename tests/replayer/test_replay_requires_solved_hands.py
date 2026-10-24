"""Unavailable or incomplete hand solving must never activate approximate conversion."""
from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from tools.lib import ts_replayer_convert as converter
from tools.lib import ts_replayer_hands as solver
from tools.lib.corpus_paths import corpus_path


def test_empty_download_needs_no_hand_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solver, "HAVE_Z3", False)
    conv = converter.convert_game({"replay_id": 214, "all_turns": []})
    assert not conv.samples and conv.failure is None and not conv.game_ended
    assert conv.entries_converted == 0 and not conv.hands_solved


@pytest.mark.parametrize("mode", ["unsolved", "missing-dependency", "missing-turn"])
@pytest.mark.parametrize("fragment", [False, True])
def test_hand_solver_failure_emits_nothing(monkeypatch: pytest.MonkeyPatch, mode: str, fragment: bool) -> None:
    with gzip.open(corpus_path(243), "rt") as stream:
        game = json.load(stream)
    if fragment:
        game["all_turns"] = game["all_turns"][:1]
    if mode == "missing-dependency":
        monkeypatch.setattr(solver, "HAVE_Z3", False)
    else:
        monkeypatch.setattr(converter, "solve_hands", lambda *args, **kwargs: {} if mode == "missing-turn" else None)
    decisions = []

    def capture(*args: Any) -> None:
        decisions.append(args)

    conv = converter.convert_game(game, on_decision=capture)
    assert conv.hands_solved == (mode == "missing-turn")
    assert not decisions and not conv.samples and conv.decisions_emitted == 0
    assert conv.failure is not None and conv.failure.kind == "hand reconstruction failed"
    assert conv.failure.replay_id == 243 and conv.failure.turn == 1
    assert conv.truncated_at is None
