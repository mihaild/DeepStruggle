"""The Playdek-scale rating fit of tools/lib/game_stats.py, on games whose truth is known.

Games are drawn from the model the fit assumes -- Elo's logistic against each human's rating --
for a model of known strength, and the fit must find it. Grouping is tested with synthetic
verified submissions; tests/web/test_game_stats.py checks rating real played games end to end.
"""
from __future__ import annotations

import base64
import json
import math
import struct
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from tools import game_stats as cli
from tools.lib.game_stats import (RatedGame, SettingsDict, Step, decode_log, fit_rating, load_export,
                                  summarize, Verdict)

SHA = "c" * 64


def _games(true_rating: float, n: int, seed: int, ussr_advantage: float = 0.0,
           players: int = 20) -> List[RatedGame]:
    rng = np.random.default_rng(seed)
    ratings = rng.normal(1700.0, 200.0, players)
    out = []
    for i in range(n):
        k = int(rng.integers(players))
        side = "USSR" if i % 2 else "US"
        x = ratings[k] + ussr_advantage * (1 if side == "USSR" else -1) - true_rating
        p = 1.0 / (1.0 + 10 ** (-x / 400.0))
        out.append(RatedGame(f"p{k}", float(ratings[k]), side, 1.0 if rng.random() < p else 0.0))
    return out


def test_the_fit_finds_a_known_rating() -> None:
    fit = fit_rating("m", _games(1900.0, 400, seed=1), bootstrap=200)
    assert abs(fit.rating - 1900.0) < 3 * fit.se
    # 400 games against a spread of players: about 400 / (ln10 * sqrt(400 * p(1-p))) points.
    assert 15 < fit.se < 30
    assert fit.interval[0] < fit.rating < fit.interval[1]
    assert fit.games == 400 and fit.humans <= 20


def test_the_side_advantage_is_taken_out() -> None:
    games = _games(1800.0, 600, seed=2, ussr_advantage=80.0)
    with_it = fit_rating("m", games, ussr_advantage=80.0, bootstrap=0)
    assert abs(with_it.rating - 1800.0) < 3 * with_it.se


def test_a_perfect_score_stays_finite() -> None:
    games = [RatedGame("a", 1500.0, "US", 0.0) for _ in range(5)]
    fit = fit_rating("m", games, bootstrap=50)
    assert math.isfinite(fit.rating) and fit.rating > 1500.0
    assert fit.record == (5, 0, 0) and fit.by_side == {"US": (0, 0, 0), "USSR": (5, 0, 0)}


def test_one_players_many_games_widen_the_interval() -> None:
    # The same games, once from one player and once from twenty: the bootstrap over players knows.
    many = _games(1900.0, 200, seed=3)
    one = [RatedGame("solo", g.human_rating, g.human_side, g.score) for g in many]
    wide = fit_rating("m", one, bootstrap=200)
    spread = fit_rating("m", many, bootstrap=200)
    assert wide.interval[1] - wide.interval[0] < 1e-6, "one player resampled is the same games every time"
    assert spread.interval[1] - spread.interval[0] > 20


# ---- the CLI on synthetic submissions -----------------------------------------------------------

def _log(seed: int, steps: List[Tuple[str, str, int]]) -> str:
    """record.ts encodeLog for (mover, source, setting) steps; the actions are not replayed here."""
    raw = bytearray(struct.pack("<BII", 1, seed, len(steps)))
    for mover, source, setting in steps:
        info = ("human", "suggested", "auto").index(source) | ({"US": 1, "USSR": 2}[mover] << 3)
        raw += bytes([1, 1, 0, 0, 0, info, setting])
    return base64.urlsafe_b64encode(zlib.compress(bytes(raw))).decode().rstrip("=")


def _submission(i: int, login: str, rating: int, human_side: str, winner: str,
                human_for_model: bool = False, search: str = "policy") -> Dict[str, Any]:
    model_side = "US" if human_side == "USSR" else "USSR"
    steps = [(model_side, "auto", 0), (human_side, "human", 0)] * 20
    if human_for_model:
        steps.append((model_side, "human", 0))
    settings: List[SettingsDict] = [{"view": human_side, "auto": model_side, "model": SHA, "search": search}]
    sub: Dict[str, Any] = {
        "schema": 1, "id": f"game-{i:04d}-xxxx", "started_at": "2026-10-18T10:00:00Z",
        "ended_at": "2026-10-18T11:00:00Z",
        "build": {"commit": "abc", "dirty": False, "engine_fingerprint": "f" * 64},
        "playdek": {"login": login, "rating": rating}, "note": "",
        "result": {"winner": winner, "margin": 20, "end_turn": 8, "reason": "20 VP"},
        "settings": settings, "models": {SHA: {"label": "E7 test", "source": "", "checkpoint": ""}},
        "events": [], "start_position": None, "log": _log(i, steps), "n_steps": len(steps),
        "final_hash": "0" * 64,
    }
    _, decoded = decode_log(sub["log"])
    sub["summary"] = summarize(decoded, settings, [], None, finished=True)
    return sub


def test_the_log_decodes_to_its_tags() -> None:
    seed, steps = decode_log(_log(9, [("US", "auto", 0), ("USSR", "suggested", 2)]))
    assert seed == 9
    assert steps[1] == Step(1, 1, 0, 0, 0, "suggested", False, "USSR", False, 2)


def test_rating_counts_only_clean_games_per_model_player(tmp_path: Any, capsys: pytest.CaptureFixture[str],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    verified: List[str] = []

    def accept(sub: Any, fingerprint: str) -> Verdict:
        verified.append(sub["id"])
        return Verdict("ok", "unit fixture")

    monkeypatch.setattr(cli, "verify", accept)
    subs = [_submission(i, f"p{i % 3}", 1600 + 50 * (i % 3), "US" if i % 2 else "USSR",
                        "USSR" if i % 2 else "US") for i in range(12)]  # the model wins every game
    subs.append(_submission(100, "p0", 1600, "US", "US", human_for_model=True))
    subs.append(_submission(101, "p1", 1650, "US", "US", search="rollout:4:32:4:z2"))
    # wrangler d1 execute --json: rows under "results", each submission in `body`.
    export = tmp_path / "games.json"
    export.write_text(json.dumps([{"results": [{"body": json.dumps(s)} for s in subs]}]))
    assert len(load_export(str(export))) == 14

    assert cli.main(["rating", str(export), "--bootstrap", "50", "--json", str(tmp_path / "fit.json")]) == 0
    out = capsys.readouterr().out
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert verified == [s["id"] for s in subs]
    players = {f["player"]: f for f in fit["fits"]}
    assert set(players) == {f"E7 test [{SHA[:12]}] policy", f"E7 test [{SHA[:12]}] rollout:4:32:4:z2"}
    policy = players[f"E7 test [{SHA[:12]}] policy"]
    assert policy["games"] == 12 and policy["record"] == [12, 0, 0] and policy["humans"] == 3
    assert policy["rating"] > 1700
    assert fit["skipped"] == {"not clean": 1}
    assert "Left out: 1 not clean" in out

    assert cli.main(["show", str(export)]) == 0
    assert "human_moves_for_model=1" in capsys.readouterr().out


def test_rating_rejects_unplayed_and_other_engine_records(tmp_path: Any) -> None:
    refused = _submission(0, "p", 2000, "USSR", "US")
    refused["build"]["engine_fingerprint"] = cli.engine_fingerprint()
    other_engine = _submission(1, "p", 2000, "USSR", "US")
    export, output = tmp_path / "games.json", tmp_path / "fit.json"
    export.write_text(json.dumps([refused, other_engine]))
    assert cli.main(["rating", str(export), "--bootstrap", "0", "--json", str(output)]) == 0
    fit = json.loads(output.read_text())
    assert not fit["fits"]
    assert fit["skipped"] == {"verification: refused": 1, "verification: other-engine": 1}
