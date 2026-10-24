"""A game submitted to the stats database is the game that was played.

The page records a game as its seed and steps (web/ui/src/game/record.ts) and submits that, not
the full replay; tools/game_stats.py re-drives it in the Python engine. So the record must carry
everything the engine needs -- forced dice included -- and the page's summary of how the game
was played must be the one Python computes from the same log. Two games are played in the
page's own GameSession under node (tests/web/js/game_stats.ts): a clean one, and one that does
everything a game can be marked for. Both go through the stats Worker's handler, with a fake D1.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.web.js_runner import PUBLIC, run_ts
from tools import game_stats as cli
from tools.lib.engine_fingerprint import fingerprint
from tools.lib.game_stats import SubmissionDict, decode_log, summarize_submission, verify


@pytest.fixture(scope="module")
def played(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    out = run_ts("game_stats.ts", str(tmp_path_factory.mktemp("game_stats")), PUBLIC)
    assert out["fingerprint"] == fingerprint(), "the WebAssembly engine is stale: tools/scripts/build_web.sh --engine"
    return out


def test_a_submitted_game_replays_to_its_end_in_python(played: Dict[str, Any]) -> None:
    for name in ("clean", "messy"):
        sub: SubmissionDict = played[name]
        v = verify(sub, fingerprint())
        assert v.status == "ok", f"{name}: {v.status} {v.detail}"


def test_rating_verifies_the_exact_record_before_counting_it(played: Dict[str, Any], tmp_path: Path) -> None:
    clean = played["clean"]
    forged = copy.deepcopy(clean)
    forged["id"] = "forged-result"
    forged["result"]["winner"] = "USSR" if clean["result"]["winner"] == "US" else "US"
    wrong_hash = copy.deepcopy(clean)
    wrong_hash["id"] = "forged-hash"
    wrong_hash["final_hash"] = "0" * 64
    export, output = tmp_path / "games.json", tmp_path / "fit.json"
    export.write_text(json.dumps([clean, played["messy"], forged, wrong_hash]))
    assert cli.main(["rating", str(export), "--bootstrap", "0", "--json", str(output)]) == 0
    fit = json.loads(output.read_text())
    assert len(fit["fits"]) == 1 and fit["fits"][0]["games"] == 1
    assert fit["skipped"] == {"not clean": 1, "verification: mismatch": 2}


def test_the_log_is_compact_and_round_trips(played: Dict[str, Any]) -> None:
    sub: SubmissionDict = played["clean"]
    assert played["roundtrip"]
    seed, steps = decode_log(sub["log"])
    assert seed == 7771 and len(steps) == sub["n_steps"] > 50
    # Under 7 bytes a step before deflate: a full game of ~500 steps is a few KB, where its replay
    # with snapshots is megabytes.
    for name in ("clean", "messy"):
        s: SubmissionDict = played[name]
        assert len(s["log"]) < 7 * s["n_steps"] + 100, name
        assert len(json.dumps(s)) < 4000 + 7 * s["n_steps"], name


def test_page_and_python_summarize_a_game_alike(played: Dict[str, Any]) -> None:
    for name in ("clean", "messy"):
        sub: SubmissionDict = played[name]
        assert summarize_submission(sub) == sub["summary"], name


def test_a_straight_game_is_clean_and_takebacks_are_counted(played: Dict[str, Any]) -> None:
    s = played["clean"]["summary"]
    assert s["clean"]
    assert (s["model_side"], s["human_side"]) == ("US", "USSR")
    assert s["models"] == ["a" * 64] and s["searches"] == ["policy"]
    assert s["undos"] == 2


def test_a_messy_game_says_what_happened(played: Dict[str, Any]) -> None:
    s = played["messy"]["summary"]
    assert not s["clean"]
    assert s["model_side"] == "US"
    assert s["human_moves_for_model"] == 3
    assert s["dev_view_moves"] == 4
    assert s["suggested_moves"] == 1 and s["analysis_moves"] == 1
    assert s["forced_dice"] > 0, "no forced six was rolled: pick moves that roll"
    assert s["debug_overrides"] == 1
    assert s["models"] == ["a" * 64, "b" * 64]
    assert set(played["messy"]["models"]) == {"a" * 64, "b" * 64}


def test_a_reload_plays_the_stored_game_back(played: Dict[str, Any]) -> None:
    assert played["resumed"] == {"same_position": True, "same_id": True, "same_tags": True, "resume_event": "resume"}


def test_the_worker_keeps_a_game_once_and_refuses_junk(played: Dict[str, Any]) -> None:
    r = played["responses"]
    assert played["invalid_clean"] is None
    assert played["invalid_no_login"] == "a Playdek login is needed"
    assert (r["first"]["status"], r["first"]["body"]) == (200, {"ok": True, "stored": True})
    assert r["first"]["cors"] == "https://mihaild.github.io"
    assert (r["again"]["status"], r["again"]["body"]) == (200, {"ok": True, "stored": False})
    assert r["other_origin"]["status"] == 403
    assert r["no_login"]["status"] == 400
    assert r["messy"]["body"] == {"ok": True, "stored": True}
    assert r["third"]["body"] == {"ok": True, "stored": True}
    assert r["fourth"]["status"] == 429
    assert played["stored_body_is_submission"]
