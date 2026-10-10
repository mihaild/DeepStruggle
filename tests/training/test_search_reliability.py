"""tools/search_reliability.py's figures on hand-built rows: reproducibility counts a departure only
when another seed picks the same move, gains are paired against raw's score, and the population
weights are the bank's weight over its inclusion probability."""
from __future__ import annotations

import argparse

import pytest

from tools.search_reliability import PER_GAME, chunk_seed, parse_spec, summarize


def _bank(i: str, weight: float, incl: float) -> dict:
    return {"id": i, "weight": weight, "pl_incl": incl}


def test_reproducibility_gain_and_precision_by_hand() -> None:
    bank = {"a": _bank("a", 2.0, 1.0), "b": _bank("b", 1.0, 0.5)}      # weights 2 and 2
    search = {
        "a": {"id": "a", "raw": 1, "picks": {"w1": [2, 3], "w16": [2, 2]}},
        "b": {"id": "b", "raw": 5, "picks": {"w1": [5, 5], "w16": [6, 6]}},
    }
    play = {
        "a": {"id": "a", "score": {"1": 0.40, "2": 0.50, "3": 0.30},
              "diff_vs_raw": {"2": [0.10, 0.02], "3": [-0.10, 0.02]}},
        "b": {"id": "b", "score": {"5": 0.60, "6": 0.62}, "diff_vs_raw": {"6": [0.02, 0.02]}},
    }
    res = summarize(bank, search, play, ["w1", "w16"])
    w1, w16 = res["w1"], res["w16"]
    # w1 departs at a in both seeds, to different moves: never reproduced.
    assert w1["reproducibility"] == pytest.approx(0.0)
    assert w1["departure_rate"] == pytest.approx(0.5)
    # w16 departs everywhere and always to the same move.
    assert w16["reproducibility"] == pytest.approx(1.0) and w16["same_move"] == pytest.approx(1.0)
    # gains: w1 a = mean(+0.1, -0.1) = 0, b = 0; w16 a = +0.1, b = +0.02; equal weights.
    assert w1["gain_per_decision"] == pytest.approx(0.0)
    assert w16["gain_per_decision"] == pytest.approx(0.06)
    assert w16["gain_per_game"] == pytest.approx(0.06 * 4.0 * PER_GAME)
    assert w16["vs_first_per_game"] == pytest.approx(0.06 * 4.0 * PER_GAME)
    assert w1["precision_positive"] == pytest.approx(0.5)
    assert w1["confirmed_better"] == pytest.approx(0.5) and w1["confirmed_worse"] == pytest.approx(0.5)
    assert w16["confirmed_better"] == pytest.approx(0.5)     # a's +0.10 is 5 SE; b's +0.02 is 1


def test_specs_and_seeds() -> None:
    assert parse_spec("w16=256:8:0.2:all:16") == ("w16", "256:8:0.2:all:16")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_spec("w16=256:8")
    assert chunk_seed("w1", 0, ["a"]) != chunk_seed("w1", 1, ["a"]) != chunk_seed("w16", 0, ["a"])
    assert 0 <= chunk_seed("w1", 0, ["a", "b"]) < 2 ** 32


def test_runs_merge_by_position(tmp_path) -> None:
    """Two runs' searchers side by side; their playouts' moves pooled; a raw move scored differently
    across runs is refused."""
    import gzip
    import json
    from tools.search_reliability import report

    def w(name: str, rows: list) -> str:
        p = str(tmp_path / name)
        with gzip.open(p, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    bank = w("bank.jsonl.gz", [_bank("a", 1.0, 1.0)])
    s1 = w("s1.jsonl.gz", [{"id": "a", "raw": 1, "picks": {"x": [2, 2]}, "evals": {"x": [256, 256]}}])
    s2 = w("s2.jsonl.gz", [{"id": "a", "raw": 1, "picks": {"y": [3, 3]}, "evals": {"y": [900, 900]}}])
    p1 = w("p1.jsonl.gz", [{"id": "a", "pairs": 4, "score": {"1": 0.4, "2": 0.5}, "diff_vs_raw": {"2": [0.1, 0.01]}}])
    p2 = w("p2.jsonl.gz", [{"id": "a", "pairs": 4, "score": {"1": 0.4, "3": 0.3}, "diff_vs_raw": {"3": [-0.1, 0.01]}}])
    out = str(tmp_path / "r.md")
    assert report(bank, [s1, s2], [p1, p2], out) == 0
    res = json.load(open(str(tmp_path / "r.json")))
    assert res["x"]["gain_per_decision"] == pytest.approx(0.1)
    assert res["y"]["gain_per_decision"] == pytest.approx(-0.1)
    bad = w("p3.jsonl.gz", [{"id": "a", "pairs": 4, "score": {"1": 0.45, "3": 0.3}, "diff_vs_raw": {"3": [-0.1, 0.01]}}])
    with pytest.raises(ValueError):
        report(bank, [s1, s2], [p1, bad], out)
