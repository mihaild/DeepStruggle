#!/usr/bin/env python3
"""Freeze the engine's decision stream, so a refactor can be checked against it.

P17 changes the action representation. Its whole verification rests on one property: **a
divergence between the old engine and the new one means the representation changed, and nothing
else**. That is only checkable against a stream recorded *before* the change, so this records it.

What is captured, per decision:

* a **state fingerprint** — every country's influence, the tracks, turn/AR/phase, and all 110 card
  locations, hashed. Two engines agreeing on this agree on the position.
* the **decision type** and **acting player**;
* a **mask fingerprint** and the legal count — what the player was offered;
* the **action taken**.

Actions are chosen by a seeded RNG over the legal set, not by a policy. A policy would be the
confound: its choices move when the action space does, so a stream recorded with one could not be
reproduced afterwards for reasons having nothing to do with the engine.

    # freeze
    tools/scripts/decision_stream_baseline.py --games 200 --out data/p17_baseline.jsonl.gz

    # after a change: regenerate and diff
    tools/scripts/decision_stream_baseline.py --games 200 --compare data/p17_baseline.jsonl.gz

**On the post-refactor side this is driven through the adapter**, which maps the merged
representation back to the old flat indices. The comparison is then like for like, and the only
expected divergence is the one §1 of the plan names — a card that can no longer be played for Ops
because no coup target exists and coup was its only usable mode.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np

import ts_engine as ts

from bindings.action_encoder import ActionEncoder

MAX_STEPS = 4000


def state_fingerprint(st: "ts.GameState") -> str:
    """A stable digest of the position. Cheap fields only -- no observation, which is
    perspective-dependent and would double the cost for no extra discrimination."""
    h = hashlib.blake2b(digest_size=16)
    for i in range(84):
        c = st.get_country(i)
        h.update(bytes((c.us_influence, c.ussr_influence)))
    h.update(bytes((
        st.victory_points & 0xFF, st.defcon, st.us_mil_ops, st.ussr_mil_ops,
        st.us_space_track, st.ussr_space_track, st.turn, st.action_round,
        int(st.current_phase),
    )))
    for cid in range(1, 111):
        h.update(bytes((int(st.get_card_location(cid)),)))
    return h.hexdigest()


def play_game(seed: int, rng: random.Random) -> Dict[str, Any]:
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    steps: List[List[Any]] = []
    guard = 0
    while not ts.Engine.is_terminal(st) and guard < MAX_STEPS:
        guard += 1
        ctx = st.ctx()
        pl = ctx.decision_player
        if pl == ts.Player.NONE:
            # chance node: drain it. try_step is the boolean half of the try_step/step split --
            # step returns None and RAISES, so its result must never be truth-tested.
            if not ts.Engine.try_step(st, ts.MicroAction(ctx.decision_type, 0, 0, 0)):
                break
            continue
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            steps.append(["EMPTY_MASK", int(ctx.decision_type), int(pl),
                          state_fingerprint(st), 0, -1])
            break
        chosen = int(legal[rng.randrange(len(legal))])
        steps.append([
            state_fingerprint(st),
            int(ctx.decision_type),
            int(pl),
            hashlib.blake2b(mask.tobytes(), digest_size=8).hexdigest(),
            int(len(legal)),
            chosen,
        ])
        if not ts.Engine.try_step(st, ts.decode_flat_action(st, chosen)):
            steps.append(["STEP_REFUSED", int(ctx.decision_type), int(pl), "", 0, chosen])
            break
    return {
        "seed": seed,
        "steps": steps,
        "n_steps": len(steps),
        "final": state_fingerprint(st),
        "vp": int(st.victory_points),
        "turn": int(st.turn),
        "terminal": bool(ts.Engine.is_terminal(st)),
    }


def generate(games: int, base_seed: int) -> List[Dict[str, Any]]:
    out = []
    for g in range(games):
        # one RNG per game, seeded from the game seed, so game N is reproducible on its own
        out.append(play_game(base_seed + g, random.Random(0xC0FFEE + base_seed + g)))
    return out


def write(path: str, rows: List[Dict[str, Any]]) -> None:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def read(path: str) -> List[Dict[str, Any]]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compare(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> int:
    """Report the FIRST divergence per game, which is the only one that means anything --
    everything after it is downstream of a different position."""
    if len(old) != len(new):
        print("game count differs: %d vs %d" % (len(old), len(new)))
        return 1
    diverged = 0
    for o, n in zip(old, new):
        if o["seed"] != n["seed"]:
            print("seed mismatch: %s vs %s" % (o["seed"], n["seed"]))
            diverged += 1
            continue
        if o["steps"] == n["steps"]:
            continue
        diverged += 1
        for i, (a, b) in enumerate(zip(o["steps"], n["steps"])):
            if a != b:
                print("seed %d: first divergence at step %d" % (o["seed"], i))
                print("   old: state=%s dt=%s pl=%s mask=%s nlegal=%s action=%s" % tuple(a))
                print("   new: state=%s dt=%s pl=%s mask=%s nlegal=%s action=%s" % tuple(b))
                break
        else:
            print("seed %d: identical prefix, length differs %d vs %d" % (
                o["seed"], len(o["steps"]), len(n["steps"])))
    print()
    print("games compared: %d" % len(old))
    print("games diverging: %d" % diverged)
    return 1 if diverged else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=770000)
    ap.add_argument("--out", default=None, help="write the stream here")
    ap.add_argument("--compare", default=None, help="regenerate and diff against this file")
    a = ap.parse_args()

    if not a.out and not a.compare:
        print("give --out to freeze a baseline, or --compare to check against one",
              file=sys.stderr)
        return 2

    rows = generate(a.games, a.base_seed)
    total = sum(r["n_steps"] for r in rows)
    terminal = sum(1 for r in rows if r["terminal"])
    print("%d games, %d decisions, %.1f per game, %d reached a terminal state"
          % (len(rows), total, total / max(1, len(rows)), terminal))

    if a.out:
        write(a.out, rows)
        digest = hashlib.blake2b(
            "".join(r["final"] for r in rows).encode(), digest_size=16).hexdigest()
        print("wrote %s" % a.out)
        print("corpus digest: %s" % digest)
        return 0

    return compare(read(a.compare), rows)


if __name__ == "__main__":
    raise SystemExit(main())
