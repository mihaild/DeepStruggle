#!/usr/bin/env python3
"""Build the behaviour-cloning dataset from the ts-replayer corpus of human games.

    PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py

Converts every cached replay and writes the materialised columns
`ai.training.human_corpus_dataset` reads. Unlike the self-play format there is no seed that
replays a human game -- the dice come from the log and the hands are solved -- so the
observations are stored rather than re-derived. See that module for the layout and why value
targets are masked on games whose recording stops.

A conversion failure is reported and counted, never skipped quietly: per AGENTS.md invariant 12
an entry the log does not determine is a bug to diagnose, and a dataset that silently drops the
games it could not reproduce hides exactly that.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ts_engine as ts

from ai.eval.agreement import group_point_runs
from ai.eval.human_dominance import _Capture, _capturing
from ai.training.human_corpus_dataset import HumanCorpusWriter
from tools.lib.corpus_paths import (corpus_files, distinct_corpus_files,
                                    missing_corpus_reason)
from tools.lib.ts_replayer_convert import Conversion, convert_game

DEFAULT_OUT = os.path.join(_ROOT, "data", "datasets", "human_corpus")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="directory to write the dataset columns into")
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N replays (for a quick check)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep every replay id, including games ts-replayer serves under more "
                         "than one. Only for reproducing an older dataset -- duplicates carry "
                         "several times their weight in behaviour cloning and in every injection "
                         "batch, and because their ids differ they land on both sides of an "
                         "id-based train/held-out split.")
    args = ap.parse_args()

    reason = missing_corpus_reason()
    if reason is not None:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    collapsed: dict = {}
    if args.keep_duplicates:
        paths = corpus_files()
    else:
        paths, collapsed = distinct_corpus_files()
        dropped = sum(len(ids) - 1 for ids in collapsed.values())
        print(f"{len(corpus_files())} replays on disk, {len(paths)} distinct games "
              f"({dropped} duplicate copies dropped from {len(collapsed)} groups)")
        for ids in sorted(collapsed.values()):
            print(f"  keeping {ids[0]}, dropping {', '.join(str(i) for i in ids[1:])}")
    if args.limit is not None:
        paths = paths[:args.limit]

    writer = HumanCorpusWriter(args.out)
    failures: List[str] = []
    skipped = empty = 0
    t0 = time.time()

    for index, path in enumerate(paths, 1):
        with gzip.open(path, "rt") as fh:
            game = json.load(fh)
        if not game.get("all_turns"):
            empty += 1
            continue

        cap = _Capture()
        with _capturing(cap):
            conv: Conversion = convert_game(game)
        if conv.skipped is not None:
            skipped += 1
            continue
        if conv.failure is not None:
            failures.append(str(conv.failure))
            continue
        if not conv.samples:
            continue

        # The engine's own verdict, and None where the recording stopped before the game did.
        # Points spent by one play are grouped so agreement can ignore the order they were
        # written in; see ai/eval/agreement.
        n = min(len(conv.samples), len(cap.states))
        plays = group_point_runs(cap.states[:n], cap.movers[:n]) if n else None
        writer.add_game(conv.samples,
                        us_utility=conv.us_utility,
                        final_vp=conv.final_victory_points,
                        plays=plays)

        if index % 25 == 0:
            print(f"  {index}/{len(paths)} replays, {len(writer.action):,} samples "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = writer.write()
    print("\n" + "=" * 70)
    print(f" Human corpus dataset -> {args.out}")
    print("=" * 70)
    print(f"  replays read          : {len(paths)}")
    print(f"  duplicate copies dropped : "
          f"{sum(len(v) - 1 for v in collapsed.values())} "
          f"from {len(collapsed)} groups")
    print(f"  empty downloads       : {empty}")
    print(f"  skipped (handicap)    : {skipped}")
    print(f"  conversion failures   : {len(failures)}")
    print(f"  games written         : {meta['games']}")
    print(f"  games with an outcome : {meta['games_with_outcome']}")
    print(f"  samples               : {meta['samples']:,}")
    print(f"  samples with a value target : {meta['samples_with_outcome']:,}")
    print(f"  elapsed               : {time.time() - t0:.0f}s")
    for failure in failures:
        print(f"  FAILED: {failure}")


if __name__ == "__main__":
    main()
