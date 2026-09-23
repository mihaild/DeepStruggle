#!/usr/bin/env python3
"""The acceptance probes for the programme's goal, run on any list of checkpoints.

`research/plans/README.md` defines the goal as a no-search player at the level of a mediocre
human, and defines that by named behaviours rather than Elo: a sane setup (Poland >= 3 as USSR,
West Germany >= 4 as US), contesting battlegrounds instead of leaving them empty from turn 8, not
losing to its own DEFCON, and disposing of a card through the exit that card has. Each has a probe
in `ai/eval/`, but until now they ran only inside training, on the model being trained. This runs
the same functions on finished checkpoints, one row each, so the distance to the goal can be read
for any snapshot rather than inferred from Elo.

The setup criterion has a human yardstick: the same statistics read off the human corpus by
`setup_probe.measure_corpus`. The others do not yet.

    PYTHONPATH=.:build/release python tools/scripts/goal_probes.py \\
        --checkpoints a.pt b.pt --output-json out.json --output-md out.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ai.eval import setup_probe  # noqa: E402
from ai.eval.blunders import measure_blunders_batched  # noqa: E402
from ai.eval.decisive_probe import measure_decisive_batched  # noqa: E402
from ai.eval.position_diagnostics import profile_self_play_batched  # noqa: E402
from tools.lib.checkpoint_id import checkpoint_label  # noqa: E402

#: The blunder rules, by the goal criterion each one measures.
BLUNDER_RULES = {
    "defcon_suicide_with_alternative": "own-DEFCON loss taken with an alternative",
    "spaced_own_or_neutral": "space exit spent on a card with its own exit",
    "olympic_games_at_defcon2": "Olympic Games played at DEFCON 2",
}


def setup_row(m: setup_probe.SetupMeasurement) -> Dict[str, float]:
    """Each setup target's rate, plus the composite: every target taken in one opening."""
    row: Dict[str, float] = {}
    for t, rate, _lo, _hi in setup_probe.target_rates(m):
        row[f"setup/{t.side} {t.name} >= {t.threshold}"] = rate
    row["setup/all targets"] = setup_probe.all_targets_met(m)[0]
    return row


def probe(path: str, device: str, setup_games: int, position_games: int, blunder_games: int,
          decisive_games: int) -> Dict[str, float]:
    from tools.lib.player_agent import NeuralAgent

    agent = NeuralAgent.from_checkpoint(path, device=device)
    model = agent.model
    model.eval()
    # Each checkpoint plays in the action view it was trained in (P23). Setup has no op-choice
    # node, so only the three game-playing probes need it.
    mv = agent.merged_influence
    row: Dict[str, float] = {"view/merged_influence": float(mv)}
    row.update(setup_row(setup_probe.measure(
        setup_probe.torch_policy(model, device=device, temperature=0.1), num_games=setup_games)))
    prof = profile_self_play_batched(model, num_envs=position_games, merged_influence=mv)
    row.update({k: float(v) for k, v in prof["scalars"].items()})
    for temperature in (0.1, 1.0):
        counts = measure_blunders_batched(model, num_games=blunder_games, temperature=temperature,
                                          merged_influence=mv)
        for rule in BLUNDER_RULES:
            row[f"blunder@{temperature:g}/{rule}"] = counts.rate(rule)
            row[f"blunder@{temperature:g}/{rule}/chances"] = float(counts.opportunities.get(rule, 0))
    dec = measure_decisive_batched(model, num_envs=decisive_games, merged_influence=mv)
    row["decisive/forced wins taken"] = dec.win_take_rate
    row["decisive/avoidable losses avoided"] = dec.loss_avoid_rate
    return row


def _fmt(v: Optional[float], pct: bool) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.2f}"


def render(rows: Dict[str, Dict[str, float]]) -> str:
    """One column per checkpoint, one row per measurement, grouped by goal criterion."""
    labels = list(rows)
    keys: List[str] = []
    for r in rows.values():
        keys.extend(k for k in r if k not in keys and not k.endswith("/chances"))
    lines = ["| measurement | " + " | ".join(labels) + " |",
             "|:---|" + "---:|" * len(labels)]
    for k in keys:
        pct = not k.startswith("diag/") or "salvageable" in k
        lines.append(f"| {k} | " + " | ".join(_fmt(rows[lab].get(k), pct) for lab in labels) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--setup-games", type=int, default=2000)
    ap.add_argument("--position-games", type=int, default=256)
    ap.add_argument("--blunder-games", type=int, default=128)
    ap.add_argument("--decisive-games", type=int, default=256)
    ap.add_argument("--no-human", action="store_true", help="skip the human-corpus setup column")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    from tools.lib.player_agent import resolve_device

    device = str(resolve_device(args.device))
    rows: Dict[str, Dict[str, float]] = {}
    if not args.no_human:
        human = setup_probe.measure_corpus()
        if human.games == 0:
            # measure_corpus returns an empty measurement rather than raising when the corpus is
            # absent, and an empty yardstick renders as a column of dashes that reads like data.
            sys.exit("the human corpus is missing or empty -- fetch it with "
                     "tools/download_ts_replayer.py, or pass --no-human")
        rows[f"human corpus ({human.games} games)"] = setup_row(human)
    for path in args.checkpoints:
        label = checkpoint_label(path)
        print(f"probing {label} ...", flush=True)
        rows[label] = probe(path, device, args.setup_games, args.position_games,
                            args.blunder_games, args.decisive_games)
    table = render(rows)
    print(table)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
