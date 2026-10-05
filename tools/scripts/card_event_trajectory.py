#!/usr/bin/env python3
"""How a card's owner uses it across a run's checkpoints: event, Ops, or not at all.

Greedy self-play of each checkpoint; every owner holding of the card (the card in the owner's
hand until it leaves) in which the event could trigger at one of the owner's card choices is
counted by how it ended -- evented (headlined, or the event in a round), played for Ops or the
space race, or not played by the owner. The per-choice bookkeeping is `checkpoint_report`'s
`OwnCardChoices`. `--max-turn` stops the games after that turn, which makes an Early War card
cheap to follow; a holding still in hand then counts as not played.

    PYTHONPATH=.:build/release python tools/scripts/card_event_trajectory.py --card "Marshall Plan" \\
        --owner US --max-turn 3 --checkpoints <pt> ... --output-md <md>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from tools.scripts.checkpoint_report import ID, OwnCardChoices, _holding_outcomes, _pct


def play(model: Any, merged: bool, card: int, owner: Any, games: int, seed: int, batch: int,
         max_turn: int) -> List[Dict[str, Any]]:
    from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
    check_obs_width(model)
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    obs_card = OwnCardChoices(card, owner, lambda st: 0)
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        if merged:
            env.set_merged_influence(True, True)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        for _ in range(20_000):
            if all(done):
                break
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)[:, :width]).to(device),
                               torch.from_numpy(np.asarray(masks)).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st) or int(st.turn) > max_turn:
                    done[i] = True
                    obs_card.done(b0 + i)
                    continue
                obs_card.see(b0 + i, st, int(actions[i]))
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d and not done[i]:
                    done[i] = True
                    obs_card.done(b0 + i)
    return obs_card.choices


def summarize(choices: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hs = [h for h in _holding_outcomes(choices).values() if h["legal"]]
    return {"holdings": len(hs),
            "headline": sum(h["played"] == "headline" for h in hs),
            "event": sum(h["played"] == "event" for h in hs),
            "ops": sum(h["played"] in ("Ops", "space race") for h in hs),
            "kept": sum(h["played"] is None for h in hs)}


def plot_svg(series: Sequence[Tuple[str, Sequence[Dict[str, Any]]]], title: str, subtitle: str,
             marks: Sequence[Tuple[float, str]] = ()) -> str:
    """The evented share of holdings against training steps, one line per sweep (its rows from
    --dump), with 95% Wilson bands; `marks` are vertical lines (step in millions, label)."""
    from tools.scripts.star_wars_play import _wilson
    colours = [("#2a78d6", "#3987e5", "circle"), ("#eb6834", "#d95926", "square")]
    W, H, L, R, T, B = 760, 380, 56, 170, 70, 44

    def xm(label: str) -> float:
        return float(label.rstrip("M").replace(",", ""))
    xmax = max(xm(r["label"]) for _, rows in series for r in rows)
    ymax = 1.0

    def X(x: float) -> float:
        return L + (W - L - R) * x / xmax

    def Y(p: float) -> float:
        return T + (H - T - B) * (1 - p / ymax)
    css_l = "".join(f".s{i}{{stroke:{c[0]};fill:{c[0]}}}.t{i}{{fill:{c[0]}}}" for i, c in enumerate(colours))
    css_d = "".join(f".s{i}{{stroke:{c[1]};fill:{c[1]}}}.t{i}{{fill:{c[1]}}}" for i, c in enumerate(colours))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" aria-label="{title}">',
         "<style>.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.mut{fill:#898781}"
         ".grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}" + css_l + "svg .mark{stroke:#fcfcfb}svg .line{fill:none}"
         "svg .band{stroke:none}@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#ffffff}"
         ".ink2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}" + css_d + "svg .mark{stroke:#1a1a19}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="ink" x="{L}" y="22" font-size="15" font-weight="600">{title}</text>',
         f'<text class="ink2" x="{L}" y="40" font-size="12">{subtitle}</text>']
    for v in range(0, 101, 20):
        y = Y(v / 100)
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="1"/>')
        o.append(f'<text class="mut" x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{v}%</text>')
    step = 400 if xmax > 2000 else 200
    for v in range(0, int(xmax) + 1, step):
        o.append(f'<text class="mut" x="{X(v):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle">{v:,}M</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke-width="1"/>')
    o.append(f'<text class="mut" x="{(L + W - R) / 2:.1f}" y="{H - 8}" font-size="11" text-anchor="middle">training steps</text>')
    for j, (xv, lab) in enumerate(marks):          # labels alternate between two rows
        x = X(xv)
        ty = T - 6 - 13 * (j % 2)
        o.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{ty + 3}" y2="{Y(0):.1f}" stroke-width="1" stroke-dasharray="3 3"/>')
        o.append(f'<text class="mut" x="{x + 4:.1f}" y="{ty}" font-size="10">{lab}</text>')
    ends = []
    for i, (name, rows) in enumerate(series):
        pts = []
        for r in rows:
            n, k = int(r["holdings"]), int(r["headline"]) + int(r["event"])
            lo, hi = _wilson(k, n)
            pts.append((xm(r["label"]), k / max(1, n), lo, hi))
        band = " ".join(f"{X(x):.1f},{Y(hi):.1f}" for x, _, _, hi in pts) + " " + \
            " ".join(f"{X(x):.1f},{Y(lo):.1f}" for x, _, lo, _ in reversed(pts))
        o.append(f'<polygon class="s{i} band" points="{band}" fill-opacity="0.14"/>')
        o.append(f'<polyline class="s{i} line" points="{" ".join(f"{X(x):.1f},{Y(p):.1f}" for x, p, _, _ in pts)}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        for x, p, _, _ in pts:
            if colours[i][2] == "circle":
                o.append(f'<circle class="s{i} mark" cx="{X(x):.1f}" cy="{Y(p):.1f}" r="3.5" style="stroke-width:1.5"/>')
            else:
                o.append(f'<rect class="s{i} mark" x="{X(x) - 3.2:.1f}" y="{Y(p) - 3.2:.1f}" width="6.4" height="6.4" '
                         f'rx="1" style="stroke-width:1.5"/>')
        ends.append((Y(pts[-1][1]), i, name, pts[-1][1]))
    ends.sort()
    ys = [e[0] for e in ends]
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + 16)
    over = ys[-1] - Y(0) if ys else 0.0           # keep the labels inside the plot
    if over > 0:
        ys = [y - over for y in ys]
    for y, (_, i, name, p) in zip(ys, ends):
        o.append(f'<text class="t{i}" x="{W - R + 10}" y="{y + 4:.1f}" font-size="12" font-weight="600">'
                 f'{name} {100 * p:.0f}%</text>')
    o.append("</svg>")
    return "\n".join(o)


def _steps(path: str) -> Optional[int]:
    m = re.search(r"(\d+)steps\.pt$", path)
    return int(m.group(1)) if m else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.action_view import checkpoint_merged_influence
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True, help="card name as in rules/cards.json")
    ap.add_argument("--owner", choices=("US", "USSR"), required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=55_000)
    ap.add_argument("--max-turn", type=int, default=10)
    ap.add_argument("--dump", default=None, help="write the per-checkpoint summaries here (JSON)")
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    card = ID[a.card]
    owner = ts.Player.US if a.owner == "US" else ts.Player.USSR
    labels = a.labels or [f"{(_steps(p) or 0) / 1e6:,.0f}M" for p in a.checkpoints]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header = [f"{a.card}, {a.owner} holdings (event could trigger), turns 1-{a.max_turn}, {a.games:,} greedy "
              f"self-play games per checkpoint (seed {a.seed:,}):", "",
              "| checkpoint | holdings | evented | headline | event in a round | Ops / space race | not played |",
              "|:---|---:|---:|---:|---:|---:|---:|"]
    out = list(header)
    print("\n".join(header), flush=True)
    rows: List[Dict[str, Any]] = []
    for path, label in zip(a.checkpoints, labels):
        model = NeuralAgent.from_checkpoint(path, device=str(dev)).model
        s = summarize(play(model, checkpoint_merged_influence(path), card, owner, a.games, a.seed, a.batch, a.max_turn))
        s.update(label=label, checkpoint=os.path.relpath(path, "/workspace"))
        rows.append(s)
        n = s["holdings"]
        line = (f"| {label} | {n:,} | **{_pct(s['headline'] + s['event'], n)}** | {_pct(s['headline'], n)} | "
                f"{_pct(s['event'], n)} | {_pct(s['ops'], n)} | {_pct(s['kept'], n)} |")
        out.append(line)
        print(line, flush=True)
        if a.dump:
            json.dump(rows, open(a.dump, "w"))
    if a.output_md:
        open(a.output_md, "w").write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
