#!/usr/bin/env python3
"""What the US does with Star Wars when it is ahead on the Space Race, across a run's snapshots.

Star Wars (US, 2 Ops): if the US is ahead on the Space Race Track, the US picks any non-scoring
card from the discard pile and plays it as an event. Behind or level, the event does nothing.

Greedy self-play of each checkpoint. A *play* is a US card choice of its own (not inside another
event's resolution) that selects Star Wars while the US space marker is strictly ahead of the
USSR's: a headline (its event), or an action round followed by the play-mode choice -- EVENT,
SPACE, or Ops (influence, coup, realign). A *holding* is Star Wars entering the US hand until it
leaves; it counts when the US was ahead at one of its card choices while holding it, and records
how it left: played as the event, for Ops or space, or not played by the US at all (the game ended,
or it was discarded or taken).

    PYTHONPATH=.:build/release python tools/scripts/star_wars_play.py --checkpoints <pt>... --games 2048
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

STAR_WARS = 85
EVENT = ActionEncoder.PLAY_MODE_OFFSET              # 110
SPACE = ActionEncoder.PLAY_MODE_OFFSET + 1          # 111; 112..114 are the three Ops modes
OUTCOMES = ("headline", "event", "ops", "space", "not played")


def _ahead(st: ts.GameState) -> bool:
    return int(st.us_space_track) > int(st.ussr_space_track)


def play(model: Any, games: int, seed: int, batch: int, game0: int = 0) -> Dict[str, Any]:
    from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
    check_obs_width(model)
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    plays: List[Dict[str, Any]] = []        # every US play of Star Wars: phase, ahead, outcome
    holdings: List[Dict[str, Any]] = []     # every US holding: ahead_seen, outcome
    nonplays: List[Dict[str, Any]] = []     # Star Wars selected at a card choice that was not a play
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + game0 + b0)
        env.set_obs_features(feats, feats)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        holding: List[Optional[Dict[str, Any]]] = [None] * n      # the current US holding, if any
        pending: List[Optional[Dict[str, Any]]] = [None] * n      # an AR play awaiting its play mode
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
                if ts.Engine.is_terminal(st):
                    continue
                in_us_hand = ts.in_hand_of(st.get_card_location(STAR_WARS), ts.Player.US)
                if in_us_hand and holding[i] is None:
                    new: Dict[str, Any] = {"game": game0 + b0 + i, "turn": int(st.turn), "ahead_seen": False, "outcome": "not played"}
                    holding[i] = new
                    holdings.append(new)
                elif not in_us_hand and pending[i] is None:
                    holding[i] = None
                ctx = st.ctx()
                a = int(actions[i])
                sel = pending[i]
                if sel is not None and not (ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                                            and int(ctx.pending_op_card) == STAR_WARS):
                    # Selected at an action-round card choice that was not a play: a forced
                    # discard (Quagmire's, for one). Kept aside, not counted as a play.
                    sel["outcome"] = "not a play"
                    sel["next"] = [str(ctx.decision_type), int(ctx.resolving_card), str(st.get_card_location(STAR_WARS))]
                    nonplays.append(sel)
                    pending[i] = None
                if int(ctx.resolving_card) != 0 or ctx.decision_player != ts.Player.US:
                    continue
                h = holding[i]
                if ctx.decision_type == ts.DecisionType.SELECT_CARD and in_us_hand and h is not None:
                    ahead = _ahead(st)
                    h["ahead_seen"] = h["ahead_seen"] or ahead
                    if a < ActionEncoder.PLAY_MODE_OFFSET and int(ts.decode_flat_action(st, a).primary_id) == STAR_WARS:
                        rec = {"game": game0 + b0 + i, "turn": int(st.turn), "ahead": ahead,
                               "ar": 0 if st.current_phase == ts.Phase.HEADLINE else int(st.action_round),
                               "space": [int(st.us_space_track), int(st.ussr_space_track)],
                               "phase": "headline" if st.current_phase == ts.Phase.HEADLINE else "ar"}
                        if rec["phase"] == "headline":
                            rec["outcome"] = "headline"
                            plays.append(rec)
                            h["outcome"] = "headline"
                        else:
                            pending[i] = rec
                elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and pending[i] is not None:
                    rec = pending[i]
                    assert rec is not None
                    rec["outcome"] = "event" if a == EVENT else "space" if a == SPACE else "ops"
                    rec["mode"] = a - ActionEncoder.PLAY_MODE_OFFSET
                    rec["event_legal"] = bool(np.asarray(masks)[i][EVENT])
                    plays.append(rec)
                    if h is not None:
                        h["outcome"] = rec["outcome"]
                    pending[i] = None
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
    return {"games": games, "plays": plays, "holdings": holdings, "nonplays": nonplays}


def collect(model: Any, target: int, seed: int, batch: int, chunk: int) -> Dict[str, Any]:
    """Play `chunk` games at a time, on fresh seeds, until there are `target` plays while ahead;
    keep exactly the first `target` of them in game order, and the games up to the last one kept."""
    plays: List[Dict[str, Any]] = []
    holdings: List[Dict[str, Any]] = []
    nonplays: List[Dict[str, Any]] = []
    game0 = 0
    while sum(p["ahead"] for p in plays) < target:
        d = play(model, chunk, seed, batch, game0)
        plays += d["plays"]
        holdings += d["holdings"]
        nonplays += d["nonplays"]
        game0 += chunk
    ahead = sorted((p for p in plays if p["ahead"]), key=lambda p: (p["game"], p["turn"], p["ar"]))[:target]
    last = ahead[-1]["game"]
    keep = {id(p) for p in ahead}
    return {"games": last + 1, "target": target,
            "plays": [p for p in plays if (id(p) in keep) or (not p["ahead"] and p["game"] <= last)],
            "holdings": [h for h in holdings if h["game"] <= last],
            "nonplays": [p for p in nonplays if p["game"] <= last]}


def split_row(label: str, d: Dict[str, Any]) -> str:
    """The plays while ahead split by what the US did: headline, Ops / event / space race in a round."""
    pl = [p for p in d["plays"] if p["ahead"]]
    n = len(pl)
    cells = []
    for k in ("headline", "ops", "event", "space"):
        c = sum(p["outcome"] == k for p in pl)
        cells.append(f"{c} ({100 * c / n:.1f}%)" if n else "—")
    return f"| {label} | {int(d['games']):,} | {n:,} | " + " | ".join(cells) + " |"


SPLIT_HEADER = ["| snapshot | games played | plays while ahead | headline | Ops in a round | event in a round | space race |",
                "|:---|---:|---:|---:|---:|---:|---:|"]


#: (outcome, label, light, dark, marker) -- categorical slots 1-3 of the reference palette.
SERIES = [("headline", "headline", "#2a78d6", "#3987e5", "circle"),
          ("ops", "Ops in a round", "#eb6834", "#d95926", "square"),
          ("event", "event in a round", "#1baf7a", "#199e70", "triangle")]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def plot_svg(points: Sequence[tuple[float, Dict[str, Any]]], title: str, branch_at: Optional[float]) -> str:
    """A line chart of the split of plays while ahead, one point per snapshot (x in millions of
    steps), with 95% Wilson bands; light and dark colours chosen by the viewer's scheme."""
    W, H = 760, 380
    L, R, T, B = 56, 150, 66, 44
    xmax = max(1200.0, max(x for x, _ in points))

    def X(x: float) -> float:
        return L + (W - L - R) * x / xmax

    def Y(p: float) -> float:
        return T + (H - T - B) * (1 - p)

    sizes = {sum(p["ahead"] for p in d["plays"]) for _, d in points}
    n_label = f"{sizes.pop():,}" if len(sizes) == 1 else "all"
    css_l = "".join(f".s{i}{{stroke:{c};fill:{c}}}.t{i}{{fill:{c}}}" for i, (_, _, c, _, _) in enumerate(SERIES))
    css_d = "".join(f".s{i}{{stroke:{c};fill:{c}}}.t{i}{{fill:{c}}}" for i, (_, _, _, c, _) in enumerate(SERIES))
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" aria-label="{title}">',
         "<style>.bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.mut{fill:#898781}"
         ".grid{stroke:#e1e0d9}.axis{stroke:#c3c2b7}" + css_l + "svg .mark{stroke:#fcfcfb}svg .line{fill:none}svg .band{stroke:none}" +
         "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#ffffff}.ink2{fill:#c3c2b7}"
         ".grid{stroke:#2c2c2a}.axis{stroke:#383835}" + css_d + "svg .mark{stroke:#1a1a19}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="ink" x="{L}" y="22" font-size="15" font-weight="600">{title}</text>',
         f'<text class="ink2" x="{L}" y="40" font-size="12">Share of {n_label} plays per snapshot, '
         f'greedy self-play; bands are 95% intervals; space race (rare) not drawn</text>']
    for v in range(0, 101, 20):
        y = Y(v / 100)
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke-width="1"/>')
        o.append(f'<text class="mut" x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{v}%</text>')
    for v in range(0, int(xmax) + 1, 200):
        x = X(v)
        o.append(f'<text class="mut" x="{x:.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle">{v}M</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke-width="1"/>')
    o.append(f'<text class="mut" x="{(L + W - R) / 2:.1f}" y="{H - 8}" font-size="11" text-anchor="middle">training steps</text>')
    if branch_at is not None:
        x = X(branch_at)
        o.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{T - 2}" y2="{Y(0):.1f}" stroke-width="1" stroke-dasharray="3 3"/>')
        o.append(f'<text class="mut" x="{x - 4:.1f}" y="{T - 6}" font-size="10" text-anchor="end">E7-01-44</text>')
        o.append(f'<text class="mut" x="{x + 4:.1f}" y="{T - 6}" font-size="10">E7-02-44</text>')
    ends: List[tuple[float, int, str, float]] = []
    for i, (key, label, _, _, marker) in enumerate(SERIES):
        rows = []
        for x, d in points:
            pl = [p for p in d["plays"] if p["ahead"]]
            k = sum(p["outcome"] == key for p in pl)
            rows.append((x, k / max(1, len(pl)), *_wilson(k, len(pl))))
        band = " ".join(f"{X(x):.1f},{Y(hi):.1f}" for x, _, _, hi in rows) + " " + \
            " ".join(f"{X(x):.1f},{Y(lo):.1f}" for x, _, lo, _ in reversed(rows))
        o.append(f'<polygon class="s{i} band" points="{band}" fill-opacity="0.14"/>')
        o.append(f'<polyline class="s{i} line" points="{" ".join(f"{X(x):.1f},{Y(p):.1f}" for x, p, _, _ in rows)}" '
                 f'stroke-width="2" stroke-linejoin="round"/>')
        for x, p, _, _ in rows:
            cx, cy = X(x), Y(p)
            if marker == "circle":
                o.append(f'<circle class="s{i} mark" cx="{cx:.1f}" cy="{cy:.1f}" r="4" style="stroke-width:1.5"/>')
            elif marker == "square":
                o.append(f'<rect class="s{i} mark" x="{cx - 3.6:.1f}" y="{cy - 3.6:.1f}" width="7.2" height="7.2" '
                         f'rx="1" style="stroke-width:1.5"/>')
            else:
                o.append(f'<polygon class="s{i} mark" points="{cx:.1f},{cy - 4.8:.1f} {cx + 4.6:.1f},{cy + 3.4:.1f} '
                         f'{cx - 4.6:.1f},{cy + 3.4:.1f}" style="stroke-width:1.5"/>')
        ends.append((Y(rows[-1][1]), i, label, rows[-1][1]))
    ends.sort()                                    # direct labels at the right end, kept 16px apart
    ys = [e[0] for e in ends]
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + 16)
    for y, (_, i, label, p) in zip(ys, ends):
        o.append(f'<text class="t{i}" x="{W - R + 10}" y="{y + 4:.1f}" font-size="12" font-weight="600">'
                 f'{label} {100 * p:.0f}%</text>')
    o.append("</svg>")
    return "\n".join(o)


def _steps(path: str) -> int:
    m = re.search(r"snapshot_(\d+)steps\.pt$", path)
    return int(m.group(1)) if m else -1


def row(label: str, d: Dict[str, Any]) -> str:
    pl = [p for p in d["plays"] if p["ahead"]]
    hs = [h for h in d["holdings"] if h["ahead_seen"]]
    n = len(pl)

    def pc(k: int) -> str:
        return "—" if n == 0 else f"{100 * k / n:.0f}%"
    hd = sum(p["outcome"] == "headline" for p in pl)
    ev = sum(p["outcome"] == "event" for p in pl)
    ops = sum(p["outcome"] == "ops" for p in pl)
    sp = sum(p["outcome"] == "space" for p in pl)
    nh = len(hs)
    kept = sum(h["outcome"] == "not played" for h in hs)
    ev_h = sum(h["outcome"] in ("headline", "event") for h in hs)
    # Headlined while not ahead: the event does nothing, so the card is spent for nothing.
    dud = sum(p["outcome"] == "headline" and not p["ahead"] for p in d["plays"])
    behind = sum(not p["ahead"] for p in d["plays"])
    return (f"| {label} | {n:,} | **{pc(hd + ev)}** | {pc(hd)} | {pc(ev)} | **{pc(ops)}** | {pc(sp)} | "
            f"{nh:,} | {'—' if nh == 0 else f'{100 * ev_h / nh:.0f}%'} | {'—' if nh == 0 else f'{100 * kept / nh:.0f}%'} | "
            f"{dud} of {behind} |")


HEADER = ["| snapshot | plays while ahead | as event | headline | AR event | for Ops | space race | "
          "holdings ahead | holding evented | holding not played | headlined while not ahead |",
          "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=57_000)
    ap.add_argument("--dump-dir", default=None, help="one JSON of plays and holdings per checkpoint")
    ap.add_argument("--load", action="store_true", help="tabulate the dumps in --dump-dir instead of playing")
    ap.add_argument("--target-plays", type=int, default=None,
                    help="instead of --games: play until this many plays while ahead, keep exactly that many")
    ap.add_argument("--chunk", type=int, default=4096, help="games per round of --target-plays")
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--plot-svg", default=None, help="also draw the split of plays while ahead (SVG)")
    ap.add_argument("--plot-title", default="Star Wars played while the US is ahead in space")
    ap.add_argument("--branch-at", type=float, default=None, help="mark a run boundary at this step (millions)")
    a = ap.parse_args(argv)
    labels = a.labels or [f"{_steps(p) / 1e6:.0f}M" for p in a.checkpoints]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header = SPLIT_HEADER if a.target_plays else HEADER
    out = list(header)
    points: List[tuple[float, Dict[str, Any]]] = []
    print("\n".join(header), flush=True)
    for path, label in zip(a.checkpoints, labels):
        dump = os.path.join(a.dump_dir, f"{label}.json") if a.dump_dir else None
        if a.load:
            assert dump is not None
            d = json.load(open(dump))
        else:
            model = NeuralAgent.from_checkpoint(path, device=str(dev)).model
            d = (collect(model, a.target_plays, a.seed, a.batch, a.chunk) if a.target_plays
                 else play(model, a.games, a.seed, a.batch))
            d["checkpoint"] = path
            if dump:
                os.makedirs(a.dump_dir, exist_ok=True)
                json.dump(d, open(dump, "w"))
        line = split_row(label, d) if a.target_plays else row(label, d)
        out.append(line)
        print(line, flush=True)
        m = re.match(r"(\d+(?:\.\d+)?)M", label)
        x = float(m.group(1)) if m else _steps(path) / 1e6
        points.append((x, d))
    if a.plot_svg:
        open(a.plot_svg, "w").write(plot_svg(points, a.plot_title, a.branch_at) + "\n")
    if a.output_md:
        open(a.output_md, "w").write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
