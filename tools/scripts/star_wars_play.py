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


def play(model: Any, games: int, seed: int, batch: int) -> Dict[str, Any]:
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
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
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
                    new: Dict[str, Any] = {"game": b0 + i, "turn": int(st.turn), "ahead_seen": False, "outcome": "not played"}
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
                        rec = {"game": b0 + i, "turn": int(st.turn), "ahead": ahead,
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
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    labels = a.labels or [f"{_steps(p) / 1e6:.0f}M" for p in a.checkpoints]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = list(HEADER)
    print("\n".join(HEADER), flush=True)
    for path, label in zip(a.checkpoints, labels):
        dump = os.path.join(a.dump_dir, f"{label}.json") if a.dump_dir else None
        if a.load:
            assert dump is not None
            d = json.load(open(dump))
        else:
            model = NeuralAgent.from_checkpoint(path, device=str(dev)).model
            d = play(model, a.games, a.seed, a.batch)
            d["checkpoint"] = path
            if dump:
                os.makedirs(a.dump_dir, exist_ok=True)
                json.dump(d, open(dump, "w"))
        line = row(label, d)
        out.append(line)
        print(line, flush=True)
    if a.output_md:
        open(a.output_md, "w").write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
