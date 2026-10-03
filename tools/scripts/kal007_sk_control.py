#!/usr/bin/env python3
"""Does South Korea's control move a checkpoint's choice to headline Soviets Shoot Down KAL-007?

KAL-007 (US, 4 Ops): DEFCON -1 and US +2 VP; *if the US controls South Korea* the US may also
place influence or realign with the card's 4 Ops. So control of South Korea is what makes the
event worth more than its VP.

In greedy self-play of the checkpoint, at every US headline choice with KAL-007 in the US hand,
read the policy's probability of headlining it twice: in the position as played, and in the same
position with South Korea's control flipped by the smallest change to US influence there --
controlled (US >= USSR + 3; stability 3) -> US set to USSR + 2; not controlled -> US set to
USSR + 3. Nothing else in the state is touched, and the engine is never stepped from the edited
copy. The observation is rebuilt from the edited state (the engine keeps no control cache:
`set_country` writes the influence and every derived feature is recomputed).

    PYTHONPATH=.:build/release python tools/scripts/kal007_sk_control.py \
        --checkpoint data/checkpoints/_soups/shallow_E7-02+03+04+05_1200M.pt --games 4096
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

KAL = 89
SOUTH_KOREA = 44
SK_STABILITY = 3


def _us_controls(us: int, ussr: int) -> bool:
    return us >= SK_STABILITY and us - ussr >= SK_STABILITY


def _flipped_us(us: int, ussr: int) -> int:
    """US influence in South Korea after the smallest change that flips US control."""
    return ussr + SK_STABILITY - 1 if _us_controls(us, ussr) else ussr + SK_STABILITY


def _kal_slot(st: ts.GameState, mask: np.ndarray) -> Optional[int]:
    for idx in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET]):
        if int(ts.decode_flat_action(st, int(idx)).primary_id) == KAL:
            return int(idx)
    return None


def _probs(model: Any, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(obs.astype(np.float32)).to(device),
                       torch.from_numpy(mask).to(device))[0].float()
    m = torch.from_numpy(mask).to(device).bool()
    logits = logits.masked_fill(~m, float("-inf"))
    return torch.softmax(logits, -1).cpu().numpy()


def play(model: Any, games: int, seed: int, batch: int) -> List[Dict[str, Any]]:
    from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
    check_obs_width(model)
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    recs: List[Dict[str, Any]] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        for _ in range(20_000):
            if all(done):
                break
            obs_np = np.asarray(obs, dtype=np.float32)[:, :width]
            masks_np = np.asarray(masks)
            with torch.no_grad():
                logits = model(torch.from_numpy(obs_np).to(device), torch.from_numpy(masks_np).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            # Collect this step's probe positions, then score them in one batch.
            probe_obs: List[np.ndarray] = []
            probe_mask: List[np.ndarray] = []
            pending: List[Dict[str, Any]] = []
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st) or st.current_phase != ts.Phase.HEADLINE:
                    continue
                ctx = st.ctx()
                if (ctx.decision_type != ts.DecisionType.SELECT_CARD or int(ctx.resolving_card) != 0
                        or ctx.decision_player != ts.Player.US):
                    continue
                if not ts.in_hand_of(st.get_card_location(KAL), ts.Player.US):
                    continue
                slot = _kal_slot(st, masks_np[i])
                if slot is None:
                    continue
                sk = st.get_country(SOUTH_KOREA)
                us_i, su_i = int(sk.us_influence), int(sk.ussr_influence)
                flipped = st.clone()
                new_us = _flipped_us(us_i, su_i)
                flipped.set_country(SOUTH_KOREA, new_us, su_i)
                assert bool(ts.Scoring.is_controlled_by(flipped, SOUTH_KOREA, ts.Player.US)) != _us_controls(us_i, su_i)
                assert bool(ts.Scoring.is_controlled_by(st, SOUTH_KOREA, ts.Player.US)) == _us_controls(us_i, su_i)
                o_nat = np.asarray(ts.extract_observation_features(st, ts.Player.US, feats), dtype=np.float32)
                o_flip = np.asarray(ts.extract_observation_features(flipped, ts.Player.US, feats), dtype=np.float32)
                m_flip = np.asarray(ts.get_flat_action_mask(flipped)).astype(masks_np.dtype)
                # The probe must read the very observation the game was played from.
                assert np.array_equal(o_nat, obs_np[i]), "extracted observation differs from the env's"
                assert np.array_equal(m_flip, masks_np[i]), "flipping South Korea changed the headline mask"
                probe_obs += [o_nat, o_flip]
                probe_mask += [masks_np[i], m_flip]
                hand = [c for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), ts.Player.US)]
                pending.append({
                    "game": b0 + i, "turn": int(st.turn), "defcon": int(st.defcon), "vp": int(st.victory_points),
                    "sk_us": us_i, "sk_ussr": su_i, "sk_us_flipped": new_us,
                    "us_controls": _us_controls(us_i, su_i), "slot": slot, "hand": hand,
                    "greedy_kal": int(actions[i]) == slot,
                })
            if pending:
                p = _probs(model, np.stack(probe_obs), np.stack(probe_mask))
                for k, r in enumerate(pending):
                    r["p_natural"] = float(p[2 * k, r["slot"]])
                    r["p_flipped"] = float(p[2 * k + 1, r["slot"]])
                    recs.append(r)
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
    return recs


def _row(label: str, xs: Sequence[float]) -> str:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return f"| {label} | 0 | — | — | — |"
    return (f"| {label} | {a.size:,} | {a.mean():.3f} | {np.median(a):.3f} | "
            f"{100 * float(np.mean(a > 0.5)):.0f}% |")


def report(recs: Sequence[Dict[str, Any]], games: int, title: str = "") -> str:
    ctl = [r for r in recs if r["us_controls"]]
    nct = [r for r in recs if not r["us_controls"]]
    out = []
    if title:
        out += [title, ""]
    out += [f"{len(recs):,} US headline choices with KAL-007 in hand, {games:,} greedy self-play games; "
            f"{len(ctl):,} with the US controlling South Korea, {len(nct):,} without.", "",
            "| scenario | positions | mean P(headline KAL-007) | median | P > 0.5 |", "|:---|---:|---:|---:|---:|",
            _row("natural, US controls SK", [r["p_natural"] for r in ctl]),
            _row("natural, US does not control SK", [r["p_natural"] for r in nct]),
            _row("flipped: controlled -> not (US = USSR + 2)", [r["p_flipped"] for r in ctl]),
            _row("flipped: not -> controlled (US = USSR + 3)", [r["p_flipped"] for r in nct]),
            "", "Paired change in the same positions (flipped − natural):", "",
            "| flip | positions | mean Δ | median Δ | Δ < 0 | Δ > 0 |", "|:---|---:|---:|---:|---:|---:|"]
    for label, rs in (("lose control", ctl), ("gain control", nct)):
        d = np.asarray([r["p_flipped"] - r["p_natural"] for r in rs], dtype=np.float64)
        if d.size:
            out.append(f"| {label} | {d.size:,} | {d.mean():+.3f} | {np.median(d):+.3f} | "
                       f"{100 * float(np.mean(d < 0)):.0f}% | {100 * float(np.mean(d > 0)):.0f}% |")
    # DEFCON improves at every turn's end, so a headline is never at DEFCON 2: 3-5 only.
    out += ["", "By DEFCON at the headline (mean P natural → flipped, paired mean Δ):", "",
            "| DEFCON | US controls SK: n | natural | flipped | Δ | not controlled: n | natural | flipped | Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dc in sorted({int(r["defcon"]) for r in recs}):
        cells = []
        for rs in ([r for r in ctl if r["defcon"] == dc], [r for r in nct if r["defcon"] == dc]):
            if rs:
                pn = np.asarray([r["p_natural"] for r in rs])
                pf = np.asarray([r["p_flipped"] for r in rs])
                cells.append(f"{len(rs):,} | {pn.mean():.3f} | {pf.mean():.3f} | {(pf - pn).mean():+.3f}")
            else:
                cells.append("0 | — | — | —")
        out.append(f"| {dc} | " + " | ".join(cells) + " |")
    out += ["", "By the size of the flip (US influence moved in South Korea):", "",
            "| flip | influence moved | n | natural | flipped | mean Δ | median Δ |", "|:---|---:|---:|---:|---:|---:|---:|"]
    for label, rs in (("lose control", ctl), ("gain control", nct)):
        by: Dict[int, List[Dict[str, Any]]] = {}
        for r in rs:
            by.setdefault(abs(int(r["sk_us_flipped"]) - int(r["sk_us"])), []).append(r)
        for k in sorted(by):
            pn = np.asarray([r["p_natural"] for r in by[k]])
            pf = np.asarray([r["p_flipped"] for r in by[k]])
            out.append(f"| {label} | {k} | {len(by[k]):,} | {pn.mean():.3f} | {pf.mean():.3f} | "
                       f"{(pf - pn).mean():+.3f} | {np.median(pf - pn):+.3f} |")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=56_000)
    ap.add_argument("--dump", default=None, help="write the per-position records here (JSON)")
    ap.add_argument("--load", default=None, help="report from a dump instead of playing")
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    if a.load:
        d = json.load(open(a.load))
        recs, games = d["records"], int(d["games"])
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model
        recs, games = play(model, a.games, a.seed, a.batch), a.games
        if a.dump:
            json.dump({"checkpoint": a.checkpoint, "games": games, "seed": a.seed, "records": recs}, open(a.dump, "w"))
    md = report(recs, games)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
