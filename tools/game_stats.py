#!/usr/bin/env python3
"""Games people played against the models in the web workbench: fetch, check, rate.

    PYTHONPATH=.:build/release python tools/game_stats.py fetch --output data/game_stats/games.json [--local]
    PYTHONPATH=.:build/release python tools/game_stats.py show data/game_stats/games.json
    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/game_stats.py \\
        verify data/game_stats/games.json [--write-replays]
    PYTHONPATH=.:build/release python tools/game_stats.py rating data/game_stats/games.json \\
        [--ussr-advantage 0] [--max-undos N] [--json out.json]

The page submits a finished game to the stats database (web/stats_worker/, Cloudflare D1);
`fetch` exports its table with wrangler. `verify` re-drives every game played on this build's
engine and checks it ends on the submitted position; a game from another engine names the commit
to check out. `rating` repeats that verification and fits each model player's rating from the clean games
(tools/lib/game_stats.py has the model). See tools/README.md, "Games against people".
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, cast

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.engine_fingerprint import fingerprint as engine_fingerprint
from tools.lib.game_stats import (RatedGame, SubmissionDict, decode_log, fit_rating, load_export, outcome,
                                  play_step, player_key, rated_game, start_state, summarize_submission, verify)
from web.server.replay import ReplayLogger, replays_dir
from web.server.replay_types import GameStateDict, ReplayActionDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_DIR = os.path.join(_ROOT, "web", "stats_worker")
#: The D1 database's name in web/stats_worker/wrangler.toml.
DATABASE = "ts-game-stats"


def cmd_fetch(a: argparse.Namespace) -> int:
    """Every submission in the database, as wrangler's JSON: the remote one (needs `wrangler login`),
    or with --local the one `wrangler dev` keeps in web/stats_worker/.wrangler/."""
    out = subprocess.run(
        ["npx", "wrangler", "d1", "execute", DATABASE, "--local" if a.local else "--remote", "--json",
         "--command", "SELECT body FROM games ORDER BY received_at"],
        cwd=WORKER_DIR, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr or out.stdout, file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(out.stdout)
    print(f"game_stats: {len(load_export(a.output))} games -> {a.output}")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    print("| ended | login | rating | human | result | model | search | clean | notes |")
    print("|:--|:--|--:|:--|:--|:--|:--|:--|:--|")
    for sub in load_export(a.export):
        s = summarize_submission(sub)
        model = sub["models"].get(s["models"][0], None) if len(s["models"]) == 1 else None
        notes = [f"{k}={v}" for k, v in s.items()
                 if k in ("human_moves_for_model", "suggested_moves", "analysis_moves", "dev_view_moves",
                          "forced_dice", "undos", "debug_overrides") and v]
        if s["loaded_position"]:
            notes.append("loaded")
        if len(s["models"]) > 1 or len(s["searches"]) > 1:
            notes.append("settings changed")
        r = sub["result"]
        print(f"| {sub['ended_at'][:16]} | {sub['playdek']['login']} | {sub['playdek']['rating']} | "
              f"{s['human_side'] or '-'} | {r['winner']} {r['margin']:+d} T{r['end_turn']} | "
              f"{model['label'] if model else '-'} | {','.join(s['searches']) or '-'} | "
              f"{'yes' if s['clean'] else 'no'} | {' '.join(notes)} |")
    return 0


def _rebuild(sub: SubmissionDict) -> str:
    """The full .tslog.json of a verified game, written flat into the replays directory."""
    seed, steps = decode_log(sub["log"])
    state = start_state(seed, sub["start_position"])
    model_side = summarize_submission(sub)["model_side"]
    log = ReplayLogger(game_id=f"stats_{sub['id']}", seed=seed,
                       us_player="model" if model_side == "US" else "human",
                       ussr_player="model" if model_side == "USSR" else "human")
    log.start_position = sub["start_position"]
    for i, st in enumerate(steps):
        before_turn, before_ar, phase = int(state.turn), int(state.action_round), str(state.current_phase.name)
        ma = st.micro_action()
        flat = int(ts.encode_micro_action(state, ma))
        desc = ActionEncoder.get_action_name(state, flat) if flat >= 0 else f"{ma.decision_type.name} {st.primary_id}"
        play_step(state, st)
        action: ReplayActionDict = {
            "flat_action_idx": flat,
            "decision_type": st.decision_type,
            "primary_id": st.primary_id,
            "secondary_id": st.secondary_id,
            "flags": st.flags,
            "card_id": st.primary_id if st.decision_type in (1, 2, 3, 4) else None,
            "target_id": st.secondary_id if st.decision_type in (5, 6, 7) else None,
        }
        log.log_step(step_index=i + 1, turn=before_turn, ar=before_ar, phase=phase, player=st.mover or "NONE",
                     action=action, description=f"[{st.source}] {desc}",
                     state_snapshot=cast(GameStateDict, ts.state_to_dict(state)))
    r = outcome(state)
    log.set_result(r["winner"], r["margin"], r["end_turn"], r["reason"])
    return log.save(os.path.join(replays_dir(), f"stats_{sub['id']}.tslog.json"))


def cmd_verify(a: argparse.Namespace) -> int:
    fp = engine_fingerprint()
    counts: Dict[str, int] = {}
    for sub in load_export(a.export):
        v = verify(sub, fp)
        counts[v.status] = counts.get(v.status, 0) + 1
        line = f"{sub['id']}  {v.status}"
        if v.detail:
            line += f"  {v.detail}"
        if v.status == "ok" and a.write_replays:
            line += f"  -> {_rebuild(sub)}"
        print(line)
    print("game_stats: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    return 0 if not set(counts) - {"ok", "other-engine"} else 1


def cmd_rating(a: argparse.Namespace) -> int:
    by_player: Dict[str, List[RatedGame]] = {}
    skipped: Dict[str, int] = {}
    fingerprint = engine_fingerprint()
    for sub in load_export(a.export):
        verdict = verify(sub, fingerprint)
        if verdict.status != "ok":
            why = f"verification: {verdict.status}"
            skipped[why] = skipped.get(why, 0) + 1
            continue
        s = summarize_submission(sub)
        why = ("not clean" if not s["clean"] else
               "too many undos" if a.max_undos is not None and s["undos"] > a.max_undos else
               "no rating" if sub["playdek"]["rating"] <= 0 else None)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        by_player.setdefault(player_key(sub, s), []).append(rated_game(sub, s))
    fits = [fit_rating(p, gs, ussr_advantage=a.ussr_advantage, prior_mean=a.prior_mean, prior_sd=a.prior_sd,
                       bootstrap=a.bootstrap)
            for p, gs in sorted(by_player.items()) if len(gs) >= a.min_games]
    print(f"Elo on the opponents' Playdek ratings; USSR advantage {a.ussr_advantage:+g}; "
          f"prior sd {a.prior_sd:g}; 95% interval: bootstrap over players.\n")
    print("| model player | games | players | W-D-L (model) | as US | as USSR | mean opponent | rating | se | 95% |")
    print("|:--|--:|--:|:--|:--|:--|--:|--:|--:|:--|")
    for f in fits:
        def wdl(t: tuple[int, int, int]) -> str:
            return f"{t[0]}-{t[1]}-{t[2]}"
        print(f"| {f.player} | {f.games} | {f.humans} | {wdl(f.record)} | {wdl(f.by_side['US'])} | "
              f"{wdl(f.by_side['USSR'])} | {f.mean_opponent:.0f} | {f.rating:.0f} | {f.se:.0f} | "
              f"{f.interval[0]:.0f}..{f.interval[1]:.0f} |")
    if skipped:
        print("\nLeft out: " + ", ".join(f"{n} {k}" for k, n in sorted(skipped.items())))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"ussr_advantage": a.ussr_advantage, "prior_sd": a.prior_sd, "skipped": skipped,
                       "fits": [dataclasses.asdict(f) for f in fits]}, fh, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="export the remote database with wrangler")
    p.add_argument("--output", required=True, help="where to write the export (JSON)")
    p.add_argument("--local", action="store_true", help="the local database of `wrangler dev`, not the deployed one")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("show", help="list the games and how each was played")
    p.add_argument("export")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("verify", help="re-drive each game in this build and check its end")
    p.add_argument("export")
    p.add_argument("--write-replays", action="store_true",
                   help="write each verified game's full .tslog.json into the replays directory")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("rating", help="fit each model player's rating on Playdek's scale")
    p.add_argument("export")
    p.add_argument("--ussr-advantage", type=float, default=0.0,
                   help="the USSR's edge in rating points between equal players (default 0)")
    p.add_argument("--prior-mean", type=float, default=None,
                   help="the prior's mean (default: the mean opponent rating)")
    p.add_argument("--prior-sd", type=float, default=400.0)
    p.add_argument("--max-undos", type=int, default=None, help="leave out games with more takebacks")
    p.add_argument("--min-games", type=int, default=1)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--json", default=None, help="also write the fits here")
    p.set_defaults(fn=cmd_rating)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
