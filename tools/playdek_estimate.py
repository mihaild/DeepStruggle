#!/usr/bin/env python3
"""Fit the leaderboard's players on Playdek's rating scale from the stats database.

    PYTHONPATH=.:build/release python tools/playdek_estimate.py --output playdek.json   # wrangler, --remote
    PYTHONPATH=.:build/release python tools/playdek_estimate.py --export dump.txt --output playdek.json
    PYTHONPATH=. python tools/leaderboard.py fit --playdek playdek.json --output leaderboard.json

Without --export it reads the deployed database with wrangler (from `wrangler login`, or
CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID on CI; read access to D1 is enough). The rows stay in
memory, and the output and this program's log hold aggregates only, so both may be published.
tools/lib/playdek_estimate.py has the model; tools/README.md, "Games against people", the use.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from tools.lib.engine_fingerprint import fingerprint as engine_fingerprint, require_fresh
from tools.lib.game_stats import SubmissionDict, rated_game, rating_check
from tools.lib.leaderboard import LeaderboardError, ROOT, load
from tools.lib.playdek_estimate import DEFAULT_DEVIATION_SD, KEYED_DEFAULT_COMMIT, Estimate, PlayedGame, estimate

#: The D1 database's name in web/stats_worker/wrangler.toml.
DATABASE = "ts-game-stats"
WORKER_DIR = os.path.join(str(ROOT), "web", "stats_worker")
#: Only what the fit reads: never `ip_hash`. `received_at` is the Worker's stamp, not the submitter's.
QUERY = "SELECT received_at, body FROM games ORDER BY received_at"
#: A game on another engine that does not replay here counts only if it arrived before this: E7's and
#: early E8's games are wanted (owner, 2026-10-31), and were all in the database by this day, while a
#: forged record cannot now arrive before it. A game arriving later must replay.
UNVERIFIED_BEFORE = "2026-10-31"


def rows_from_text(text: str) -> List[Tuple[str, SubmissionDict]]:
    """(received_at, submission) of `wrangler d1 execute --json` output, one per id. A terminal's dump
    has wrangler's banner above the JSON, which is skipped."""
    start = 0 if text.startswith("[") else text.find("\n[") + 1
    data: Any = json.loads(text[start:])
    rows: Dict[str, Tuple[str, SubmissionDict]] = {}
    for item in data:
        for row in item["results"]:
            sub = cast(SubmissionDict, json.loads(row["body"]))
            rows.setdefault(sub["id"], (row["received_at"], sub))
    return list(rows.values())


def read_database() -> List[Tuple[str, SubmissionDict]]:
    out = subprocess.run(["npx", "wrangler", "d1", "execute", DATABASE, "--remote", "--json", "--command", QUERY],
                         cwd=WORKER_DIR, capture_output=True, text=True)
    if out.returncode != 0:
        # Only that it failed and its exit code: nothing wrangler printed can reach a public log.
        raise LeaderboardError(f"wrangler exited {out.returncode}")
    return rows_from_text(out.stdout)


def counted_games(rows: Sequence[Tuple[str, SubmissionDict]], fingerprint: str) -> Tuple[List[PlayedGame], Dict[str, int]]:
    """The games that pass `tools/game_stats.py rating`'s check, and how many do not and why."""
    played: List[PlayedGame] = []
    left_out: Dict[str, int] = {}
    for received_at, sub in rows:
        check = rating_check(sub, fingerprint, include_unverified=received_at < UNVERIFIED_BEFORE)
        s = check.summary
        if check.why is not None or s is None:
            why = check.why or "no summary"
            left_out[why] = left_out.get(why, 0) + 1
            continue
        game = rated_game(sub, s)
        played.append(PlayedGame(game.login, game.human_rating, game.score, sub["models"][s["models"][0]]["label"],
                                 s["searches"][0], sub["build"]["commit"]))
    return played, left_out


def build_precedes_keyed_default(cache: Dict[str, Optional[bool]], commit: str) -> Optional[bool]:
    """Whether a build's commit precedes KEYED_DEFAULT_COMMIT; None if this clone does not have it
    (CI's checkout needs the whole history)."""
    if commit not in cache:
        code = subprocess.run(["git", "merge-base", "--is-ancestor", KEYED_DEFAULT_COMMIT, commit],
                              cwd=str(ROOT), capture_output=True).returncode
        cache[commit] = False if code == 0 else True if code == 1 else None
    return cache[commit]


def summary(est: Estimate) -> str:
    excluded = ", ".join(f"{n} {why}" for why, n in sorted(est["excluded"].items())) or "none"
    lines = [f"playdek: {est['games']} games against {est['humans']} people, to {est['as_of']}; left out: {excluded}",
             f"playdek: offset {est['offset']:+.1f} on {est['reference_epoch']}'s scale, deviation sd {est['deviation_sd']:g}"]
    played = [(pid, p) for pid, p in est["players"].items() if p["games"]]
    for pid, p in sorted(played, key=lambda x: -x[1]["rating"]):
        lines.append(f"  {p['rating']:6.0f} +- {p['se']:3.0f}  {p['w']}-{p['d']}-{p['l']:<3d} {pid}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    require_fresh()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="where to write the estimate (JSON, aggregates only)")
    ap.add_argument("--export", default=None, help="a dump of the games table with received_at and body, "
                                                   "instead of reading the database")
    ap.add_argument("--deviation-sd", type=float, default=DEFAULT_DEVIATION_SD,
                    help="how far a played player may stray from its leaderboard gap, in rating points")
    a = ap.parse_args(argv)
    try:
        if a.export:
            with open(a.export, encoding="utf-8") as f:
                rows = rows_from_text(f.read())
        else:
            rows = read_database()
        if not rows:
            raise LeaderboardError("the database holds no games")
        played, left_out = counted_games(rows, engine_fingerprint())
        cache: Dict[str, Optional[bool]] = {}
        est = estimate(played, left_out, max(r[0] for r in rows), load(),
                       lambda c: build_precedes_keyed_default(cache, c), deviation_sd=a.deviation_sd)
    except LeaderboardError as e:
        print(f"playdek_estimate: {e}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(est, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(summary(est))
    return 0


if __name__ == "__main__":
    sys.exit(main())
