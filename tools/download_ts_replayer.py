#!/usr/bin/env python3
"""Download human Twilight Struggle games from ts-replayer.fly.dev, once.

The site hosts community-uploaded logs produced by TS Espionage, a mod for the Playdek/Steam
app. Each replay page embeds four JSON islands, which together are everything we need -- there
is no HTML or prose parsing involved:

    all-turns     one entry per action: turn, player, phase, card played, a structured move
                  text ("USSR +2 in Thailand [0][2]"), score, DEFCON, and the full board
    unique-turns  one entry per game turn
    hands         BOTH players' hands per turn, so every decision is usable, not half
    stats         ops usage by category, coup rolls, score trajectory

Fetched once and cached locally so experiments never touch the host again. The site has no
robots.txt; this still throttles to one request a second, identifies itself, retries politely,
and skips anything already on disk so a re-run costs nothing.

    PYTHONPATH=. .venv/bin/python3 tools/download_ts_replayer.py
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://ts-replayer.fly.dev/replay/{}"
UA = "twilight-struggle-ai-research/1.0 (offline dataset build; one request/sec)"
ISLANDS = ("all-turns", "unique-turns", "hands", "stats")
ISLAND_RE = re.compile(
    r'<script id="([^"]+)" type="application/json">(.*?)</script>', re.S)


def fetch(url: str, timeout: float, retries: int = 3) -> "tuple[int, str]":
    """Returns (status, body). Retries 5xx and network errors with backoff; 404 is final."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, ""
            if attempt == retries - 1:
                return e.code, ""
        except Exception:
            if attempt == retries - 1:
                return 0, ""
        time.sleep(2.0 * (attempt + 1))
    return 0, ""


def extract(html: str) -> "dict | None":
    found = {m.group(1): m.group(2) for m in ISLAND_RE.finditer(html)}
    if not all(k in found for k in ISLANDS):
        return None
    out = {}
    for k in ISLANDS:
        try:
            out[k.replace("-", "_")] = json.loads(found[k])
        except json.JSONDecodeError:
            return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out", default=None,
        help="Where to store the replays. Defaults to datasets/ts_replayer in the shared "
             "data tree (tools/lib/data_root.py: the main checkout's data/), so the corpus is "
             "downloaded once rather than once per checkout or git worktree.")
    ap.add_argument("--start", type=int, default=1)
    # 324 is the last id of the validated corpus: 300 files, 266 distinct games, the set every
    # replayer test and the human dataset were built on. The site has since grown (ids 325-340
    # existed on 2026-09-23); those are not validated by the converter, so a default download
    # must not pull them in silently. Pass --end explicitly to fetch past it.
    ap.add_argument("--end", type=int, default=324,
                    help="inclusive; 324 is the last id of the validated 300-file corpus")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    if args.out is None:
        from tools.lib.corpus_paths import corpus_dir
        args.out = str(corpus_dir())
    os.makedirs(args.out, exist_ok=True)
    print(f"Caching replays in {args.out}")
    got = missing = cached = failed = 0

    for gid in range(args.start, args.end + 1):
        path = os.path.join(args.out, f"{gid}.json.gz")
        if os.path.exists(path):
            cached += 1
            continue

        status, body = fetch(BASE.format(gid), args.timeout)
        if status == 404:
            missing += 1
        elif status == 200:
            data = extract(body)
            if data is None:
                failed += 1
                print(f"  {gid}: 200 but no parseable data islands", flush=True)
            else:
                data["replay_id"] = gid
                data["source"] = BASE.format(gid)
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
                got += 1
        else:
            failed += 1
            print(f"  {gid}: HTTP {status} after retries", flush=True)

        if gid % 25 == 0:
            print(f"  ...{gid}: {got} saved, {cached} cached, {missing} absent, "
                  f"{failed} failed", flush=True)
        time.sleep(args.delay)

    print(f"\ndone: {got} downloaded, {cached} already cached, {missing} absent, "
          f"{failed} failed -> {args.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
