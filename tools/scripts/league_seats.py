#!/usr/bin/env python3
"""P45: a league with one exploiter lineage per seat, beside a main agent that is already training.

    PYTHONPATH=.:build/release python tools/scripts/league_seats.py \\
        --main-dir <main run dir> --main-pid <pid> --main-steps <the main's --train-steps> \\
        --league-dir <the main's --league-dirs> --log-dir <dir> --every 800000000 --gen-steps 400000000 \\
        --us-start <resume state> --ussr-start <resume state> \\
        --us-name '<name with {step}>' --ussr-name '<name with {step}>' \\
        --exploiter-args "<R108 / R109's flags: architecture, recipe, --zero-avoid-scope setup, ...>" \\
        --description "..."

The main agent is an ordinary `tools/train.py` run launched with `--league-dirs <league-dir>`; this driver
adopts it by PID and writes nothing into it. The league directory is the only thing they share.

* **A generation every `--every` steps of the main:** the main's newest 80M SWA is the generation's target; the
  US exploiter trains `--gen-steps` against it (`--opponent-lock-side us`, `--opponent-frac 1.0`, the target
  played greedily), then the USSR exploiter does, one trainer at a time beside the main.
* **Continued lineages:** each exploiter resumes from its own previous generation, so its counter-play carries
  over; the first generation of a seat starts from `--{seat}-start`. A lineage whose margin has gone negative
  twice in a row restarts from the main's newest tagged resume state.
* **Every generation is published** -- its final snapshot linked into the league directory, where the main's
  pool reads its seat from the exploiter's run metadata -- with its margin in `<league-dir>/events.jsonl`: its
  greedy win rate in its seat against the target minus the target's own greedy self-play win rate in that
  seat, `--judge-games` deals a side, on the CPU. A record, not a gate (P24's publish bar kept that league
  empty).
* **Capacity per seat:** the league directory keeps the newest `--capacity` members of each seat; older links
  are removed (the main's pool evicts the same way).

Events go to `<league-dir>/events.jsonl`. The driver ends when the main does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai.training.run_name import is_run_name  # noqa: E402  (ROOT must be importable first)
from tools.lib.data_root import checkpoints_dir  # noqa: E402

SEATS = ("us", "ussr")
SNAP_RE = re.compile(r"^snapshot_(\d+)steps\.pt$")
RESUME_RE = re.compile(r"^resume_(\d+)steps\.pt$")
LINK_RE = re.compile(r"^(us|ussr)_g(\d+)_.*_snapshot_(\d+)steps\.pt$")


def snapshots(run_dir: str) -> List[int]:
    return sorted(int(m.group(1)) for f in os.listdir(run_dir) if (m := SNAP_RE.match(f)))


def latest_resume(run_dir: str) -> Optional[str]:
    found = [(int(m.group(1)), f) for f in os.listdir(run_dir) if (m := RESUME_RE.match(f))]
    return os.path.join(run_dir, max(found)[1]) if found else None


def steps_of_state(path: str) -> int:
    m = re.search(r"(\d+)steps", os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot tell the steps of {path}: name a resume_<n>steps.pt state")
    return int(m.group(1))


@dataclass
class Ops:
    """What the driver does to the world, injectable so its logic is tested without training."""
    #: train(seat, start_state, target, run_name, train_steps, description) -> the exploiter's run dir, after it exits
    train: Callable[[str, str, str, str, int, str], str]
    #: swa(run_dir, hi_steps) -> the path of that run's (hi-80M .. hi) SWA
    swa: Callable[[str, int], str]
    #: match(a, b) -> {"a_as_us": win rate, "a_as_ussr": win rate}, greedy, both seats
    match: Callable[[str, str], Dict[str, float]]
    #: the main's logged steps, and whether it is still running
    main_steps: Callable[[], int]
    main_alive: Callable[[], bool]
    sleep: Callable[[float], None] = time.sleep


@dataclass
class SeatLeague:
    main_dir: str
    league_dir: str
    every: int
    gen_steps: int
    starts: Dict[str, str]
    names: Dict[str, str]
    description: str
    ops: Ops
    capacity: int = 4
    first_at: int = 0
    poll: float = 60.0
    lineage: Dict[str, str] = field(default_factory=dict)       # seat -> the state its next generation resumes
    negatives: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SEATS})
    generation: int = 0

    def event(self, kind: str, **kw: Any) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind, **kw}
        with open(os.path.join(self.league_dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[league] {kind} {kw}", flush=True)

    def publish(self, seat: str, run_dir: str) -> str:
        """Link the generation's final snapshot into the league directory and keep `capacity` a seat."""
        n = snapshots(run_dir)[-1]
        src = os.path.join(run_dir, f"snapshot_{n}steps.pt")
        run = os.path.basename(run_dir).rsplit("_", 2)[0]
        link = os.path.join(self.league_dir, f"{seat}_g{self.generation}_{run}_snapshot_{n}steps.pt")
        if not os.path.exists(link):
            os.symlink(src, link)
        mine = sorted((int(m.group(2)), f) for f in os.listdir(self.league_dir)
                      if (m := LINK_RE.match(f)) and m.group(1) == seat)
        for _, f in mine[:-self.capacity] if len(mine) > self.capacity else []:
            os.unlink(os.path.join(self.league_dir, f))
            self.event("evicted", seat=seat, link=f)
        return link

    def run_generation(self) -> None:
        self.generation += 1
        main_now = snapshots(self.main_dir)[-1]
        hi = main_now - main_now % 10_000_000
        target = self.ops.swa(self.main_dir, hi)
        bar = self.ops.match(target, target)
        bars = {"us": bar["a_as_us"], "ussr": bar["a_as_ussr"]}
        self.event("generation", generation=self.generation, target=target, main_snapshot=main_now, bars=bars)
        for seat in SEATS:                                        # US first, then the USSR, one at a time
            start = self.lineage.get(seat) or self.starts[seat]
            if start == "main":
                newest = latest_resume(self.main_dir)
                if newest is None:
                    raise RuntimeError(f"no tagged resume state in {self.main_dir} to start the {seat} lineage from")
                start = newest
            begin = steps_of_state(start)
            name = self.names[seat].format(step=round(begin / 1e6), gen=self.generation)
            run_dir = self.ops.train(seat, start, target, name, begin + self.gen_steps,
                                     f"{self.description} Generation {self.generation}, the {seat.upper()} lineage, "
                                     f"against {os.path.basename(target)}.")
            last = snapshots(run_dir)[-1]
            swa = self.ops.swa(run_dir, last - last % 10_000_000)
            res = self.ops.match(swa, target)
            rate = res[f"a_as_{seat}"]
            margin = rate - bars[seat]
            link = self.publish(seat, run_dir)
            self.event("published", generation=self.generation, seat=seat, run=run_dir, link=link,
                       rate=rate, bar=bars[seat], margin=margin)
            self.negatives[seat] = self.negatives[seat] + 1 if margin < 0 else 0
            if self.negatives[seat] >= 2:
                self.lineage[seat] = "main"                       # restart from the main's state
                self.negatives[seat] = 0
                self.event("lineage_restart", seat=seat, generation=self.generation)
            else:
                resume = os.path.join(run_dir, "resume_state.pt")
                self.lineage[seat] = latest_resume(run_dir) or resume

    def run(self) -> int:
        next_at = self.first_at
        while self.ops.main_alive():
            if self.ops.main_steps() >= next_at:
                self.run_generation()
                next_at = max(next_at + self.every, self.ops.main_steps())
                continue
            self.ops.sleep(self.poll)
        self.event("main_exited", generations=self.generation)
        return 0


# ---- the real operations ------------------------------------------------------------------------------

def _logged_steps(run_dir: str) -> int:
    last = 0
    try:
        with open(os.path.join(run_dir, "training_metrics.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("total_steps"), int):
                    last = row["total_steps"]
    except OSError:
        pass
    return last


def _run_dir(name: str, started: float, timeout: float = 900.0) -> str:
    pat = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}$")
    for _ in range(int(timeout / 5)):
        root = checkpoints_dir()
        cands = [os.path.join(root, d) for d in os.listdir(root) if pat.match(d)]
        cands = [d for d in cands if os.path.getmtime(d) >= started - 5]
        if cands:
            return max(cands, key=os.path.getmtime)
        time.sleep(5)
    raise RuntimeError(f"no run directory for {name} appeared within {timeout:.0f}s")


def real_ops(args: argparse.Namespace) -> Ops:
    common = shlex.split(args.exploiter_args)
    py = sys.executable

    def train(seat: str, start: str, target: str, name: str, steps: int, desc: str) -> str:
        t = time.time()
        with open(os.path.join(args.log_dir, f"{name}.log"), "a", encoding="utf-8") as log:
            proc = subprocess.Popen([py, "tools/train.py", *common, "--resume", start, "--opponent-checkpoints", target,
                                     "--opponent-frac", "1.0", "--opponent-lock-side", seat,
                                     "--train-steps", str(steps), "--run-name", name, "--description", desc],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            run_dir = _run_dir(name, t)
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"exploiter {name} exited {proc.returncode}; see {args.log_dir}/{name}.log")
        return run_dir

    def swa(run_dir: str, hi: int) -> str:
        lo = hi - 80_000_000
        out = os.path.join(run_dir, f"swa_{lo // 1_000_000}-{hi // 1_000_000}M.pt")
        if not os.path.exists(out):
            snaps = [os.path.join(run_dir, f"snapshot_{n}steps.pt") for n in snapshots(run_dir)
                     if lo + 5_000_000 < n <= hi + 5_000_000]
            subprocess.run([py, "tools/scripts/average_weights.py", "--snapshots", *snaps, "--output", out],
                           cwd=ROOT, check=True, capture_output=True)
        return out

    def match(a: str, b: str) -> Dict[str, float]:
        out = os.path.join(args.log_dir, f"match_{int(time.time())}.json")
        subprocess.run([py, "tools/tournament.py", "--models", a, b, "--games-per-side", str(args.judge_games),
                        "--temperature", "0", "--device", "cpu", "--output-json", out],
                       cwd=ROOT, check=True, capture_output=True)
        m = json.load(open(out))["matchup"]
        return {"a_as_us": float(m["win_rate_a_as_us"]), "a_as_ussr": float(m["win_rate_a_as_ussr"])}

    def alive() -> bool:
        try:
            os.kill(args.main_pid, 0)
        except ProcessLookupError:
            return False
        return _logged_steps(args.main_dir) < args.main_steps

    return Ops(train=train, swa=swa, match=match, main_steps=lambda: _logged_steps(args.main_dir), main_alive=alive)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-dir", required=True)
    ap.add_argument("--main-pid", type=int, required=True)
    ap.add_argument("--main-steps", type=int, required=True, help="the main's --train-steps: the driver ends there")
    ap.add_argument("--league-dir", required=True, help="the main's --league-dirs")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--every", type=int, default=800_000_000, help="main steps between generations")
    ap.add_argument("--gen-steps", type=int, default=400_000_000, help="steps of each exploiter generation")
    ap.add_argument("--first-at", type=int, default=0, help="the main's step at which the first generation starts")
    ap.add_argument("--us-start", required=True, help="resume state of the US lineage's first generation, or 'main'")
    ap.add_argument("--ussr-start", required=True, help="the USSR lineage's, or 'main'")
    ap.add_argument("--us-name", required=True, help="run name template with {step} (and optionally {gen})")
    ap.add_argument("--ussr-name", required=True)
    ap.add_argument("--exploiter-args", required=True, help="the exploiters' tools/train.py flags, one string")
    ap.add_argument("--capacity", type=int, default=4, help="league members kept per seat")
    ap.add_argument("--judge-games", type=int, default=1000)
    ap.add_argument("--description", required=True)
    a = ap.parse_args(argv)
    for seat in SEATS:
        name = getattr(a, f"{seat}_name").format(step=10**9, gen=1)
        if not is_run_name(name):
            raise SystemExit(f"--{seat}-name {name!r} is not a run name; tools/train.py would refuse it")
    os.makedirs(a.league_dir, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)
    league = SeatLeague(main_dir=a.main_dir, league_dir=a.league_dir, every=a.every, gen_steps=a.gen_steps,
                        starts={"us": a.us_start, "ussr": a.ussr_start}, names={"us": a.us_name, "ussr": a.ussr_name},
                        description=a.description, ops=real_ops(a), capacity=a.capacity, first_at=a.first_at)
    return league.run()


if __name__ == "__main__":
    raise SystemExit(main())
