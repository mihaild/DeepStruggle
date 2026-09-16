#!/usr/bin/env python3
"""Compare two training runs at matched step counts, and say what else differs between them.

The config diff comes first, and it is the point of the tool. E3-22-28's first attempt was
compared against E3-20-28 by hand, window by window, and read as "no detectable difference" --
but the two runs also differed in their snapshot cadence, and snapshots feed the self-play
opponent pool, so at 45M steps the arm had 2 pool opponents against the baseline's 9 while a
third of its games were played against that pool. The critic comparison was confounded and the
null meant nothing. Nobody had to be careless for that to happen: the difference was in a flag
nobody thought to diff, and the arm's own metadata did not even record its snapshot cadence.

So: diff the configuration, then compare the metrics, and never present the second without the
first.

Two further things this encodes, both learned the same day:

* **Average over windows, not iterations.** A single iteration's `critic_auc` swings by more than
  the effect being looked for. Reading one point against a remembered baseline point is how the
  first attempt was reported as "comfortably ahead" when ten windows put it at -0.008 +/- 0.016.
* **Watch the base rate.** `critic_auc` is a discrimination score over whatever mix of won and
  lost positions the window happened to contain. When two windows have different base rates they
  are not the same problem, and part of any AUC gap is the mix rather than the critic. Base rates
  are printed beside every row and a material divergence is called out.

    tools/compare_runs.py <baseline-run-dir> <arm-run-dir> [--window-steps 5000000]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Metrics compared by default. All are "higher is better".
DEFAULT_METRICS: Tuple[str, ...] = ("critic_auc", "critic_brier_skill")

#: Config keys that change what the run *trains against*, not merely what it reports. A difference
#: in any of these makes the comparison multi-factor, which is the failure this tool exists for.
TRAINING_AFFECTING: Tuple[str, ...] = (
    "seed", "num_envs", "train_steps", "reward_scheme", "eta", "vf_coef", "ent_coef",
    "ref_update_freq", "opponent_frac", "opponent_self_pool", "opponent_pool_size",
    "start_pool_frac", "decisiveness_turns", "arch", "obs_layout", "identity_dim",
    "per_entity_heads", "graph_layers", "value_dist_coef", "per_player_gae",
    "same_perspective_bootstrap", "inject_dataset", "inject_every", "inject_weight",
    # Not obviously a training knob, which is exactly why it is listed: the opponent pool is fed
    # from snapshots, so the snapshot cadence sets how fast and how diverse the pool grows.
    "snapshot_every_steps",
)

#: Recorded for provenance; a difference here is worth seeing but is not itself a confound.
INFORMATIONAL: Tuple[str, ...] = ("base_commit", "git_dirty", "run_name", "run_id")


def load_rows(run_dir: str) -> List[Dict[str, Any]]:
    """Every JSON line of a run's training_metrics.jsonl, in step order where steps are known."""
    path = os.path.join(run_dir, "training_metrics.jsonl")
    rows: List[Dict[str, Any]] = []
    try:
        handle = open(path, encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a run killed mid-write leaves a partial last line
    rows.sort(key=lambda r: r.get("total_steps") or -1)
    return rows


def load_metadata(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "metadata.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def window_mean(rows: Sequence[Dict[str, Any]], key: str, lo: int, hi: int) -> Optional[float]:
    """Mean of `key` over rows whose total_steps falls in [lo, hi). None if the window is empty."""
    vals = [r[key] for r in rows
            if isinstance(r.get("total_steps"), int)
            and lo <= r["total_steps"] < hi
            and isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def mean_ci(xs: Sequence[float]) -> Tuple[float, float, float]:
    """Mean and a normal-approximation 95% CI half-width, plus the SD. Zero-width for n < 2."""
    if not xs:
        return 0.0, 0.0, 0.0
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0, 0.0
    sd = statistics.stdev(xs)
    return m, 1.96 * sd / len(xs) ** 0.5, sd


def config_diff(base_meta: Dict[str, Any], arm_meta: Dict[str, Any]
                ) -> Tuple[List[str], List[str], List[str]]:
    """(recorded differences, unverifiable keys, informational differences).

    A key missing from one run's metadata is NOT a difference and NOT a match -- it is a thing
    that cannot be compared, and it gets its own list. Reporting it as a difference invents a
    confound (an older run that predates a flag records None where a newer run records False);
    reporting it as a match hides a real one. `snapshot_every_steps` was unrecorded by BOTH runs
    in the case that motivated this tool, which is why the observed checks below exist: what a
    run did is evidence, what its metadata claims is only a claim.
    """
    recorded: List[str] = []
    unverifiable: List[str] = []
    info: List[str] = []
    for key in TRAINING_AFFECTING:
        in_b, in_a = key in base_meta, key in arm_meta
        b, a = base_meta.get(key), arm_meta.get(key)
        if in_b and in_a:
            if b != a:
                note = ("   <- also sets how fast the self-play opponent pool grows"
                        if key == "snapshot_every_steps" else "")
                recorded.append(f"  {key}: baseline {b!r}  vs  arm {a!r}{note}")
        elif in_b or in_a:
            which = "arm" if in_b else "baseline"
            val = b if in_b else a
            unverifiable.append(
                f"  {key}: not recorded by the {which}; the other says {val!r}")
        else:
            if key == "snapshot_every_steps":
                unverifiable.append(
                    f"  {key}: recorded by NEITHER run -- compare the observed pool growth below")
    for key in INFORMATIONAL:
        b, a = base_meta.get(key), arm_meta.get(key)
        if b != a:
            info.append(f"  {key}: baseline {b!r}  vs  arm {a!r}")
    return recorded, unverifiable, info


def observed_confounds(base_rows: Sequence[Dict[str, Any]], arm_rows: Sequence[Dict[str, Any]],
                       horizon: int) -> List[str]:
    """Differences derived from what the runs DID, so unrecorded flags cannot hide them.

    The opponent pool is the one that matters here: it is fed from snapshots, and with a non-zero
    --opponent-frac a share of every arm's games is played against it. Two runs whose pools grew
    at different rates trained against different opponents, whatever their metadata says.
    """
    found: List[str] = []
    b_trace = [(s, n) for s, n in pool_trace(base_rows) if s <= horizon]
    a_trace = [(s, n) for s, n in pool_trace(arm_rows) if s <= horizon]
    b_final = b_trace[-1][1] if b_trace else 0.0
    a_final = a_trace[-1][1] if a_trace else 0.0
    if b_final and a_final and (max(b_final, a_final) >= 2 * min(b_final, a_final)
                                or abs(b_final - a_final) >= 3):
        found.append(
            f"  opponent pool at {horizon / 1e6:.1f}M: baseline {b_final:g} models, "
            f"arm {a_final:g} -- the runs trained against different opponent distributions")
    return found


def pool_trace(rows: Sequence[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Every point at which opp_pool_size changed, as (steps, size)."""
    out: List[Tuple[int, float]] = []
    seen: Optional[float] = None
    for r in rows:
        size = r.get("opp_pool_size")
        steps = r.get("total_steps")
        if not isinstance(size, (int, float)) or not isinstance(steps, int):
            continue
        if size != seen:
            out.append((steps, float(size)))
            seen = size
    return out


def compare(base_dir: str, arm_dir: str, window_steps: int,
            metrics: Sequence[str]) -> int:
    base_rows, arm_rows = load_rows(base_dir), load_rows(arm_dir)
    base_meta, arm_meta = load_metadata(base_dir), load_metadata(arm_dir)

    print("=" * 96)
    print(f"baseline : {base_dir}")
    print(f"arm      : {arm_dir}")
    print("=" * 96)

    recorded, unverifiable, info = config_diff(base_meta, arm_meta)

    arm_steps = max((r["total_steps"] for r in arm_rows
                     if isinstance(r.get("total_steps"), int)), default=0)
    base_steps = max((r["total_steps"] for r in base_rows
                      if isinstance(r.get("total_steps"), int)), default=0)
    horizon = min(arm_steps, base_steps)
    observed = observed_confounds(base_rows, arm_rows, horizon)

    print("\nCONFIGURATION")
    if recorded:
        print("  Differences recorded in metadata that affect what the run trains against:")
        for line in recorded:
            print(line)
    else:
        print("  No recorded training-affecting differences.")
    if unverifiable:
        print("\n  Cannot be compared -- not recorded by one or both runs:")
        for line in unverifiable:
            print(line)
    if info:
        print("\n  Informational:")
        for line in info:
            print(line)

    if observed:
        print("\nOBSERVED DIFFERENCES  (from the runs' own logs, not their metadata)")
        for line in observed:
            print(line)

    factors = len(recorded) + len(observed)
    if factors > 1:
        print("\n  *** MORE THAN ONE FACTOR DIFFERS. No metric difference below can be attributed")
        print("      to any one of them. This is the confound that voided E3-22-28's first")
        print("      attempt: it differed from its baseline in the estimator under test AND in")
        print("      its opponent pool, and the pool difference was in a flag neither run")
        print("      recorded. ***")
    elif observed:
        print("\n  *** The single differing factor is an OBSERVED one, not the flag under test.")
        print("      Whatever is measured below is a property of that, not of the intended")
        print("      change. ***")
    print(f"\nComparing to {horizon:,} steps "
          f"(baseline reached {base_steps:,}, arm {arm_steps:,})")
    if horizon < window_steps:
        print("  Not enough overlap for even one window.")
        return 1

    for metric in metrics:
        print(f"\n{metric.upper()}  (windowed mean over {window_steps:,}-step windows)")
        hdr = (f"  {'window':>16} | {'baseline':>9} {'arm':>9} {'delta':>9} | "
               f"{'base rate':>19}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        deltas: List[float] = []
        br_gaps: List[float] = []
        for lo in range(0, horizon, window_steps):
            hi = lo + window_steps
            b = window_mean(base_rows, metric, lo, hi)
            a = window_mean(arm_rows, metric, lo, hi)
            if b is None or a is None:
                continue
            deltas.append(a - b)
            bb = window_mean(base_rows, "critic_base_rate", lo, hi)
            ba = window_mean(arm_rows, "critic_base_rate", lo, hi)
            br = ""
            if bb is not None and ba is not None:
                br_gaps.append(abs(ba - bb))
                br = f"{bb:.3f} vs {ba:.3f}"
                if abs(ba - bb) > 0.08:
                    br += "  !"
            label = f"{lo // 1_000_000}-{hi // 1_000_000}M"
            print(f"  {label:>16} | {b:>9.4f} {a:>9.4f} {a - b:>+9.4f} | {br:>19}")

        if not deltas:
            print("  no overlapping windows carried this metric")
            continue
        m, half, sd = mean_ci(deltas)
        print(f"\n    mean delta {m:+.4f}   SD {sd:.4f}   "
              f"95% CI [{m - half:+.4f}, {m + half:+.4f}]   n={len(deltas)} windows")
        verdict = ("indistinguishable from zero" if m - half <= 0.0 <= m + half
                   else ("arm is better" if m > 0 else "arm is worse"))
        print(f"    -> {verdict}")
        if br_gaps and max(br_gaps) > 0.08:
            print(f"    NOTE: base rates diverge by up to {max(br_gaps):.3f} in some windows "
                  f"(marked !).")
            print("    Those windows are not the same discrimination problem; part of the gap is")
            print("    the win/loss mix, not the critic.")

    print("\nOPPONENT POOL GROWTH  (the evidence behind the observed check above)")
    for tag, rows in (("baseline", base_rows), ("arm", arm_rows)):
        trace = [(s, n) for s, n in pool_trace(rows) if s <= horizon]
        if not trace:
            print(f"  {tag:>8}: no opp_pool_size logged")
            continue
        shown = ", ".join(f"{n:g}@{s / 1e6:.1f}M" for s, n in trace[:12])
        print(f"  {tag:>8}: {shown}{' ...' if len(trace) > 12 else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    ap.add_argument("baseline_dir", help="the control run's checkpoint directory")
    ap.add_argument("arm_dir", help="the experimental run's checkpoint directory")
    ap.add_argument("--window-steps", type=int, default=5_000_000,
                    help="Average each metric over windows this wide (default 5,000,000). "
                         "Single iterations are noisier than the effects usually looked for.")
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                    help=f"Metric keys to compare (default: {' '.join(DEFAULT_METRICS)})")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return compare(args.baseline_dir, args.arm_dir, args.window_steps, args.metrics)


if __name__ == "__main__":
    sys.exit(main())
