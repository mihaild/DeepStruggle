#!/usr/bin/env python3
"""Measure training throughput against environment count, to find where the bottleneck is.

Answers two questions with one measurement:

* **Is there headroom on this GPU?** If steps/s keeps climbing with `--num-envs`, the GPU was
  batch-starved and the current setting is leaving speed on the table. If it flattens, the device
  is saturated and only a faster card helps.
* **Which card is cheapest per step?** Run the same sweep on a rented candidate and divide its
  hourly price by its steps/s. Spec sheets do not predict this: the workload is a small network
  (a 512-wide trunk over 3,824 inputs) plus a CPU-side simulator, which is exactly the shape that
  fails to scale with headline FLOPS.

Reports the CPU cost too, because the engine is a C++ simulator that links no CUDA and the host
can be the limit instead -- on a rented marketplace slice, often is.

**Measures a full training iteration, not just rollout collection.** The first version called
`collect_rollouts()` alone and reported 57,193 steps/s at num_envs=512 on a 4090, against the
13,169 steps/s that arm actually trained at -- a 4.3x overstatement, because the four inner SGD
epochs are roughly three quarters of the wall clock. A rollout-only number is the wrong basis
both for "can we go faster" and for comparing cards, since a GPU can be good at batched inference
and bad at the backward pass.

The model is discarded afterwards; this trains nothing that is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np


def measure(num_envs: int, buffer_size: int, iterations: int, device: str) -> Dict[str, Any]:
    import torch

    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from ai.training.nash_pg import NashPGTrainer
    from bindings.ts_env import TsVectorizedEnv

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=1234)
    trainer = NashPGTrainer(active_net=create_coldwar_net_v2(), env=env, num_envs=num_envs,
                            buffer_size=buffer_size, device=device)

    trainer.train_iteration()  # warm up: allocator, autotune, CUDA context
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    for _ in range(iterations):
        trainer.train_iteration()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0

    steps = num_envs * buffer_size * iterations
    out: Dict[str, Any] = {
        "num_envs": num_envs,
        "steps": steps,
        "wall_seconds": wall,
        "steps_per_sec": steps / wall,
        # >1 means several cores busy at once; near the core count means CPU-bound.
        "cpu_cores_busy": cpu / wall,
    }
    if device == "cuda" and torch.cuda.is_available():
        out["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del trainer, env
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--buffer-size", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--price-per-hour", type=float, default=None,
                    help="If given, also report $ per 80M steps at each setting")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        import torch
        name = (torch.cuda.get_device_name(0)
                if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    except Exception:
        name = "unknown"
    print(f"device: {name}   host cores: {os.cpu_count()}")
    print(f"{'num_envs':>9}{'steps/s':>11}{'cores busy':>12}{'GPU GB':>9}"
          + (f"{'$/80M':>9}" if args.price_per_hour else ""))
    print("-" * (41 + (9 if args.price_per_hour else 0)))

    rows: List[Dict[str, Any]] = []
    for n in args.num_envs:
        try:
            r = measure(n, args.buffer_size, args.iterations, args.device)
        except RuntimeError as e:
            # Out of memory at this setting is a result, not a crash: it is the point at which
            # the card stops being able to hold the batch.
            print(f"{n:>9}   failed: {str(e).splitlines()[0][:48]}")
            continue
        rows.append(r)
        line = (f"{r['num_envs']:>9}{r['steps_per_sec']:>11.0f}"
                f"{r['cpu_cores_busy']:>12.1f}{r.get('gpu_mem_gb', float('nan')):>9.2f}")
        if args.price_per_hour:
            line += f"{args.price_per_hour * 80e6 / r['steps_per_sec'] / 3600:>9.2f}"
        print(line)

    if len(rows) > 1:
        best = max(rows, key=lambda r: r["steps_per_sec"])
        base = rows[0]
        print(f"\nfastest at num_envs={best['num_envs']}: {best['steps_per_sec']:.0f} steps/s, "
              f"{best['steps_per_sec'] / base['steps_per_sec']:.2f}x the smallest setting")
        gain = best["steps_per_sec"] / rows[-1]["steps_per_sec"]
        if abs(gain - 1.0) < 0.05 and best["num_envs"] != rows[-1]["num_envs"]:
            print("throughput has flattened -- the device is saturated, more envs will not help")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"device": name, "rows": rows}, f, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
