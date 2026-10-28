"""Online distillation: a fresh network of any shape trained to the control's policy and value on the control's own
self-play (P39 arm 2, research/plans/P39_deep_trunk_optimisation.md).

The plan asks for the A8 control distilled into a deep network before RL, so the deep network starts at the
control's level with its plasticity intact: "the deep trunk's problem is a slow start, not a ceiling it has
reached". 50M positions stored offline would be ~750 GB (3,824 floats each), so the positions are not stored: the
teacher plays its own games in the vectorized environment (sampling at `play_temperature`, 0.25 by default, so the
student sees the teacher's distribution of positions with some breadth) and the student trains on each rollout's
positions as they come, to the teacher's full policy at temperature 1 (the plan's caution: a greedy target is sharper
than an RL policy) by KL, and to its value by MSE.

A held-out set from games on another seed measures the fit: KL to the teacher, agreement with the teacher's argmax
over the legal moves (the plan's bar: >= 95% before any RL) and the value's MSE. Training stops when the held-out KL
has not improved for `patience` evaluations or at the position budget; the best student is `best.pt`, a plain state
dict that `--warmup-checkpoint` loads into the RL arm.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.training.nash_pg import make_optimizer
from bindings.ts_env import TsVectorizedEnv, model_obs_features
from tools.lib.player_agent import NeuralAgent


@dataclass
class DistillConfig:
    teacher: str
    positions: int = 50_000_000              # the position budget
    num_envs: int = 512
    rollout_len: int = 32                    # positions per env between student updates
    batch: int = 4096
    epochs: int = 1                          # passes over each rollout's positions
    lr: float = 3e-4
    weight_decay: float = 1e-4
    play_temperature: float = 0.25
    value_coef: float = 0.5
    holdout: int = 32_768
    holdout_stride: int = 4                  # one position kept per env every this many steps
    eval_every: int = 2_000_000
    patience: int = 5
    min_improvement: float = 0.005           # relative held-out KL improvement that counts
    seed_env: int = 44
    device: str = "cuda"


def _teacher_step(teacher: nn.Module, obs: np.ndarray, masks: np.ndarray, dev: torch.device,
                  temperature: float, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                      torch.Tensor, torch.Tensor, np.ndarray]:
    o = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(masks)).to(dev).bool()
    with torch.no_grad():
        logits, v_win, v_vp = teacher(o, m)
        logits = logits.float()
        play = torch.softmax(logits / temperature, dim=-1)
        actions = torch.multinomial(play, 1, generator=gen).squeeze(-1)
    return o, m, logits, v_win.float().view(-1), v_vp.float().view(-1), actions.cpu().numpy()


def _losses(student: nn.Module, o: torch.Tensor, m: torch.Tensor, t_logits: torch.Tensor, t_win: torch.Tensor,
            t_vp: torch.Tensor, value_coef: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    s_logits, s_win, s_vp = student(o, m)
    t_logp = torch.log_softmax(t_logits, dim=-1)
    s_logp = torch.log_softmax(s_logits.float(), dim=-1)
    p = t_logp.exp()
    kl = (p * (t_logp - s_logp)).masked_fill(~m, 0.0).sum(-1)
    v = F.mse_loss(s_win.float().view(-1), t_win) + F.mse_loss(s_vp.float().view(-1) / 20.0, t_vp / 20.0)
    agree = (s_logits.masked_fill(~m, -1e9).argmax(-1) == t_logits.argmax(-1)).float()
    return kl.mean() + value_coef * v, {"kl": kl.mean().detach(), "value_mse": v.detach(), "agree": agree.mean()}


def _collect_holdout(teacher: nn.Module, cfg: DistillConfig, features: int, dev: torch.device
                     ) -> Dict[str, torch.Tensor]:
    """Teacher positions from games on another seed, one per env every `holdout_stride` steps (so the set spans
    whole games), kept on the CPU: observations in half precision, the masks, the teacher's logits and values."""
    seed = cfg.seed_env + 7919
    env = TsVectorizedEnv(num_envs=min(cfg.num_envs, 256), base_seed=seed, game_seed=seed)
    env.set_obs_features(features, features)
    obs, masks, _ = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    keep: Dict[str, List[torch.Tensor]] = {"obs": [], "mask": [], "logits": [], "win": [], "vp": []}
    n, step = 0, 0
    while n < cfg.holdout:
        o, m, lg, vw, vv, acts = _teacher_step(teacher, obs, masks, dev, cfg.play_temperature, gen)
        if step % cfg.holdout_stride == 0:
            for k, t in (("obs", o.half()), ("mask", m), ("logits", lg), ("win", vw), ("vp", vv)):
                keep[k].append(t.cpu())
            n += o.shape[0]
        obs, masks, _r, _d, _i = env.step(acts)
        step += 1
    return {k: torch.cat(v)[:cfg.holdout] for k, v in keep.items()}


@torch.no_grad()
def _evaluate(student: nn.Module, hold: Dict[str, torch.Tensor], dev: torch.device, value_coef: float,
              chunk: int = 4096) -> Dict[str, float]:
    was = student.training
    student.eval()
    tot = {"kl": 0.0, "value_mse": 0.0, "agree": 0.0}
    n = int(hold["obs"].shape[0])
    for i in range(0, n, chunk):
        sl = slice(i, min(n, i + chunk))
        _, parts = _losses(student, hold["obs"][sl].to(dev).float(), hold["mask"][sl].to(dev),
                           hold["logits"][sl].to(dev), hold["win"][sl].to(dev), hold["vp"][sl].to(dev), value_coef)
        w = (sl.stop - sl.start) / n
        for k in tot:
            tot[k] += float(parts[k]) * w
    student.train(was)
    return tot


def run_online_distillation(student: nn.Module, cfg: DistillConfig, run_dir: str, meta: Dict[str, Any],
                            lr_mults: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Train `student` (fresh weights, any LadderNet shape) to the teacher on the teacher's self-play; returns the
    best held-out evaluation. Writes metadata.json, distill_metrics.jsonl, best.pt and snapshot_<positions>pos.pt."""
    dev = torch.device(cfg.device)
    teacher = NeuralAgent.from_checkpoint(cfg.teacher, device=dev).model
    teacher.eval()
    features = model_obs_features(student)
    if model_obs_features(teacher) != features:
        raise ValueError(f"the teacher reads observation features {model_obs_features(teacher):#x}, the student "
                         f"{features:#x}")
    student.to(dev).train()
    opt = make_optimizer(student, cfg.lr, cfg.weight_decay, lr_mults=lr_mults)
    os.makedirs(run_dir, exist_ok=True)
    meta = dict(meta) | {"mode": "online-distill", "config": asdict(cfg),
                         "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "parameters": sum(p.numel() for p in student.parameters())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log = open(os.path.join(run_dir, "distill_metrics.jsonl"), "a", encoding="utf-8")

    t0 = time.time()
    hold = _collect_holdout(teacher, cfg, features, dev)
    print(f"[distill] held-out set: {hold['obs'].shape[0]:,} teacher positions ({time.time() - t0:.0f}s)", flush=True)
    env = TsVectorizedEnv(num_envs=cfg.num_envs, base_seed=cfg.seed_env, game_seed=cfg.seed_env)
    env.set_obs_features(features, features)
    obs, masks, _ = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(cfg.seed_env)
    torch.manual_seed(cfg.seed_env)

    best: Dict[str, Any] = {"kl": float("inf")}
    stale, seen, next_eval = 0, 0, cfg.eval_every
    first = _evaluate(student, hold, dev, cfg.value_coef) | {"positions": 0, "seconds": 0.0}
    log.write(json.dumps(first) + "\n")
    log.flush()
    print(f"[distill] start: held-out KL {first['kl']:.4f}, agreement {first['agree']:.1%}", flush=True)
    while seen < cfg.positions:
        roll: Dict[str, List[torch.Tensor]] = {"obs": [], "mask": [], "logits": [], "win": [], "vp": []}
        for _ in range(cfg.rollout_len):
            o, m, lg, vw, vv, acts = _teacher_step(teacher, obs, masks, dev, cfg.play_temperature, gen)
            for k, t in (("obs", o), ("mask", m), ("logits", lg), ("win", vw), ("vp", vv)):
                roll[k].append(t)
            obs, masks, _r, _d, _i = env.step(acts)
        data = {k: torch.cat(v) for k, v in roll.items()}
        n = int(data["obs"].shape[0])
        train: Dict[str, float] = {"kl": 0.0, "value_mse": 0.0, "agree": 0.0}
        batches = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch]
                loss, parts = _losses(student, data["obs"][idx], data["mask"][idx], data["logits"][idx],
                                      data["win"][idx], data["vp"][idx], cfg.value_coef)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                for k in train:
                    train[k] += float(parts[k])
                batches += 1
        seen += n
        if seen >= next_eval or seen >= cfg.positions:
            next_eval += cfg.eval_every
            ev = _evaluate(student, hold, dev, cfg.value_coef)
            row = ev | {"positions": seen, "seconds": round(time.time() - t0, 1),
                        **{f"train_{k}": v / max(1, batches) for k, v in train.items()}}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[distill] {seen:,} positions: held-out KL {ev['kl']:.4f}, agreement {ev['agree']:.1%}, value MSE "
                  f"{ev['value_mse']:.4f} ({row['seconds']:.0f}s)", flush=True)
            torch.save(student.state_dict(), os.path.join(run_dir, f"snapshot_{seen}pos.pt"))
            if ev["kl"] < best["kl"] * (1.0 - cfg.min_improvement):
                best, stale = row, 0
                torch.save(student.state_dict(), os.path.join(run_dir, "best.pt"))
            else:
                stale += 1
                if stale >= cfg.patience:
                    print(f"[distill] held-out KL flat for {stale} evaluations; stopping", flush=True)
                    break
    log.close()
    (pathlib.Path(run_dir) / "distill_result.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"[distill] best: {best.get('positions', 0):,} positions, held-out KL {best['kl']:.4f}, agreement "
          f"{best.get('agree', 0.0):.1%} -> {os.path.join(run_dir, 'best.pt')}", flush=True)
    return best
