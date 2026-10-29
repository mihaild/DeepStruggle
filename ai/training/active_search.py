"""P36's gated active search: the rollout root acting inside a from-scratch run (owner, 2026-10-20).

The search does not join from step 0: a random network's root is noise at four times the cost. Every
`gate_every` steps a 200-game check plays the run's raw snapshot under the root against the same
snapshot raw (`tools/tournament.py`, run as a separate process so training does not wait for it); the
first time the root wins `gate_threshold` or more, the gate opens -- "search is now worth something to
this network", a rule that transfers to any architecture. From then on, at `frac` of the learner's
entry decisions (a headline card choice, or an action round's card play; a step is a search step with
probability `frac` and the root plays all of that step's, searched in one batch), the root's move is played as
ENVIRONMENT -- learner = 0, no policy gradient on it, the forced-openings mechanism -- and the root's
improved policy (R32's form, softmax(logit + sigma(completed Q))) is that row's soft cross-entropy
target. The root searches a lagged snapshot of the learner (against the R29 loop, which searched the
live network), refreshed every `refresh_every` steps until `late_from` and every `refresh_every_late`
after.

`gate_only` runs the checks and records when the gate opened without ever joining: a run launched so
can switch search on at a later resume, the moment its own gate is open, with no relaunch.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import ts_engine as ts

from ai.search.rollout_root import RolloutTeacher, is_entry_decision, rollout_spec_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def child_env() -> Dict[str, str]:
    """The environment of a gate check's process: the trainer's, with the repository and the directory
    the trainer's own engine was imported from first on PYTHONPATH -- a run launched with
    PYTHONPATH=. finds the engine without it being there, and the child would not (2026-10-20: both
    40M checks failed on `No module named 'ts_engine'`)."""
    env = dict(os.environ)
    paths = [str(REPO_ROOT), str(Path(ts.__file__).resolve().parent)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


@dataclass
class GateCheck:
    """One 200-game check: the step it was taken at and, once played, the root's score."""

    step: int
    snapshot: str
    out: str
    proc: Optional[subprocess.Popen] = None
    result: Optional[Dict[str, float]] = None


@dataclass
class ActiveSearchState:
    """What a resume needs: when the gate opened, and the checks taken so far."""

    open_step: Optional[int] = None
    checks: List[Dict[str, float]] = field(default_factory=list)


class ActiveSearch:
    def __init__(self, learner: torch.nn.Module, spec: str, device: Any, frac: float, gate_every: int,
                 gate_games: int, gate_threshold: float, gate_only: bool, refresh_every: int,
                 refresh_every_late: int, late_from: int, seed: int = 97531) -> None:
        """`spec` is a rollout spec tail without its checkpoint (e.g.
        "4:16:4:z2:earlier:own:keyed:aligned:box4"), read by the same parser as every other player."""
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--active-search-frac must be in (0, 1], got {frac}")
        if gate_every <= 0 or gate_games < 2 or gate_games % 2:
            raise ValueError("the gate needs a positive cadence and an even number of games")
        self.spec = spec
        _, self.config, self.label = rollout_spec_config(f"rollout:-:{spec}")
        if self.config.belief or self.config.oracle:
            raise ValueError("active search plays as environment: no belief or oracle worlds")
        self.device = device
        self.frac = float(frac)
        self.gate_every = int(gate_every)
        self.gate_games = int(gate_games)
        self.gate_threshold = float(gate_threshold)
        self.gate_only = bool(gate_only)
        self.refresh_every = int(refresh_every)
        self.refresh_every_late = int(refresh_every_late)
        self.late_from = int(late_from)
        self._rng = np.random.default_rng(seed)
        self.snapshot = copy.deepcopy(learner).to(device).eval()
        for p in self.snapshot.parameters():
            p.requires_grad_(False)
        self.teacher = RolloutTeacher(self.snapshot, self.config, device=device, subsample=1.0,
                                      node_filter="all", seed=seed + 1)
        self.state = ActiveSearchState()
        self.pending: Optional[GateCheck] = None
        self.next_gate_step = self.gate_every       # a fresh run's first check (a random network's root is noise)
        self.next_refresh_step: Optional[int] = None
        #: rows played by the root in the current rollout, and over the run
        self.rows_rollout = 0
        self.rows_total = 0

    # ---- the gate ---------------------------------------------------------------------------------
    @property
    def joined(self) -> bool:
        return self.state.open_step is not None and not self.gate_only

    def restore(self, state: Optional[Dict[str, Any]], steps: int) -> None:
        self.state = (ActiveSearchState(open_step=state.get("open_step"), checks=list(state.get("checks", [])))
                      if state else ActiveSearchState())
        done = [int(c["step"]) for c in self.state.checks]
        last = max(done) if done else int(steps)
        self.next_gate_step = max(self.gate_every, (last // self.gate_every + 1) * self.gate_every)

    def state_dict(self) -> Dict[str, Any]:
        return {"open_step": self.state.open_step, "checks": list(self.state.checks)}

    def gate_due(self, steps: int) -> bool:
        return self.state.open_step is None and self.pending is None and steps >= self.next_gate_step

    def start_check(self, steps: int, model: torch.nn.Module, out_dir: Path) -> None:
        """Save the raw snapshot and play the check in a separate process (`tools/tournament.py`)."""
        gate_dir = out_dir / "gate_checks"
        gate_dir.mkdir(parents=True, exist_ok=True)
        snap = out_dir / f"gate_snapshot_{steps}steps.pt"
        torch.save(model.state_dict(), snap)
        out = gate_dir / f"gate_{steps}steps.json"
        cmd = [sys.executable, str(REPO_ROOT / "tools" / "tournament.py"),
               "--models", f"rollout:{snap}:{self.spec}", str(snap),
               "--games-per-side", str(self.gate_games // 2), "--temperature", "0",
               "--base-seed", str(40_000 + steps // 1_000_000), "--output-json", str(out)]
        log = open(gate_dir / f"gate_{steps}steps.log", "w")
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=child_env())
        self.pending = GateCheck(step=int(steps), snapshot=str(snap), out=str(out), proc=proc)
        self.next_gate_step = (int(steps) // self.gate_every + 1) * self.gate_every
        print(f"[active search] gate check at {steps / 1e6:.0f}M: {self.gate_games} games, root "
              f"({self.label}) against the raw snapshot", flush=True)

    def poll(self, steps: int) -> Optional[Dict[str, float]]:
        """The finished check's record (and opens the gate if it passed), else None."""
        if self.pending is None or self.pending.proc is None or self.pending.proc.poll() is None:
            return None
        check, self.pending = self.pending, None
        if check.proc is not None and check.proc.returncode != 0:
            print(f"[active search] gate check at {check.step / 1e6:.0f}M FAILED (exit "
                  f"{check.proc.returncode}); see {check.out[:-5]}.log -- retried at the next cadence",
                  flush=True)
            return None
        m = json.loads(Path(check.out).read_text())["matchup"]
        n = int(m["total_games"])
        rate = (float(m["a_wins"]) + 0.5 * float(m["draws"])) / max(1, n)
        rec = {"step": float(check.step), "rate": rate, "games": float(n),
               "rate_as_us": float(m["win_rate_a_as_us"]), "rate_as_ussr": float(m["win_rate_a_as_ussr"]),
               "read_at_step": float(steps)}
        self.state.checks.append(rec)
        opened = rate >= self.gate_threshold and self.state.open_step is None
        if opened:
            self.state.open_step = int(check.step)
        verb = ("OPEN" + ("" if self.gate_only else ": search joins the rollouts") if opened else "closed")
        print(f"[active search] gate check at {check.step / 1e6:.0f}M: root {100 * rate:.1f}% against the raw "
              f"snapshot (US {100 * rec['rate_as_us']:.1f} / USSR {100 * rec['rate_as_ussr']:.1f}) -- gate {verb}",
              flush=True)
        return {**rec, "open": float(self.state.open_step is not None)}

    # ---- acting -----------------------------------------------------------------------------------
    def maybe_refresh(self, learner: torch.nn.Module, steps: int) -> bool:
        """Copy the learner into the searched snapshot when its schedule says so (in place, so the
        root's network object stays the same)."""
        if not self.joined:
            return False
        if self.next_refresh_step is not None and steps < self.next_refresh_step:
            return False
        self.snapshot.load_state_dict(learner.state_dict())
        every = self.refresh_every if steps < self.late_from else self.refresh_every_late
        self.next_refresh_step = (int(steps) // every + 1) * every
        return True

    def begin_rollout(self, buffer: Any) -> None:
        self.rows_rollout = 0
        buffer.search_pi.zero_()
        buffer.has_search.zero_()

    def select(self, learner_np: np.ndarray, runner: Any) -> List[int]:
        """The learner's entry decisions the root plays this step. A step is a search step with
        probability `frac`, and at a search step it plays every one of them: each entry decision is
        still played with probability `frac` (the games are at unrelated points, so a step's are a
        random sample of the run's), but they are searched together. The root's cost hardly grows with
        the positions searched at once -- 2.1 s for one, 3.2 s for ten, 3.5 s for forty (A8, k = 4,
        16 worlds) -- so one at a time cost 40x a plain rollout and this ~25x less."""
        if self._rng.random() >= self.frac:
            return []
        envs: List[int] = []
        for e in np.flatnonzero(np.asarray(learner_np, dtype=bool)):
            st = runner.get_state(int(e))
            if not ts.Engine.is_terminal(st) and is_entry_decision(st):
                envs.append(int(e))
        return envs

    def apply(self, actions_t: torch.Tensor, learner_np: np.ndarray, runner: Any, buffer: Any,
              step: int) -> np.ndarray:
        """Plays the root's move at the selected envs (overwriting `actions_t`), files its improved
        policy as the row's search target, and returns the mask of envs it played."""
        played = np.zeros(len(learner_np), dtype=bool)
        if not self.joined:
            return played
        envs = self.select(learner_np, runner)
        if not envs:
            return played
        states = [runner.get_state(e) for e in envs]
        picks, targets = self.teacher.decide(states)
        for e, pick, (acts, probs) in zip(envs, picks, targets):
            if not acts:
                continue
            actions_t[e] = int(pick)
            buffer.search_pi[step, e].zero_()
            buffer.search_pi[step, e, torch.tensor(acts, device=buffer.search_pi.device)] = torch.as_tensor(
                probs, dtype=buffer.search_pi.dtype, device=buffer.search_pi.device)
            buffer.has_search[step, e] = 1.0
            played[e] = True
        n = int(played.sum())
        self.rows_rollout += n
        self.rows_total += n
        return played


def gate_metrics(rec: Dict[str, float]) -> Dict[str, float]:
    """A gate check's record as training_metrics.jsonl keys."""
    return {f"active_search_gate_{k}": v for k, v in rec.items()}

