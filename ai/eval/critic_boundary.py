"""Does the critic agree with itself across a hand-over between the two sides?

At a hand-over -- the last decision of one side, s_t, followed by the first decision of the other,
s_{t+1} -- each side's critic is read from its own perspective: v_A(s_t) and v_B(s_{t+1}). They see
different things (each knows its own hand), so they need not agree position by position. But each
predicts its own return, so averaged over positions

    E[v_A(s_t) + v_B(s_{t+1})] = E[G_A] + E[G_B],

which is 0 whenever the targets are zero-sum -- no nesting of information is needed for the
average, and it holds within any class of positions both sides can see (turn, phase), though not
within a class defined by one side's hand. The recipe's targets are zero-sum (they were not in the
turn of a blunder under the `blunder_aware` scheme and its window, removed on 2026-11-22: the
blunderer's target was -1 and the winner's pinned to its own value), so a calibrated critic should
show a mean of ~0.

This plays self-play games, records the mover's value at every decision, and reports the mean of
v_prev + v_next over hand-overs (standard errors clustered by game), split by direction, turn and
phase, and decomposed into each side's bias against the realised result.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.loading import load_for_eval
from ai.eval.playout import Sampler, check_finished, make_runner, model_view


@dataclass
class Decision:
    side: int          # +1 US, -1 USSR
    value: float       # v_win from the mover's perspective
    turn: int
    phase: str
    defcon: int = 0


@dataclass
class Game:
    decisions: List[Decision] = field(default_factory=list)
    us_util: float = 0.0
    blunder: str = ""              # "", "defcon_self" or "held_scoring"


def _blunder_kind(state: ts.GameState) -> str:
    """How the game ended, as far as the blunder window is concerned (ai/rewards/reward_calculator)."""
    if int(state.defcon) <= 1:
        return "defcon"            # provoked or not -- recorded coarsely; see the report
    if ts.Engine.is_held_scoring_game_over(state):
        return "held_scoring"
    return ""


def play(model: Any, device: Any, num_envs: int = 512, base_seed: int = 4242, temperature: float = 1.0,
         max_iters: int = 4000) -> List[Game]:
    """Self-play games with `model` on both sides, every decision's mover value recorded. The model
    reads its own observation feature set; E4 action view (load with merged_ok=False)."""
    runner = make_runner(num_envs, base_seed, model_view(model, False))
    sampler = Sampler(model, temperature, base_seed)
    games = [Game() for _ in range(num_envs)]
    done = np.zeros(num_envs, dtype=bool)
    model.eval()
    for _ in range(max_iters):
        term = np.asarray(runner.get_terminals(), dtype=bool)
        newly = term & ~done
        for i in np.flatnonzero(newly):
            st = runner.get_state(int(i))
            games[i].blunder = _blunder_kind(st)
        done |= term
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        logits, v_win, _ = sampler.forward(obs, masks, term)
        values = v_win.float().squeeze(-1).cpu().numpy()
        picks = sampler.pick(logits.float())
        for i in np.flatnonzero(~done):
            side = int(players[i])
            if side == 0:
                continue
            st = runner.get_state(int(i))
            games[i].decisions.append(Decision(side, float(values[i]), int(st.turn), str(st.current_phase).split(".")[-1],
                                               int(st.defcon)))
        runner.step_flat_all([int(a) for a in picks], auto_advance=True)
    check_finished(done, max_iters, "critic_boundary.play")
    utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
    for i in range(num_envs):
        games[i].us_util = float(utils[i])
    return games


def _clustered(values: Sequence[float], clusters: Sequence[int]) -> Tuple[float, float, int]:
    """Mean, its standard error clustered by game, and the count."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    c = np.asarray(clusters)
    m = float(v.mean())
    sums = np.array([float((v[c == k] - m).sum()) for k in np.unique(c)])
    se = float(np.sqrt((sums ** 2).sum()) / v.size)
    return m, se, int(v.size)


def handovers(games: Sequence[Game]) -> List[Dict[str, Any]]:
    """One row per hand-over: the two values, their sum, and context."""
    rows: List[Dict[str, Any]] = []
    for gi, g in enumerate(games):
        d = g.decisions
        for a, b in zip(d, d[1:]):
            if a.side == b.side:
                continue
            rows.append({"game": gi, "dir": "US->USSR" if a.side == 1 else "USSR->US", "sum": a.value + b.value,
                         "turn": b.turn, "phase": b.phase, "same_turn": a.turn == b.turn,
                         "blunder": g.blunder,
                         # each side's error against what happened, in its own frame
                         "err_a": a.value - g.us_util * a.side, "err_b": b.value - g.us_util * b.side})
    return rows


def report(games: Sequence[Game]) -> str:
    rows = handovers(games)
    out: List[str] = []

    def line(label: str, sel: List[Dict[str, Any]], key: str = "sum") -> None:
        m, se, n = _clustered([r[key] for r in sel], [r["game"] for r in sel])
        out.append(f"  {label:38s} {m:+.4f} ± {se:.4f}   (n={n:,})")

    ends = [g.blunder for g in games]
    out.append(f"{len(games):,} games; endings: DEFCON 1 {ends.count('defcon')/len(games):.1%}, "
               f"held scoring {ends.count('held_scoring')/len(games):.1%}; US wins {np.mean([g.us_util > 0 for g in games]):.1%}")
    out.append("\nMean of v_prev + v_next over hand-overs (0 if calibrated to zero-sum targets):")
    line("all hand-overs", rows)
    for dname in ("US->USSR", "USSR->US"):
        line(dname, [r for r in rows if r["dir"] == dname])
    line("within a turn", [r for r in rows if r["same_turn"]])
    line("across a turn boundary", [r for r in rows if not r["same_turn"]])
    for ph in sorted({r["phase"] for r in rows}):
        line(f"phase {ph}", [r for r in rows if r["phase"] == ph])
    line("games without a blunder-type ending", [r for r in rows if not r["blunder"]])
    line("games ending in DEFCON 1", [r for r in rows if r["blunder"] == "defcon"])
    line("games ending on held scoring", [r for r in rows if r["blunder"] == "held_scoring"])
    # The same mover's own critic along its own consecutive decisions should be a martingale too:
    # E[v(s_{t+1}) - v(s_t)] = 0. A drift here says where in a run of decisions the bias sits.
    same: List[float] = []
    same_g: List[int] = []
    first_after: List[float] = []
    first_g: List[int] = []
    for gi, g in enumerate(games):
        d = g.decisions
        for k, (a, b) in enumerate(zip(d, d[1:])):
            if a.side == b.side:
                same.append(b.value - a.value)
                same_g.append(gi)
        # within each run of one mover: last value minus first value
        start = 0
        for k in range(1, len(d) + 1):
            if k == len(d) or d[k].side != d[start].side:
                if k - 1 > start:
                    first_after.append(d[k - 1].value - d[start].value)
                    first_g.append(gi)
                start = k
    m, se, n = _clustered(same, same_g)
    out.append(f"\nSame mover, consecutive decisions, v_next - v_prev:  {m:+.4f} ± {se:.4f}   (n={n:,})")
    m, se, n = _clustered(first_after, first_g)
    out.append(f"Same mover, last minus first decision of a run:     {m:+.4f} ± {se:.4f}   (n={n:,})")
    out.append("\nBy turn (all hand-overs):")
    for t in sorted({r["turn"] for r in rows}):
        line(f"turn {t}", [r for r in rows if r["turn"] == t])
    out.append("\nWhere a non-zero sum comes from -- each side's bias against the realised result:")
    for dname in ("US->USSR", "USSR->US"):
        sel = [r for r in rows if r["dir"] == dname]
        line(f"{dname}: first mover's v - result", sel, "err_a")
        line(f"{dname}: second mover's v - result", sel, "err_b")
    return "\n".join(out)


def _auc(scores: Any, labels: Any) -> float:
    """Area under the ROC curve by the rank-sum (Mann-Whitney) formula, ties averaged."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels) > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _log_loss(p: Any, y: Any) -> Any:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=np.float64)
    return -(yy * np.log(q) + (1 - yy) * np.log(1 - q))


def _decision_table(games: Sequence[Game]) -> Dict[str, np.ndarray]:
    """Every decision of every decided game: predicted p, outcome, game id, side, turn, DEFCON."""
    cols: Dict[str, List[float]] = {"p": [], "y": [], "g": [], "side": [], "turn": [], "defcon": []}
    for gi, g in enumerate(games):
        if g.us_util == 0.0:
            continue
        for d in g.decisions:
            cols["p"].append((d.value + 1.0) / 2.0)
            cols["y"].append(1.0 if g.us_util * d.side > 0 else 0.0)
            cols["g"].append(gi)
            cols["side"].append(d.side)
            cols["turn"].append(d.turn)
            cols["defcon"].append(d.defcon)
    return {k: np.asarray(v) for k, v in cols.items()}


def compare_report(by_temperature: Dict[float, Sequence[Game]]) -> str:
    """P33b: the same critic on games played at different temperatures -- the signed error (bias) and
    the log loss per slice, each game-clustered, and the difference of the first temperature from the
    last with its SE (independent game sets). The hypothesis it tests: the critic, trained on the returns
    of sampled play, is biased on greedy play, most at DEFCON 2 and late."""
    temps = list(by_temperature)
    tabs = {t: _decision_table(g) for t, g in by_temperature.items()}
    slices: List[Tuple[str, Any]] = [
        ("all", lambda t: np.ones(len(t["p"]), dtype=bool)),
        ("as US", lambda t: t["side"] == 1), ("as USSR", lambda t: t["side"] == -1),
        ("turns 1-3", lambda t: t["turn"] <= 3), ("turns 4-7", lambda t: (t["turn"] >= 4) & (t["turn"] <= 7)),
        ("turns 8-10", lambda t: t["turn"] >= 8),
    ] + [(f"DEFCON {dc}", (lambda dc: lambda t: t["defcon"] == dc)(dc)) for dc in (2, 3, 4, 5)] + [
        ("DEFCON 2, turns 8-10", lambda t: (t["defcon"] == 2) & (t["turn"] >= 8)),
    ]
    head = "  ".join(f"T={t:g}: bias, log loss".ljust(34) for t in temps)
    out = [f"{'slice':22s}  {head}  T={temps[0]:g} - T={temps[-1]:g}: |bias|, log loss"]
    for name, sel in slices:
        cells: List[str] = []
        stats = {}
        for t in temps:
            tab = tabs[t]
            m = sel(tab)
            if not m.any():
                cells.append("–".ljust(34))
                continue
            b, bse, n = _clustered(list(tab["p"][m] - tab["y"][m]), list(tab["g"][m]))
            ll, llse, _ = _clustered(list(_log_loss(tab["p"][m], tab["y"][m])), list(tab["g"][m]))
            stats[t] = (b, bse, ll, llse)
            cells.append(f"{b:+.4f}±{bse:.4f}  {ll:.4f}±{llse:.4f}  n={n:,}".ljust(34))
        diff = ""
        if temps[0] in stats and temps[-1] in stats:
            b0, s0, l0, ls0 = stats[temps[0]]
            b1, s1, l1, ls1 = stats[temps[-1]]
            db, dl = abs(b0) - abs(b1), l0 - l1
            sd, sl = math.sqrt(s0 ** 2 + s1 ** 2), math.sqrt(ls0 ** 2 + ls1 ** 2)
            diff = f"{db:+.4f} ({db / max(sd, 1e-9):+.1f} SE)  {dl:+.4f} ({dl / max(sl, 1e-9):+.1f} SE)"
        out.append(f"{name:22s}  {'  '.join(cells)}  {diff}")
    return "\n".join(out)


def calibration_report(games: Sequence[Game], bins: int = 10) -> str:
    """Predicted win probability (v + 1) / 2 at every decision, from the mover's side, against
    whether the mover won. Reliability by bin, bias, ECE, Brier and its skill over the base rate,
    and AUC; SEs clustered by game. Each game contributes hundreds of correlated positions."""
    p_l: List[float] = []
    y_l: List[float] = []
    g_l: List[int] = []
    side_l: List[int] = []
    turn_l: List[int] = []
    end_l: List[str] = []
    for gi, g in enumerate(games):
        if g.us_util == 0.0:
            continue
        for d in g.decisions:
            p_l.append((d.value + 1.0) / 2.0)
            y_l.append(1.0 if g.us_util * d.side > 0 else 0.0)
            g_l.append(gi)
            side_l.append(d.side)
            turn_l.append(d.turn)
            end_l.append(g.blunder or "normal")
    p, y, gid = np.asarray(p_l), np.asarray(y_l), np.asarray(g_l)
    side, turn, end = np.asarray(side_l), np.asarray(turn_l), np.asarray(end_l)
    out: List[str] = []

    def summary(sel: Any) -> str:
        ps, ys, gs = p[sel], y[sel], gid[sel]
        bias, bse, n = _clustered(list(ps - ys), list(gs))
        brier = float(np.mean((ps - ys) ** 2))
        base = float(np.mean(ys))
        skill = 1.0 - brier / max(base * (1 - base), 1e-9)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(ps, edges) - 1, 0, bins - 1)
        ece = float(sum(abs(ps[idx == b].mean() - ys[idx == b].mean()) * (idx == b).mean()
                        for b in range(bins) if (idx == b).any()))
        auc = _auc(ps, ys)
        ll, llse, _ = _clustered(list(_log_loss(ps, ys)), list(gs))
        return (f"bias {bias:+.4f} ± {bse:.4f}  log loss {ll:.4f} ± {llse:.4f}  ECE {ece:.4f}  Brier {brier:.4f}  "
                f"skill {skill:+.3f}  AUC {auc:.3f}  (n={n:,}, win rate {base:.3f})")

    allsel = np.ones_like(p, dtype=bool)
    out.append(f"Calibration of v_win at every decision (mover's side), {len(set(g_l)):,} games:")
    out.append("  all:      " + summary(allsel))
    out.append("  as US:    " + summary(side == 1))
    out.append("  as USSR:  " + summary(side == -1))
    out.append("\nReliability (predicted win probability bin -> realised win rate, game-clustered SE):")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        m, se, n = _clustered(list(y[sel]), list(gid[sel]))
        out.append(f"  [{edges[b]:.1f}, {edges[b + 1]:.1f})  predicted {p[sel].mean():.3f}  realised {m:.3f} ± {se:.3f}"
                   f"  gap {m - p[sel].mean():+.3f}  (n={n:,})")
    out.append("\nBy turn:")
    for t in sorted(set(turn_l)):
        out.append(f"  turn {t:2d}: " + summary(turn == t))
    out.append("\nBy DEFCON:")
    defcon = np.asarray([d.defcon for g in games if g.us_util != 0.0 for d in g.decisions])
    for dc in sorted(set(defcon.tolist())):
        out.append(f"  DEFCON {dc}: " + summary(defcon == dc))
    out.append("\nBy how the game ended:")
    for e in sorted(set(end_l)):
        out.append(f"  {e:13s}: " + summary(end == e))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperatures", type=float, nargs="+", default=None,
                    help="P33b: play at each temperature and compare the critic across them (first minus last)")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_eval(a.checkpoint, dev, merged_ok=False).model
    if a.temperatures:
        by_t: Dict[float, Sequence[Game]] = {}
        for t in a.temperatures:
            gs: List[Game] = []
            for b in range(a.batches):
                gs += play(model, dev, num_envs=a.envs, base_seed=a.seed + 1000 * b, temperature=t)
            by_t[t] = gs
            print(f"checkpoint {a.checkpoint}, temperature {t}, {len(gs)} games")
            print(calibration_report(gs))
            print()
        print(compare_report(by_t))
        return 0
    games: List[Game] = []
    for b in range(a.batches):
        games += play(model, dev, num_envs=a.envs, base_seed=a.seed + 1000 * b, temperature=a.temperature)
    print(f"checkpoint {a.checkpoint}, temperature {a.temperature}")
    print(report(games))
    print()
    print(calibration_report(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
