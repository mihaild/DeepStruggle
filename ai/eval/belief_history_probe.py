"""P35a: how sharp can a belief about the opponent's hidden hand get from the public record?

P33f (`ai.eval.belief_probe`) found that the current observation supports a belief only 3.4% better
than the arithmetic prior (log loss 0.3914 -> 0.3780), and that the trunk holds exactly that. The
observation shows no history. This probe adds the public record, by two routes, read on the same
kind of held-out self-play positions (a card in the decider's hidden pool -- the opponent's unseen
hand or the deck -- labelled by whether the opponent holds it):

* **a history probe**: the P33f public probe's input plus, per card, when it last entered the
  decider's hidden pool, how many cards the opponent has drawn since, and whether it entered this
  turn -- the arithmetic of who could have drawn it;
* **the policy-likelihood posterior**: hands drawn uniformly from the hidden pool, each weighted by
  the probability the network gives the opponent's actual decisions this turn, with that hand put
  back into each decision's position (the hand then = the hypothesis plus the cards that have since
  left the hidden pool for anywhere but the decider's hand). Self-play at T = 1 makes that
  likelihood exact for these games. It uses only the public record and the shared policy, and it is
  the posterior a belief-weighted root (P35 item 3b) would deal its worlds from;
* both combined: the history probe with the posterior's logit as an extra input.

P35's rule: a log loss at most ~0.35 (three times the public probe's gain) opens 3b and 3c.

    PYTHONPATH=.:build/release python -m ai.eval.belief_history_probe --checkpoint swa.pt
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.belief_probe import (CARD_OFFSET, CARD_WIDTH, GLOBAL_OFFSET, GLOBAL_WIDTH, fit_probe, prior_logit,
                                  scores)
from ai.search.belief import Decision
from bindings.ts_env import model_obs_features
from ai.eval.loading import load_for_eval

DECK = int(ts.CardLocation.DRAW_DECK)
UNKNOWN = {1: int(ts.CardLocation.HAND_US_UNKNOWN), 2: int(ts.CardLocation.HAND_USSR_UNKNOWN)}
KNOWN = {1: int(ts.CardLocation.HAND_US_KNOWN), 2: int(ts.CardLocation.HAND_USSR_KNOWN)}
#: Locations whose meaning to the decider is transient; a position holding one is not sampled.
TRANSIENT = {int(ts.CardLocation.PEEKED_TEMP), int(ts.CardLocation.HEADLINE_COMMITTED)}
HIST_WIDTH = 5
#: The opponent's decisions the posterior conditions on: which card, and how to play it -- the choices
#: a hand constrains directly. Targets and branches inside a play are left out (cost, not principle).
CHOICES = {int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)}


def _player(code: int) -> ts.Player:
    return ts.Player.US if code == 1 else ts.Player.USSR


@dataclass
class Sample:
    pub: np.ndarray          # the P33f public input: the card block and the globals
    hist: np.ndarray         # 110 x HIST_WIDTH, flattened
    label: np.ndarray        # 110: in the opponent's unseen hand
    pool: np.ndarray         # 110: in the decider's hidden pool
    game: int
    decider: int
    locs: np.ndarray
    turn_log: List[Decision] = field(default_factory=list)
    post: Optional[np.ndarray] = None
    used: int = 0            # opponent decisions the posterior conditioned on
    ess: float = 0.0         # the posterior's effective sample size, in hands
    post_more: Optional[np.ndarray] = None   # the same posterior from --more-worlds hands
    ess_more: float = 0.0


class _History:
    """Per game and per side, the public record's arithmetic for every card: the turn it last entered
    that side's hidden pool and the cards the opponent has drawn since."""

    def __init__(self) -> None:
        self.entry = np.zeros((2, 110), dtype=np.float32)
        self.draws = np.zeros((2, 110), dtype=np.float32)
        self.prev_pool = np.zeros((2, 110), dtype=bool)
        self.prev_opp = np.zeros(2, dtype=np.int64)

    def update(self, locs: np.ndarray, turn: int) -> None:
        for p, me in ((0, 1), (1, 2)):
            opp = 3 - me
            pool = (locs == DECK) | (locs == UNKNOWN[opp])
            new = pool & ~self.prev_pool[p]
            self.entry[p, new] = turn
            self.draws[p, new] = 0
            n_opp = int(((locs == UNKNOWN[opp]) | (locs == KNOWN[opp])).sum())
            self.draws[p, pool] += max(0, n_opp - int(self.prev_opp[p]))
            self.prev_pool[p], self.prev_opp[p] = pool, n_opp

    def features(self, me: int, turn: int) -> np.ndarray:
        p = me - 1
        pool = self.prev_pool[p].astype(np.float32)
        return np.stack([pool, pool * (turn - self.entry[p]) / 10.0, pool * self.draws[p] / 10.0,
                         pool * self.entry[p] / 10.0, pool * (self.entry[p] == turn)], axis=1).reshape(-1)


def _locs(st: ts.GameState) -> np.ndarray:
    return np.asarray([int(st.get_card_location(c)) for c in range(1, 111)], dtype=np.int64)


def collect(model: torch.nn.Module, device: torch.device, games: int, every: int, temperature: float,
            seed: int, keep_log: set[int]) -> List[Sample]:
    """Self-play positions of `model`, one decision in `every`; for games in `keep_log` each sample also
    keeps the turn's decisions, which the posterior needs."""
    runner = ts.VectorizedBatchRunner(games, seed)
    feats = model_obs_features(model)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    rng = np.random.default_rng(seed)
    done = np.zeros(games, dtype=bool)
    hist = [_History() for _ in range(games)]
    logs: List[List[Decision]] = [[] for _ in range(games)]
    turns = [-1] * games
    out: List[Sample] = []
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        with torch.no_grad():
            lg = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))[0].float()
            legal = torch.from_numpy(masks).to(device) > 0
            legal[~legal.any(-1), 0] = True               # a finished game has no legal move; its pick is ignored
            lg = lg.masked_fill(~legal, float("-inf"))
            acts = (torch.multinomial(torch.softmax(lg / temperature, -1), 1).squeeze(-1) if temperature > 0
                    else lg.argmax(-1)).cpu().numpy()
        for i in np.flatnonzero(~done):
            st = runner.get_state(int(i))
            locs = _locs(st)
            turn = int(st.turn)
            if turn != turns[i]:
                logs[i], turns[i] = [], turn
            hist[i].update(locs, turn)
            if int(players[i]) == 0:
                continue
            me = 1 if int(players[i]) == 1 else 2          # the runner codes US 1, USSR -1
            if rng.random() < 1.0 / every and not np.isin(locs, list(TRANSIENT)).any():
                label = (locs == UNKNOWN[3 - me]).astype(np.float32)
                pool = ((locs == UNKNOWN[3 - me]) | (locs == DECK)).astype(np.float32)
                if label.sum() > 0:
                    out.append(Sample(obs[i, CARD_OFFSET:GLOBAL_OFFSET + GLOBAL_WIDTH].copy(),
                                      hist[i].features(me, turn), label, pool, int(i), me, locs,
                                      list(logs[i]) if int(i) in keep_log else []))
            logs[i].append(Decision(st, me, int(acts[i]), locs))
        acts[done] = 0
        runner.step_flat_all([int(a) for a in acts], True)
    return out


def left_since(d: Decision, s: Sample) -> Optional[np.ndarray]:
    """The cards that were in the decider's hidden pool at `d` and have left it since, for anywhere but
    the decider's own hand: with the opponent's unseen hand at `s`, its unseen hand at `d`. None when `d`
    is not on this hand's record (a card entered the pool since, or the hand changed by other means)."""
    me, opp = s.decider, 3 - s.decider
    pool_tau = (d.locs == DECK) | (d.locs == UNKNOWN[opp])
    if not pool_tau[s.pool > 0.5].all():
        return None
    in_my_hand = (s.locs == UNKNOWN[me]) | (s.locs == KNOWN[me])
    if (in_my_hand & (d.locs == UNKNOWN[opp])).any():
        return None   # a card the opponent held then is the decider's now (bindings/belief.cpp left_since)
    left = pool_tau & (s.pool < 0.5) & ~in_my_hand
    if int((d.locs == UNKNOWN[opp]).sum()) != int(left.sum()) + int(s.label.sum()):
        return None
    return left


def world_at(d: Decision, opp: int, held: np.ndarray) -> ts.GameState:
    """`d`'s position with the opponent's unseen hand made `held` (110 booleans) and the rest of the
    decider's hidden pool put in the deck."""
    w = d.state.clone()
    for c in np.flatnonzero((d.locs == DECK) | (d.locs == UNKNOWN[opp])):
        w.set_card_location(int(c) + 1, ts.CardLocation(UNKNOWN[opp] if held[c] else DECK))
    return w


def posterior(model: torch.nn.Module, device: torch.device, s: Sample, worlds: int, temperature: float,
              rng: np.random.Generator, all_decisions: bool = False) -> Tuple[np.ndarray, int, float]:
    """The policy-likelihood posterior over the opponent's unseen hand at `s`, and how many of the
    opponent's decisions it conditioned on (0: the uniform prior), and its effective sample size
    (sum w)^2 / sum w^2 in hands. `all_decisions` conditions on every one of the opponent's decisions
    this turn, targets and branches included, not only CHOICES."""
    opp = 3 - s.decider
    h = int(s.label.sum())
    hands = np.stack([rng.choice(np.flatnonzero(s.pool > 0.5), size=h, replace=False) for _ in range(worlds)])
    member = np.zeros((worlds, 110), dtype=np.float32)
    np.put_along_axis(member, hands, 1.0, axis=1)
    logw = np.zeros(worlds, dtype=np.float64)
    used = 0
    feats = model_obs_features(model)
    for d in s.turn_log:
        if d.decider != opp or (not all_decisions and int(d.state.ctx().decision_type) not in CHOICES):
            continue
        left = left_since(d, s)
        if left is None:
            continue
        obs_l: List[np.ndarray] = []
        mask_l: List[np.ndarray] = []
        for k in range(worlds):
            w = world_at(d, opp, left | (member[k] > 0.5))
            obs_l.append(ts.extract_observation_features(w, _player(opp), feats))
            mask_l.append(np.asarray(ts.ActionMask.generate_flat_mask(w)))
        obs, masks = np.stack(obs_l), np.stack(mask_l)
        with torch.no_grad():
            lg = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))[0].float()
            legal = torch.from_numpy(masks).to(device) > 0
            lg = (lg / temperature).masked_fill(~legal, float("-inf"))
            lp = torch.log_softmax(lg, -1)[:, d.action].double().cpu().numpy()
        logw += np.where(np.isfinite(lp), lp, -1e9)
        used += 1
    if used == 0:
        return member.mean(0), 0, float(worlds)
    w = np.exp(logw - logw.max())
    return (w[:, None] * member).sum(0) / w.sum(), used, float(w.sum() ** 2 / (w ** 2).sum())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--every", type=int, default=3, help="sample one decision in this many")
    ap.add_argument("--temperature", type=float, default=1.0, help="self-play sampling temperature")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=256, help="hands drawn per position for the posterior")
    ap.add_argument("--posterior-positions", type=int, default=4000,
                    help="held-out positions (and as many validation ones) the posterior is computed for")
    ap.add_argument("--more-worlds", type=int, default=0,
                    help="also compute the held-out posterior from this many hands: a sampling-noise check")
    ap.add_argument("--all-decisions", action="store_true",
                    help="condition the posterior on every opponent decision this turn, not only card and play-mode choices")
    ap.add_argument("--seed", type=int, default=9091)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_eval(a.checkpoint, str(dev), merged_ok=False).model
    test_games = set(range(0, a.games, 5))                       # one game in five held out
    val_games = set(range(1, a.games, 10))                       # and one in ten for early stopping
    samples = collect(model, dev, a.games, a.every, a.temperature, a.seed, test_games | val_games)
    va = [s for s in samples if s.game in val_games]
    tr = [s for s in samples if s.game not in test_games and s.game not in val_games]
    te = [s for s in samples if s.game in test_games]
    rng = np.random.default_rng(a.seed)
    # The posterior is the expensive part: computed on a subsample of the held-out and validation positions.
    te = [te[i] for i in sorted(rng.choice(len(te), size=min(a.posterior_positions, len(te)), replace=False))]
    va_post = [va[i] for i in sorted(rng.choice(len(va), size=min(a.posterior_positions, len(va)), replace=False))]
    for s in te + va_post:
        s.post, s.used, s.ess = posterior(model, dev, s, a.worlds, a.temperature, rng, a.all_decisions)
    if a.more_worlds:
        for s in te:
            s.post_more, _, s.ess_more = posterior(model, dev, s, a.more_worlds, a.temperature, rng, a.all_decisions)

    def stack(ss: List[Sample]) -> Dict[str, torch.Tensor]:
        pub = torch.from_numpy(np.stack([s.pub for s in ss])).to(dev)
        hist = torch.from_numpy(np.stack([s.hist for s in ss])).to(dev)
        y = torch.from_numpy(np.stack([s.label for s in ss])).to(dev)
        m = torch.from_numpy(np.stack([s.pool for s in ss])).to(dev)
        d: Dict[str, torch.Tensor] = {"pub": pub, "hist": torch.cat([pub, hist], dim=1), "y": y, "m": m}
        if all(s.post is not None for s in ss):
            post = np.stack([s.post for s in ss if s.post is not None]).clip(1e-4, 1 - 1e-4)
            pl = torch.from_numpy(np.log(post / (1 - post)).astype(np.float32)).to(dev)
            d["post_logit"] = pl
            d["both"] = torch.cat([pub, hist, pl], dim=1)
        return d

    dtr, dva, dte, dvp = stack(tr), stack(va), stack(te), stack(va_post)
    y, m = dte["y"].cpu().numpy(), dte["m"].cpu().numpy()
    prior = (y.sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) * m
    rows: List[Tuple[str, np.ndarray]] = [("prior: hand size / pool size", prior)]
    for key, title in (("pub", "public probe (P33f)"), ("hist", "history probe: public + per-card record")):
        probe = fit_probe(dtr[key], dtr["y"], dtr["m"], a.epochs, a.seed, (dva[key], dva["y"], dva["m"]))
        with torch.no_grad():
            rows.append((title, torch.sigmoid(probe(dte[key]) + prior_logit(dte["y"], dte["m"])).cpu().numpy()))
    rows.append(("policy-likelihood posterior (this turn)", np.stack([s.post for s in te if s.post is not None]) * m))
    if a.more_worlds:
        rows.append((f"the same from {a.more_worlds} hands", np.stack([s.post_more for s in te if s.post_more is not None]) * m))
    # The combination is fitted on the validation positions that carry a posterior (it needs one as input),
    # early-stopped on half of them, and read on the held-out set.
    half = dvp["y"].shape[0] // 2
    fit = {k: v[:half] for k, v in dvp.items()}
    stop = {k: v[half:] for k, v in dvp.items()}
    probe = fit_probe(fit["both"], fit["y"], fit["m"], 4 * a.epochs, a.seed, (stop["both"], stop["y"], stop["m"]))
    with torch.no_grad():
        rows.append(("history probe + posterior", torch.sigmoid(probe(dte["both"]) + prior_logit(dte["y"], dte["m"])).cpu().numpy()))
    used = np.asarray([s.used for s in te])
    for k, ess in ((a.worlds, np.asarray([s.ess for s in te])), (a.more_worlds, np.asarray([s.ess_more for s in te]))):
        if k:
            q = ess[used > 0]
            print(f"  effective hands of {k}: median {np.median(q):.1f}, mean {q.mean():.1f}, "
                  f"below 10 on {100 * (q < 10).mean():.0f}% of the positions with a decision")
    print(f"checkpoint {a.checkpoint}: {len(tr):,} training, {len(va):,} validation and {len(te):,} held-out positions "
          f"({a.games} games at T={a.temperature:g}, one decision in {a.every}; the posterior from {a.worlds} hands, "
          f"on {100 * (used > 0).mean():.0f}% of positions with an opponent decision this turn, mean {used.mean():.2f}); "
          f"pool entries {int(m.sum()):,}, of which in the opponent's hand {100 * y[m > 0.5].mean():.1f}%")
    base_ll = scores(prior, y, m)[0]
    for title, p in rows:
        ll, auc = scores(p, y, m)
        print(f"  {title:42s} log loss {ll:.4f} ({100 * (base_ll - ll) / base_ll:+5.1f}% vs the prior)  AUC {auc:.3f}")
    sel = used > 0
    if sel.any() and (~sel).any():
        for title, p in rows:
            a1 = scores(p[sel], y[sel], m[sel])[0]
            a0 = scores(p[~sel], y[~sel], m[~sel])[0]
            print(f"    {title:40s} with an opponent decision this turn {a1:.4f}, without {a0:.4f}")
    print(f"  P35 rule: <= ~0.35 opens 3b/3c (the prior is {base_ll:.4f}; "
          f"{math.nan if not rows else min(scores(p, y, m)[0] for _, p in rows):.4f} is the best here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
