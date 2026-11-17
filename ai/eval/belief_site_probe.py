"""How well does the belief root's belief guess a person's hand, against how well it guesses its own?

P35's human probe (`ai.eval.belief_human_probe`) scored the policy-likelihood belief on the ts-replayer
corpus, conditioning on card and play-mode choices only (those logs order a play's points arbitrarily).
The games people play against the models on the site (`tools/game_stats.py`) record every one of the
person's decisions as the engine asked it, so the belief can be scored here exactly as the root runs it:
`ts.BeliefTracker` (bindings/belief.hpp) -- `particles` hands, every opposing decision of the turn,
the temperature model-averaged from the opponent's play, box 4's revealed headline (`--no-reveal` drops
it) -- fed every decision of the game, asked for a belief at each decision of the AI's side.

Any network can be the belief's model of the person on any site game: the belief needs only the record
and the network's likelihoods, not that the network played the game. Two networks on the same games see
the same positions and, the draws being keyed by the tracker's seed and the position, the same hands --
their difference is paired.

The reference is the network's own self-play (`--selfplay-games`, the policy sampled at T = 1), where the
likelihood is exact, scored the same way at every decision of either side.

Per position (the decider's hidden pool: the opponent's unseen hand and the deck; the hand's true cards):

* **log loss** per pool card against the prior (hand size / pool size) -- P35's number;
* **hit rate**: the share of a dealt world's hand that the opponent really holds (the expected overlap,
  sum_c p_c y_c / hand); the prior's is hand / pool, the oracle's 1. *Lift* = (hit - prior) / (1 - prior):
  the way from random worlds to the oracle root's;
* **joint bits**: log2 of p_belief(true hand) / p_prior(true hand) = log2 sum_T w_T L_T(h*) / Z_T, the
  normaliser Z_T the mean likelihood of the uniform hands. Its expectation is the information the
  record gives about the hand when the model is exact (>= 0); below 0 the belief misleads. Estimated:
  log Z_T from the particles is biased low when few carry the weight, so it reads high there;
* the effective number of hands, the opposing decisions conditioned on, and the temperature weights.

    PYTHONPATH=.:build/release python -m ai.eval.belief_site_probe --checkpoint net.pt \\
        --games data/game_stats/games.json --selfplay-games 256 --output out.npz
    python -m ai.eval.belief_site_probe --summarize a.npz [b.npz ...]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.loading import load_for_eval
from ai.search.belief import (DECK, KNOWN, TRANSIENT, UNKNOWN, Decision, Now, card_locations, device_log_probs, player,
                              record, side)
from tools.lib.game_stats import SubmissionDict, decode_log, final_hash, load_export, play_step, start_state

#: Positions per prepare() call: bounds the rows (positions x decisions x particles) held at once.
_GROUP = 24
#: The war's stages, by turn, for the strata.
STAGES = ((1, 3, "early"), (4, 7, "mid"), (8, 10, "late"))
#: Per-position record fields, in the .npz's column order.
FIELDS = ("source", "game", "me", "turn", "hand", "pool", "decisions", "ll_prior", "ll_post", "hit_prior",
          "hit_post", "joint_bits", "ess")


def model_features_of(label: str) -> int:
    """The feature set (rule bits) the page seated a model with, from its run name's engine epoch."""
    if label.startswith("E8.1"):
        return int(ts.OBS_FEATURE_HEADLINE_REVEAL) | int(ts.OBS_FEATURE_OPP_EVENT_GATED)
    if label.startswith("E8"):
        return int(ts.OBS_FEATURE_HEADLINE_REVEAL)
    return 0


@dataclass
class SiteGame:
    """A site game that re-drives exactly on this build: its seed, steps and seats."""
    sub: SubmissionDict
    seed: int
    steps: list
    human: int                 # 1 US, 2 USSR
    seated: bool               # the person seated with OBS_FEATURES_PERSON_RULES (else unseated)

    def start(self) -> ts.GameState:
        st = start_state(self.seed, None)
        if self.seated:
            st.seat_obs_features(player(self.human), int(ts.OBS_FEATURES_PERSON_RULES))
            label = next(iter(self.sub["models"].values()))["label"]
            st.seat_obs_features(player(3 - self.human), model_features_of(label))
        return st


def _replays(g: SiteGame) -> bool:
    try:
        st = g.start()
        for s in g.steps:
            play_step(st, s)
    except Exception:  # the engine's refusal, whatever its type: the game is not this build's
        return False
    return final_hash(st) == g.sub["final_hash"]


def site_games(path: str, clean_only: bool) -> Tuple[List[SiteGame], Dict[str, int]]:
    """The submissions with one person against one model from the deal, each re-driven: seated as the page
    seats them (the person every person rule bit, the model its own), else unseated (what an older page
    did); kept when one ends on the submitted position."""
    out: List[SiteGame] = []
    counts: Dict[str, int] = {"submitted": 0, "not one person vs one model": 0, "not clean": 0,
                              "does not replay": 0, "kept": 0}
    for sub in load_export(path):
        counts["submitted"] += 1
        hs = sub["summary"]["human_side"]
        if hs not in ("US", "USSR") or sub["start_position"] is not None or len(sub["models"]) != 1:
            counts["not one person vs one model"] += 1
            continue
        if clean_only and not sub["summary"]["clean"]:
            counts["not clean"] += 1
            continue
        seed, steps = decode_log(sub["log"])
        human = 1 if hs == "US" else 2
        for seated in (True, False):
            g = SiteGame(sub, seed, steps, human, seated)
            if _replays(g):
                out.append(g)
                counts["kept"] += 1
                break
        else:
            counts["does not replay"] += 1
    return out, counts


def flat_of(state: ts.GameState, micro: ts.MicroAction) -> Optional[int]:
    """The legal flat action that decodes to `micro` (as the page's ts_belief_observe finds it), or None."""
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    for a in np.flatnonzero(mask):
        m = ts.decode_flat_action(state, int(a))
        if (int(m.decision_type), int(m.primary_id), int(m.secondary_id), int(m.flags)) == \
                (int(micro.decision_type), int(micro.primary_id), int(micro.secondary_id), int(micro.flags)):
            return int(a)
    return None


@dataclass
class _Belief:
    """One side's last belief in one game, while the tracker reuses it."""
    marginal: np.ndarray
    pool: np.ndarray           # the scored pool (the tracker's, a revealed headline left out)
    hand: int
    joint_bits: float
    ess: float
    decisions: int
    locs: np.ndarray           # every card's location when it was weighed


@dataclass
class Prober:
    """The tracker as the root runs it, plus what it takes to score its belief against the true hand."""
    model: torch.nn.Module
    device: torch.device
    features: int
    particles: int
    seed: int
    reveal: bool
    tracker: ts.BeliefTracker = field(init=False)
    logs: Dict[int, List[Decision]] = field(default_factory=dict)
    turns: Dict[int, int] = field(default_factory=dict)
    cache: Dict[Tuple[int, int], _Belief] = field(default_factory=dict)
    temperatures: List[float] = field(init=False)
    temp_weight_sum: np.ndarray = field(init=False)
    joint_mismatch: int = 0

    def __post_init__(self) -> None:
        self.tracker = ts.BeliefTracker(self.particles, True, self.seed, self.reveal)
        self.temperatures = list(ts.BeliefTracker.temperatures())
        self.temp_weight_sum = np.zeros(len(self.temperatures))

    def observe(self, game: int, state: ts.GameState, decider: int, action: int) -> None:
        """`decider` 1 US / 2 USSR; mirrors the tracker's own turn log for the true hand's likelihood."""
        self.tracker.observe(game, state, player(decider), action)
        if self.turns.get(game) != int(state.turn):
            self.logs[game], self.turns[game] = [], int(state.turn)
        self.logs[game].append(Decision(state.clone(), decider, action, card_locations(state)))

    def _true_hand_lp(self, game: int, now: Now, truth: np.ndarray, n_decisions: int) -> Optional[np.ndarray]:
        """(T x D) log-probabilities of the opponent's recorded decisions with its true hand, through the
        tracker's own rows (`device_log_probs`); None if the Python record is not the tracker's."""
        rec = record(self.logs.get(game, []), now, True)
        if len(rec) != n_decisions:
            return None
        if not rec:
            return np.zeros((len(self.temperatures), 0))
        opp = 3 - now.me
        held = truth.astype(np.uint8)[None, :]
        bases, pools, lefts, packed, acts = [], [], [], [], []
        for d, left in rec:
            h = np.ascontiguousarray(held | left.astype(np.uint8)[None, :])
            bases.append(np.asarray(ts.extract_observation_features(d.state, player(opp), self.features), dtype=np.float32))
            pools.append(((d.locs == DECK) | (d.locs == UNKNOWN[opp])).astype(np.uint8))
            lefts.append(left.astype(np.uint8))
            packed.append(np.packbits(ts.hypothesis_masks(d.state, player(now.me), h), axis=1))
            acts.append(d.action)
        rows = (np.stack(bases), np.stack(pools), np.stack(lefts), np.zeros(len(rec), dtype=np.int64),
                np.asarray(acts, dtype=np.int64), held[None, :, :], np.stack(packed))
        lp = device_log_probs(self.model, self.device, rows, self.temperatures)   # T x D x 1
        return lp[:, :, 0].double().cpu().numpy()

    def score(self, games: Sequence[int], states: Sequence[ts.GameState]) -> List[Optional[Tuple[_Belief, np.ndarray]]]:
        """Each position's belief (its decider's, about the opponent's unseen hand) and the true hand, or
        None where the tracker has none."""
        out: List[Optional[Tuple[_Belief, np.ndarray]]] = []
        for lo in range(0, len(games), _GROUP):
            out.extend(self._score_group(list(games[lo:lo + _GROUP]), list(states[lo:lo + _GROUP])))
        return out

    def _score_group(self, games: List[int], states: List[ts.GameState]) -> List[Optional[Tuple[_Belief, np.ndarray]]]:
        has = self.tracker.prepare(games, states)
        jobs = int(self.tracker.job_count())
        fresh: Dict[int, _Belief] = {}
        if jobs:
            rows = self.tracker.rows_patch(self.features)
            R = len(rows[0])
            lp = device_log_probs(self.model, self.device, rows, self.temperatures) if R else \
                torch.zeros(len(self.temperatures), 0, self.particles, device=self.device)
            if R:
                self.tracker.set_log_likelihoods(np.ascontiguousarray(lp.cpu().numpy(), dtype=np.float32))
            else:
                self.tracker.set_log_likelihoods(np.zeros((len(self.temperatures), 0, self.particles), dtype=np.float32))
            job_of = torch.from_numpy(np.asarray(rows[3], dtype=np.int64)).to(self.device)
            logw = torch.zeros(len(self.temperatures), jobs, self.particles, dtype=torch.float64, device=self.device)
            if R:
                logw.index_add_(1, job_of, lp.double())
            for j in range(jobs):
                p = int(self.tracker.job_position(j))
                fresh[p] = self._belief(games[p], states[p], j, logw[:, j, :], int(self.tracker.job_decisions(j)))
        out: List[Optional[Tuple[_Belief, np.ndarray]]] = []
        for p, (g, st) in enumerate(zip(games, states)):
            if not has[p]:
                out.append(None)
                continue
            me = side(1 if st.ctx().decision_player == ts.Player.US else -1)
            key = (g, me)
            locs = card_locations(st)
            if p in fresh:
                self.cache[key] = fresh[p]
            elif key not in self.cache or not np.array_equal(self.cache[key].locs, locs):
                # No job and not the last belief (the tracker reuses one only on an unchanged board): no
                # opposing decision this turn to condition on, the uniform prior.
                self.cache[key] = self._prior(st, me)
            self.temp_weight_sum += np.asarray(self.tracker.temperature_weights(p))
            out.append((self.cache[key], locs == UNKNOWN[3 - me]))
        return out

    def _prior(self, state: ts.GameState, me: int) -> _Belief:
        now = Now.of(state, me)
        n = int(now.pool.sum())
        return _Belief(np.where(now.pool, now.hand / max(n, 1), 0.0), now.pool, now.hand, 0.0, float(self.particles), 0,
                       now.locs)

    def _belief(self, game: int, state: ts.GameState, j: int, logw: torch.Tensor, n_dec: int) -> _Belief:
        """Job j's mixture over its hands -- the tracker's `set_log_likelihoods`, rebuilt -- and its scores."""
        hands = np.asarray(self.tracker.job_hands(j), dtype=np.float64)          # N x 110
        tw = np.asarray(self.tracker.temperature_weights(int(self.tracker.job_position(j))), dtype=np.float64)
        lw = logw.cpu().numpy()                                                   # T x N
        mx = lw.max(1, keepdims=True)
        e = np.exp(lw - mx)
        per_t = e / e.sum(1, keepdims=True)
        mixture = (tw[:, None] * per_t).sum(0)
        mixture /= mixture.sum()
        marginal = mixture @ hands
        me = side(1 if state.ctx().decision_player == ts.Player.US else -1)
        now = Now.of(state, me)
        pool = now.pool.copy()
        revealed = hands.min(0) > 0.5                                             # in every hand: box 4's
        pool &= ~revealed
        truth = now.locs == UNKNOWN[3 - me]
        ess = float(1.0 / (mixture ** 2).sum())
        # The true hand's likelihood at each temperature over the uniform hands' mean (the evidence).
        lp_true = self._true_hand_lp(game, now, truth, n_dec)
        if lp_true is None:
            self.joint_mismatch += 1
            joint = math.nan
        else:
            log_l = lp_true.sum(1)                                                  # T
            log_z = mx[:, 0] + np.log(e.mean(1))
            r = log_l - log_z
            top = r.max()
            joint = float((top + np.log((tw * np.exp(r - top)).sum())) / math.log(2))
        return _Belief(marginal, pool, int(truth[pool].sum()), joint, ess, n_dec, now.locs)


def _row(source: int, game: int, me: int, turn: int, b: _Belief, truth: np.ndarray) -> List[float]:
    m = b.pool
    y = truth[m].astype(np.float64)
    n = int(m.sum())
    q = b.hand / n
    p = np.clip(b.marginal[m], 1e-6, 1 - 1e-6)
    ll_post = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    ll_prior = float(-(y * math.log(q) + (1 - y) * math.log(1 - q)).sum()) if 0 < q < 1 else 0.0
    hit_post = float((b.marginal[m] * y).sum() / b.hand)
    return [source, game, me, turn, b.hand, n, b.decisions, ll_prior, ll_post, q, hit_post, b.joint_bits, b.ess]


def run_site(prober: Prober, games: List[SiteGame]) -> List[List[float]]:
    """Every site game in lockstep: at each decision of the AI's side the belief about the person's hand."""
    states = [g.start() for g in games]
    k = [0] * len(games)
    rows: List[List[float]] = []
    while True:
        live = [i for i, g in enumerate(games) if k[i] < len(g.steps)]
        if not live:
            break
        ask = [i for i in live if side(1 if states[i].ctx().decision_player == ts.Player.US else -1) == 3 - games[i].human
               and states[i].ctx().decision_player in (ts.Player.US, ts.Player.USSR)
               and not np.isin(card_locations(states[i]), TRANSIENT).any()]
        for i, res in zip(ask, prober.score(ask, [states[i] for i in ask])):
            if res is not None and res[0].hand > 0:
                rows.append(_row(0, i, 3 - games[i].human, int(states[i].turn), res[0], res[1]))
        for i in live:
            st = states[i]
            step = games[i].steps[k[i]]
            dp = st.ctx().decision_player
            if dp in (ts.Player.US, ts.Player.USSR):
                a = flat_of(st, step.micro_action())
                if a is not None:
                    prober.observe(i, st, 1 if dp == ts.Player.US else 2, a)
            play_step(st, step)
            k[i] += 1
    return rows


def run_selfplay(prober: Prober, n_games: int, seed: int) -> List[List[float]]:
    """The network's own games (the policy at T = 1, both seats in its feature set): at each decision, the
    decider's belief about the other's hand."""
    runner = ts.VectorizedBatchRunner(n_games, seed)
    if prober.features:
        runner.set_obs_features([prober.features] * n_games, [prober.features] * n_games)
    done = np.zeros(n_games, dtype=bool)
    rows: List[List[float]] = []
    for _ in range(8000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        with torch.no_grad():
            legal = torch.from_numpy(masks).to(prober.device) > 0
            legal[~legal.any(-1), 0] = True
            lg = prober.model(torch.from_numpy(obs).to(prober.device), torch.from_numpy(masks).to(prober.device))[0].float()
            acts = torch.multinomial(torch.softmax(lg.masked_fill(~legal, float("-inf")), -1), 1).squeeze(-1).cpu().numpy()
        live = [int(i) for i in np.flatnonzero(~done) if int(players[i]) != 0]
        states = {i: runner.get_state(i) for i in live}
        ask = [i for i in live if not np.isin(card_locations(states[i]), TRANSIENT).any()]
        for i, res in zip(ask, prober.score(ask, [states[i] for i in ask])):
            if res is not None and res[0].hand > 0:
                rows.append(_row(1, i, side(int(players[i])), int(states[i].turn), res[0], res[1]))
        for i in live:
            prober.observe(i, states[i], side(int(players[i])), int(acts[i]))
        acts[done] = 0
        runner.step_flat_all([int(a) for a in acts], True)
    return rows


# ---- the summary -------------------------------------------------------------------------------

def _stage(turn: np.ndarray) -> np.ndarray:
    out = np.full(len(turn), -1)
    for s, (lo, hi, _) in enumerate(STAGES):
        out[(turn >= lo) & (turn <= hi)] = s
    return out


def _metrics(r: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Pooled metrics of position rows `r` (FIELDS columns) with position weights `w`."""
    c = {f: i for i, f in enumerate(FIELDS)}
    pool, hand = r[:, c["pool"]], r[:, c["hand"]]
    llq = (w * r[:, c["ll_prior"]]).sum() / (w * pool).sum()
    llp = (w * r[:, c["ll_post"]]).sum() / (w * pool).sum()
    hq = (w * hand * r[:, c["hit_prior"]]).sum() / (w * hand).sum()
    hp = (w * hand * r[:, c["hit_post"]]).sum() / (w * hand).sum()
    jb = r[:, c["joint_bits"]]
    ok = np.isfinite(jb)
    return {"ll_prior": llq, "ll_post": llp, "ll_gain_pct": 100 * (llq - llp) / llq,
            "bits_per_position": float((w * (r[:, c["ll_prior"]] - r[:, c["ll_post"]])).sum() / w.sum() / math.log(2)),
            "hit_prior": hq, "hit_post": hp, "lift_pct": 100 * (hp - hq) / (1 - hq),
            "joint_bits": float((w[ok] * jb[ok]).sum() / w[ok].sum()) if ok.any() else math.nan,
            "positions": float(len(r))}


def _rows_by_game(game: np.ndarray) -> List[np.ndarray]:
    """Each game's row indices, for resampling whole games."""
    order = np.argsort(game, kind="stable")
    _, starts = np.unique(game[order], return_index=True)
    return np.split(order, starts[1:])


def _strata(r: np.ndarray) -> np.ndarray:
    c = {f: i for i, f in enumerate(FIELDS)}
    return (r[:, c["me"]].astype(int) - 1) * len(STAGES) + _stage(r[:, c["turn"]].astype(int))


def summarize(paths: Sequence[str], boot: int = 1000, seed: int = 0) -> None:
    """Site and self-play side by side for each file; self-play reweighted to the site's (side, stage) mix;
    bootstrap over games."""
    rng = np.random.default_rng(seed)
    keys = ("ll_gain_pct", "bits_per_position", "hit_prior", "hit_post", "lift_pct", "joint_bits")
    site_by_file: Dict[str, np.ndarray] = {}
    for path in paths:
        z = np.load(path, allow_pickle=False)
        rows = z["rows"]
        label = str(z["label"])
        c = {f: i for i, f in enumerate(FIELDS)}
        site = rows[rows[:, c["source"]] == 0]
        sp = rows[rows[:, c["source"]] == 1]
        site_by_file[path] = site
        print(f"\n{label}  ({path})")
        print(f"  temperature weights at site positions {z['tw_site']}, self-play {z['tw_self']} (T = {z['temps']})")
        target = np.bincount(_strata(site), minlength=2 * len(STAGES)).astype(np.float64) if len(site) else None
        groups: List[Tuple[str, np.ndarray, np.ndarray]] = []
        if len(site):
            groups.append(("site (people)", site, np.ones(len(site))))
        if len(sp):
            groups.append(("self-play", sp, np.ones(len(sp))))
            if target is not None:
                have = np.bincount(_strata(sp), minlength=2 * len(STAGES)).astype(np.float64)
                ratio = np.where(have > 0, target / np.maximum(have, 1) * (len(sp) / max(target.sum(), 1)), 0.0)
                groups.append(("self-play, site's mix", sp, ratio[_strata(sp)]))
        print(f"  {'':24s} {'positions':>9s} {'LL gain':>8s} {'bits/pos':>8s} {'hit prior':>9s} {'hit':>6s} "
              f"{'lift':>6s} {'joint bits':>10s}")
        for name, r, w in groups:
            m = _metrics(r, w)
            by_game = _rows_by_game(r[:, c["game"]])
            bs: Dict[str, List[float]] = {k: [] for k in keys}
            for _ in range(boot):
                idx = np.concatenate([by_game[g] for g in rng.integers(0, len(by_game), len(by_game))])
                mb = _metrics(r[idx], w[idx])
                for k in keys:
                    bs[k].append(mb[k])
            se = {k: float(np.nanstd(bs[k])) for k in keys}
            print(f"  {name:24s} {int(m['positions']):9d} {m['ll_gain_pct']:+7.1f}% {m['bits_per_position']:8.3f} "
                  f"{100 * m['hit_prior']:8.1f}% {100 * m['hit_post']:5.1f}% {m['lift_pct']:+5.1f}% {m['joint_bits']:+10.3f}")
            print(f"  {'  (bootstrap s.e.)':24s} {'':9s} {se['ll_gain_pct']:7.1f}  {se['bits_per_position']:8.3f} "
                  f"{'':9s} {100 * se['hit_post']:5.1f}  {se['lift_pct']:5.1f}  {se['joint_bits']:10.3f}")
            for s_me in (1, 2):
                for s, (lo, hi, nm) in enumerate(STAGES):
                    sel = (r[:, c["me"]] == s_me) & (_stage(r[:, c["turn"]].astype(int)) == s)
                    if sel.sum() < 50:
                        continue
                    ms = _metrics(r[sel], w[sel])
                    who = ("US" if s_me == 2 else "USSR") + "'s hand, " + nm
                    print(f"    {who:22s} {int(sel.sum()):9d} {ms['ll_gain_pct']:+7.1f}% "
                          f"{ms['bits_per_position']:8.3f} {100 * ms['hit_prior']:8.1f}% {100 * ms['hit_post']:5.1f}% "
                          f"{ms['lift_pct']:+5.1f}% {ms['joint_bits']:+10.3f}")
    if len(site_by_file) == 2:
        (pa, a), (pb, b) = site_by_file.items()
        c = {f: i for i, f in enumerate(FIELDS)}
        if len(a) == len(b) and np.array_equal(a[:, [c["game"], c["turn"], c["hand"]]], b[:, [c["game"], c["turn"], c["hand"]]]):
            by_game = _rows_by_game(a[:, c["game"]])
            d = {k: _metrics(b, np.ones(len(b)))[k] - _metrics(a, np.ones(len(a)))[k] for k in keys}
            bs = {k: [] for k in keys}
            for _ in range(boot):
                idx = np.concatenate([by_game[g] for g in rng.integers(0, len(by_game), len(by_game))])
                for k in keys:
                    bs[k].append(_metrics(b[idx], np.ones(len(idx)))[k] - _metrics(a[idx], np.ones(len(idx)))[k])
            print(f"\npaired on the site positions, second minus first ({pb} - {pa}):")
            for k in keys:
                print(f"  {k:18s} {d[k]:+.4f}  (s.e. {float(np.nanstd(bs[k])):.4f})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--label", default="", help="the network's name in the summary")
    ap.add_argument("--games", help="the site's games: tools/game_stats.py fetch's JSON export")
    ap.add_argument("--all-games", action="store_true", help="also games the page did not mark clean")
    ap.add_argument("--limit", type=int, default=None, help="only the first N kept site games")
    ap.add_argument("--selfplay-games", type=int, default=0)
    ap.add_argument("--particles", type=int, default=2048)
    ap.add_argument("--no-reveal", action="store_true", help="the root without box4")
    ap.add_argument("--seed", type=int, default=9093)
    ap.add_argument("--output", help=".npz of the per-position rows")
    ap.add_argument("--summarize", nargs="+", help="summarize these .npz files (no checkpoint needed)")
    a = ap.parse_args(argv)
    if a.summarize:
        summarize(a.summarize)
        return 0
    if not a.checkpoint or not a.output:
        ap.error("--checkpoint and --output are required unless --summarize")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = load_for_eval(a.checkpoint, str(dev), merged_ok=False)
    rows: List[List[float]] = []
    tw_site = np.zeros(len(ts.BeliefTracker.temperatures()))
    tw_self = tw_site.copy()
    mismatch = 0
    if a.games:
        games, counts = site_games(a.games, not a.all_games)
        if a.limit is not None:
            games = games[:a.limit]
        print("site games: " + ", ".join(f"{k} {v}" for k, v in counts.items())
              + f"; seated {sum(g.seated for g in games)}, unseated {sum(not g.seated for g in games)}", flush=True)
        pr = Prober(ev.model, dev, ev.features, a.particles, a.seed, not a.no_reveal)
        rows += run_site(pr, games)
        tw_site, mismatch = pr.temp_weight_sum / max(pr.temp_weight_sum.sum(), 1e-9), pr.joint_mismatch
        print(f"site: {len(rows):,} positions", flush=True)
    if a.selfplay_games:
        pr = Prober(ev.model, dev, ev.features, a.particles, a.seed + 1, not a.no_reveal)
        n0 = len(rows)
        rows += run_selfplay(pr, a.selfplay_games, a.seed)
        tw_self, mismatch = pr.temp_weight_sum / max(pr.temp_weight_sum.sum(), 1e-9), mismatch + pr.joint_mismatch
        print(f"self-play: {len(rows) - n0:,} positions", flush=True)
    print(f"joint bits not computed (the Python record is not the tracker's) at {mismatch} beliefs")
    np.savez(a.output, rows=np.asarray(rows, dtype=np.float64), label=a.label or a.checkpoint,
             temps=np.asarray(ts.BeliefTracker.temperatures()), tw_site=tw_site, tw_self=tw_self)
    summarize([a.output], boot=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
