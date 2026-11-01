"""The policy-likelihood belief over the opponent's unseen hand (P35 item 3b).

P35a (`ai.eval.belief_history_probe`) measured it: hands drawn uniformly from the decider's hidden
pool, each weighted by the probability the network gives the opponent's actual decisions this turn
with that hand put back into each decision's position, predict the hand at log loss 0.336 against the
uniform prior's 0.384 and the observation's 0.371. This module makes it a world sampler for the
rollout root: `BeliefTracker` logs every decision of every game it is shown (`observe`), and at a
root decision `worlds` deals the root's worlds by resampling the weighted hands.

The record and the hypothesis. Within a turn the opponent's unseen hand only loses cards, and every
card it loses leaves the decider's hidden pool for somewhere visible. So a hypothesis for the hand
now fixes the hand at each earlier decision of the turn: the hypothesis plus the cards that have left
the pool since, for anywhere but the decider's own hand (`left_since`). A decision whose record does
not fit (a card entered the pool since, or the hand changed some other way -- an event's draw) is left
out, not approximated. Earlier turns are not used: the deal mixes the held cards with new ones.

The weights are the network's own: exact in self-play, a model of the opponent otherwise. The hands
are importance-sampled, so their number (`particles`) sets the noise: P35a's log loss was 0.353 at
256 and 0.336 at 2,048 on the same positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

DECK = int(ts.CardLocation.DRAW_DECK)
UNKNOWN = {1: int(ts.CardLocation.HAND_US_UNKNOWN), 2: int(ts.CardLocation.HAND_USSR_UNKNOWN)}
KNOWN = {1: int(ts.CardLocation.HAND_US_KNOWN), 2: int(ts.CardLocation.HAND_USSR_KNOWN)}
#: Locations whose meaning to the decider is transient; at a position holding one there is no belief.
TRANSIENT = (int(ts.CardLocation.PEEKED_TEMP), int(ts.CardLocation.HEADLINE_COMMITTED))
#: The rows of one network call.
_CHUNK = 16384
#: The flat action space's width.
ACTIONS = ActionEncoder.FLAT_ACTION_SIZE


def side(code: int) -> int:
    """The runner's decision-player code (US 1, USSR -1) as 1 (US) / 2 (USSR)."""
    return 1 if code == 1 else 2


def player(s: int) -> ts.Player:
    return ts.Player.US if s == 1 else ts.Player.USSR


def card_locations(st: ts.GameState) -> np.ndarray:
    return np.asarray([int(st.get_card_location(c)) for c in range(1, 111)], dtype=np.int64)


@dataclass
class Decision:
    """One decision of the current turn: the position before it, who decided (1 US, 2 USSR), what
    they chose, and every card's location then."""
    state: ts.GameState
    decider: int
    action: int
    locs: np.ndarray


@dataclass
class Now:
    """The decider's view of the position a belief is for."""
    me: int                  # 1 US, 2 USSR
    locs: np.ndarray         # every card's location
    pool: np.ndarray         # 110 booleans: the decider's hidden pool (the opponent's unknown hand, the deck)
    hand: int                # the opponent's unknown hand size
    headline_hidden: bool = False   # an opposing headline chosen now is not revealed to the decider

    @staticmethod
    def of(st: ts.GameState, me: int) -> "Now":
        locs = card_locations(st)
        opp = 3 - me
        return Now(me, locs, (locs == DECK) | (locs == UNKNOWN[opp]), int((locs == UNKNOWN[opp]).sum()),
                   st.current_phase == ts.Phase.HEADLINE and int(st.headline_stage) == 0)


def left_since(d: Decision, now: Now) -> Optional[np.ndarray]:
    """The cards that were in the decider's hidden pool at `d` and have left it since, for anywhere but
    the decider's own hand: with the opponent's unseen hand now, its unseen hand at `d`. None when `d`
    is not on this hand's record (a card entered the pool since, or the hand changed by other means)."""
    me, opp = now.me, 3 - now.me
    pool_tau = (d.locs == DECK) | (d.locs == UNKNOWN[opp])
    if not pool_tau[now.pool].all():
        return None
    in_my_hand = (now.locs == UNKNOWN[me]) | (now.locs == KNOWN[me])
    left = pool_tau & ~now.pool & ~in_my_hand
    if int((d.locs == UNKNOWN[opp]).sum()) != int(left.sum()) + now.hand:
        return None
    return left


def record(log: Sequence[Decision], now: Now, all_decisions: bool) -> List[Tuple[Decision, np.ndarray]]:
    """The opponent's decisions this turn that the belief conditions on, each with `left_since`."""
    opp = 3 - now.me
    out: List[Tuple[Decision, np.ndarray]] = []
    for d in log:
        if d.decider != opp:
            continue
        if now.headline_hidden and d.state.current_phase == ts.Phase.HEADLINE and int(d.state.headline_stage) == 0 \
                and d.state.ctx().decision_type == ts.DecisionType.SELECT_CARD and int(d.state.ctx().resolving_card) == 0:
            continue   # chosen but not revealed: it stays in the hand, and would deal into every world
        if not all_decisions and int(d.state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            continue
        left = left_since(d, now)
        if left is not None:
            out.append((d, left))
    return out


def sample_hands(now: Now, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` hands of the opponent's unknown size drawn uniformly from the decider's hidden pool, as an
    n x 110 uint8 membership array."""
    pool = np.flatnonzero(now.pool)
    member = np.zeros((n, 110), dtype=np.uint8)
    if now.hand:
        idx = np.argsort(rng.random((n, len(pool))), axis=1)[:, :now.hand]
        np.put_along_axis(member, pool[idx], 1, axis=1)
    return member


#: The card block's offset in the observation, and the two location slots a hypothesis decides for a
#: hidden card seen from the opponent's side (engine/include/ts/game_state.hpp `card_slots`).
CARD_BLOCK = 84 * 26
CARD_FEATURES = 14
SLOT_DECK = 0
SLOT_MY_HAND = 1


_CARD_HAND = torch.tensor([CARD_BLOCK + c * CARD_FEATURES + SLOT_MY_HAND for c in range(110)])
_CARD_DECK = torch.tensor([CARD_BLOCK + c * CARD_FEATURES + SLOT_DECK for c in range(110)])
_BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)


def _hypothesis_obs(base: torch.Tensor, held: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Observation rows of past opposing decisions under hypothesised hands: `base` (b x width, each
    decision's own observation from the opponent's side), `held` (b x n x 110) and `pool` (b x 110, the
    decider's hidden pool then) -> (b * n) x width, each hidden card's own-hand / deck slot set by its
    hand. Exactly `ts.hypothesis_rows`'s rows (tested): from that side the hypotheses differ in nothing
    else -- the hand and deck counts are fixed by the hand size."""
    b, n = held.shape[0], held.shape[1]
    obs = base.unsqueeze(1).expand(b, n, base.shape[1]).clone()
    hand_idx, deck_idx = _CARD_HAND.to(base.device), _CARD_DECK.to(base.device)
    h = held.to(obs.dtype)
    in_pool = pool.unsqueeze(1).expand(b, n, 110)
    obs[:, :, hand_idx] = torch.where(in_pool, h, obs[:, :, hand_idx])
    obs[:, :, deck_idx] = torch.where(in_pool, 1.0 - h, obs[:, :, deck_idx])
    return obs.reshape(b * n, base.shape[1])


def _unpack(packed: torch.Tensor, width: int) -> torch.Tensor:
    """np.packbits rows (big-endian bits) back to `width` uint8 0/1 columns."""
    bits = (packed.unsqueeze(-1) & _BITS.to(packed.device)) != 0
    return bits.reshape(*packed.shape[:-1], -1)[..., :width].to(torch.uint8)


def device_log_probs(model: Any, device: torch.device, rows: Tuple[np.ndarray, ...],
                     temperatures: Sequence[float]) -> torch.Tensor:
    """(T x R x N) log-probabilities of each recorded decision's move under each hand at each temperature,
    from `rows` = (bases R x W, pools R x 110, lefts R x 110, job_of R, actions R, hands J x N x 110,
    packed masks R x N x 28) -- `ts.BeliefTracker.rows_patch`'s layout, which `log_weights` also builds.
    One transfer; the rows are built on the device (`_hypothesis_obs`) in chunks of `_CHUNK`, and each
    chunk's logits serve every temperature."""
    bases, pools, lefts, job_of, acts, hands, packed = rows
    R, n = len(bases), hands.shape[1]
    out = torch.empty(len(temperatures), R, n, dtype=torch.float32, device=device)
    if R == 0:
        return out
    with torch.no_grad():
        base_t = torch.from_numpy(np.ascontiguousarray(bases)).to(device)
        left_t = torch.from_numpy(np.ascontiguousarray(lefts)).to(device)
        pool_t = torch.from_numpy(np.ascontiguousarray(pools)).to(device).bool()
        mask_t = torch.from_numpy(np.ascontiguousarray(packed)).to(device)
        hands_t = torch.from_numpy(np.ascontiguousarray(hands)).to(device)
        act_t = torch.from_numpy(np.ascontiguousarray(acts)).to(device).long()
        job_t = torch.from_numpy(np.ascontiguousarray(job_of)).to(device).long()
        per = max(1, _CHUNK // max(n, 1))
        for lo in range(0, R, per):
            sl = slice(lo, lo + per)
            held = hands_t[job_t[sl]] | left_t[sl].unsqueeze(1)
            obs = _hypothesis_obs(base_t[sl], held, pool_t[sl])
            m = _unpack(mask_t[sl], ACTIONS).reshape(-1, ACTIONS)
            lg = model(obs, m)[0].float().masked_fill(m == 0, float("-inf"))
            a = act_t[sl].repeat_interleave(n)[:, None]
            for ti, t in enumerate(temperatures):
                lp = torch.log_softmax(lg / t, -1).gather(1, a).squeeze(1)
                out[ti, sl] = torch.where(torch.isfinite(lp), lp, torch.full_like(lp, -1e9)).reshape(-1, n)
    return out


def log_weights(model: Any, device: torch.device, jobs: Sequence[Tuple[Now, np.ndarray, List[Tuple[Decision, np.ndarray]]]],
                temperature: float = 1.0) -> List[np.ndarray]:
    """For each job (the decider's view, its hands, the record), every hand's log-likelihood of the
    opponent's recorded decisions under the network at one temperature: the probes' path, through the same
    device code as the tracker (`device_log_probs`)."""
    feats = model_obs_features(model)
    n = len(jobs[0][1]) if jobs else 0
    if any(len(h) != n for _, h, _ in jobs):
        raise ValueError("every job must carry the same number of hands")
    bases: List[np.ndarray] = []
    lefts: List[np.ndarray] = []
    pools: List[np.ndarray] = []
    packed: List[np.ndarray] = []
    acts: List[int] = []
    job_of: List[int] = []
    for j, (now, hands, rec) in enumerate(jobs):
        opp = 3 - now.me
        for d, left in rec:
            held = np.ascontiguousarray(hands | left.astype(np.uint8)[None, :])
            bases.append(np.asarray(ts.extract_observation_features(d.state, player(opp), feats), dtype=np.float32))
            lefts.append(left.astype(np.uint8))
            pools.append(((d.locs == DECK) | (d.locs == UNKNOWN[opp])).astype(np.uint8))
            packed.append(np.packbits(ts.hypothesis_masks(d.state, player(now.me), held), axis=1))
            acts.append(d.action)
            job_of.append(j)
    if not bases:
        return [np.zeros(n, dtype=np.float64) for _ in jobs]
    rows = (np.stack(bases), np.stack(pools), np.stack(lefts), np.asarray(job_of, dtype=np.int64),
            np.asarray(acts, dtype=np.int64), np.stack([h for _, h, _ in jobs]), np.stack(packed))
    lp = device_log_probs(model, device, rows, [temperature])[0].double()
    logw = torch.zeros(len(jobs), n, dtype=torch.float64, device=lp.device)
    logw.index_add_(0, torch.from_numpy(rows[3]).to(lp.device), lp)
    out = logw.cpu().numpy()
    return [out[j] for j in range(len(jobs))]


def posterior(hands: np.ndarray, logw: np.ndarray) -> Tuple[np.ndarray, float]:
    """The per-card posterior (110) and the effective number of hands."""
    w = np.exp(logw - logw.max())
    return (w[:, None] * hands).sum(0) / w.sum(), float(w.sum() ** 2 / (w ** 2).sum())


class BeliefTracker:
    """The root's belief: `ts.BeliefTracker` (bindings/belief.hpp, shared with the browser) keeps every
    game's record, draws the hands, weighs them and the likelihood temperatures, and deals the worlds;
    this wrapper evaluates the rows it asks for on `device` with `model` -- the opponent's model.

    `observe` is called with every decision of every game the agent plays (both sides), before it is
    applied; `worlds` for the decisions the agent is about to search."""

    def __init__(self, model: Any, device: torch.device, particles: int = 2048, all_decisions: bool = True,
                 seed: int = 0, reveal: bool = False) -> None:
        """`reveal` (the root's box4): every hand holds the opponent's headline revealed by Space Race box 4."""
        self.model = model
        self.device = device
        self.tracker = ts.BeliefTracker(int(particles), bool(all_decisions), int(seed), bool(reveal))
        self.temperatures = list(ts.BeliefTracker.temperatures())
        self.features = model_obs_features(model)
        #: positions given a belief, jobs weighed, decisions conditioned on, and summed temperature weights
        self.stats: Dict[str, float] = {"positions": 0, "jobs": 0, "decisions_sum": 0}
        self.temperature_weight_sum = np.zeros(len(self.temperatures))

    def reseed(self, seed: int) -> None:
        self.tracker.reseed(int(seed))

    def new_games(self) -> None:
        self.tracker.new_games()

    def observe(self, games: Sequence[int], states: Sequence[ts.GameState], actions: Sequence[int],
                players: Sequence[int]) -> None:
        for g, st, a, p in zip(games, states, actions, players):
            if int(p) != 0:
                self.tracker.observe(int(g), st, player(side(int(p))), int(a))

    def worlds(self, games: Sequence[int], states: Sequence[ts.GameState], n_worlds: int) -> List[Optional[np.ndarray]]:
        """For each game's position (the agent to move): `n_worlds` opposing hands (n_worlds x 110 uint8)
        resampled from the belief, or None where there is nothing to believe (no hidden hand, a
        transient location) -- the root then redeals at random as before."""
        has = self.tracker.prepare([int(g) for g in games], list(states))
        jobs = self.tracker.job_count()
        if jobs:
            rows = self.tracker.rows_patch(self.features)
            lp = device_log_probs(self.model, self.device, rows, self.temperatures)
            self.tracker.set_log_likelihoods(np.ascontiguousarray(lp.cpu().numpy(), dtype=np.float32))
            self.stats["jobs"] += jobs
            self.stats["decisions_sum"] += len(rows[0])
        out: List[Optional[np.ndarray]] = []
        for i, h in enumerate(has):
            out.append(cast(Optional[np.ndarray], self.tracker.worlds(i, n_worlds)) if h else None)
            if h:
                self.stats["positions"] += 1
                self.temperature_weight_sum += np.asarray(self.tracker.temperature_weights(i))
        return out
