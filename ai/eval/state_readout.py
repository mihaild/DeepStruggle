"""What can be read linearly off the trunk? A sanity check on what the network knows.

The trunk is 512 floats that every head reads. If a fact about the position cannot be recovered
from it by a *linear* probe, no head can condition on that fact either -- a deeper probe would
only show the information is recoverable in principle, which is not the question.

Four groups, chosen because each is either obviously necessary or diagnostic:

* **board** -- the viewer's influence in each of the 84 countries. Necessary for any positional
  play; if this fails, nothing else matters.
* **hand** -- whether each of the 110 cards is in the viewer's hand. This is the diagnostic one.
  The card block encodes a card's *properties* and not which card it is (`metrics.md` §21.4), and
  v2's card branch pools over shared per-card weights, so position -- the only carrier of
  identity -- is discarded. A v2 model without identity embeddings should fail here and pass
  everything else. An `--arch mlp` model reads the flat vector positionally and should pass. A v2
  model *with* `--identity-dim` should pass.
* **tracks** -- DEFCON, victory points, military operations and space race, per side.
* Per-card *effects* are deliberately not probed: most are irrelevant to most positions, so a
  low score would say nothing.

Scores are out-of-sample. Continuous targets report R^2, binary ones report AUC, both averaged
over the group, so 0 is chance and 1 is perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

import ts_engine as ts


@dataclass
class Readout:
    scores: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, float] = field(default_factory=dict)
    samples: int = 0

    def summary(self, label: str = "") -> str:
        head = f"=== {label}   {self.samples:,} positions"
        rows = [f"    {k:28s} {v:6.3f}" for k, v in self.scores.items()]
        return "\n".join([head] + rows)


def _ridge(x: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Closed-form ridge. One solve for all targets at once."""
    n_feat = x.shape[1]
    a = x.T @ x + lam * np.eye(n_feat)
    return np.linalg.solve(a, x.T @ y)


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean R^2 over columns, skipping columns that never vary."""
    out = []
    for j in range(true.shape[1]):
        var = true[:, j].var()
        if var < 1e-9:
            continue
        out.append(1.0 - ((true[:, j] - pred[:, j]) ** 2).mean() / var)
    return float(np.mean(out)) if out else float("nan")


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mean AUC over columns with both classes present."""
    out = []
    for j in range(label.shape[1]):
        y = label[:, j] > 0.5
        if y.all() or not y.any():
            continue
        order = np.argsort(score[:, j])
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(len(order))
        n1, n0 = y.sum(), (~y).sum()
        out.append((ranks[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else float("nan")


def collect(model: Any, num_envs: int = 128, steps: int = 900,
            every: int = 6, temperature: float = 0.1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Trunk features and ground truth from real decision positions."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=525_252)
    obs, masks, _ = env.reset_all()

    feats: List[np.ndarray] = []
    board: List[np.ndarray] = []
    hand: List[np.ndarray] = []
    tracks: List[np.ndarray] = []

    for step in range(steps):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(masks)).to(device)
        with torch.no_grad():
            acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)
            trunk = model.extract_features(obs_t)
            if isinstance(trunk, tuple):
                trunk = trunk[0]
        if step % every == 0:
            tr = trunk.cpu().numpy()
            for i in range(num_envs):
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                ctx = st.ctx()
                p = ctx.decision_player if ctx.decision_player != ts.Player.NONE \
                    else st.phasing_player
                if p == ts.Player.NONE:
                    continue
                mine = p == ts.Player.US
                feats.append(tr[i].copy())
                board.append(np.array(
                    [float(st.get_country(c).us_influence if mine
                           else st.get_country(c).ussr_influence) for c in range(84)],
                    dtype=np.float32))
                loc = ts.hand_of(p)
                hand.append(np.array(
                    [1.0 if st.get_card_location(c) == loc else 0.0 for c in range(1, 111)],
                    dtype=np.float32))
                sign = 1.0 if mine else -1.0
                tracks.append(np.array([
                    float(st.defcon),
                    float(st.victory_points) * sign,
                    float(st.us_mil_ops if mine else st.ussr_mil_ops),
                    float(st.ussr_mil_ops if mine else st.us_mil_ops),
                    float(st.us_space_track if mine else st.ussr_space_track),
                    float(st.ussr_space_track if mine else st.us_space_track),
                ], dtype=np.float32))
        obs, masks, *_ = env.step(acts.cpu().numpy().astype(np.int64))

    return np.asarray(feats, dtype=np.float64), {
        "board": np.asarray(board, dtype=np.float64),
        "hand": np.asarray(hand, dtype=np.float64),
        "tracks": np.asarray(tracks, dtype=np.float64),
    }


def collision_groups() -> List[List[int]]:
    """Cards that share one feature vector, so only identity can separate them."""
    import collections

    from ai.eval.positions import PositionBuilder
    from ai.models.coldwar_net_v2 import ColdWarNetV2

    st = PositionBuilder(hand=tuple(range(1, 111)), side=ts.Player.USSR,
                         defcon=3, turn=5).build()
    o = np.asarray(ts.extract_observation(st, ts.Player.USSR), dtype=np.float32)
    off, cf = ColdWarNetV2.BOARD_FEATURES * 84, ColdWarNetV2.CARD_FEATURES
    groups: Dict[Any, List[int]] = collections.defaultdict(list)
    for cid in range(1, 111):
        groups[tuple(o[off + (cid - 1) * cf: off + cid * cf].tolist())].append(cid)
    return [g for g in groups.values() if len(g) > 1]


def within_group_accuracy(x: np.ndarray, hand: np.ndarray, tr: np.ndarray,
                          te: np.ndarray) -> Tuple[float, float, int]:
    """Can the probe say *which* of a set of identical-looking cards is held?

    The aggregate hand AUC is not the test: most cards differ in Ops, side or era, so a model
    that knows only properties already scores well above chance. Restricting to positions where
    exactly one member of a collision group is in hand removes every cue except identity, and
    the question becomes a clean multi-class one against a known chance rate.
    """
    w = _ridge(x[tr], hand[tr])
    pred = x[te] @ w
    hits = trials = 0
    chance_num = 0.0
    for g in collision_groups():
        cols = [c - 1 for c in g]
        held = hand[te][:, cols]
        one = held.sum(axis=1) == 1
        if one.sum() < 20:
            continue
        truth = held[one].argmax(axis=1)
        guess = pred[one][:, cols].argmax(axis=1)
        hits += int((truth == guess).sum())
        trials += int(one.sum())
        chance_num += one.sum() / len(g)
    return (hits / trials if trials else float("nan"),
            chance_num / trials if trials else float("nan"), trials)


def probe(model: Any, **kw) -> Readout:
    x, targets = collect(model, **kw)
    out = Readout(samples=len(x))
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = np.hstack([x, np.ones((len(x), 1))])          # bias column
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    cut = int(0.7 * len(x))
    tr, te = perm[:cut], perm[cut:]

    for name, y in targets.items():
        w = _ridge(x[tr], y[tr])
        pred = x[te] @ w
        out.scores[f"{name} ({'AUC' if name == 'hand' else 'R^2'})"] = (
            _auc(pred, y[te]) if name == "hand" else _r2(pred, y[te]))

    acc, chance, n = within_group_accuracy(x, targets["hand"], tr, te)
    out.scores["hand: which twin (acc)"] = acc
    out.detail["which-twin chance"] = chance
    out.detail["which-twin trials"] = float(n)

    # The tracks are worth naming individually: DEFCON failing would mean something very
    # different from the space race failing.
    y = targets["tracks"]
    w = _ridge(x[tr], y[tr])
    pred = x[te] @ w
    for j, label in enumerate(["defcon", "victory points", "my military ops",
                               "opp military ops", "my space", "opp space"]):
        var = y[te][:, j].var()
        out.detail[label] = (float("nan") if var < 1e-9 else
                             1.0 - ((y[te][:, j] - pred[:, j]) ** 2).mean() / var)
    return out
