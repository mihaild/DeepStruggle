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
from typing import Any, Dict, List, Sequence, Tuple

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


def collect(model: Any, num_envs: int = 128, steps: int = 900, every: int = 6,
            temperature: float = 0.1
            ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Trunk features, ground truth, and the env each position came from.

    The env index is returned so a split can hold out whole *games*. Positions from one
    environment are a single game sampled every `every` steps, so they are heavily correlated --
    a random split over positions puts step t in train and step t+6 of the same game in test,
    and the probe scores partly on having seen the position already.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=525_252)
    obs, masks, _ = env.reset_all()

    feats: List[np.ndarray] = []
    env_ids: List[int] = []
    board: List[np.ndarray] = []
    board_opp: List[np.ndarray] = []
    control: List[np.ndarray] = []
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
                env_ids.append(i)
                cs = [st.get_country(c) for c in range(84)]
                board.append(np.array(
                    [float(c.us_influence if mine else c.ussr_influence) for c in cs],
                    dtype=np.float32))
                board_opp.append(np.array(
                    [float(c.ussr_influence if mine else c.us_influence) for c in cs],
                    dtype=np.float32))
                # CountryState exposes influence only, so control is derived: a side controls
                # a country when its influence exceeds the opponent's by at least the stability.
                control.append(np.array([
                    1.0 if (float(c.us_influence if mine else c.ussr_influence)
                            - float(c.ussr_influence if mine else c.us_influence))
                    >= float(ts.MapData.get_country_info(i)["stability"]) else 0.0
                    for i, c in enumerate(cs)], dtype=np.float32))
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
        "board_opp": np.asarray(board_opp, dtype=np.float64),
        "control": np.asarray(control, dtype=np.float64),
        "hand": np.asarray(hand, dtype=np.float64),
        "tracks": np.asarray(tracks, dtype=np.float64),
    }, np.asarray(env_ids, dtype=np.int64)


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


def battleground_detail(model: Any, **kw) -> List[Tuple[str, bool, int, float, float, float]]:
    """Per-country read-out, so the aggregate board R^2 can be split by what matters.

    A battleground is where the game is decided -- it is what regional scoring counts and the
    only place a coup moves DEFCON -- so a trunk that prices the board on average but loses the
    battlegrounds specifically would be a different and worse problem than a uniform blur.

    Returns, per country: name, battleground flag, region, and the held-out R^2 for the viewer's
    influence, the opponent's influence, and the AUC for "the viewer controls it".
    """
    x, targets, _env = collect(model, **kw)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = np.hstack([x, np.ones((len(x), 1))])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    cut = int(0.7 * len(x))
    tr, te = perm[:cut], perm[cut:]

    rows: List[Tuple[str, bool, int, float, float, float]] = []
    preds = {}
    for key in ("board", "board_opp", "control"):
        w = _ridge(x[tr], targets[key][tr])
        preds[key] = x[te] @ w

    for c in range(84):
        info = ts.MapData.get_country_info(c)
        def r2(key: str) -> float:
            y = targets[key][te][:, c]
            return float("nan") if y.var() < 1e-9 else 1.0 - ((y - preds[key][:, c]) ** 2).mean() / y.var()
        y = targets["control"][te][:, c] > 0.5
        if y.all() or not y.any():
            auc = float("nan")
        else:
            order = np.argsort(preds["control"][:, c])
            ranks = np.empty(len(order), dtype=float)
            ranks[order] = np.arange(len(order))
            n1, n0 = y.sum(), (~y).sum()
            auc = float((ranks[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
        rows.append((str(info["name"]), bool(info["battleground"]), int(info["region"]),
                     r2("board"), r2("board_opp"), auc))
    return rows


def probe(model: Any, **kw) -> Readout:
    x, targets, _env = collect(model, **kw)
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


def split_by_env(env_ids: np.ndarray, frac: float = 0.7,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test indices that keep every position of one game on the same side.

    The alternative, permuting positions, leaks: successive samples from one environment are the
    same game six steps apart, so a probe scored on them is partly scored on positions it has
    already fitted. Holding out environments makes "can this be read off the trunk" a question
    about the representation rather than about memorising a rollout.
    """
    envs = np.unique(env_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(envs))
    cut = max(1, int(frac * len(envs)))
    train_envs = set(envs[perm[:cut]].tolist())
    is_train = np.array([e in train_envs for e in env_ids], dtype=bool)
    return np.flatnonzero(is_train), np.flatnonzero(~is_train)


def _exact_accuracy(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-column share of positions where the rounded prediction is the exact influence.

    Influence is a non-negative integer, so the natural question about a linear read-out is not
    how much variance it explains but whether it names the number. R^2 0.4 on a country that is
    usually 0-4 can still be wrong about the value nearly every time.
    """
    hat = np.clip(np.rint(pred), 0.0, None)
    return (hat == true).mean(axis=0)


def _mode_baseline(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Accuracy of always predicting each column's most common training value.

    The floor the exact-match number has to clear. Most countries sit at 0 influence most of the
    time, so a probe that has learned nothing still scores high here, and without this column the
    raw accuracy reads far better than it is.
    """
    out = np.zeros(train.shape[1])
    for j in range(train.shape[1]):
        vals, counts = np.unique(train[:, j], return_counts=True)
        out[j] = (test[:, j] == vals[np.argmax(counts)]).mean()
    return out


MAX_LEVEL = 6


def multinomial_accuracy(x: np.ndarray, lev: np.ndarray, tr: np.ndarray, te: np.ndarray,
                         l2: float = 1e-5, steps: int = 1500,
                         lr: float = 0.3) -> np.ndarray:
    """Per-country exact-level accuracy from a linear **multinomial logistic** probe.

    This replaces a least-squares-onto-one-hot classifier that was wrong in a way worth writing
    down, because its failure looked exactly like a result. Regressing indicators masks
    intermediate classes once there are three or more (ESL 4.2) and shrinks rare ones out of the
    argmax altogether. Handed a *noiseless* influence/10 column -- the exact thing the engine puts
    in board slot 0 -- it scored 79% against a 72% constant baseline, closing 24% of the gap with
    the answer sitting in front of it. That ceiling, not the network, was what an earlier version
    of 21.12 reported as "the trunk recovers about a fifth".

    Softmax regression has no such masking: the fit is a proper likelihood, so a rare level is
    learned rather than suppressed. `x` is (N, D) for features shared across countries or
    (N, 84, D) for per-country ones; every country is fitted in one batched problem.

    Any probe used for this question must first clear `tests/training/test_state_readout_probe.py`,
    which feeds it the noiseless column and requires near-perfect recovery.

    The defaults are set by that test rather than by taste. `l2` has to be small -- a 7-way split
    of one scalar needs sharp boundaries, and at 1e-3 the penalty caps recovery of the *known*
    answer at 92% -- but the same test confirms pure noise still scores exactly the constant
    baseline at 1e-5, so the loose penalty buys recovery without buying false signal.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    per_country = x.ndim == 3
    n_country = lev.shape[1]

    xf = torch.tensor(x, dtype=torch.float32, device=dev)
    mu, sd = xf.mean(0, keepdim=True), xf.std(0, keepdim=True) + 1e-6
    xf = (xf - mu) / sd
    ones = torch.ones(*xf.shape[:-1], 1, device=dev)
    xf = torch.cat([xf, ones], dim=-1)
    d = xf.shape[-1]

    y = torch.tensor(lev, dtype=torch.long, device=dev)
    tr_t = torch.tensor(tr, dtype=torch.long, device=dev)
    te_t = torch.tensor(te, dtype=torch.long, device=dev)

    w = torch.zeros(n_country, d, MAX_LEVEL + 1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    eq = "ncd,cdk->nck" if per_country else "nd,cdk->nck"
    x_tr, x_te = xf[tr_t], xf[te_t]
    y_tr, y_te = y[tr_t], y[te_t]

    for _ in range(steps):
        opt.zero_grad()
        logits = torch.einsum(eq, x_tr, w)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, MAX_LEVEL + 1), y_tr.reshape(-1)) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = torch.einsum(eq, x_te, w).argmax(-1)
        return (pred == y_te).float().mean(0).cpu().numpy()


def collect_board_stages(model: Any, num_envs: int = 128, steps: int = 450, every: int = 6,
                         temperature: float = 0.1) -> Dict[str, Any]:
    """Per-country representations at four points, plus the influence to recover from each.

    21.12 measured what the 512-float trunk holds about each country but not *where* the rest
    went. This taps one rollout at four stages so the loss can be localised rather than inferred:

    * ``raw``    -- country i's own 26 observation floats. Upper bound and sanity check: slot 0 is
      literally ``my_influence / 10``, so anything well below 1.0 here means the probe is broken
      and nothing else on the ladder can be read. That check has already caught one broken probe.
    * ``gconv1`` -- after one graph convolution over the map adjacency, 64 floats for country i.
    * ``gconv2`` -- after the second, still per-country and still pre-pooling. This is the token
      an end-of-trunk attention block would attend over, so it decides whether such a block has
      anything left to find.
    * ``trunk``  -- the 512 floats every head actually reads.

    Only v2-shaped models have the graph layers; anything else raises rather than quietly
    returning a shorter ladder.
    """
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    if not hasattr(model, "gconv1") or not hasattr(model, "gconv2"):
        raise TypeError(f"{type(model).__name__} has no graph layers to tap")

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    taps: Dict[str, Any] = {}

    def hook(name: str) -> Any:
        def fn(_m: Any, _i: Any, out: Any) -> None:
            taps[name] = out.detach()
        return fn

    handles = [model.gconv1.register_forward_hook(hook("gconv1")),
               model.gconv2.register_forward_hook(hook("gconv2"))]

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=525_252)
    obs, masks, _ = env.reset_all()
    keys = ("raw", "gconv1", "gconv2", "trunk", "own", "opp")
    env_ids: List[int] = []
    acc: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    try:
        for step in range(steps):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)
                trunk = model.extract_features(obs_t)
                if isinstance(trunk, tuple):
                    trunk = trunk[0]
            if step % every == 0:
                bf = int(model.board_features)
                raw = obs_t[:, :84 * bf].view(-1, 84, bf).cpu().numpy()
                g1 = taps["gconv1"].cpu().numpy()
                g2 = taps["gconv2"].cpu().numpy()
                tk = trunk.cpu().numpy()
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
                    cs = [st.get_country(c) for c in range(84)]
                    env_ids.append(i)
                    acc["raw"].append(raw[i])
                    acc["gconv1"].append(g1[i])
                    acc["gconv2"].append(g2[i])
                    acc["trunk"].append(tk[i])
                    acc["own"].append(np.array(
                        [float(c.us_influence if mine else c.ussr_influence) for c in cs],
                        dtype=np.float32))
                    acc["opp"].append(np.array(
                        [float(c.ussr_influence if mine else c.us_influence) for c in cs],
                        dtype=np.float32))
            obs, masks, *_ = env.step(acts.cpu().numpy().astype(np.int64))
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, Any] = {k: np.asarray(v, dtype=np.float32)
                           for k, v in acc.items()}
    out["env_id"] = np.asarray(env_ids, dtype=np.int64)
    return out


def stage_gap_closed(data: Dict[str, Any], stage: str,
                     target: str = "own") -> Dict[str, np.ndarray]:
    """Per-country accuracy, constant baseline and gap closed, read from one stage.

    Same estimator as `influence_control_detail`, so the numbers are comparable across the ladder
    and against 21.12. A per-country stage is fitted from country i's own vector alone: the
    question is what *that token* carries about *that country*, not what the whole board carries.

    Accuracy and baseline are returned alongside the ratio because the ratio alone cannot be
    averaged. A country with 4% headroom divides by 0.04, so a couple of points of probe noise
    becomes -400%, and a mean over countries is then decided by whichever near-constant country
    happened to wobble. Aggregate with `weighted_gap_closed` instead.
    """
    y = data[target]
    # Whole games held out, not permuted positions. With a probe strong enough to fit, the
    # difference is not cosmetic: trunk battleground influence reads 25.8% under a permuted
    # split and 10.3% held out. An earlier check called the leak negligible, but it was run
    # with an estimator too weak to exploit it.
    tr, te = split_by_env(data["env_id"])

    lev = np.clip(y, 0, MAX_LEVEL).astype(int)
    base = _mode_baseline(y[tr], y[te])
    acc = tuned_multinomial_accuracy(data[stage].astype(np.float64), lev,
                                     data["env_id"], tr, te)

    gap = np.full(84, np.nan)
    for c in range(84):
        head = 1.0 - base[c]
        if head >= 0.02:
            gap[c] = (acc[c] - base[c]) / head
    return {"acc": acc, "base": base, "gap": gap}


def weighted_gap_closed(res: Dict[str, np.ndarray],
                        countries: Sequence[int] | None = None) -> float:
    """Total gap closed across countries: sum(acc - base) / sum(headroom).

    The aggregate that survives a near-constant country. Averaging the per-country ratio does
    not: it weights a country with 4% headroom the same as one with 71%, and the small one's
    ratio is mostly noise divided by a small number.
    """
    idx = list(range(84)) if countries is None else list(countries)
    num = float(sum(res["acc"][c] - res["base"][c] for c in idx))
    den = float(sum(1.0 - res["base"][c] for c in idx))
    return num / den if den > 1e-9 else float("nan")


L2_GRID: Tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def tuned_multinomial_accuracy(x: np.ndarray, lev: np.ndarray, env_ids: np.ndarray,
                               tr: np.ndarray, te: np.ndarray,
                               grid: Sequence[float] = L2_GRID) -> np.ndarray:
    """Per-country accuracy with the penalty chosen on a validation split, not fixed.

    A single `l2` cannot serve every rung of the ladder, and using one is how the `raw` rung came
    to read 85% when the answer is in the input. The penalty was tuned on a synthetic *one
    feature* problem; applied to 26, 64 and 512 features it is far too strong for the small ones.
    At 1e-7 the raw rung recovers 98% of the gap instead of 85%.

    Loosening it globally is not the fix either: the same sweep moved the 512-float trunk between
    8.9% and 28.0%, because a wide feature space with a weak penalty overfits the probe's own
    training split. One number per stage, each at its own best penalty, is the comparison that
    means "what a linear read-out can get out of this representation".

    The penalty is picked per country on a validation split carved out of the training games --
    never from the test games -- and the returned accuracy is always on the untouched test split.
    """
    inner_tr, val = split_by_env(env_ids[tr], frac=0.75, seed=1)
    tr_fit, tr_val = tr[inner_tr], tr[val]

    val_acc = np.stack([multinomial_accuracy(x, lev, tr_fit, tr_val, l2=g) for g in grid])
    best = np.argmax(val_acc, axis=0)

    test_acc = np.stack([multinomial_accuracy(x, lev, tr, te, l2=g) for g in grid])
    return test_acc[best, np.arange(test_acc.shape[1])]


def influence_control_detail(model: Any, hold_out_envs: bool = True,
                             **kw: Any) -> Dict[str, Any]:
    """Per-country: can the exact influence, and control, be read off the trunk?

    `battleground_detail` answers this in R^2 and AUC, which are the right scales for "is the
    signal there at all" and the wrong ones for "does it know the position". This adds the two
    numbers a person actually wants -- the share of positions where the rounded read-out is the
    *exact* influence, and the share where control is called correctly -- each against the
    best-constant baseline, since most countries are empty most of the time.
    """
    x, targets, env_ids = collect(model, **kw)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = np.hstack([x, np.ones((len(x), 1))])
    if hold_out_envs:
        tr, te = split_by_env(env_ids)
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(x))
        cut = int(0.7 * len(x))
        tr, te = perm[:cut], perm[cut:]

    preds = {k: x[te] @ _ridge(x[tr], targets[k][tr])
             for k in ("board", "board_opp", "control")}

    own_exact = _exact_accuracy(preds["board"], targets["board"][te])
    own_base = _mode_baseline(targets["board"][tr], targets["board"][te])
    lev_own = np.clip(targets["board"], 0, MAX_LEVEL).astype(int)
    lev_opp = np.clip(targets["board_opp"], 0, MAX_LEVEL).astype(int)
    own_clf = tuned_multinomial_accuracy(x, lev_own, env_ids, tr, te)
    opp_clf = tuned_multinomial_accuracy(x, lev_opp, env_ids, tr, te)
    opp_exact = _exact_accuracy(preds["board_opp"], targets["board_opp"][te])
    opp_base = _mode_baseline(targets["board_opp"][tr], targets["board_opp"][te])

    ctrl_true = targets["control"][te]
    ctrl_hat = (preds["control"] > 0.5).astype(float)
    ctrl_acc = (ctrl_hat == ctrl_true).mean(axis=0)
    ctrl_base = _mode_baseline(targets["control"][tr], ctrl_true)

    rows: List[Dict[str, Any]] = []
    for c in range(84):
        info = ts.MapData.get_country_info(c)
        y = targets["board"][te][:, c]
        r2 = float("nan") if y.var() < 1e-9 else \
            float(1.0 - ((y - preds["board"][:, c]) ** 2).mean() / y.var())
        rows.append({
            "name": str(info["name"]),
            "battleground": bool(info["battleground"]),
            "own_r2": r2,
            "own_exact": float(own_exact[c]),
            "own_clf": float(own_clf[c]),
            "own_base": float(own_base[c]),
            "opp_exact": float(opp_exact[c]),
            "opp_clf": float(opp_clf[c]),
            "opp_base": float(opp_base[c]),
            "control_acc": float(ctrl_acc[c]),
            "control_base": float(ctrl_base[c]),
        })

    bg = [r for r in rows if r["battleground"]]
    bg_idx = [c for c in range(84) if rows[c]["battleground"]]

    def m(rs: List[Dict[str, Any]], k: str) -> float:
        return float(np.mean([r[k] for r in rs]))

    # Weighted, not a mean of per-country ratios: a country with 4% headroom divides by 0.04 and
    # its ratio is mostly noise, which then decides the average. See `weighted_gap_closed`.
    inf_res = {"acc": own_clf, "base": own_base}
    ctl_res = {"acc": ctrl_acc, "base": ctrl_base}

    return {
        "rows": rows,
        "positions": int(len(x)),
        "test_positions": int(len(te)),
        "held_out_envs": bool(hold_out_envs),
        "all_own_exact": m(rows, "own_exact"), "all_own_clf": m(rows, "own_clf"),
        "all_own_base": m(rows, "own_base"),
        "all_control_acc": m(rows, "control_acc"), "all_control_base": m(rows, "control_base"),
        "bg_own_exact": m(bg, "own_exact"), "bg_own_clf": m(bg, "own_clf"),
        "bg_own_base": m(bg, "own_base"), "bg_own_r2": m(bg, "own_r2"),
        "bg_control_acc": m(bg, "control_acc"), "bg_control_base": m(bg, "control_base"),
        # The headline figures.
        "all_influence_gap": weighted_gap_closed(inf_res),
        "bg_influence_gap": weighted_gap_closed(inf_res, bg_idx),
        "all_control_gap": weighted_gap_closed(ctl_res),
        "bg_control_gap": weighted_gap_closed(ctl_res, bg_idx),
    }
