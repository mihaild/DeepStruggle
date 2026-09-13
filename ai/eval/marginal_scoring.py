"""Does the network know what a placement is *worth*, or only what the board is worth now?

The observation already hands the network the live net VP differential for every region, so a
probe asking it to recover regional score recovers an input. It would pass on a network that
copies six floats and understands nothing.

The question that separates the two is **counterfactual**: what would this region be worth if I
put one influence *here*? That is not in the observation, and it is not a smooth function of what
is. Regional scoring is a stack of thresholds -- presence, domination, control, battleground
counts, superpower adjacency -- so the marginal value of a placement is **zero almost everywhere
and several VP at the point that crosses one**. Adding to a country already safely held is worth
nothing; the influence that flips a battleground and tips domination can be worth five.

So a network hill-climbing the region's current number cannot produce this target, and one that
has computed the rules can. The engine supplies the answer key exactly, via
`Scoring::evaluate_region` on a mutated copy -- no rule is reimplemented here.

Three feature sets are compared, and the first is the control that makes the result mean
something:

* ``input`` -- the country's own raw observation slots plus the six regional VP scalars. This is
  "the magic number and the local facts". If the trunk cannot beat this, the network has added
  nothing.
* ``token`` -- the country's pre-pooling encoder output.
* ``trunk`` -- the 512 floats every head reads.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

import ts_engine as ts

#: `global_features[64..69]` carry the live net VP differential per region.
REGION_VP_SLICE = (64, 70)
N_REGIONS = 6


def _net_delta_for(state: ts.GameState, region: ts.Region, player: ts.Player) -> float:
    """Region VP differential, signed so positive is good for `player`."""
    summary = ts.Scoring.evaluate_region(state, region)
    net = float(summary.net_delta)          # US-positive by convention
    return net if player == ts.Player.US else -net


def marginal_placement_value(state: ts.GameState, player: ts.Player) -> np.ndarray:
    """(84,) VP change in a country's own region from one more influence there, for `player`.

    Uses the engine's own scorer on a mutated clone. The clone is per country and thrown away,
    so nothing here can disturb the state being probed.
    """
    out = np.zeros(84, dtype=np.float32)
    base: Dict[Any, float] = {}
    for c in range(84):
        region = ts.MapData.get_country_info(c)["region"]
        if region not in base:
            base[region] = _net_delta_for(state, region, player)

        cs = state.get_country(c)
        us, ussr = int(cs.us_influence), int(cs.ussr_influence)
        if player == ts.Player.US:
            us += 1
        else:
            ussr += 1

        nxt = state.clone()
        nxt.set_country(c, us, ussr)
        out[c] = _net_delta_for(nxt, region, player) - base[region]
    return out


def collect(model: Any, num_envs: int = 128, steps: int = 300, every: int = 6,
            temperature: float = 0.1) -> Dict[str, Any]:
    """Feature sets and the marginal-value answer key, from real decision positions."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    taps: Dict[str, Any] = {}

    def hook(name: str) -> Any:
        def fn(_m: Any, _i: Any, out: Any) -> None:
            taps[name] = out.detach()
        return fn

    tap_names = [n for n in ("gconv2", "gconv1", "board_fc") if hasattr(model, n)]
    if not tap_names:
        raise TypeError(f"{type(model).__name__} has no board encoder to tap")
    handle = getattr(model, tap_names[0]).register_forward_hook(hook("token"))

    env = TsVectorizedEnv(num_envs=num_envs, base_seed=717_000)
    obs, masks, _ = env.reset_all()
    acc: Dict[str, List[np.ndarray]] = {k: [] for k in ("input", "token", "trunk", "target")}
    env_ids: List[int] = []

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
                gl = obs_t[:, model.GLOBAL_OFFSET:model.GLOBAL_OFFSET + model.GLOBAL_SIZE]
                region_vp = gl[:, REGION_VP_SLICE[0]:REGION_VP_SLICE[1]].cpu().numpy()
                tok = taps["token"].cpu().numpy()
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
                    # The control feature set: this country's own slots, plus the six regional
                    # VP scalars broadcast to every country -- the "magic number" made available.
                    per_country_vp = np.repeat(region_vp[i][None, :], 84, axis=0)
                    acc["input"].append(np.hstack([raw[i], per_country_vp]))
                    acc["token"].append(tok[i])
                    acc["trunk"].append(tk[i])
                    acc["target"].append(marginal_placement_value(st, p))
                    env_ids.append(i)
            obs, masks, *_ = env.step(acts.cpu().numpy().astype(np.int64))
    finally:
        handle.remove()

    out: Dict[str, Any] = {k: np.asarray(v, dtype=np.float32) for k, v in acc.items()}
    out["env_id"] = np.asarray(env_ids, dtype=np.int64)
    return out


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    y = label > 0.5
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order))
    n1, n0 = float(y.sum()), float((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def probe(data: Dict[str, Any], stage: str) -> Dict[str, float]:
    """How well one feature set predicts the marginal value of a placement.

    Two readings, because the target is zero almost everywhere:

    * ``auc_matters`` -- ranking countries by whether placing there changes the region's score at
      all. This is the threshold question, and the one a copied magic number cannot answer.
    * ``r2`` -- how much of the magnitude is captured, over the countries where it is nonzero.
    """
    from ai.eval.state_readout import _ridge, split_by_env

    y = data["target"]
    tr, te = split_by_env(data["env_id"])
    x = data[stage].astype(np.float64)

    aucs: List[float] = []
    r2s: List[float] = []
    for c in range(84):
        xc = x[:, c, :] if x.ndim == 3 else x
        xc = (xc - xc.mean(0)) / (xc.std(0) + 1e-6)
        xc = np.hstack([xc, np.ones((len(xc), 1))])
        yc = y[:, c].astype(np.float64)
        if np.abs(yc).max() < 1e-9:
            continue
        w = _ridge(xc[tr], yc[tr].reshape(-1, 1))
        pred = (xc[te] @ w).ravel()
        matters = (np.abs(yc[te]) > 1e-9).astype(float)
        a = _auc(np.abs(pred), matters)
        if np.isfinite(a):
            aucs.append(a)
        var = yc[te].var()
        if var > 1e-9:
            r2s.append(1.0 - ((yc[te] - pred) ** 2).mean() / var)

    return {
        "auc_matters": float(np.mean(aucs)) if aucs else float("nan"),
        "r2": float(np.mean(r2s)) if r2s else float("nan"),
        "countries_scored": float(len(aucs)),
        "nonzero_share": float((np.abs(y) > 1e-9).mean()),
    }
