"""Does the critic predict the actual game outcome better than something trivial?

`explained_variance` cannot answer this. The GAE return is built as `G = A + V`, so `G - V = A`
exactly and `EV = 1 - Var(A)/Var(G)`: a critic whose advantages collapse scores near 1.0 by
construction, and the 0.996 measured at 160M is the advantage collapse restated, not independent
evidence of a good critic.

The non-circular question is whether `v_win(s)` predicts who actually wins that game, against
baselines that require no learning:

* **base rate** -- always predict the side that wins more often. Accuracy = the base rate itself.
* **VP sign** -- predict whoever is ahead on the victory-point track right now.

A critic that cannot beat the VP track has learned nothing the score does not already say.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import ts_engine as ts


def measure(model: Any, num_games: int = 512, temperature: float = 1.0,
            sample_every: int = 20, max_iters: int = 6000,
            seed: int = 717_000) -> Dict[str, Any]:
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=seed)
    obs, masks, _ = env.reset_all()

    # Samples are held per env until that env's game ends and its winner is known.
    pending: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(num_games)}
    done_flag = [False] * num_games
    rows: List[Dict[str, float]] = []

    for it in range(max_iters):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(masks)).to(device)
        with torch.no_grad():
            acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)

        if it % sample_every == 0:
            for i in range(num_games):
                if done_flag[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                o = np.asarray(ts.extract_observation(st, ts.Player.US),
                               dtype=np.float32).reshape(1, -1)
                with torch.no_grad():
                    _, vw, _ = model(torch.from_numpy(o).to(device), None)
                pending[i].append({
                    "v_us": float(vw.item()),
                    "vp": float(st.victory_points),
                    "turn": float(st.turn),
                })

        obs, masks, _, dones, info = env.step(acts.cpu().numpy().astype(np.int64))

        for i, reason in enumerate(info["ending_reasons"]):
            if not reason or done_flag[i]:
                continue
            done_flag[i] = True
            vp_final = float(info["victory_points"][i])
            # +1 if the US won this game, -1 if the USSR did; draws dropped.
            if vp_final == 0:
                pending[i] = []
                continue
            label = 1.0 if vp_final > 0 else -1.0
            for row in pending[i]:
                row["label"] = label
                rows.append(row)
            pending[i] = []

        if all(done_flag):
            break

    v = np.array([r["v_us"] for r in rows])
    vp = np.array([r["vp"] for r in rows])
    y = np.array([r["label"] for r in rows])
    turn = np.array([r["turn"] for r in rows])

    us_rate = float((y > 0).mean()) if len(y) else float("nan")
    base_rate = max(us_rate, 1.0 - us_rate)

    def acc(pred: np.ndarray, target: np.ndarray | None = None) -> float:
        # Ties count as half, so a predictor that abstains gets no free credit.
        t = y if target is None else target
        if pred.size == 0:
            return float("nan")
        return float((np.sign(pred) == t).mean() + 0.5 * (np.sign(pred) == 0).mean())

    return {
        "n": len(rows),
        "us_win_rate": us_rate,
        "acc_base_rate": base_rate,
        "acc_vp_sign": acc(vp),
        "acc_critic": acc(v),
        "corr_critic": float(np.corrcoef(v, y)[0, 1]) if len(v) > 1 else float("nan"),
        "corr_vp": float(np.corrcoef(vp, y)[0, 1]) if len(vp) > 1 else float("nan"),
        "by_turn": {
            int(t): {
                "n": int((turn == t).sum()),
                "acc_critic": acc(v[turn == t], y[turn == t]),
                "acc_vp": acc(vp[turn == t], y[turn == t]),
                "base": float(max((y[turn == t] > 0).mean(), 1 - (y[turn == t] > 0).mean())),
            }
            for t in sorted(set(int(x) for x in turn)) if (turn == t).sum() >= 50
        },
    }
