"""How much of the critic's zero-sum residual is information, and how much is error?

`v(s, US) + v(s, USSR)` is zero for a perfect-information zero-sum game, so it is tempting to read
the residual as pure model error. Twilight Struggle is not perfect information: each perspective
sees its own hand, so the two evaluations condition on *different information sets* and a nonzero
residual is expected. Holding a scoring card for a region you dominate genuinely raises your
value without lowering the opponent's estimate, because they cannot see it.

So the residual decomposes into information and error, and the two are separable by how they
respond to how much private information exists:

* **Hand size.** At the end of a turn hands are nearly empty and little is hidden; just after the
  deal both hold a full hand. If the residual is information, it should grow with hand size.
* **Hand asymmetry.** A residual driven by information should track how *unequal* the private
  information is -- one side holding a scoring card the other cannot see -- rather than sitting at
  a constant level.

What does not vary with either is the floor: that part is model error.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import ts_engine as ts

SCORING_CARDS = [c for c in range(1, 111) if ts.CardData.get_card_info(c)["is_scoring"]]
HANDS = {
    "US": (ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN),
    "USSR": (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN),
}


def _hand_stats(state: ts.GameState) -> Dict[str, Any]:
    sizes = {"US": 0, "USSR": 0}
    scoring = {"US": 0, "USSR": 0}
    for cid in range(1, 111):
        loc = state.get_card_location(cid)
        for side in ("US", "USSR"):
            if loc in HANDS[side]:
                sizes[side] += 1
                if cid in SCORING_CARDS:
                    scoring[side] += 1
    return {"sizes": sizes, "scoring": scoring}


def measure(model: Any, num_games: int = 256, temperature: float = 1.0,
            sample_every: int = 25, max_iters: int = 4000) -> Dict[str, Any]:
    """Residual against hand size and against scoring-card asymmetry."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=626_000)
    obs, masks, _ = env.reset_all()

    rows: List[Dict[str, Any]] = []
    for it in range(max_iters):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        mask_t = torch.from_numpy(np.asarray(masks)).to(device)
        with torch.no_grad():
            acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)

        if it % sample_every == 0:
            for i in range(0, num_games, 4):
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                vals = {}
                for tag, p in (("us", ts.Player.US), ("ussr", ts.Player.USSR)):
                    o = np.asarray(ts.extract_observation(st, p),
                                   dtype=np.float32).reshape(1, -1)
                    with torch.no_grad():
                        _, vw, _ = model(torch.from_numpy(o).to(device), None)
                    vals[tag] = float(vw.item())
                hs = _hand_stats(st)
                rows.append({
                    "residual": vals["us"] + vals["ussr"],
                    "hand_total": hs["sizes"]["US"] + hs["sizes"]["USSR"],
                    "scoring_gap": hs["scoring"]["US"] - hs["scoring"]["USSR"],
                    "scoring_total": hs["scoring"]["US"] + hs["scoring"]["USSR"],
                    "turn": int(st.turn),
                })

        obs, masks, _, dones, _ = env.step(acts.cpu().numpy().astype(np.int64))
        if it > 400 and all(bool(d) for d in dones):
            break
    return {"rows": rows}
