"""Which scoring cards get stranded, and in whose hand?

A player holding a scoring card at the end of a turn loses immediately. It is the most avoidable
loss in the game -- the card is in your hand, the rule is unconditional, and playing it is always
legal. So the interesting question is not the rate but the *pattern*: if some scoring cards are
stranded far more than others, that is a card-identity failure rather than a general one, and it
points at the same blind spot as the scoring-card probes -- the six cards share only two feature
vectors, so only the identity embedding can tell Asia Scoring from Europe Scoring.

Two readings:

* **per card** -- how often each scoring card is the one being held when the game ends that way,
  against how often it is in a hand at all. A card stranded in proportion to how often it is held
  is a general failure; one stranded far more often is specific.
* **per side** -- whether one superpower strands cards more than the other.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import ts_engine as ts

SCORING_CARDS = [c for c in range(1, 111) if ts.CardData.get_card_info(c)["is_scoring"]]

# A hand is two card locations, not one: a card the opponent has seen (played off a Space Race
# peek, a "reveal your hand" event) moves to the KNOWN half and stays in hand. Testing only
# `hand_of(player)` -- which is the UNKNOWN half -- silently misses those, and it missed the
# stranded card in 10 of the first 23 held-scoring endings measured.
HANDS = {
    "US": (ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN),
    "USSR": (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN),
}
ANY_HAND = HANDS["US"] + HANDS["USSR"]


def measure(model: Any, num_games: int = 512, temperature: float = 1.0,
            max_iters: int = 20_000) -> Dict[str, Any]:
    """Play out games, and for each held-scoring ending record what was stranded."""
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    device = next(model.parameters()).device
    model.eval()

    env = TsVectorizedEnv(num_envs=num_games, base_seed=313_000)
    obs, masks, _ = env.reset_all()
    finished = [False] * num_games

    stranded: Dict[int, int] = {c: 0 for c in SCORING_CARDS}
    held_exposure: Dict[int, int] = {c: 0 for c in SCORING_CARDS}
    by_side: Dict[str, int] = {"US": 0, "USSR": 0}
    by_turn: Dict[int, int] = {}
    endings = 0
    games = 0
    # Sampled rather than counted every step: exposure only needs to be proportional.
    sample_every = 12

    for it in range(max_iters):
        if all(finished):
            break
        with torch.no_grad():
            acts, *_ = model.sample_action(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(masks)).to(device), temperature=temperature)
        a = acts.cpu().numpy().astype(np.int64)

        if it % sample_every == 0:
            for i in range(num_games):
                if finished[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                for c in SCORING_CARDS:
                    if st.get_card_location(c) in ANY_HAND:
                        held_exposure[c] += 1

        # .clone() is load-bearing: runner.get_state returns a live view onto the batch
        # runner's slot, so an un-cloned handle follows the env through the step and
        # through auto-reset. Without it every held-scoring ending read back as turn 1
        # with a freshly dealt hand -- the reset game, not the one that just ended.
        prev = [env.runner.get_state(i).clone() if not finished[i] else None
                for i in range(num_games)]
        obs, masks, _, dones, info = env.step(a)

        for i, reason in enumerate(info["ending_reasons"]):
            if not reason or finished[i]:
                continue
            finished[i] = True
            games += 1
            if reason != "held_scoring":
                continue
            endings += 1
            st = prev[i]
            if st is None:
                continue
            # Who actually lost comes from `info`, not from re-testing the state: the env
            # evaluates `is_held_scoring_loss` on the post-step terminal state and auto-reset
            # has already rewound that state to a fresh turn 1 by the time step() returns.
            # `st` is the pre-step state, whose hands still hold the stranded card.
            for side in ("US", "USSR"):
                if not bool(info[f"held_scoring_{side.lower()}"][i]):
                    continue
                by_side[side] += 1
                t = int(st.turn)
                by_turn[t] = by_turn.get(t, 0) + 1
                for c in SCORING_CARDS:
                    if st.get_card_location(c) in HANDS[side]:
                        stranded[c] += 1

    total_exposure = sum(held_exposure.values()) or 1
    total_stranded = sum(stranded.values()) or 1
    rows: List[Dict[str, Any]] = []
    for c in SCORING_CARDS:
        rows.append({
            "card": str(ts.CardData.get_card_name(c)),
            "stranded": stranded[c],
            "stranded_share": stranded[c] / total_stranded,
            "exposure_share": held_exposure[c] / total_exposure,
            "over_representation": ((stranded[c] / total_stranded)
                                    / max(held_exposure[c] / total_exposure, 1e-9)),
        })
    return {"games": games, "held_scoring_endings": endings,
            "by_side": by_side, "by_turn": dict(sorted(by_turn.items())),
            "cards": rows}
