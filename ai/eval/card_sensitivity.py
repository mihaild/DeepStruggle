"""Does the critic read the hand at all?

Replace the acting player's hand with random cards from the draw deck and see how far the value
moves. On its own |dv| is uninterpretable -- 0.01 is small or large depending on what the critic
does with anything -- so every run carries two yardsticks measured on the same states:

* **null**: perturb nothing. Must be exactly 0, or the measurement is broken.
* **board**: hand the acting player four influence in a European battleground, or take four away.
  That is a large, unambiguous positional change, and it sets the scale |dv| should be read
  against.

Only the value head is read and the engine is never stepped from the perturbed state, so the
mutation has to be consistent for the *observation* and nothing else. The probe checks that the
card block of the observation actually changed, because a perturbation that silently did nothing
would report perfect insensitivity.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import ts_engine as ts

SCORING_CARDS = [c for c in range(1, 111) if ts.CardData.get_card_info(c)["is_scoring"]]
HANDS = {
    ts.Player.US: (ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN),
    ts.Player.USSR: (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN),
}
#: West Germany, France, Italy -- the battlegrounds the Europe-control games turn on.
YARDSTICK_COUNTRIES = [7, 8, 10]
#: Card-feature block: 110 cards x 14 features, after the 84 x 26 board block.
BOARD_FLOATS = 84 * 26
CARD_FLOATS = 110 * 14


def _hand(state: ts.GameState, player: Any) -> List[int]:
    return [c for c in range(1, 111) if state.get_card_location(c) in HANDS[player]]


def _deck(state: ts.GameState) -> List[int]:
    return [c for c in range(1, 111)
            if state.get_card_location(c) == ts.CardLocation.DRAW_DECK]


def _value(model: Any, state: ts.GameState, player: Any) -> float:
    import torch

    device = next(model.parameters()).device
    obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        _, v_win, _ = model(torch.from_numpy(obs).to(device), None)
    return float(v_win.item())


def _card_block(state: ts.GameState, player: Any) -> np.ndarray:
    obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32)
    return obs[BOARD_FLOATS:BOARD_FLOATS + CARD_FLOATS].copy()


def measure(model: Any, num_games: int = 128, temperature: float = 1.0,
            sample_every: int = 20, max_iters: int = 4000,
            seed: int = 808_000) -> Dict[str, Any]:
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    check_obs_width(model)
    model.eval()
    rng = np.random.default_rng(seed)

    env = TsVectorizedEnv(num_envs=num_games, base_seed=seed)
    obs, masks, _ = env.reset_all()

    rows: List[Dict[str, float]] = []
    inert = 0

    for it in range(max_iters):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(next(model.parameters()).device)
        mask_t = torch.from_numpy(np.asarray(masks)).to(next(model.parameters()).device)
        with torch.no_grad():
            acts, *_ = model.sample_action(obs_t, mask_t, temperature=temperature)

        if it % sample_every == 0:
            for i in range(0, num_games, 4):
                live = env.runner.get_state(i)
                if ts.Engine.is_terminal(live):
                    continue
                p = live.ctx().decision_player
                if p == ts.Player.NONE:
                    continue
                base_state = live.clone()
                hand = _hand(base_state, p)
                deck = _deck(base_state)
                if len(hand) < 2 or len(deck) < len(hand):
                    continue

                v_base = _value(model, base_state, p)
                block_base = _card_block(base_state, p)

                # --- random hand -------------------------------------------------------
                swapped = base_state.clone()
                picks = rng.choice(len(deck), size=len(hand), replace=False)
                for c in hand:
                    swapped.set_card_location(c, ts.CardLocation.DRAW_DECK)
                for k, c in zip(picks, hand):
                    swapped.set_card_location(int(deck[int(k)]), base_state.get_card_location(c))
                if np.array_equal(_card_block(swapped, p), block_base):
                    inert += 1
                    continue
                v_rand = _value(model, swapped, p)

                # --- null control ------------------------------------------------------
                v_null = _value(model, base_state.clone(), p)

                # --- board yardstick: +4 influence in a European battleground ----------
                boosted = base_state.clone()
                cid = int(YARDSTICK_COUNTRIES[rng.integers(len(YARDSTICK_COUNTRIES))])
                c_now = boosted.get_country(cid)
                us_i, su_i = int(c_now.us_influence), int(c_now.ussr_influence)
                if p == ts.Player.US:
                    boosted.set_country(cid, us_i + 4, su_i)
                else:
                    boosted.set_country(cid, us_i, su_i + 4)
                v_board = _value(model, boosted, p)

                # --- hand made maximally different: all scoring cards in hand ----------
                scored = base_state.clone()
                loc = base_state.get_card_location(hand[0])
                for c in SCORING_CARDS:
                    scored.set_card_location(c, loc)
                v_scoring = _value(model, scored, p)

                rows.append({
                    "v_base": v_base,
                    "d_null": abs(v_null - v_base),
                    "d_random_hand": abs(v_rand - v_base),
                    "d_all_scoring": abs(v_scoring - v_base),
                    "d_board_plus4": abs(v_board - v_base),
                    "hand_size": float(len(hand)),
                })

        obs, masks, _, dones, _ = env.step(acts.cpu().numpy().astype(np.int64))
        if it > 400 and all(bool(d) for d in dones):
            break

    return {"rows": rows, "inert_perturbations": inert}
