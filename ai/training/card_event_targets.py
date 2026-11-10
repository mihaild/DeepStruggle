"""Engine-labelled card x board targets (P30), shared by the probe tools and the auxiliary target.

Every card in the mover's hand is labelled with quantities that depend on that card *and* the board:

* **T2, Ops reach** (5 numbers), with the engine's own Ops arithmetic replicated in `effective_ops`
  (Red Scare/Purge, Containment/Brezhnev, the China Card in Asia, Vietnam Revolts; pinned to the
  engine's grants by `tests/training/test_card_event_aux.py`):
  the card's effective Ops; the countries the mover could bring under control with them (it can
  place there, does not control it, and the Ops to reach control -- two per point while the
  opponent controls the country, `Operations::get_influence_cost` -- fit the card's Ops *in that
  country*), the same for battlegrounds; and the best coup success chance over countries it can
  coup without losing the game (Cuban Missile Crisis, DEFCON 1), with SALT and Latin American
  Death Squads on the roll, overall and among battlegrounds. None for 0-Ops cards.
* **T3, the event can fire**: `CardHandlers.can_trigger_event` for the player it fires for -- the
  card's owner for a side card (an opponent's card played for Ops fires for the opponent), the
  mover for a neutral one.
* **T4, what the event does** (12 numbers): the event fired on a clone for that player, and the
  change read from the mover's observation before and after -- VP, DEFCON, the six regional
  scoring margins, battlegrounds controlled by each side, total influence of each side. An event
  that stops for choices (T5: Decolonization, Junta, the wars, ...) is played out on the clone,
  each choice taken greedily by whoever makes it for their own best `_score`, until the card is no
  longer the active one; die rolls are left to the engine. Averaged over the given engine seeds.

`research/log/P30_card_board_targets.md` has what these showed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

N_COUNTRIES, BOARD_W, N_CARDS, CARD_W = 84, 26, 110, 14
CARD_OFFSET = N_COUNTRIES * BOARD_W
GLOBAL_OFFSET = CARD_OFFSET + N_CARDS * CARD_W
MY_HAND = 1
DECISION_TYPE = GLOBAL_OFFSET + 72
CHINA = 6
MAX_HAND = 12
T2_NAMES = ("effective Ops", "countries to control", "battlegrounds to control", "best coup chance",
            "best battleground coup chance")
N_T2 = len(T2_NAMES)
T4_NAMES = ("VP", "DEFCON", "margin Europe", "margin Asia", "margin Middle East", "margin Africa",
            "margin Central America", "margin South America", "my battlegrounds", "their battlegrounds",
            "my influence", "their influence")

_INFO = {c: ts.CardData.get_card_info(c) for c in range(1, N_CARDS + 1)}
OPS = np.array([0] + [int(_INFO[c]["ops"]) for c in range(1, N_CARDS + 1)], dtype=np.int64)
SIDE = {c: str(_INFO[c]["side"]) for c in range(1, N_CARDS + 1)}
SCORING = sorted(c for c in range(1, N_CARDS + 1) if _INFO[c]["is_scoring"])

#: The auxiliary target's 17 outputs per card, T2 then T4, and their scales: the mean and standard
#: deviation over held cards of self-play at E6-12-44@560M / E6-03-44@560M, labelled by this module
#: (`data/datasets/card_board/e6_12_03_560M_v2.npz`, 2026-09-30). Fixed rather than running, so a
#: checkpoint's head means the same thing whenever it was trained.
AUX_DIM = 17
AUX_MEAN = np.array([2.441, 18.349, 3.476, 0.710, 0.232,
                     -0.650, -0.015, -0.099, -0.045, -0.041, -0.103, -0.057, -0.027, -0.131, -0.002, 0.089, 0.504],
                    dtype=np.float32)
AUX_SCALE = np.array([0.989, 9.794, 2.960, 0.408, 0.382,
                      3.996, 0.350, 1.349, 1.007, 0.896, 1.080, 0.823, 0.565, 0.630, 0.508, 1.592, 1.674],
                     dtype=np.float32)


# effect_bits (engine/include/ts/constants.hpp)
CONTAINMENT, PURGE_US, PURGE_USSR, VIETNAM = 1 << 6, 1 << 7, 1 << 8, 1 << 9
CMC_US, CMC_USSR, SALT, BREZHNEV = 1 << 11, 1 << 12, 1 << 16, 1 << 18
DEATH_SQUADS_US, DEATH_SQUADS_USSR = 1 << 22, 1 << 23

_MAP = [ts.MapData.get_country_info(c) for c in range(N_COUNTRIES)]
REGION = np.array([int(m["region"]) for m in _MAP])
IN_SE_ASIA = np.array([bool(m["in_southeast_asia"]) for m in _MAP])
STABILITY = np.array([int(m["stability"]) for m in _MAP])
BATTLEGROUND = np.array([bool(m["battleground"]) for m in _MAP])
ASIA, CENTRAL_AMERICA, SOUTH_AMERICA = 1, 4, 5


def effective_ops(st: "ts.GameState", card: int, player: "ts.Player", in_asia: bool = False,
                  in_southeast_asia: bool = False) -> int:
    """`Operations::combine_ops` (engine/src/ops.cpp), exactly: a card's Ops for `player` with every
    modifier in force, for an action in Asia / Southeast Asia or (both False) elsewhere."""
    base = int(OPS[card])
    if base == 0:
        return 0
    us = player == ts.Player.US
    raise_ = bool(st.has_flag(CONTAINMENT if us else BREZHNEV))
    purge = bool(st.has_flag(PURGE_US if us else PURGE_USSR))
    ops = base + (1 if raise_ else 0) - (1 if purge else 0)
    cap = 4
    if card == CHINA and in_asia:
        ops += 1
        cap = 5
    if raise_:
        ops = min(ops, max(cap, base))
    if not us and in_southeast_asia and bool(st.has_flag(VIETNAM)):
        ops += 1
    return max(ops, 1)


def ops_to_control(mine: int, theirs: int, stability: int) -> int:
    """Ops to bring a country under control by placing, one point at a time, at two Ops a point while
    the opponent controls it (`Operations::get_influence_cost`)."""
    ops = 0
    while mine - theirs < stability:
        ops += 2 if theirs - mine >= stability else 1
        mine += 1
    return ops


def coup_roll_mod(st: "ts.GameState", player: "ts.Player", country: int) -> int:
    """The coup roll's modifiers (`Operations::execute_coup`): Death Squads in Latin America, SALT."""
    mod = 0
    if REGION[country] in (CENTRAL_AMERICA, SOUTH_AMERICA):
        us = player == ts.Player.US
        if st.has_flag(DEATH_SQUADS_US):
            mod += 1 if us else -1
        if st.has_flag(DEATH_SQUADS_USSR):
            mod += -1 if us else 1
    if st.has_flag(SALT):
        mod -= 1
    return mod


def _summary(obs: np.ndarray) -> np.ndarray:
    """The quantities T4 differences, from one observation (mover's side)."""
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    bg = b[:, 4] > 0.5
    return np.concatenate([
        (obs[GLOBAL_OFFSET + 64:GLOBAL_OFFSET + 70] * 20.0),
        [float(((b[:, 5] > 0.5) & bg).sum()), float(((b[:, 6] > 0.5) & bg).sum()),
         float((b[:, 0] * 10.0).sum()), float((b[:, 1] * 10.0).sum())]])


def t2(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, card: int) -> Optional[np.ndarray]:
    if int(OPS[card]) <= 0:
        return None
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    my_inf = np.rint(b[:, 0] * 10.0).astype(int)
    their_inf = np.rint(b[:, 1] * 10.0).astype(int)
    can_place = b[:, 19] > 0.5
    mine = b[:, 5] > 0.5
    can_coup = b[:, 21] > 0.5
    # a coup that loses the game is no option: DEFCON 1 (board[7]) or the Cuban Missile Crisis
    cmc = bool(st.has_flag(CMC_US if mover == ts.Player.USSR else CMC_USSR))
    can_coup &= ~(b[:, 7] > 0.5)
    if cmc:
        can_coup[:] = False
    reach = np.zeros(N_COUNTRIES, dtype=bool)
    p = np.zeros(N_COUNTRIES)
    for c in range(N_COUNTRIES):
        if not (can_place[c] or can_coup[c]):
            continue
        ops_c = effective_ops(st, card, mover, REGION[c] == ASIA, bool(IN_SE_ASIA[c]))
        if can_place[c] and not mine[c]:
            reach[c] = ops_to_control(int(my_inf[c]), int(their_inf[c]), int(STABILITY[c])) <= ops_c
        if can_coup[c]:
            need = 2 * int(STABILITY[c]) - ops_c - coup_roll_mod(st, mover, c)   # die must exceed this
            p[c] = min(max(6 - need, 0), 6) / 6.0
    best = float(p[can_coup].max()) if can_coup.any() else 0.0
    bgc = can_coup & BATTLEGROUND
    best_bg = float(p[bgc].max()) if bgc.any() else 0.0
    return np.array([float(effective_ops(st, card, mover)), float(reach.sum()),
                     float((reach & BATTLEGROUND).sum()), best, best_bg], dtype=np.float32)


@dataclass(frozen=True)
class _Ctx:
    """The decision a state waits on, compared before and after an event fires."""
    decision_type: int
    decision_player: int
    pending_op_card: int
    remaining_steps: int


def _ctx(st: "ts.GameState") -> _Ctx:
    c = st.ctx()
    return _Ctx(int(c.decision_type), int(c.decision_player), int(c.pending_op_card), int(c.remaining_steps))


#: `_score`'s weights: a won or lost game, VP, the regional scoring margins, battleground balance,
#: influence balance. A fixed measure, so "the best choice" is defined without a network.
SCORE_WEIGHTS = (100.0, 1.0, 0.5, 1.0, 0.2)
MAX_EVENT_STEPS = 60


def _score(x: "ts.GameState", player: "ts.Player") -> float:
    """How good the board is for `player`, by SCORE_WEIGHTS."""
    w_end, w_vp, w_margin, w_bg, w_inf = SCORE_WEIGHTS
    sign = 1.0 if player == ts.Player.US else -1.0
    if ts.Engine.is_terminal(x):
        u = float(ts.Engine.get_terminal_utility(x)) * sign
        return w_end * (1.0 if u > 0 else -1.0 if u < 0 else 0.0)
    s = _summary(np.asarray(ts.extract_observation(x, player)))
    return (w_vp * sign * float(int(x.victory_points)) + w_margin * float(s[:6].sum())
            + w_bg * float(s[6] - s[7]) + w_inf * float(s[8] - s[9]))


def _active(x: "ts.GameState", card: int) -> float:
    p = int(x.ctx().decision_player)
    if p == 0:
        return -1.0
    o = np.asarray(ts.extract_observation(x, ts.Player(p)))
    return float(o[CARD_OFFSET + (card - 1) * CARD_W + 13])


def resolve_choices(x: "ts.GameState", card: int) -> bool:
    """Play out an event that stopped for choices, in place: each choice taken greedily by the
    player making it for their own best `_score`; die rolls and other decisions with no player are
    left to the engine. Done when `card` is no longer the active card. False if it does not finish
    within MAX_EVENT_STEPS."""
    for _ in range(MAX_EVENT_STEPS):
        if ts.Engine.is_terminal(x):
            return True
        p = int(x.ctx().decision_player)
        legal = [int(a) for a in np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(x, False)))]
        if not legal:
            return False
        # An event fired outside its normal place in the turn can leave the mask offering an action
        # the state machine refuses, so every step is a try_step: a refused option is skipped, and
        # an event with no accepted option is left unlabelled.
        if p == 0:                                    # a die roll: the engine's own RNG
            if not any(ts.Engine.try_step_flat(x, a, True, False) for a in legal):
                return False
            continue
        if _active(x, card) < 0.99:
            return True
        player = ts.Player(p)
        best_a: Optional[int] = None
        best_v = -np.inf
        for a in legal:
            y = x.clone()
            if not ts.Engine.try_step_flat(y, a, True, False):
                continue
            v = _score(y, player) if len(legal) > 1 else 0.0
            if best_a is None or v > best_v:
                best_a, best_v = a, v
        if best_a is None or not ts.Engine.try_step_flat(x, best_a, True, False):
            return False
    return False


def t3_t4(st: "ts.GameState", mover: "ts.Player", card: int, base_obs: np.ndarray,
          seeds: Sequence[int]) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    if card == CHINA:
        return None, None
    side = SIDE[card]
    fire = ts.Player.US if side == "US" else ts.Player.USSR if side == "USSR" else mover
    if not bool(ts.CardHandlers.can_trigger_event(st, card, fire)):
        return False, None
    sign = 1.0 if mover == ts.Player.US else -1.0
    base = _summary(base_obs)
    c0 = _ctx(st)
    deltas: List[np.ndarray] = []
    for s in seeds:
        x = st.clone()
        x.rng_state = int(s)
        vp0, dc0 = int(x.victory_points), int(x.defcon)
        ts.CardHandlers.trigger_event(x, card, fire)
        c1 = _ctx(x)
        if c1 != c0 and c1.decision_type != 0 and not resolve_choices(x, card):
            return True, None                         # did not finish: not labelled
        after = _summary(np.asarray(ts.extract_observation(x, mover)))
        deltas.append(np.concatenate([[sign * (int(x.victory_points) - vp0), float(int(x.defcon) - dc0)],
                                      after - base]).astype(np.float32))
    return True, np.mean(deltas, 0)


def held_cards(obs: np.ndarray) -> List[int]:
    rows = obs[CARD_OFFSET:GLOBAL_OFFSET].reshape(N_CARDS, CARD_W)
    return [c for c in range(1, N_CARDS + 1) if rows[c - 1, MY_HAND] > 0.5][:MAX_HAND]


def label(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, seeds: Sequence[int]) -> Dict[str, np.ndarray]:
    """All targets for the mover's held cards, padded to MAX_HAND (card id 0 = none)."""
    cards = np.zeros(MAX_HAND, dtype=np.int16)
    y2 = np.zeros((MAX_HAND, N_T2), dtype=np.float32)
    y3 = np.zeros(MAX_HAND, dtype=np.float32)
    y4 = np.zeros((MAX_HAND, 12), dtype=np.float32)
    m2 = np.zeros(MAX_HAND, dtype=bool)
    m3 = np.zeros(MAX_HAND, dtype=bool)
    m4 = np.zeros(MAX_HAND, dtype=bool)
    for j, c in enumerate(held_cards(obs)):
        cards[j] = c
        v2 = t2(st, mover, obs, c)
        if v2 is not None:
            y2[j], m2[j] = v2, True
        can, v4 = t3_t4(st, mover, c, obs, seeds)
        if can is not None:
            y3[j], m3[j] = float(can), True
        if v4 is not None:
            y4[j], m4[j] = v4, True
    return {"cards": cards, "y2": y2, "y3": y3, "y4": y4, "m2": m2, "m3": m3, "m4": m4}


def aux_label(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, seed: int
              ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The auxiliary target for one position: (cards (12,), standardised targets (12, 16), mask
    (12, 16)), or None when no held card has any label. One engine seed, for speed."""
    lab = label(st, mover, obs, (seed,))
    mask = np.concatenate([np.repeat(lab["m2"][:, None], N_T2, 1), np.repeat(lab["m4"][:, None], 12, 1)], 1)
    if not mask.any():
        return None
    y = (np.concatenate([lab["y2"], lab["y4"]], 1) - AUX_MEAN) / AUX_SCALE
    return lab["cards"], np.where(mask, y, 0.0).astype(np.float32), mask
