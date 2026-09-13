"""The same card, the same board, played for its Event or for Operations. Where does it aim?

This is the sharpest available test of whether the network prices a country *for the operation it
is performing*, because the two branches differ in exactly one decision and nothing else -- same
position, same card, same Ops value, same legal-target machinery.

The rule that makes the two branches want different countries is `get_influence_cost`: placing
influence costs **2 Ops per point in a country the opponent controls** and 1 elsewhere. Event
placement does not go through that -- `place_influence` never consults the cost -- so:

* **Ordinary Ops placement should avoid opponent-controlled countries.** Breaking control by
  direct placement is the most expensive thing Ops can buy.
* **Free-placement events should prefer them.** Decolonization, Colonial Rear Guards and Ussuri
  River Skirmish place at no premium, so eroding a country the opponent holds is a bargain there
  and a waste with plain Ops.
* **Removal events should prefer them most of all.** The Voice of America takes Soviet influence
  away; removing from a controlled country breaks control at a price direct placement could not
  match.

A network with one static country preference scores the same on both branches. The number that
matters is therefore not either branch alone but the **gap** between them.

"Controlled but not overcontrolled" is reported separately because it is the actionable subset: a
country the opponent holds by twenty points is not a target for anybody.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts
from ai.eval.positions import PLAY_MODE_ACTION, card_action, legal_mask

#: (card id, the side that holds it). All own-side, so playing for Ops does not fire the event
#: and the two branches really are the same card.
DEFAULT_CARDS: Tuple[Tuple[int, ts.Player], ...] = (
    (74, ts.Player.US),      # The Voice of America -- removes USSR influence outside Europe
    (76, ts.Player.US),      # Ussuri River Skirmish -- free placement in Asia
    (63, ts.Player.US),      # Colonial Rear Guards -- free placement in Africa/SE Asia
    (30, ts.Player.USSR),    # Decolonization -- free placement in Africa/SE Asia
)

OP_MODE_INFLUENCE = 116      # flat action for SELECT_OP_MODE / INFLUENCE
COUNTRY_OFFSET = 119


@dataclass
class BranchStats:
    nodes: int = 0
    controlled_mass: float = 0.0
    reachable_mass: float = 0.0
    legal_controlled: float = 0.0
    legal_reachable: float = 0.0

    def summary(self) -> Dict[str, float]:
        n = max(self.nodes, 1)
        return {
            "nodes": float(self.nodes),
            "controlled": self.controlled_mass / n,
            "controlled_if_uniform": self.legal_controlled / n,
            "reachable": self.reachable_mass / n,
            "reachable_if_uniform": self.legal_reachable / n,
        }


def _country_flags(state: ts.GameState, me: ts.Player,
                   legal: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per legal country: opponent-controlled, and controlled-but-still-reachable."""
    opp = ts.Player.USSR if me == ts.Player.US else ts.Player.US
    controlled = np.zeros(len(legal))
    reachable = np.zeros(len(legal))
    for k, c in enumerate(legal):
        if not ts.Scoring.is_controlled_by(state, c, opp):
            continue
        controlled[k] = 1.0
        cs = state.get_country(c)
        mine = float(cs.us_influence if me == ts.Player.US else cs.ussr_influence)
        theirs = float(cs.ussr_influence if me == ts.Player.US else cs.us_influence)
        stability = float(ts.MapData.get_country_info(c)["stability"])
        # Held by no more than one point beyond the control threshold: still worth contesting.
        reachable[k] = 1.0 if (theirs - mine) <= stability + 1 else 0.0
    return controlled, reachable


def _step(state: ts.GameState, action: int) -> ts.GameState:
    nxt = state.clone()
    ts.Engine.step_flat(nxt, action)
    return nxt


def _to_point_node(state: ts.GameState, card: int, mode: str,
                   max_steps: int = 6) -> Optional[ts.GameState]:
    """Play `card` for `mode`, advancing past any forced choice, and stop at the country node.

    Returns None when this branch does not reach a country choice in this position -- an event
    with no legal target, or Ops that cannot place. Those positions are dropped from *both*
    branches by the caller, so the pairing stays exact.
    """
    mask = legal_mask(state)
    if mask[card_action(card)] != 1:
        return None
    st = _step(state, card_action(card))
    if st.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return None
    if legal_mask(st)[PLAY_MODE_ACTION[mode]] != 1:
        return None
    st = _step(st, PLAY_MODE_ACTION[mode])

    for _ in range(max_steps):
        dt = st.ctx().decision_type
        if dt == ts.DecisionType.POINT_NODE:
            return st
        if dt == ts.DecisionType.SELECT_OP_MODE:
            if legal_mask(st)[OP_MODE_INFLUENCE] != 1:
                return None
            st = _step(st, OP_MODE_INFLUENCE)
            continue
        # Any other forced node (a timing branch, a die roll) -- take the first legal action.
        m = legal_mask(st)
        idx = int(np.flatnonzero(m)[0]) if m.any() else -1
        if idx < 0:
            return None
        st = _step(st, idx)
    return None


def _distribution(model: Any, state: ts.GameState, temperature: float = 0.1
                  ) -> Optional[Tuple[np.ndarray, List[int]]]:
    """The model's probability over the legal country targets at this node."""
    import torch

    mask = legal_mask(state)
    legal = [c for c in range(84) if mask[COUNTRY_OFFSET + c]]
    if len(legal) < 2:
        return None
    device = next(model.parameters()).device
    obs = np.asarray(ts.extract_observation(state, state.ctx().decision_player),
                     dtype=np.float32)
    with torch.no_grad():
        logits, _v, _vp = model(torch.from_numpy(obs).unsqueeze(0).to(device),
                                torch.from_numpy(mask).unsqueeze(0).to(device))
        p = torch.softmax(logits[0] / max(temperature, 1e-4), dim=-1).cpu().numpy()
    sub = np.array([p[COUNTRY_OFFSET + c] for c in legal], dtype=np.float64)
    total = sub.sum()
    if total <= 0:
        return None
    return sub / total, legal


def compare(model: Any, states: Sequence[ts.GameState],
            cards: Sequence[Tuple[int, ts.Player]] = DEFAULT_CARDS,
            temperature: float = 0.1) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each card, where the model aims when it plays it for Event versus for Operations.

    Only positions where *both* branches reach a country choice are counted, so every number in a
    card's row comes from the same set of boards.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for card, holder in cards:
        name = str(ts.CardData.get_card_name(card))
        ev, op = BranchStats(), BranchStats()
        for base in states:
            st = base.clone()
            st.set_card_location(card, ts.hand_of(holder))
            branches = {m: _to_point_node(st, card, m) for m in ("event", "ops")}
            if any(v is None for v in branches.values()):
                continue
            dists = {}
            ok = True
            for m, node in branches.items():
                assert node is not None
                d = _distribution(model, node, temperature)
                if d is None:
                    ok = False
                    break
                dists[m] = (node, *d)
            if not ok:
                continue
            for m, acc in (("event", ev), ("ops", op)):
                node, probs, legal = dists[m]
                controlled, reachable = _country_flags(node, holder, legal)
                acc.nodes += 1
                acc.controlled_mass += float((probs * controlled).sum())
                acc.reachable_mass += float((probs * reachable).sum())
                acc.legal_controlled += float(controlled.mean())
                acc.legal_reachable += float(reachable.mean())
        if ev.nodes:
            e, o = ev.summary(), op.summary()
            out[name] = {"event": e, "ops": o, "gap": {
                "controlled": e["controlled"] - o["controlled"],
                "reachable": e["reachable"] - o["reachable"],
            }}
    return out
