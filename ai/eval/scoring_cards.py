"""Does the network know which region a scoring card scores?

This is the one association the observation cannot supply. The card block encodes a card's
*properties* -- Ops, side, era, one-time, is-scoring -- and its location, but not which card it
is. Asia Scoring and Europe Scoring therefore share a feature vector exactly, so a network can
only tell them apart through the identity embedding indexed by position. Everything here is a
direct test of what that embedding bought.

Two questions, both keyed to the engine's own scorer:

**Ordering.** Holding several scoring cards, does the model prefer to play the one that pays
most? `Scoring::evaluate_region` gives the exact VP each would yield in this position, so the
model's preference over those cards can be ranked against the truth.

**Location.** Moving Europe Scoring between the draw deck, the player's hand, the opponent's hand
and the discard pile changes nothing about Europe on the board -- but it changes a great deal
about whether investing in Europe pays. If the network associates the card with the region, its
appetite for placing influence in Europe should move when the card moves, and specifically in
Europe rather than everywhere.

The second is reported as a difference in differences: how much moving the card shifts placement
into *its own* region, minus how much it shifts placement into the others. That subtracts any
generic "a scoring card is in hand, play differently" reflex, which needs no card identity at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts
from ai.eval.event_vs_ops import COUNTRY_OFFSET, _step, _to_point_node
from ai.eval.positions import legal_mask

#: scoring card -> the region it scores. Not derivable from the observation.
SCORING_CARDS: Tuple[Tuple[int, Any], ...] = (
    (1, ts.Region.ASIA),
    (2, ts.Region.EUROPE),
    (3, ts.Region.MIDDLE_EAST),
    (37, ts.Region.CENTRAL_AMERICA),
    (79, ts.Region.AFRICA),
    (81, ts.Region.SOUTH_AMERICA),
)

#: An ordinary own-side card, used to reach an influence-placement node without firing an event.
OPS_CARD = {ts.Player.US: 63, ts.Player.USSR: 30}

REGION_OF = {c: ts.MapData.get_country_info(c)["region"] for c in range(84)}


def true_region_value(state: ts.GameState, card: int, player: ts.Player) -> float:
    """VP this scoring card would pay `player` if played now, from the engine's own scorer."""
    region = dict(SCORING_CARDS)[card]
    summary = ts.Scoring.evaluate_region(state, region)
    net = float(summary.net_delta)
    return net if player == ts.Player.US else -net


def preference_ordering(model: Any, states: Sequence[ts.GameState], player: ts.Player,
                        cards: Sequence[int] = (1, 2, 3, 37, 79, 81),
                        temperature: float = 0.1) -> Dict[str, float]:
    """Given several scoring cards in hand, does the model prefer the one that pays most?

    All the named cards are put in hand at once and the model's distribution over *which card to
    play* is read at the card-selection node. Preference is scored against the engine's VP for
    each, which the observation does not contain.
    """
    import torch

    device = next(model.parameters()).device
    hand_loc = ts.hand_of(player)
    took_best: List[float] = []
    spearman: List[float] = []
    value_gap: List[float] = []

    for base in states:
        st = base.clone()
        for c in cards:
            st.set_card_location(c, hand_loc)
        mask = legal_mask(st)
        available = [c for c in cards if mask[c - 1]]
        if len(available) < 3:
            continue
        truth = np.array([true_region_value(st, c, player) for c in available])
        if truth.max() - truth.min() < 1e-9:
            continue

        obs = np.asarray(ts.extract_observation(st, player), dtype=np.float32)
        with torch.no_grad():
            logits, _w, _v = model(torch.from_numpy(obs).unsqueeze(0).to(device),
                                   torch.from_numpy(mask).unsqueeze(0).to(device))
            p = torch.softmax(logits[0] / max(temperature, 1e-4), dim=-1).cpu().numpy()
        pref = np.array([p[c - 1] for c in available], dtype=np.float64)

        took_best.append(1.0 if int(np.argmax(pref)) == int(np.argmax(truth)) else 0.0)
        if len(available) > 2:
            pr = np.argsort(np.argsort(pref)).astype(float)
            tr = np.argsort(np.argsort(truth)).astype(float)
            if pr.std() > 1e-9 and tr.std() > 1e-9:
                spearman.append(float(np.corrcoef(pr, tr)[0, 1]))
        best, worst = float(truth.max()), float(truth.min())
        chosen = float(truth[int(np.argmax(pref))])
        if best > worst:
            value_gap.append((chosen - worst) / (best - worst))

    def m(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {"positions": float(len(took_best)), "took_best_scoring_card": m(took_best),
            "rank_correlation": m(spearman), "value_captured": m(value_gap),
            "chance_took_best": 1.0 / max(len(cards), 1)}


def _region_mass(model: Any, state: ts.GameState, player: ts.Player,
                 temperature: float) -> Optional[Dict[Any, float]]:
    """Probability the model puts on each region at an influence-placement node."""
    import torch

    # The Ops card has to be in hand before it can be played -- without this the branch never
    # reaches a placement node and the probe silently measures nothing.
    st = state.clone()
    st.set_card_location(OPS_CARD[player], ts.hand_of(player))
    node = _to_point_node(st, OPS_CARD[player], "ops")
    if node is None:
        return None
    mask = legal_mask(node)
    legal = [c for c in range(84) if mask[COUNTRY_OFFSET + c]]
    if len(legal) < 4:
        return None
    device = next(model.parameters()).device
    obs = np.asarray(ts.extract_observation(node, node.ctx().decision_player), dtype=np.float32)
    with torch.no_grad():
        logits, _w, _v = model(torch.from_numpy(obs).unsqueeze(0).to(device),
                               torch.from_numpy(mask).unsqueeze(0).to(device))
        p = torch.softmax(logits[0] / max(temperature, 1e-4), dim=-1).cpu().numpy()
    sub = np.array([p[COUNTRY_OFFSET + c] for c in legal], dtype=np.float64)
    total = sub.sum()
    if total <= 0:
        return None
    sub = sub / total
    out: Dict[Any, float] = {}
    for k, c in enumerate(legal):
        r = REGION_OF[c]
        out[r] = out.get(r, 0.0) + float(sub[k])
    return out


def location_sensitivity(model: Any, states: Sequence[ts.GameState], player: ts.Player,
                         cards: Sequence[Tuple[int, Any]] = SCORING_CARDS,
                         temperature: float = 0.1) -> Dict[str, Dict[str, float]]:
    """Does moving a scoring card change where the model wants to place -- in *that* region?

    For each card, the board is held fixed and only the card's location changes. The headline is
    the difference in differences: the shift in placement into the card's own region minus the
    mean shift into the other regions, so a generic reflex to a scoring card being in hand
    cancels out.
    """
    hand_loc = ts.hand_of(player)
    opp_loc = ts.hand_of(ts.Player.USSR if player == ts.Player.US else ts.Player.US)
    places = {"deck": ts.CardLocation.DRAW_DECK, "my_hand": hand_loc,
              "opp_hand": opp_loc, "discard": ts.CardLocation.DISCARD_PILE}

    out: Dict[str, Dict[str, float]] = {}
    for card, region in cards:
        own: Dict[str, List[float]] = {k: [] for k in places}
        other: Dict[str, List[float]] = {k: [] for k in places}
        for base in states:
            per_place: Dict[str, Dict[Any, float]] = {}
            for label, loc in places.items():
                st = base.clone()
                st.set_card_location(card, loc)
                m = _region_mass(model, st, player, temperature)
                if m is None:
                    break
                per_place[label] = m
            if len(per_place) != len(places):
                continue
            for label, mass in per_place.items():
                own[label].append(mass.get(region, 0.0))
                others = [v for r, v in mass.items() if r != region]
                other[label].append(float(np.mean(others)) if others else 0.0)

        n = len(own["my_hand"])
        if n < 5:
            continue

        def mean(d: Dict[str, List[float]], k: str) -> float:
            return float(np.mean(d[k])) if d[k] else float("nan")

        res = {f"own_{k}": mean(own, k) for k in places}
        res.update({f"other_{k}": mean(other, k) for k in places})
        # Hand versus deck is the cleanest contrast: the card is equally unplayed in both, and
        # only in one can this player play it.
        res["did_own"] = (mean(own, "my_hand") - mean(own, "deck")) \
            - (mean(other, "my_hand") - mean(other, "deck"))
        res["did_opp_hand"] = (mean(own, "opp_hand") - mean(own, "deck")) \
            - (mean(other, "opp_hand") - mean(other, "deck"))
        res["positions"] = float(n)
        out[str(ts.CardData.get_card_name(card))] = res
    return out


def card_swap(model: Any, states: Sequence[ts.GameState], player: ts.Player,
              cards: Sequence[Tuple[int, Any]] = SCORING_CARDS,
              temperature: float = 0.1) -> Dict[str, Dict[str, float]]:
    """Two copies of one position differing only in *which* scoring card is held.

    The cleanest identity test available. Every copy holds exactly one scoring card, the rest in
    the deck, so "a scoring card is in hand" is constant across the comparison and the only thing
    that varies is which one it is. If the model places more into Europe when the card it holds is
    Europe Scoring than when it is Asia Scoring, it has associated the card with the region -- and
    only the identity embedding can carry that, since the two share a feature vector.

    Reported per card as the mass it draws into its own region while held, minus the mass that
    same region draws while a different scoring card is held instead.
    """
    hand_loc = ts.hand_of(player)
    per_card: Dict[int, List[Dict[Any, float]]] = {c: [] for c, _ in cards}
    kept = 0

    for base in states:
        masses: Dict[int, Dict[Any, float]] = {}
        for card, _region in cards:
            st = base.clone()
            for other, _r in cards:
                st.set_card_location(other, ts.CardLocation.DRAW_DECK)
            st.set_card_location(card, hand_loc)
            m = _region_mass(model, st, player, temperature)
            if m is None:
                break
            masses[card] = m
        if len(masses) != len(cards):
            continue
        for card in masses:
            per_card[card].append(masses[card])
        kept += 1

    out: Dict[str, Dict[str, float]] = {}
    if kept < 5:
        return out
    for card, region in cards:
        held = float(np.mean([m.get(region, 0.0) for m in per_card[card]]))
        others = [float(np.mean([per_card[o][k].get(region, 0.0) for k in range(kept)]))
                  for o, _r in cards if o != card]
        out[str(ts.CardData.get_card_name(card))] = {
            "positions": float(kept),
            "region_mass_when_held": held,
            "region_mass_when_other_held": float(np.mean(others)),
            "swap_effect": held - float(np.mean(others)),
        }
    return out


def unknown_to_discard(model: Any, states: Sequence[ts.GameState], player: ts.Player,
                       cards: Sequence[Tuple[int, Any]] = SCORING_CARDS,
                       temperature: float = 0.1) -> Dict[str, Dict[str, float]]:
    """Does a scoring card reaching the discard cool the model's interest in its region?

    While a scoring card sits in the draw deck or the opponent's hand it is unresolved: it may
    still be played, so the region may still be scored and is worth investing in. Once discarded
    it cannot score again until the deck is reshuffled, so the same investment is less urgent.

    "Unknown" pools the deck and the opponent's hand because they share exactly that property --
    out of this player's control and still live. Reported as a difference in differences against
    the other regions, so a general shift in placement cancels. **Negative is the competent
    direction.**
    """
    opp_loc = ts.hand_of(ts.Player.USSR if player == ts.Player.US else ts.Player.US)
    places = {"deck": ts.CardLocation.DRAW_DECK, "opp_hand": opp_loc,
              "discard": ts.CardLocation.DISCARD_PILE}

    out: Dict[str, Dict[str, float]] = {}
    for card, region in cards:
        own: Dict[str, List[float]] = {k: [] for k in places}
        other: Dict[str, List[float]] = {k: [] for k in places}
        for base in states:
            got: Dict[str, Dict[Any, float]] = {}
            for label, loc in places.items():
                st = base.clone()
                st.set_card_location(card, loc)
                m = _region_mass(model, st, player, temperature)
                if m is None:
                    break
                got[label] = m
            if len(got) != len(places):
                continue
            for label, mass in got.items():
                own[label].append(mass.get(region, 0.0))
                rest = [v for r, v in mass.items() if r != region]
                other[label].append(float(np.mean(rest)) if rest else 0.0)

        n = len(own["discard"])
        if n < 5:
            continue

        def mean(d: Dict[str, List[float]], k: str) -> float:
            return float(np.mean(d[k])) if d[k] else float("nan")

        unknown_own = 0.5 * (mean(own, "deck") + mean(own, "opp_hand"))
        unknown_other = 0.5 * (mean(other, "deck") + mean(other, "opp_hand"))
        out[str(ts.CardData.get_card_name(card))] = {
            "positions": float(n),
            "own_unknown": unknown_own,
            "own_discard": mean(own, "discard"),
            "cooling": (mean(own, "discard") - unknown_own)
            - (mean(other, "discard") - unknown_other),
        }
    return out
