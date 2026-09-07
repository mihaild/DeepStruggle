"""Early-war handling of the two big USSR events, surveyed across the whole human corpus.

§13 asked this of six hand-picked games, which locates a behaviour but does not measure a rate.
This asks it of every early-war position in the corpus where the card was actually in hand, so the
denominator is positions where the choice existed.

The right answers are not symmetric, and neither is close:

* **Decolonization and De-Stalinization are USSR events strong enough that the USSR should fire
  them.** Holding one from turn 1 into turn 2 is sometimes right; holding Decolonization *through*
  turn 2 is not. Played for Ops by its owner the event never fires at all, so a 2-Ops
  Decolonization or a 3-Ops one-time De-Stalinization is the whole return on the card.
* **The US should never play either for Ops in the early war.** An opponent's card played for
  Operations still owes its event (`engine/src/state_machine.cpp:271`), so Ops buys the Ops and
  hands over the full event. Hold it, or put it on the space track.

Firing the event is only half of playing it well, so where the USSR does fire it this also records
what it did with it -- which countries Decolonization reached, and what De-Stalinization moved.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

from ai.eval.card_probe import (DECOLONIZATION, DE_STALINIZATION, MODE_NAMES, Node, _drain,
                                capture_nodes, card_action, card_name, give_card, hand, policy)

EVENT, OPS, SPACE = 0, 1, 2
EARLY_WAR_TURNS = (1, 2, 3)


@dataclass
class Placement:
    """One country the event touched, and which way the Influence went."""

    country: int
    before: int
    after: int

    @property
    def moved(self) -> int:
        return self.after - self.before


@dataclass
class Survey:
    """Mode choices over a set of positions, for one card, one side, one snapshot."""

    card: int
    side: ts.Player
    positions: int = 0
    # Mean probability the policy puts on each mode, over positions where that mode was legal.
    mode_prob: Dict[int, List[float]] = field(default_factory=dict)
    # How often each mode was the greedy pick.
    greedy: Counter = field(default_factory=Counter)
    # How often each mode was even available.
    legal: Counter = field(default_factory=Counter)
    # Where the card ended up when the turn was played out.
    fate: Counter = field(default_factory=Counter)
    # Countries the fired event touched, pooled over positions.
    placements: List[Placement] = field(default_factory=list)
    turns: Counter = field(default_factory=Counter)

    def mean_prob(self, mode: int) -> float:
        vals = self.mode_prob.get(mode) or []
        return float(np.mean(vals)) if vals else float("nan")

    def greedy_share(self, mode: int) -> float:
        return 100.0 * self.greedy[mode] / max(1, self.positions)


def _mode_node(state: ts.GameState, card: int) -> Optional[ts.GameState]:
    """The position right after this card is selected, where the mode is chosen."""
    action = card_action(state, card)
    if action is None:
        return None
    probe = state.clone()
    ts.Engine.step_flat(probe, action)
    _drain(probe)
    if ts.Engine.is_terminal(probe):
        return None
    if probe.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return None
    return probe


def _mode_actions(state: ts.GameState) -> Dict[int, int]:
    """Legal play modes at a SELECT_PLAY_MODE node, as {mode: flat action}."""
    out: Dict[int, int] = {}
    for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(state))):
        ma = ts.decode_flat_action(state, int(a))
        if ma.decision_type != ts.DecisionType.SELECT_PLAY_MODE:
            continue
        out[int(ma.secondary_id) or int(ma.primary_id)] = int(a)
    return out


def event_placements(model: Any, state: ts.GameState, mover: ts.Player, card: int,
                     device: Any, max_steps: int = 40) -> List[Placement]:
    """Fire the event and record every country the policy sends it to.

    Recorded as before/after Influence for the acting side rather than as bare country ids,
    because De-Stalinization both removes and places and the two read identically as targets.
    """
    modes = _mode_actions(state)
    if EVENT not in modes:
        return []
    probe = state.clone()
    ts.Engine.step_flat(probe, modes[EVENT])
    _drain(probe)

    out: List[Placement] = []
    for _ in range(max_steps):
        if ts.Engine.is_terminal(probe):
            break
        ctx = probe.ctx()
        if int(ctx.resolving_card) != card:
            break
        who = ctx.decision_player if ctx.decision_player != ts.Player.NONE else mover
        probs = policy(model, probe, who, device)
        if not probs.any():
            break
        action = int(np.argmax(probs))
        ma = ts.decode_flat_action(probe, action)
        target = int(ma.primary_id)
        if ctx.decision_type == ts.DecisionType.POINT_NODE and 0 <= target < 84:
            country = probe.get_country(target)
            before = int(country.ussr_influence if mover == ts.Player.USSR
                         else country.us_influence)
            ts.Engine.step_flat(probe, action)
            _drain(probe)
            country = probe.get_country(target)
            after = int(country.ussr_influence if mover == ts.Player.USSR
                        else country.us_influence)
            out.append(Placement(target, before, after))
            continue
        ts.Engine.step_flat(probe, action)
        _drain(probe)
    return out


def survey(model: Any, nodes: Sequence[Node], card: int, side: ts.Player, device: Any,
           swap_in: bool = False, record_placements: bool = False,
           fates: bool = False) -> Survey:
    """Mode choice for `card` in `side`'s hand, over every position in `nodes`."""
    from ai.eval.card_probe import card_fate

    out = Survey(card=card, side=side)
    for node in nodes:
        state = (give_card(node.state, side, card) if swap_in
                 else node.state)
        if card not in hand(state, side):
            continue
        at_mode = _mode_node(state, card)
        if at_mode is None:
            continue

        modes = _mode_actions(at_mode)
        if not modes:
            continue
        probs = policy(model, at_mode, side, device)
        out.positions += 1
        out.turns[int(node.turn)] += 1
        for mode, action in modes.items():
            out.legal[mode] += 1
            out.mode_prob.setdefault(mode, []).append(float(probs[action]))
        best = max(modes, key=lambda m: probs[modes[m]])
        out.greedy[best] += 1

        if record_placements and best == EVENT:
            out.placements.extend(event_placements(model, at_mode, side, card, device))
        if fates:
            out.fate[card_fate(model, state, side, card, device)] += 1
    return out


def collect(games: Iterable[Dict], cards: Sequence[int],
            turns: Sequence[int] = EARLY_WAR_TURNS,
            ) -> Tuple[Dict[int, List[Node]], Dict[int, List[Node]]]:
    """Early-war Action Round card selections from the corpus, split by mover.

    Returns `(ussr_nodes, us_nodes)` keyed by card: the USSR list holds only positions where that
    side genuinely held the card, so the USSR figures are about real hands rather than constructed
    ones. The US list is every US node, since the card has to be swapped in there regardless.
    """
    ussr: Dict[int, List[Node]] = {c: [] for c in cards}
    us: Dict[int, List[Node]] = {c: [] for c in cards}
    for game in games:
        try:
            # One conversion per game, not one per turn: converting is the expensive part.
            nodes = capture_nodes(game, turn=turns)
        except Exception:
            # A game that will not convert is not evidence about card play; §12 and §13 both run
            # on the games that do. The corpus-wide conversion test is what guards this.
            continue
        for node in nodes:
            for card in cards:
                if node.mover == ts.Player.USSR and card in hand(node.state, ts.Player.USSR):
                    ussr[card].append(node)
                elif node.mover == ts.Player.US:
                    us[card].append(node)
    return ussr, us


def load_games(replay_ids: Iterable[int]) -> List[Dict]:
    from tools.lib.corpus_paths import corpus_path

    out = []
    for rid in replay_ids:
        try:
            out.append(json.load(gzip.open(corpus_path(int(rid)), "rt")))
        except Exception:
            continue
    return out


def region_name(cid: int) -> str:
    from ai.eval.battleground_value import REGION_NAMES, region_of

    return REGION_NAMES.get(region_of(cid), "?")


def placement_summary(placements: Sequence[Placement], top: int = 8) -> List[str]:
    """The countries the event reached, most frequent first."""
    gained = Counter(p.country for p in placements if p.moved > 0)
    lost = Counter(p.country for p in placements if p.moved < 0)
    lines = []
    for label, counter in (("into", gained), ("out of", lost)):
        if not counter:
            continue
        parts = [f"{ts.MapData.get_country_name(c)}"
                 f"{'*' if bool(ts.MapData.get_country_info(c)['battleground']) else ''} x{n}"
                 for c, n in counter.most_common(top)]
        lines.append(f"{label}: " + ", ".join(parts))
    return lines
