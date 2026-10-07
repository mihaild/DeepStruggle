#!/usr/bin/env python3
"""Per-checkpoint behaviour report: individual cards first, then events vs Ops for every card.

One pass of greedy self-play per checkpoint feeds every section:

1. **Star Wars** -- the US's plays of it while ahead in space (headline / Ops / event / space race
   in a round) and the card its event takes from the discard pile, split by who played Star Wars
   (`tools/scripts/star_wars_play.py`).
2. **Five Year Plan played by the USSR** -- the USSR playing the US card from its own hand so the
   US event fires (headline, or a round with the event first or the Ops first; not the space
   race): the action round, split Early War / Mid+Late War, and the USSR hand at that moment.
3. **Aldrich Ames Remix played by the US** -- the same for the US playing the USSR card.
4. **OPEC and Alliance for Progress** -- how often the owner plays the event by the VP it would
   score at that moment (0-2 / 3-4 / 5+), per card choice and per holding.
5. **Soviets Shoot Down KAL-007** -- the US's choices holding it, split by South Korea's control.
6. **UN Intervention** -- the opponent card each side plays with it.
7. **Space race** -- the cards each side sends there most.
8. **Events vs Ops, every card** -- the event census of `tools/scripts/event_play_census.py`, the
   same logic (one count per holding where the owner could have played the event).

    PYTHONPATH=.:build/release python tools/scripts/checkpoint_report.py \\
        --checkpoints <pt> [<pt> ...] --games 16384

Each checkpoint gets `research/log/per_checkpoint/<run>_<step>M.md` (and the raw records in
`data/eval/per_checkpoint/`); the directory's `README.md` index is rewritten to list every report.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.scripts.event_play_census import (EVENT, SCORING, Holding, HoldingTracker, dump_holding,
                                             load_holding, table as census_table)
from tools.scripts.star_wars_play import _tag_retrievals, retrieval_table

US, USSR = int(ts.Player.US), int(ts.Player.USSR)
SPACE = ActionEncoder.PLAY_MODE_OFFSET + 1
CARDS = {int(c["id"]): c for c in json.load(open("rules/cards.json"))}
ID = {c["name"]: i for i, c in CARDS.items()}
STAR_WARS = ID["Star Wars"]
FIVE_YEAR_PLAN = ID["Five Year Plan"]
ALDRICH_AMES = ID["Aldrich Ames Remix"]
KAL_007 = ID["Soviets Shoot Down KAL-007"]
OPEC = ID["OPEC"]
ALLIANCE = ID["Alliance for Progress"]
UN_INTERVENTION = ID["UN Intervention"]
_MAP = {c["name"]: c for c in json.load(open("rules/map.json"))["countries"]}
SOUTH_KOREA = int(_MAP["South Korea"]["id"])
#: The engine's lists (`trigger_opec`, `trigger_alliance_for_progress`): 1 VP per country controlled.
OPEC_COUNTRIES = [int(_MAP[n]["id"]) for n in
                  ("Egypt", "Iran", "Libya", "Saudi Arabia", "Iraq", "Gulf States", "Venezuela")]
ALLIANCE_COUNTRIES = [int(c["id"]) for c in _MAP.values()
                      if c["battleground"] and c["region"] in ("Central America", "South America")]
REPORT_DIR = "research/log/per_checkpoint"
DATA_DIR = "/workspace/data/eval/per_checkpoint"


def _side(card: int) -> str:
    return str(CARDS[card].get("side", "neutral")).lower()


def _hand(st: ts.GameState, player: Any) -> List[int]:
    return [c for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), player)]


def _mode(a: int) -> str:
    return "event first" if a == EVENT else "space race" if a == SPACE else "Ops first"


# --------------------------------------------------------------------------------------------
# Observers. Each sees every live env at every step, before the step is taken, with the action
# about to be played; game ids are global across batches.
# --------------------------------------------------------------------------------------------

class Census:
    """The event census: `event_play_census.HoldingTracker`, one per game, so the counting is the
    census tool's own -- each holding's legality and how it ended (headline, event, Ops, space race
    or kept)."""

    def __init__(self) -> None:
        self.holdings: List[Holding] = []
        self.trackers: Dict[int, HoldingTracker] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        self.trackers.setdefault(g, HoldingTracker()).observe(st, a)

    def done(self, g: int) -> None:
        t = self.trackers.pop(g, None)
        if t is not None:
            self.holdings += t.holdings


class StarWars:
    """`star_wars_play.play`'s plays and retrievals."""

    def __init__(self) -> None:
        self.plays: List[Dict[str, Any]] = []
        self.retrievals: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}

    def see(self, g: int, st: ts.GameState, a: int, mask: np.ndarray) -> None:
        ctx = st.ctx()
        sel = self.pending.get(g)
        if sel is not None and not (ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                                    and int(ctx.pending_op_card) == STAR_WARS):
            del self.pending[g]                      # selected at a choice that was not a play
            sel = None
        if (int(ctx.resolving_card) == STAR_WARS and ctx.decision_player == ts.Player.US
                and ctx.decision_type == ts.DecisionType.SELECT_CARD):
            headline = st.current_phase == ts.Phase.HEADLINE
            by_us = (int(st.headline_us_card) == STAR_WARS if headline else st.phasing_player == ts.Player.US)
            self.retrievals.append({
                "game": g, "turn": int(st.turn), "by": "US" if by_us else "USSR",
                "phase": "headline" if headline else "ar", "card": int(ts.decode_flat_action(st, a).primary_id),
                "options": [int(ts.decode_flat_action(st, int(k)).primary_id)
                            for k in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET])]})
        if int(ctx.resolving_card) != 0 or ctx.decision_player != ts.Player.US:
            return
        if (ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET
                and int(ts.decode_flat_action(st, a).primary_id) == STAR_WARS
                and ts.in_hand_of(st.get_card_location(STAR_WARS), ts.Player.US)):
            headline = st.current_phase == ts.Phase.HEADLINE
            rec = {"game": g, "turn": int(st.turn), "ahead": int(st.us_space_track) > int(st.ussr_space_track),
                   "ar": 0 if headline else int(st.action_round), "phase": "headline" if headline else "ar",
                   "space": [int(st.us_space_track), int(st.ussr_space_track)]}
            if headline:
                rec["outcome"] = "headline"
                self.plays.append(rec)
            else:
                self.pending[g] = rec
        elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and sel is not None:
            sel["outcome"] = "event" if a == EVENT else "space" if a == SPACE else "ops"
            self.plays.append(sel)
            del self.pending[g]

    def done(self, g: int) -> None:
        self.pending.pop(g, None)


class OpponentCardPlay:
    """`player` playing `card` -- the opponent's card -- from its own hand: a headline, or an
    action-round play followed by the play mode. The event fires unless the mode is the space race.
    Records the turn, the action round (0 = headline) and the player's hand without the card."""

    def __init__(self, card: int, player: Any) -> None:
        self.card, self.player = card, player
        self.plays: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        sel = self.pending.get(g)
        if sel is not None:
            del self.pending[g]
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == self.card:
                sel["mode"] = _mode(a)
                self.plays.append(sel)
            # otherwise it was selected at a choice that was not a play (a forced discard)
        if (int(ctx.resolving_card) != 0 or ctx.decision_player != self.player
                or ctx.decision_type != ts.DecisionType.SELECT_CARD or a >= ActionEncoder.PLAY_MODE_OFFSET):
            return
        if int(ts.decode_flat_action(st, a).primary_id) != self.card:
            return
        if not ts.in_hand_of(st.get_card_location(self.card), self.player):
            return
        headline = st.current_phase == ts.Phase.HEADLINE
        rec: Dict[str, Any] = {"game": g, "turn": int(st.turn), "ar": 0 if headline else int(st.action_round),
                               "hand": [c for c in _hand(st, self.player) if c != self.card],
                               "vp": int(st.victory_points), "defcon": int(st.defcon)}
        if headline:
            rec["mode"] = "headline"
            self.plays.append(rec)
        else:
            self.pending[g] = rec

    def done(self, g: int) -> None:
        self.pending.pop(g, None)


class OwnCardChoices:
    """Every card choice of `owner` (headline or action round, not inside an event) while `card`
    is in its hand: whether the event could trigger, `feature(st)` (a number that conditions the
    event's worth), and whether the card was played and how -- "headline", or in a round "event",
    "Ops" or "space race". `holding` numbers the spells of the card in the owner's hand."""

    def __init__(self, card: int, owner: Any, feature: Any) -> None:
        self.card, self.owner, self.feature = card, owner, feature
        self.choices: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.held: Dict[int, bool] = {}
        self.spell: Dict[int, int] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        sel = self.pending.pop(g, None)
        if sel is not None:
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == self.card:
                sel["played"] = "event" if a == EVENT else "space race" if a == SPACE else "Ops"
            else:
                sel["played"] = "not a play"
        holding = ts.in_hand_of(st.get_card_location(self.card), self.owner)
        if holding and not self.held.get(g, False):
            self.spell[g] = self.spell.get(g, 0) + 1
        self.held[g] = holding
        if (not holding or int(ctx.resolving_card) != 0 or ctx.decision_player != self.owner
                or ctx.decision_type != ts.DecisionType.SELECT_CARD
                or st.current_phase not in (ts.Phase.HEADLINE, ts.Phase.ACTION_ROUND)):
            return
        headline = st.current_phase == ts.Phase.HEADLINE
        chosen = a < ActionEncoder.PLAY_MODE_OFFSET and int(ts.decode_flat_action(st, a).primary_id) == self.card
        rec: Dict[str, Any] = {"game": g, "holding": self.spell[g], "turn": int(st.turn),
                               "phase": "headline" if headline else "ar",
                               "legal": bool(ts.CardHandlers.can_trigger_event(st, self.card, self.owner)),
                               "feature": self.feature(st), "played": "headline" if chosen and headline else None}
        self.choices.append(rec)
        if chosen and not headline:
            self.pending[g] = rec

    def done(self, g: int) -> None:
        for d in (self.pending, self.held, self.spell):
            d.pop(g, None)


class UNIntervention:
    """The card a side plays with UN Intervention: the opponent card whose Ops it uses."""

    def __init__(self) -> None:
        self.picks: List[Dict[str, Any]] = []

    def see(self, g: int, st: ts.GameState, a: int, mask: np.ndarray) -> None:
        ctx = st.ctx()
        if int(ctx.resolving_card) != UN_INTERVENTION or ctx.decision_type != ts.DecisionType.SELECT_CARD:
            return
        self.picks.append({"game": g, "turn": int(st.turn), "side": "US" if ctx.decision_player == ts.Player.US else "USSR",
                           "card": int(ts.decode_flat_action(st, a).primary_id),
                           "options": [int(ts.decode_flat_action(st, int(k)).primary_id)
                                       for k in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET])]})

    def done(self, g: int) -> None:
        pass


class PlayModes:
    """Every play-mode choice (a card played in an action round, not inside an event): which card,
    which side, and whether it went to the space race -- the denominator for the space-race list."""

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {"US": {}, "USSR": {}}
        self.space: Dict[str, Dict[str, int]] = {"US": {}, "USSR": {}}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        if (ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE or int(ctx.resolving_card) != 0
                or st.current_phase != ts.Phase.ACTION_ROUND):
            return
        side = "US" if ctx.decision_player == ts.Player.US else "USSR"
        card = str(int(ctx.pending_op_card))
        self.counts[side][card] = self.counts[side].get(card, 0) + 1
        if a == SPACE:
            self.space[side][card] = self.space[side].get(card, 0) + 1

    def done(self, g: int) -> None:
        pass


def _controlled(st: ts.GameState, countries: Sequence[int], side: Any) -> int:
    return sum(1 for c in countries if ts.Scoring.is_controlled_by(st, c, side))


def play(model: Any, merged: bool, games: int, seed: int, batch: int) -> Dict[str, Any]:
    from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
    check_obs_width(model)
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    census, sw = Census(), StarWars()
    fyp, ames = OpponentCardPlay(FIVE_YEAR_PLAN, ts.Player.USSR), OpponentCardPlay(ALDRICH_AMES, ts.Player.US)
    kal = OwnCardChoices(KAL_007, ts.Player.US,
                         lambda st: int(ts.Scoring.is_controlled_by(st, SOUTH_KOREA, ts.Player.US)))
    opec = OwnCardChoices(OPEC, ts.Player.USSR, lambda st: _controlled(st, OPEC_COUNTRIES, ts.Player.USSR))
    alliance = OwnCardChoices(ALLIANCE, ts.Player.US, lambda st: _controlled(st, ALLIANCE_COUNTRIES, ts.Player.US))
    un, modes = UNIntervention(), PlayModes()
    observers = (census, sw, fyp, ames, kal, opec, alliance, un, modes)
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        if merged:
            env.set_merged_influence(True, True)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        for _ in range(20_000):
            if all(done):
                break
            masks_np = np.asarray(masks)
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)[:, :width]).to(device),
                               torch.from_numpy(masks_np).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                g, a = b0 + i, int(actions[i])
                census.see(g, st, a)
                sw.see(g, st, a, masks_np[i])
                fyp.see(g, st, a)
                ames.see(g, st, a)
                kal.see(g, st, a)
                opec.see(g, st, a)
                alliance.see(g, st, a)
                un.see(g, st, a, masks_np[i])
                modes.see(g, st, a)
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d and not done[i]:
                    done[i] = True
                    for o in observers:
                        o.done(b0 + i)
        for i in range(n):                      # a game still running at the step cap
            if not done[i]:
                for o in observers:
                    o.done(b0 + i)
    return {"games": games, "seed": seed, "batch": batch,
            "holdings": [dump_holding(h) for h in census.holdings],
            "star_wars": {"plays": sw.plays, "retrievals": sw.retrievals},
            "five_year_plan": fyp.plays, "aldrich_ames": ames.plays,
            "kal_007": kal.choices, "opec": opec.choices, "alliance_for_progress": alliance.choices,
            "un_intervention": un.picks, "play_modes": {"all": modes.counts, "space": modes.space}}


# --------------------------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------------------------

def _pct(k: int, n: int) -> str:
    return "—" if n == 0 else f"{100 * k / n:.1f}%"


def star_wars_section(d: Dict[str, Any]) -> str:
    plays = d["plays"]
    ahead = [p for p in plays if p["ahead"]]
    n = len(ahead)
    k = collections.Counter(p["outcome"] for p in ahead)
    behind = [p for p in plays if not p["ahead"]]
    dud = sum(p["outcome"] == "headline" for p in behind)
    tagged = _tag_retrievals({"plays": plays, "retrievals": d["retrievals"]})
    inside = sum(r["via"] == "inside another event" for r in tagged)
    out = ["## Star Wars", "",
           "Star Wars (US, 2 Ops): if the US is ahead on the Space Race, the US takes a non-scoring card from the "
           "discard pile and plays it as an event. A *play* is the US selecting Star Wars at its own card choice; "
           "\"ahead\" is the US space marker strictly above the USSR's at that choice.", "",
           "| plays while ahead | headline | Ops in a round | event in a round | space race |",
           "|---:|---:|---:|---:|---:|",
           f"| {n:,} | {k['headline']} ({_pct(k['headline'], n)}) | {k['ops']} ({_pct(k['ops'], n)}) | "
           f"{k['event']} ({_pct(k['event'], n)}) | {k['space']} ({_pct(k['space'], n)}) |", "",
           f"Not ahead: {len(behind):,} plays, {dud} of them headlines (the event does nothing).", "",
           "### What Star Wars takes from the discard pile", "",
           "\"Offered in\": the share of retrievals where the card was a legal option; \"picked when offered\": how "
           "often it was taken then."]
    for via, what in (("US play", "the US played Star Wars (headline or event in a round)"),
                      ("USSR play", "the USSR played Star Wars, firing the US event")):
        rs = [r for r in tagged if r["via"] == via]
        out += ["", f"#### {what[0].upper()}{what[1:]}", "", retrieval_table(rs) if rs else "No retrievals."]
    out += ["", f"Not counted: {inside} retrievals where the US event fired inside another event "
                f"(Star Wars handed to the US by an event such as Missile Envy)."]
    return "\n".join(out)


def _era(turn: int) -> str:
    return "Early War" if turn <= 3 else "Mid+Late War"


def opponent_card_section(title: str, intro: str, plays: Sequence[Dict[str, Any]], holder: str,
                          show_mode: bool = True) -> str:
    fired = [p for p in plays if p["mode"] != "space race"]
    eras = ["Early War", "Mid+Late War"]
    by = {e: [p for p in fired if _era(p["turn"]) == e] for e in eras}
    out = [f"## {title}", "", intro, "",
           f"{len(plays):,} plays from hand; {len(fired):,} fire the event "
           f"({sum(p['mode'] == 'space race' for p in plays)} went to the space race and are left out below).", ""]
    if not fired:
        return "\n".join(out + ["No plays."])
    # When: the action round
    ars = sorted({p["ar"] for p in fired})
    out += ["### When (action round; H = headline)", "",
            "| | " + " | ".join(f"{e} (n={len(by[e]):,})" for e in eras) + " |", "|:---|" + "---:|" * len(eras)]
    for ar in ars:
        cells = []
        for e in eras:
            k = sum(p["ar"] == ar for p in by[e])
            cells.append(f"{k} ({_pct(k, len(by[e]))})" if by[e] else "—")
        out.append(f"| {'H' if ar == 0 else f'AR{ar}'} | " + " | ".join(cells) + " |")
    if show_mode:
        out += ["", "| how it was played | " + " | ".join(eras) + " |", "|:---|" + "---:|" * len(eras)]
        for m in ("headline", "event first", "Ops first"):
            out.append(f"| {m} | " + " | ".join(
                f"{_pct(sum(p['mode'] == m for p in by[e]), len(by[e]))}" for e in eras) + " |")
    by_turn = collections.Counter(p["turn"] for p in fired)
    out += ["", "By turn: " + ", ".join(f"T{t} {by_turn[t]}" for t in sorted(by_turn)) + "."]
    # What is in the hand
    opp = "ussr" if holder == "us" else "us"
    out += ["", f"### The {holder.upper()} hand at that moment (without the card played)", "",
            "| | " + " | ".join(eras) + " |", "|:---|" + "---:|" * len(eras)]

    def stat(fn: Any, fmt: str = "{:.2f}") -> str:
        cells = []
        for e in eras:
            xs = [fn(p) for p in by[e]]
            cells.append(fmt.format(float(np.mean(xs))) if xs else "—")
        return " | ".join(cells)

    def frac(p: Dict[str, Any], pred: Any) -> float:
        return sum(1 for c in p["hand"] if pred(c)) / max(1, len(p["hand"]))
    out += [f"| cards in hand (mean) | {stat(lambda p: len(p['hand']))} |",
            f"| last card (hand empty after it) | {stat(lambda p: 100.0 * (len(p['hand']) == 0), '{:.1f}%')} |",
            f"| scoring cards (mean) | {stat(lambda p: sum(c in SCORING for c in p['hand']))} |",
            f"| plays with a scoring card in hand | {stat(lambda p: 100.0 * any(c in SCORING for c in p['hand']), '{:.1f}%')} |",
            f"| {holder.upper()} cards, own events (mean) | {stat(lambda p: sum(_side(c) == holder and c not in SCORING for c in p['hand']))} |",
            f"| {opp.upper()} cards, opponent events (mean) | {stat(lambda p: sum(_side(c) == opp for c in p['hand']))} |",
            f"| neutral non-scoring cards (mean) | {stat(lambda p: sum(_side(c) == 'neutral' and c not in SCORING for c in p['hand']))} |",
            f"| share of hand: scoring | {stat(lambda p: 100 * frac(p, lambda c: c in SCORING), '{:.1f}%')} |",
            f"| share of hand: {opp.upper()} cards | {stat(lambda p: 100 * frac(p, lambda c: _side(c) == opp), '{:.1f}%')} |"]
    for e in eras:
        if not by[e]:
            continue
        cnt = collections.Counter(c for p in by[e] for c in p["hand"])
        out += ["", f"Most frequent cards in hand, {e} (share of plays holding it): " + ", ".join(
            f"{CARDS[c]['name']} {100 * k / len(by[e]):.0f}%" for c, k in cnt.most_common(15)) + "."]
    return "\n".join(out)


VP_BUCKETS = (("0-2 VP", 0, 2), ("3-4 VP", 3, 4), ("5+ VP", 5, 99))


def _bucket(v: int) -> str:
    return next(name for name, lo, hi in VP_BUCKETS if lo <= v <= hi)


def _holding_outcomes(choices: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Per holding (game, spell): its choices where the event could trigger, the best `feature`
    among them, and how the holding ended -- the card's play, if it was played at one of them."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for c in choices:
        h = out.setdefault((c["game"], c["holding"]), {"max": -1, "played": None, "at": None, "legal": False})
        if c["legal"]:
            h["legal"] = True
            h["max"] = max(h["max"], int(c["feature"]))
        if c["played"] in ("headline", "event", "Ops", "space race"):
            h["played"], h["at"] = c["played"], int(c["feature"])
    return out


def vp_card_section(name: str, intro: str, choices: Sequence[Dict[str, Any]]) -> str:
    legal = [c for c in choices if c["legal"]]
    out = [f"### {name}", "", intro, "",
           "At each of the owner's card choices holding the card where the event can trigger, by the VP the event "
           "would score at that moment:", "",
           "| VP the event would score | headline choices | headlined | action-round choices | event in the round "
           "| Ops / space race in the round |", "|:---|---:|---:|---:|---:|---:|"]
    for b, _, _ in VP_BUCKETS:
        hl = [c for c in legal if c["phase"] == "headline" and _bucket(int(c["feature"])) == b]
        ar = [c for c in legal if c["phase"] == "ar" and _bucket(int(c["feature"])) == b]
        out.append(f"| {b} | {len(hl):,} | {_pct(sum(c['played'] == 'headline' for c in hl), len(hl))} | {len(ar):,} | "
                   f"{_pct(sum(c['played'] == 'event' for c in ar), len(ar))} | "
                   f"{_pct(sum(c['played'] in ('Ops', 'space race') for c in ar), len(ar))} |")
    hs = [h for h in _holding_outcomes(choices).values() if h["legal"]]
    out += ["", "Per holding (the card in the owner's hand until it leaves), by the most the event would have scored "
                "at any of those choices -- how the holding ended:", "",
            "| best VP while held | holdings | evented (headline or round) | Ops / space race | not played by the owner |",
            "|:---|---:|---:|---:|---:|"]
    for b, _, _ in VP_BUCKETS:
        xs = [h for h in hs if _bucket(h["max"]) == b]
        out.append(f"| {b} | {len(xs):,} | {_pct(sum(h['played'] in ('headline', 'event') for h in xs), len(xs))} | "
                   f"{_pct(sum(h['played'] in ('Ops', 'space race') for h in xs), len(xs))} | "
                   f"{_pct(sum(h['played'] is None for h in xs), len(xs))} |")
    ev = [h["at"] for h in hs if h["played"] in ("headline", "event")]
    if ev:
        out += ["", f"When evented, the VP it would score at that choice: mean {float(np.mean(ev)):.2f}, "
                    f"5+ in {_pct(sum(v >= 5 for v in ev), len(ev))} of {len(ev):,} plays (a headline resolves after "
                    f"the other side's may have changed the board)."]
    return "\n".join(out)


def kal_section(choices: Sequence[Dict[str, Any]]) -> str:
    out = ["## Soviets Shoot Down KAL-007", "",
           "KAL-007 (US, 4 Ops): DEFCON −1 and the US +2 VP; *if the US controls South Korea* the US may also place "
           "influence or realign with the card's Ops. Each of the US's card choices while holding it, by South "
           "Korea's control at that moment:", "",
           "| | US controls South Korea | does not |", "|:---|---:|---:|"]
    by = {k: [c for c in choices if int(c["feature"]) == k] for k in (1, 0)}
    hl = {k: [c for c in by[k] if c["phase"] == "headline"] for k in by}
    ar = {k: [c for c in by[k] if c["phase"] == "ar"] for k in by}
    out += [f"| headline choices | {len(hl[1]):,} | {len(hl[0]):,} |",
            f"| headlined | {_pct(sum(c['played'] == 'headline' for c in hl[1]), len(hl[1]))} | "
            f"{_pct(sum(c['played'] == 'headline' for c in hl[0]), len(hl[0]))} |",
            f"| action-round choices | {len(ar[1]):,} | {len(ar[0]):,} |"]
    for m, label in (("event", "event in the round"), ("Ops", "Ops in the round"), ("space race", "space race")):
        out.append(f"| {label} | {_pct(sum(c['played'] == m for c in ar[1]), len(ar[1]))} | "
                   f"{_pct(sum(c['played'] == m for c in ar[0]), len(ar[0]))} |")
    return "\n".join(out)


def _pick_table(recs: Sequence[Dict[str, Any]], top: int = 15) -> str:
    n = len(recs)
    if n == 0:
        return "None."
    picked = collections.Counter(r["card"] for r in recs)
    offered = collections.Counter(c for r in recs for c in r["options"])
    out = [f"{n:,} plays, {sum(len(r['options']) for r in recs) / n:.1f} eligible cards in hand on average.", "",
           "| card | played with it | share | eligible in | played when eligible |", "|:---|---:|---:|---:|---:|"]
    for c, k in picked.most_common(top):
        out.append(f"| {CARDS[c]['name']} | {k:,} | {100 * k / n:.1f}% | {100 * offered[c] / n:.0f}% | "
                   f"{100 * k / offered[c]:.0f}% |")
    return "\n".join(out)


def un_section(picks: Sequence[Dict[str, Any]]) -> str:
    out = ["## UN Intervention: the card played with it", "",
           "UN Intervention is played with an opponent card from the same hand, whose Ops are used without its event. "
           "\"Eligible\": the opponent's non-scoring cards in hand at that choice (the engine's options)."]
    for side in ("US", "USSR"):
        out += ["", f"### {side}", "", _pick_table([p for p in picks if p["side"] == side])]
    return "\n".join(out)


def space_section(modes: Dict[str, Dict[str, Dict[str, int]]], top: int = 15) -> str:
    out = ["## Space race: the cards sent", "",
           "Every card played in an action round goes through the play-mode choice; this counts those sent to the "
           "space race. \"Share of its plays\": of that side's plays of the card in a round, the share spent on the "
           "space race."]
    for side in ("US", "USSR"):
        allp, sp = modes["all"][side], modes["space"][side]
        n, tot = sum(sp.values()), sum(allp.values())
        out += ["", f"### {side}", "", f"{n:,} space race plays, {_pct(n, tot)} of the {tot:,} cards it played in a round.", "",
                "| card | sent to space | share of space plays | share of its plays |", "|:---|---:|---:|---:|"]
        for c, k in sorted(sp.items(), key=lambda kv: -kv[1])[:top]:
            out.append(f"| {CARDS[int(c)]['name']} | {k:,} | {_pct(k, n)} | {_pct(k, allp.get(c, 0))} |")
    return "\n".join(out)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(path: str) -> Tuple[str, str, str]:
    """(run short name, step label, file stem) from a checkpoint path."""
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
    short = re.sub(r"_\d{8}_\d{6}.*$", "", run_dir)
    m = re.search(r"(\d+)steps\.pt$", path)
    step = f"{int(m.group(1)) / 1e6:,.0f}M" if m else os.path.splitext(os.path.basename(path))[0]
    stem = f"{short}_{step.replace(',', '')}" if m else f"{short}_{step}"
    return short, step, stem


def report(path: str, d: Dict[str, Any], merged: bool, feats: int,
           label: Optional[str] = None, note: Optional[str] = None) -> str:
    short, step, _ = _label(path)
    title = label or f"{short} @ {step}"
    meta: Dict[str, Any] = {}
    try:
        meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(path)), "metadata.json")))
    except (OSError, ValueError):
        pass
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    holdings = [load_holding(r) for r in d["holdings"]]
    rel = os.path.relpath(path, "/workspace")
    out = [f"# {title}: behaviour report", "",
           f"Generated {datetime.date.today().isoformat()} by `tools/scripts/checkpoint_report.py` (commit {commit}).", "",
           "## The checkpoint", "",
           f"* **File:** `{rel}` (sha256 `{_sha(path)[:12]}…`)",
           f"* **Run:** {note or meta.get('description', '(no metadata.json description)')}"]
    if meta.get("resumed_from"):
        out.append(f"* **Resumed from:** `{os.path.relpath(str(meta['resumed_from']), '/workspace')}`")
    out += [f"* **Action view:** {'merged influence (E4.1)' if merged else 'E4'}; observation feature bits: {feats}",
            f"* **Games:** {d['games']:,} greedy self-play games of the checkpoint against itself, seed "
            f"{d['seed']:,}, batches of {d['batch']}. Every section reads the same games.", "",
            "Sections: Star Wars · Five Year Plan played by the USSR · Aldrich Ames Remix played by the US · OPEC "
            "and Alliance for Progress · Soviets Shoot Down KAL-007 · UN Intervention · Space race · and, last, "
            "Events vs Ops for every card.", "",
            star_wars_section(d["star_wars"]), "",
            opponent_card_section(
                "Five Year Plan played by the USSR",
                "Five Year Plan (US, 3 Ops): the USSR discards a random card from its hand, and if that card's event "
                "is a US event it fires. Here: the USSR playing it from its own hand so that the event fires -- "
                "headlined, or played in a round with the event first or the Ops first. Early War is turns 1-3.",
                d["five_year_plan"], "ussr"), "",
            opponent_card_section(
                "Aldrich Ames Remix played by the US",
                "Aldrich Ames Remix (USSR, 3 Ops): the US reveals its hand for the rest of the turn and the USSR "
                "discards a card of its choice from it. Here: the US playing it from its own hand so that the event "
                "fires (a Late War card, so all plays fall in Mid+Late War).",
                d["aldrich_ames"], "us", show_mode=False), "",
            "## OPEC and Alliance for Progress", "",
            vp_card_section("OPEC (USSR)", "OPEC (USSR, 3 Ops): the USSR gains 1 VP for each of Egypt, Iran, Libya, "
                            "Saudi Arabia, Iraq, the Gulf States and Venezuela it controls (cancelled by North Sea Oil, "
                            "when the event cannot trigger).", d["opec"]), "",
            vp_card_section("Alliance for Progress (US)", "Alliance for Progress (US, 3 Ops): the US gains 1 VP for "
                            "each battleground it controls in Central and South America (Mexico, Panama, Cuba, "
                            "Venezuela, Brazil, Chile, Argentina).", d["alliance_for_progress"]), "",
            kal_section(d["kal_007"]), "",
            un_section(d["un_intervention"]), "",
            space_section(d["play_modes"]), "",
            "## Events vs Ops, every card", "", census_table(holdings, d["games"])]
    return "\n".join(out) + "\n"


def write_index() -> None:
    rows = []
    for f in sorted(os.listdir(REPORT_DIR)):
        if f.endswith(".md") and f != "README.md":
            first = open(os.path.join(REPORT_DIR, f)).readline().lstrip("# ").strip()
            rows.append(f"* [{first}]({f})")
    open(os.path.join(REPORT_DIR, "README.md"), "w").write(
        "# Per-checkpoint behaviour reports\n\nGenerated by `tools/scripts/checkpoint_report.py`: for one "
        "checkpoint, greedy self-play and from it the event census (events vs Ops), Star Wars plays and "
        "retrievals, and when and with what hand the USSR plays Five Year Plan and the US plays Aldrich Ames "
        "Remix.\n\n" + "\n".join(rows) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from bindings.ts_env import model_obs_features
    from tools.lib.action_view import checkpoint_merged_influence
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=55_000)
    ap.add_argument("--load", action="store_true", help="rebuild the reports from the saved records")
    ap.add_argument("--output-dir", default=REPORT_DIR)
    ap.add_argument("--data-dir", default=DATA_DIR, help="where the raw records go (and --load reads them)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="a name per checkpoint, for the title and the file name (default: run @ step); "
                         "needed for a checkpoint outside a run directory, such as an SWA or a soup")
    ap.add_argument("--notes", nargs="+", default=None,
                    help="a description per checkpoint, in place of its run's metadata.json description")
    a = ap.parse_args(argv)
    if a.labels and len(a.labels) != len(a.checkpoints) or a.notes and len(a.notes) != len(a.checkpoints):
        ap.error("--labels and --notes need one entry per checkpoint")
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(a.data_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, path in enumerate(a.checkpoints):
        label = a.labels[k] if a.labels else None
        note = a.notes[k] if a.notes else None
        stem = re.sub(r"[^A-Za-z0-9.+-]+", "_", label).strip("_") if label else _label(path)[2]
        dump = os.path.join(a.data_dir, f"{stem}.json")
        merged = checkpoint_merged_influence(path)
        model = NeuralAgent.from_checkpoint(path, device=str(dev)).model
        feats = model_obs_features(model)
        if a.load:
            d = json.load(open(dump))
        else:
            d = play(model, merged, a.games, a.seed, a.batch)
            json.dump(d, open(dump, "w"))
        md = report(path, d, merged, feats, label, note)
        out = os.path.join(a.output_dir, f"{stem}.md")
        open(out, "w").write(md)
        print(f"{path} -> {out}", flush=True)
    if os.path.abspath(a.output_dir) == os.path.abspath(REPORT_DIR):
        write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
