"""Regression tests for the 2026-11-22 audit's engine findings, fixed for engine E9 (engine/AGENTS.md section 24;
research/log/audit_2026-11-22_verification.md). Each drives a real game into the finding's situation -- seeded random
play to an action round or a headline, then the cards placed -- and checks the fixed behaviour."""

from __future__ import annotations

import random
from typing import Callable, List

import numpy as np
import ts_engine as ts

E = ts.Engine
L = ts.CardLocation
P = ts.Player
DT = ts.DecisionType
CARD = ts.FLAT_CARD
ROLL = ts.FLAT_ROLL_DIE
BR = ts.FLAT_BRANCH
DONE = ts.FLAT_CONFIRM_DONE
EVENT_FIRST = 110   # FLAT_RESOLUTION + 0
OPS = 112           # FLAT_RESOLUTION + 2: Ops (Influence)


def card(name: str) -> int:
    return int(ts.CardData.get_card_by_name(name))


def mask(s: ts.GameState) -> List[int]:
    return [i for i, v in enumerate(E.get_flat_action_mask(s)) if v]


def step(s: ts.GameState, i: int) -> None:
    assert E.try_step_flat(s, i), f"refused {i}"


def rand_play(s: ts.GameState, rng: random.Random, cond: Callable[[ts.GameState], bool], n: int = 20000) -> bool:
    for _ in range(n):
        if E.is_terminal(s):
            return False
        if cond(s):
            return True
        E.step_flat(s, rng.choice(mask(s)))
    return False


def to_ar(seed: int, phasing: ts.Player, lo: int, hi: int) -> ts.GameState:
    for k in range(200):
        s = ts.GameState()
        E.init_game(s, seed + k)
        if rand_play(s, random.Random(seed + k), lambda s: (
                s.current_phase == ts.Phase.ACTION_ROUND and s.ctx_stack_depth == 0
                and s.ctx().decision_type == DT.SELECT_CARD and s.ctx().decision_player == phasing
                and s.phasing_player == phasing and lo <= s.turn <= hi and s.action_round <= 5)):
            return s
    raise AssertionError("no action round reached")


def to_headline(seed: int, min_turn: int) -> ts.GameState:
    for k in range(100):
        s = ts.GameState()
        E.init_game(s, seed + k)
        if rand_play(s, random.Random(seed + k), lambda s: (
                s.current_phase == ts.Phase.HEADLINE and s.turn >= min_turn and s.headline_stage == 0
                and s.ctx().decision_type == DT.SELECT_CARD and s.headline_us_card == 0
                and s.headline_ussr_card == 0)):
            return s
    raise AssertionError("no headline reached")


def hand(s: ts.GameState, p: ts.Player) -> List[int]:
    locs = (L.HAND_US_KNOWN, L.HAND_US_UNKNOWN) if p == P.US else (L.HAND_USSR_KNOWN, L.HAND_USSR_UNKNOWN)
    return [c for c in range(1, 111) if s.get_card_location(c) in locs]


def set_hand(s: ts.GameState, p: ts.Player, cards: List[int]) -> None:
    loc = L.HAND_US_UNKNOWN if p == P.US else L.HAND_USSR_UNKNOWN
    for c in hand(s, p):
        if c != 6:
            s.set_card_location(c, L.DRAW_DECK)
    for c in cards:
        s.set_card_location(c, loc)


def headline(s: ts.GameState, us_card: int, ussr_card: int) -> None:
    for _ in range(2):
        step(s, CARD + (us_card if s.ctx().decision_player == P.US else ussr_card) - 1)


def advance_to(s: ts.GameState, cond: Callable[[ts.GameState], bool], n: int = 30) -> None:
    for _ in range(n):
        if cond(s):
            return
        step(s, mask(s)[0])
    assert cond(s), "situation not reached"


def grain_sales_offer(s: ts.GameState, gs: int) -> None:
    advance_to(s, lambda s: s.ctx().resolving_card == gs and s.ctx().decision_type == DT.SELECT_PLAY_MODE)


def olympics_after_grain_sales() -> ts.GameState:
    """A USSR Grain Sales draws Olympic Games for the US, who plays it as its Event; the USSR participates."""
    gs, og = card("Grain Sales to Soviets"), card("Olympic Games")
    s = to_ar(100, P.USSR, 4, 7)
    set_hand(s, P.USSR, [gs, og])
    step(s, CARD + gs - 1)
    step(s, EVENT_FIRST)
    advance_to(s, lambda s: s.ctx().decision_type == DT.SELECT_PLAY_MODE and s.ctx().pending_op_card == og)
    step(s, EVENT_FIRST)
    return s


def test_e1_the_olympic_sponsor_is_the_player_whose_event_it_is() -> None:
    s = olympics_after_grain_sales()
    step(s, BR + 0)   # the USSR participates
    vp0 = s.victory_points
    win = s.clone()   # the sponsor's die 6 (+2) against 1: the sponsor -- the US -- wins
    assert E.try_step(win, ts.MicroAction(DT.ROLL_DIE, 6, 1, 0))
    assert win.victory_points - vp0 == 2
    lose = s.clone()
    assert E.try_step(lose, ts.MicroAction(DT.ROLL_DIE, 1, 6, 0))
    assert lose.victory_points - vp0 == -2


def test_e8_grain_sales_leaves_no_stray_decline_at_the_drawn_cards_branch() -> None:
    s = olympics_after_grain_sales()
    assert s.ctx().decision_type == DT.CHOOSE_BRANCH and DONE not in mask(s)


def test_e2_star_wars_runs_a_ussr_war_as_the_ussrs() -> None:
    sw, kw = card("Star Wars"), card("Korean War")
    s = to_ar(500, P.US, 8, 10)
    set_hand(s, P.US, [sw, card("CIA Created")])
    s.set_card_location(kw, L.DISCARD_PILE)
    s.us_space_track = 5
    s.ussr_space_track = 2
    step(s, CARD + sw - 1)
    step(s, EVENT_FIRST)
    step(s, CARD + kw - 1)
    assert s.ctx().roll_actor == P.USSR
    mil = (s.us_mil_ops, s.ussr_mil_ops)
    step(s, ROLL)
    assert (s.us_mil_ops, s.ussr_mil_ops) == (mil[0], mil[1] + 2)


def _grain_sales_with(drawn: int, seed: int) -> ts.GameState:
    gs = card("Grain Sales to Soviets")
    s = to_ar(seed, P.USSR, 4, 7)
    set_hand(s, P.USSR, [gs, drawn])
    return s


def test_e3_we_will_bury_you_waits_for_the_us_round() -> None:
    gs, un, cia = card("Grain Sales to Soviets"), card("UN Intervention"), card("CIA Created")
    for mode in (EVENT_FIRST, OPS):
        s = _grain_sales_with(cia, 600)
        set_hand(s, P.US, hand(s, P.US) + [un])
        s.set_flag(ts.EffectBits.WE_WILL_BURY_YOU_PENDING)
        vp0 = s.victory_points
        step(s, CARD + gs - 1)
        step(s, OPS)
        grain_sales_offer(s, gs)
        step(s, mode)
        assert s.has_flag(ts.EffectBits.WE_WILL_BURY_YOU_PENDING), mode
        assert s.victory_points == vp0, mode


def test_e4_only_missile_envy_discharges_its_obligation() -> None:
    gs, me, cia = card("Grain Sales to Soviets"), card("Missile Envy"), card("CIA Created")
    s = _grain_sales_with(cia, 700)
    set_hand(s, P.US, hand(s, P.US) + [me])
    s.forced_card_player = P.US
    s.forced_card_id = me
    step(s, CARD + gs - 1)
    step(s, OPS)
    grain_sales_offer(s, gs)
    step(s, OPS)
    assert s.forced_card_player == P.US and s.forced_card_id == me


def test_e5_five_year_plan_puts_shuttle_diplomacy_in_play() -> None:
    fyp, sd = card("Five Year Plan"), card("Shuttle Diplomacy")
    s = to_ar(200, P.USSR, 4, 7)
    set_hand(s, P.USSR, [fyp, sd])
    step(s, CARD + fyp - 1)
    step(s, EVENT_FIRST)
    assert s.get_card_location(sd) == L.ONGOING_EVENT
    assert s.has_flag(ts.EffectBits.SHUTTLE_DIPLOMACY_ACTIVE)


def test_e6_the_debt_crisis_never_counts_itself_as_the_discard() -> None:
    ladc, cia = card("Latin American Debt Crisis"), card("CIA Created")
    for mode in (EVENT_FIRST, OPS):
        s = to_ar(400, P.US, 8, 10)
        set_hand(s, P.US, [ladc, cia])
        s.set_flag(ts.EffectBits.CONTAINMENT_ACTIVE)   # +1: Latin American Debt Crisis itself reaches 3 Ops
        step(s, CARD + ladc - 1)
        step(s, mode)
        advance_to(s, lambda s: s.ctx().resolving_card == ladc, 12)
        # No 3-Ops card but the one being played: straight to the USSR's doubling, no US prompt.
        assert s.ctx().decision_player == P.USSR and s.ctx().decision_type == DT.POINT_NODE, mode


def test_e7_missile_envy_spent_in_a_headline_is_discarded() -> None:
    gs, me = card("Grain Sales to Soviets"), card("Missile Envy")
    base = to_headline(900, 4)
    for trial in range(200):
        s = base.clone()
        s.rng_state = 1000 + trial
        set_hand(s, P.US, [me, card("CIA Created"), card("Duck and Cover"), card("NATO")])
        set_hand(s, P.USSR, [gs, card("Comecon"), card("Decolonization"), card("Fidel")])
        headline(s, me, gs)
        for _ in range(40):
            if (s.ctx().resolving_card == gs and s.ctx().decision_type == DT.SELECT_PLAY_MODE) or \
                    s.current_phase != ts.Phase.HEADLINE:
                break
            step(s, mask(s)[0])
        if not (s.ctx().resolving_card == gs and s.ctx().pending_op_card == me):
            continue
        step(s, OPS)
        advance_to(s, lambda s: s.current_phase != ts.Phase.HEADLINE, 40)
        assert s.get_card_location(me) == L.DISCARD_PILE
        return
    raise AssertionError("Grain Sales never drew Missile Envy")


def test_e9_an_owed_event_leaves_no_start_of_round_snapshot() -> None:
    cia = card("CIA Created")
    s = to_ar(1000, P.USSR, 2, 3)
    set_hand(s, P.USSR, [cia, card("Comecon")])
    step(s, CARD + cia - 1)
    step(s, OPS)
    while s.ctx().decision_type == DT.POINT_NODE and s.ctx().decision_player == P.USSR:
        step(s, mask(s)[0])
    sd = s.to_save_dict()
    assert sd["ctx_stack"][sd["ctx_stack_depth"]]["start_influence_nodes"] == [0, 0]


def test_e10_flower_power_charges_a_us_headline_of_a_ussr_war() -> None:
    base = to_headline(1100, 5)
    for war in (card("Korean War"), card("Arab-Israeli War")):
        s = base.clone()
        s.set_flag(ts.EffectBits.FLOWER_POWER_ACTIVE)
        set_hand(s, P.US, [war, card("NATO"), card("Duck and Cover")])
        set_hand(s, P.USSR, [card("Fidel"), card("Comecon"), card("Decolonization")])
        vp0 = s.victory_points
        headline(s, war, card("Fidel"))
        while s.ctx().decision_type == DT.ROLL_DIE and s.current_phase == ts.Phase.HEADLINE:
            step(s, ROLL)
        war_vp = -2 if s.to_save_dict()["last_roll"]["success"] else 0
        assert s.victory_points - vp0 == war_vp - 2, war


def test_e11_a_headlined_missile_envy_uses_un_intervention_for_ops() -> None:
    me, un = card("Missile Envy"), card("UN Intervention")
    s = to_headline(1200, 4)
    set_hand(s, P.US, [me, card("Decolonization"), card("NATO")])
    set_hand(s, P.USSR, [un, card("Asia Scoring"), card("Fidel")])
    headline(s, me, card("Fidel"))
    advance_to(s, lambda s: s.ctx().pending_op_card == un or s.ctx().resolving_card == un
               or s.current_phase != ts.Phase.HEADLINE, 10)
    assert s.ctx().resolving_card != un
    assert s.ctx().pending_op_card == un and s.ctx().decision_type == DT.SELECT_OP_MODE


def _peeked(o: np.ndarray) -> List[int]:
    cards = np.asarray(o)[84 * 26:84 * 26 + 110 * 14].reshape(110, 14)
    return [i + 1 for i in range(110) if cards[i, 6] == 1.0]


def test_e14_our_man_in_tehran_is_the_uss_peek_alone() -> None:
    omt = card("Our Man in Tehran")
    s = to_ar(1300, P.USSR, 8, 10)
    s.set_country(int(ts.MapData.get_country_by_name("Iran")), 5, 0)   # the US controls Iran
    set_hand(s, P.USSR, [omt, card("Comecon")])
    step(s, CARD + omt - 1)
    step(s, EVENT_FIRST)
    peek = [c for c in range(1, 111) if s.get_card_location(c) == L.PEEKED_TEMP]
    assert peek and _peeked(ts.extract_observation(s, P.US)) == peek
    assert _peeked(ts.extract_observation(s, P.USSR)) == []


def test_e16_the_first_headline_picker_sees_its_own_pick() -> None:
    s = to_headline(1400, 3)
    p = s.ctx().decision_player
    before = np.asarray(ts.extract_observation_features(s, p, 4))
    outs: List[np.ndarray] = []
    for c in [c for c in hand(s, p) if c != 6][:4]:
        t = s.clone()
        step(t, CARD + c - 1)
        outs.append(np.asarray(ts.extract_observation_features(t, p, 4)))
    assert not np.array_equal(outs[0], before)
    assert len({o.tobytes() for o in outs}) == len(outs)


def test_e17_suez_under_a_canonical_order_auto_advances_to_its_end() -> None:
    suez = card("Suez Crisis")
    names = {ts.MapData.get_country_name(i): i for i in range(84)}
    uk = [i for n, i in names.items() if "United Kingdom" in n or n == "UK"][0]
    fr, isr = names["France"], names["Israel"]
    for feat in (ts.OBS_FEATURE_CANONICAL_FIRST, ts.OBS_FEATURE_CANONICAL_LAST, 0):
        s = to_ar(1500, P.USSR, 2, 3)
        s.seat_obs_features(P.USSR, feat)
        s.seat_obs_features(P.US, 0)
        for cid, inf in ((uk, 1), (fr, 1), (isr, 2)):
            s.set_country(cid, inf, s.get_country(cid).ussr_influence)
        set_hand(s, P.USSR, [suez, card("Comecon")])
        step(s, CARD + suez - 1)
        step(s, EVENT_FIRST)
        E.auto_advance_step(s, 20)
        assert (s.get_country(uk).us_influence, s.get_country(fr).us_influence,
                s.get_country(isr).us_influence) == (0, 0, 0), feat
        assert s.ctx().resolving_card != suez, feat


def test_e19_an_eighth_round_win_is_not_a_held_scoring_loss() -> None:
    s = ts.GameState()
    E.init_game(s, 5)
    s.set_card_location(card("Asia Scoring"), ts.hand_of(P.USSR))
    s.current_phase = ts.Phase.GAME_OVER
    s.turn = 6
    s.action_round = 8
    s.victory_points = 20
    assert not E.is_held_scoring_game_over(s)
    s.set_flag(ts.EffectBits.HELD_SCORING_LOSS)
    assert E.is_held_scoring_game_over(s) and E.is_held_scoring_loss(s, P.USSR)




def _europe() -> List[int]:
    return [i for i in range(84) if ts.MapData.get_country_info(i)["region"] == 0]


def test_t13_europe_control_is_booked_as_forty_whatever_the_score() -> None:
    """Audit T-13 / E9: the Event that wins by Europe Control records +/-40 VP for Europe scoring, not the track's
    jump to +/-20 (which depended on the score before); the game ends at 20 as Europe Control."""
    eu = card("Europe Scoring")
    for vp_before in (-15, 0, 12):
        s = to_ar(1600, P.US, 4, 7)
        for i in _europe():
            s.set_country(i, 20, 0)
        s.victory_points = vp_before
        set_hand(s, P.US, [eu, card("NATO")])
        step(s, CARD + eu - 1)
        sd = s.to_save_dict()
        tracks = [t for t in sd["last_event_tracks"] if t["card_id"] == eu]
        assert tracks and tracks[0]["vp_delta"] == 40, (vp_before, tracks)
        assert s.victory_points == 20 and str(ts.game_ending_reason(s)) == "Europe Control"


def test_t10_wargames_and_final_scoring_are_read_from_what_happened() -> None:
    wg = card("Wargames")
    s = to_ar(1700, P.US, 8, 10)
    s.defcon = 2
    set_hand(s, P.US, [wg, card("NATO")])
    step(s, CARD + wg - 1)
    advance_to(s, lambda s: s.ctx().resolving_card == wg and s.ctx().decision_type == DT.CHOOSE_BRANCH, 5)
    step(s, BR + 0)
    assert E.is_terminal(s) and str(ts.game_ending_reason(s)) == "wargames"
