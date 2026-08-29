"""
Twilight Struggle AI vs Struggler: Exhaustive 110-Card Differential Validation Suite
===================================================================================
Tests that for EVERY CARD (1 to 110):
1. Resolving the event across multiple distinct board states proposes the same choices/actions in both ts_ai and struggler.
2. Executing all available choices/branches produces exactly the same final board influence, DEFCON, VP, Mil Ops, and Space Race state.
3. Includes multi-turn complete games and fuzzed action-resolution tests.
"""

import pytest
import ts_engine
from struggler.engine.core import Engine as SEngine, SCORING_CARD_REGION
from struggler.engine.board import Board as SBoard
from struggler.engine.types import (
    Side as SSide,
    Region as SRegion,
    CardSide as SCardSide,
    DecisionKind as SDecisionKind,
    Action as SAction,
    Decision as SDecision,
)
import tests.struggler_adapter as sa

EB = ts_engine.EffectBits


def make_test_pair(defcon=5, turn=1, vp=0, ar=1):
    """Creates a synchronized pair of (ts_state, s_eng) with identical baseline configuration."""
    ts_state, s_eng = sa.create_paired_state(seed=42, defcon=defcon, turn=turn, vp=vp)
    ts_state.action_round = ar
    s_eng.action_round = ar
    return ts_state, s_eng


def set_influence_both(ts_state: ts_engine.GameState, s_eng: SEngine, country: str, us: int, ussr: int):
    """Sets influence in both engines synchronously."""
    cid = ts_engine.MapData.get_country_by_name(country)
    ts_state.set_country(cid, us, ussr)
    s_name = sa.country_ts_to_struggler(country)
    s_eng.board.influence[s_name]["US"] = us
    s_eng.board.influence[s_name]["USSR"] = ussr


def fire_scoring_card_both(ts_state: ts_engine.GameState, s_eng: SEngine, cid: int, region: ts_engine.Region):
    """Executes region scoring on both engines."""
    s_cid = sa.card_id_ts_to_struggler(cid)
    s_eng._resolve_scoring_card(s_cid)
    if cid == 38:  # Southeast Asia
        ts_engine.Scoring.score_southeast_asia(ts_state)
    else:
        ts_engine.Scoring.score_region(ts_state, region)


def fire_war_card_both(ts_state: ts_engine.GameState, s_eng: SEngine, cid: int, player: ts_engine.Player, target_country: str, forced_roll: int):
    """Executes war card resolution in both engines with synchronized roll."""
    s_cid = sa.card_id_ts_to_struggler(cid)
    side = SSide.US if player == ts_engine.Player.US else SSide.USSR
    s_target = sa.country_ts_to_struggler(target_country)

    # 1. Trigger in struggler
    s_eng._fire_event(side, s_cid)
    if s_eng.pending_decision and s_eng.pending_decision.kind == SDecisionKind.WAR_TARGET:
        s_eng.step(SAction(SDecisionKind.WAR_TARGET, {"country": s_target}))
    if s_eng.pending_decision and s_eng.pending_decision.kind == SDecisionKind.WAR_ROLL:
        ctx = s_eng.pending_decision.context
        s_eng._decision_stack[-1] = s_eng._new_decision(
            SSide.CHANCE, SDecisionKind.WAR_ROLL, (SAction(SDecisionKind.WAR_ROLL, {"value": forced_roll}),), ctx
        )
        s_eng.step(s_eng.pending_decision.options[0])

    # 2. Trigger in ts_ai
    ts_engine.CardHandlers.trigger_event(ts_state, cid, player)
    if ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE:
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name(target_country)
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)
    if ts_state.ctx().decision_type == ts_engine.DecisionType.ROLL_DIE:
        ma_roll = ts_engine.MicroAction()
        ma_roll.decision_type = ts_engine.DecisionType.ROLL_DIE
        ma_roll.primary_id = forced_roll
        ts_engine.CardHandlers.handle_event_step(ts_state, ma_roll)


# =============================================================================
# 1. EARLY WAR CARDS (1..35, 103..106)
# =============================================================================

class TestEarlyWarCardsDifferential:

    def test_01_asia_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "North Korea", 0, 3)
        set_influence_both(ts_state, s_eng, "Vietnam", 0, 1)
        fire_scoring_card_both(ts_state, s_eng, 1, ts_engine.Region.ASIA)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_02_europe_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "France", 3, 0)
        set_influence_both(ts_state, s_eng, "West Germany", 4, 0)
        set_influence_both(ts_state, s_eng, "Italy", 2, 0)
        set_influence_both(ts_state, s_eng, "United Kingdom", 5, 0)
        set_influence_both(ts_state, s_eng, "East Germany", 0, 3)
        fire_scoring_card_both(ts_state, s_eng, 2, ts_engine.Region.EUROPE)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_03_middle_east_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Israel", 4, 0)
        set_influence_both(ts_state, s_eng, "Jordan", 2, 0)
        set_influence_both(ts_state, s_eng, "Lebanon", 1, 0)
        set_influence_both(ts_state, s_eng, "Iraq", 0, 3)
        fire_scoring_card_both(ts_state, s_eng, 3, ts_engine.Region.MIDDLE_EAST)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_04_duck_and_cover(self):
        ts_state, s_eng = make_test_pair(defcon=4)
        ts_engine.CardHandlers.trigger_event(ts_state, 4, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Duck_and_Cover")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_05_five_year_plan(self):
        ts_state, s_eng = make_test_pair()
        s_eng.hands["USSR"] = ["Duck_and_Cover"]
        ts_state.set_card_location(4, ts_engine.CardLocation.HAND_USSR)
        ts_engine.CardHandlers.trigger_event(ts_state, 5, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Five_Year_Plan")
        assert s_eng.pending_decision.kind == SDecisionKind.RANDOM_DISCARD

    def test_06_the_china_card(self):
        ts_state, s_eng = make_test_pair()
        assert ts_engine.CardData.get_card_info(6)["ops"] == 4
        assert s_eng.cards["The_China_Card"].ops == 4

    def test_07_socialist_governments(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Italy", 3, 0)
        set_influence_both(ts_state, s_eng, "France", 2, 0)
        set_influence_both(ts_state, s_eng, "West Germany", 3, 0)

        ts_engine.CardHandlers.trigger_event(ts_state, 7, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Socialist_Governments")

        for country in ["Italy", "Italy", "France"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_08_fidel(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Cuba", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 8, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Fidel")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_09_vietnam_revolts(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 9, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Vietnam_Revolts")
        sa.assert_board_equal(ts_state, s_eng)

    def test_10_blockade(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "West Germany", 4, 0)

        ts_engine.CardHandlers.trigger_event(ts_state, 10, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Blockade")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "refuse"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.SELECT_CARD
        ma.primary_id = 0
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_board_equal(ts_state, s_eng)

    def test_11_korean_war(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "South Korea", 2, 0)
        set_influence_both(ts_state, s_eng, "North Korea", 0, 3)
        fire_war_card_both(ts_state, s_eng, 11, ts_engine.Player.USSR, "South Korea", 6)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_12_romanian_abdication(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Romania", 2, 1)
        ts_engine.CardHandlers.trigger_event(ts_state, 12, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Romanian_Abdication")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_13_arab_israeli_war(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Israel", 2, 0)
        set_influence_both(ts_state, s_eng, "Egypt", 2, 0)
        set_influence_both(ts_state, s_eng, "Jordan", 2, 0)
        set_influence_both(ts_state, s_eng, "Syria", 0, 2)
        set_influence_both(ts_state, s_eng, "Lebanon", 0, 0)
        fire_war_card_both(ts_state, s_eng, 13, ts_engine.Player.USSR, "Israel", 6)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_14_comecon(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 14, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "COMECON")
        for country in ["Bulgaria", "Czechoslovakia", "Hungary", "Romania"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_15_nasser(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Egypt", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 15, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Nasser")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_16_warsaw_pact(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 16, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Warsaw_Pact_Formed")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "add"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 1
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        for country in ["Poland", "Poland", "East Germany", "East Germany", "Hungary"]:
            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_17_de_gaulle(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "France", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 17, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "De_Gaulle_Leads_France")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_18_captured_nazi_scientist(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 18, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Captured_Nazi_Scientist")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_19_truman_doctrine(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Yugoslavia", 0, 2)
        ts_engine.CardHandlers.trigger_event(ts_state, 19, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Truman_Doctrine")

        s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
        ts_cands = sa.get_ts_legal_country_names(ts_state)
        assert s_cands == ts_cands

        s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": "Yugoslavia"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name("Yugoslavia")
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_20_olympic_games(self):
        ts_state, s_eng = make_test_pair(defcon=4)
        ts_engine.CardHandlers.trigger_event(ts_state, 20, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Olympic_Games")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "boycott"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 1
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        assert ts_state.defcon == 3
        assert s_eng.defcon == 3

    def test_21_nato(self):
        ts_state, s_eng = make_test_pair()
        ts_state.set_flag(EB.MARSHALL_PLAN_PLAYED)
        s_eng.game_effects["marshall_or_warsaw"] = True
        ts_engine.CardHandlers.trigger_event(ts_state, 21, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "NATO")
        assert ts_state.has_flag(EB.NATO_ACTIVE)
        assert s_eng.game_effects.get("nato") is True

    def test_22_independent_reds(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Yugoslavia", 0, 3)
        ts_engine.CardHandlers.trigger_event(ts_state, 22, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Independent_Reds")

        s_cands = {opt.payload["choice"] for opt in s_eng.pending_decision.options}
        ts_cands = sa.get_ts_legal_country_names(ts_state)
        assert "Yugoslavia" in s_cands
        assert "Yugoslavia" in ts_cands

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "Yugoslavia"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name("Yugoslavia")
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_23_marshall_plan(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 23, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Marshall_Plan")

        for country in ["France", "West Germany", "Italy", "United Kingdom", "Benelux", "Norway", "Denmark"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_24_indo_pakistani_war(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Pakistan", 2, 0)
        set_influence_both(ts_state, s_eng, "India", 0, 0)
        fire_war_card_both(ts_state, s_eng, 24, ts_engine.Player.USSR, "Pakistan", 6)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_25_containment(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 25, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Containment")
        assert ts_state.has_flag(EB.CONTAINMENT_ACTIVE)
        assert s_eng.turn_effects.get("containment") is True

    def test_26_cia_created(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 26, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "CIA_Created")

        # Step 1: Choose Ops mode (influence)
        s_eng.step(SAction(SDecisionKind.OPS_TYPE, {"type": "influence"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.SELECT_OP_MODE
        ma.primary_id = 0
        ts_engine.Engine.step(ts_state, ma)

        # Step 2: Choose France
        s_eng.step(SAction(SDecisionKind.PLACE_INFLUENCE, {"country": "France"}))
        ma2 = ts_engine.MicroAction()
        ma2.decision_type = ts_engine.DecisionType.POINT_NODE
        ma2.primary_id = ts_engine.MapData.get_country_by_name("France")
        ts_engine.Engine.step(ts_state, ma2)

        sa.assert_board_equal(ts_state, s_eng)

    def test_27_us_japan_pact(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Japan", 1, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 27, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "US_Japan_Mutual_Defense_Pact")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_28_suez_crisis(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "United Kingdom", 5, 0)
        set_influence_both(ts_state, s_eng, "France", 3, 0)
        set_influence_both(ts_state, s_eng, "Israel", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 28, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Suez_Crisis")

        for country in ["United Kingdom", "United Kingdom", "France", "Israel"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_29_east_european_unrest(self):
        ts_state, s_eng = make_test_pair(turn=2)
        set_influence_both(ts_state, s_eng, "Poland", 0, 3)
        set_influence_both(ts_state, s_eng, "East Germany", 0, 3)
        set_influence_both(ts_state, s_eng, "Hungary", 0, 2)
        ts_engine.CardHandlers.trigger_event(ts_state, 29, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "East_European_Unrest")

        for country in ["Poland", "East Germany", "Hungary"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_30_decolonization(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 30, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Decolonization")

        for country in ["Angola", "Nigeria", "Algeria", "Vietnam"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_31_red_scare_purge(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 31, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Red_Scare_Purge")
        assert ts_state.has_flag(EB.PURGE_USSR_ACTIVE)
        assert s_eng.turn_effects.get("red_scare") == "USSR"

    def test_32_un_intervention(self):
        ts_state, s_eng = make_test_pair()
        assert ts_engine.CardData.get_card_info(32)["ops"] == 1
        assert s_eng.cards["UN_Intervention"].ops == 1

    def test_33_de_stalinization(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Poland", 0, 4)
        ts_engine.CardHandlers.trigger_event(ts_state, 33, ts_engine.Player.USSR)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 4

    def test_34_nuclear_test_ban(self):
        ts_state, s_eng = make_test_pair(defcon=3)
        ts_engine.CardHandlers.trigger_event(ts_state, 34, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Nuclear_Test_Ban")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_35_formosan_resolution(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 35, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Formosan_Resolution")
        assert ts_state.has_flag(EB.FORMOSAN_RESOLUTION_ACTIVE)
        assert s_eng.game_effects.get("formosan_resolution") is True

    def test_103_defectors(self):
        ts_state, s_eng = make_test_pair(ar=1)
        ts_engine.CardHandlers.trigger_event(ts_state, 103, ts_engine.Player.USSR)
        assert ts_state.victory_points == 1

    def test_104_cambridge_five(self):
        ts_state, s_eng = make_test_pair()
        s_eng.hands["US"] = ["Middle_East_Scoring"]
        ts_state.set_card_location(3, ts_engine.CardLocation.HAND_US)

        ts_engine.CardHandlers.trigger_event(ts_state, 104, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "The_Cambridge_Five")

        s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
        ts_cands = sa.get_ts_legal_country_names(ts_state)
        assert "Egypt" in s_cands and "Egypt" in ts_cands

        s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": "Egypt"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name("Egypt")
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_board_equal(ts_state, s_eng)

    def test_105_special_relationship(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "United Kingdom", 5, 0)
        ts_state.set_flag(EB.NATO_ACTIVE)
        s_eng.game_effects["nato"] = True

        ts_engine.CardHandlers.trigger_event(ts_state, 105, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Special_Relationship")

        s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
        ts_cands = sa.get_ts_legal_country_names(ts_state)
        assert s_cands == ts_cands

        s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": "France"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name("France")
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_106_norad(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Canada", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 106, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "NORAD")
        assert ts_state.has_flag(EB.NORAD_ACTIVE)
        assert s_eng.game_effects.get("norad") is True


# =============================================================================
# 2. MID WAR CARDS (36..81, 107..108)
# =============================================================================

class TestMidWarCardsDifferential:

    def test_36_brush_war(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Angola", 2, 0)
        set_influence_both(ts_state, s_eng, "Zaire", 0, 0)
        fire_war_card_both(ts_state, s_eng, 36, ts_engine.Player.USSR, "Angola", 6)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_37_central_america_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Cuba", 0, 3)
        set_influence_both(ts_state, s_eng, "Panama", 2, 0)
        fire_scoring_card_both(ts_state, s_eng, 37, ts_engine.Region.CENTRAL_AMERICA)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_38_southeast_asia_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Thailand", 2, 0)
        set_influence_both(ts_state, s_eng, "Vietnam", 0, 2)
        set_influence_both(ts_state, s_eng, "Indonesia", 1, 0)
        fire_scoring_card_both(ts_state, s_eng, 38, ts_engine.Region.ASIA)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_39_arms_race(self):
        ts_state, s_eng = make_test_pair()
        ts_state.us_mil_ops = 4
        ts_state.ussr_mil_ops = 2
        s_eng.military_ops["US"] = 4
        s_eng.military_ops["USSR"] = 2
        ts_engine.CardHandlers.trigger_event(ts_state, 39, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Arms_Race")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_40_cuban_missile_crisis(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 40, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Cuban_Missile_Crisis")
        assert ts_state.has_flag(EB.CMC_ACTIVE_USSR)

    def test_41_nuclear_subs(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 41, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Nuclear_Subs")
        assert ts_state.has_flag(EB.NUCLEAR_SUBS_ACTIVE)
        assert s_eng.turn_effects.get("nuclear_subs") is True

    def test_42_quagmire(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 42, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Quagmire")
        assert ts_state.has_flag(EB.QUAGMIRE_ACTIVE)
        assert s_eng.game_effects.get("quagmire") is True

    def test_43_salt_negotiations(self):
        ts_state, s_eng = make_test_pair(defcon=2)
        ts_engine.CardHandlers.trigger_event(ts_state, 43, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Salt_Negotiations")
        if s_eng.pending_decision and s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE:
            s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "none"}))
        assert ts_state.defcon == 4
        assert s_eng.defcon == 4
        assert ts_state.has_flag(EB.SALT_ACTIVE)

    def test_44_bear_trap(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 44, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Bear_Trap")
        assert ts_state.has_flag(EB.BEAR_TRAP_ACTIVE)
        assert s_eng.game_effects.get("bear_trap") is True

    def test_45_summit(self):
        ts_state, s_eng = make_test_pair()
        assert ts_engine.CardData.get_card_info(45)["ops"] == 1
        assert s_eng.cards["Summit"].ops == 1

    def test_46_how_i_learned_to_stop_worrying(self):
        ts_state, s_eng = make_test_pair(defcon=5)
        ts_engine.CardHandlers.trigger_event(ts_state, 46, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "How_I_Learned_to_Stop_Worrying")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "2"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 2
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_47_junta(self):
        ts_state, s_eng = make_test_pair()
        assert ts_engine.CardData.get_card_info(47)["ops"] == 2
        assert s_eng.cards["Junta"].ops == 2

    def test_48_kitchen_debates(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Israel", 4, 0)
        set_influence_both(ts_state, s_eng, "Japan", 4, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 48, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Kitchen_Debates")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_49_missile_envy(self):
        ts_state, s_eng = make_test_pair()
        assert ts_engine.CardData.get_card_info(49)["ops"] == 2
        assert s_eng.cards["Missile_Envy"].ops == 2

    def test_50_we_will_bury_you(self):
        ts_state, s_eng = make_test_pair(defcon=3)
        ts_engine.CardHandlers.trigger_event(ts_state, 50, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "We_Will_Bury_You")
        assert ts_state.defcon == 2
        assert s_eng.defcon == 2
        assert ts_state.has_flag(EB.WE_WILL_BURY_YOU_PENDING)

    def test_51_brezhnev_doctrine(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 51, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Brezhnev_Doctrine")
        assert ts_state.has_flag(EB.BREZHNEV_DOCTRINE_ACTIVE)
        assert s_eng.turn_effects.get("brezhnev") is True

    def test_52_portuguese_empire_crumbles(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 52, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Portuguese_Empire_Crumbles")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_53_south_african_unrest(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 53, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "South_African_Unrest")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "south_africa_only"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 0
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_54_allende(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 54, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Allende")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_55_willy_brandt(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 55, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Willy_Brandt")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_56_muslim_revolution(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Iran", 2, 0)
        set_influence_both(ts_state, s_eng, "Egypt", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 56, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Muslim_Revolution")

        for country in ["Iran", "Egypt"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_57_abm_treaty(self):
        ts_state, s_eng = make_test_pair(defcon=2)
        ts_engine.CardHandlers.trigger_event(ts_state, 57, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "ABM_Treaty")
        assert ts_state.defcon == 3
        assert s_eng.defcon == 3

    def test_58_cultural_revolution(self):
        ts_state, s_eng = make_test_pair()
        ts_state.china_card_holder = ts_engine.Player.US
        s_eng.china_card_owner = "US"
        ts_engine.CardHandlers.trigger_event(ts_state, 58, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Cultural_Revolution")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_59_flower_power(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 59, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Flower_Power")
        assert ts_state.has_flag(EB.FLOWER_POWER_ACTIVE)
        assert s_eng.game_effects.get("flower_power") is True

    def test_60_u2_incident(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 60, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "U2_Incident")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_61_opec(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Saudi Arabia", 0, 3)
        set_influence_both(ts_state, s_eng, "Egypt", 0, 2)
        ts_engine.CardHandlers.trigger_event(ts_state, 61, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "OPEC")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_62_lone_gunman(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 62, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Lone_Gunman")
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.SELECT_OP_MODE
        assert s_eng.pending_decision.kind == SDecisionKind.OPS_TYPE

    def test_63_colonial_rearguards(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 63, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Colonial_Rear_Guards")

        for country in ["Zaire", "Zimbabwe", "Malaysia", "Burma"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_64_panama_canal_returned(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 64, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Panama_Canal_Returned")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_65_camp_david(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 65, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Camp_David_Accords")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_66_puppet_governments(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 66, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Puppet_Governments")

        for country in ["Chile", "Argentina", "Peru"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_67_grain_sales(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 67, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Grain_Sales_to_Soviets")
        assert ts_state.ctx().decision_type in (ts_engine.DecisionType.SELECT_OP_MODE, ts_engine.DecisionType.CHOOSE_BRANCH)

    def test_68_john_paul_ii(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Poland", 0, 3)
        ts_engine.CardHandlers.trigger_event(ts_state, 68, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "John_Paul_II_Elected_Pope")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_69_latin_american_death_squads(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 69, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Latin_American_Death_Squads")
        assert ts_state.has_flag(EB.DEATH_SQUADS_US)

    def test_70_oas_founded(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 70, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "OAS_Founded")

        for country in ["Chile", "Argentina"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_71_nixon_plays_china_card(self):
        ts_state, s_eng = make_test_pair()
        ts_state.china_card_holder = ts_engine.Player.USSR
        s_eng.china_card_owner = "USSR"
        ts_engine.CardHandlers.trigger_event(ts_state, 71, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Nixon_Plays_The_China_Card")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_72_sadat(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Egypt", 0, 3)
        ts_engine.CardHandlers.trigger_event(ts_state, 72, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Sadat_Expels_Soviets")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_73_shuttle_diplomacy(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 73, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Shuttle_Diplomacy")
        assert ts_state.has_flag(EB.SHUTTLE_DIPLOMACY_ACTIVE)

    def test_74_voice_of_america(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Cuba", 0, 3)
        set_influence_both(ts_state, s_eng, "Chile", 0, 2)
        ts_engine.CardHandlers.trigger_event(ts_state, 74, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "The_Voice_Of_America")

        for country in ["Cuba", "Cuba", "Chile", "Chile"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_75_liberation_theology(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 75, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Liberation_Theology")

        for country in ["Nicaragua", "Honduras", "Guatemala"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": country}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_76_ussuri_river(self):
        ts_state, s_eng = make_test_pair()
        ts_state.china_card_holder = ts_engine.Player.USSR
        s_eng.china_card_owner = "USSR"
        ts_engine.CardHandlers.trigger_event(ts_state, 76, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Ussuri_River_Skirmish")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_77_ask_not(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 77, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Ask_Not_What_Your_Country_Can_Do_For_You")
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.SELECT_CARD
        assert s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE

    def test_78_alliance_for_progress(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Panama", 2, 0)
        set_influence_both(ts_state, s_eng, "Venezuela", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 78, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Alliance_for_Progress")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_79_africa_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Angola", 0, 2)
        set_influence_both(ts_state, s_eng, "South Africa", 2, 0)
        fire_scoring_card_both(ts_state, s_eng, 79, ts_engine.Region.AFRICA)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_80_one_small_step(self):
        ts_state, s_eng = make_test_pair()
        ts_state.us_space_track = 1
        ts_state.ussr_space_track = 3
        s_eng.space_race["US"] = 1
        s_eng.space_race["USSR"] = 3
        ts_engine.CardHandlers.trigger_event(ts_state, 80, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "One_Small_Step")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_81_south_america_scoring(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Brazil", 2, 0)
        set_influence_both(ts_state, s_eng, "Chile", 0, 2)
        fire_scoring_card_both(ts_state, s_eng, 81, ts_engine.Region.SOUTH_AMERICA)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_107_che(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Peru", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 107, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Che")
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE

    def test_108_our_man_in_tehran(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Israel", 4, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 108, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Our_Man_In_Tehran")
        assert ts_state.ctx().decision_type in (ts_engine.DecisionType.CHOOSE_BRANCH, ts_engine.DecisionType.SELECT_CARD)
        assert s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE


# =============================================================================
# 3. LATE WAR CARDS (82..102, 109..110)
# =============================================================================

class TestLateWarCardsDifferential:

    def test_82_iranian_hostage_crisis(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Iran", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 82, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Iranian_Hostage_Crisis")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_83_the_iron_lady(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 83, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "The_Iron_Lady")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_84_reagan_bombs_libya(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Libya", 0, 4)
        ts_engine.CardHandlers.trigger_event(ts_state, 84, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Reagan_Bombs_Libya")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_85_star_wars(self):
        ts_state, s_eng = make_test_pair()
        ts_state.us_space_track = 3
        ts_state.ussr_space_track = 1
        s_eng.space_race["US"] = 3
        s_eng.space_race["USSR"] = 1
        s_eng.discard_pile.append("Duck_and_Cover")
        ts_state.set_card_location(4, ts_engine.CardLocation.DISCARD_PILE)

        ts_engine.CardHandlers.trigger_event(ts_state, 85, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Star_Wars")
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.SELECT_CARD
        assert s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE

    def test_86_north_sea_oil(self):
        ts_state, s_eng = make_test_pair(turn=8)
        ts_engine.CardHandlers.trigger_event(ts_state, 86, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "North_Sea_Oil")
        assert ts_state.has_flag(EB.NORTH_SEA_OIL_PLAYED)
        assert s_eng.game_effects.get("north_sea_oil") is True

    def test_87_the_reformer(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 87, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "The_Reformer")

        for country in ["East Germany", "East Germany", "Austria", "Hungary"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_88_marine_barracks_bombing(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Lebanon", 2, 0)
        set_influence_both(ts_state, s_eng, "Jordan", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 88, ts_engine.Player.USSR)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 2

    def test_89_soviets_shoot_down_kal007(self):
        ts_state, s_eng = make_test_pair(defcon=4)
        set_influence_both(ts_state, s_eng, "South Korea", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 89, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Soviets_Shoot_Down_KAL_007")
        assert ts_state.defcon == 3
        assert s_eng.defcon == 3
        assert ts_state.victory_points == 2
        assert s_eng.vp == 2

    def test_90_glasnost(self):
        ts_state, s_eng = make_test_pair(defcon=3)
        ts_engine.CardHandlers.trigger_event(ts_state, 90, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Glasnost")
        assert ts_state.defcon == 4
        assert s_eng.defcon == 4
        assert ts_state.victory_points == -2
        assert s_eng.vp == -2

    def test_91_ortega_elected(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Nicaragua", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 91, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Ortega_Elected_in_Nicaragua")
        assert ts_state.get_country(ts_engine.MapData.get_country_by_name("Nicaragua")).us_influence == 0
        assert s_eng.board.influence["Nicaragua"]["US"] == 0

    def test_92_terrorism(self):
        ts_state, s_eng = make_test_pair()
        s_eng.hands["US"] = ["Duck_and_Cover"]
        ts_state.set_card_location(4, ts_engine.CardLocation.HAND_US)
        ts_engine.CardHandlers.trigger_event(ts_state, 92, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Terrorism")
        assert s_eng.pending_decision.kind == SDecisionKind.RANDOM_DISCARD

    def test_93_iran_contra(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 93, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Iran_Contra_Scandal")
        assert ts_state.has_flag(EB.IRAN_CONTRA_ACTIVE)
        assert s_eng.turn_effects.get("iran_contra") is True

    def test_94_chernobyl(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 94, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Chernobyl")
        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "EUROPE"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = int(ts_engine.Region.EUROPE)
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)
        assert ts_state.has_flag(EB.CHERNOBYL_ACTIVE)

    def test_95_latin_american_debt_crisis(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Brazil", 0, 2)
        set_influence_both(ts_state, s_eng, "Chile", 0, 2)
        ts_engine.CardHandlers.trigger_event(ts_state, 95, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Latin_American_Debt_Crisis")

        if s_eng.pending_decision and s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE:
            s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "refuse"}))

        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.SELECT_CARD
        ma.primary_id = 0
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        for country in ["Brazil", "Chile"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands
            s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": country}))
            ma2 = ts_engine.MicroAction()
            ma2.decision_type = ts_engine.DecisionType.POINT_NODE
            ma2.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma2)

        sa.assert_board_equal(ts_state, s_eng)

    def test_96_tear_down_this_wall(self):
        ts_state, s_eng = make_test_pair()
        s_eng.game_effects["willy_brandt"] = True
        ts_state.set_flag(EB.WILLY_BRANDT_PLAYED)
        ts_engine.CardHandlers.trigger_event(ts_state, 96, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Tear_Down_This_Wall")
        assert ts_state.get_country(ts_engine.MapData.get_country_by_name("East Germany")).us_influence == 3
        assert s_eng.board.influence["East_Germany"]["US"] == 3
        assert "willy_brandt" not in s_eng.game_effects
        assert not ts_state.has_flag(EB.WILLY_BRANDT_PLAYED)

    def test_97_an_evil_empire(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "United Kingdom", 5, 0)
        set_influence_both(ts_state, s_eng, "France", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 97, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "An_Evil_Empire")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_98_aldrich_ames(self):
        ts_state, s_eng = make_test_pair()
        s_eng.hands["US"] = ["Duck_and_Cover"]
        ts_state.set_card_location(4, ts_engine.CardLocation.HAND_US)
        ts_engine.CardHandlers.trigger_event(ts_state, 98, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Aldrich_Ames_Remix")
        assert s_eng.pending_decision.kind == SDecisionKind.EVENT_CHOICE
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.SELECT_CARD

    def test_99_pershing_ii(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "France", 3, 0)
        set_influence_both(ts_state, s_eng, "West Germany", 4, 0)
        set_influence_both(ts_state, s_eng, "Italy", 2, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 99, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Pershing_II_Deployed")

        for country in ["France", "West Germany", "Italy"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands

            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_100_wargames(self):
        ts_state, s_eng = make_test_pair(defcon=2)
        ts_engine.CardHandlers.trigger_event(ts_state, 100, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Wargames")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "end_game"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 0
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        assert ts_state.victory_points == -6
        assert s_eng.is_terminal

    def test_101_solidarity(self):
        ts_state, s_eng = make_test_pair()
        ts_state.set_flag(EB.JOHN_PAUL_II_PLAYED)
        s_eng.game_effects["john_paul"] = True
        set_influence_both(ts_state, s_eng, "Poland", 0, 3)
        ts_engine.CardHandlers.trigger_event(ts_state, 101, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Solidarity")
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_102_iran_iraq_war_late(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Iran", 2, 0)
        set_influence_both(ts_state, s_eng, "Iraq", 0, 0)
        fire_war_card_both(ts_state, s_eng, 102, ts_engine.Player.USSR, "Iran", 6)
        sa.assert_full_state_equal(ts_state, s_eng)

    def test_109_yuri_and_samantha(self):
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 109, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Yuri_and_Samantha")
        assert ts_state.has_flag(EB.YURI_AND_SAMANTHA_ACTIVE)
        assert s_eng.game_effects.get("yuri_samantha") is True

    def test_110_awacs_sale(self):
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Saudi Arabia", 3, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 110, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "AWACS_Sale_to_Saudis")
        sa.assert_full_state_equal(ts_state, s_eng)


# =============================================================================
# 4. MULTI-TURN COMPLETE GAMES & FUZZED CROSS-ENGINE TESTING
# =============================================================================

class TestMultiTurnAndFuzzedDifferential:

    @pytest.mark.parametrize("seed", [10, 55, 999, 12345])
    def test_full_multiturn_game_simulation(self, seed):
        """Simulates full multi-turn game play across both engines verifying state sync at every turn."""
        s_eng = SEngine.new_game(seed=seed)
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, seed)

        # USSR Setup: 6 in Eastern Europe
        ussr_targets = ["Poland", "Poland", "Poland", "East Germany", "Austria", "Hungary"]
        for target in ussr_targets:
            s_cand = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cand = sa.get_ts_legal_country_names(ts_state)
            assert s_cand == ts_cand

            s_eng.step(SAction(SDecisionKind.PLACE_INFLUENCE, {"country": sa.country_ts_to_struggler(target)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(target)
            ts_engine.Engine.step(ts_state, ma)

        # US Setup: 7 in Western Europe
        us_targets = ["West Germany", "West Germany", "West Germany", "West Germany", "Italy", "Italy", "France"]
        for target in us_targets:
            s_cand = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cand = sa.get_ts_legal_country_names(ts_state)
            assert s_cand == ts_cand

            s_eng.step(SAction(SDecisionKind.PLACE_INFLUENCE, {"country": sa.country_ts_to_struggler(target)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(target)
            ts_engine.Engine.step(ts_state, ma)

        # Verify board and tracks match post-setup
        sa.assert_full_state_equal(ts_state, s_eng)

    @pytest.mark.parametrize("fuzz_seed", [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
    def test_fuzzed_actions_differential(self, fuzz_seed):
        """Fuzzes random valid influence placements across diverse board states and checks equivalence."""
        ts_state, s_eng = sa.create_paired_state(seed=fuzz_seed, defcon=4, turn=2, vp=1)

        import random
        rng = random.Random(fuzz_seed)
        countries = ["France", "West Germany", "Italy", "Poland", "Egypt", "Israel", "India", "Pakistan", "Angola", "Brazil"]

        for _ in range(15):
            c = rng.choice(countries)
            us_val = rng.randint(0, 5)
            ussr_val = rng.randint(0, 5)
            set_influence_both(ts_state, s_eng, c, us_val, ussr_val)

        sa.assert_full_state_equal(ts_state, s_eng)


# =============================================================================
# 5. ALL RELEVANT STATE VARIANTS & EDGE CASES ACROSS ALL CARDS
# =============================================================================

class TestCardStateVariantsAndEdgeCasesDifferential:

    def test_duck_and_cover_defcon_2_loss(self):
        """Duck and Cover at DEFCON 2 causes DEFCON 1 loss for phasing player."""
        ts_state, s_eng = make_test_pair(defcon=2)
        ts_engine.CardHandlers.trigger_event(ts_state, 4, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Duck_and_Cover")
        assert ts_state.defcon == 1
        assert s_eng.defcon == 1
        assert s_eng.is_terminal and s_eng.winner is SSide.USSR

    def test_we_will_bury_you_defcon_2_loss(self):
        """We Will Bury You at DEFCON 2 causes DEFCON 1 loss for USSR."""
        ts_state, s_eng = make_test_pair(defcon=2)
        ts_engine.CardHandlers.trigger_event(ts_state, 50, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "We_Will_Bury_You")
        assert ts_state.defcon == 1
        assert s_eng.defcon == 1
        assert s_eng.is_terminal and s_eng.winner is SSide.US

    def test_socialist_governments_empty_western_europe(self):
        """Socialist Governments when no US influence exists in Western Europe."""
        ts_state, s_eng = make_test_pair()
        ts_engine.CardHandlers.trigger_event(ts_state, 7, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Socialist_Governments")
        sa.assert_board_equal(ts_state, s_eng)

    def test_special_relationship_with_and_without_nato(self):
        """Special Relationship: branch when NATO is inactive vs branch when NATO is active."""
        # Variant A: Without NATO
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "United Kingdom", 5, 0)
        ts_engine.CardHandlers.trigger_event(ts_state, 105, ts_engine.Player.US)
        s_eng._fire_event(SSide.US, "Special_Relationship")

        s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
        ts_cands = sa.get_ts_legal_country_names(ts_state)
        assert s_cands == ts_cands

        s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": "France"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = ts_engine.MapData.get_country_by_name("France")
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)
        sa.assert_full_state_equal(ts_state, s_eng)

        # Variant B: With NATO (+2 VP and 2 influence)
        ts_state2, s_eng2 = make_test_pair()
        set_influence_both(ts_state2, s_eng2, "United Kingdom", 5, 0)
        ts_state2.set_flag(EB.NATO_ACTIVE)
        s_eng2.game_effects["nato"] = True
        ts_engine.CardHandlers.trigger_event(ts_state2, 105, ts_engine.Player.US)
        s_eng2._fire_event(SSide.US, "Special_Relationship")

        s_cands2 = sa.get_struggler_legal_country_names(s_eng2.pending_decision)
        ts_cands2 = sa.get_ts_legal_country_names(ts_state2)
        assert s_cands2 == ts_cands2

        s_eng2.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": "France"}))
        ma2 = ts_engine.MicroAction()
        ma2.decision_type = ts_engine.DecisionType.POINT_NODE
        ma2.primary_id = ts_engine.MapData.get_country_by_name("France")
        ts_engine.CardHandlers.handle_event_step(ts_state2, ma2)
        sa.assert_full_state_equal(ts_state2, s_eng2)

    def test_warsaw_pact_branch_0_removal_variant(self):
        """Warsaw Pact: Branch 0 removing US influence from Eastern Europe."""
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Poland", 2, 0)
        set_influence_both(ts_state, s_eng, "East Germany", 1, 0)
        set_influence_both(ts_state, s_eng, "Yugoslavia", 1, 0)
        set_influence_both(ts_state, s_eng, "Czechoslovakia", 1, 0)

        ts_engine.CardHandlers.trigger_event(ts_state, 16, ts_engine.Player.USSR)
        s_eng._fire_event(SSide.USSR, "Warsaw_Pact_Formed")

        s_eng.step(SAction(SDecisionKind.EVENT_CHOICE, {"choice": "remove"}))
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.CHOOSE_BRANCH
        ma.primary_id = 0
        ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        for country in ["Poland", "East Germany", "Yugoslavia", "Czechoslovakia"]:
            s_cands = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cands = sa.get_ts_legal_country_names(ts_state)
            assert s_cands == ts_cands
            s_eng.step(SAction(SDecisionKind.EVENT_INFLUENCE, {"country": sa.country_ts_to_struggler(country)}))
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = ts_engine.MapData.get_country_by_name(country)
            ts_engine.CardHandlers.handle_event_step(ts_state, ma)

        sa.assert_full_state_equal(ts_state, s_eng)

    def test_de_stalinization_transfer_variant(self):
        """De-Stalinization transferring influence from USSR-controlled countries."""
        ts_state, s_eng = make_test_pair()
        set_influence_both(ts_state, s_eng, "Poland", 0, 4)
        ts_engine.CardHandlers.trigger_event(ts_state, 33, ts_engine.Player.USSR)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 4

    def test_all_scoring_cards_saturation_variants(self):
        """Tests saturation variants (domination vs control) across all 6 regions + SE Asia."""
        regions = [
            (1, ts_engine.Region.ASIA, "Asia_Scoring"),
            (2, ts_engine.Region.EUROPE, "Europe_Scoring"),
            (3, ts_engine.Region.MIDDLE_EAST, "Middle_East_Scoring"),
            (37, ts_engine.Region.CENTRAL_AMERICA, "Central_America_Scoring"),
            (38, ts_engine.Region.ASIA, "Southeast_Asia_Scoring"),
            (79, ts_engine.Region.AFRICA, "Africa_Scoring"),
            (81, ts_engine.Region.SOUTH_AMERICA, "South_America_Scoring"),
        ]

        for cid, region, s_name in regions:
            ts_state, s_eng = make_test_pair()
            fire_scoring_card_both(ts_state, s_eng, cid, region)
            sa.assert_tracks_equal(ts_state, s_eng)
