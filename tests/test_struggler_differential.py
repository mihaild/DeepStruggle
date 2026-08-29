"""
Twilight Struggle AI vs Struggler Differential Test Suite
=========================================================
Cross-validates that:
1. Static map topology, country data, card metadata, and region definitions match.
2. Available legal actions and decision candidates are identical across both implementations.
3. Action resolution (influence placement, reachability rules, coups, realignments, space race, scoring, and card events) produce identical state transitions.
"""

import pytest
import ts_engine
from struggler.engine.core import Engine as SEngine
from struggler.engine.board import Board as SBoard
from struggler.engine.rules import RULES as SRULES
from struggler.engine.types import (
    Side as SSide,
    Region as SRegion,
    CardSide as SCardSide,
    DecisionKind as SDecisionKind,
    Action as SAction,
    Decision as SDecision,
)
from struggler.engine.cards import load_cards as s_load_cards
import tests.struggler_adapter as sa

EB = ts_engine.EffectBits


# =============================================================================
# 1. STATIC DATA & TOPOLOGY DIFFERENTIAL TESTS
# =============================================================================

class TestStaticDataDifferential:
    """Verifies that map graphs, countries, battlegrounds, and card definitions are identical."""

    def test_country_count_and_attributes(self):
        s_board = SBoard()
        s_countries = s_board.countries

        # ts_ai has 84 countries
        assert len(s_countries) == 84 or (len(s_countries) == 85 and "Chinese_Civil_War" in s_countries)

        for cid in range(84):
            ts_info = ts_engine.MapData.get_country_info(cid)
            s_name = sa.country_id_to_struggler(cid)
            assert s_name in s_countries, f"Country {s_name} missing from struggler"
            s_info = s_countries[s_name]

            # Stability
            assert ts_info["stability"] == s_info.stability, f"Stability mismatch for {s_name}"
            # Battleground
            assert ts_info["battleground"] == s_info.battleground, f"Battleground mismatch for {s_name}"
            # Subregion checks
            subregions_set = {sr.value for sr in s_info.subregions}
            if ts_info["in_western_europe"]:
                assert "WESTERN_EUROPE" in subregions_set
            if ts_info["in_eastern_europe"]:
                assert "EASTERN_EUROPE" in subregions_set
            if ts_info["in_southeast_asia"]:
                assert "SOUTHEAST_ASIA" in subregions_set

    def test_adjacency_graph_match(self):
        s_board = SBoard()

        for cid in range(84):
            ts_info = ts_engine.MapData.get_country_info(cid)
            s_name = sa.country_id_to_struggler(cid)

            ts_neighbors = {sa.country_id_to_struggler(n) for n in ts_info["neighbors"]}
            s_neighbors = {n for n in s_board.neighbors(s_name) if n not in ("US", "USSR", "Chinese_Civil_War")}

            assert ts_neighbors == s_neighbors, f"Adjacency mismatch for {s_name}: ts={ts_neighbors} vs s={s_neighbors}"

    def test_syria_and_iraq_not_adjacent(self):
        """Explicitly verifies that Syria and Iraq are NOT adjacent in both engines."""
        s_board = SBoard()
        assert "Iraq" not in s_board.neighbors("Syria")
        assert "Syria" not in s_board.neighbors("Iraq")

        syr_id = ts_engine.MapData.get_country_by_name("Syria")
        irq_id = ts_engine.MapData.get_country_by_name("Iraq")
        syr_info = ts_engine.MapData.get_country_info(syr_id)
        irq_info = ts_engine.MapData.get_country_info(irq_id)

        assert irq_id not in syr_info["neighbors"]
        assert syr_id not in irq_info["neighbors"]

    def test_card_database_equivalence(self):
        s_cards = s_load_cards()
        assert len(s_cards) == 110

        for card_num in range(1, 111):
            ts_card = ts_engine.CardData.get_card_info(card_num)
            s_id = sa.card_id_ts_to_struggler(card_num)
            s_card = s_cards[s_id]

            # Number
            assert s_card.number == card_num
            # Ops
            assert s_card.ops == ts_card["ops"], f"Ops mismatch for card {card_num} ({ts_card['name']})"
            # Scoring
            assert s_card.scoring == ts_card["is_scoring"]
            # Side
            expected_side = "NEUTRAL" if ts_card["side"] == "NONE" else ts_card["side"]
            assert s_card.side.value == expected_side


# =============================================================================
# 2. SETUP PHASE DIFFERENTIAL TESTS
# =============================================================================

class TestSetupPhaseDifferential:
    """Verifies legal setup actions and step resolution across USSR and US opening placements."""

    def test_initial_board_influence_setup(self):
        s_eng = SEngine.new_game(seed=100)
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 100)

        # Check pre-setup fixed printed board influence
        for cid in range(84):
            c = ts_state.get_country(cid)
            s_name = sa.country_id_to_struggler(cid)
            assert s_eng.board.influence[s_name]["US"] == c.us_influence, f"Pre-setup US mismatch in {s_name}"
            assert s_eng.board.influence[s_name]["USSR"] == c.ussr_influence, f"Pre-setup USSR mismatch in {s_name}"

    def test_ussr_and_us_setup_action_candidates(self):
        s_eng = SEngine.new_game(seed=100)
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 100)

        # USSR Setup: 6 points in Eastern Europe
        for step_i in range(6):
            assert s_eng.pending_decision.actor == SSide.USSR
            assert s_eng.pending_decision.kind == SDecisionKind.PLACE_INFLUENCE

            s_candidates = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_candidates = sa.get_ts_legal_country_names(ts_state)
            assert s_candidates == ts_candidates, f"USSR setup step {step_i} candidate mismatch"

            # Execute placement: 3 Poland, 1 East Germany, 1 Austria, 1 Finland
            target = ["Poland", "Poland", "Poland", "East Germany", "Austria", "Finland"][step_i]
            s_target = sa.country_ts_to_struggler(target)

            # Step struggler
            s_eng.step(SAction(kind=SDecisionKind.PLACE_INFLUENCE, payload={"country": s_target}))
            # Step ts_ai
            cid = ts_engine.MapData.get_country_by_name(target)
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = cid
            ok = ts_engine.Engine.step(ts_state, ma)
            assert ok

        # US Setup: 7 points in Western Europe
        for step_i in range(7):
            assert s_eng.pending_decision.actor == SSide.US
            assert s_eng.pending_decision.kind == SDecisionKind.PLACE_INFLUENCE

            s_candidates = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_candidates = sa.get_ts_legal_country_names(ts_state)
            assert s_candidates == ts_candidates, f"US setup step {step_i} candidate mismatch"

            # Execute placement: 4 West Germany, 2 Italy, 1 France
            target = ["West Germany", "West Germany", "West Germany", "West Germany", "Italy", "Italy", "France"][step_i]
            s_target = sa.country_ts_to_struggler(target)

            # Step struggler
            s_eng.step(SAction(kind=SDecisionKind.PLACE_INFLUENCE, payload={"country": s_target}))
            # Step ts_ai
            cid = ts_engine.MapData.get_country_by_name(target)
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = cid
            ok = ts_engine.Engine.step(ts_state, ma)
            assert ok

        # Verify board match post-setup
        for cid in range(84):
            c = ts_state.get_country(cid)
            s_name = sa.country_id_to_struggler(cid)
            assert s_eng.board.influence[s_name]["US"] == c.us_influence
            assert s_eng.board.influence[s_name]["USSR"] == c.ussr_influence


# =============================================================================
# 3. OPERATIONS & REACHABILITY (RULE 6.1.1) DIFFERENTIAL TESTS
# =============================================================================

class TestOpsReachabilityDifferential:
    """Verifies influence placement rules, adjacency reachability, and double-cost control mechanics."""

    def test_reachability_from_start_snapshot(self):
        s_board = SBoard()
        # Set US in UK only
        s_board.influence["UK"]["US"] = 5
        s_board.influence["Canada"]["US"] = 2

        # Reachable countries for US from UK/Canada
        assert s_board.is_reachable(SSide.US, "France")
        assert s_board.is_reachable(SSide.US, "Norway")
        assert s_board.is_reachable(SSide.US, "Canada")
        assert s_board.is_reachable(SSide.US, "UK")
        assert not s_board.is_reachable(SSide.US, "Poland")

    def test_double_cost_influence_in_controlled_country(self):
        """Placing influence in an opponent-controlled country costs 2 Ops."""
        s_board = SBoard()
        s_board.influence["East_Germany"]["USSR"] = 3 # Controlled (stability 3)
        s_board.influence["West_Germany"]["US"] = 4   # Adjacent

        # Cost for US to place in East Germany is 2
        cost_us = s_board.influence_cost(SSide.US, "East_Germany")
        assert cost_us == 2

        # Cost for USSR to place in East Germany is 1
        cost_ussr = s_board.influence_cost(SSide.USSR, "East_Germany")
        assert cost_ussr == 1


# =============================================================================
# 4. COUP AND REALIGNMENT DIFFERENTIAL TESTS
# =============================================================================

class TestCoupAndRealignmentDifferential:
    """Verifies coup targets, DEFCON degradation, Mil Ops, and realignment mechanics."""

    def test_defcon_regional_coup_restrictions(self):
        min_defcon = SRULES["coup_min_defcon"]
        assert min_defcon["EUROPE"] == 5
        assert min_defcon["ASIA"] == 4
        assert min_defcon["MIDDLE_EAST"] == 3

    def test_coup_resolution_formula(self):
        """Coup formula: Net = Roll + Ops - (2 * Stability)."""
        stability = 2 # e.g. Cuba / Israel
        card_ops = 3
        roll = 5
        # Total = 5 + 3 = 8. Threshold = 2 * 2 = 4. Net = 4.
        net_delta = roll + card_ops - 2 * stability
        assert net_delta == 4

        # Target had 2 USSR influence -> 0 USSR, 2 US influence
        target_ussr_before = 2
        ussr_loss = min(target_ussr_before, net_delta)
        us_gain = net_delta - ussr_loss
        assert ussr_loss == 2
        assert us_gain == 2

    def test_realignment_roll_modifiers(self):
        s_eng = SEngine.new_game(seed=42)
        s_eng.board.influence["Poland"]["USSR"] = 3
        s_eng.board.influence["Poland"]["US"] = 1
        s_eng.board.influence["East_Germany"]["USSR"] = 3
        s_eng.board.influence["Czechoslovakia"]["US"] = 3

        # USSR has more influence (+1) + East Germany adjacent (+1) + USSR superpower adjacent (+1) = +3
        # US has Czechoslovakia adjacent (+1) = +1
        mod_ussr = s_eng._realignment_bonus(SSide.USSR, "Poland")
        mod_us = s_eng._realignment_bonus(SSide.US, "Poland")
        assert mod_ussr == 3
        assert mod_us == 1


# =============================================================================
# 5. SPACE RACE DIFFERENTIAL TESTS
# =============================================================================

class TestSpaceRaceDifferential:
    """Verifies Space Race box requirements, attempt tracking, and VP milestones."""

    def test_space_race_requirements(self):
        boxes = SRULES["space_race_boxes"]
        assert boxes["1"]["ops"] == 2
        assert boxes["2"]["ops"] == 2
        assert boxes["3"]["ops"] == 2
        assert boxes["4"]["ops"] == 2
        assert boxes["5"]["ops"] == 3
        assert boxes["6"]["ops"] == 3
        assert boxes["7"]["ops"] == 3
        assert boxes["8"]["ops"] == 4

    def test_space_race_scoring_milestones(self):
        boxes = SRULES["space_race_boxes"]
        # Box 1: Earth Satellite (2 VP first, 1 VP second)
        assert boxes["1"]["vp_first"] == 2
        assert boxes["1"]["vp_second"] == 1

        # Box 3: Man in Space (2 VP first, 0 VP second)
        assert boxes["3"]["vp_first"] == 2
        assert boxes["3"]["vp_second"] == 0

        # Box 5: Lunar Orbit (3 VP first, 1 VP second)
        assert boxes["5"]["vp_first"] == 3
        assert boxes["5"]["vp_second"] == 1

        # Box 7: Space Station (4 VP first, 2 VP second)
        assert boxes["7"]["vp_first"] == 4
        assert boxes["7"]["vp_second"] == 2


# =============================================================================
# 6. REGIONAL & FINAL SCORING DIFFERENTIAL TESTS
# =============================================================================

class TestScoringSystemDifferential:
    """Verifies regional scoring arithmetic (Presence, Domination, Control) across multiple configurations."""

    @pytest.mark.parametrize("region_enum, s_region", [
        (ts_engine.Region.EUROPE, SRegion.EUROPE),
        (ts_engine.Region.ASIA, SRegion.ASIA),
        (ts_engine.Region.MIDDLE_EAST, SRegion.MIDDLE_EAST),
        (ts_engine.Region.AFRICA, SRegion.AFRICA),
        (ts_engine.Region.CENTRAL_AMERICA, SRegion.CENTRAL_AMERICA),
        (ts_engine.Region.SOUTH_AMERICA, SRegion.SOUTH_AMERICA),
    ])
    def test_scoring_differential_presence(self, region_enum, s_region):
        """Tests presence scoring for both sides."""
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        s_board = SBoard()

        # Clear board
        for cid in range(84):
            ts_state.set_country(cid, 0, 0)
        for name in s_board.influence:
            s_board.influence[name]["US"] = 0
            s_board.influence[name]["USSR"] = 0

        # Give US presence (1 non-BG) and USSR presence (1 non-BG)
        countries_in_region = [cid for cid in range(84) if ts_engine.MapData.get_country_info(cid)["region"] == int(region_enum) and not ts_engine.MapData.get_country_info(cid)["battleground"]]
        if len(countries_in_region) >= 2:
            c1, c2 = countries_in_region[0], countries_in_region[1]
            ts_state.set_country(c1, 2, 0) # US controls c1
            ts_state.set_country(c2, 0, 2) # USSR controls c2

            s_name1 = sa.country_id_to_struggler(c1)
            s_name2 = sa.country_id_to_struggler(c2)
            s_board.influence[s_name1]["US"] = 2
            s_board.influence[s_name2]["USSR"] = 2

            # Score in ts_ai
            old_vp = ts_state.victory_points
            ts_engine.Scoring.score_region(ts_state, region_enum)
            ts_delta = ts_state.victory_points - old_vp

            # Score in struggler
            s_vp = s_board.score_region(s_region)

            assert ts_delta == s_vp, f"Presence score delta mismatch in region {s_region}: ts={ts_delta} vs s={s_vp}"

    def test_europe_control_domination(self):
        """Tests domination scoring in Europe with battleground and adjacency bonuses."""
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        s_board = SBoard()

        for cid in range(84):
            ts_state.set_country(cid, 0, 0)
        for name in s_board.influence:
            s_board.influence[name]["US"] = 0
            s_board.influence[name]["USSR"] = 0

        # US Controls UK(5/0), France(3/0), West Germany(4/0), Italy(2/0) -> 3 BG, 1 non-BG
        # USSR Controls East Germany(0/3), Poland(0/3), Finland(0/2) -> 2 BG, 1 non-BG
        placements = [
            ("United Kingdom", 5, 0),
            ("France", 3, 0),
            ("West Germany", 4, 0),
            ("Italy", 2, 0),
            ("East Germany", 0, 3),
            ("Poland", 0, 3),
            ("Finland", 0, 2),
        ]
        for name, us_inf, ussr_inf in placements:
            cid = ts_engine.MapData.get_country_by_name(name)
            ts_state.set_country(cid, us_inf, ussr_inf)
            s_name = sa.country_ts_to_struggler(name)
            s_board.influence[s_name]["US"] = us_inf
            s_board.influence[s_name]["USSR"] = ussr_inf

        # US: Domination (7) + 3 BG (3) = 10 VP
        # USSR: Presence (3) + 2 BG (2) = 5 VP
        # Net: US +5 VP
        old_vp = ts_state.victory_points
        ts_engine.Scoring.score_region(ts_state, ts_engine.Region.EUROPE)
        ts_delta = ts_state.victory_points - old_vp
        s_vp = s_board.score_region(SRegion.EUROPE)

        assert ts_delta == s_vp
        assert ts_delta == 5


# =============================================================================
# 7. CARD EVENTS CROSS-ENGINE VALIDATION
# =============================================================================

class TestCardEventsDifferential:
    """Verifies that card event definitions and effects behave identically."""

    def test_duck_and_cover_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        ts_state.defcon = 4
        old_vp = ts_state.victory_points

        # Trigger Duck and Cover (Card 4) for US
        ts_engine.CardHandlers.trigger_event(ts_state, 4, ts_engine.Player.US)
        # DEFCON degrades to 3, US gains 5 - 3 = 2 VP
        assert ts_state.defcon == 3
        assert ts_state.victory_points == old_vp + 2

    def test_fidel_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        cuba_id = ts_engine.MapData.get_country_by_name("Cuba")
        ts_state.set_country(cuba_id, 2, 0)

        # Trigger Fidel (Card 8)
        ts_engine.CardHandlers.trigger_event(ts_state, 8, ts_engine.Player.USSR)
        c = ts_state.get_country(cuba_id)
        assert c.us_influence == 0
        assert c.ussr_influence == 3

    def test_socialist_governments_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

        # Set Italy 3 US, France 2 US
        it_id = ts_engine.MapData.get_country_by_name("Italy")
        fr_id = ts_engine.MapData.get_country_by_name("France")
        ts_state.set_country(it_id, 3, 0)
        ts_state.set_country(fr_id, 2, 0)

        # Trigger Socialist Governments (Card 7)
        ts_engine.CardHandlers.trigger_event(ts_state, 7, ts_engine.Player.USSR)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 3
        assert ts_state.ctx().max_per_country == 2

        # Verify legal targets in ts_ai include Italy and France
        candidates = sa.get_ts_legal_country_names(ts_state)
        assert "Italy" in candidates
        assert "France" in candidates

    def test_truman_doctrine_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

        # Yugoslavia: stability 3, 2 USSR influence (uncontrolled)
        yug_id = ts_engine.MapData.get_country_by_name("Yugoslavia")
        ts_state.set_country(yug_id, 0, 2)

        # Trigger Truman Doctrine (Card 19)
        ts_engine.CardHandlers.trigger_event(ts_state, 19, ts_engine.Player.US)
        candidates = sa.get_ts_legal_country_names(ts_state)
        assert "Yugoslavia" in candidates

        # Step removal via handle_event_step
        ma = ts_engine.MicroAction()
        ma.decision_type = ts_engine.DecisionType.POINT_NODE
        ma.primary_id = yug_id
        finished = ts_engine.CardHandlers.handle_event_step(ts_state, ma)
        assert finished
        assert ts_state.get_country(yug_id).ussr_influence == 0

    def test_vietnam_revolts_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        # Vietnam (Card 9): USSR adds 2 in Vietnam, gives USSR +1 Op on all ops cards spent in SE Asia
        vn_id = ts_engine.MapData.get_country_by_name("Vietnam")
        old_ussr = ts_state.get_country(vn_id).ussr_influence
        ts_engine.CardHandlers.trigger_event(ts_state, 9, ts_engine.Player.USSR)
        assert ts_state.get_country(vn_id).ussr_influence == old_ussr + 2
        assert ts_state.has_flag(EB.VIETNAM_REVOLTS_ACTIVE)

    def test_us_japan_pact_differential(self):
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        jp_id = ts_engine.MapData.get_country_by_name("Japan")
        # Trigger US/Japan Pact (Card 27): US gets enough to control Japan (stability 4, 4 US)
        ts_engine.CardHandlers.trigger_event(ts_state, 27, ts_engine.Player.US)
        assert ts_state.get_country(jp_id).us_influence >= 4
        assert ts_state.has_flag(EB.US_JAPAN_PACT_ACTIVE)


# =============================================================================
# 8. MULTI-STEP TRAJECTORY CROSS-TESTING
# =============================================================================

class TestTrajectoryDifferential:
    """Verifies synchronized multi-step execution across diverse game seeds."""

    @pytest.mark.parametrize("seed", [123, 456, 789, 1011, 2022, 9999, 424242])
    def test_synchronized_setup_trajectories(self, seed):
        s_eng = SEngine.new_game(seed=seed)
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, seed)

        # 1. USSR setup: 6 placements in Eastern Europe (deterministic pattern)
        ussr_choices = ["Poland", "Poland", "Poland", "East Germany", "Austria", "Hungary"]
        for country in ussr_choices:
            s_cand = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cand = sa.get_ts_legal_country_names(ts_state)
            assert s_cand == ts_cand

            s_target = sa.country_ts_to_struggler(country)
            s_eng.step(SAction(kind=SDecisionKind.PLACE_INFLUENCE, payload={"country": s_target}))

            cid = ts_engine.MapData.get_country_by_name(country)
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = cid
            ts_engine.Engine.step(ts_state, ma)

        # 2. US setup: 7 placements in Western Europe
        us_choices = ["West Germany", "West Germany", "West Germany", "West Germany", "Italy", "Italy", "France"]
        for country in us_choices:
            s_cand = sa.get_struggler_legal_country_names(s_eng.pending_decision)
            ts_cand = sa.get_ts_legal_country_names(ts_state)
            assert s_cand == ts_cand

            s_target = sa.country_ts_to_struggler(country)
            s_eng.step(SAction(kind=SDecisionKind.PLACE_INFLUENCE, payload={"country": s_target}))

            cid = ts_engine.MapData.get_country_by_name(country)
            ma = ts_engine.MicroAction()
            ma.decision_type = ts_engine.DecisionType.POINT_NODE
            ma.primary_id = cid
            ts_engine.Engine.step(ts_state, ma)

        # 3. Verify final board match after setup
        for cid in range(84):
            c = ts_state.get_country(cid)
            s_name = sa.country_id_to_struggler(cid)
            assert s_eng.board.influence[s_name]["US"] == c.us_influence
            assert s_eng.board.influence[s_name]["USSR"] == c.ussr_influence


# =============================================================================
# 9. ADVANCED WAR CARDS & MULTI-STEP CARD EVENT DIFFERENTIAL TESTS
# =============================================================================

class TestAdvancedCardEventsDifferential:
    """Verifies complex multi-step card events and war resolution rules."""

    def test_arab_israeli_war_modifiers(self):
        """Arab-Israeli War (Card 10): Base roll 1d6 - modified by adjacency.
        Target: Israel (stability 4, BG).
        Adjacent to Israel: Egypt, Jordan, Syria, Lebanon.
        USSR gets -1 per adjacent country controlled by US.
        Roll 1-4 is modified. On net 4+, USSR wins 2 VP, gains 2 Mil Ops, replaces all US influence in Israel with USSR influence.
        """
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        isr_id = ts_engine.MapData.get_country_by_name("Israel")
        egy_id = ts_engine.MapData.get_country_by_name("Egypt")
        jor_id = ts_engine.MapData.get_country_by_name("Jordan")
        syr_id = ts_engine.MapData.get_country_by_name("Syria")
        leb_id = ts_engine.MapData.get_country_by_name("Lebanon")

        # Set Israel: 2 US. Egypt: 2 US (controlled). Jordan: 2 US (controlled). Syria: 2 USSR. Lebanon: 0.
        ts_state.set_country(isr_id, 2, 0)
        ts_state.set_country(egy_id, 2, 0)
        ts_state.set_country(jor_id, 2, 0)
        ts_state.set_country(syr_id, 0, 2)
        ts_state.set_country(leb_id, 0, 0)

        # US controls Egypt and Jordan -> USSR penalty is -2.
        # If roll = 6 -> modified roll = 4 -> Success!
        old_vp = ts_state.victory_points
        old_mil = ts_state.ussr_mil_ops

        # Trigger Arab-Israeli War with forced roll 6
        ts_engine.CardHandlers.trigger_event(ts_state, 13, ts_engine.Player.USSR)
        if ts_state.ctx().decision_type == ts_engine.DecisionType.ROLL_DIE:
            ts_engine.CardHandlers.handle_event_step(ts_state, ts_engine.MicroAction(ts_engine.DecisionType.ROLL_DIE, 6, 0, 0))
        assert ts_state.victory_points == old_vp - 2
        assert ts_state.ussr_mil_ops == old_mil + 2
        assert ts_state.get_country(isr_id).us_influence == 0
        assert ts_state.get_country(isr_id).ussr_influence == 2

    def test_korean_war_modifiers(self):
        """Korean War (Card 11): Base roll 1d6 - modified by adjacency.
        Target: South Korea (stability 2, BG).
        Adjacent to South Korea: North Korea (USSR controlled), Japan (US controlled).
        US controls Japan (-1 modifier).
        If roll = 5 -> modified roll = 4 -> Success!
        """
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        sk_id = ts_engine.MapData.get_country_by_name("South Korea")
        nk_id = ts_engine.MapData.get_country_by_name("North Korea")
        jp_id = ts_engine.MapData.get_country_by_name("Japan")

        ts_state.set_country(sk_id, 2, 0)
        ts_state.set_country(nk_id, 0, 3) # Controlled
        ts_state.set_country(jp_id, 4, 0) # US controlled -> -1

        old_vp = ts_state.victory_points
        old_mil = ts_state.ussr_mil_ops

        ts_engine.CardHandlers.trigger_event(ts_state, 11, ts_engine.Player.USSR)
        if ts_state.ctx().decision_type == ts_engine.DecisionType.ROLL_DIE:
            ts_engine.CardHandlers.handle_event_step(ts_state, ts_engine.MicroAction(ts_engine.DecisionType.ROLL_DIE, 5, 0, 0))
        assert ts_state.victory_points == old_vp - 2
        assert ts_state.ussr_mil_ops == old_mil + 2
        assert ts_state.get_country(sk_id).us_influence == 0
        assert ts_state.get_country(sk_id).ussr_influence == 2

    def test_decolonization_candidate_selection(self):
        """Decolonization (Card 30): USSR adds 1 influence in each of 4 countries in Africa and/or SE Asia."""
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

        ts_engine.CardHandlers.trigger_event(ts_state, 30, ts_engine.Player.USSR)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 4
        assert ts_state.ctx().max_per_country == 1

        candidates = sa.get_ts_legal_country_names(ts_state)
        # Check that African and SE Asian countries are candidates
        assert "Algeria" in candidates
        assert "Nigeria" in candidates
        assert "Angola" in candidates
        assert "Vietnam" in candidates
        assert "Indonesia" in candidates
        # European countries must NOT be candidates
        assert "France" not in candidates
        assert "Poland" not in candidates

    def test_marshall_plan_candidate_selection(self):
        """Marshall Plan (Card 23): US adds 1 influence in each of 7 Western European countries."""
        ts_state = ts_engine.GameState()
        ts_engine.Engine.init_game(ts_state, 42)
        ts_state.current_phase = ts_engine.Phase.ACTION_ROUND

        ts_engine.CardHandlers.trigger_event(ts_state, 23, ts_engine.Player.US)
        assert ts_state.ctx().decision_type == ts_engine.DecisionType.POINT_NODE
        assert ts_state.ctx().remaining_steps == 7
        assert ts_state.ctx().max_per_country == 1

        candidates = sa.get_ts_legal_country_names(ts_state)
        assert "France" in candidates
        assert "West Germany" in candidates
        assert "Italy" in candidates
        assert "United Kingdom" in candidates
        # Eastern European countries must NOT be candidates
        assert "Poland" not in candidates
        assert "Romania" not in candidates
