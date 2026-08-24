import pytest
import ts_engine as ts
EB = ts.EffectBits

def cid(name):
    return ts.MapData.get_country_by_name(name)

def make_state(seed=42):
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    return s

def set_inf(s, name_or_id, us=0, ussr=0):
    c_id = cid(name_or_id) if isinstance(name_or_id, str) else name_or_id
    s.set_country(c_id, us, ussr)

def get_inf(s, name_or_id):
    c_id = cid(name_or_id) if isinstance(name_or_id, str) else name_or_id
    c = s.get_country(c_id)
    return c.us_influence, c.ussr_influence

def test_fix_card_20_olympic_games_boycott_no_vp():
    s = make_state()
    s.victory_points = 0
    s.defcon = 4
    # US triggers Olympic Games
    done = ts.CardHandlers.trigger_event(s, 20, ts.Player.US)
    assert not done
    assert s.ctx().decision_player == ts.Player.USSR
    assert s.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH
    # USSR boycotts (branch 1)
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 1))
    assert not done
    # Sponsor (US) gets 4 Ops, DEFCON degrades by 1, NO VP AWARDED
    assert s.victory_points == 0
    assert s.defcon == 3
    assert s.ctx().pending_ops_value == 4
    assert s.ctx().decision_player == ts.Player.US

def test_fix_card_32_un_intervention_cannot_be_headlined():
    s = make_state()
    s.current_phase = ts.Phase.HEADLINE
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD
    s.ctx().decision_player = ts.Player.US
    s.set_card_location(32, ts.CardLocation.HAND_US) # UN Intervention
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[32] == 0 # Cannot headline UN Intervention

def test_fix_card_33_de_stalinization_stage2_cannot_stop_early():
    s = make_state()
    poland_id = cid("Poland")
    set_inf(s, poland_id, 0, 3)
    done = ts.CardHandlers.trigger_event(s, 33, ts.Player.USSR)
    assert not done
    # Remove 2 USSR from Poland in Stage 1
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, poland_id))
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, poland_id))
    # Confirm removal stage done -> enter Stage 2 (placement)
    action = ts.MicroAction(ts.DecisionType.POINT_NODE, 255)
    action.flags = 0x80 # CONFIRM_DONE
    ts.CardHandlers.handle_event_step(s, action)
    assert s.ctx().remaining_steps == 2
    assert s.ctx().allow_early_stop == 0
    # Attempting confirm done early in Stage 2 must be rejected (returns false, remaining_steps unchanged)
    done = ts.CardHandlers.handle_event_step(s, action)
    assert not done
    assert s.ctx().remaining_steps == 2

def test_fix_card_36_brush_war_nato_cancellation_exceptions():
    s = make_state()
    s.set_flag(EB.NATO_ACTIVE)
    spain_id = cid("Spain/Portugal") # Western Europe, stability 2
    mexico_id = cid("Mexico") # CA, stability 2
    set_inf(s, spain_id, 2, 0)
    set_inf(s, mexico_id, 2, 0)
    
    # USSR triggers Brush War
    ts.CardHandlers.trigger_event(s, 36, ts.Player.USSR)
    mask = ts.Engine.get_legal_action_mask(s)
    # Spain is protected by NATO -> illegal
    assert mask[spain_id] == 0
    # Mexico is non-European, stability 2 -> legal
    assert mask[mexico_id] == 1

def test_fix_card_39_arms_race_only_phasing_player():
    s = make_state()
    s.defcon = 4
    s.us_mil_ops = 4
    s.ussr_mil_ops = 1
    s.victory_points = 0
    # USSR plays Arms Race (USSR is phasing player, but USSR has LESS mil ops than US)
    done = ts.CardHandlers.trigger_event(s, 39, ts.Player.USSR)
    assert done == True
    # No VP awarded because triggering/phasing player USSR is not ahead!
    assert s.victory_points == 0

def test_fix_card_40_cuban_missile_crisis_cancellation():
    s = make_state()
    s.set_flag(EB.CMC_ACTIVE_US)
    cuba_id = cid("Cuba")
    colombia_id = cid("Colombia")
    set_inf(s, cuba_id, 0, 3)
    set_inf(s, colombia_id, 2, 0)
    assert s.has_flag(EB.CMC_ACTIVE_US)
    
    # USSR plays card 14 (COMECON, 3 Ops, USSR card) for Coup while CMC is active
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.USSR
    s.ctx().decision_player = ts.Player.USSR
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD
    s.set_card_location(14, ts.CardLocation.HAND_USSR)
    s.defcon = 3
    
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 14))
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1)) # OPS
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1)) # COUP
    # Coup Colombia with roll 3 -> removes 2 from Cuba, clears CMC, executes coup
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, colombia_id, 3))
    
    # CMC flag must now be cancelled and 2 influence removed from Cuba!
    assert not s.has_flag(EB.CMC_ACTIVE_US)
    assert get_inf(s, cuba_id)[1] == 1

def test_fix_card_47_junta_free_ops_restricted_to_ca_sa():
    s = make_state()
    nicaragua_id = cid("Nicaragua") # CA
    set_inf(s, nicaragua_id, 2, 0)
    ts.CardHandlers.trigger_event(s, 47, ts.Player.USSR)
    # Put 2 influence in Nicaragua
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, nicaragua_id))
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().pending_op_card == 47
    # In SELECT_OP_MODE, INFLUENCE mode is illegal for Junta
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[0] == 0 # INFLUENCE illegal
    assert mask[1] == 1 # COUP legal

def test_fix_card_50_we_will_bury_you_pending_ar_check():
    s = make_state()
    s.set_flag(EB.WE_WILL_BURY_YOU_PENDING)
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = 4 # Duck and Cover (not UN Intervention)
    s.victory_points = 0
    # US plays card for OPS without playing UN Intervention as Event
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1)) # OPS
    # USSR is awarded +3 VP (-3 on VP track) and pending flag is cleared
    assert s.victory_points == -3
    assert not s.has_flag(EB.WE_WILL_BURY_YOU_PENDING)

def test_fix_card_53_south_african_unrest_branch1_adjacent():
    s = make_state()
    sa_id = cid("South Africa")
    angola_id = cid("Angola")
    set_inf(s, sa_id, 0, 0)
    set_inf(s, angola_id, 0, 0)
    ts.CardHandlers.trigger_event(s, 53, ts.Player.USSR)
    # Branch 1: 1 in SA, 2 in adjacent
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 1))
    assert not done
    assert get_inf(s, sa_id)[1] == 1
    assert s.ctx().decision_type == ts.DecisionType.POINT_NODE
    # Place 2 in Angola
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, angola_id))
    assert done == True
    assert get_inf(s, angola_id)[1] == 2

def test_fix_card_59_flower_power_war_cards_for_ops():
    s = make_state()
    s.set_flag(EB.FLOWER_POWER_ACTIVE)
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = 97 # An Evil Empire (War card)
    s.victory_points = 0
    # US plays An Evil Empire for Ops
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1)) # OPS
    # USSR is awarded +2 VP (-2 on VP track)
    assert s.victory_points == -2

def test_fix_card_60_u2_incident_awards_vp_on_un_intervention():
    s = make_state()
    s.set_flag(EB.U2_INCIDENT_ACTIVE)
    s.set_card_location(11, ts.CardLocation.HAND_US) # Korean War (USSR card)
    s.victory_points = 0
    # US plays UN Intervention
    ts.CardHandlers.trigger_event(s, 32, ts.Player.US)
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 11))
    assert not done
    # USSR is awarded +1 VP (-1 on VP track)
    assert s.victory_points == -1

def test_fix_card_73_shuttle_diplomacy_ongoing_until_scoring():
    s = make_state()
    s.set_card_location(73, ts.CardLocation.HAND_US)
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = 73
    # US plays Shuttle Diplomacy for Event
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0)) # EVENT
    assert s.has_flag(EB.SHUTTLE_DIPLOMACY_ACTIVE)
    assert s.get_card_location(73) == ts.CardLocation.ONGOING_EVENT

def test_fix_card_89_soviets_shoot_down_kal_007_no_coups():
    s = make_state()
    sk_id = cid("South Korea")
    set_inf(s, sk_id, 3, 0) # South Korea controlled
    ts.CardHandlers.trigger_event(s, 89, ts.Player.US)
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().pending_op_card == 89
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[0] == 1 # INFLUENCE legal
    assert mask[1] == 0 # COUP forbidden

def test_fix_card_90_glasnost_reformer_no_coups():
    s = make_state()
    s.set_flag(EB.THE_REFORMER_PLAYED)
    ts.CardHandlers.trigger_event(s, 90, ts.Player.USSR)
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().pending_op_card == 90
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[0] == 1 # INFLUENCE legal
    assert mask[1] == 0 # COUP forbidden

def test_fix_card_94_chernobyl_all_regions_persistent():
    s = make_state()
    ts.CardHandlers.trigger_event(s, 94, ts.Player.US)
    assert s.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH
    mask = ts.Engine.get_legal_action_mask(s)
    for r in range(6): assert mask[r] == 1 # All 6 regions legal
    # Choose South America (region 5)
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 5))
    assert done == True
    assert s.has_flag(EB.CHERNOBYL_ACTIVE)
    ch_reg = (s.persistent_effects & EB.CHERNOBYL_REGION_MASK) >> EB.CHERNOBYL_REGION_SHIFT
    assert ch_reg == 5

def test_fix_card_96_tear_down_this_wall_europe_only():
    s = make_state()
    ts.CardHandlers.trigger_event(s, 96, ts.Player.US)
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().pending_op_card == 96
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[0] == 0 # INFLUENCE forbidden

def test_fix_card_105_special_relationship_nato_branch():
    s = make_state()
    s.set_flag(EB.NATO_ACTIVE)
    uk_id = cid("United Kingdom")
    wg_id = cid("West Germany")
    set_inf(s, uk_id, 5, 0)
    set_inf(s, wg_id, 0, 0)
    s.victory_points = 0
    ts.CardHandlers.trigger_event(s, 105, ts.Player.US)
    assert s.victory_points == 2 # +2 VP
    assert s.ctx().remaining_steps == 1
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, wg_id))
    assert done == True
    assert get_inf(s, wg_id)[0] == 2 # 2 influence added to WG

def test_fix_card_106_norad_influence_placement():
    s = make_state()
    canada_id = cid("Canada")
    set_inf(s, canada_id, 3, 0)
    # Trigger NORAD event -> sets NORAD flag
    ts.CardHandlers.trigger_event(s, 106, ts.Player.US)
    assert s.has_flag(EB.NORAD_ACTIVE)
    # When NORAD reaction is executed, US adds 1 influence to country with US influence
    s.ctx().resolving_card = 106
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.POINT_NODE
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[canada_id] == 1
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, canada_id))
    assert done == True
    assert get_inf(s, canada_id)[0] == 4

def test_fix_card_107_che_ca_sa_africa_conditional_second_coup():
    s = make_state()
    colombia_id = cid("Colombia") # SA non-battleground
    peru_id = cid("Peru") # SA non-battleground
    set_inf(s, colombia_id, 2, 0)
    set_inf(s, peru_id, 2, 0)
    ts.CardHandlers.trigger_event(s, 107, ts.Player.USSR)
    # First coup in Colombia with forced roll 6 -> removes 2 US, adds 5 USSR
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, colombia_id, 6))
    assert not done # Second coup offered because US influence removed!
    # Second coup in Peru with forced roll 6 -> removes 2 US, adds 3 USSR
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, peru_id, 6))
    assert done == True
    assert get_inf(s, colombia_id)[0] == 0
    assert get_inf(s, peru_id)[0] == 0

def test_fix_card_108_our_man_in_tehran_peek_discard_and_return():
    s = make_state()
    iran_id = cid("Iran")
    set_inf(s, iran_id, 4, 0) # US controls ME country
    for i in range(1, 111): s.set_card_location(i, ts.CardLocation.DISCARD_PILE)
    # Put 5 known cards into draw deck
    s.set_card_location(10, ts.CardLocation.DRAW_DECK)
    s.set_card_location(11, ts.CardLocation.DRAW_DECK)
    s.set_card_location(12, ts.CardLocation.DRAW_DECK)
    s.set_card_location(13, ts.CardLocation.DRAW_DECK)
    s.set_card_location(14, ts.CardLocation.DRAW_DECK)
    
    ts.CardHandlers.trigger_event(s, 108, ts.Player.US)
    assert s.ctx().decision_type == ts.DecisionType.SELECT_CARD
    assert s.get_card_location(10) == ts.CardLocation.PEEKED_TEMP
    
    # Discard card 10
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 10))
    assert not done
    assert s.get_card_location(10) == ts.CardLocation.DISCARD_PILE
    
    # Confirm done -> remaining cards return to DRAW_DECK
    action = ts.MicroAction(ts.DecisionType.SELECT_CARD, 0)
    action.flags = 0x80 # CONFIRM_DONE
    done = ts.CardHandlers.handle_event_step(s, action)
    assert done == True
    assert s.get_card_location(11) == ts.CardLocation.DRAW_DECK
    assert s.get_card_location(12) == ts.CardLocation.DRAW_DECK

# =============================================================================
# Verification of 13 Specific Bug Fixes
# =============================================================================

def test_fix_china_card_asia_bonus_in_action_round():
    s = make_state()
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.china_card_holder = ts.Player.US
    s.china_card_playable = 1
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # US plays China Card for Ops
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 6)) # China Card
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1)) # Ops
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0)) # Influence

    # In Asia, pending ops should be 5
    assert s.ctx().pending_ops_value == 5
    assert s.ctx().remaining_steps == 5

    japan_id = cid("Japan")
    for _ in range(5):
        ts.Engine.step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, japan_id))
    
    # Finished all 5 points in Asia -> passes China Card to USSR
    assert s.china_card_holder == ts.Player.USSR
    assert s.china_card_playable == 0

def test_fix_nato_brush_war_europe_control_check():
    s = make_state()
    s.set_flag(EB.NATO_ACTIVE)
    greece_id = cid("Greece") # Western Europe, stability 2, uncontrolled
    spain_id = cid("Spain/Portugal") # Western Europe, stability 2, US controlled
    
    set_inf(s, greece_id, 0, 0) # Uncontrolled
    set_inf(s, spain_id, 2, 0)  # US controlled
    
    ts.CardHandlers.trigger_event(s, 36, ts.Player.USSR)
    mask = ts.Engine.get_legal_action_mask(s)
    
    # Spain is US-controlled in Europe -> blocked by NATO
    assert mask[spain_id] == 0
    # Greece is uncontrolled Western Europe, stability 2 -> legal target (fixed from bug which blocked all Western Europe)
    assert mask[greece_id] == 1

def test_fix_independent_reds_sets_equal_influence():
    s = make_state()
    yugo_id = cid("Yugoslavia")
    set_inf(s, yugo_id, 2, 3) # US 2, USSR 3
    
    ts.CardHandlers.trigger_event(s, 22, ts.Player.US)
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, yugo_id))
    
    # US influence should equal USSR influence (3, not 5!)
    assert get_inf(s, yugo_id)[0] == 3
    assert get_inf(s, yugo_id)[1] == 3

def test_fix_we_will_bury_you_vp_penalty_on_scoring_card():
    s = make_state()
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.victory_points = 0
    s.set_flag(EB.WE_WILL_BURY_YOU_PENDING)
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # US plays Asia Scoring
    s.set_card_location(1, ts.CardLocation.HAND_US)
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 1))

    # USSR should have gained 3 VP (-3) + 1 VP from default Asia Scoring (-1) = -4 VP
    assert not s.has_flag(EB.WE_WILL_BURY_YOU_PENDING)
    assert s.victory_points == -4

def test_fix_ask_not_discard_six_cards_safely():
    s = make_state()
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD
    # Give US 7 cards
    for c in range(10, 17):
        s.set_card_location(c, ts.CardLocation.HAND_US)
    
    ts.CardHandlers.trigger_event(s, 77, ts.Player.US)
    
    # Discard 6 cards
    for c in range(10, 16):
        done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, c))
        assert not done
    
    # Finish discard
    action = ts.MicroAction(ts.DecisionType.SELECT_CARD, 0)
    action.flags = 0x80
    done = ts.CardHandlers.handle_event_step(s, action)
    assert done == True

def test_fix_star_wars_replays_shuttle_diplomacy_as_ongoing():
    s = make_state()
    s.us_space_track = 5
    s.ussr_space_track = 2
    s.set_card_location(73, ts.CardLocation.DISCARD_PILE) # Shuttle Diplomacy
    
    ts.CardHandlers.trigger_event(s, 85, ts.Player.US)
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 73))
    
    assert s.has_flag(EB.SHUTTLE_DIPLOMACY_ACTIVE)
    assert s.get_card_location(73) == ts.CardLocation.ONGOING_EVENT

def test_fix_kal007_glasnost_tear_down_ops_reachable():
    s = make_state()
    sk_id = cid("South Korea")
    set_inf(s, sk_id, 3, 0) # US controls SK
    
    # KAL-007
    done = ts.CardHandlers.trigger_event(s, 89, ts.Player.US)
    assert not done
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().resolving_card == 0
    assert s.ctx().pending_ops_value == 4
    
    # Glasnost with Reformer
    s.set_flag(EB.THE_REFORMER_PLAYED)
    done = ts.CardHandlers.trigger_event(s, 90, ts.Player.USSR)
    assert not done
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().resolving_card == 0
    assert s.ctx().pending_ops_value == 4
    
    # Tear Down This Wall
    done = ts.CardHandlers.trigger_event(s, 96, ts.Player.US)
    assert not done
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert s.ctx().resolving_card == 0
    assert s.ctx().pending_ops_value == 3

def test_fix_defectors_no_vp_on_us_action_round():
    s = make_state()
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.victory_points = 0
    
    # US plays Defectors event on US AR
    ts.CardHandlers.trigger_event(s, 103, ts.Player.US)
    assert s.victory_points == 0 # NO VP awarded to US!

def test_fix_cambridge_five_reveals_up_to_seven_scoring_cards():
    s = make_state()
    s.turn = 3
    # Give US 6 scoring cards
    score_cards = [1, 2, 3, 37, 38, 81] # Asia, Europe, ME, CA, SEA, SA
    for sc in score_cards:
        s.set_card_location(sc, ts.CardLocation.HAND_US)
    
    argentina_id = cid("Argentina") # SA
    set_inf(s, argentina_id, 0, 0)
    
    done = ts.CardHandlers.trigger_event(s, 104, ts.Player.USSR)
    assert not done
    
    # USSR should be able to target South America (from 6th scoring card)
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[argentina_id] == 1
