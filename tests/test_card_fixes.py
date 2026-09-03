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
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, colombia_id, 0))
    assert s.ctx().decision_type == ts.DecisionType.ROLL_DIE
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.ROLL_DIE, 3, 0, 0))
    
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
    # In SELECT_OP_MODE, INFLUENCE mode is legal for Junta to allow declining bonus coup/realign
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[0] == 1 # INFLUENCE legal (decline option)
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
    # The 2 adjacent Influence are placed one at a time -- the card says "any countries", so
    # they may be split. Both into Angola is still reachable by choosing it twice.
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, angola_id))
    assert not done
    assert get_inf(s, angola_id)[1] == 1
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, angola_id))
    assert done == True
    assert get_inf(s, angola_id)[1] == 2


def test_fix_card_53_south_african_unrest_branch1_splits_across_neighbours():
    """Replay 112 turn 5 AR2: the USSR put 1 in Botswana and 1 in Angola, not 2 in one."""
    s = make_state()
    sa_id, angola_id, botswana_id = cid("South Africa"), cid("Angola"), cid("Botswana")
    set_inf(s, sa_id, 0, 0)
    set_inf(s, angola_id, 0, 0)
    set_inf(s, botswana_id, 0, 0)
    ts.CardHandlers.trigger_event(s, 53, ts.Player.USSR)
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 1))
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, botswana_id))
    assert not done
    done = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, angola_id))
    assert done == True
    assert get_inf(s, sa_id)[1] == 1
    assert get_inf(s, botswana_id)[1] == 1
    assert get_inf(s, angola_id)[1] == 1

def test_fix_card_59_flower_power_war_cards_for_ops():
    s = make_state()
    s.set_flag(EB.FLOWER_POWER_ACTIVE)
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().pending_op_card = 36 # Brush War (actual War card)
    s.victory_points = 0
    # US plays Brush War for Ops
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
    assert mask[0] == 1 # INFLUENCE allowed (decline option)

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

    # 1. can_trigger_event returns False for US on Action Round
    assert not ts.CardHandlers.can_trigger_event(s, 103, ts.Player.US)

    # 2. SELECT_PLAY_MODE mask has EVENT=0, OPS=1
    s.ctx().decision_type = ts.DecisionType.SELECT_PLAY_MODE
    s.ctx().decision_player = ts.Player.US
    s.ctx().pending_op_card = 103
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[int(ts.PlayMode.EVENT)] == 0
    assert mask[int(ts.PlayMode.OPS)] == 1

    flat_mask = ts.Engine.get_flat_action_mask(s)
    assert flat_mask[110] == 0
    assert flat_mask[111] == 1

    # 3. StateMachine rejects EVENT play mode
    step_ok = ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.EVENT), 0, 0))
    assert not step_ok

    # 4. Trigger event directly does not change VP
    ts.CardHandlers.trigger_event(s, 103, ts.Player.US)
    assert s.victory_points == 0

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

def test_fix_card_97_an_evil_empire_not_war_card():
    # Card 97 is not a war card; Flower Power should not trigger when US plays it for ops
    card_info = ts.CardData.get_card_info(97)
    assert not card_info["is_war_card"], "Card 97 (An Evil Empire) should not have is_war flag set"

    s = make_state()
    s.set_flag(EB.FLOWER_POWER_ACTIVE)
    s.victory_points = 0
    s.turn = 8
    s.current_phase = ts.Phase.ACTION_ROUND
    s.phasing_player = ts.Player.US
    s.set_card_location(97, ts.CardLocation.HAND_US)

    # US selects Card 97 for play mode OPS
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 97))
    ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, int(ts.PlayMode.OPS)))

    # VP should still be 0 (Flower Power did not award 2 VP to USSR)
    assert s.victory_points == 0

def test_fix_card_46_how_i_learned_defcon_levels():
    # Branches 1..5 set DEFCON to levels 1..5
    for target_defcon in range(1, 6):
        s = make_state()
        s.defcon = 3
        done = ts.CardHandlers.trigger_event(s, 46, ts.Player.US)
        assert not done
        mask = ts.Engine.get_legal_action_mask(s)
        assert mask[target_defcon] == 1
        ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, target_defcon))
        assert s.defcon == target_defcon

def test_fix_card_45_summit_pass_branch():
    s = make_state()
    s.defcon = 2
    # Equal domination -> roll determines winner or tie
    set_inf(s, "West Germany", 4, 0)
    done = ts.CardHandlers.trigger_event(s, 45, ts.Player.US)
    if not done:
        mask = ts.Engine.get_legal_action_mask(s)
        assert mask[2] == 1 # Branch 2 (pass / no change) is legal
        ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 2))
        assert s.defcon == 2 # DEFCON unchanged

def test_fix_card_36_brush_war_nato_validation_us_control():
    s = make_state()
    s.set_flag(EB.NATO_ACTIVE)
    # 1. Uncontrolled Western Europe country (e.g. Spain/Portugal, stab 2) can be targeted by USSR
    spain_id = cid("Spain/Portugal")
    set_inf(s, spain_id, 0, 0)
    ts.CardHandlers.trigger_event(s, 36, ts.Player.USSR)
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[spain_id] == 1
    # Step targeting Spain/Portugal is accepted -> transitions to ROLL_DIE
    res = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, spain_id, 0))
    assert res == False
    assert s.ctx().decision_type == ts.DecisionType.ROLL_DIE
    res = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.ROLL_DIE, 6, 0, 0))
    assert res == True

    # 2. US-controlled European country (e.g. Greece, stab 2, US inf 2) cannot be targeted by USSR
    s2 = make_state()
    s2.set_flag(EB.NATO_ACTIVE)
    greece_id = cid("Greece")
    set_inf(s2, greece_id, 2, 0) # US control
    ts.CardHandlers.trigger_event(s2, 36, ts.Player.USSR)
    mask2 = ts.Engine.get_legal_action_mask(s2)
    assert mask2[greece_id] == 0
    res2 = ts.CardHandlers.handle_event_step(s2, ts.MicroAction(ts.DecisionType.POINT_NODE, greece_id, 6))
    assert res2 == False

def test_fix_card_23_marshall_plan_canada_selection():
    s = make_state()
    canada_id = cid("Canada") # Country 0
    set_inf(s, canada_id, 0, 0)
    done = ts.CardHandlers.trigger_event(s, 23, ts.Player.US)
    assert not done
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[canada_id] == 1
    # Placing influence in Canada (Country 0) should add 1 US influence, not exit early
    res = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, canada_id))
    assert res == False # Still has remaining steps to place
    assert s.get_country(canada_id).us_influence == 1

def test_fix_card_66_puppet_governments_canada_selection():
    s = make_state()
    canada_id = cid("Canada") # Country 0
    set_inf(s, canada_id, 0, 0)
    done = ts.CardHandlers.trigger_event(s, 66, ts.Player.US)
    assert not done
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[canada_id] == 1
    # Placing influence in Canada (Country 0) should add 1 US influence, not exit early
    res = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, canada_id))
    assert res == False
    assert s.get_country(canada_id).us_influence == 1

def test_fix_card_14_comecon_metadata():
    card_info = ts.CardData.get_card_info(14)
    assert card_info["name"] == "Comecon"


def test_fix_card_12_romanian_abdication_preserves_higher_ussr_influence():
    s = make_state()
    rom_id = cid("Romania")
    # USSR already has 5 influence, US has 2
    set_inf(s, rom_id, 2, 5)
    ts.CardHandlers.trigger_event(s, 12, ts.Player.USSR)
    assert s.get_country(rom_id).us_influence == 0
    assert s.get_country(rom_id).ussr_influence == 5 # Does not downgrade 5 to 3

    # When USSR has 1 influence, tops up to 3
    s2 = make_state()
    set_inf(s2, rom_id, 1, 1)
    ts.CardHandlers.trigger_event(s2, 12, ts.Player.USSR)
    assert s2.get_country(rom_id).us_influence == 0
    assert s2.get_country(rom_id).ussr_influence == 3


def test_fix_card_13_arab_israeli_war_checks_israel_us_control():
    s = make_state()
    israel_id = cid("Israel")
    # US controls Israel (stab 4, US 4, USSR 0)
    set_inf(s, israel_id, 4, 0)
    # Adjacent countries are uncontrolled
    for c in ["Egypt", "Jordan", "Lebanon", "Syria"]:
        set_inf(s, cid(c), 0, 0)
    # Forced roll 4 with Israel US controlled: modifier is -1, total = 3 -> Failure
    ts.CardHandlers.trigger_event(s, 13, ts.Player.USSR)
    assert s.ctx().decision_type == ts.DecisionType.ROLL_DIE
    ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.ROLL_DIE, 4, 0, 0))
    assert s.victory_points == 0
    assert s.get_country(israel_id).us_influence == 4

    # Forced roll 5 with Israel US controlled: total = 4 -> Success
    s2 = make_state()
    set_inf(s2, israel_id, 4, 0)
    for c_name in ["Egypt", "Jordan", "Lebanon", "Syria"]:
        set_inf(s2, cid(c_name), 0, 0)
    ts.CardHandlers.trigger_event(s2, 13, ts.Player.USSR)
    assert s2.ctx().decision_type == ts.DecisionType.ROLL_DIE
    ts.CardHandlers.handle_event_step(s2, ts.MicroAction(ts.DecisionType.ROLL_DIE, 5, 0, 0))
    assert s2.victory_points == -2
    assert s2.get_country(israel_id).ussr_influence == 4


def test_fix_card_14_comecon_prevents_duplicate_country_placement():
    s = make_state()
    poland_id = cid("Poland")
    set_inf(s, poland_id, 0, 0)
    done = ts.CardHandlers.trigger_event(s, 14, ts.Player.USSR)
    assert not done
    # Step 1: place 1 on Poland
    res = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, poland_id))
    assert res == False
    assert s.get_country(poland_id).ussr_influence == 1

    # Step 2: raw action attempting to place on Poland again must be rejected (returns False and does not increment)
    res2 = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, poland_id))
    assert res2 == False
    assert s.get_country(poland_id).ussr_influence == 1


def test_fix_card_27_us_japan_pact_guarantees_us_control_with_existing_ussr_inf():
    s = make_state()
    japan_id = cid("Japan")
    # USSR has 2 influence in Japan. Japan stability is 4. US needs 2 + 4 = 6 for Control.
    set_inf(s, japan_id, 0, 2)
    ts.CardHandlers.trigger_event(s, 27, ts.Player.US)
    assert s.get_country(japan_id).us_influence == 6
    assert ts.Scoring.is_controlled_by(s, japan_id, ts.Player.US)


def test_fix_card_32_un_intervention_applies_ops_modifiers():
    s = make_state()
    # USSR plays Red Scare/Purge on US
    s.set_flag(EB.PURGE_US_ACTIVE)
    # US holds UN Intervention (32) and Arab-Israeli War (13, 2 Ops USSR card)
    s.set_card_location(32, ts.CardLocation.HAND_US)
    s.set_card_location(13, ts.CardLocation.HAND_US)
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # US plays UN Intervention as event
    done = ts.CardHandlers.trigger_event(s, 32, ts.Player.US)
    assert not done
    # US selects Arab-Israeli War (13)
    done2 = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.SELECT_CARD, 13))
    assert not done2
    # Pending ops value should be 2 - 1 = 1 (Red Scare applied)
    assert s.ctx().pending_ops_value == 1
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE


def test_fix_card_47_junta_allows_optional_bonus_op_decline():
    s = make_state()
    nicaragua_id = cid("Nicaragua")
    set_inf(s, nicaragua_id, 0, 2) # USSR has influence in CA
    # US plays Junta
    s.ctx().decision_player = ts.Player.US
    s.ctx().decision_type = ts.DecisionType.POINT_NODE
    done = ts.CardHandlers.trigger_event(s, 47, ts.Player.US)
    assert not done
    # Step 1: Place 2 influence in Nicaragua
    done2 = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, nicaragua_id))
    assert not done2
    assert s.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    # Action mask MUST allow INFLUENCE (mode 0) as option to decline bonus coup/realign
    mask = ts.Engine.get_legal_action_mask(s)
    assert mask[int(ts.OpMode.INFLUENCE)] == 1
    # Selecting INFLUENCE mode cleanly finishes the card ops immediately
    step_ok = ts.Engine.step(s, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, int(ts.OpMode.INFLUENCE)))
    assert step_ok == True


def test_fix_card_73_shuttle_diplomacy_excluded_from_final_scoring():
    s = make_state()
    s.set_flag(EB.SHUTTLE_DIPLOMACY_ACTIVE)
    # Middle East setup:
    # Egypt (BG): USSR 2 (controlled)
    # Israel (BG): US 4 (controlled)
    # Iraq (BG): USSR 3 (controlled)
    # Saudi Arabia (BG): USSR 3 (controlled)
    # Total ME BG: 5 (Egypt, Israel, Iraq, Saudi Arabia, Libya)
    # USSR has 3 BG, US has 1 BG.
    set_inf(s, cid("Egypt"), 0, 2)
    set_inf(s, cid("Israel"), 4, 0)
    set_inf(s, cid("Iraq"), 0, 3)
    set_inf(s, cid("Saudi Arabia"), 0, 3)
    set_inf(s, cid("Libya"), 0, 0)
    # During normal scoring, Shuttle Diplomacy reduces USSR effective BG from 3 to 2
    me_normal = ts.Scoring.evaluate_region(s, ts.Region.MIDDLE_EAST, False)
    assert me_normal.ussr_battlegrounds == 3

    # During final scoring, Shuttle Diplomacy is NOT applied
    s.victory_points = 0
    ts.Scoring.execute_final_scoring(s)
    # In final scoring, Shuttle Diplomacy flag is cleared and card moved to discard
    assert not s.has_flag(EB.SHUTTLE_DIPLOMACY_ACTIVE)


def test_fix_card_76_ussuri_river_skirmish_adds_us_influence():
    s = make_state()
    s.china_card_holder = ts.Player.US # US already holds China Card
    japan_id = cid("Japan") # Asia
    set_inf(s, japan_id, 0, 0)
    done = ts.CardHandlers.trigger_event(s, 76, ts.Player.US)
    assert not done
    assert s.ctx().decision_type == ts.DecisionType.POINT_NODE
    assert s.ctx().remaining_steps == 4

    # US places 2 influence in Japan
    step1 = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, japan_id))
    assert step1 == False
    assert s.get_country(japan_id).us_influence == 1
    assert s.ctx().remaining_steps == 3

    step2 = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.POINT_NODE, japan_id))
    assert step2 == False
    assert s.get_country(japan_id).us_influence == 2
    assert s.ctx().remaining_steps == 2


def test_fix_card_106_norad_triggers_on_olympic_games_boycott_defcon_2():
    s = make_state()
    s.defcon = 3
    s.turn = 4
    s.current_phase = ts.Phase.ACTION_ROUND
    s.set_flag(EB.NORAD_ACTIVE)
    set_inf(s, cid("Canada"), 2, 0) # US controls Canada
    # USSR plays Olympic Games as event
    done = ts.CardHandlers.trigger_event(s, 20, ts.Player.USSR)
    assert not done
    assert s.ctx().decision_player == ts.Player.US # US decides to participate or boycott
    # US boycotts (branch 1)
    done_boycott = ts.CardHandlers.handle_event_step(s, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 1))
    assert not done_boycott
    assert s.defcon == 2
    assert s.defcon_dropped_to_2_in_ar == 1


def test_chain_scenario_1_fyp_grainsales_starwars_abmtreaty_full_ar():
    """Scenario 1: US plays FYP -> Grain Sales -> Star Wars -> ABM Treaty -> Coup -> AR complete."""
    st = ts.GameState()
    ts.Engine.init_game(st, 42)

    st.turn = 7
    st.action_round = 1
    st.current_phase = ts.Phase.ACTION_ROUND
    st.phasing_player = ts.Player.US
    st.defcon = 4
    st.us_space_track = 4
    st.ussr_space_track = 1

    for i in range(1, 111):
        st.set_card_location(i, ts.CardLocation.DRAW_DECK)

    st.set_card_location(5, ts.CardLocation.HAND_US) # Five Year Plan
    st.set_card_location(67, ts.CardLocation.HAND_USSR) # Grain Sales
    st.set_card_location(85, ts.CardLocation.HAND_USSR) # Star Wars
    st.set_card_location(57, ts.CardLocation.DISCARD_PILE) # ABM Treaty

    st.set_country(25, 0, 2) # Iran (ID 25)

    st.ctx().decision_player = ts.Player.US
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # 1. US plays Five Year Plan (#5) for EVENT
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0))

    # 2. Grain Sales prompts Branch 0 (play drawn Star Wars)
    if st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0))

    # 3. Star Wars prompts SELECT_CARD from discard
    assert st.ctx().decision_player == ts.Player.US
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 57, 0, 0)) # ABM Treaty

    # 4. ABM Treaty triggers: DEFCON 5, 4 Ops
    assert st.defcon == 5
    assert st.ctx().decision_player == ts.Player.US
    assert st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE
    assert st.ctx().pending_ops_value == 4

    # 5. US chooses COUP (1) on Iran with forced roll 5
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 25, 0, 0))
    assert st.ctx().decision_type == ts.DecisionType.ROLL_DIE
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 5, 0, 0))

    assert st.defcon == 4
    assert st.us_mil_ops == 4

    # Verification: Full AR finishes cleanly and advances to USSR AR 2
    assert st.current_phase == ts.Phase.ACTION_ROUND
    assert st.phasing_player == ts.Player.USSR
    assert st.action_round == 2
    assert st.ctx().decision_player == ts.Player.USSR
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD


def test_chain_scenario_2_starwars_fyp_grainsales_glasnost_full_ar():
    """Scenario 2: US plays Star Wars -> FYP -> Grain Sales -> Glasnost -> Coup -> USSR Realign -> AR complete."""
    st = ts.GameState()
    ts.Engine.init_game(st, 42)

    st.turn = 9
    st.action_round = 2
    st.current_phase = ts.Phase.ACTION_ROUND
    st.phasing_player = ts.Player.US
    st.defcon = 4
    st.us_space_track = 5
    st.ussr_space_track = 2

    for i in range(1, 111):
        st.set_card_location(i, ts.CardLocation.DRAW_DECK)

    st.set_card_location(85, ts.CardLocation.HAND_US) # Star Wars
    st.set_card_location(5, ts.CardLocation.DISCARD_PILE) # Five Year Plan
    st.set_card_location(67, ts.CardLocation.HAND_USSR) # Grain Sales
    st.set_card_location(90, ts.CardLocation.HAND_USSR) # Glasnost

    st.set_country(67, 0, 3) # Cuba
    st.set_country(3, 1, 2)  # Poland

    st.ctx().decision_player = ts.Player.US
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # 1. US plays Star Wars (#85) for EVENT
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 85, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0))

    # 2. Star Wars selects Five Year Plan (#5) from discard
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))

    # 3. FYP discards Grain Sales -> Grain Sales draws Glasnost -> Branch 0
    if st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0))

    # 4. US plays Glasnost (#90) for OPS
    if st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0)) # OPS

    if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0)) # OPS_FIRST

    # 5. US chooses COUP (1) on Cuba with roll 4
    if st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 1, 0, 0))
    if st.ctx().decision_type == ts.DecisionType.POINT_NODE:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 67, 0, 0))
    if st.ctx().decision_type == ts.DecisionType.ROLL_DIE:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 4, 0, 0))

    # 6. USSR Glasnost event resolves
    while st.ctx().decision_player == ts.Player.USSR and st.ctx().decision_type != ts.DecisionType.SELECT_CARD:
        if st.ctx().decision_type == ts.DecisionType.POINT_NODE:
            if st.ctx().allow_early_stop:
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 255, 0, 0x80))
            else:
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 3, 0, 0))
        elif st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0))
        elif st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 2, 0, 0)) # REALIGN
        elif st.ctx().decision_type == ts.DecisionType.ROLL_DIE or st.ctx().decision_player == ts.Player.NONE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        else:
            break

    # Verification: Full AR finishes cleanly and advances to USSR AR 3
    assert st.current_phase == ts.Phase.ACTION_ROUND
    assert st.phasing_player == ts.Player.USSR
    assert st.action_round == 3
    assert st.ctx().decision_player == ts.Player.USSR
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD


def test_chain_scenario_3_ussr_grainsales_starwars_fyp_kal007_full_ar():
    """Scenario 3: USSR plays Grain Sales -> Star Wars -> FYP -> KAL-007 -> Realign -> USSR Ops -> AR complete."""
    st = ts.GameState()
    ts.Engine.init_game(st, 42)

    st.turn = 9
    st.action_round = 3
    st.current_phase = ts.Phase.ACTION_ROUND
    st.phasing_player = ts.Player.USSR
    st.defcon = 4
    st.us_space_track = 6
    st.ussr_space_track = 3

    for i in range(1, 111):
        st.set_card_location(i, ts.CardLocation.DRAW_DECK)

    st.set_card_location(67, ts.CardLocation.HAND_USSR)
    st.set_card_location(5, ts.CardLocation.DISCARD_PILE)
    st.set_card_location(85, ts.CardLocation.HAND_USSR)

    st.set_country(44, 4, 1) # South Korea (ID 44)
    st.set_country(26, 3, 0) # Japan
    st.set_country(29, 0, 0) # Afghanistan

    st.ctx().decision_player = ts.Player.USSR
    st.ctx().decision_type = ts.DecisionType.SELECT_CARD

    # 1. USSR plays Grain Sales for OPS
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 67, 0, 0))
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 1, 0, 0))

    # 2. EVENT_FIRST
    if st.ctx().decision_type == ts.DecisionType.CHOOSE_TIMING_BRANCH:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_TIMING_BRANCH, 1, 0, 0))

    # 3. US chooses Branch 0 (play drawn Star Wars)
    if st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0))

    st.set_card_location(89, ts.CardLocation.HAND_USSR) # Give USSR KAL-007

    # 3b. US plays Star Wars for EVENT
    if st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE:
        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_PLAY_MODE, 0, 0, 0))

    # 4. Star Wars selects Five Year Plan from discard
    assert st.ctx().decision_player == ts.Player.US
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD
    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_CARD, 5, 0, 0))

    # 5. FYP discards KAL-007 -> US executes realignment
    while st.ctx().decision_player == ts.Player.US and st.ctx().decision_type != ts.DecisionType.SELECT_CARD:
        if st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 2, 0, 0)) # REALIGN
        elif st.ctx().decision_type == ts.DecisionType.POINT_NODE:
            if st.ctx().allow_early_stop:
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 255, 0, 0x80))
            else:
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 44, 0, 0))
        elif st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.CHOOSE_BRANCH, 0, 0, 0))
        else:
            break

    # 6. USSR executes Grain Sales Ops
    while st.ctx().decision_player == ts.Player.USSR and st.ctx().decision_type != ts.DecisionType.SELECT_CARD:
        if st.ctx().decision_type == ts.DecisionType.SELECT_OP_MODE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.SELECT_OP_MODE, 0, 0, 0)) # INFLUENCE
        elif st.ctx().decision_type == ts.DecisionType.POINT_NODE:
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.POINT_NODE, 29, 0, 0))
        else:
            break

    # Verification: AR finishes cleanly and advances
    assert st.current_phase == ts.Phase.ACTION_ROUND
    assert st.ctx().decision_type == ts.DecisionType.SELECT_CARD
