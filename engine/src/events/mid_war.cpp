#include "ts/card_handlers.hpp"
#include "ts/invariant.hpp"
#include "ts/war_events.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include <algorithm>

namespace ts {

namespace mid_war {

bool trigger_brush_war(GameState& state, Player p) noexcept {
    return war_helpers::trigger_war(state, card_ids::BRUSH_WAR, p);
}


bool trigger_arms_race(GameState& state, Player p) noexcept {
    uint8_t us_mo = state.us_mil_ops;
    uint8_t ussr_mo = state.ussr_mil_ops;
    if (p == Player::US) {
        if (us_mo > ussr_mo) {
            int8_t vp = (us_mo >= state.defcon) ? 3 : 1;
            state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + vp));
            if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
        }
    } else if (p == Player::USSR) {
        if (ussr_mo > us_mo) {
            int8_t vp = (ussr_mo >= state.defcon) ? 3 : 1;
            state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - vp));
            if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
        }
    }
    return true;
}

bool trigger_cuban_missile_crisis(GameState& state, Player p) noexcept {
    state.defcon = 2;
    state.defcon_dropped_to_2 = 1;
    if (p == Player::US) {
        state.set_flag(effect_bits::CMC_ACTIVE_US);
    } else {
        state.set_flag(effect_bits::CMC_ACTIVE_USSR);
    }
    return true;
}

bool trigger_nuclear_subs(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::NUCLEAR_SUBS_ACTIVE);
    return true;
}

bool trigger_quagmire(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::QUAGMIRE_ACTIVE);
    state.clear_flag(effect_bits::NORAD_ACTIVE);
    return true;
}

bool trigger_salt_negotiations(GameState& state, Player p) noexcept {
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 2));
    state.set_flag(effect_bits::SALT_ACTIVE);

    // Player may look through discard pile and retrieve 1 non-scoring card
    uint8_t discard_non_scoring[111];
    uint8_t cnt = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::DISCARD_PILE && !CardData::is_scoring_card(i)) {
            discard_non_scoring[cnt++] = i;
        }
    }
    if (cnt == 0) return true;

    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::SALT_NEGOTIATIONS;
    return false;
}

bool trigger_bear_trap(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::BEAR_TRAP_ACTIVE);
    return true;
}

bool trigger_summit(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::NONE;
    state.ctx().decision_type = DecisionType::ROLL_DIE;
    state.ctx().resolving_card = card_ids::SUMMIT;
    state.ctx().pending_roll = RollType::SUMMIT;
    return false;
}

bool trigger_how_i_learned_to_stop_worrying(GameState& state, Player p) noexcept {
    if (p == Player::US) {
        state.us_mil_ops = 5;
    } else {
        state.ussr_mil_ops = 5;
    }
    // Phasing player sets DEFCON to 1..5: Branches 1..5
    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = card_ids::HOW_I_LEARNED_TO_STOP_WORRYING;
    return false;
}

bool trigger_junta(GameState& state, Player p) noexcept {
    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::POINT_NODE; // Stage 1: Add 2 influence in CA/SA
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 0;   // placement is mandatory; see trigger_war
    state.ctx().resolving_card = card_ids::JUNTA;
    return false;
}

bool trigger_kitchen_debates(GameState& state, Player p) noexcept {
    // The US must control more battlegrounds than the USSR. Asked through event_has_effect for
    // the same reason Our Man in Tehran does: the condition that decides whether the Event
    // happens is the condition that decides whether the card leaves the game, and it should not
    // be written twice.
    //
    // This card sets its own location -- every caller in the state machine skips it -- because
    // the Event's outcome is only known here. The generic path would now reach the same answer,
    // but leaving the card in charge of itself keeps the two from disagreeing.
    if (CardHandlers::event_has_effect(state, card_ids::KITCHEN_DEBATES, p)) {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
        state.card_locations[card_ids::KITCHEN_DEBATES] = CardLocation::REMOVED_FROM_GAME;
    } else {
        state.card_locations[card_ids::KITCHEN_DEBATES] = CardLocation::DISCARD_PILE;
    }
    return true;
}

bool trigger_missile_envy(GameState& state, Player p) noexcept {
    Player opp = get_opponent(p);

    // What the card may take, asked once and shared with the legal mask and the validation
    // rather than scanned separately here. The helper skips ctx().resolving_card for the reason
    // trigger_grain_sales does: an opponent's card played for Operations fires its own event,
    // so the event can otherwise find the very card in front of it.
    const uint8_t best_ops = CardHandlers::highest_takeable_ops(state, opp);
    uint8_t tied_cnt = 0;
    uint8_t first_tied = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (!CardHandlers::missile_envy_may_take(state, i, opp, best_ops)) continue;
        if (tied_cnt == 0) first_tied = i;
        tied_cnt++;
    }

    if (tied_cnt == 0) return true;

    if (tied_cnt == 1) {
        uint8_t chosen_card = first_tied;
        // Exchange cards
        // Both halves of the exchange are public: the opponent hands over a named card and
        // receives Missile Envy in return, so each side has seen the other's new card.
        state.card_locations[chosen_card] = hand_of(p, /*known=*/true);
        state.card_locations[card_ids::MISSILE_ENVY] = hand_of(opp, /*known=*/true);
        state.forced_card_player = opp;
        state.forced_card_id = card_ids::MISSILE_ENVY;

        const auto& c_info = CardData::get_card(chosen_card);
        if (c_info.side == p || c_info.side == Player::NONE) {
            // Event occurs immediately
            if (!state.push_context()) {
                invariant_failed("event chain too deep; card", static_cast<int>(chosen_card));
            }
            state.ctx().decision_player = p;
            state.ctx().resolving_card = chosen_card;
            const bool fired = CardHandlers::event_has_effect(state, chosen_card, p);
            bool done = CardHandlers::trigger_event(state, chosen_card, p);
            CardHandlers::relocate_played_card(state, chosen_card, fired,
                                               CardHandlers::Handover::Respect);
            if (done) state.pop_context();
            return done;
        } else {
            // Use for Ops (opponent event does NOT occur)
            state.ctx().pending_op_card = chosen_card;
            state.ctx().pending_ops_value = Operations::grant_ops_for_card(state, chosen_card, p);
            state.ctx().decision_player = p;
            state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
            state.ctx().timing_branch = 255;
            // Two different questions read two different fields, and 255 only answers one of
            // them. It stops the Event *firing* -- the guard in advance_after_ops requires
            // OPS_FIRST -- but event_occurred_on_ops_play, which decides whether a starred card
            // is removed from the game or discarded, never looks at timing_branch: it asks
            // is_opponent_card and this flag. Without it the taken card counted as an Event that
            // occurred, so the USSR taking NATO and spending its 4 Ops removed NATO from the
            // game for good. UN Intervention, the other card that plays an opponent's card
            // without its Event, has always set this.
            state.ctx().suppress_op_card_event = 1;
            state.ctx().resolving_card = 0;
            return false;
        }
    } else {
        // Opponent selects which tied card to give. Which cards those are is recomputed from
        // their hand wherever it is asked -- the mask and the validation both do it -- so the
        // tie is not written down anywhere it could go stale.
        state.ctx().decision_player = opp;
        state.ctx().decision_type = DecisionType::SELECT_CARD;
        state.ctx().remaining_steps = 1;
        // P17 6.1: set, not inherited. "If 2 or more cards are tied, opponent chooses" --
        // choosing is not optional, and declining is not one of the things they may do.
        state.ctx().allow_early_stop = 0;
        state.ctx().resolving_card = card_ids::MISSILE_ENVY;
        return false;
    }
}

bool trigger_we_will_bury_you(GameState& state, Player p) noexcept {
    if (state.defcon > 1) {
        state.defcon--;
        if (state.defcon == 2) state.defcon_dropped_to_2 = 1;
    }
    if (state.defcon == 1) {
        Player loser = state.phasing_player;
        state.victory_points = (loser == Player::US) ? -20 : 20;
        state.current_phase = Phase::GAME_OVER;
        return true;
    }
    state.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    return true;
}

bool trigger_brezhnev_doctrine(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::BREZHNEV_DOCTRINE_ACTIVE);
    return true;
}

bool trigger_portuguese_empire(GameState& state, Player p) noexcept {
    state.countries[countries::ANGOLA].add_influence(Player::USSR, 2);
    state.countries[countries::SE_AFRICAN_STS].add_influence(Player::USSR, 2);
    return true;
}

bool trigger_south_african_unrest(GameState& state, Player p) noexcept {
    // P17 section 5: no bespoke branch, and every influence the card places gets its own node.
    //
    //   stage 0: South Africa only        -- both readings require it, so this is forced
    //   stage 1: South Africa or adjacent -- South Africa ends it, an adjacent starts the split
    //   stage 2: adjacent only            -- the second half of the split
    //
    // Placing stage 0 implicitly was tried and broke six corpus games: the log writes "+1 in
    // South Africa" as its own line, so the reconstruction's first placement was South Africa,
    // which the choice node then read as "spend the rest here". An implicit placement is still a
    // decision the record contains. A forced node costs nothing at play time -- auto-advance
    // settles it -- and keeps the reconstruction able to follow the log placement for placement.
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().event_stage = 0;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 0;     // both readings place everything; set, never inherited
    state.ctx().resolving_card = card_ids::SOUTH_AFRICAN_UNREST;
    return false;
}

bool trigger_allende(GameState& state, Player p) noexcept {
    state.countries[countries::CHILE].add_influence(Player::USSR, 2);
    return true;
}

bool trigger_willy_brandt(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::TEAR_DOWN_THIS_WALL_PLAYED)) return true;

    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
    if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;

    state.countries[countries::WEST_GERMANY].add_influence(Player::USSR, 1);
    state.set_flag(effect_bits::WILLY_BRANDT_PLAYED);
    state.set_flag(effect_bits::NATO_CANCELED_WEST_GERMANY);
    return true;
}

bool trigger_muslim_revolution(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::AWACS_PLAYED)) return true;

    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 2;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 0;   // removal is mandatory; see trigger_war
    state.ctx().resolving_card = card_ids::MUSLIM_REVOLUTION;
    return false;
}

bool trigger_abm_treaty(GameState& state, Player p) noexcept {
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));
    state.ctx().decision_player = p;
    state.ctx().pending_op_card = card_ids::ABM_TREATY;
    state.ctx().pending_ops_value = Operations::grant_ops(state, 4, p);
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().resolving_card = 0;
    return false;
}

bool trigger_cultural_revolution(GameState& state, Player p) noexcept {
    if (state.china_card_holder == Player::US) {
        state.china_card_holder = Player::USSR;
        state.china_card_playable = 1;
    } else {
        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
        if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_flower_power(GameState& state, Player p) noexcept {
    if (!state.has_flag(effect_bits::EVIL_EMPIRE_PLAYED)) {
        state.set_flag(effect_bits::FLOWER_POWER_ACTIVE);
    }
    return true;
}

bool trigger_u2_incident(GameState& state, Player p) noexcept {
    state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - 1));
    if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
    state.set_flag(effect_bits::U2_INCIDENT_ACTIVE);
    return true;
}

bool trigger_opec(GameState& state, Player p) noexcept {
    if (state.has_flag(effect_bits::NORTH_SEA_OIL_PLAYED)) return true;

    constexpr std::array<uint8_t, 7> OPEC_COUNTRIES = {
        countries::EGYPT, countries::IRAN, countries::LIBYA,
        countries::SAUDI_ARABIA, countries::IRAQ, countries::GULF_STATES,
        countries::VENEZUELA
    };

    uint8_t pts = 0;
    for (uint8_t cid : OPEC_COUNTRIES) {
        if (Scoring::is_controlled_by(state, cid, Player::USSR)) {
            pts++;
        }
    }
    if (pts > 0) {
        state.victory_points = static_cast<int8_t>(std::max(-20, state.victory_points - pts));
        if (state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_lone_gunman(GameState& state, Player p) noexcept {
    // "The US reveals their hand of cards."
    reveal_hand(state, Player::US);
    // USSR conducts Operations using card Ops value (1 Op base)
    state.ctx().decision_player = Player::USSR;
    state.ctx().pending_op_card = card_ids::LONE_GUNMAN;
    state.ctx().pending_ops_value = Operations::grant_ops(state, 1, Player::USSR);
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().resolving_card = 0;
    return false;
}

bool trigger_colonial_rear_guards(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 1;
    // P17 6a: mandatory -- 1 Influence to each of any 4 -- exact count
    state.ctx().allow_early_stop = 0;
    state.ctx().resolving_card = card_ids::COLONIAL_REAR_GUARDS;
    return false;
}

bool trigger_panama_canal(GameState& state, Player p) noexcept {
    state.countries[countries::PANAMA].add_influence(Player::US, 1);
    state.countries[countries::COSTA_RICA].add_influence(Player::US, 1);
    state.countries[countries::VENEZUELA].add_influence(Player::US, 1);
    return true;
}

bool trigger_camp_david(GameState& state, Player p) noexcept {
    state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 1));
    if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    state.countries[countries::ISRAEL].add_influence(Player::US, 1);
    state.countries[countries::JORDAN].add_influence(Player::US, 1);
    state.countries[countries::EGYPT].add_influence(Player::US, 1);
    state.set_flag(effect_bits::CAMP_DAVID_PLAYED);
    return true;
}

bool trigger_puppet_governments(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3;
    state.ctx().max_per_country = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::PUPPET_GOVERNMENTS;
    return false;
}

bool trigger_grain_sales(GameState& state, Player p) noexcept {
    // Check if USSR has cards in hand. Not the card being played: it is in play, and the
    // engine leaves an Ops card in its owner's hand until the play finishes. The USSR playing
    // Grain Sales for Operations fires the US's event, which then found the very card in front
    // of it and offered it back -- at turn 9 AR6 of ts-replayer game 229 the USSR's hand is
    // empty but for Grain Sales itself, the log reads "USSR has no cards in hand to reveal",
    // and the engine asked the US how it wished to play Grain Sales.
    //
    // resolving_card is that card: an opponent's card played for Operations fires its own
    // event, and both timing branches set the field to it before triggering. Where Grain Sales
    // is fired some other way -- out of a discard pile, or by Five Year Plan -- the field is
    // Grain Sales too and it is not in anyone's hand, so skipping it changes nothing.
    const uint8_t in_play = state.ctx().resolving_card;
    uint8_t ussr_cards[111];
    uint8_t cnt = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (i != in_play && in_hand_of(state.card_locations[i], Player::USSR)) {
            ussr_cards[cnt++] = i;
        }
    }
    if (cnt == 0) {
        // USSR has no cards; US conducts Ops using 2 Ops.
        //
        // resolving_card has to be cleared, as both of the branches below clear it. Any
        // decision taken while it is set is handed to that card's own handler whatever its
        // type, so the op mode chosen here would reach Grain Sales' CHOOSE_BRANCH reader and
        // be taken for a branch: INFLUENCE is 0, which is "play the drawn card", and there is
        // no drawn card. Played directly the field is already 0 and nothing showed; fired
        // through another card's event it is not. At turn 8 AR7 of ts-replayer game 219 the US
        // plays Five Year Plan, whose event has the USSR discard Grain Sales, and the two
        // Influence it then owes were lost to a play mode for card 0.
        state.ctx().decision_player = Player::US;
        state.ctx().pending_op_card = card_ids::GRAIN_SALES;
        state.ctx().pending_ops_value = Operations::grant_ops(state, 2, Player::US);
        state.ctx().resolving_card = 0;
        state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
        return false;
    }

    uint32_t chosen_idx = Prng::random_index(state.rng_state, cnt);
    uint8_t chosen_card = ussr_cards[chosen_idx];
    // Drawn out of the USSR hand and shown to the US, which is a location and not a note kept
    // beside one. PEEKED_TEMP is where the engine already puts a card someone is looking at,
    // and it is what the observation reads, so the player being asked to keep or return the
    // card can now see which card it is without a special case.
    //
    // Both players know it: the USSR watches it leave their hand, and the log records the
    // reveal. Whichever way the US answers, the card leaves PEEKED_TEMP in the same step.
    state.card_locations[chosen_card] = CardLocation::PEEKED_TEMP;

    // US chooses: Branch 0 = Play drawn card, Branch 1 = Return card and conduct Ops with 2 Ops
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::CHOOSE_BRANCH;
    state.ctx().resolving_card = card_ids::GRAIN_SALES;
    return false;
}

bool trigger_john_paul_ii(GameState& state, Player p) noexcept {
    state.countries[countries::POLAND].remove_influence(Player::USSR, 2);
    state.countries[countries::POLAND].add_influence(Player::US, 1);
    state.set_flag(effect_bits::JOHN_PAUL_II_PLAYED);
    return true;
}

bool trigger_latin_death_squads(GameState& state, Player p) noexcept {
    if (p == Player::US) {
        state.set_flag(effect_bits::DEATH_SQUADS_US);
    } else {
        state.set_flag(effect_bits::DEATH_SQUADS_USSR);
    }
    return true;
}

bool trigger_oas_founded(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 2;
    state.ctx().max_per_country = 2;
    // P17 6a: mandatory -- a total of 2 -- exact count
    state.ctx().allow_early_stop = 0;
    state.ctx().resolving_card = card_ids::OAS_FOUNDED;
    return false;
}

bool trigger_nixon_china_card(GameState& state, Player p) noexcept {
    if (state.china_card_holder == Player::USSR) {
        state.china_card_holder = Player::US;
        state.china_card_playable = 0; // Face down
    } else {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + 2));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_sadat_expels_soviets(GameState& state, Player p) noexcept {
    state.countries[countries::EGYPT].ussr_influence = 0;
    state.countries[countries::EGYPT].add_influence(Player::US, 1);
    return true;
}

bool trigger_shuttle_diplomacy(GameState& state, Player p) noexcept {
    state.set_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    return true;
}

bool trigger_voice_of_america(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 4;
    state.ctx().max_per_country = 2;
    // P17 6a: mandatory -- remove 4 -- exact count
    state.ctx().allow_early_stop = 0;
    state.ctx().resolving_card = card_ids::THE_VOICE_OF_AMERICA;
    return false;
}

bool trigger_liberation_theology(GameState& state, Player p) noexcept {
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 3;
    state.ctx().max_per_country = 2;
    // P17 6a: mandatory -- a total of 3 -- exact count
    state.ctx().allow_early_stop = 0;
    state.ctx().resolving_card = card_ids::LIBERATION_THEOLOGY;
    return false;
}

bool trigger_ussuri_river(GameState& state, Player p) noexcept {
    if (state.china_card_holder == Player::USSR) {
        state.china_card_holder = Player::US;
        state.china_card_playable = 1; // Face up
        return true;
    } else {
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 4;
        state.ctx().max_per_country = 2;
        // P17 6a: mandatory -- a total of 4 to Asia -- exact count
        state.ctx().allow_early_stop = 0;
        state.ctx().resolving_card = card_ids::USSURI_RIVER_SKIRMISH;
        return false;
    }
}

bool trigger_ask_not(GameState& state, Player p) noexcept {
    // US selects cards to discard: SELECT_CARD with allow_early_stop (CONFIRM_DONE / 0x80)
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = ask_not::MAX_DISCARDS;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::ASK_NOT_WHAT_YOUR_COUNTRY_CAN_DO_FOR_YOU;
    return false;
}

bool trigger_alliance_for_progress(GameState& state, Player p) noexcept {
    uint8_t pts = 0;
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& c_info = MapData::get_country(i);
        if (c_info.battleground && (c_info.region == Region::CENTRAL_AMERICA || c_info.region == Region::SOUTH_AMERICA)) {
            if (Scoring::is_controlled_by(state, i, Player::US)) {
                pts++;
            }
        }
    }
    if (pts > 0) {
        state.victory_points = static_cast<int8_t>(std::min(20, state.victory_points + pts));
        if (state.victory_points >= 20) state.current_phase = Phase::GAME_OVER;
    }
    return true;
}

bool trigger_one_small_step(GameState& state, Player p) noexcept {
    uint8_t cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    uint8_t opp_track = (p == Player::US) ? state.ussr_space_track : state.us_space_track;
    if (cur_track < opp_track) {
        uint8_t new_track = std::min(8, cur_track + 2);
        if (p == Player::US) state.us_space_track = new_track;
        else state.ussr_space_track = new_track;

        // VP award from final space moved into
        const auto& box = SpaceRace::get_box_info(new_track);
        uint8_t vp = (new_track > opp_track) ? box.vp_first : box.vp_second;
        if (vp > 0) {
            int32_t vp_delta = (p == Player::US) ? vp : -static_cast<int32_t>(vp);
            state.victory_points = static_cast<int8_t>(std::clamp(static_cast<int32_t>(state.victory_points) + vp_delta, -20, 20));
            if (state.victory_points >= 20 || state.victory_points <= -20) state.current_phase = Phase::GAME_OVER;
        }
    }
    return true;
}

bool trigger_che(GameState& state, Player p) noexcept {
    // USSR performs Coup 1 in non-BG in CA/SA/Africa
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 1;
    state.ctx().event_stage = 0;   // the first of Che's two coups
    state.ctx().resolving_card = card_ids::CHE;
    return false;
}

bool trigger_our_man_in_tehran(GameState& state, Player p) noexcept {
    // The US must control a Middle East country. Asked through event_has_effect so the same
    // answer decides whether the card is removed from the game or discarded -- the condition
    // has one definition, not one here and a copy at the removal site.
    if (!CardHandlers::event_has_effect(state, card_ids::OUR_MAN_IN_TEHRAN, p)) return true;

    // Draw up to 5 cards from draw deck
    uint8_t draw_pool[111];
    uint8_t draw_cnt = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (state.card_locations[i] == CardLocation::DRAW_DECK) draw_pool[draw_cnt++] = i;
    }
    if (draw_cnt == 0) {
        for (uint8_t i = 1; i <= 110; ++i) {
            if (i == card_ids::THE_CHINA_CARD) continue;
            if (state.card_locations[i] == CardLocation::DISCARD_PILE) {
                state.card_locations[i] = CardLocation::DRAW_DECK;
                draw_pool[draw_cnt++] = i;
            }
        }
    }
    if (draw_cnt == 0) return true;

    uint8_t sample_count = std::min(static_cast<uint8_t>(5), draw_cnt);
    for (uint8_t k = 0; k < sample_count; ++k) {
        uint32_t chosen_idx = Prng::random_index(state.rng_state, draw_cnt);
        uint8_t drawn_card = draw_pool[chosen_idx];
        draw_pool[chosen_idx] = draw_pool[--draw_cnt];
        state.card_locations[drawn_card] = CardLocation::PEEKED_TEMP;
    }
    // The peek is the set of cards at PEEKED_TEMP. A list of the same ids was kept beside it
    // and shuffled down on each discard; the handler already trusted the locations over it.

    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    state.ctx().remaining_steps = sample_count;
    state.ctx().allow_early_stop = 1;
    state.ctx().resolving_card = card_ids::OUR_MAN_IN_TEHRAN;
    return false;
}

} // namespace mid_war

} // namespace ts
