#include "ts/card_handlers.hpp"
#include "ts/defcon.hpp"
#include "ts/invariant.hpp"
#include "ts/war_events.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/state_machine.hpp"
#include <algorithm>
#include "ts/victory_points.hpp"

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
            award_vp(state, vp);
        }
    } else if (p == Player::USSR) {
        if (ussr_mo > us_mo) {
            int8_t vp = (ussr_mo >= state.defcon) ? 3 : 1;
            award_vp(state, -vp);
        }
    }
    return true;
}

bool trigger_cuban_missile_crisis(GameState& state, Player p) noexcept {
    // NORAD asks whether DEFCON "moved to 2" during the round. Setting it to 2 from 2 is not a
    // move, so it arms nothing -- the case at hand whenever the crisis is played at DEFCON 2.
    if (state.defcon != 2) state.defcon_dropped_to_2 = 1;
    state.defcon = 2;
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
    // The US must control more battlegrounds than the USSR. Asked through event_has_effect, as Our
    // Man in Tehran is: the condition that decides whether the Event happens is the one every
    // caller already asks before firing it, and relocate_played_card then removes the card or
    // discards it on that answer -- the same path as NATO or Solidarity, whose prerequisites
    // live in can_trigger_event.
    //
    // Until E6 this handler placed the card itself and relocate_played_card skipped Kitchen
    // Debates unconditionally, so a US play for Operations -- where no Event runs -- left it in
    // the US hand, playable again every action round.
    if (CardHandlers::event_has_effect(state, card_ids::KITCHEN_DEBATES, p)) {
        award_vp(state, 2);
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

    // Under ALWAYS_ASK_DISCARD the giver is asked even when one card qualifies (the mask then
    // holds just that card): exchanging at once told the Missile Envy player there was no tie,
    // which at a table nobody learns. Read on either seat, as for Blockade.
    if (tied_cnt == 1 && !either_seat_plays_rule(state, obs_features::ALWAYS_ASK_DISCARD)) {
        uint8_t chosen_card = first_tied;
        // Exchange cards
        // Both halves of the exchange are public: the opponent hands over a named card and
        // receives Missile Envy in return, so each side has seen the other's new card.
        state.card_locations[chosen_card] = hand_of(p, /*known=*/true);
        state.card_locations[card_ids::MISSILE_ENVY] = hand_of(opp, /*known=*/true);
        state.forced_card_player = opp;
        state.forced_card_id = card_ids::MISSILE_ENVY;

        const auto& c_info = CardData::get_card(chosen_card);
        // UN Intervention "cannot be played during the Headline Phase": taken by a headlined Missile
        // Envy it is used for its Ops, as Star Wars and Grain Sales already refuse it there (audit
        // E-11, 2026-11-22; engine/AGENTS.md section 23).
        const bool un_in_headline = chosen_card == card_ids::UN_INTERVENTION && state.current_phase == Phase::HEADLINE;
        if ((c_info.side == p || c_info.side == Player::NONE) && !un_in_headline) {
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
            Operations::grant_card_ops_to_ctx(state, chosen_card, p);
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
        // Opponent selects which tied card to give (or, under ALWAYS_ASK_DISCARD, hands over
        // the only one). Which cards those are is recomputed from
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
        // The phasing player loses, as everywhere; resolve_defcon_one_loss also records a
        // loss the US was forced into by playing this USSR card for its Ops.
        resolve_defcon_one_loss(state, p);
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

    award_vp(state, -1);

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
    Operations::grant_event_ops_to_ctx(state, 4, p);
    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
    state.ctx().resolving_card = 0;
    return false;
}

bool trigger_cultural_revolution(GameState& state, Player p) noexcept {
    if (state.china_card_holder == Player::US) {
        state.china_card_holder = Player::USSR;
        state.china_card_playable = 1;
    } else {
        award_vp(state, -1);
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
    award_vp(state, -1);
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
        award_vp(state, -pts);
    }
    return true;
}

bool trigger_lone_gunman(GameState& state, Player p) noexcept {
    // "The US reveals their hand of cards."
    reveal_hand(state, Player::US);
    // USSR conducts Operations using card Ops value (1 Op base)
    state.ctx().decision_player = Player::USSR;
    state.ctx().pending_op_card = card_ids::LONE_GUNMAN;
    Operations::grant_event_ops_to_ctx(state, 1, Player::USSR);
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
    award_vp(state, 1);
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
        if (i != in_play && !state.card_in_play(i) && in_hand_of(state.card_locations[i], Player::USSR)) {
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
        Operations::grant_event_ops_to_ctx(state, 2, Player::US);
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
    // Into the US hand, marked known, BEFORE the decision rather than after it.
    //
    // PEEKED_TEMP would say "shown, not taken", which is the truer picture -- but it splits the
    // engine in two. The mask is generated against this position and `step` runs against it, and
    // every predicate that asks where the card IS then disagrees between them: with the card
    // staged, ActionMask's Missile Envy forced-play test reads `in_hand_of` as false and offers
    // the Event, and `step`'s identical test reads it as true and refuses. The old two-decision
    // shape hid this, because the card moved BETWEEN the branch and the play mode, so each mask
    // was generated against the location its own step would see. Collapsing to one decision
    // removes that gap, and the card must therefore already be where the resolution expects it.
    //
    // Nothing is leaked by moving it early: the USSR is never the decision player anywhere in
    // Grain Sales, so no observation is taken for them while the card sits in the US hand, and
    // the decline puts it straight back. It also settles the Our Man in Tehran hazard for free --
    // a card still at PEEKED_TEMP when that Event peeks five cards of its own would be inside
    // its own peek.
    state.card_locations[chosen_card] = hand_of(Player::US, /*known=*/true);

    // P17 section 5: the merged resolution node, on Grain Sales' OWN frame. Nothing is pushed.
    //
    // The first attempt at this pushed a child frame for the drawn card before the US had
    // answered, so the parent had to survive a possible decline -- and nothing popped it when a
    // play completed normally. The choice comes first, and once it is made Grain Sales owes
    // nothing either way, so there is nothing to come back to:
    //
    //   CONFIRM_DONE  -> hand the card back and take Grain Sales' own 2 Ops, on this frame
    //   any Resolution-> discard Grain Sales and REPLACE this frame with the drawn card's play
    //
    state.ctx().decision_player = Player::US;
    state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
    state.ctx().pending_op_card = chosen_card;
    state.ctx().resolving_card = card_ids::GRAIN_SALES;
    // P17 6.1: set, not inherited. Handing the card back is always one of the answers, and it is
    // the ONLY one when a headline draws UN Intervention -- so callers that read this rather than
    // the flat mask still know the decline exists.
    state.ctx().allow_early_stop = 1;
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
        award_vp(state, 2);
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
        award_vp(state, pts);
    }
    return true;
}

bool trigger_one_small_step(GameState& state, Player p) noexcept {
    uint8_t cur_track = (p == Player::US) ? state.us_space_track : state.ussr_space_track;
    uint8_t opp_track = (p == Player::US) ? state.ussr_space_track : state.us_space_track;
    if (cur_track < opp_track) {
        // The VP is the final box's alone: the box skipped over pays nothing.
        SpaceRace::advance_to(state, p, static_cast<uint8_t>(std::min(8, cur_track + 2)));
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

    // Draw 5 cards. The deck running out is handled as at the deal (StateMachine::
    // deal_cards_to_hands): the discard pile is reshuffled at once, so a deck holding fewer
    // than five is drawn to the end and the rest come from the new deck. The cards already
    // drawn are at PEEKED_TEMP and stay out of it. This used to draw only what the deck held:
    // at turn 7 AR2 of ts-replayer game 207 the deck held two cards, the log prints *RESHUFFLE*
    // inside the event, and the engine drew the two and left the reshuffle to the deal.
    //
    // At the deal an empty deck makes both hands public. Here only the USSR's is: every card
    // the US cannot see is in the USSR hand, but the USSR cannot tell the US hand from the
    // cards being looked at.
    uint8_t draw_pool[111];
    uint8_t draw_cnt = 0;
    auto collect_deck = [&]() {
        draw_cnt = 0;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (state.card_locations[i] == CardLocation::DRAW_DECK) draw_pool[draw_cnt++] = i;
        }
    };
    auto run_out = [&]() {
        reveal_hand(state, Player::USSR);
        StateMachine::reshuffle_discard_into_draw(state);
        collect_deck();
    };
    collect_deck();
    if (draw_cnt == 0) run_out();
    if (draw_cnt == 0) return true;

    uint8_t sample_count = 0;
    while (sample_count < 5 && draw_cnt > 0) {
        uint32_t chosen_idx = Prng::random_index(state.rng_state, draw_cnt);
        uint8_t drawn_card = draw_pool[chosen_idx];
        draw_pool[chosen_idx] = draw_pool[--draw_cnt];
        state.card_locations[drawn_card] = CardLocation::PEEKED_TEMP;
        ++sample_count;
        // That was the deck's last card. The pool is rebuilt from the new deck only now, so a
        // deck of more than five is drawn exactly as before.
        if (draw_cnt == 0) run_out();
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
