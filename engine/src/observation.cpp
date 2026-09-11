#include "ts/observation.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"
#include <cstring>
#include <algorithm>

namespace ts {

namespace {

int16_t compute_net_realign_mod(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (country_id >= 84 || p == Player::NONE) return 0;
    const auto& c_info = MapData::get_country(country_id);
    Player opp = (p == Player::US) ? Player::USSR : Player::US;

    int16_t my_mod = 0;
    int16_t opp_mod = 0;

    for (uint8_t n = 0; n < c_info.num_neighbors; ++n) {
        uint8_t n_id = c_info.neighbors[n];
        if (Scoring::is_controlled_by(state, n_id, p)) my_mod++;
        if (Scoring::is_controlled_by(state, n_id, opp)) opp_mod++;
    }

    uint8_t my_inf = state.countries[country_id].get_influence(p);
    uint8_t opp_inf = state.countries[country_id].get_influence(opp);
    if (my_inf > opp_inf) my_mod++;
    else if (opp_inf > my_inf) opp_mod++;

    if (c_info.superpower_adjacent == p) my_mod++;
    if (c_info.superpower_adjacent == opp) opp_mod++;

    if (p == Player::US && state.has_flag(effect_bits::IRAN_CONTRA_ACTIVE)) my_mod -= 1;
    if (opp == Player::US && state.has_flag(effect_bits::IRAN_CONTRA_ACTIVE)) opp_mod -= 1;

    return my_mod - opp_mod;
}

bool is_coup_nuclear_hazard(const GameState& state, Player p, uint8_t country_id) noexcept {
    if (state.defcon > 2 || country_id >= 84) return false;
    const auto& c_info = MapData::get_country(country_id);
    if (!c_info.battleground) return false;
    
    // In TS, couping a battleground reduces DEFCON by 1.
    // If DEFCON is 2 and country is legal to coup, couping causes DEFCON 1 Nuclear Loss!
    return Operations::can_coup(state, p, country_id);
}

} // anonymous namespace

void Observation::extract(const GameState& state, Player perspective,
                          ObservationBufferV23* out_buf) noexcept {
    if (!out_buf) return;
    std::memset(out_buf, 0, sizeof(ObservationBufferV23));

    Player my_player = perspective;
    if (my_player == Player::NONE) {
        my_player = (state.ctx().decision_player != Player::NONE)
            ? state.ctx().decision_player : state.phasing_player;
        if (my_player == Player::NONE) my_player = Player::US;
    }
    const Player opp_player = (my_player == Player::US) ? Player::USSR : Player::US;

    // 1. Board features (84 * 26) -- canonical (myself vs opponent).
    //
    // can_my_realign and can_opp_realign used to sit between "can coup" and the node count.
    // can_realign is exactly can_coup_or_realign, and can_coup is that plus "The Reformer blocks
    // USSR coups in Europe", so the pair said almost nothing the coup pair does not -- 168 floats
    // for one late-war card. They are not computed here at all, which is also why this loop does
    // not call Operations::can_realign.
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& c_info = MapData::get_country(i);
        const size_t offset = i * V23_BOARD_FEATURES;

        const float my_inf = static_cast<float>(my_player == Player::US ? state.countries[i].us_influence : state.countries[i].ussr_influence);
        const float opp_inf = static_cast<float>(my_player == Player::US ? state.countries[i].ussr_influence : state.countries[i].us_influence);

        out_buf->board_features[offset + 0] = my_inf / 10.0f;
        out_buf->board_features[offset + 1] = opp_inf / 10.0f;

        // Net realignment modifier: the dice advantage, normalised by 5.
        const int16_t net_realign = compute_net_realign_mod(state, my_player, i);
        out_buf->board_features[offset + 2] = std::clamp(static_cast<float>(net_realign) / 5.0f, -1.0f, 1.0f);

        out_buf->board_features[offset + 3] = static_cast<float>(c_info.stability) / 5.0f;
        out_buf->board_features[offset + 4] = c_info.battleground ? 1.0f : 0.0f;

        const Player ctrl = Scoring::get_country_control(state, i);
        out_buf->board_features[offset + 5] = (ctrl == my_player) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 6] = (ctrl == opp_player) ? 1.0f : 0.0f;

        // 1.0 if couping here would take DEFCON to 1 and lose the game outright.
        out_buf->board_features[offset + 7] = is_coup_nuclear_hazard(state, my_player, i) ? 1.0f : 0.0f;

        out_buf->board_features[offset + 8] = (c_info.superpower_adjacent == my_player) ? 1.0f : 0.0f;
        out_buf->board_features[offset + 9] = (c_info.superpower_adjacent == opp_player) ? 1.0f : 0.0f;

        const size_t r_idx = static_cast<size_t>(c_info.region);
        if (r_idx < 6) {
            out_buf->board_features[offset + 10 + r_idx] = 1.0f;
        }

        out_buf->board_features[offset + 16] = c_info.in_western_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 17] = c_info.in_eastern_europe ? 1.0f : 0.0f;
        out_buf->board_features[offset + 18] = c_info.in_southeast_asia ? 1.0f : 0.0f;

        bool can_my_place = false;
        bool can_opp_place = false;
        if (state.current_phase == Phase::SETUP) {
            auto check_setup_place = [&](Player pl) -> bool {
                if (pl == Player::USSR) return c_info.in_eastern_europe;
                if (state.ctx().pending_ops_value == 0) return c_info.in_western_europe;
                return state.countries[i].us_influence > 0;
            };
            can_my_place = check_setup_place(my_player);
            can_opp_place = check_setup_place(opp_player);
        } else {
            can_my_place = Operations::can_place_influence(state, my_player, i);
            can_opp_place = Operations::can_place_influence(state, opp_player, i);
        }
        out_buf->board_features[offset + 19] = can_my_place ? 1.0f : 0.0f;
        out_buf->board_features[offset + 20] = can_opp_place ? 1.0f : 0.0f;

        const bool can_my_coup = (state.current_phase != Phase::SETUP) && Operations::can_coup(state, my_player, i);
        const bool can_opp_coup = (state.current_phase != Phase::SETUP) && Operations::can_coup(state, opp_player, i);
        out_buf->board_features[offset + 21] = can_my_coup ? 1.0f : 0.0f;
        out_buf->board_features[offset + 22] = can_opp_coup ? 1.0f : 0.0f;

        out_buf->board_features[offset + 23] = static_cast<float>(state.ctx().node_count(i)) / 5.0f;

        // Influence deficit to my control: how many Ops it would take to reach control.
        const int my_def_stab = std::max(0, static_cast<int>(c_info.stability) - static_cast<int>(my_inf));
        const int my_def_margin = std::max(0, static_cast<int>(opp_inf) + static_cast<int>(c_info.stability) - static_cast<int>(my_inf));
        const int my_deficit = std::max(my_def_stab, my_def_margin);
        out_buf->board_features[offset + 24] = std::min(static_cast<float>(my_deficit) / 5.0f, 2.0f);

        // Influence deficit to opponent control.
        const int opp_def_stab = std::max(0, static_cast<int>(c_info.stability) - static_cast<int>(opp_inf));
        const int opp_def_margin = std::max(0, static_cast<int>(my_inf) + static_cast<int>(c_info.stability) - static_cast<int>(opp_inf));
        const int opp_deficit = std::max(opp_def_stab, opp_def_margin);
        out_buf->board_features[offset + 25] = std::min(static_cast<float>(opp_deficit) / 5.0f, 2.0f);
    }

    // 2. Card features (110 * 14) -- canonical (myself vs opponent).
    //
    // The counts below are keyed off the card's location alone, which is why the China Card
    // override further down cannot disturb them: it is fixed at ONGOING_EVENT and so counts
    // toward neither hand.
    uint8_t draw_pile_cnt = 0;
    uint8_t discard_pile_cnt = 0;
    uint8_t my_hand_cnt = 0;
    uint8_t opp_hand_cnt = 0;

    const auto& ctx = state.ctx();
    for (uint8_t i = 1; i <= 110; ++i) {
        const auto& c_info = CardData::get_card(i);
        const size_t offset = static_cast<size_t>(i - 1) * card_slots::V23_FEATURES;
        const CardLocation loc = state.card_locations[i];

        // Defaults to DECK_OR_HIDDEN, which is where anything the chain below does not claim
        // ends up -- in practice HEADLINE_COMMITTED, a card face down in the headline, which is
        // then given its owner's hand slot further down.
        size_t slot = card_slots::DECK_OR_HIDDEN;
        if (loc == CardLocation::UNAVAILABLE) {
            slot = card_slots::NOT_IN_GAME;
        } else if (loc == CardLocation::DRAW_DECK) {
            slot = card_slots::DECK_OR_HIDDEN;
            draw_pile_cnt++;
        } else if (in_hand_of(loc, my_player)) {
            slot = card_slots::MY_HAND;
            my_hand_cnt++;
        } else if (in_hand_of(loc, opp_player)) {
            // A card the opponent holds that I have seen is public knowledge and gets its own
            // slot; one I have not seen goes back in with the draw deck, because from here those
            // two are the same thing. Anything else would leak the hand.
            slot = known_to_opponent(loc) ? card_slots::KNOWN_OPPONENT_HAND
                                          : card_slots::DECK_OR_HIDDEN;
            opp_hand_cnt++;
        } else if (loc == CardLocation::DISCARD_PILE) {
            slot = card_slots::DISCARD;
            discard_pile_cnt++;
        } else if (loc == CardLocation::REMOVED_FROM_GAME) {
            slot = card_slots::REMOVED;
        } else if (loc == CardLocation::ONGOING_EVENT) {
            slot = card_slots::ONGOING;
        } else if (loc == CardLocation::PEEKED_TEMP) {
            slot = card_slots::PEEKED;
        }

        // The China Card is not in card_locations and must not be. It is fixed at ONGOING_EVENT
        // for the whole game because the cards that scan a hand -- Grain Sales To Soviets, Five
        // Year Plan, Missile Envy, The Cambridge Five, Terrorism -- test membership through
        // card_locations, and a hand variant there would let it be stolen, discarded or forced,
        // which the rules forbid. So the observation reads china_card_holder directly, which is
        // public knowledge: both players always know who holds it.
        //
        // Without this the card block showed it as an ongoing event in every position, to holder
        // and opponent alike, and the card branch never saw the one card that is always safe to
        // play -- 4 Ops, no event of its own, so it can never fire an opponent event and can never
        // move DEFCON. Whether it is playable *this* turn stays in global_features[11]; being face
        // down does not take it out of the hand.
        if (i == card_ids::THE_CHINA_CARD) {
            slot = (state.china_card_holder == my_player) ? card_slots::MY_HAND
                 : (state.china_card_holder == opp_player) ? card_slots::KNOWN_OPPONENT_HAND
                 : card_slots::DECK_OR_HIDDEN;
        }

        out_buf->card_features[offset + slot] = 1.0f;

        const size_t base = card_slots::PROPERTY_BASE;
        out_buf->card_features[offset + base + 0] = static_cast<float>(c_info.ops) / 4.0f;
        const float rel_side = (c_info.side == my_player) ? 1.0f
                             : ((c_info.side == opp_player) ? -1.0f : 0.0f);
        out_buf->card_features[offset + base + 1] = rel_side;
        out_buf->card_features[offset + base + 2] = static_cast<float>(c_info.era) / 2.0f;
        out_buf->card_features[offset + base + 3] = c_info.one_time ? 1.0f : 0.0f;
        out_buf->card_features[offset + base + 4] = c_info.is_scoring ? 1.0f : 0.0f;

        // The card this decision belongs to. resolving_card is the card whose event is
        // executing; pending_op_card is the one whose Ops are being spent. They are usually the
        // same card and are both marked, because "which card am I in the middle of" is the
        // question this feature answers. Without it the network was asked to place a point with
        // no indication of what it was spending.
        //
        // Graded across the whole chain, not a flag on the top frame. Events nest -- Missile
        // Envy takes a card whose Event fires, and that Event can make its opponent discard a
        // card whose Event fires in turn -- and a model mid-chain could previously see only the
        // innermost card. Walking ctx_stack costs no extra floats, because the slot is already
        // one per card.
        float active = 0.0f;
        for (size_t d = 0; d <= state.ctx_stack_depth && d < state.ctx_stack.size(); ++d) {
            const DecisionContext& frame = state.ctx_stack[d];
            if (frame.resolving_card != i && frame.pending_op_card != i) continue;
            active = (d == state.ctx_stack_depth) ? card_slots::ACTIVE_NOW
                                                  : card_slots::ACTIVE_SUSPENDED;
            if (d == state.ctx_stack_depth) break;
        }

        // The card committed to resolve after this one. Both headlines are revealed together and
        // then resolved in Ops order, so while the first is resolving the second is public and
        // known to be next -- which the model could see the owner of (HEADLINE_SECOND_MINE) but
        // not the identity of.
        if (active == 0.0f && state.headline_stage == 1 &&
            state.headline_second_card == i) {
            active = card_slots::ACTIVE_NEXT;
        }

        // A card committed to the headline has no branch in the location chain above, so it
        // arrives here as DECK_OR_HIDDEN -- indistinguishable from a card still in the deck.
        // That hides a player's own headline from them, and hides the opponent's once both are
        // revealed, which is public information. Both go to their owner's hand slot: the card
        // has technically left the hand, but whose it is and that it is in play is what the slot
        // is read for, and a dedicated slot would cost 110 floats to say the same thing.
        if (loc == CardLocation::HEADLINE_COMMITTED) {
            const Player owner = (state.headline_us_card == i) ? Player::US
                               : (state.headline_ussr_card == i) ? Player::USSR
                               : Player::NONE;
            if (owner != Player::NONE) {
                float* row = &out_buf->card_features[offset];
                for (size_t sl = 0; sl < 8; ++sl) row[sl] = 0.0f;
                row[(owner == my_player) ? card_slots::MY_HAND
                                         : card_slots::KNOWN_OPPONENT_HAND] = 1.0f;
                // Committed to the headline is in play. Only raise it -- a card already
                // resolving must not be demoted to "next".
                if (active < card_slots::ACTIVE_NOW) {
                    active = (state.headline_stage == 1 && state.headline_second_card == i)
                        ? card_slots::ACTIVE_NEXT : card_slots::ACTIVE_NOW;
                }
            }
        }

        out_buf->card_features[offset + card_slots::ACTIVE_CARD] = active;
    }

    // 3. Global features (72 board-and-track, then the 28-float decision context).
    const float my_vp = (my_player == Player::US) ? static_cast<float>(state.victory_points) : -static_cast<float>(state.victory_points);
    out_buf->global_features[0] = my_vp / 20.0f; // +1.0 = I am at +20 VP, -1.0 = I am at -20 VP
    out_buf->global_features[1] = static_cast<float>(state.defcon) / 5.0f;

    const float my_mil_ops = static_cast<float>(my_player == Player::US ? state.us_mil_ops : state.ussr_mil_ops);
    const float opp_mil_ops = static_cast<float>(my_player == Player::US ? state.ussr_mil_ops : state.us_mil_ops);
    out_buf->global_features[2] = my_mil_ops / 5.0f;
    out_buf->global_features[3] = opp_mil_ops / 5.0f;

    const float my_space = static_cast<float>(my_player == Player::US ? state.us_space_track : state.ussr_space_track);
    const float opp_space = static_cast<float>(my_player == Player::US ? state.ussr_space_track : state.us_space_track);
    out_buf->global_features[4] = my_space / 8.0f;
    out_buf->global_features[5] = opp_space / 8.0f;

    out_buf->global_features[6] = static_cast<float>(state.turn) / 10.0f;
    out_buf->global_features[7] = static_cast<float>(state.action_round) / 8.0f;
    out_buf->global_features[8] = (state.phasing_player == my_player) ? 1.0f : -1.0f;
    out_buf->global_features[9] = static_cast<float>(state.current_phase) / 6.0f;
    out_buf->global_features[10] = (state.china_card_holder == my_player) ? 1.0f : -1.0f;
    out_buf->global_features[11] = state.china_card_playable ? 1.0f : 0.0f;

    // Bits 0..44 only. The span is global_features[12..56]: 57 and 58 are defcon_dropped_to_2
    // and ctx_stack_depth, assigned just below, so a loop to 47 would write bits 45 and 46
    // (SPACE_USSR_ATTEMPT_1/2) and have them overwritten on the next two lines. Their information
    // reaches the model through global_features[59]/[60] instead. Anything added at bit >= 45
    // needs its own feature; it will not appear here.
    for (size_t b = 0; b < 45; ++b) {
        out_buf->global_features[12 + b] = ((state.persistent_effects & (1ULL << b)) != 0) ? 1.0f : 0.0f;
    }

    out_buf->global_features[57] = state.defcon_dropped_to_2 ? 1.0f : 0.0f;
    out_buf->global_features[58] = static_cast<float>(state.ctx_stack_depth) / 3.0f;
    out_buf->global_features[59] = static_cast<float>(state.get_space_turns_used(my_player)) / 2.0f;
    out_buf->global_features[60] = static_cast<float>(state.get_space_turns_used(opp_player)) / 2.0f;

    // Side identity.
    out_buf->global_features[61] = (my_player == Player::US) ? 1.0f : 0.0f; // I_AM_US

    // Deck tracking.
    out_buf->global_features[62] = static_cast<float>(draw_pile_cnt) / 100.0f;
    out_buf->global_features[63] = static_cast<float>(discard_pile_cnt) / 100.0f;

    // Real-time regional scoring VP differentials for the 6 regions.
    for (size_t r = 0; r < 6; ++r) {
        auto summary = Scoring::evaluate_region(state, static_cast<Region>(r));
        int16_t net = summary.net_delta;
        // Europe Control ends the game, but its control_vp is 0 against domination_vp 7, so
        // net_delta ranks a won position *below* a dominated one -- about 6 against 12 once
        // battlegrounds and adjacency are added. Report the win as a win; the engine's own
        // shaping potential already special-cases this the same way.
        if (static_cast<Region>(r) == Region::EUROPE) {
            if (summary.us_status == RegionalStatus::CONTROL)         net = 20;
            else if (summary.ussr_status == RegionalStatus::CONTROL)  net = -20;
        }
        const float my_region_vp = (my_player == Player::US) ? static_cast<float>(net) : -static_cast<float>(net);
        out_buf->global_features[64 + r] = std::clamp(my_region_vp / 20.0f, -1.0f, 1.0f);
    }

    out_buf->global_features[70] = static_cast<float>(opp_hand_cnt) / 10.0f;
    out_buf->global_features[71] = static_cast<float>(my_hand_cnt) / 10.0f;

    // 4. Decision context. Every field here is a pure function of the state, so an observation
    // remains reproducible from a GameState alone -- which is what lets a run branch from a
    // snapshot, and what any decision-time search will need.
    const size_t dt = static_cast<size_t>(ctx.decision_type);
    if (dt < 8) out_buf->global_features[ctx_slots::DECISION_TYPE + dt] = 1.0f;

    // op_mode is only meaningful while an Op is being spent; SELECT_OP_MODE is where it is
    // chosen, and before that the field holds whatever the last Op left. Gate on there being a
    // pending Op so it reads as "no mode" rather than as a stale one.
    if (ctx.pending_op_card != 0 || ctx.decision_type == DecisionType::POINT_NODE) {
        const size_t om = static_cast<size_t>(ctx.op_mode);
        if (om < 3) out_buf->global_features[ctx_slots::OP_MODE + om] = 1.0f;
    }

    out_buf->global_features[ctx_slots::REMAINING_STEPS]    = static_cast<float>(ctx.remaining_steps) / 7.0f;
    out_buf->global_features[ctx_slots::PENDING_OPS_VALUE]  = static_cast<float>(ctx.pending_ops_value) / 5.0f;
    out_buf->global_features[ctx_slots::MAX_PER_COUNTRY]    = static_cast<float>(ctx.max_per_country) / 5.0f;
    out_buf->global_features[ctx_slots::ALLOW_EARLY_STOP]   = ctx.allow_early_stop ? 1.0f : 0.0f;
    // 255 means "no branch chosen", which is neither of these.
    out_buf->global_features[ctx_slots::TIMING_OPS_FIRST]   = (ctx.timing_branch == 0) ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::TIMING_EVENT_FIRST] = (ctx.timing_branch == 1) ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::EVENT_GRANTED_OPS]  = ctx.event_granted_ops ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::SUPPRESS_OP_EVENT]  = ctx.suppress_op_card_event ? 1.0f : 0.0f;

    // Headline stage and resolution order. Which card resolves first is a mechanic (Space box 4
    // lets a player see the opponent's headline before choosing), and none of this reached the
    // model before.
    out_buf->global_features[ctx_slots::HEADLINE_STAGE] =
        static_cast<float>(state.headline_stage) / 3.0f;
    out_buf->global_features[ctx_slots::HEADLINE_FIRST_MINE] =
        (state.headline_first_owner == my_player) ? 1.0f : 0.0f;
    out_buf->global_features[ctx_slots::HEADLINE_SECOND_MINE] =
        (state.headline_second_owner == my_player) ? 1.0f : 0.0f;

    // Chernobyl's forbidden region, one-hot. All zero when it is not in play.
    if (state.has_flag(effect_bits::CHERNOBYL_ACTIVE)) {
        const size_t ch = static_cast<size_t>(
            (state.persistent_effects & effect_bits::CHERNOBYL_REGION_MASK)
            >> effect_bits::CHERNOBYL_REGION_SHIFT);
        if (ch < 6) out_buf->global_features[ctx_slots::CHERNOBYL_REGION + ch] = 1.0f;
    }
}

void extract_observation(const GameState& state, Player perspective,
                         ObservationBufferV23* out_buf) noexcept {
    Observation::extract(state, perspective, out_buf);
}

} // namespace ts
