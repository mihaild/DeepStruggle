#include "ts/state_machine.hpp"
#include "ts/invariant.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/scoring.hpp"
#include "ts/space_race.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/action_mask.hpp"
#include "ts/canonical.hpp"
#include <algorithm>
#include <cstring>
#include "ts/victory_points.hpp"

namespace ts {

namespace {

// Does the pending POINT_NODE have any legal country to point at? Asked of generate_mask
// rather than the flat mask so that checking cannot itself re-report the anomaly.
bool has_legal_point_target(const GameState& state) noexcept {
    uint8_t buf[FLAT_ACTION_SPACE_SIZE] = {0};
    size_t n = 0;
    ActionMask::generate_mask(state, buf, &n);
    const size_t limit = (n < 84) ? n : 84;
    for (size_t i = 0; i < limit; ++i) {
        if (buf[i]) return true;
    }
    return false;
}

} // namespace


void StateMachine::add_era_cards_to_deck(GameState& state, WarEra era) noexcept {
    for (uint8_t i = 1; i <= 110; ++i) {
        if (i == card_ids::THE_CHINA_CARD) continue; // The China Card is not in deck
        if (CardData::get_card(i).era == era) {
            state.card_locations[i] = CardLocation::DRAW_DECK;
        }
    }
}

void StateMachine::reshuffle_discard_into_draw(GameState& state) noexcept {
    for (uint8_t i = 1; i <= 110; ++i) {
        if (i == card_ids::THE_CHINA_CARD) continue;
        if (state.card_locations[i] == CardLocation::DISCARD_PILE) {
            state.card_locations[i] = CardLocation::DRAW_DECK;
        }
    }
}

// The next card a replay source named for this player's deal, or 0 when it named none.
//
// Consuming is unconditional: a named card that is not in the draw deck is still taken off the
// queue, because leaving it there would hand it to the next draw and silently shift every card
// after it. The caller reports the contradiction and falls back to the PRNG for that one draw.
static uint8_t take_forced_deal(GameState& state, Player p) noexcept {
    uint8_t* pos = (p == Player::US) ? &state.forced_deal_us_pos : &state.forced_deal_ussr_pos;
    const uint8_t count = (p == Player::US) ? state.forced_deal_us_count
                                           : state.forced_deal_ussr_count;
    const uint8_t* q = (p == Player::US) ? state.forced_deal_us : state.forced_deal_ussr;
    if (*pos >= count) return 0;
    return q[(*pos)++];
}

void StateMachine::deal_cards_to_hands(GameState& state) noexcept {
    uint8_t target_hand = (state.turn <= 3) ? 8 : 9;

    // Draws one card into p's hand; false only when the deck is empty even after a reshuffle.
    auto draw_one = [&](Player p) -> bool {
        // Count cards in draw deck
        uint8_t draw_cards[111];
        uint8_t draw_count = 0;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (state.card_locations[i] == CardLocation::DRAW_DECK) {
                draw_cards[draw_count++] = i;
            }
        }

        if (draw_count == 0) {
            // The deck is empty, so every card is now in a hand, the discard, removed, in
            // play, or never in the game -- and all of those but the hands are public. Each
            // player can therefore name the other's hand as the complement of what they can
            // see. The shuffle that follows is the consequence, not the cause. It is also the
            // only reshuffle there is: the era transitions at turns 4 and 8 add cards to the
            // deck and leave the discard alone (rule 4.4).
            reveal_both_hands(state);
            reshuffle_discard_into_draw(state);
            for (uint8_t i = 1; i <= 110; ++i) {
                if (state.card_locations[i] == CardLocation::DRAW_DECK) {
                    draw_cards[draw_count++] = i;
                }
            }
            if (draw_count == 0) return false; // Entire deck exhausted
        }

        // A replay source may name this card. Inert in normal play: the queue is empty
        // and this costs one comparison.
        uint8_t chosen_card = 0;
        if (const uint8_t named = take_forced_deal(state, p)) {
            if (named >= 1 && named <= 110 &&
                state.card_locations[named] == CardLocation::DRAW_DECK) {
                chosen_card = named;
            } else {
                // The recording and the engine disagree about what is in the deck. Never
                // paper over it: a wrong card here changes the game from this point on and
                // every later entry would be compared against a position that never existed.
                report_anomaly("forced deal named a card that is not in the draw deck", state);
            }
        }
        if (chosen_card == 0) {
            const uint32_t chosen_idx = Prng::random_index(state.rng_state, draw_count);
            chosen_card = draw_cards[chosen_idx];
        }
        state.card_locations[chosen_card] = hand_of(p);
        if (draw_count == 1) {
            // That was the last card in the deck; same deduction as above, and it covers
            // the card just drawn too, since the opponent can see the deck is now empty.
            reveal_both_hands(state);
            reshuffle_discard_into_draw(state);
        }
        return true;
    };

    // The players draw alternately, one card at a time, the USSR first, and a full hand sits
    // out (ENG-5). The order only shows when the deck runs out mid-deal, as it does on turn 3:
    // filling the USSR hand first handed the USSR nearly every card left in the old deck.
    constexpr Player order[2] = {Player::USSR, Player::US};
    uint8_t count[2] = {0, 0};
    for (uint8_t i = 1; i <= 110; ++i) {
        for (int k = 0; k < 2; ++k) {
            if (in_hand_of(state.card_locations[i], order[k])) count[k]++;
        }
    }
    bool exhausted = false;
    while (!exhausted && (count[0] < target_hand || count[1] < target_hand)) {
        for (int k = 0; k < 2 && !exhausted; ++k) {
            if (count[k] >= target_hand) continue;
            if (draw_one(order[k])) count[k]++;
            else exhausted = true;
        }
    }

    // One deal, one queue. A source that wants to drive the next deal refills before it; a
    // leftover here would be applied to a deal it was never recorded for, which is exactly the
    // silent divergence this affordance exists to prevent.
    state.forced_deal_us_count = 0;
    state.forced_deal_ussr_count = 0;
    state.forced_deal_us_pos = 0;
    state.forced_deal_ussr_pos = 0;
}

void StateMachine::init_new_game(GameState& state, uint64_t seed) noexcept {
    state = GameState{};
    state.rng_state = (seed != 0) ? seed : 0x123456789ABCDEF0ULL;

    // 1. Tracks initialization
    state.victory_points = 0;
    state.defcon = 5;
    state.us_mil_ops = 0;
    state.ussr_mil_ops = 0;
    state.us_space_track = 0;
    state.ussr_space_track = 0;
    state.turn = 1;
    state.action_round = 0;

    // 2. Fixed Setup Influence Placements
    // USSR: Syria(22)=1, Iraq(24)=1, North Korea(43)=3, East Germany(14)=3, Finland(5)=1
    state.countries[countries::SYRIA].ussr_influence = 1;
    state.countries[countries::IRAQ].ussr_influence = 1;
    state.countries[countries::NORTH_KOREA].ussr_influence = 3;
    state.countries[countries::EAST_GERMANY].ussr_influence = 3;
    state.countries[countries::FINLAND].ussr_influence = 1;

    // US: Iran(25)=1, Israel(23)=1, Japan(45)=1, Australia(41)=4, Philippines(40)=1, South Korea(44)=1, Panama(70)=1, South Africa(63)=1, United Kingdom(1)=5
    state.countries[countries::IRAN].us_influence = 1;
    state.countries[countries::ISRAEL].us_influence = 1;
    state.countries[countries::JAPAN].us_influence = 1;
    state.countries[countries::AUSTRALIA].us_influence = 4;
    state.countries[countries::PHILIPPINES].us_influence = 1;
    state.countries[countries::SOUTH_KOREA].us_influence = 1;
    state.countries[countries::PANAMA].us_influence = 1;
    state.countries[countries::SOUTH_AFRICA].us_influence = 1;
    state.countries[countries::UNITED_KINGDOM].us_influence = 5;
    state.countries[countries::CANADA].us_influence = 2;

    // 3. China Card to USSR
    state.china_card_holder = Player::USSR;
    state.china_card_playable = 1;
    state.forced_card_player = Player::NONE;
    state.forced_card_id = 0;
    state.card_locations[card_ids::THE_CHINA_CARD] = CardLocation::ONGOING_EVENT;

    // 4. Early War Cards & Deal Hands
    add_era_cards_to_deck(state, WarEra::EARLY);
    deal_cards_to_hands(state);

    // 5. Setup Phase
    state.current_phase = Phase::SETUP;
    state.phasing_player = Player::USSR;
    state.ctx().decision_player = Player::USSR;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 6; // USSR places 6 in Eastern Europe
}

void StateMachine::start_turn(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;

    // Reset decision context cleanly for the new turn
    state.ctx() = DecisionContext{};

    // Phase A: Improve DEFCON
    state.defcon = static_cast<uint8_t>(std::min(5, static_cast<int>(state.defcon) + 1));

    // Phase B: Deal Cards
    deal_cards_to_hands(state);

    // Phase C: Headline Phase
    state.current_phase = Phase::HEADLINE;
    state.action_round = 0;
    state.headline_us_card = 0;
    state.headline_ussr_card = 0;
    state.headline_first_card = 0;
    state.headline_second_card = 0;
    state.headline_stage = 0;
    state.headline_first_owner = Player::NONE;
    state.headline_second_owner = Player::NONE;

    // Space 4 Priority check
    bool us_space4 = SpaceRace::has_man_in_earth_orbit(state, Player::US);
    bool ussr_space4 = SpaceRace::has_man_in_earth_orbit(state, Player::USSR);

    if (us_space4 && !ussr_space4) {
        state.ctx().decision_player = Player::USSR; // USSR reveals first
    } else if (ussr_space4 && !us_space4) {
        state.ctx().decision_player = Player::US; // US reveals first
    } else {
        state.ctx().decision_player = Player::US;
    }

    state.ctx().decision_type = DecisionType::SELECT_CARD;
}

void StateMachine::advance_headline_step(GameState& state) noexcept {
    // Check if we need to resolve headline 2 or enter AR 1
    if (state.headline_stage == 1) {
        // Move to Stage 2: Second headline
        state.headline_stage = 2;
        uint8_t h2_card = state.headline_second_card;
        Player h2_owner = state.headline_second_owner;

        if (h2_card != 0 && state.current_phase != Phase::GAME_OVER) {
            bool ussr_cancelled = (state.headline_us_card == card_ids::DEFECTORS && h2_owner == Player::USSR);
            if (ussr_cancelled) {
                state.card_locations[h2_card] = CardLocation::DISCARD_PILE;
                advance_headline_step(state);
                return;
            }
            const auto& c_info = CardData::get_card(h2_card);
            Player exec_player = (c_info.side == get_opponent(h2_owner)) ? get_opponent(h2_owner) : h2_owner;
            // A fresh context: the first headline is finished, and nothing it left behind is
            // the second one's. What lasts belongs to the state -- the effect bits, the card
            // Missile Envy forced on its recipient -- not to the decision frame. The countries
            // a card has already placed in are kept in the frame's visited bitmap, which is how
            // "no more than one per country" is enforced, and both headline cards can want it:
            // at turn 7's headline of ts-replayer game 92 the US's Colonial Rear Guards places
            // in Zaire, Angola, Zimbabwe and Nigeria, and the USSR's Decolonization is offered
            // none of them, so three of its four Influence had nowhere to go.
            state.ctx() = DecisionContext{};
            state.phasing_player = h2_owner;
            state.ctx().decision_player = exec_player;
            state.ctx().resolving_card = h2_card;

            const bool h2_fired = CardHandlers::event_has_effect(state, h2_card, exec_player);
            bool done = CardHandlers::trigger_event(state, h2_card, exec_player);
            CardHandlers::relocate_played_card(state, h2_card, h2_fired, CardHandlers::Handover::Respect);
            if (done) {
                advance_headline_step(state);
            }
            return;
        }
    }

    // Both headlines resolved -> Start Action Round 1
    state.headline_stage = 3;
    // NORAD asks whether DEFCON reached 2 during an *action round*, and a headline that drops
    // it to 2 sets the same flag. Clearing it as the headline ends keeps that drop from
    // claiming the round that follows: at turn 5 of ts-replayer game 219 DEFCON falls to 2 in
    // the headline, and the US was handed a free Influence placement at the end of AR1, where
    // the USSR had done nothing but discard a card to escape Bear Trap.
    state.defcon_dropped_to_2 = 0;
    if (state.current_phase != Phase::GAME_OVER) {
        state.current_phase = Phase::ACTION_ROUND;
        state.action_round = 1;
        state.phasing_player = Player::USSR;
        state.ctx() = DecisionContext{};
        state.ctx().decision_player = Player::USSR;
        state.ctx().decision_type = DecisionType::SELECT_CARD;
        offer_cuban_missile_payoff(state);
    }
}

// The crisis can be paid off at any time; it is offered here, at the head of the payer's own
// action round, and inside a coup they make. At turn 4 AR2 of ts-replayer game 264 the USSR
// pays 2 Influence out of Cuba, then plays "We Will Bury You" for its 4 Operations and puts
// two of them straight back into Cuba and two into Saudi Arabia -- no coup anywhere near it.
//
// Declining is always allowed, which is what most players do, so this costs a pass and not a
// decision wherever the crisis is not in play at all.
void StateMachine::offer_cuban_missile_payoff(GameState& state) noexcept {
    if (state.current_phase != Phase::ACTION_ROUND) return;
    Player p = state.ctx().decision_player;
    if (!can_pay_off_cuban_missile_crisis(state, p)) return;
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = 1;
    state.ctx().allow_early_stop = 1;    // paying is a choice, and usually declined
    state.ctx().resolving_card = card_ids::CUBAN_MISSILE_CRISIS;
}

namespace {

// Did the event on a card spent for Operations actually occur?
//
// Only an opponent's card fires its event when played for Ops, and only when it was not
// suppressed. The phasing player's own card and a neutral card never do, which is what decides
// whether a starred card is removed from the game or discarded.
bool event_occurred_on_ops_play(const GameState& state, uint8_t card, Player p) noexcept {
    if (card == 0) return false;
    if (!CardData::is_opponent_card(card, p)) return false;
    return state.ctx().suppress_op_card_event == 0;
}

// Ops spent on a pushed frame are usually granted by an Event, whose card was placed as soon as
// the Event was triggered. But Grain Sales fired through another card's Event -- Star Wars, Five
// Year Plan, Missile Envy -- plays the card it drew on the frame that was pushed for it, and that is
// a real play: the card is still in the hand it was drawn into, and nothing else will place it. At
// turn 10's headline of stats game b97d05b2 the US headlines Star Wars, takes Grain Sales, draws
// NORAD and places three Influence with it, and NORAD stayed in the US hand, playable again; drawn
// from the USSR, a card's Event never happened either. So "still in a hand" -- the test the frame
// underneath uses -- separates this from granted Ops, at any depth.
//
// Returns true when the frame must stop here: the card's Event has asked a question, whose answer
// unwinds this frame when it finishes, or the game ended.
bool settle_card_played_on_pushed_frame(GameState& state) noexcept {
    const uint8_t card = state.ctx().pending_op_card;
    if (card == 0 || card > 110 || card == card_ids::THE_CHINA_CARD ||
        card == state.headline_first_card || card == state.headline_second_card ||
        !is_in_any_hand(state.card_locations[card])) {
        return false;
    }
    const Player spender = state.ctx().decision_player;
    if (!state.ctx().suppress_op_card_event &&
        state.ctx().timing_branch == static_cast<uint8_t>(TimingBranch::OPS_FIRST) &&
        CardData::is_opponent_card(card, spender)) {
        const Player opp = get_opponent(spender);
        state.ctx().decision_player = opp;
        state.ctx().resolving_card = card;
        state.ctx().timing_branch = 255; // cleared
        state.ctx().allow_early_stop = 0;  // an Event starts from none, as below
        const bool fired = CardHandlers::event_has_effect(state, card, opp);
        const bool done = CardHandlers::trigger_event(state, card, opp);
        CardHandlers::relocate_played_card(state, card, fired, CardHandlers::Handover::Respect);
        if (!done || state.current_phase == Phase::GAME_OVER) return true;
        state.ctx().resolving_card = 0;
        return false;
    }
    CardHandlers::relocate_played_card(state, card, event_occurred_on_ops_play(state, card, spender),
                                       CardHandlers::Handover::Ignore);
    return false;
}

} // namespace

// A play that ended without Ops -- an Event, a Space attempt -- has to leave the frames it was
// played in. Without this, Star Wars firing Grain Sales, whose drawn card was played as its Event,
// went on to the next action round one frame deep.
void StateMachine::finish_play_without_ops(GameState& state) noexcept {
    while (state.ctx_stack_depth > 0) {
        state.pop_context();
        if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
            return;   // the player's own Ops are still owed
        }
        state.ctx().resolving_card = 0;
    }
    if (state.current_phase == Phase::HEADLINE) {
        advance_headline_step(state);
    } else {
        advance_after_action_round(state);
    }
}

void StateMachine::advance_after_ops(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;
    if (state.ctx_stack_depth > 0 && settle_card_played_on_pushed_frame(state)) return;
    if (state.current_phase == Phase::HEADLINE) {
        // An opponent's card played for Operations still owes its Event, in a headline exactly
        // as in an action round. The two orderings are a choice, and OPS_FIRST leaves the
        // Event to be fired here -- which this path used to return without doing, because it
        // unwinds and advances the headline before ever reaching the check below. At turn 4's
        // headline of ts-replayer game 137 the US headlines Grain Sales To Soviets, takes
        // Willy Brandt out of the USSR hand, chooses its Operations first and realigns Cuba
        // twice; Willy Brandt's Event -- 1 VP to the USSR, 1 Influence into West Germany, and
        // the card into play -- simply never happened.
        //
        // Same guards as the action round's: a pushed frame means these Ops were granted by an
        // event rather than bought with the card, and a pushed frame starts zeroed, which is
        // OPS_FIRST. And one the action round does not need -- a headlined card is played as
        // its Event and never for Operations, so Ops that belong to one are Ops its Event gave
        // away and its Event has already happened. Without that, the USSR headlining CIA
        // Created (a US card) fired it a second time when the US finished spending the Op it
        // grants them, at turn 3's headline of ts-replayer game 101 and 8 games besides.
        uint8_t owed_card = state.ctx().pending_op_card;
        Player card_player = state.phasing_player;
        if (state.ctx_stack_depth == 0 && !state.ctx().suppress_op_card_event &&
            owed_card != state.headline_first_card && owed_card != state.headline_second_card &&
            state.ctx().timing_branch == static_cast<uint8_t>(TimingBranch::OPS_FIRST) &&
            CardData::is_opponent_card(owed_card, card_player)) {
            Player opp = get_opponent(card_player);
            state.ctx().decision_player = opp;
            state.ctx().resolving_card = owed_card;
            state.ctx().timing_branch = 255; // cleared
            // The Event runs in the frame the Ops were spent in, so it must not inherit their stop
            // -- see the action-round twin below.
            state.ctx().allow_early_stop = 0;

            const bool owed_fired = CardHandlers::event_has_effect(state, owed_card, opp);
            bool done = CardHandlers::trigger_event(state, owed_card, opp);
            CardHandlers::relocate_played_card(state, owed_card, owed_fired, CardHandlers::Handover::Respect);
            // An event that asks something of its player is not finished, and the headline
            // resumes here once it is.
            if (!done || state.current_phase == Phase::GAME_OVER) return;
            state.ctx().resolving_card = 0;
        }
        // Unwind whatever the headline's own events pushed. A card that grants Ops does not
        // finish when it is triggered, so the frame it was fired in stays open until those Ops
        // are spent -- and this path used to return without popping it. At turn 6's headline
        // of ts-replayer game 240 the USSR headlines Missile Envy, the US hands over ABM
        // Treaty, its event improves DEFCON and grants the USSR 3 Ops, and the USSR coups
        // Argentina; the headline then ended with a frame still pushed, and turn 6 AR1 was
        // refused because the engine was one frame deep in a headline that was over.
        //
        // A frame holding an unanswered SELECT_OP_MODE is Ops still owed inside the headline,
        // and resuming there is the whole point of the stack -- the same reading the action
        // round takes below.
        while (state.ctx_stack_depth > 0) {
            state.pop_context();
            if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
                return;
            }
            state.ctx().resolving_card = 0;
        }

        // Ops spent during a headline still belong to a card, and that card is spent with
        // them. The headline machinery relocates the two headline cards themselves, but a card
        // played *through* one of them is nobody's to clear away and stayed in the hand it was
        // played from: at turn 4's headline of ts-replayer game 14 the US headlines Grain
        // Sales To Soviets, draws Marshall Plan out of the USSR hand and coups Brazil with it,
        // and Marshall Plan was still sitting in the US hand afterwards -- in the running when
        // the USSR's Missile Envy asked for the highest Ops card, and playable a second time.
        uint8_t op_card = state.ctx().pending_op_card;
        if (op_card != 0 && op_card != card_ids::THE_CHINA_CARD &&
            op_card != state.headline_first_card && op_card != state.headline_second_card &&
            is_in_any_hand(state.card_locations[op_card])) {
            // Still in a hand means nothing has placed it yet. On an EVENT_FIRST play the
            // card was placed when its Event fired, by the site that knew whether the Event
            // could occur; re-deciding here on "it is an opponent's card" alone would throw
            // that answer away and remove a card whose Event never happened.
            CardHandlers::relocate_played_card(state, op_card,
                                 event_occurred_on_ops_play(state, op_card,
                                                            state.phasing_player),
                                 CardHandlers::Handover::Respect);
        }
        advance_headline_step(state);
        return;
    }
    // Ops spent inside a pushed frame were granted by an event, and finishing them has to
    // return to the frame underneath rather than end the action round. On an EVENT_FIRST play
    // that frame holds the phasing player's own Ops: CIA Created reveals the USSR hand and
    // hands the US one Op, and the USSR then still owes the card's own Op. Ending the round
    // here dropped it -- at turn 2 AR1 of ts-replayer game 103 the USSR's influence in Poland
    // never went in. The SELECT_OP_MODE test mirrors how the resolving_card path resumes.
    if (state.ctx_stack_depth > 0) {
        bool resumed = false;
        while (state.ctx_stack_depth > 0) {
            state.pop_context();
            if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
                resumed = true;
                break;
            }
            state.ctx().resolving_card = 0;
        }
        if (!resumed) {
            advance_after_action_round(state);
        }
        return;
    }

    uint8_t card = state.ctx().pending_op_card;
    Player p = state.phasing_player;
    uint8_t timing = state.ctx().timing_branch;

    // Only the phasing player's own card play can owe an opponent event here. When the
    // opponent is spending Ops granted BY that event -- Grain Sales handing the US two Ops,
    // say -- a frame is pushed, and the pushed frame starts zeroed. Zero is OPS_FIRST, so
    // without the depth guard this fires again on an EVENT_FIRST play: the opponent
    // resolves the event twice and the phasing player loses the Ops they paid for.
    if (state.ctx_stack_depth == 0 && !state.ctx().suppress_op_card_event &&
        timing == static_cast<uint8_t>(TimingBranch::OPS_FIRST) && CardData::is_opponent_card(card, p)) {
        // Trigger opponent event!
        Player opp = get_opponent(p);
        state.ctx().decision_player = opp;
        state.ctx().resolving_card = card;
        state.ctx().timing_branch = 255; // cleared
        // The Event runs in the frame the Ops were just spent in, and a handler that does not set
        // allow_early_stop itself inherited theirs: after a realignment -- and after an influence
        // play, while that could stop -- Warsaw Pact Formed's mandatory choice offered
        // CONFIRM_DONE and could be declined, and the free Op from CIA Created or "Lone Gunman"
        // showed the stop in the observation. An Event starts from none; the handlers that
        // allow one set it.
        state.ctx().allow_early_stop = 0;

        const bool owed_fired = CardHandlers::event_has_effect(state, card, opp);
        bool done = CardHandlers::trigger_event(state, card, opp);
        CardHandlers::relocate_played_card(state, card, owed_fired, CardHandlers::Handover::Respect);
        if (done) {
            advance_after_action_round(state);
        }
        return;
    }

    // Friendly / Neutral / Already triggered event
    if (card == card_ids::THE_CHINA_CARD) {
        state.china_card_holder = get_opponent(p);
        state.china_card_playable = 0; // Passes to opponent face down
        if (p == Player::US) {
            state.clear_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
        }
    } else if (card != 0 && is_in_any_hand(state.card_locations[card])) {
        // A starred card is removed when its *event* occurs, not when it is spent for
        // Operations (rules.md 280-282; the Space Race path says the same at 252).
        //
        // The in-hand test is what separates the two ways this branch is reached. The phasing
        // player's own or a neutral card is still in their hand, and its Event never fired.
        // An opponent's card played EVENT_FIRST was already placed when the Event resolved,
        // by the site that knew whether the Event could occur -- so it is left alone here
        // rather than re-judged on "it is an opponent's card", which cannot tell an Event that
        // happened from one that was fired and could not occur.
        CardHandlers::relocate_played_card(state, card, event_occurred_on_ops_play(state, card, p),
                             CardHandlers::Handover::Ignore);
    }
    advance_after_action_round(state);
}

// We Will Bury You: "unless UN Intervention is played as an Event on the US's next action
// round, the USSR receives 3 VP". Called on that round, however it goes -- a card played, a
// pass, an empty hand, a trapped discard -- with `un_event` saying whether it is UN
// Intervention's Event that answers it. A no-op when nothing is owed. Returns true when the
// 3 VP end the game.
static bool settle_we_will_bury_you(GameState& state, bool un_event) noexcept {
    if (!state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING)) return false;
    state.clear_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    return !un_event && award_vp(state, -3);
}

// Conditional Ops bonuses are granted up front and withdrawn the moment their region condition
// breaks, after an Op is spent in `cid` (an influence placement or a realignment). Track the
// *current* budget in pending_ops_value and derive what is left from what has been spent; a
// version that rewrote remaining_steps against a recomputed base while leaving
// pending_ops_value at the optimistic figure charged the next withdrawal a second time.
//
// The ladder itself is Operations::budget_if_spent_in, which the influence and realignment
// masks also ask (mask_region_budget), so no target is offered whose cost the withdrawal
// would leave unpaid.
//
// Only engaged when a conditional bonus is actually in play, and only when pending_ops_value
// is being maintained. Event-driven placements set remaining_steps directly and leave
// pending_ops_value at 0; recomputing from a zero budget there would wipe out the remaining Ops.
static void withdraw_conditional_ops_bonus(GameState& state, Player p, uint8_t cid) noexcept {
    const uint8_t budget = state.ctx().pending_ops_value;
    const uint8_t keep = Operations::budget_if_spent_in(state, p, cid);
    if (keep == budget) return;
    const uint8_t spent = static_cast<uint8_t>(
        (budget > state.ctx().remaining_steps) ? (budget - state.ctx().remaining_steps) : 0);
    state.ctx().pending_ops_value = keep;
    state.ctx().remaining_steps = (spent >= keep) ? 0 : static_cast<uint8_t>(keep - spent);
}

void StateMachine::advance_after_action_round(GameState& state) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return;
    // Check NORAD
    if (state.defcon_dropped_to_2 && state.has_flag(effect_bits::NORAD_ACTIVE) &&
        Scoring::is_controlled_by(state, countries::CANADA, Player::US)) {
        // US gets 1 free influence placement.
        state.defcon_dropped_to_2 = 0;
        // P17 6.1: a FRESH context, the way every normal path opens a decision. Setting four
        // fields onto whatever was there left max_per_country, visited_nodes and
        // node_count_bits carrying over from the decision before this one -- so the same NORAD
        // placement behaved differently depending on what had just happened.
        state.ctx() = DecisionContext{};
        state.ctx().decision_player = Player::US;
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 1;
        // "Add 1 US Influence to a single country containing US Influence" -- no "may". The
        // pool cannot run dry either: the event requires US-controlled Canada, which is itself
        // a country containing US Influence, so this needs no may_fizzle entry.
        state.ctx().allow_early_stop = 0;
        state.ctx().resolving_card = card_ids::NORAD;
        return;
    }
    state.defcon_dropped_to_2 = 0;

    // Determine max ARs for this turn
    uint8_t max_ar = (state.turn <= 3) ? 6 : 7;
    bool us_has_ar8 = state.has_flag(effect_bits::NORTH_SEA_OIL_ACTIVE) || SpaceRace::has_space_station_ar8(state, Player::US);
    bool ussr_has_ar8 = SpaceRace::has_space_station_ar8(state, Player::USSR);
    if ((us_has_ar8 || ussr_has_ar8) && state.turn >= 4) {
        max_ar = 8;
    }

    // Alternate phasing player
    if (state.phasing_player == Player::USSR) {
        state.phasing_player = Player::US;
    } else {
        state.phasing_player = Player::USSR;
        state.action_round++;
    }

    if (state.action_round > max_ar) {
        // All Action Rounds completed for this turn -> proceed to turn end
        begin_turn_cleanup(state);
        return;
    }

    // The eighth Action Round belongs only to the player who earned it. max_ar is a
    // property of the turn, so raising it for one side handed the extra round to both:
    // North Sea Oil gave the USSR a free action round, and so did a Space Station only
    // the opponent had reached. Skip whoever is not entitled, and end the turn once the
    // entitled player has taken theirs.
    if (state.action_round == 8) {
        while (true) {
            bool entitled = (state.phasing_player == Player::US) ? us_has_ar8 : ussr_has_ar8;
            if (entitled) break;
            if (state.phasing_player == Player::USSR) {
                state.phasing_player = Player::US;  // not owed to the USSR; offer to the US
            } else {
                begin_turn_cleanup(state);          // both sides done with round eight
                return;
            }
        }
        state.ctx().decision_player = state.phasing_player;
    }

    // Start next Action Round
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = state.phasing_player;

    // Check if player has cards
    uint8_t count = 0;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (in_hand_of(state.card_locations[i], state.phasing_player)) count++;
    }

    if (count == 0 && !(state.china_card_holder == state.phasing_player && state.china_card_playable)) {
        // Player has no cards -> automatically pass to next player. A US round started with an
        // empty hand is still the US's next action round for We Will Bury You, and no UN
        // Intervention can be played in it.
        if (state.phasing_player == Player::US && settle_we_will_bury_you(state, false)) return;
        advance_after_action_round(state);
        return;
    }

    state.ctx().decision_type = DecisionType::SELECT_CARD;
    offer_cuban_missile_payoff(state);
}

void StateMachine::begin_turn_cleanup(GameState& state) noexcept {
    // A decided game has no turn end at all, so there is nothing to park.
    if (state.current_phase == Phase::GAME_OVER ||
        state.victory_points >= 20 || state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
        return;
    }
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = Player::NONE;
    state.ctx().decision_type = DecisionType::ROLL_DIE;
    state.ctx().pending_roll = RollType::TURN_CLEANUP;
}

void StateMachine::end_turn(GameState& state) noexcept {
    // Nothing in the turn's cleanup happens if the game is already decided -- the win took
    // effect in the action round that produced it, and there is no turn end after it.
    if (state.current_phase == Phase::GAME_OVER ||
        state.victory_points >= 20 || state.victory_points <= -20) {
        state.current_phase = Phase::GAME_OVER;
        return;
    }

    // Phase E: Military Operations Status Check
    Scoring::evaluate_military_ops(state);
    if (state.current_phase == Phase::GAME_OVER) return;

    // Check Eagle/Bear Has Landed (Box 6) Privilege
    bool us_landed = SpaceRace::has_eagle_bear_landed(state, Player::US);
    bool ussr_landed = SpaceRace::has_eagle_bear_landed(state, Player::USSR);
    Player landed_player = us_landed ? Player::US : (ussr_landed ? Player::USSR : Player::NONE);

    if (landed_player != Player::NONE) {
        bool has_cards = false;
        for (uint8_t i = 1; i <= 110; ++i) {
            if (in_hand_of(state.card_locations[i], landed_player)) {
                has_cards = true;
                break;
            }
        }
        if (has_cards) {
            state.ctx() = DecisionContext{};
            state.ctx().decision_player = landed_player;
            state.ctx().decision_type = DecisionType::SELECT_CARD;
            state.ctx().resolving_card = card_ids::EAGLE_BEAR_LANDED_DISCARD;
            state.ctx().allow_early_stop = 1;
            return;
        }
    }

    finish_end_turn(state);
}

void StateMachine::finish_end_turn(GameState& state) noexcept {
    // Phase F: Check held cards (Scoring cards cannot be held!)
    bool us_holds = false;
    bool ussr_holds = false;
    for (uint8_t i = 1; i <= 110; ++i) {
        if (CardData::is_scoring_card(i)) {
            if (in_hand_of(state.card_locations[i], Player::US)) us_holds = true;
            if (in_hand_of(state.card_locations[i], Player::USSR)) ussr_holds = true;
        }
    }
    if (us_holds || ussr_holds) {
        state.current_phase = Phase::GAME_OVER;
        if (us_holds && ussr_holds) {
            state.victory_points = 0;
        } else if (us_holds) {
            state.victory_points = -20;
        } else {
            state.victory_points = 20;
        }
        return;
    }

    // Phase G: Flip China Card face up
    state.china_card_playable = 1;

    // Phase H: Advance Turn
    state.persistent_effects &= ~effect_bits::TURN_CLEANUP_MASK;
    state.turn_aggregates.clear();

    state.turn++;
    state.action_round = 0;

    // Rule 4.4: the new era's cards join the existing deck, and the discard pile stays where it
    // is until the deck next runs out. Until E6 both transitions also shuffled the discard back
    // in, so everything spent since the last reshuffle -- scoring cards included -- came round
    // again at turns 4 and 8. The human corpus never prints *RESHUFFLE* at turn 4 or 8, and
    // prints one at turns 3 and 7, where the deck really does run out
    // (research/findings/engine/engine_revisions.md, "The E5 -> E6 boundary").
    if (state.turn == 4) {
        add_era_cards_to_deck(state, WarEra::MID);
    } else if (state.turn == 8) {
        add_era_cards_to_deck(state, WarEra::LATE);
    }

    if (state.turn <= 10) {
        start_turn(state);
    } else {
        // Phase I: Final Scoring
        Scoring::execute_final_scoring(state);
    }
}

// Where `p` may place Influence is fixed by the board at the start of the action round (rules.md,
// Adjacency Requirement): the target or a neighbour must have held friendly Influence then. The
// snapshot is taken when the Ops begin, except on an opponent's card played event-first, where it
// is taken before the Event so that Influence the Event gives `p` -- The Iron Lady's USSR point in
// Argentina -- opens nothing (ENG-8).
static void snapshot_op_influence(GameState& state, Player p) noexcept {
    state.ctx().start_influence_nodes = {};
    for (uint8_t i = 0; i < 84; ++i) {
        if (state.countries[i].get_influence(p) > 0) {
            state.ctx().set_start_influence(i);
        }
    }
    state.ctx().set_start_influence_owner(p);
}

// P17 helpers. The merged resolution node and the deferred Ops node reach the same three Ops
// modes by different routes, so the work of starting one lives in a single place.

static OpMode to_op_mode(Resolution r) noexcept {
    switch (r) {
        case Resolution::OPS_COUP:    return OpMode::COUP;
        case Resolution::OPS_REALIGN: return OpMode::REALIGN;
        default:                      return OpMode::INFLUENCE;
    }
}

// Playing a card for Operations discharges a Missile Envy obligation.
static void clear_force_for_card(GameState& state, Player p, uint8_t card) noexcept {
    if (state.forced_card_player == p &&
        (state.forced_card_id == card || state.forced_card_id == card_ids::MISSILE_ENVY ||
         state.forced_card_id == 0)) {
        state.forced_card_player = Player::NONE;
        state.forced_card_id = 0;
    }
}

// Side effects of committing a card to its Operations, independent of WHEN the Ops are spent.
//
// P17 split the old single OPS branch in two -- ops-first and event-first now arrive at different
// places -- and these were left behind in the ops-first one. On an opponent's card, event-first is
// still a play FOR OPERATIONS, so it owes the same charges. Missing them cost the US its Flower
// Power penalty on four corpus games, every one a war card played at a turn's last action round:
// replays 56, 138, 258 and 272.
//
// Returns false if the game ended (Flower Power can take the US to -20), in which case the caller
// must stop.
static bool commit_card_to_ops(GameState& state, Player p, uint8_t card) noexcept {
    clear_force_for_card(state, p, card);

    // Flower Power charges the US 2 VP for a war card played for Operations.
    if (!CardHandlers::charge_flower_power_for_ops(state, p, card)) return false;
    if (card == card_ids::THE_CHINA_CARD && p == Player::US) {
        state.clear_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    }
    return true;
}

// Open the POINT_NODE that spends the Ops. `pending_ops_value` must already be set.
//
// Realignment is repeatable and keeps its stop; a coup is a single action and does not. Influence
// has no stop either: every point must be placed, and CONFIRM_DONE is offered only once the board
// has no legal target left (DecisionContext::is_ops_influence_play, read by the mask). The merged
// mask gates coup on a target existing and offers the other two unconditionally -- choosing
// realignment or influence with nothing to hit ends at once, while a coup with nothing to hit
// would have no way out.
static bool begin_op_mode(GameState& state, Player p, OpMode op_mode) noexcept {
    const uint8_t ops = state.ctx().pending_ops_value;
    state.ctx().op_mode = op_mode;

    if (op_mode == OpMode::COUP) {
        state.ctx().decision_type = DecisionType::POINT_NODE;
        state.ctx().remaining_steps = 1;
        state.ctx().max_per_country = 1;
        state.ctx().allow_early_stop = 0;
        return true;
    }

    // An event-first play took its snapshot before the Event (snapshot_op_influence); the Ops
    // beginning puts it in force. A country the Event emptied leaves it too, as it always has: the
    // fix withholds what the Event gave, and gives nothing back that the Event took.
    if (op_mode == OpMode::INFLUENCE) {
        if (state.ctx().timing_branch == static_cast<uint8_t>(TimingBranch::EVENT_FIRST) &&
            state.ctx().start_influence_owned_by(p)) {
            state.ctx().start_influence_nodes[1] &= ~DecisionContext::START_INFLUENCE_PENDING;
            for (uint8_t i = 0; i < 84; ++i) {
                if (state.countries[i].get_influence(p) == 0) state.ctx().clear_start_influence(i);
            }
        } else {
            snapshot_op_influence(state, p);
        }
    }
    state.ctx().decision_type = DecisionType::POINT_NODE;
    state.ctx().remaining_steps = ops;
    state.ctx().allow_early_stop = (op_mode == OpMode::INFLUENCE) ? 0 : 1;
    return true;
}

bool StateMachine::step(GameState& state, const MicroAction& action) noexcept {
    return step(state, action, nullptr);
}

// P43 (ts/canonical.hpp): a placement in a covered run moves the run's floor to the country placed,
// in the placing seat's order. Written only for a seat playing a canonical bit, so a game without one
// never touches the fields; anything else that happens in the frame clears a floor it left. The step
// itself is validated against the flat mask, which applies the floor, so a caller cannot place below it.
bool StateMachine::step(GameState& state, const MicroAction& action, const uint8_t* legal) noexcept {
    const uint32_t order = canonical::run_order(state);
    const uint16_t tag = order != 0 ? canonical::run_tag(state) : 0;
    const uint8_t depth = state.ctx_stack_depth;
    if (!step_impl(state, action, legal)) return false;
    if (state.ctx_stack_depth != depth) return true;
    DecisionContext& c = state.ctx();
    if (order != 0 && action.decision_type == DecisionType::POINT_NODE && action.primary_id < 84) {
        c.canon_floor = static_cast<uint8_t>(canonical::rank_in(order, action.primary_id) + 1);
        c.canon_tag = tag;
        c.canon_left = c.remaining_steps;
    } else if (c.canon_floor != 0) {
        c.canon_floor = 0;
        c.canon_tag = 0;
        c.canon_left = 0;
    }
    return true;
}

bool StateMachine::step_impl(GameState& state, const MicroAction& action, const uint8_t* legal) noexcept {
    // Reset ephemeral die roll record for the current step
    state.last_roll = DieRollRecord{};
    state.last_die_roll = 0;
    state.last_opp_die_roll = 0;
    state.last_event_track_count = 0;
    if (state.current_phase == Phase::GAME_OVER) return false;

    // An action must answer the question being asked. Without this, a mismatched action can fall
    // through to a branch that acts on it anyway -- a SELECT_CARD action was once accepted at a
    // ROLL_DIE node and silently consumed the die roll, so a coup reached resolution and never
    // rolled. Rejecting is strictly better than that: the caller sees false and the state is
    // untouched, instead of the game quietly taking a different course.
    //
    // This rejects nothing legal: every action in the 212-wide legal mask decodes to a
    // MicroAction whose decision_type equals the context's (verified over 177,258 state/action
    // pairs). Callers that construct actions by hand must set the type the context asks for.
    //
    // P17 section 5 adds the one legitimate exception. UN Intervention played ON the card Grain
    // Sales drew arrives card-shaped at a resolution node: everywhere else UN Intervention is
    // the card played and NAMES a companion, but here the companion already exists and UN
    // Intervention is named as its resolution. It is the only node in the game where a card slot
    // is legal at a SELECT_PLAY_MODE decision, and the flat-mask check below still has the final
    // say on whether this particular card is offered.
    const bool grain_sales_un_intervention =
        state.ctx().decision_type == DecisionType::SELECT_PLAY_MODE &&
        state.ctx().resolving_card == card_ids::GRAIN_SALES &&
        action.decision_type == DecisionType::SELECT_CARD &&
        action.primary_id == card_ids::UN_INTERVENTION;
    if (action.decision_type != state.ctx().decision_type && !grain_sales_un_intervention) {
        return false;
    }

    // ROLL_DIE is the one decision the 212-wide mask cannot constrain: its primary_id and
    // secondary_id are the forced dice themselves (actor's and opponent's) rather than indices
    // into the mask, and forcing a die is a deliberate replayer and test affordance. So the range
    // has to be checked here or nowhere -- a hand-built action carrying 26, or the 255 sentinel,
    // was applied as though it were a die. 0 means "roll normally".
    if (action.decision_type == DecisionType::ROLL_DIE
        && (action.primary_id > 6 || action.secondary_id > 6)) {
        return false;
    }

    // One definition of legality. The mask decides; this function applies.
    //
    // Both used to decide, from separate code, and where they differed the mask offered an action
    // this function refused: six of twelve forced-play conditions, two of them deadlocks where the
    // mask's ENTIRE legal set was refused and the position could not advance. Callers either
    // crashed or -- before the batched-step guard -- silently failed to advance while the trainer
    // credited the transition anyway. See research/plans/P14_one_definition_of_legality.md.
    //
    // Safe because the encoding round-trips: 559,500 masked actions across 300 games re-encoded to
    // the index they decoded from, 0 failures. A failure here would refuse a legal move, which is
    // worse than the bug being fixed, so it is worth re-measuring if the flat space ever changes.
    //
    // NOT sufficient on its own, which is why the two guards above stay: several MicroActions map
    // to one flat index, so a forced die's value and other sub-index fields are invisible here.
    //
    // A caller may hand the mask in (`legal`): the mask of this very state, generated before the
    // three die-roll fields above were cleared. The mask reads neither them nor the RNG, which
    // engine test CachedState.MaskIgnoresRngAndLastRolls pins, so it is the mask generated here.
    {
        uint8_t legal_flat[FLAT_ACTION_SPACE_SIZE];
        if (legal == nullptr) {
            ActionMask::generate_flat_mask_212(state, legal_flat);
            legal = legal_flat;
        }
        const int16_t flat = ActionMask::encode_micro_action_212(state, action);
        if (flat < 0 || flat >= static_cast<int16_t>(FLAT_ACTION_SPACE_SIZE) || !legal[flat]) {
            return false;
        }
    }

    // 1. SETUP PHASE
    if (state.current_phase == Phase::SETUP) {
        if (state.ctx().decision_player == Player::USSR) {
            uint8_t cid = action.primary_id;
            if (cid < 84 && MapData::get_country(cid).in_eastern_europe) {
                state.countries[cid].add_influence(Player::USSR, 1);
                state.ctx().remaining_steps--;
                if (state.ctx().remaining_steps == 0) {
                    // Transition to US placing 7 in Western Europe
                    state.ctx().decision_player = Player::US;
                    state.ctx().remaining_steps = 7;
                }
                return true;
            }
            return false;
        } else if (state.ctx().decision_player == Player::US) {
            uint8_t cid = action.primary_id;
            if (state.ctx().pending_ops_value == 0) {
                // Stage 0: 7 points in Western Europe
                if (cid < 84 && MapData::get_country(cid).in_western_europe) {
                    state.countries[cid].add_influence(Player::US, 1);
                    state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        // Transition to US placing 2 bonus influence in countries with existing US presence
                        state.ctx().pending_ops_value = 1; // Stage 1: Bonus placement
                        state.ctx().remaining_steps = 2;
                    }
                    return true;
                }
                return false;
            } else {
                // Stage 1: 2 bonus influence in any country with existing US influence
                if (cid < 84 && state.countries[cid].us_influence > 0) {
                    state.countries[cid].add_influence(Player::US, 1);
                    state.ctx().remaining_steps--;
                    if (state.ctx().remaining_steps == 0) {
                        // Setup complete -> start Turn 1
                        start_turn(state);
                    }
                    return true;
                }
                return false;
            }
        }
    }

    // 2. HEADLINE PHASE CARD SELECTION
    // `state.headline_stage == 0` is "both cards still to be chosen". This used to read
    // temp_cards[4], which held the headline stage in a much older design; that role moved to
    // headline_stage and the read survived, guarding on a slot whose only remaining writer was
    // realignment staging.
    if (state.current_phase == Phase::HEADLINE && state.headline_stage == 0 &&
        state.ctx().decision_type == DecisionType::SELECT_CARD && state.ctx().resolving_card == 0) {
        Player p = state.ctx().decision_player;
        uint8_t card = action.primary_id;
        if (p == Player::US && state.headline_us_card == 0) {
            state.headline_us_card = card;
            if (state.headline_ussr_card == 0) {
                state.ctx().decision_player = Player::USSR;
                state.ctx().decision_type = DecisionType::SELECT_CARD;
                return true;
            }
        } else if (p == Player::USSR && state.headline_ussr_card == 0) {
            state.headline_ussr_card = card;
            if (state.headline_us_card == 0) {
                state.ctx().decision_player = Player::US;
                state.ctx().decision_type = DecisionType::SELECT_CARD;
                return true;
            }
        }

        // Both headlines selected -> resolve headlines in ops priority order
        uint8_t us_h = state.headline_us_card;
        uint8_t ussr_h = state.headline_ussr_card;
        bool ussr_cancelled = (us_h == card_ids::DEFECTORS);

        uint8_t us_hv = CardData::get_card(us_h).ops;
        uint8_t ussr_hv = CardData::get_card(ussr_h).ops;
        bool us_first = (us_hv >= ussr_hv); // US wins ties

        uint8_t first_card = us_first ? us_h : ussr_h;
        Player first_owner = us_first ? Player::US : Player::USSR;
        uint8_t second_card = us_first ? ussr_h : us_h;
        Player second_owner = us_first ? Player::USSR : Player::US;

        state.headline_first_card = first_card;
        state.headline_second_card = second_card;
        state.headline_first_owner = first_owner;
        state.headline_second_owner = second_owner;
        state.headline_stage = 1; // Stage 1: Resolving 1st headline

        // Both headlines are committed face down before either resolves, so neither is in a
        // hand while the other plays out. Leaving the second one there let every card that
        // reads a hand count it: at turn 2's headline of replay 144 the US headlines Middle
        // East Scoring and the USSR's The Cambridge Five found it still in the US hand and
        // asked where to place, where the log reads "US has no cards to reveal". Missile Envy
        // is the sharper case -- it takes the opponent's highest Ops card, and their headline
        // card was in the running.
        // The real destination is set after the event resolves, overwriting this.
        state.card_locations[first_card] = CardLocation::HEADLINE_COMMITTED;
        state.card_locations[second_card] = CardLocation::HEADLINE_COMMITTED;

        if (first_owner == Player::USSR && ussr_cancelled) {
            state.card_locations[first_card] = CardLocation::DISCARD_PILE;
            advance_headline_step(state);
            return true;
        }

        const auto& c_info = CardData::get_card(first_card);
        Player exec_player = (c_info.side == get_opponent(first_owner)) ? get_opponent(first_owner) : first_owner;
        state.phasing_player = first_owner;
        state.ctx().decision_player = exec_player;
        state.ctx().resolving_card = first_card;

        const bool first_fired = CardHandlers::event_has_effect(state, first_card, exec_player);
        bool done = CardHandlers::trigger_event(state, first_card, exec_player);
        CardHandlers::relocate_played_card(state, first_card, first_fired, CardHandlers::Handover::Respect);
        if (done) {
            advance_headline_step(state);
        }
        return true;
    }

    // 3. INTERACTIVE DECISIONS (HEADLINE OR ACTION ROUNDS)
    if (state.current_phase == Phase::ACTION_ROUND || state.current_phase == Phase::HEADLINE) {
        DecisionType dt = state.ctx().decision_type;
        Player p = state.ctx().decision_player;

        // P17 section 5: Grain Sales resolves the card it drew on its OWN frame, so a Resolution
        // here REPLACES the frame instead of nesting. Done before the dispatch below, so the
        // ordinary SELECT_PLAY_MODE path plays the card -- repeating that logic inside the
        // handler would be the second definition of legality P14 forbids, and the first attempt
        // at this card foundered on a child frame that nothing popped when a play completed.
        //
        // The decline and UN Intervention are NOT handled here: both are Grain Sales' own
        // business rather than the drawn card's, so they go to the handler like any other event
        // sub-decision. Grain Sales' own card is left alone, exactly as before -- whoever played
        // it relocates it, and that is unchanged by which answer the US gives.
        if (state.ctx().resolving_card == card_ids::GRAIN_SALES &&
            state.ctx().decision_type == DecisionType::SELECT_PLAY_MODE &&
            !action.is_confirm_done() &&
            action.decision_type != DecisionType::SELECT_CARD) {
            // Only the frame changes hands. The drawn card is already in the US hand -- the
            // trigger put it there, so that the mask and this step agree about where it is.
            // Moving it here instead would reintroduce exactly the split that costs.
            state.ctx().resolving_card = 0;
        }

        // If resolving active event sub-decision
        if (state.ctx().resolving_card != 0) {
            if (state.ctx().resolving_card == card_ids::EAGLE_BEAR_LANDED_DISCARD) {
                uint8_t card = action.primary_id;
                if (card >= 1 && card <= 110 && !action.is_confirm_done()) {
                    state.card_locations[card] = CardLocation::DISCARD_PILE;
                }
                state.ctx() = DecisionContext{};
                finish_end_turn(state);
                return true;
            }

            // What the handler says is the answer. A confirm-done used to end the event
            // frame here whatever the handler had just done, which silently discarded a
            // decision it had opened: Latin American Debt Crisis asks the US whether to
            // discard a 3 Ops card, and the US declining is exactly what lets the USSR double
            // its Influence in two South American countries. At turn 8 AR4 of ts-replayer
            // game 117 the doubling vanished and the US went straight on to spend the card's
            // own Ops.
            const DecisionContext before_ctx = state.ctx();
            const uint8_t before_card = state.ctx().resolving_card;
            bool finished = CardHandlers::handle_event_step(state, action);
            if (!finished && action.is_confirm_done()) {
                // A handler that neither finishes nor changes anything at all has not consumed
                // the decline: the same decision would be asked again, forever. Ending the
                // frame for it would paper over a defect in that handler with a game that
                // quietly skipped part of a card, so it fails here instead.
                //
                // The whole context is compared rather than a chosen few fields. Picking
                // fields is guesswork of exactly the kind this replaces: De-Stalinization's
                // stage 1 decline moves on to stage 2 without touching decision_player or
                // decision_type, and when two Influence were removed remaining_steps happens
                // to land on the value it already had -- a real continuation that a
                // three-field test called a defect.
                if (std::memcmp(&before_ctx, &state.ctx(), sizeof(DecisionContext)) == 0) {
                    // One sanctioned exception: a POINT_NODE the board cannot offer a single
                    // legal target for. The decline is then the escape the mask added rather
                    // than a choice the card was written to understand, and ending the event
                    // is right -- an event does as much as the board allows. The mask has
                    // already reported that position in full, so it is not lost here.
                    if (state.ctx().decision_type != DecisionType::POINT_NODE ||
                        has_legal_point_target(state)) {
                        invariant_failed("event handler did not consume a confirm-done; card",
                                         static_cast<int>(before_card));
                    }
                    finished = true;
                }
            }
            if (finished) {
                state.ctx().resolving_card = 0;
                if (state.ctx_stack_depth > 0) {
                    // Unwind every finished frame, not just one. Events nest: Five Year Plan
                    // makes the USSR discard a card and that card's own event runs a frame
                    // deeper, so on an EVENT_FIRST play the stack is two deep and a single pop
                    // landed on Five Year Plan's own frame -- which is not SELECT_OP_MODE, so
                    // the action round ended and the USSR lost the Ops they had paid for. At
                    // turn 2 AR6 of ts-replayer game 105 their influence in Iran never went in.
                    bool resumed = false;
                    while (state.ctx_stack_depth > 0) {
                        state.pop_context();
                        if (state.ctx().decision_type == DecisionType::SELECT_OP_MODE) {
                            resumed = true;   // the player's own Ops are still owed
                            break;
                        }
                        state.ctx().resolving_card = 0;
                    }
                    if (!resumed) {
                        if (state.current_phase == Phase::HEADLINE) {
                            advance_headline_step(state);
                        } else {
                            advance_after_action_round(state);
                        }
                    }
                } else if (state.current_phase == Phase::HEADLINE) {
                    advance_headline_step(state);
                } else {
                    advance_after_action_round(state);
                }
            }
            return true;
        }

        switch (dt) {
            case DecisionType::SELECT_CARD: {
                uint8_t card = action.primary_id;

                // We Will Bury You on a trapped US round. Quagmire spends the round on a discard
                // and a die, or on nothing at all when no card qualifies, so UN Intervention
                // cannot be played in it -- and that round is still the US's "next action round"
                // the card names. The settlement below sits past the trap branch, which returns
                // first, so the 3 VP waited for whichever round the US next played a card in.
                if (p == Player::US && state.current_phase == Phase::ACTION_ROUND &&
                    state.has_flag(effect_bits::QUAGMIRE_ACTIVE) &&
                    settle_we_will_bury_you(state, false)) {
                    return true;
                }

                if (card == 0 || action.is_confirm_done()) {
                    // A US round passed -- holding nothing but the China Card, or leaving an
                    // eighth round -- is still the US's next action round, and no UN
                    // Intervention was played in it.
                    if (p == Player::US && state.current_phase == Phase::ACTION_ROUND &&
                        settle_we_will_bury_you(state, false)) {
                        return true;
                    }
                    if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                    else advance_after_action_round(state);
                    return true;
                }
                state.ctx().pending_op_card = card;

                // Quagmire / Bear Trap handling. Action rounds only, as in the mask: a
                // headline is not constrained by the trap and its event resolves normally.
                bool trapped = state.current_phase == Phase::ACTION_ROUND &&
                               ((p == Player::US && state.has_flag(effect_bits::QUAGMIRE_ACTIVE)) ||
                                (p == Player::USSR && state.has_flag(effect_bits::BEAR_TRAP_ACTIVE)));

                if (trapped) {
                    // Effective Ops, and the same reading the mask takes: a card is eligible
                    // for the trap on what it is worth now, not on what is printed.
                    // Containment and Brezhnev Doctrine each add one to their side's cards and
                    // Red Scare/Purge takes one away. Judged here on the printed value, a card
                    // the mask had offered as a discard fell through to an ordinary play: at
                    // turn 6 AR1 of ts-replayer game 229 the USSR discards Panama Canal
                    // Returned, 1 Op printed and 2 under Brezhnev Doctrine, and the engine
                    // asked how it wanted to spend them.
                    if (Operations::get_effective_ops(state, card, p) >= 2) {
                        if (state.forced_card_player == p && (state.forced_card_id == card || state.forced_card_id == 0)) {
                            state.forced_card_player = Player::NONE;
                            state.forced_card_id = 0;
                        }
                        state.card_locations[card] = CardLocation::DISCARD_PILE;
                        state.ctx().pending_roll = RollType::TRAP_ESCAPE;
                        state.ctx().roll_target = card;
                        state.ctx().roll_actor = p;
                        state.ctx().decision_player = Player::NONE;
                        state.ctx().decision_type = DecisionType::ROLL_DIE;
                        return true;
                    }
                }

                // We Will Bury You on a US action round. UN Intervention waits for its play
                // mode below: only its Event answers the debt.
                if (p == Player::US && state.current_phase == Phase::ACTION_ROUND &&
                    card != card_ids::UN_INTERVENTION && settle_we_will_bury_you(state, false)) {
                    return true;
                }

                // Scoring card auto-resolve
                if (CardData::is_scoring_card(card)) {
                    state.card_locations[card] = CardData::get_card(card).one_time ? CardLocation::REMOVED_FROM_GAME : CardLocation::DISCARD_PILE;
                    CardHandlers::trigger_event(state, card, p);
                    if (state.current_phase != Phase::GAME_OVER) {
                        if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                        else advance_after_action_round(state);
                    }
                    return true;
                }

                // Transition to SELECT_PLAY_MODE
                state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;
                return true;
            }

            case DecisionType::SELECT_PLAY_MODE: {
                uint8_t card = state.ctx().pending_op_card;
                // P17: five-way resolution. See `Resolution` in types.hpp.
                Resolution mode = static_cast<Resolution>(action.primary_id);
                const bool opponent_card = CardData::is_opponent_card(card, p);
                const bool is_ops = (mode == Resolution::OPS_INFLUENCE ||
                                     mode == Resolution::OPS_COUP ||
                                     mode == Resolution::OPS_REALIGN);

                // Forced play (Missile Envy recipient must play it for Operations). Mirrors
                // ActionMask::generate_flat_mask_212 exactly -- see the comment there for why each
                // test is present. This copy had none of the three, which is how the mask came to
                // offer play modes the engine then refused: a headline, a stale force, the China
                // Card and, worst, a scoring card, where the mask's only legal action was refused
                // and the position could not advance at all.
                if (state.current_phase == Phase::ACTION_ROUND &&
                    state.forced_card_player == p &&
                    card == card_ids::MISSILE_ENVY &&
                    in_hand_of(state.card_locations[card_ids::MISSILE_ENVY], p)) {
                    if (!is_ops) {
                        return false;
                    }
                }

                // We Will Bury You on the US action round UN Intervention is played in.
                if (p == Player::US && state.current_phase == Phase::ACTION_ROUND &&
                    settle_we_will_bury_you(state, mode == Resolution::EVENT)) {
                    return true;
                }

                if (mode == Resolution::SPACE) {
                    state.ctx().pending_roll = RollType::SPACE_RACE;
                    state.ctx().roll_target = card;
                    state.ctx().roll_actor = p;
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                if (mode == Resolution::EVENT && !opponent_card) {
                    // A friendly or neutral card played as its Event: that is the whole play.
                    if (card == card_ids::THE_CHINA_CARD ||
                        (card == card_ids::DEFECTORS && p == Player::US && state.current_phase == Phase::ACTION_ROUND) ||
                        !CardHandlers::can_trigger_event(state, card, p)) {
                        return false; // Illegal event play
                    }
                    const bool fired = CardHandlers::event_has_effect(state, card, p);
                    bool done = CardHandlers::trigger_event(state, card, p);
                    CardHandlers::relocate_played_card(state, card, fired, CardHandlers::Handover::Respect);
                    if (done && state.current_phase != Phase::GAME_OVER) {
                        finish_play_without_ops(state);
                    }
                    return true;
                }

                if (mode == Resolution::EVENT && opponent_card) {
                    // Event-first on an opponent's card. The Ops choice is STAGED on this frame
                    // and a new frame is pushed for the event; pop_context() returns here with
                    // timing_branch intact, which is what tells the model at that deferred node
                    // that the event has already resolved (observation TIMING_EVENT_FIRST).
                    //
                    // timing_branch must keep being set even though CHOOSE_TIMING_BRANCH is gone
                    // as a decision: it is still the only thing distinguishing "Ops before the
                    // event" from "Ops after it".
                    // Event-first is still a play for Operations, so it owes the same charges.
                    if (!commit_card_to_ops(state, p, card)) return true;   // game ended
                    state.ctx().timing_branch = static_cast<uint8_t>(TimingBranch::EVENT_FIRST);
                    snapshot_op_influence(state, p);   // the board before the Event (ENG-8)
                    state.ctx().start_influence_nodes[1] |= DecisionContext::START_INFLUENCE_PENDING;
                    Operations::grant_card_ops_to_ctx(state, card, p);
                    state.ctx().decision_type = DecisionType::SELECT_OP_MODE;
                    state.ctx().decision_player = p;
                    if (!state.push_context()) {
                        invariant_failed("event chain too deep; card", static_cast<int>(card));
                    }

                    Player opp = get_opponent(p);
                    state.ctx().decision_player = opp;
                    state.ctx().resolving_card = card;

                    const bool fired = CardHandlers::event_has_effect(state, card, opp);
                    bool done = CardHandlers::trigger_event(state, card, opp);
                    CardHandlers::relocate_played_card(state, card, fired, CardHandlers::Handover::Respect);
                    if (done) {
                        state.pop_context();
                    }
                    return true;
                }

                if (is_ops) {
                    if (!commit_card_to_ops(state, p, card)) return true;   // game ended
                    // Ops-first on an opponent's card; timing_branch records which, so the
                    // event fires after the Ops are spent (advance_after_ops reads it).
                    // 255 is the "no branch chosen" sentinel the observation reads, and it is
                    // the right value for a friendly card: no event is pending either way.
                    state.ctx().timing_branch = opponent_card
                        ? static_cast<uint8_t>(TimingBranch::OPS_FIRST)
                        : static_cast<uint8_t>(255);
                    Operations::grant_card_ops_to_ctx(state, card, p);
                    return begin_op_mode(state, p, to_op_mode(mode));
                }
                return false;
            }

            case DecisionType::SELECT_OP_MODE: {
                // Junta and Tear Down This Wall grant free Ops usable only for a coup or a
                // realignment, so choosing Influence with those is a decline. It is the Ops
                // that are restricted, not the card: the player whose card it is still has
                // their own Ops afterwards and may place Influence with them. Keying this on
                // pending_op_card conflated the two, and since the mask then offered nothing
                // but Influence, those Ops could not be spent at all -- at turn 9 AR1 of
                // ts-replayer game 146 the USSR plays Tear Down This Wall, the US takes its
                // three free realignments against France, and the USSR's own three Ops, which
                // the log spends placing in France, silently ended the action round instead.
                const bool free_action_bars_influence = free_action::region_locked(
                    state.ctx().pending_op_card, state.ctx().event_granted_ops);
                if (action.is_confirm_done() || action.primary_id == 255 ||
                    (action.primary_id == 0 && free_action_bars_influence)) {
                    if (state.current_phase == Phase::HEADLINE && state.ctx_stack_depth == 0) advance_headline_step(state);
                    else advance_after_ops(state);
                    return true;
                }
                OpMode op_mode = static_cast<OpMode>(action.primary_id);
                if (op_mode == OpMode::INFLUENCE && free_action_bars_influence) {
                    // P17: INFLUENCE is not legal here and the mask no longer offers it. Declining
                    // the free action is the shared decline index, handled above as confirm/done.
                    // Refusing keeps one definition of legality (P14) rather than accepting an
                    // action the mask withheld.
                    return false;
                }
                if (op_mode != OpMode::INFLUENCE && op_mode != OpMode::COUP &&
                    op_mode != OpMode::REALIGN) {
                    return false;
                }
                return begin_op_mode(state, p, op_mode);
            }

            case DecisionType::POINT_NODE: {
                if (action.is_confirm_done() || action.primary_id == 84) {
                    // Early stop Ops
                    advance_after_ops(state);
                    return true;
                }

                uint8_t cid = action.primary_id;
                // No forced dice are read here any more. A POINT_NODE names a target; the die
                // that resolves it is supplied on the ROLL_DIE action that follows, which is
                // the only place either die is read. `secondary_id` and `flags` are ignored.

                if (state.ctx().op_mode == OpMode::COUP) {
                    uint8_t coup_ops = state.ctx().pending_ops_value;
                    uint8_t op_card = state.ctx().pending_op_card;
                    const bool china_card = (op_card == card_ids::THE_CHINA_CARD);
                    const bool vietnam_bonus =
                        (p == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE));

                    if (china_card || vietnam_bonus) {
                        // A card played for Operations takes every Ops modifier and its limit
                        // (get_effective_ops_in). Ops an event granted were modified when they
                        // were granted (rule 7.4.3), so the coup takes that grant as it stands
                        // outside the bonus regions -- ops_plain -- plus the Vietnam Revolts bonus.
                        const uint8_t plain = Operations::plain_budget(state, op_card, p);
                        if (plain == Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION)) {
                            coup_ops = Operations::get_effective_ops_in(state, op_card, p, cid);
                        } else {
                            coup_ops = plain;
                            if (vietnam_bonus && MapData::get_country(cid).in_southeast_asia) {
                                coup_ops += 1;
                            }
                        }
                    }
                    state.ctx().pending_roll = RollType::COUP;
                    state.ctx().roll_target = cid;
                    state.ctx().roll_actor = p;
                    state.ctx().pending_ops_value = coup_ops;

                    // Couping under Cuban Missile Crisis cancels it, and the US may pay from
                    // West Germany or from Turkey -- its choice, and only its own when both
                    // can pay. Taken silently, always from West Germany, that was right in
                    // eight of the corpus's ten US cancellations and wrong in two: at turn 5
                    // AR7 of ts-replayer game 234 the US holds 5 in West Germany and 2 in
                    // Turkey and pays from Turkey.
                    //
                    // Asked here, before the die, so the coup's own setup above is already in
                    // the roll fields and the chance node opens on the far side of the answer with
                    // nothing to rebuild. Where only one country can pay, or neither, there is
                    // no choice to make and execute_coup settles it as before.
                    if (p == Player::US && state.has_flag(effect_bits::CMC_ACTIVE_USSR) &&
                        state.countries[countries::WEST_GERMANY].us_influence >= 2 &&
                        state.countries[countries::TURKEY].us_influence >= 2) {
                        state.ctx().decision_player = Player::US;
                        state.ctx().decision_type = DecisionType::POINT_NODE;
                        state.ctx().remaining_steps = 1;
                        state.ctx().allow_early_stop = 0;   // cancelling is not optional
                        state.ctx().resolving_card = card_ids::CUBAN_MISSILE_CRISIS;
                        return true;
                    }

                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                if (state.ctx().op_mode == OpMode::REALIGN) {
                    state.ctx().pending_roll = RollType::REALIGNMENT;
                    state.ctx().roll_target = cid;
                    state.ctx().roll_actor = p;
                    state.ctx().decision_player = Player::NONE;
                    state.ctx().decision_type = DecisionType::ROLL_DIE;
                    return true;
                }

                // Influence placement
                if (cid < 84) {
                    uint8_t cost = Operations::get_influence_cost(state, p, cid);
                    if (state.ctx().remaining_steps >= cost && Operations::can_place_influence(state, p, cid)) {
                        Operations::place_influence(state, p, cid);
                        state.ctx().remaining_steps -= cost;

                        withdraw_conditional_ops_bonus(state, p, cid);

                        if (state.ctx().remaining_steps == 0) {
                            advance_after_ops(state);
                        }
                        return true;
                    }
                }

                // Only the confirm-done (primary 255) ends the Ops here. A country the placement
                // above refused is not a request to stop -- 0 is Canada -- and the mask never
                // offers one, so it is refused rather than read as one.
                if (action.primary_id >= 84) {
                    advance_after_ops(state);
                    return true;
                }
                return false;
            }

            case DecisionType::ROLL_DIE: {
                const RollType rt = state.ctx().pending_roll;

                if (rt == RollType::TURN_CLEANUP) {
                    end_turn(state);
                    return true;
                }

                // The only source of a forced die is this action. `primary_id` is the acting
                // player's die and `secondary_id` the opponent's; realignment is the only roll
                // that uses both. Nothing is read back out of the state, which is what makes
                // the two dice impossible to transpose.
                const uint8_t forced_actor = action.primary_id;
                const uint8_t forced_opponent = action.secondary_id;
                const Player roll_actor = (state.ctx().roll_actor != Player::NONE)
                    ? state.ctx().roll_actor : state.phasing_player;

                if (rt == RollType::COUP) {
                    uint8_t cid = state.ctx().roll_target;
                    uint8_t coup_ops = state.ctx().pending_ops_value;
                    Player coup_player = roll_actor;
                    Operations::execute_coup(state, coup_player, cid, coup_ops, forced_actor);
                    if (state.current_phase != Phase::GAME_OVER) {
                        advance_after_ops(state);
                    }
                    return true;
                }

                if (rt == RollType::REALIGNMENT) {
                    uint8_t cid = state.ctx().roll_target;
                    Player realign_player = roll_actor;
                    // Mapping the acting player's die onto the side that rolls it, in the one
                    // place that knows who is acting. Previously the two dice were stored per
                    // side and re-read per actor, and a USSR realignment came out transposed.
                    uint8_t forced_us = (realign_player == Player::US) ? forced_actor : forced_opponent;
                    uint8_t forced_ussr = (realign_player == Player::USSR) ? forced_actor : forced_opponent;
                    Operations::execute_realign(state, realign_player, cid, forced_us, forced_ussr);
                    state.ctx().remaining_steps -= 1;

                    withdraw_conditional_ops_bonus(state, realign_player, cid);

                    if (state.ctx().remaining_steps == 0) {
                        advance_after_ops(state);
                    } else {
                        state.ctx().decision_player = realign_player;
                        state.ctx().decision_type = DecisionType::POINT_NODE;
                        state.ctx().allow_early_stop = 1;
                    }
                    return true;
                }

                if (rt == RollType::SPACE_RACE) {
                    uint8_t card = state.ctx().roll_target;
                    Player space_player = roll_actor;
                    SpaceRace::attempt_space(state, space_player, card, forced_actor);
                    if (state.current_phase != Phase::GAME_OVER) {
                        finish_play_without_ops(state);
                    }
                    return true;
                }

                if (rt == RollType::TRAP_ESCAPE) {
                    uint8_t card = state.ctx().roll_target;
                    Player trapped_p = roll_actor;
                    uint8_t roll = (forced_actor >= 1 && forced_actor <= 6) ? forced_actor : Prng::roll_d6(state.rng_state);
                    state.last_die_roll = roll;
                    bool escaped = (roll <= 4);
                    if (escaped) {
                        if (trapped_p == Player::US) state.clear_flag(effect_bits::QUAGMIRE_ACTIVE);
                        else state.clear_flag(effect_bits::BEAR_TRAP_ACTIVE);
                    }
                    state.last_roll = DieRollRecord{
                        .type = RollType::TRAP_ESCAPE,
                        .roller = trapped_p,
                        .card_id = card,
                        .country_id = 255,
                        .roll1 = roll,
                        .mod1 = 0,
                        .roll2 = 0,
                        .mod2 = 4, // escape threshold
                        .success = escaped,
                        .net_delta = 0
                    };
                    if (state.current_phase == Phase::HEADLINE) advance_headline_step(state);
                    else advance_after_action_round(state);
                    return true;
                }

                return false;
            }

            default:
                break;
        }
    }

    return false;
}

} // namespace ts
