#include "ts/action_mask.hpp"
#include "ts/invariant.hpp"
#include "ts/map_data.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/ops.hpp"
#include "ts/space_race.hpp"
#include <cstring>

namespace ts {

void ActionMask::generate_mask(const GameState& state, uint8_t* mask_out, size_t* out_size) noexcept {
    if (!mask_out || !out_size) return;

    if (state.current_phase == Phase::GAME_OVER) {
        *out_size = 0;
        return;
    }

    const auto& ctx = state.ctx();

    // A decision needs somebody to make it. The one exception is a chance node -- a ROLL_DIE the
    // engine owns -- which is resolved by the caller draining it, not by choosing from this mask.
    // Substituting `phasing_player` here used to offer that player's moves at a node they do not
    // own; harmless while `step` refused them, and legality-inventing once `step` trusts the mask.
    if (ctx.decision_player == Player::NONE && ctx.decision_type != DecisionType::ROLL_DIE) {
        *out_size = 0;
        return;
    }
    Player p = (ctx.decision_player != Player::NONE) ? ctx.decision_player : state.phasing_player;

    switch (ctx.decision_type) {
        case DecisionType::NONE:
            // Not a decision, so nothing is legal. This used to emit a confirm/done.
            *out_size = 0;
            break;

        case DecisionType::SELECT_CARD: {
            *out_size = 112;
            std::memset(mask_out, 0, 112);

            // 0. Space Walk (Box 6) end-of-turn discard
            if (ctx.resolving_card == card_ids::SPACE_WALK_DISCARD) {
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p)) {
                        mask_out[i] = 1;
                    }
                }
                mask_out[0] = 1; // Allow early stop / pass
                return;
            }

            // 1. If currently inside an active event's sub-decision:
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                bool any_legal = false;
                for (size_t i = 0; i < *out_size; ++i) {
                    if (mask_out[i]) { any_legal = true; break; }
                }
                if (!any_legal || ctx.allow_early_stop) {
                    mask_out[0] = 1;
                }
                return;
            }

            // 2. UN Intervention's companion selection used to be handled here, and was
            // unreachable: block 1 above returns whenever resolving_card is set, and the
            // companion decision always sets it. The rule existed twice and this was the copy
            // nobody could run -- see ENG-1, where the reachable copy in
            // CardHandlers::get_event_action_mask was offering the player's whole hand while
            // this one had the rule right.
            //
            // Confirmed dead before removal rather than argued: a report_anomaly probe in this
            // branch fired 0 times over 20,000 fuzz games (3.57M steps) and all 300 corpus games.

            // 3. Forced play (Missile Envy). Action rounds only: the card must be used "during
            // their next action round", so it constrains no headline. At turn 6's headline of
            // ts-replayer game 114 the USSR held Missile Envy from a turn 5 exchange and the
            // engine would let them headline nothing else.
            //
            // Held scoring outranks the force and DEFERS it: holding scoring cards past the end
            // of the turn loses the game outright, so the obligation that can lose it wins, and
            // the forced flags are left alone to bind on the next action round instead.
            //
            // Under a trap the mask is the same single card -- Missile Envy is 2 Ops, so it is a
            // legal discard, and discarding it is the whole action round and satisfies both
            // obligations. The state machine's trap branch already clears the force when the
            // discarded card is the forced one.
            if (state.current_phase == Phase::ACTION_ROUND &&
                state.forced_card_player == p &&
                in_hand_of(state.card_locations[card_ids::MISSILE_ENVY], p)) {
                uint8_t held = 0;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p) && CardData::is_scoring_card(i)) {
                        held++;
                    }
                }
                uint8_t cap = (state.turn <= 3) ? 6 : 7;
                const bool ar8 = (p == Player::US)
                    ? (state.has_flag(effect_bits::NORTH_SEA_OIL_ACTIVE) ||
                       SpaceRace::has_space_station_ar8(state, Player::US))
                    : SpaceRace::has_space_station_ar8(state, Player::USSR);
                if (state.turn >= 4 && ar8) cap = 8;
                const uint8_t left = (state.action_round <= cap)
                    ? static_cast<uint8_t>(cap - state.action_round + 1) : 0;

                if (held > 0 && held >= left) {
                    for (uint8_t i = 1; i <= 110; ++i) {
                        if (in_hand_of(state.card_locations[i], p) && CardData::is_scoring_card(i)) {
                            mask_out[i] = 1;
                        }
                    }
                    return;
                }
                mask_out[card_ids::THE_CHINA_CARD] = 0;   // the force blocks it too
                mask_out[card_ids::MISSILE_ENVY] = 1;
                return;
            }

            // 4. Quagmire / Bear Trap. Action rounds only: the trap constrains what a player
            // may *do on their turn*, and says nothing about the headline. Headlining any card
            // stays legal and its event resolves normally -- at turn 9's headline of
            // ts-replayer game 105 the USSR headlined Europe Scoring while trapped, which the
            // engine would not allow because a scoring card is not a 2 Ops card.
            bool trapped = state.current_phase == Phase::ACTION_ROUND &&
                           ((p == Player::US && state.has_flag(effect_bits::QUAGMIRE_ACTIVE)) ||
                            (p == Player::USSR && state.has_flag(effect_bits::BEAR_TRAP_ACTIVE)));

            if (trapped) {
                // Effective Ops, as the discard itself is judged -- the same reading Latin
                // American Debt Crisis takes of its own 3. Containment and Brezhnev Doctrine
                // each add one to their side's cards and Red Scare/Purge takes one away, and
                // a card is eligible for the trap on what it is worth now, not on what is
                // printed. At turn 6 AR1 of ts-replayer game 229 the USSR discards Panama
                // Canal Returned, 1 Op printed, which Brezhnev Doctrine makes 2.
                //
                // No target region, because a discard has none: the Asia bonuses that Vietnam
                // Revolts and the China Card carry are for operations conducted there.
                bool has_2ops = false;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p) &&
                        Operations::get_effective_ops(state, i, p) >= 2) {
                        has_2ops = true;
                        mask_out[i] = 1;
                    }
                }
                // A scoring card may be played out of a trap on either of two counts: there
                // is nothing the trap will take, or the player holds as many scoring cards as
                // they have action rounds left to play them in. The second is what stops a
                // trap from costing a player the game by holding their scoring cards past the
                // turn's end, which is a loss outright. At turn 4 AR7 of ts-replayer game 63
                // the USSR has spent two rounds discarding to Bear Trap and plays Central
                // America Scoring on the last one; at turn 7 AR7 of game 237 the USSR lays
                // Quagmire and the US plays Africa Scoring on the very next round.
                uint8_t scoring_held = 0;
                for (uint8_t i = 1; i <= 110; ++i) {
                    if (in_hand_of(state.card_locations[i], p) && CardData::is_scoring_card(i)) {
                        scoring_held++;
                    }
                }
                uint8_t max_ar = (state.turn <= 3) ? 6 : 7;
                const bool us_ar8 = state.has_flag(effect_bits::NORTH_SEA_OIL_ACTIVE) ||
                                    SpaceRace::has_space_station_ar8(state, Player::US);
                const bool ussr_ar8 = SpaceRace::has_space_station_ar8(state, Player::USSR);
                if (state.turn >= 4 && ((p == Player::US && us_ar8) ||
                                        (p == Player::USSR && ussr_ar8))) {
                    max_ar = 8;
                }
                const uint8_t rounds_left = (state.action_round <= max_ar)
                    ? static_cast<uint8_t>(max_ar - state.action_round + 1) : 0;
                if (!has_2ops || scoring_held >= rounds_left) {
                    for (uint8_t i = 1; i <= 110; ++i) {
                        if (in_hand_of(state.card_locations[i], p) && CardData::is_scoring_card(i)) {
                            mask_out[i] = 1;
                        }
                    }
                }
                bool any = false;
                for (size_t i = 0; i < 112; ++i) if (mask_out[i]) { any = true; break; }
                if (!any) mask_out[0] = 1;
                return;
            }

            // 5. Standard Hand Selection
            for (uint8_t i = 1; i <= 110; ++i) {
                if (i == card_ids::THE_CHINA_CARD) continue; // China card handled below
                if (state.current_phase == Phase::HEADLINE && i == card_ids::UN_INTERVENTION) continue;
                if (in_hand_of(state.card_locations[i], p)) {
                    mask_out[i] = 1;
                }
            }

            // A player holding cards must play one; passing is only for a player who has run
            // out. The China Card is the exception: it is not part of the hand a player is
            // required to spend, so holding it and nothing else is still a choice between
            // playing it and passing the action round.
            bool holds_a_card = false;
            for (size_t i = 1; i <= 110; ++i) if (mask_out[i]) { holds_a_card = true; break; }
            if (!holds_a_card) mask_out[0] = 1;

            // The eighth action round is the other exception. It is not part of the turn: it
            // is granted by North Sea Oil or a Space Station to the one player who earned it,
            // and it is theirs to take or to leave. At turn 9 AR8 of ts-replayer game 16 the
            // USSR has it and declines -- the log records "Turn 9, USSR AR8:" and no card.
            if (state.current_phase == Phase::ACTION_ROUND && state.action_round == 8) {
                mask_out[0] = 1;
            }

            // The China Card
            if (state.china_card_holder == p && state.china_card_playable && state.current_phase != Phase::HEADLINE) {
                mask_out[card_ids::THE_CHINA_CARD] = 1;
            }
            break;
        }

        case DecisionType::SELECT_PLAY_MODE: {
            // P17: the merged resolution node. Five options, not four -- see `Resolution`.
            //
            // Ops legality follows the engine's own mechanics rather than a uniform rule.
            // Placement and realignment are REPEATABLE actions with a stop, so choosing one with
            // nothing to do resolves to "stop immediately" and they are offered unconditionally.
            // A coup is a SINGLE action with no stop: once chosen it must happen, so it is gated
            // on a target existing. That is the one lookahead this node does.
            *out_size = static_cast<size_t>(Resolution::COUNT);
            std::memset(mask_out, 0, static_cast<size_t>(Resolution::COUNT));
            uint8_t card = ctx.pending_op_card;

            // Ops options, shared by every branch below that allows Ops play at all.
            auto set_ops_options = [&]() {
                mask_out[static_cast<size_t>(Resolution::OPS_INFLUENCE)] = 1;
                mask_out[static_cast<size_t>(Resolution::OPS_REALIGN)] = 1;
                uint8_t coup_mask[84];
                Operations::get_coup_target_mask(state, p, coup_mask);
                for (uint8_t i = 0; i < 84; ++i) {
                    if (coup_mask[i]) {
                        mask_out[static_cast<size_t>(Resolution::OPS_COUP)] = 1;
                        break;
                    }
                }
            };

            if (CardData::is_scoring_card(card)) {
                mask_out[static_cast<size_t>(Resolution::EVENT)] = 1;
                return;
            }

            // The China Card has no Event of its own, so Operations or the Space Race. Racing
            // with it is a poor play -- it is the best Ops card in the game and passes to the
            // opponent either way -- but it is a legal one, and at turn 10 AR4 of ts-replayer
            // game 247 the US does exactly that, to box 5.
            if (card == card_ids::THE_CHINA_CARD) {
                set_ops_options();
                if (SpaceRace::can_attempt_space(state, p, card)) {
                    mask_out[static_cast<size_t>(Resolution::SPACE)] = 1;
                }
                return;
            }

            // Forced play (Missile Envy recipient must play it for Operations). Three tests,
            // and each rules out a defect this branch used to have:
            //   ACTION_ROUND      -- the card says "on their next action round", so a headline is
            //                        unconstrained and its card is played as its Event as usual;
            //   MISSILE_ENVY held -- the obligation is live only while the card is still in hand,
            //                        so a stale forced_card_id restricts nothing;
            //   card == MISSILE_ENVY -- the restriction is on THAT card. `forced_card_id` only
            //                        ever holds MISSILE_ENVY, so the old disjunct
            //                        `forced_card_id == card || forced_card_id == MISSILE_ENVY`
            //                        was always true and read as "every card must go to Ops".
            // The last one is also what makes ordering irrelevant: a scoring card and the China
            // Card are never card 49, so their branches above can no longer be overtaken.
            if (state.current_phase == Phase::ACTION_ROUND &&
                state.forced_card_player == p &&
                card == card_ids::MISSILE_ENVY &&
                in_hand_of(state.card_locations[card_ids::MISSILE_ENVY], p)) {
                set_ops_options();
                return;
            }

            // Defectors cannot be played as an event by US during Action Round
            if (card == card_ids::DEFECTORS && p == Player::US && state.current_phase == Phase::ACTION_ROUND) {
                set_ops_options();
                if (SpaceRace::can_attempt_space(state, p, card)) {
                    mask_out[static_cast<size_t>(Resolution::SPACE)] = 1;
                }
                return;
            }

            // EVENT means "the event resolves now". For a friendly or neutral card that is the
            // whole play. For an OPPONENT's card it is event-first: the event fires and the Ops
            // choice is deferred until afterwards, which is why it is legal exactly when Ops play
            // is -- the old CHOOSE_TIMING_BRANCH offered EVENT_FIRST unconditionally there.
            if (CardData::is_opponent_card(card, p)) {
                mask_out[static_cast<size_t>(Resolution::EVENT)] = 1;
            } else if (CardHandlers::can_trigger_event(state, card, p)) {
                mask_out[static_cast<size_t>(Resolution::EVENT)] = 1;
            }

            // Ops play: always legal for non-scoring cards. Which Ops MODES are legal is the
            // gating above -- coup needs a target, the other two do not.
            set_ops_options();

            // Space play: legal if prerequisites are met
            if (SpaceRace::can_attempt_space(state, p, card)) {
                mask_out[static_cast<size_t>(Resolution::SPACE)] = 1;
            }
            break;
        }

        case DecisionType::CHOOSE_TIMING_BRANCH: {
            // P17: retired. The merged resolution node carries the timing -- EVENT on an
            // opponent's card IS event-first, any OPS_* on one is ops-first -- so this node no
            // longer occurs. Emitting nothing means an unexpected arrival is refused loudly
            // instead of being handed a choice that no longer means anything.
            *out_size = 0;
            break;
        }

        case DecisionType::SELECT_OP_MODE: {
            *out_size = 3;
            std::memset(mask_out, 0, 3);
            uint8_t ops = ctx.pending_ops_value;
            uint8_t op_card = ctx.pending_op_card;

            // Junta and Tear Down This Wall grant free Ops usable only for a coup or a
            // realignment, and their bonus action is optional, so INFLUENCE stands for
            // declining it. That restriction belongs to the Ops the event granted, not to the
            // card: the player whose card it is still has their own Ops afterwards and may
            // place Influence with them. Keying it on the card alone left those Ops with
            // nothing but a decline on offer -- at turn 9 AR1 of ts-replayer game 146 the USSR
            // could not spend Tear Down This Wall's own three Ops at all.
            const bool free_action_bars_influence =
                ctx.event_granted_ops
                && (op_card == card_ids::JUNTA || op_card == card_ids::TEAR_DOWN_THIS_WALL);
            if (!free_action_bars_influence) {
                uint8_t inf_mask[84];
                Operations::get_influence_placement_mask(state, p, ops, inf_mask);
                for (uint8_t i = 0; i < 84; ++i) {
                    if (inf_mask[i]) {
                        mask_out[static_cast<size_t>(OpMode::INFLUENCE)] = 1;
                        break;
                    }
                }
            }
            // P17: when the free action bars Influence, INFLUENCE is simply NOT legal. It used to
            // be set here to stand for "decline the free action", which is what gave index 116
            // three meanings and cost three engine bugs (ts-replayer games 141, 146, 105). The
            // decline is the shared index, added by the flat layer below.

            // KAL-007 and Glasnost restrict the *free action their event grants*, not the card
            // itself: played for Ops, either allows a coup like any other card. Barring it
            // whenever the card was the Ops source conflated the two, and at turn 8 AR1 of
            // ts-replayer game 105 the US played Glasnost for its 4 Ops and couped Mexico --
            // which the engine could not do, so the Ops went into influence instead.
            const bool free_action_bars_coup =
                state.ctx().event_granted_ops
                && (op_card == card_ids::SOVIETS_SHOOT_DOWN_KAL_007 || op_card == card_ids::GLASNOST);
            if (!free_action_bars_coup) {
                uint8_t coup_mask[84];
                Operations::get_coup_target_mask(state, p, coup_mask);
                for (uint8_t i = 0; i < 84; ++i) {
                    if (coup_mask[i]) {
                        if (free_action::region_locked(op_card, ctx.event_granted_ops)) {
                            const auto& c = MapData::get_country(i);
                            if (op_card == card_ids::JUNTA) {
                                if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) continue;
                            } else if (c.region != Region::EUROPE) {
                                continue;
                            }
                        }
                        mask_out[static_cast<size_t>(OpMode::COUP)] = 1;
                        break;
                    }
                }
            }

            // Check if realign is possible
            uint8_t realign_mask[84];
            Operations::get_realign_target_mask(state, p, realign_mask);
            for (uint8_t i = 0; i < 84; ++i) {
                if (realign_mask[i]) {
                    if (free_action::region_locked(op_card, ctx.event_granted_ops)) {
                        const auto& c = MapData::get_country(i);
                        if (op_card == card_ids::JUNTA) {
                            if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) continue;
                        } else if (c.region != Region::EUROPE) {
                            continue;
                        }
                    }
                    mask_out[static_cast<size_t>(OpMode::REALIGN)] = 1;
                    break;
                }
            }
            // P17: no fallback here. This used to set INFLUENCE(0) to mean "skip", which is the
            // third meaning that index carried -- a real placement, "nothing is legal", and
            // "decline this event's free bonus". An all-zero mask here is answered by the flat
            // layer with the shared decline index, which means exactly one thing.
            break;
        }

        case DecisionType::POINT_NODE: {
            *out_size = 84;
            std::memset(mask_out, 0, 84);

            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                *out_size = 84;
                return;
            }

            // Setup Phase placement
            if (state.current_phase == Phase::SETUP) {
                if (p == Player::USSR) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        if (MapData::get_country(i).in_eastern_europe) mask_out[i] = 1;
                    }
                } else if (p == Player::US) {
                    if (ctx.pending_ops_value == 0) {
                        // Stage 0: 7 Western Europe placements
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (MapData::get_country(i).in_western_europe) mask_out[i] = 1;
                        }
                    } else {
                        // Stage 1: 2 Bonus placements in countries with existing US presence
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (state.countries[i].us_influence > 0) mask_out[i] = 1;
                        }
                    }
                }
                return;
            }

            // Standard Op Mode Node Selection
            if (ctx.op_mode == OpMode::COUP) {
                Operations::get_coup_target_mask(state, p, mask_out);
                if (free_action::region_locked(ctx.pending_op_card, ctx.event_granted_ops)) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        const auto& c = MapData::get_country(i);
                        if (ctx.pending_op_card == card_ids::JUNTA) {
                            if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) mask_out[i] = 0;
                        } else if (c.region != Region::EUROPE) {
                            mask_out[i] = 0;
                        }
                    }
                }
            } else if (ctx.op_mode == OpMode::REALIGN) {
                Operations::get_realign_target_mask(state, p, mask_out);
                if (free_action::region_locked(ctx.pending_op_card, ctx.event_granted_ops)) {
                    for (uint8_t i = 0; i < 84; ++i) {
                        const auto& c = MapData::get_country(i);
                        if (ctx.pending_op_card == card_ids::JUNTA) {
                            if (c.region != Region::CENTRAL_AMERICA && c.region != Region::SOUTH_AMERICA) mask_out[i] = 0;
                        } else if (c.region != Region::EUROPE) {
                            mask_out[i] = 0;
                        }
                    }
                }
            } else {
                if (ctx.remaining_steps > 0) {
                    Operations::get_influence_placement_mask(state, p, ctx.remaining_steps, mask_out);
                    uint8_t op_card = ctx.pending_op_card;
                    uint8_t total_spent = ctx.pending_ops_value - ctx.remaining_steps;
                    if (op_card == card_ids::THE_CHINA_CARD) {
                        uint8_t non_asia_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (mask_out[i] && MapData::get_country(i).region != Region::ASIA) {
                                uint8_t cost = Operations::get_influence_cost(state, p, i);
                                if (total_spent + cost > non_asia_base) {
                                    mask_out[i] = 0;
                                }
                            }
                        }
                    } else if (p == Player::USSR && state.has_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE)) {
                        uint8_t non_se_base = Operations::get_effective_ops(state, op_card, p, Region::NONE_REGION);
                        for (uint8_t i = 0; i < 84; ++i) {
                            if (mask_out[i] && !MapData::get_country(i).in_southeast_asia) {
                                uint8_t cost = Operations::get_influence_cost(state, p, i);
                                if (total_spent + cost > non_se_base) {
                                    mask_out[i] = 0;
                                }
                            }
                        }
                    }
                }
            }
            *out_size = 84;
            break;
        }

        case DecisionType::CHOOSE_BRANCH: {
            *out_size = 8;
            std::memset(mask_out, 0, 8);
            if (ctx.resolving_card != 0) {
                CardHandlers::get_event_action_mask(state, mask_out, out_size);
                bool any_legal = false;
                for (size_t i = 0; i < *out_size; ++i) {
                    if (mask_out[i]) { any_legal = true; break; }
                }
                if (!any_legal) {
                    mask_out[0] = 1;
                }
            }
            // No else. A CHOOSE_BRANCH outside an event frame offered branch 0 and branch 1, and
            // `StateMachine::step` has no case for it -- that type is handled only inside the
            // `resolving_card != 0` block -- so both were refused and the position could not
            // advance. Every setter of CHOOSE_BRANCH is in a card handler that also sets
            // `resolving_card`, so the branch was dead; 7,745 CHOOSE_BRANCH nodes sampled from
            // self-play were all inside an event frame.
            break;
        }

        case DecisionType::ROLL_DIE: {
            *out_size = 7;
            std::memset(mask_out, 0, 7);
            mask_out[0] = 1; // 0 = Auto-roll
            for (uint8_t r = 1; r <= 6; ++r) mask_out[r] = 1; // 1..6 = Forced roll
            break;
        }
    }
}


void ActionMask::generate_flat_mask_212(const GameState& state, uint8_t* mask_212) noexcept {
    if (!mask_212) return;
    std::memset(mask_212, 0, FLAT_ACTION_SPACE_SIZE);

    if (state.current_phase == Phase::GAME_OVER || state.victory_points >= 20 || state.victory_points <= -20) {
        return;
    }

    uint8_t temp_mask[128];
    size_t temp_size = 0;
    generate_mask(state, temp_mask, &temp_size);

    const auto& ctx = state.ctx();
    switch (ctx.decision_type) {
        case DecisionType::NONE:
            // Not a decision, so nothing is legal -- `step` has no case for it either.
            break;

        case DecisionType::SELECT_CARD:
            for (size_t i = 1; i <= 110 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[i - 1] = 1;
                }
            }
            if (temp_mask[0]) {
                mask_212[211] = 1; // Pass / early stop
            }
            break;

        case DecisionType::SELECT_PLAY_MODE:
            // P17: the merged resolution node, five slots [110..114].
            for (size_t i = 0; i < static_cast<size_t>(Resolution::COUNT) && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[110 + i] = 1;
                }
            }
            break;

        case DecisionType::CHOOSE_TIMING_BRANCH:
            // P17: unreachable. The merged node carries the timing choice -- EVENT on an
            // opponent's card IS event-first, and any OPS_* on one is ops-first. Kept as a case
            // so an unexpected arrival emits an empty mask and is refused loudly, rather than
            // falling through to something that looks legal.
            break;

        case DecisionType::SELECT_OP_MODE:
            // The deferred Ops choice after an event-first event. Reuses the resolution node's
            // three OPS_* slots [112..114]: the same question reached by another route, so the
            // head sees one concept rather than two.
            for (size_t i = 0; i < 3 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[112 + i] = 1;
                }
            }
            // The decline, explicitly rather than borrowed. Two cases need it: nothing is
            // spendable at all, or the Ops came from an event whose bonus action is optional
            // (Junta, Tear Down This Wall -- `region_locked` is exactly that predicate).
            if ((!mask_212[112] && !mask_212[113] && !mask_212[114]) ||
                free_action::region_locked(ctx.pending_op_card, ctx.event_granted_ops)) {
                mask_212[211] = 1;
            }
            break;

        case DecisionType::POINT_NODE: {
            bool any_target = false;
            for (size_t i = 0; i < 84 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[119 + i] = 1;
                    any_target = true;
                }
            }
            if (ctx.allow_early_stop) {
                mask_212[211] = 1;
            } else if (!any_target) {
                // A choice the board cannot supply a single legal target for. The mask would
                // otherwise be empty and the game would sit on this decision until something
                // else timed out, far from the cause, so the player is let out either way.
                //
                // Whether that is worth hearing about depends on the card. A few may
                // legitimately come up empty and fizzle quietly; for every other card this is
                // a defect, and the whole position is reported so it can be replayed rather
                // than guessed at. Keeping the list explicit is what keeps the report worth
                // reading.
                if (!may_fizzle::allowed(ctx.resolving_card)) {
                    report_anomaly("POINT_NODE with no legal target and no early stop", state);
                }
                mask_212[211] = 1;
            }
            break;
        }

        case DecisionType::CHOOSE_BRANCH:
            for (size_t i = 0; i < 8 && i < temp_size; ++i) {
                if (temp_mask[i]) {
                    mask_212[203 + i] = 1;
                }
            }
            if (ctx.allow_early_stop) {
                mask_212[211] = 1;
            }
            break;

        case DecisionType::ROLL_DIE:
            // P17: its own index. A chance node with one legal action is not a decline, and
            // sharing 211 was the same aliasing being removed elsewhere -- cheaper only because
            // the decode already special-cased it. 211 now means exactly one thing.
            mask_212[115] = 1;
            break;
    }

    // No blanket fallback. There used to be one here -- "ensure at least one action is legal" --
    // which set flat 211 whenever a case emitted nothing. It is what turned a missing ROLL_DIE
    // case into a die roll of 255 (`9f78026`), and it would silently re-add an action to exactly
    // the three cases above that deliberately emit none: NONE, a CHOOSE_BRANCH outside an event
    // frame, and a node with no decision player.
    //
    // An empty mask now means what it says. `StateMachine::step` validates against this mask, so
    // a position that offers nothing refuses everything -- loudly, at the point of the mistake,
    // rather than by inventing a move.
}

MicroAction ActionMask::decode_flat_action_212(const GameState& state, uint16_t action_idx) noexcept {
    const auto& ctx = state.ctx();

    // P17: ROLL_DIE has its own index now. primary_id carries the forced die value, so 0 means
    // "roll normally" (ops.cpp reads `forced_roll > 0`) and the 255 sentinel used for
    // confirm/done would have been read as a forced roll of 255.
    if (action_idx == 115) {
        return MicroAction{DecisionType::ROLL_DIE, 0, 0, 0};
    }

    if (action_idx == 211) {
        if (ctx.decision_type == DecisionType::SELECT_CARD) {
            // primary_id stays 0, which every card sub-decision already accepts as "decline",
            // but the flag has to be set too: without it is_confirm_done() reported false for
            // the one action that means exactly that, so a caller inspecting the decoded
            // action read a pass as a request to discard card 0.
            return MicroAction{DecisionType::SELECT_CARD, 0, 0, action_flags::CONFIRM_DONE};
        }
        return MicroAction{ctx.decision_type, 255, 0, action_flags::CONFIRM_DONE};
    }

    if (action_idx < 110) {
        return MicroAction{DecisionType::SELECT_CARD, static_cast<uint8_t>(action_idx + 1), 0, 0};
    }
    // [110..114] are the merged resolution node. [112..114] do double duty as the deferred Ops
    // choice, so the decision type decides which of the two this is -- the same slot means
    // "spend Ops on X" either way, which is why they share.
    if (action_idx < 115) {
        if (ctx.decision_type == DecisionType::SELECT_OP_MODE && action_idx >= 112) {
            return MicroAction{DecisionType::SELECT_OP_MODE,
                               static_cast<uint8_t>(action_idx - 112), 0, 0};
        }
        return MicroAction{DecisionType::SELECT_PLAY_MODE, static_cast<uint8_t>(action_idx - 110), 0, 0};
    }
    if (action_idx < 119) {
        // [116..118] are unassigned after the merge. Nothing should arrive here; refuse rather
        // than decode to a neighbouring meaning.
        return MicroAction{DecisionType::NONE, 255, 0, 0};
    }
    if (action_idx < 203) {
        return MicroAction{DecisionType::POINT_NODE, static_cast<uint8_t>(action_idx - 119), 0, 0};
    }
    if (action_idx < 211) {
        return MicroAction{DecisionType::CHOOSE_BRANCH, static_cast<uint8_t>(action_idx - 203), 0, 0};
    }

    return MicroAction{ctx.decision_type, 255, 0, action_flags::CONFIRM_DONE};
}

int16_t ActionMask::encode_micro_action_212(const GameState& state, const MicroAction& action) noexcept {
    (void)state;
    if (action.is_confirm_done() || action.primary_id == 255) {
        return 211;
    }

    switch (action.decision_type) {
        case DecisionType::SELECT_CARD:
            if (action.primary_id == 0) return 211;
            if (action.primary_id >= 1 && action.primary_id <= 110) {
                return static_cast<int16_t>(action.primary_id - 1);
            }
            return 211;

        case DecisionType::SELECT_PLAY_MODE:
            if (action.primary_id < static_cast<uint8_t>(Resolution::COUNT)) {
                return static_cast<int16_t>(110 + action.primary_id);
            }
            return 211;

        case DecisionType::CHOOSE_TIMING_BRANCH:
            // P17: retired. No slot encodes it; the merged node carries the timing.
            return 211;

        case DecisionType::SELECT_OP_MODE:
            // Shares the resolution node's OPS_* slots.
            if (action.primary_id < 3) return static_cast<int16_t>(112 + action.primary_id);
            return 211;

        case DecisionType::ROLL_DIE:
            return 115;

        case DecisionType::POINT_NODE:
            if (action.primary_id < 84) return static_cast<int16_t>(119 + action.primary_id);
            return 211;

        case DecisionType::CHOOSE_BRANCH:
            if (action.primary_id < 8) return static_cast<int16_t>(203 + action.primary_id);
            return 211;

        case DecisionType::NONE:
        default:
            return 211;
    }
}

} // namespace ts
