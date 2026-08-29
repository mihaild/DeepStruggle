#pragma once

#include "ts/constants.hpp"
#include "ts/game_state.hpp"
#include "ts/types.hpp"

namespace ts {

// Resolves a DEFCON-1 thermonuclear war game end. The phasing player always loses,
// regardless of who drove DEFCON down.
//
// `cause_owner` is the player the triggering action belongs to: for a card event, the
// side the event benefits (which is what `CardHandlers::trigger_event` is handed); for a
// coup or a direct DEFCON manipulation, the player performing it.
//
// DEFCON_SUICIDE_PROVOKED records whether the loss was forced rather than self-inflicted.
// It is set when the cause belongs to the phasing player's opponent, which is the case
// when the phasing player plays an opponent-associated card for Operations and is
// obliged to trigger its event. That is a legitimate strategic squeeze: a player can run
// out of safe cards, and even strong players are sometimes forced into it. The flag is
// left clear when the phasing player chose the action themselves -- their own or a
// neutral event, or a battleground coup at DEFCON 2 -- which is an entirely avoidable
// blunder.
//
// Only the classification depends on `cause_owner`; the game outcome is identical either
// way. Analytics (`classify_game_ending_reason`) and blunder-aware reward shaping rely on
// the distinction.
inline void resolve_defcon_one_loss(GameState& state, Player cause_owner) noexcept {
    const Player loser = state.phasing_player;
    if (cause_owner != Player::NONE && loser != Player::NONE && cause_owner != loser) {
        state.set_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);
    }
    state.victory_points = (loser == Player::US) ? -20 : 20;
    state.current_phase = Phase::GAME_OVER;
}

} // namespace ts
