"""Game length in *plies*: a continuous index over the player-slots of a whole game.

`turn` and `round` are both taken -- a turn is one of the ten, an action round is a
turn-relative index shared by both players -- so the continuous count needs its own word. A
**ply** is one player's single opportunity to act: one headline, or one action round for one
side. The numeration runs from the start of the game:

    ply  1  turn 1 first headline       ply  3  USSR turn 1 AR1
    ply  2  turn 1 second headline      ply  4  US   turn 1 AR1
                                        ...
                                        ply 14  US   turn 1 AR6
    ply 15  turn 2 first headline

The two headline plies are numbered by **resolution order, not by side**. Both players reveal
simultaneously and the higher-ops card resolves first, so the US headline is the turn's first ply
whenever the US played the bigger card. Numbering them USSR-then-US instead made the index run
backwards inside a headline (30 -> 29 at turn 3), which is why `headline_stage` is required
rather than inferred from `is_us`. Every action round still alternates USSR then US, because
`advance_after_action_round` genuinely does.

This is the engine's own schedule, not a re-derivation of it: `advance_after_action_round`
alternates `phasing_player` USSR -> US and increments `action_round` when the US finishes, over
`max_ar = (turn <= 3) ? 6 : 7`. So a turn is 2 headline plies plus 2 per action round: 14 plies
in the early war, 16 from turn 4, and 154 for a game that plays all ten turns out.

Why not just count turns. The turn counter answers "which of the ten" and nothing finer, so a
game abandoned at turn 7 AR1 and one that runs to turn 7 AR7 are the same number. It also has a
reporting artifact at the end: `finish_end_turn` increments past 10 *before* testing `turn <= 10`,
so a game that goes to final scoring terminates holding `turn == 11` while a human log numbers
that same game turn 10. Plies have neither problem -- a completed game is 154 on both sides.

The eighth action round. North Sea Oil and a Space Station can grant a turn an AR8, which adds
plies to that turn for the side entitled to it. This module counts the *nominal* schedule and
does not model it: an AR8 slot is numbered 17/18 within its turn, but earlier AR8s do not shift
later turns. A game containing one is therefore under-counted by 2 per occurrence. That keeps a
ply index comparable across games, which is the point of it, and the case is rare -- 36 of the
~26,800 action-round entries in the human corpus, about 0.1%. Report `ar8_games` alongside any
comparison rather than silently absorbing it.

This module sits at the top of `ai/` rather than in `ai/eval/` deliberately: it is imported by
`bindings/ts_env.py` on the training path, and `ai/eval/__init__` pulls in torch and the model
zoo, which that module goes out of its way to keep out of its import chain.
"""

from typing import Final

# Action rounds per turn, by the engine's `max_ar`. Index 0 is unused so `turn` indexes directly.
_ARS_PER_TURN: Final[tuple[int, ...]] = (0, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7)

FIRST_TURN: Final[int] = 1
LAST_TURN: Final[int] = 10

#: Plies in a game that plays all ten turns out, and so the ply at which final scoring happens.
FULL_GAME_PLIES: Final[int] = 154


def plies_in_turn(turn: int) -> int:
    """The nominal ply count of one turn: two headline plies plus two per action round."""
    if not FIRST_TURN <= turn <= LAST_TURN:
        raise ValueError(f"turn {turn} is outside 1..10")
    return 2 + 2 * _ARS_PER_TURN[turn]


def plies_before_turn(turn: int) -> int:
    """Plies completed before `turn` begins. `plies_before_turn(11)` is the whole game."""
    if not FIRST_TURN <= turn <= LAST_TURN + 1:
        raise ValueError(f"turn {turn} is outside 1..11")
    return sum(plies_in_turn(t) for t in range(FIRST_TURN, turn))


def ply(turn: int, action_round: int, is_us: bool,
        headline_stage: int | None = None) -> int:
    """The ply index of one player-slot.

    `action_round` follows the engine: 0 during the headline, otherwise 1..8. In an action round
    `is_us` picks the second of the pair, since the USSR phases first in every one of them.

    In a headline it does not, and `headline_stage` is **required** there: the engine resolves
    the two headline cards in descending Ops, so which side goes first is a property of the cards
    played, not of the side. `state.headline_stage` is 0 while both players are still choosing,
    1 while the first card resolves and 2 while the second does; 0 and 1 are the turn's first
    headline ply, 2 is its second.

    Passing `is_us` alone for a headline is refused rather than guessed. It is the whole of the
    bug this parameter fixes, and a wrong ply here is invisible -- it returns a plausible number.

    `turn == 11` is the post-final-scoring state the engine terminates in; it has no slot of its
    own and maps to the last ply actually played.
    """
    if turn == LAST_TURN + 1:
        return FULL_GAME_PLIES
    if action_round == 0:
        if headline_stage is None:
            raise ValueError(
                "ply() needs headline_stage when action_round is 0: headlines resolve in "
                "descending Ops, so is_us does not order them. Pass state.headline_stage.")
        return plies_before_turn(turn) + 1 + (1 if headline_stage >= 2 else 0)
    return plies_before_turn(turn) + (2 * action_round + 1) + (1 if is_us else 0)
