"""Game length in *plies*: a continuous index over the player-slots of a whole game.

`turn` and `round` are both taken -- a turn is one of the ten, an action round is a
turn-relative index shared by both players -- so the continuous count needs its own word. A
**ply** is one player's single opportunity to act: one headline, or one action round for one
side. The numeration runs from the start of the game:

    ply  1  USSR turn 1 headline        ply  3  USSR turn 1 AR1
    ply  2  US   turn 1 headline        ply  4  US   turn 1 AR1
                                        ...
                                        ply 14  US   turn 1 AR6
    ply 15  USSR turn 2 headline

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


def ply(turn: int, action_round: int, is_us: bool) -> int:
    """The ply index of one player-slot.

    `action_round` follows the engine: 0 during the headline, otherwise 1..8. `is_us` picks the
    second of the pair, since the USSR phases first in every action round.

    `turn == 11` is the post-final-scoring state the engine terminates in; it has no slot of its
    own and maps to the last ply actually played.
    """
    if turn == LAST_TURN + 1:
        return FULL_GAME_PLIES
    within = 1 if action_round == 0 else 2 * action_round + 1
    return plies_before_turn(turn) + within + (1 if is_us else 0)
