"""Named opening setups that can be forced on an agent in place of its own placements.

A trained policy fights where it was placed and rarely opens a new front, so its own setup
decides much of what the rest of the game can look like. Forcing a known opening is how that gets
separated: same agent, same engine, different starting board.

One definition, used by both the probe that measures the effect (`ai/eval/forced_setup.py`) and
the replay generators that produce games to watch (`tools/lib/self_play.py`,
`tools/play_match.py`). If these drifted, a replay labelled "the human opening" would not be the
opening the numbers were measured on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle

#: Flat action index of country `cid` at a POINT_NODE (`bindings/action_encoder.py`).
NODE_OFFSET = ActionEncoder.NODE_OFFSET

#: 6 USSR placements, then 7 US in Western Europe, then 2 US bonus.
SETUP_DECISIONS = 15

WEST_GERMANY, ITALY, EAST_GERMANY, POLAND, YUGOSLAVIA, IRAN = 7, 10, 14, 15, 18, 25
CANADA, FRANCE, HUNGARY, SOUTH_KOREA = 0, 8, 17, 44
AUSTRIA = 13
ROMANIA, AUSTRALIA = 19, 41

#: The standard human opening, as (country, points) in placement order. The USSR overcontrols
#: Poland and East Germany by a point each; the US takes West Germany and Italy with a buffer and
#: spends its two bonus points on Iran.
USSR_OPENING: Sequence[Tuple[int, int]] = ((EAST_GERMANY, 1), (POLAND, 4), (YUGOSLAVIA, 1))
US_OPENING: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 4), (ITALY, 3), (IRAN, 2))


#: The owner's variants of the human opening (2026-10-02), final influence in brackets. USSR: East
#: Germany +1 (4), Poland +4, and the last point in Yugoslavia or Austria. US: West Germany 4 and
#: Italy 3 in Western Europe, then the two bonus points either both in Iran (Iran 3, Italy 3) or one
#: in Italy and one in Iran (Italy 4, Iran 2).
USSR_OPENING_AUSTRIA: Sequence[Tuple[int, int]] = ((EAST_GERMANY, 1), (POLAND, 4), (AUSTRIA, 1))
US_OPENING_ITALY4: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 4), (ITALY, 3), (ITALY, 1), (IRAN, 1))


#: A deliberately bad opening, as a control for --setup-script-frac (owner, 2026-10-02): does the
#: mechanism make the network adopt whatever is forced, or only what wins? USSR all six in Romania;
#: US all seven Western Europe points in West Germany and both bonus points in Australia.
USSR_STUPID: Sequence[Tuple[int, int]] = ((ROMANIA, 6),)
US_STUPID: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 7), (AUSTRALIA, 2))


def expand(opening: Sequence[Tuple[int, int]]) -> List[int]:
    """(country, points) pairs to one country id per influence point, in placement order."""
    return [cid for cid, n in opening for _ in range(n)]


#: A symmetric USSR opening that commits nothing to East Germany, so the two US setups below face
#: the same board and differ only in whether the US holds West Germany.
USSR_POLAND_HUNGARY: Sequence[Tuple[int, int]] = ((POLAND, 3), (HUNGARY, 3))
FINLAND = 5
#: Poland 3 / Finland 3: no route to Yugoslavia (P44's control from 8,000M).
USSR_POLAND_FINLAND: Sequence[Tuple[int, int]] = ((POLAND, 3), (FINLAND, 3))
#: The setup M2d s3 settled on from 185M: West Germany left empty, Canada taken instead.
US_NO_WEST_GERMANY: Sequence[Tuple[int, int]] = (
    (CANADA, 2), (ITALY, 2), (FRANCE, 3), (IRAN, 1), (SOUTH_KOREA, 1))

PANAMA = 70

#: P44 (owner, 2026-11-13): J1's US setups either side of its dropping West Germany (~8,000 -> 8,400M), seven Western
#: Europe points first, then the two bonus points: France 3 / West Germany 3 / Italy 2 / Iran 1 (its 8,000M mode) and
#: France 3 / Italy 3 / Canada 2 / Iran 1 (from 8,400M to 15,200M).
US_FR3_WG3_IT2_IR1: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 3), (FRANCE, 3), (ITALY, 1), (ITALY, 1), (IRAN, 1))
US_FR3_IT3_CA2_IR1: Sequence[Tuple[int, int]] = ((CANADA, 2), (FRANCE, 3), (ITALY, 2), (ITALY, 1), (IRAN, 1))

#: The US openings E5-16 settled on (research/log/E5_16_setup_mc_credit.md), in placement order:
#: seven Western Europe points, then the two bonus points where the US already has influence.
US_E516_43: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 4), (FRANCE, 2), (ITALY, 1), (ITALY, 1), (IRAN, 1))
US_E516_44: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 2), (FRANCE, 3), (ITALY, 2), (IRAN, 1), (PANAMA, 1))

#: The owner's training openings (2026-10-13), one side each, for --forced-openings (the network plays
#: on from them; they are not taught as its own choice). US: West Germany 4 / Italy 3 / Iran 2, West
#: Germany 4 / Italy 4 / Iran 1, and -- with Marshall Plan in hand -- West Germany 3 / Italy 2 / France 2
#: / Iran 2 with Marshall Plan headlined on turn 1 (`US_MARSHALL`, chosen at the game's setup by
#: `ai/training/forced_openings.py`). Seven Western Europe points first, then the two bonus points.
US_MARSHALL: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 3), (ITALY, 2), (FRANCE, 2), (IRAN, 2))
MARSHALL_PLAN = 23

#: A name maps to a script per side. A side with no script sets up for itself -- one-sided
#: openings exist for evaluation (the other side replying as the agent would); training's
#: --forced-opening needs both sides.
OPENINGS: Dict[str, Dict[str, List[int]]] = {
    "human": {"US": expand(US_OPENING), "USSR": expand(USSR_OPENING)},
    "human_yugo_it4": {"US": expand(US_OPENING_ITALY4), "USSR": expand(USSR_OPENING)},
    "human_austria_it3": {"US": expand(US_OPENING), "USSR": expand(USSR_OPENING_AUSTRIA)},
    "human_austria_it4": {"US": expand(US_OPENING_ITALY4), "USSR": expand(USSR_OPENING_AUSTRIA)},
    "stupid_romania_australia": {"US": expand(US_STUPID), "USSR": expand(USSR_STUPID)},
    "us_e516_43": {"US": expand(US_E516_43)},
    "us_e516_44": {"US": expand(US_E516_44)},
    "ph_west_germany": {"US": expand(US_OPENING), "USSR": expand(USSR_POLAND_HUNGARY)},
    "ph_no_west_germany": {"US": expand(US_NO_WEST_GERMANY), "USSR": expand(USSR_POLAND_HUNGARY)},
    # One side each: the owner's training openings (--forced-openings), and for evaluation.
    "us_wg4_it3_ir2": {"US": expand(US_OPENING)},
    "us_wg4_it4_ir1": {"US": expand(US_OPENING_ITALY4)},
    "us_marshall": {"US": expand(US_MARSHALL)},
    "ussr_eg1_pl4_au1": {"USSR": expand(USSR_OPENING_AUSTRIA)},
    "ussr_eg1_pl4_yu1": {"USSR": expand(USSR_OPENING)},
    # P44 (owner, 2026-11-13): the USSR setup step 1's control drifted to by 8,000M, forced on its earlier self.
    "ussr_pl3_fi3": {"USSR": expand(USSR_POLAND_FINLAND)},
    "us_fr3_wg3_it2_ir1": {"US": expand(US_FR3_WG3_IT2_IR1)},
    "us_fr3_it3_ca2_ir1": {"US": expand(US_FR3_IT3_CA2_IR1)},
}

#: The owner's --forced-openings sets per side. `us_marshall` is conditional: with Marshall Plan
#: in the US hand at setup it is played and Marshall Plan headlined on turn 1; without, one of the
#: other two US openings is played instead.
FORCED_OPENINGS_US: Tuple[str, ...] = ("us_wg4_it3_ir2", "us_wg4_it4_ir1", "us_marshall")
FORCED_OPENINGS_USSR: Tuple[str, ...] = ("ussr_eg1_pl4_au1", "ussr_eg1_pl4_yu1")


#: The four human-opening pairs --setup-script-frac draws from by default.
HUMAN_OPENING_MIX: Tuple[str, ...] = ("human", "human_yugo_it4", "human_austria_it3", "human_austria_it4")


def scripted_setup_index(state: ts.GameState, side: str, opening: str,
                         cursor: Dict[str, int]) -> Optional[int]:
    """The flat action index of `side`'s next scripted placement, or None if there is none.

    None means "let the agent decide": either setup is over, or this side's script is spent.
    An *illegal* scripted placement raises instead, because falling through to the agent would
    produce a partly-forced opening -- which is neither opening, and would be reported as one.

    `cursor` is per-side and mutated in place; pass `{"US": 0, "USSR": 0}` at the start of a game.
    """
    if state.current_phase != ts.Phase.SETUP or side not in OPENINGS[opening]:
        return None
    script = OPENINGS[opening][side]
    k = cursor[side]
    if k >= len(script):
        return None
    idx = NODE_OFFSET + script[k]
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    if not mask[idx]:
        raise RuntimeError(
            f"{side} placement {k} of the '{opening}' opening (country {script[k]}, flat action "
            f"{idx}) is not legal at this setup decision -- refusing to fall back to the agent.")
    cursor[side] = k + 1
    return idx


def acting_side(state: ts.GameState) -> str:
    """Which side is making this decision.

    The decision player, never `phasing_player`: the USSR is the phasing player for the whole of
    setup, so keying off it sends every US placement down the USSR script.
    """
    return "US" if state.ctx().decision_player == ts.Player.US else "USSR"


def play_scripted_setup(state: ts.GameState, opening: str) -> ts.GameState:
    """A copy of `state`, a fresh game in SETUP, with the whole setup played by `opening`'s script
    for both sides. Raises if the script does not cover the setup or a placement is illegal, rather
    than returning a partly scripted opening."""
    missing = {"US", "USSR"} - set(OPENINGS[opening])
    if missing:
        raise ValueError(f"the '{opening}' opening scripts only one side (no {sorted(missing)}); "
                         f"a scripted game start needs both")
    st = state.clone()
    cursor = {"US": 0, "USSR": 0}
    for _ in range(4 * SETUP_DECISIONS):
        if st.current_phase != ts.Phase.SETUP:
            return st
        if st.ctx().decision_player == ts.Player.NONE:
            settle(st, SettleMode.CHANCE)
            continue
        idx = scripted_setup_index(st, acting_side(st), opening, cursor)
        if idx is None:
            raise RuntimeError(f"the '{opening}' opening ran out before setup ended ({cursor})")
        ts.Engine.step_flat(st, idx, False)
        settle(st, SettleMode.CHANCE)
    raise RuntimeError("setup did not end")


class ScriptedSetupOverride:
    """Replaces an agent's setup placements with a named opening inside a batched game loop.

    One instance per batch of games. Each game keeps a cursor per side, so a side's k-th setup
    placement is the script's k-th point whichever agent is on the other side, and a scripted side
    can face a side that sets up for itself. Setup rows are recognised from the observation's phase
    slot (exactly 0 in SETUP), the test training uses."""

    def __init__(self, num_games: int) -> None:
        self.cursor = {"US": np.zeros(num_games, dtype=np.int32),
                       "USSR": np.zeros(num_games, dtype=np.int32)}

    def apply(self, actions: np.ndarray, obs: np.ndarray, masks: np.ndarray,
              d_players: np.ndarray, rows: np.ndarray, opening: str) -> None:
        """Overwrite `actions[rows]` in place wherever the row is a setup placement."""
        from ai.training.rollout_buffer import setup_phase_slot

        rows = np.asarray(rows)
        if len(rows) == 0:
            return
        setup_rows = rows[np.asarray(obs)[rows, setup_phase_slot()] < (0.5 / 6.0)]
        for r in setup_rows:
            side = "US" if int(d_players[r]) == 1 else "USSR"
            if side not in OPENINGS[opening]:
                continue                      # this side sets up for itself
            k = int(self.cursor[side][r])
            script = OPENINGS[opening][side]
            if k >= len(script):
                raise RuntimeError(f"game {r}: {side} has no scripted placement {k} in '{opening}'")
            idx = NODE_OFFSET + script[k]
            if not masks[r][idx]:
                raise RuntimeError(f"game {r}: {side} placement {k} of '{opening}' (country "
                                   f"{script[k]}) is not legal here")
            actions[r] = idx
            self.cursor[side][r] = k + 1
