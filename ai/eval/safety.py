"""Detection of decisions that immediately win or lose the game.

Two implementations, for two different jobs:

* :func:`classify_legal_actions` -- a **rule-based** detector, fast enough to run inside a
  bot's move loop. Its rules are derived from, and verified against, the engine; the card
  list lives in ``docs/instant_decisions.md`` with the observed terminal values.
Coup targets are the one exception: `DecisionContext` does not expose `op_mode` to Python,
so a coup cannot be told from an influence placement by inspection, and that case applies
the action and checks for a terminal state instead -- exact, and one step deep.

The rule-based path is deliberately the one used in anger. A general minimax over this game
exhausts any sane node budget on breadth (a single influence-placement node offers 30+
actions), and a search that returns "unknown" is indistinguishable from one that returns
"safe" -- which is exactly the failure mode a safety check must not have.

Key engine facts these rules encode, each verified by stepping to a terminal state:

* The DEFCON-1 loser is always the **phasing player**, whoever's event fired.
* An opponent-associated card cannot be played as an event; playing it for **Ops** fires
  the owner's event unavoidably, on either timing branch.
* Wargames awards the **opponent** 6 VP and ends the game, so it wins only at a lead of 7+.
"""

from typing import Dict, List, Optional

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from ai.eval.positions import PLAY_MODE_ACTION

# Events that degrade DEFCON with no intervening choice. Firing one at DEFCON 2 ends the
# game against whoever is phasing. Verified in both directions (own event, and opponent's
# card played for Ops).
DEFCON_DEGRADING_CARDS = {
    4: "Duck and Cover",
    50: "We Will Bury You",
    89: "Soviets Shoot Down KAL-007",
    46: "How I Learned to Stop Worrying",
}

# Degrades DEFCON only if the opponent picks the boycott branch, so it is decisive only
# under an opponent who takes it.
OLYMPIC_GAMES = 20
WARGAMES = 100

# Events granting the OPPONENT a free coup. At DEFCON 2 that coup can take DEFCON to 1,
# losing the game for the player who played the card -- conditional on the opponent taking
# the line.
OPPONENT_COUP_CARDS = {
    91: "Ortega Elected in Nicaragua",
    62: "Lone Gunman",
    67: "Grain Sales to Soviets",
    96: "Tear Down This Wall",
}

EVENT_ACTION = PLAY_MODE_ACTION["event"]
OPS_ACTION = PLAY_MODE_ACTION["ops"]
NODE_OFFSET = 119


def _acting(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


def _vp_for(state: ts.GameState, player: ts.Player) -> int:
    """Victory points from `player`'s perspective; +20 wins, -20 loses."""
    vp = int(state.victory_points)
    return vp if player == ts.Player.US else -vp


def classify_legal_actions(
    state: ts.GameState,
    player: Optional[ts.Player] = None,
) -> Dict[int, str]:
    """Classifies each legal action as 'win', 'loss', 'risky', or 'normal'.

    * ``win``    -- ends the game in this player's favour.
    * ``loss``   -- ends the game against them, with no choice of theirs to avoid it.
    * ``risky``  -- hands the opponent a line that ends the game against them.
    * ``normal`` -- nothing decisive.
    """
    who = player if player is not None else _acting(state)
    ctx = state.ctx()
    legal = [int(a) for a in np.flatnonzero(ActionEncoder.get_legal_mask(state))]
    out: Dict[int, str] = {a: "normal" for a in legal}

    card = int(ctx.resolving_card) or int(ctx.pending_op_card)
    defcon = int(state.defcon)

    # --- play-mode decisions on a card already selected ---------------------------------
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and card:
        if defcon <= 2:
            if card in DEFCON_DEGRADING_CARDS:
                # Firing it loses outright; for an opponent card the only way to fire it is
                # Ops, and for our own card it is the event.
                if EVENT_ACTION in out:
                    out[EVENT_ACTION] = "loss"
                if OPS_ACTION in out and ts.CardData.get_card_info(card)["side"] != _side_name(who):
                    out[OPS_ACTION] = "loss"
            if card == OLYMPIC_GAMES and EVENT_ACTION in out:
                out[EVENT_ACTION] = "loss"
            if card in OPPONENT_COUP_CARDS and OPS_ACTION in out:
                out[OPS_ACTION] = "risky"

    # --- Wargames branch ----------------------------------------------------------------
    if card == WARGAMES and ctx.decision_type == ts.DecisionType.CHOOSE_BRANCH:
        lead = _vp_for(state, who)
        # Branch 0 hands the opponent 6 VP and ends the game.
        trigger = NODE_OFFSET - 119  # branch actions are CHOOSE_BRANCH indices 0/1
        for a in legal:
            if a == trigger:
                out[a] = "win" if lead - 6 > 0 else ("loss" if lead - 6 < 0 else "normal")

    # --- couping a battleground at DEFCON 2 ---------------------------------------------
    # DecisionContext does not expose op_mode to Python, so a coup target cannot be told
    # from an influence placement by inspection. Apply the action instead and see whether
    # the game ends: one step plus any forced die rolls is cheap, and it is exact.
    if defcon <= 2 and ctx.decision_type == ts.DecisionType.POINT_NODE:
        for a in legal:
            if a < NODE_OFFSET:
                continue
            probe = state.clone()
            try:
                ts.Engine.step_flat(probe, a)
                for _ in range(4):
                    if ts.Engine.is_terminal(probe):
                        break
                    pctx = probe.ctx()
                    if pctx.decision_type != ts.DecisionType.ROLL_DIE:
                        break
                    ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            except Exception:
                continue
            if ts.Engine.is_terminal(probe):
                util = float(ts.Engine.get_terminal_utility(probe))
                mine = util if who == ts.Player.US else -util
                out[a] = "win" if mine > 0 else ("loss" if mine < 0 else "normal")

    return out


def _side_name(player: ts.Player) -> str:
    return "US" if player == ts.Player.US else "USSR"


def find_instant_win(state: ts.GameState, player: Optional[ts.Player] = None) -> Optional[int]:
    for action, kind in classify_legal_actions(state, player).items():
        if kind == "win":
            return action
    return None


def safe_actions(
    state: ts.GameState,
    player: Optional[ts.Player] = None,
    avoid_risky: bool = True,
) -> List[int]:
    """Legal actions that do not lose outright, falling back when every option is bad."""
    kinds = classify_legal_actions(state, player)
    bad = {"loss", "risky"} if avoid_risky else {"loss"}
    safe = [a for a, k in kinds.items() if k not in bad]
    if safe:
        return safe
    safe = [a for a, k in kinds.items() if k != "loss"]
    return safe if safe else list(kinds.keys())
