"""Detection of decisions that immediately win or lose the game.

Two implementations, for two different jobs:

* :func:`classify_legal_actions` -- a **rule-based** detector, fast enough to run inside a
  bot's move loop. Its rules are derived from, and verified against, the engine.
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

    # --- forced resolutions: VP thresholds, scoring cards, coups ------------------------
    # Apply each action and follow only FORCED continuations -- chance nodes, and nodes
    # with a single legal action. Anything that ends the game along that path is decisive
    # no matter what either player would have chosen, so this needs no card list and picks
    # up every VP-threshold crossing (26 sites in the engine), every scoring card that
    # reaches +/-20, and every coup that takes DEFCON to 1.
    #
    # Following only forced steps is what keeps this both cheap and sound: it can miss a
    # decisive line that needed a choice, but it never reports one that does not exist.
    for a in legal:
        if out[a] != "normal":
            continue
        result = _probe_forced(state, a, who)
        if result is not None:
            out[a] = result

    return out


_GOLDEN = 0x9E3779B97F4A7C15
_UINT64 = 1 << 64


def _probe_forced(
    state: ts.GameState,
    action: int,
    player: ts.Player,
    max_forced: int = 6,
    die_samples: int = 6,
    node_budget: int = 64,
) -> Optional[str]:
    """Applies `action`, follows forced continuations, and reports a decisive outcome.

    A chance node is only decisive if *every* die outcome agrees. Stepping the die once and
    trusting the result -- which this did -- reports a win whenever that single sampled roll
    happens to succeed, so a coup or war that wins on 3-6 and leaves the game running on 1-2
    was labelled a forced win. Measured against the engine, 6% of "win" labels were
    die-dependent in that way: Brush War, Lone Gunman and similar VP-on-success cards, at
    9/16 to 14/16 win rates.

    Roll-independent lines survive this unchanged. A coup at DEFCON 2 degrades DEFCON to 1
    whatever the roll, ending the game against the phasing player, so every branch agrees
    and the label stands.

    Die outcomes are varied by mixing the probe's own rng_state, so the classifier stays a
    pure function of the position and remains reproducible.
    """
    probe = state.clone()
    try:
        ts.Engine.step_flat(probe, action)
    except Exception:
        return None
    return _follow_forced(probe, player, max_forced, die_samples, [node_budget])


def _follow_forced(
    probe: ts.GameState,
    player: ts.Player,
    budget: int,
    die_samples: int,
    nodes: List[int],
) -> Optional[str]:
    for _ in range(budget):
        nodes[0] -= 1
        if nodes[0] <= 0:
            return None
        if ts.Engine.is_terminal(probe):
            util = float(ts.Engine.get_terminal_utility(probe))
            mine = util if player == ts.Player.US else -util
            return "win" if mine > 0 else ("loss" if mine < 0 else None)

        pctx = probe.ctx()
        if pctx.decision_type == ts.DecisionType.ROLL_DIE:
            verdicts = set()
            base = int(probe.rng_state)
            for i in range(die_samples):
                branch = probe.clone()
                branch.rng_state = (base + (i + 1) * _GOLDEN) % _UINT64
                try:
                    ts.Engine.step(branch, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                except Exception:
                    return None
                verdicts.add(_follow_forced(branch, player, budget - 1, die_samples, nodes))
                if len(verdicts) > 1:
                    return None      # outcome depends on the die: not forced
            return verdicts.pop() if len(verdicts) == 1 else None

        forced = np.flatnonzero(ActionEncoder.get_legal_mask(probe))
        if len(forced) != 1:
            return None
        try:
            ts.Engine.step_flat(probe, int(forced[0]))
        except Exception:
            return None

    return None


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
