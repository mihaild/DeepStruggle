"""MCTS with a rules-derived leaf value and no network.

`pimcts` needs a checkpoint: it takes the policy as prior and `v_win` as leaf. That makes it
useless as a *baseline*, because its strength is the network's and it cannot exist before one is
trained. This supplies the same search with `heuristic_eval` at the leaf and a one-ply lookahead
prior, so the resulting agent's strength is attributable entirely to the search and the
hand-written evaluation.

Why this is wanted: `HeuristicBot` is the ladder's only engine-independent reference point, and it
plays a fixed rule list with no lookahead at all. This gives the ladder a second anchor that needs
no training run.

**`simulations` is a coarse dial, not a smooth one.** Measured against HeuristicBot over 20 games
a side: 1 sim -> 9/20, 16 -> 18/20, 96 -> 17/20, 256 -> 18/20. Search buys the jump from 1 to 16
and nothing after, because the one-ply prior below is sharp enough that PUCT settles on its top
action early and further simulations deepen a branch already chosen. If a finer dial is wanted,
`prior_temperature` is the parameter to vary, not `simulations`.

The search itself is `PIMCTSAgent`, unchanged -- PUCT, backup and the US-perspective sign
convention are shared rather than copied, so a fix to one is a fix to both. Only the leaf differs.

**Perfect information.** Like `pimcts`, this searches the true `GameState`, so it sees the
opponent's hand. That is fine for a *baseline* -- a benchmark opponent is allowed to be stronger
than anything deployable -- but it means a win rate against this bot is not a claim about play
under the real information set. Said plainly because the distinction is easy to lose when the
thing is registered next to the ordinary bots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import ts_engine as ts
from ai.eval import dominance
from ai.search.heuristic_eval import HeuristicWeights, evaluate
from bindings.action_encoder import ActionEncoder
from ai.search.pimcts import PIMCTSAgent, PIMCTSConfig, drain_chance_nodes


@dataclass
class HeuristicMCTSConfig:
    """Search shape. `simulations` is the difficulty dial."""

    simulations: int = 64
    c_puct: float = 1.5
    temperature: float = 0.0
    seed: int = 12345
    #: Cost control. A prior built by stepping every legal action costs one engine step per
    #: action per node; at ~10 actions that is 10x the node cost. Above this many legal actions
    #: the prior falls back to uniform, which keeps the worst node bounded.
    prior_lookahead_max_actions: int = 16
    #: Softmax temperature over the one-ply values that form the prior. Lower = sharper.
    prior_temperature: float = 0.25
    #: How much a named mistake costs in the prior. In units of the leaf value, which spans
    #: (-1, 1), so 0.5 is a large but not absolute bias -- the tree can still overrule it.
    blunder_penalty: float = 0.5
    #: default_factory, not `= DEFAULT_WEIGHTS`: a dataclass refuses a mutable default, and
    #: sharing one instance across agents would let a tuning arm mutate every other agent's.
    weights: HeuristicWeights = field(default_factory=HeuristicWeights)


class HeuristicLeaf:
    """`leaf_fn` for `PIMCTSAgent`: one-ply-lookahead prior, heuristic value.

    A uniform prior would waste most of the simulation budget: PUCT explores in prior order, and
    with ~10 legal actions and 64 simulations a uniformly-primed search visits each branch about
    six times, which is not enough to separate them. Ranking the children by a single engine step
    each costs one step per action and buys a prior that is already roughly right, so the
    simulations go into resolving the close calls rather than discovering which calls are close.

    The step is taken on a clone with chance nodes drained, so a die roll does not leak one
    sampled outcome into the prior for every later visit.
    """

    def __init__(self, cfg: HeuristicMCTSConfig) -> None:
        self.cfg = cfg

    def _play_rule_penalties(self, state: ts.GameState, legal: np.ndarray,
                             mover: ts.Player) -> np.ndarray:
        """Named mistakes, subtracted from the prior before the softmax.

        A one-ply value cannot see these. Spacing a card removes it without firing its Event, so
        every space play looks identical to the evaluation -- the question of *which* card to
        space is invisible to a position score and decided entirely by what the card was worth.
        Same for the timing of an Event whose value is conditional on DEFCON.

        The rules are not reimplemented here. `ai.eval.blunders` already defines them, is already
        what the training loop measures against, and `dominance.space_dominance_outcome` already
        encodes the space comparison -- a second copy would be a second definition, and the one
        that drifted would be this one.

        A penalty biases the prior; it never removes an action. The search can still play a
        "blunder" if the tree says the position after it is good, which matters because these
        rules are position-independent generalisations and the tree is not.
        """
        penalties = np.zeros(len(legal), dtype=float)
        if state.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
            return penalties

        card = int(state.ctx().pending_op_card)
        if not (1 <= card <= 110):
            return penalties

        space_idx = int(ts.Resolution.SPACE)
        for i, action in enumerate(legal):
            resolution = int(action) - ActionEncoder.PLAY_MODE_OFFSET
            if resolution != space_idx:
                continue
            # "Do not spend your own or a neutral card on the track while holding an equal-Ops
            # opponent card": the track is the one outlet that spends a card without firing its
            # Event, so spending yours and keeping theirs is backwards.
            try:
                outcome = dominance.space_dominance_outcome(state, mover, card)
            except Exception:
                outcome = None
            if outcome is False:
                penalties[i] += self.cfg.blunder_penalty
        return penalties

    def __call__(self, state: ts.GameState,
                 legal: np.ndarray) -> Tuple[np.ndarray, float]:
        value_us = evaluate(state, self.cfg.weights)
        n = len(legal)
        if n <= 1 or n > self.cfg.prior_lookahead_max_actions:
            return np.full(max(n, 1), 1.0 / max(n, 1)), value_us

        child_values = np.empty(n, dtype=float)
        for i, action in enumerate(legal):
            probe = state.clone()
            try:
                ts.Engine.step_flat(probe, int(action))
            except Exception:
                # An action the mask offered and `step` refused is an engine defect, not
                # something to paper over with a guess -- but a search is the wrong place to
                # raise it. Rank it last and let the fuzzer and the differential tests be what
                # reports it.
                child_values[i] = -np.inf
                continue
            drain_chance_nodes(probe)
            child_values[i] = evaluate(probe, self.cfg.weights)

        # The mover picks what is good for THEM, so orient before the softmax.
        ctx = state.ctx()
        mover = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
        oriented = child_values if mover == ts.Player.US else -child_values
        oriented = oriented - self._play_rule_penalties(state, legal, mover)

        finite = np.isfinite(oriented)
        if not finite.any():
            return np.full(n, 1.0 / n), value_us
        oriented = np.where(finite, oriented, oriented[finite].min() - 1.0)

        temp = max(self.cfg.prior_temperature, 1e-3)
        z = (oriented - oriented.max()) / temp
        priors = np.exp(z)
        total = priors.sum()
        return (priors / total if total > 1e-12 else np.full(n, 1.0 / n)), value_us


def make_heuristic_mcts_agent(name: str = "HeuristicMCTS",
                              config: Optional[HeuristicMCTSConfig] = None) -> PIMCTSAgent:
    """A `PIMCTSAgent` that needs no checkpoint."""
    cfg = config or HeuristicMCTSConfig()
    return PIMCTSAgent(
        model=None,
        name=name,
        config=PIMCTSConfig(simulations=cfg.simulations, c_puct=cfg.c_puct,
                            temperature=cfg.temperature, seed=cfg.seed),
        leaf_fn=HeuristicLeaf(cfg),
    )
