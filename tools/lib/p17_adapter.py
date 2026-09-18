"""Drive a pre-P17 checkpoint on the post-P17 engine (P17 plan section 7).

P17 merged the three-step card play -- SELECT_PLAY_MODE, CHOOSE_TIMING_BRANCH, SELECT_OP_MODE --
into one resolution node. The *observation* is untouched by that refactor, so an old checkpoint's
input stays valid and only its action semantics differ. This adapter reconstructs the old two- and
three-step masks from the merged one, asks the old policy the questions it was trained to answer,
and emits the merged index its answers add up to. No legacy C++ is needed: the merged mask carries
everything required.

    own / neutral                          opponent
      old EVENT <= bit0                      old EVENT  never legal
      old SPACE <= bit1                      old SPACE  <= bit1
      old OPS   <= bit2|bit3|bit4            old OPS    <= bit0|bit2|bit3|bit4
                                             timing: EVENT_FIRST <= bit0
                                                     OPS_FIRST   <= bit2|bit3|bit4
      if OPS -> op mode = {influence: bit2, coup: bit3, realign: bit4}

where bitN is merged index 110+N.

**The sampling structure is preserved, which is the whole point.** On the old engine the policy
drew once per node -- play mode, then timing, then op mode. The adapter draws in the same places,
so a comparison against the old engine tests the representation change and not a change in how
many times the distribution was sampled.

Divergences are counted, never papered over. The merge tightens legality in one corner: OPS used
to be legal unconditionally for any non-scoring card, and coup-only positions with no coup target
now lose that option. Where the reconstructed old mask would be empty -- a position the old engine
would have offered something in -- the adapter records it rather than guessing.
"""

from __future__ import annotations

import atexit
import sys

from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import torch

import ts_engine as ts
from tools.lib.player_agent import ColdWarModel

# Old layout, the one the checkpoint's policy head was trained against.
OLD_PLAY_MODE = 110          # EVENT 0, OPS 1, SPACE 2, PASS 3
OLD_TIMING = 114             # OPS_FIRST 0, EVENT_FIRST 1
OLD_OP_MODE = 116            # INFLUENCE 0, COUP 1, REALIGN 2

# New layout.
NEW_PLAY_MODE = 110          # EVENT 0, SPACE 1, OPS_INFLUENCE 2, OPS_COUP 3, OPS_REALIGN 4
NEW_ROLL_DIE = 115
FLAT = 212

# The decision-context block of the observation, which the old policy was conditioned on.
# ctx_slots::BASE is 72 within the 100-float global block, and the global block starts at
# OBS_SIZE - 100. DECISION_TYPE is an 8-wide one-hot over DecisionType.
#
# This matters because the new engine never enters CHOOSE_TIMING_BRANCH or the ops-first
# SELECT_OP_MODE -- the merge removed those nodes -- so a naive adapter asks the old policy its
# 2nd and 3rd questions while the state still says SELECT_PLAY_MODE. Measured on the pre-P17
# build, that is a 2-3 float difference from what the policy saw in training. The one-hot is
# restated here so each question is asked under the node identity it was trained on.
GLOBAL_BASE = int(ts.OBS_SIZE) - 100
DECISION_TYPE_SLOT = GLOBAL_BASE + 72
DT_SELECT_PLAY_MODE = 2
DT_CHOOSE_TIMING_BRANCH = 3
DT_SELECT_OP_MODE = 4


class LegacyPolicyAdapter:
    """A tournament agent that plays an old-representation checkpoint on the new engine.

    Deliberately NOT a NeuralAgent subclass: `batch_tournament` dispatches on that type for the
    fast path that feeds the merged mask straight to the model, which is exactly what must not
    happen here. It exposes `select_actions_batch`, the state-based batch hook.
    """

    def __init__(self, model: ColdWarModel, device: str = "cuda",
                 temperature: float = 0.1, deterministic: bool = False,
                 name: str = "legacy") -> None:
        #: Declared, not set by the caller alone: PlayerAgent is a Protocol requiring it.
        self.name = name
        self.model = model
        self.device = device
        self.temperature = temperature
        self.deterministic = deterministic
        #: Positions where the reconstructed old mask offered nothing, keyed by a short reason.
        self.divergences: Dict[str, int] = {}
        self.decisions = 0
        # The plan requires divergences COUNTED, not assumed away, and the tournament CLI has
        # nowhere to return them -- so they are reported on the way out rather than discarded.
        atexit.register(self._report_at_exit)

    def _report_at_exit(self) -> None:
        if self.decisions:
            print("[p17-adapter] %s: %s" % (self.name, self.report()), file=sys.stderr)

    # -- plumbing ---------------------------------------------------------------------------

    def _ask(self, state: ts.GameState, mask: npt.NDArray[np.uint8]) -> int:
        """One draw from the old policy, restricted to `mask`.

        The observation is taken as the new engine reports it. Restating the decision-type
        one-hot to name the old node was tried and measured WORSE (largest |z| 3.04 -> 8.87):
        the other context slots stay at their SELECT_PLAY_MODE values, so the result is an
        input combination that never occurred in training. See this module's header.
        """
        obs = np.asarray(ts.extract_observation(state, state.ctx().decision_player),
                         dtype=np.float32)[None, :]
        obs_t = torch.from_numpy(obs).float().to(self.device)
        mask_t = torch.from_numpy(mask[None, :].astype(np.uint8)).to(self.device)
        with torch.no_grad():
            act, _, _, _, _ = self.model.sample_action(
                obs_t, mask_t, temperature=self.temperature, deterministic=self.deterministic)
        return int(act.cpu().numpy()[0])

    def _note(self, reason: str) -> None:
        self.divergences[reason] = self.divergences.get(reason, 0) + 1

    @staticmethod
    def _blank() -> npt.NDArray[np.uint8]:
        return np.zeros(FLAT, dtype=np.uint8)

    # -- the mapping ------------------------------------------------------------------------

    def _op_mode_from_old(self, state: ts.GameState,
                          bits: Sequence[int]) -> Optional[int]:
        """Ask the old SELECT_OP_MODE question; return the merged index, or None if empty."""
        m = self._blank()
        for i in range(3):
            if bits[i]:
                m[OLD_OP_MODE + i] = 1
        if not m.any():
            self._note("op mode: no Ops mode offered")
            return None
        return NEW_PLAY_MODE + 2 + (self._ask(state, m) - OLD_OP_MODE)

    def _resolution(self, state: ts.GameState,
                    mask: npt.NDArray[np.uint8]) -> Optional[int]:
        """The merged resolution node, answered by up to three old-style questions."""
        ctx = state.ctx()
        bit = [int(mask[NEW_PLAY_MODE + i]) for i in range(5)]
        ops_bits = bit[2:5]

        card = int(ctx.pending_op_card)
        side = ts.CardData.get_card_info(card)["side"] if 1 <= card <= 110 else "neutral"
        me = "US" if int(ctx.decision_player) == int(ts.Player.US) else "USSR"
        opponent = side not in (me, "neutral")

        old = self._blank()
        if bit[1]:
            old[OLD_PLAY_MODE + 2] = 1                                   # SPACE
        if opponent:
            # EVENT alone is never a legal way to play an opponent's card; bit0 is the
            # event-first branch, which the old engine reached through OPS + a timing choice.
            if bit[0] or any(ops_bits):
                old[OLD_PLAY_MODE + 1] = 1                               # OPS
        else:
            if bit[0]:
                old[OLD_PLAY_MODE + 0] = 1                               # EVENT
            if any(ops_bits):
                old[OLD_PLAY_MODE + 1] = 1                               # OPS

        if not old.any():
            self._note("resolution: reconstructed old mask empty")
            return None

        choice = self._ask(state, old) - OLD_PLAY_MODE
        if choice == 0:                                                  # EVENT
            return NEW_PLAY_MODE + 0
        if choice == 2:                                                  # SPACE
            return NEW_PLAY_MODE + 1

        # OPS. On an opponent's card the old engine asked the timing next.
        if opponent:
            tm = self._blank()
            if any(ops_bits):
                tm[OLD_TIMING + 0] = 1                                   # OPS_FIRST
            if bit[0]:
                tm[OLD_TIMING + 1] = 1                                   # EVENT_FIRST
            if not tm.any():
                self._note("timing: neither branch offered")
                return None
            if (self._ask(state, tm) - OLD_TIMING) == 1:
                # Event first. The Ops mode is asked later, at the deferred SELECT_OP_MODE --
                # which is a real node on the new engine, so it is answered when it arrives
                # rather than guessed at now.
                return NEW_PLAY_MODE + 0
        return self._op_mode_from_old(state, ops_bits)

    # -- the agent interface ----------------------------------------------------------------

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        out: List[int] = []
        for st in states:
            mask = np.asarray(ts.get_flat_action_mask(st), dtype=np.uint8)
            dt = st.ctx().decision_type
            self.decisions += 1

            if dt == ts.DecisionType.ROLL_DIE:
                # A chance node, not a policy decision, and index 115 meant EVENT_FIRST in the
                # old space -- so the old policy must never be asked about it.
                out.append(NEW_ROLL_DIE)
                continue

            if dt == ts.DecisionType.SELECT_PLAY_MODE:
                pick = self._resolution(st, mask)
            elif dt == ts.DecisionType.SELECT_OP_MODE:
                pick = self._op_mode_from_old(
                    st, [int(mask[NEW_PLAY_MODE + 2 + i]) for i in range(3)])
            else:
                # Every other decision occupies the same indices in both spaces -- cards,
                # country nodes, branches, confirm/done -- so the merged mask is already the
                # old mask and one draw answers it.
                pick = self._ask(st, mask) if mask.any() else None

            if pick is None or not mask[pick]:
                if pick is not None:
                    self._note("mapped to an index the merged mask refuses")
                legal = np.flatnonzero(mask)
                pick = int(legal[0]) if legal.size else 0
            out.append(int(pick))
        return out

    def select_action(self, state: ts.GameState, player: ts.Player,
                      temperature: float = 0.1) -> int:
        """One state at a time, for the CLIs that step decision by decision.

        `temperature` is honoured per call rather than pinned at construction, so play_match.py
        and the tournament's per-agent temperature override reach the old policy the same way
        they reach every other agent.
        """
        prev, self.temperature = self.temperature, temperature
        try:
            return self.select_actions_batch([state])[0]
        finally:
            self.temperature = prev

    def report(self) -> str:
        if not self.divergences:
            return "adapter: %d decisions, 0 divergences" % self.decisions
        rows = ", ".join("%s x%d" % (k, v) for k, v in sorted(self.divergences.items()))
        return "adapter: %d decisions, divergences -- %s" % (self.decisions, rows)
