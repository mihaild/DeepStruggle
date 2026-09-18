"""Drive a pre-P17 checkpoint on the post-P17 engine (P17 plan section 7).

P17 merged the three-step card play -- SELECT_PLAY_MODE, CHOOSE_TIMING_BRANCH, SELECT_OP_MODE --
into one resolution node. The observation is untouched by that refactor, so an old checkpoint's
input stays valid; only its action semantics differ.

**The old policy is asked its questions on a real old engine, never on a reconstruction.** Each
merged resolution is a *sequence* of old decisions, so the adapter transplants the current
position into a pre-P17 build, walks that sequence there -- taking the old engine's own
observation and mask at every node -- and maps the answers onto the one merged index.

Why not reconstruct the intermediate observation instead: it was tried and measured worse. The
observation carries an 8-wide one-hot over DecisionType (ctx_slots::DECISION_TYPE), so asking the
old policy its 2nd question while the state still says SELECT_PLAY_MODE feeds it an input it never
saw at that node. Restating just the one-hot, leaving pending_ops_value / op_mode / the timing
slots at their SELECT_PLAY_MODE values, builds an internally inconsistent vector -- off the
training manifold rather than merely at the wrong point on it. Largest |z| against the old engine
went 3.04 -> 8.87. Only a real old engine produces a consistent intermediate state.

The transplant is exact: `to_save_dict` / `state_from_save_dict` carry scalars, RNG, card
locations, influence, headline owners, the die-roll record and the ctx_stack, and the two builds
share a byte-identical `game_state.hpp`. Measured on a live position, the transplanted state gives
an identical legal mask and an identical observation (max abs diff 0). Plan section 7 assumed this
was impossible -- "GameState cannot be serialised through the bindings" -- and designed around it;
that constraint no longer holds.

Loading two engines in one process needs both of: a distinct module name, because CPython finds an
extension by its `PyInit_<name>` symbol, and a distinct nanobind domain, because nanobind's type
registry is process-global and the second build otherwise refuses to re-register `ts::RollType`.
The old build is compiled as `ts_engine_old` with `NB_DOMAIN ts_old`.

Set TS_OLD_ENGINE_SO to the old extension; without it the adapter refuses to run rather than
silently falling back to the approximate path.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

import ts_engine as ts
from tools.lib.player_agent import ColdWarModel
from bindings.action_encoder import ActionEncoder

# Old layout, the one the checkpoint's policy head was trained against.
OLD_PLAY_MODE = 110          # EVENT 0, OPS 1, SPACE 2, PASS 3
OLD_TIMING = 114             # OPS_FIRST 0, EVENT_FIRST 1
OLD_OP_MODE = 116            # INFLUENCE 0, COUP 1, REALIGN 2

# New layout.
NEW_PLAY_MODE = 110          # EVENT 0, SPACE 1, OPS_INFLUENCE 2, OPS_COUP 3, OPS_REALIGN 4
NEW_ROLL_DIE = 115
FLAT = ActionEncoder.FLAT_ACTION_SIZE

_OLD_MODULE: Optional[Any] = None


def old_engine() -> Any:
    """The pre-P17 extension, loaded once per process."""
    global _OLD_MODULE
    if _OLD_MODULE is not None:
        return _OLD_MODULE
    # Reuse an already-imported copy. Loading the same extension twice re-enters nanobind's
    # registration and aborts the process ("refusing to add duplicate key" on ts::RollType),
    # so this is a hard requirement, not an optimisation.
    cached = sys.modules.get("ts_engine_old")
    if cached is not None:
        _OLD_MODULE = cached
        return cached
    so = os.environ.get("TS_OLD_ENGINE_SO", "")
    if not so or not os.path.exists(so):
        raise RuntimeError(
            "TS_OLD_ENGINE_SO must point at a pre-P17 ts_engine_old extension. The adapter asks "
            "the old policy its questions on a real old engine; there is no approximate fallback, "
            "because the approximation was measured worse than none.")
    spec = importlib.util.spec_from_file_location("ts_engine_old", so)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load an extension from %s" % so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ts_engine_old"] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "Resolution"):
        raise RuntimeError("%s is a POST-P17 build; the adapter needs the pre-merge one" % so)
    _OLD_MODULE = mod
    return mod


class LegacyPolicyAdapter:
    """A tournament agent playing an old-representation checkpoint on the new engine.

    Deliberately NOT a NeuralAgent subclass: `batch_tournament` dispatches on that type for the
    fast path that feeds the merged mask straight to the model, which is what must not happen
    here. It exposes `select_actions_batch`, the state-based batch hook.
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
        self.old = old_engine()
        #: Positions the mapping could not answer, keyed by a short reason.
        self.divergences: Dict[str, int] = {}
        self.decisions = 0
        self.transplants = 0
        # The plan requires divergences COUNTED, not assumed away, and the tournament CLI has
        # nowhere to return them -- so they are reported on the way out rather than discarded.
        atexit.register(self._report_at_exit)

    def _report_at_exit(self) -> None:
        if self.decisions:
            print("[p17-adapter] %s: %s" % (self.name, self.report()), file=sys.stderr)

    def _note(self, reason: str) -> None:
        self.divergences[reason] = self.divergences.get(reason, 0) + 1

    # -- asking the old policy ---------------------------------------------------------------

    def _draw(self, obs: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> int:
        import torch
        obs_t = torch.from_numpy(obs[None, :]).float().to(self.device)
        mask_t = torch.from_numpy(mask[None, :].astype(np.uint8)).to(self.device)
        with torch.no_grad():
            act, _, _, _, _ = self.model.sample_action(
                obs_t, mask_t, temperature=self.temperature,
                deterministic=self.deterministic)
        return int(act.cpu().numpy()[0])

    def _draw_many(self, obs: npt.NDArray[np.float32],
                   masks: npt.NDArray[np.uint8]) -> List[int]:
        """One forward pass for a whole batch of (observation, mask) pairs."""
        import torch
        if obs.shape[0] == 0:
            return []
        obs_t = torch.from_numpy(obs).float().to(self.device)
        mask_t = torch.from_numpy(masks.astype(np.uint8)).to(self.device)
        with torch.no_grad():
            act, _, _, _, _ = self.model.sample_action(
                obs_t, mask_t, temperature=self.temperature,
                deterministic=self.deterministic)
        return [int(a) for a in act.cpu().numpy()]

    def _ask_old_many(self, old_states: List[Any]) -> List[int]:
        """One draw each, against every old state's OWN observation and mask."""
        o = self.old
        if not old_states:
            return []
        obs = np.stack([np.asarray(
            o.extract_observation(st, st.ctx().decision_player), dtype=np.float32)
            for st in old_states])
        masks = np.stack([np.asarray(
            o.Engine.get_flat_action_mask(st), dtype=np.uint8) for st in old_states])
        empty = ~masks.any(axis=1)
        picks = self._draw_many(obs, masks)
        return [-1 if empty[i] else picks[i] for i in range(len(old_states))]

    def _ask_old(self, old_state: Any) -> int:
        """One draw, against the OLD engine's own observation and mask at its current node."""
        o = self.old
        obs = np.asarray(o.extract_observation(old_state, old_state.ctx().decision_player),
                         dtype=np.float32)
        mask = np.asarray(o.Engine.get_flat_action_mask(old_state), dtype=np.uint8)
        if not mask.any():
            return -1
        return self._draw(obs, mask)

    # -- the merged resolution, walked on the old engine --------------------------------------

    def _resolutions(self, states: List[ts.GameState]) -> List[Optional[int]]:
        """The merged resolution for a batch, walking the old chain one stage at a time.

        Each stage depends on the previous answer, so the chain itself stays sequential -- but
        every state currently AT a given stage is asked in one forward pass.
        """
        o = self.old
        n = len(states)
        res: List[Optional[int]] = [None] * n
        live: List[int] = []
        old_states: Dict[int, Any] = {}

        for i, st in enumerate(states):
            self.transplants += 1
            try:
                osx = o.state_from_save_dict(dict(st.to_save_dict()))
            except Exception:
                self._note("transplant refused")
                continue
            if osx.ctx().decision_type != o.DecisionType.SELECT_PLAY_MODE:
                self._note("old engine is not at SELECT_PLAY_MODE after transplant")
                continue
            old_states[i] = osx
            live.append(i)

        # stage 1 -- play mode
        picks = self._ask_old_many([old_states[i] for i in live])
        still: List[int] = []
        for i, pick in zip(live, picks):
            if pick < 0:
                self._note("old SELECT_PLAY_MODE mask empty")
                continue
            pm = pick - OLD_PLAY_MODE
            if pm == 0:
                res[i] = NEW_PLAY_MODE + 0
            elif pm == 2:
                res[i] = NEW_PLAY_MODE + 1
            elif pm != 1:
                self._note("old play mode %d is not one the merge maps" % pm)
            elif not o.Engine.try_step(old_states[i], o.MicroAction(
                    o.DecisionType.SELECT_PLAY_MODE, 1, 0, 0)):
                self._note("old engine refused OPS")
            else:
                still.append(i)

        # stage 2 -- timing, only for those the old engine actually asks
        timing = [i for i in still
                  if old_states[i].ctx().decision_type == o.DecisionType.CHOOSE_TIMING_BRANCH]
        picks = self._ask_old_many([old_states[i] for i in timing])
        for i, pick in zip(timing, picks):
            if pick < 0:
                self._note("old timing mask empty")
                still.remove(i)
                continue
            if (pick - OLD_TIMING) == 1:
                # Event first. The Ops mode is a real deferred node on the new engine.
                res[i] = NEW_PLAY_MODE + 0
                still.remove(i)
            elif not o.Engine.try_step(old_states[i], o.MicroAction(
                    o.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0)):
                self._note("old engine refused OPS_FIRST")
                still.remove(i)

        # stage 3 -- op mode
        ready = []
        for i in still:
            if old_states[i].ctx().decision_type != o.DecisionType.SELECT_OP_MODE:
                self._note("old engine reached %s, not SELECT_OP_MODE"
                           % str(old_states[i].ctx().decision_type).split(".")[-1])
            else:
                ready.append(i)
        picks = self._ask_old_many([old_states[i] for i in ready])
        for i, pick in zip(ready, picks):
            if pick < 0:
                self._note("old op-mode mask empty")
            else:
                res[i] = NEW_PLAY_MODE + 2 + (pick - OLD_OP_MODE)
        return res

    def _resolution(self, state: ts.GameState) -> Optional[int]:
        """Walk the old card-play chain on a transplanted state; return the merged index.

        The old state is a throwaway: stepping it resolves events and consumes its own RNG, and
        none of that is kept. Only the *decisions* are harvested -- the new engine then plays the
        merged action and resolves its own chance outcomes.
        """
        o = self.old
        self.transplants += 1
        try:
            old_state = o.state_from_save_dict(dict(state.to_save_dict()))
        except Exception:
            self._note("transplant refused")
            return None

        if old_state.ctx().decision_type != o.DecisionType.SELECT_PLAY_MODE:
            self._note("old engine is not at SELECT_PLAY_MODE after transplant")
            return None

        pick = self._ask_old(old_state)
        if pick < 0:
            self._note("old SELECT_PLAY_MODE mask empty")
            return None
        play_mode = pick - OLD_PLAY_MODE
        if play_mode == 0:                                      # EVENT (own/neutral card)
            return NEW_PLAY_MODE + 0
        if play_mode == 2:                                      # SPACE
            return NEW_PLAY_MODE + 1
        if play_mode != 1:                                      # PASS, never masked in
            self._note("old play mode %d is not one the merge maps" % play_mode)
            return None

        # OPS. Step the old engine and let it say which node comes next.
        if not o.Engine.try_step(old_state, o.MicroAction(
                o.DecisionType.SELECT_PLAY_MODE, 1, 0, 0)):
            self._note("old engine refused OPS")
            return None

        if old_state.ctx().decision_type == o.DecisionType.CHOOSE_TIMING_BRANCH:
            t = self._ask_old(old_state)
            if t < 0:
                self._note("old timing mask empty")
                return None
            if (t - OLD_TIMING) == 1:
                # Event first. The Ops mode is a real deferred node on the new engine, answered
                # when it arrives rather than guessed at now.
                return NEW_PLAY_MODE + 0
            if not o.Engine.try_step(old_state, o.MicroAction(
                    o.DecisionType.CHOOSE_TIMING_BRANCH, 0, 0, 0)):
                self._note("old engine refused OPS_FIRST")
                return None

        if old_state.ctx().decision_type != o.DecisionType.SELECT_OP_MODE:
            self._note("old engine reached %s, not SELECT_OP_MODE"
                       % str(old_state.ctx().decision_type).split(".")[-1])
            return None
        m = self._ask_old(old_state)
        if m < 0:
            self._note("old op-mode mask empty")
            return None
        return NEW_PLAY_MODE + 2 + (m - OLD_OP_MODE)

    # -- the agent interface ------------------------------------------------------------------

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        n = len(states)
        self.decisions += n
        masks = [np.asarray(ts.get_flat_action_mask(st), dtype=np.uint8) for st in states]
        picks: List[Optional[int]] = [None] * n

        res_idx: List[int] = []
        op_idx: List[int] = []
        plain_idx: List[int] = []
        for i, st in enumerate(states):
            dt = st.ctx().decision_type
            if dt == ts.DecisionType.ROLL_DIE:
                # A chance node, not a policy decision, and 115 meant EVENT_FIRST in the old
                # space -- so the old policy must never be asked about it.
                picks[i] = NEW_ROLL_DIE
            elif dt == ts.DecisionType.SELECT_PLAY_MODE:
                res_idx.append(i)
            elif dt == ts.DecisionType.SELECT_OP_MODE:
                op_idx.append(i)
            elif masks[i].any():
                plain_idx.append(i)

        for i, r in zip(res_idx, self._resolutions([states[i] for i in res_idx])):
            picks[i] = r

        # The deferred Ops choice after an event-first event. The NODE exists in both engines,
        # but NOT at the same index: the merge put it on the OPS_* resolution slots 112..114,
        # while the old policy's head learned it at 116..118. Asking the old network on the new
        # mask reads logits that meant SPACE / PASS / OPS_FIRST to it -- noise in place of a
        # decision, on every event-first play. Remap.
        if op_idx:
            usable, obs_l, mk_l = [], [], []
            for i in op_idx:
                om = np.zeros(FLAT, dtype=np.uint8)
                for k in range(3):
                    if masks[i][NEW_PLAY_MODE + 2 + k]:
                        om[OLD_OP_MODE + k] = 1
                if not om.any():
                    self._note("deferred op mode: no Ops mode offered")
                    continue
                usable.append(i)
                mk_l.append(om)
                obs_l.append(np.asarray(ts.extract_observation(
                    states[i], states[i].ctx().decision_player), dtype=np.float32))
            if usable:
                got = self._draw_many(np.stack(obs_l), np.stack(mk_l))
                for i, g in zip(usable, got):
                    picks[i] = NEW_PLAY_MODE + 2 + (g - OLD_OP_MODE)

        # Everything else -- cards, country nodes, branches, confirm/done -- does occupy the
        # same index in both spaces AND presents the same observation, verified by transplant.
        if plain_idx:
            obs = np.stack([np.asarray(ts.extract_observation(
                states[i], states[i].ctx().decision_player), dtype=np.float32)
                for i in plain_idx])
            got = self._draw_many(obs, np.stack([masks[i] for i in plain_idx]))
            for i, g in zip(plain_idx, got):
                picks[i] = g

        out: List[int] = []
        for i in range(n):
            p = picks[i]
            if p is None or not masks[i][p]:
                if p is not None:
                    self._note("mapped to an index the merged mask refuses")
                legal = np.flatnonzero(masks[i])
                p = int(legal[0]) if legal.size else 0
            out.append(int(p))
        return out

    def select_action(self, state: ts.GameState, player: ts.Player,
                      temperature: float = 0.1) -> int:
        """One state at a time, for the CLIs that step decision by decision."""
        prev, self.temperature = self.temperature, temperature
        try:
            return self.select_actions_batch([state])[0]
        finally:
            self.temperature = prev

    def report(self) -> str:
        head = "%d decisions, %d transplants" % (self.decisions, self.transplants)
        if not self.divergences:
            return head + ", 0 divergences"
        rows = ", ".join("%s x%d" % (k, v) for k, v in sorted(self.divergences.items()))
        return head + ", divergences -- " + rows
