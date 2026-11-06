import gzip
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

import ts_engine as ts
from bindings.ts_env import legal_mask, observe
from tools.lib.game_step import drain_chance


@dataclass
class ReplayedDecision:
    """One recorded decision, at the replayed position it was taken in.

    `state` is the live replay state BEFORE `action` is applied: valid only until the generator
    advances (clone it to keep it). `mask` is the legal mask in the view the game was played in,
    and the action is legal in it -- a record that is not ends its game (see `_replay`).
    """
    game: Dict[str, Any]
    record: Dict[str, Any]
    state: "ts.GameState"
    player: "ts.Player"
    mask: np.ndarray
    action: int


def _side_merged(game: Dict[str, Any], player: "ts.Player") -> bool:
    """Whether `player` decided in the merged-influence view in a tools/generate_dataset.py game.

    The generator records `us_merged_influence` / `ussr_merged_influence` on every game. A game
    without them comes from a generator that never set a view on its runner, so it was played in
    E4: False is what that file is, not a guess.
    """
    key = "us_merged_influence" if player == ts.Player.US else "ussr_merged_influence"
    return bool(game.get(key, False))


def _side_obs_features(game: Dict[str, Any], player: "ts.Player") -> int:
    """The feature set `player`'s seat played a game in, whose rule bits are the rules it played.
    Absent (games from before the first rule bit, converted human games): 0, the pre-bit rule,
    whose mask offers everything the current one does, so such a game replays whole."""
    key = "us_obs_features" if player == ts.Player.US else "ussr_obs_features"
    return int(game.get(key, 0))


class WarmupDataset:
    """
    Forward-compatible Warmup Dataset Loader.
    Reads stored action traces from warmup games and dynamically re-extracts
    observation tensors using the current C++ engine state machine and observation code.

    Every stream takes `features`, the observation feature set (`ts.OBS_FEATURE_*` bits,
    `bindings.ts_env.model_obs_features(model)`) of the network the observations are FOR -- the
    student being trained or the model being evaluated, not the one that generated the games. It
    is required: a featured network fed the base layout misreads without failing
    (`HEADLINE_REVEAL` has no width).
    """
    def __init__(self, filepath: str):
        # No default. The one that used to sit here pointed at a file that had moved, and a
        # stale default is worse than a missing argument: it resolves to something plausible
        # and trains on it. Every caller passes the path explicitly.
        self.filepath = filepath

    def _games(self, max_games: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        count = 0
        with gzip.open(self.filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if max_games is not None and count >= max_games:
                    break
                yield json.loads(line)
                count += 1

    def _replay(self, max_games: Optional[int],
                merged: Optional[bool]) -> Iterator[ReplayedDecision]:
        """The one replay loop: every recorded decision of every game, at its replayed position.

        `merged` is the action view the games were played in: a bool for a dataset whose sidecar
        names one view for both seats (tools/generate_policy_targets.py), None to read each side's
        view from the game record (tools/generate_dataset.py, `_side_merged`). A composed E4.1
        action is applied as the two E4 steps it names.

        Datasets are generated through VectorizedBatchRunner::step_flat_all, which resolves
        ROLL_DIE chance nodes internally and therefore records only player decisions. The same
        chance nodes are drained here or the replay stalls at the first die roll, misapplies every
        subsequent recorded action, and diverges the RNG stream along with the hands.

        A record the replay cannot play -- an empty mask, or a demonstrated action outside it --
        ends that game: a desynchronised replay is a different game from there on, and training
        on it would regress cross-entropy against masked logits.
        """
        for game in self._games(max_games):
            st = ts.GameState()
            ts.Engine.init_game(st, game["seed"])
            for seat in (ts.Player.US, ts.Player.USSR):
                st.seat_obs_features(seat, _side_obs_features(game, seat))
            for rec in game["actions"]:
                drain_chance(st)
                p = st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE else st.phasing_player
                view = _side_merged(game, p) if merged is None else merged
                # The seats were seated from the record above; the replay must not reseat them.
                mask = legal_mask(st, view, None)
                flat_act = int(rec["flat_action"])
                if mask.sum() == 0 or flat_act < 0 or flat_act >= mask.shape[0] or mask[flat_act] == 0:
                    break
                yield ReplayedDecision(game=game, record=rec, state=st, player=p, mask=mask, action=flat_act)
                # Raises if the engine refuses an action its own mask offered (an engine defect).
                ts.Engine.step_flat(st, flat_act, False, view)

    @staticmethod
    def _returns(game: Dict[str, Any], player: "ts.Player") -> Tuple[float, float]:
        """(win, vp) targets from `player`'s perspective."""
        winner = game.get("winner", "DRAW")
        final_vp = game.get("final_vp", 0)
        sign = 1.0 if player == ts.Player.US else -1.0
        win_ret = (1.0 if winner == "US" else (-1.0 if winner == "USSR" else 0.0)) * sign
        return win_ret, (final_vp / 20.0) * sign

    def stream_transitions(self, *, features: int, max_games: Optional[int] = None
                           ) -> Iterator[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        """
        Streams (observation, action_mask, action_id, win_target, vp_target) tuples.
        Replays the actual state machine on-the-fly, ensuring full compatibility
        with future changes to Observation::extract or feature representations.
        """
        for d in self._replay(max_games, None):
            win_ret, vp_ret = self._returns(d.game, d.player)
            yield observe(d.state, d.player, features), d.mask, d.action, win_ret, vp_ret

    def stream_with_plays(self, *, features: int, order_matters: Callable[["ts.GameState"], bool],
                          max_games: Optional[int] = None
                          ) -> Iterator[Tuple[np.ndarray, np.ndarray, int, float, float, int]]:
        """As `stream_transitions`, plus the id of the play each decision belongs to.

        A card spends its Operations one point at a time and the order the points were recorded
        in carries no decision, so agreement is scored over the play rather than the sequence
        (`ai/eval/agreement`). The grouping is recovered here rather than stored, because this
        format keeps only a seed and the actions and rebuilds everything else by replaying.
        Coups and realignments are excluded from grouping -- a die resolves between their points
        and the board the next one faces depends on the last. Which point decisions are a
        sequence is `order_matters` (`ai.eval.agreement.order_matters`), passed in by the caller.
        """
        play_id = 0
        key: Optional[Tuple[int, ...]] = None
        for d in self._replay(max_games, None):
            win_ret, vp_ret = self._returns(d.game, d.player)
            st = d.state
            ctx = st.ctx()
            groupable = ctx.decision_type == ts.DecisionType.POINT_NODE and not order_matters(st)
            this_key = ((int(st.turn), int(st.action_round), int(d.player),
                         int(ctx.resolving_card), int(ctx.pending_op_card))
                        if groupable else None)
            if this_key is None or this_key != key:
                play_id += 1
            key = this_key
            yield observe(st, d.player, features), d.mask, d.action, win_ret, vp_ret, play_id

    def stream_policy_transitions(
        self, *, features: int, max_games: Optional[int] = None, merged: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Streams (observation, action_mask, policy_target) for SEARCHED decisions only.

        A dataset from `tools/generate_search_targets.py` carries `search_pi` on the decisions a
        searcher answered: `{"a": [flat ids], "v": [visit counts]}`. Sparse on disk, dense here.
        Records without it are replayed (the game has to advance) but not yielded — they have no
        target, and inventing one from the played action would quietly turn expert iteration back
        into behaviour cloning of the policy that generated the data, which teaches nothing.

        The target is renormalised over the LEGAL actions of the replayed state. A visit on an
        action the current engine calls illegal is dropped rather than trusted: the mask is the
        authority, and a target with mass outside it would train the policy toward a move it
        cannot play.

        The fourth element is the decision type, so a caller can ask where the searcher and
        the policy actually differ. X4a found 93.1% agreement over card/play-mode nodes, which
        average 4.1 legal actions; whether that holds at POINT_NODE placements, which average
        17.5, is a different question and needs the split.

        `merged` replays in the E4.1 merged-influence view (P23): masks from the merged view, and a
        composed action applied as the two E4 steps it names. A dataset played in that view
        (`tools/generate_policy_targets.py --merged-view`) desynchronises at its first composed
        action if replayed in E4, so the caller reads the view from the dataset's sidecar.
        """
        for d in self._replay(max_games, merged):
            pi = d.record.get("search_pi")
            if not pi:
                continue
            target = np.zeros(d.mask.shape[0], dtype=np.float32)
            for act, vis in zip(pi["a"], pi["v"]):
                if 0 <= act < target.shape[0] and d.mask[act]:
                    target[act] += float(vis)
            tot = target.sum()
            if tot > 0:
                yield (observe(d.state, d.player, features), d.mask, target / tot,
                       int(d.state.ctx().decision_type))

    def stream_policy_batches(
        self,
        *,
        features: int,
        batch_size: int = 512,
        max_games: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        shuffle_buffer_size: int = 4096,
        merged: bool = False,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Bounded-memory shuffled batches of (obs, mask, policy_target, decision_type); `merged`
        and `features` as in stream_policy_transitions."""
        b_obs: List[np.ndarray] = []
        b_mask: List[np.ndarray] = []
        b_pi: List[np.ndarray] = []
        b_dt: List[int] = []

        def emit(idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.from_numpy(np.stack([b_obs[i] for i in idx])).float().to(device),
                torch.from_numpy(np.stack([b_mask[i] for i in idx])).to(device),
                torch.from_numpy(np.stack([b_pi[i] for i in idx])).float().to(device),
                torch.tensor([b_dt[i] for i in idx], dtype=torch.long, device=device),
            )

        for obs, mask, pi, dt in self.stream_policy_transitions(features=features, max_games=max_games,
                                                                merged=merged):
            b_obs.append(obs)
            b_mask.append(mask)
            b_pi.append(pi)
            b_dt.append(dt)
            if len(b_obs) >= shuffle_buffer_size:
                idx = np.random.choice(len(b_obs), size=batch_size, replace=False)
                yield emit(idx)
                keep = sorted(set(range(len(b_obs))) - set(idx.tolist()))
                b_obs = [b_obs[i] for i in keep]
                b_mask = [b_mask[i] for i in keep]
                b_pi = [b_pi[i] for i in keep]
                b_dt = [b_dt[i] for i in keep]

        while len(b_obs) >= batch_size:
            idx = np.arange(batch_size)
            yield emit(idx)
            b_obs, b_mask, b_pi, b_dt = (b_obs[batch_size:], b_mask[batch_size:],
                                         b_pi[batch_size:], b_dt[batch_size:])
        if b_obs:
            yield emit(np.arange(len(b_obs)))

    def stream_batches(
        self,
        *,
        features: int,
        batch_size: int = 512,
        max_games: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        shuffle_buffer_size: int = 2048,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Streams mini-batches directly as PyTorch tensors using an online reservoir buffer.
        Peak memory is strictly bounded by shuffle_buffer_size (< 40 MB), completely eliminating OOM.
        """
        buf: List[Tuple[np.ndarray, np.ndarray, int, float, float]] = []

        def emit(rows: List[Tuple[np.ndarray, np.ndarray, int, float, float]]
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                torch.from_numpy(np.array([r[0] for r in rows], dtype=np.float32)).to(device),
                torch.from_numpy(np.array([r[1] for r in rows], dtype=np.uint8)).to(device),
                torch.tensor([r[2] for r in rows], dtype=torch.long, device=device),
                torch.tensor([r[3] for r in rows], dtype=torch.float32, device=device),
                torch.tensor([r[4] for r in rows], dtype=torch.float32, device=device),
            )

        def draw() -> List[Tuple[np.ndarray, np.ndarray, int, float, float]]:
            nonlocal buf
            indices = np.random.choice(len(buf), size=batch_size, replace=False)
            keep = np.ones(len(buf), dtype=bool)
            keep[indices] = False
            rows = [buf[i] for i in indices]
            buf = [buf[i] for i in range(len(buf)) if keep[i]]
            return rows

        for row in self.stream_transitions(features=features, max_games=max_games):
            buf.append(row)
            if len(buf) >= shuffle_buffer_size:
                yield emit(draw())

        while len(buf) >= batch_size:
            yield emit(draw())
