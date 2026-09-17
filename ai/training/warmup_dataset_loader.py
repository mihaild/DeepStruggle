import gzip
import json
import torch
import numpy as np
from typing import Iterator, Tuple, List, Dict, Any, Optional
import ts_engine as ts

class WarmupDataset:
    """
    Forward-compatible Warmup Dataset Loader.
    Reads stored action traces from warmup games and dynamically re-extracts
    observation tensors using the current C++ engine state machine and observation code.
    """
    def __init__(self, filepath: str):
        # No default. The one that used to sit here pointed at a file that had moved, and a
        # stale default is worse than a missing argument: it resolves to something plausible
        # and trains on it. Every caller passes the path explicitly.
        self.filepath = filepath

    def stream_transitions(self, max_games: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        """
        Streams (observation, action_mask, action_id, win_target, vp_target) tuples.
        Replays the actual state machine on-the-fly, ensuring full compatibility
        with future changes to Observation::extract or feature representations.
        """
        game_count = 0
        with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if max_games is not None and game_count >= max_games:
                    break
                game = json.loads(line)
                st = ts.GameState()
                ts.Engine.init_game(st, game['seed'])
                winner = game.get('winner', 'DRAW')
                final_vp = game.get('final_vp', 0)
                
                for a in game['actions']:
                    p = st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE else st.phasing_player
                    obs = np.array(ts.extract_observation(st, p), copy=True)
                    mask = np.array(ts.get_flat_action_mask(st), copy=True)
                    
                    # Compute canonical perspective returns
                    sign = 1.0 if (p == ts.Player.US or p == 1) else -1.0
                    win_ret = (1.0 if winner == 'US' else (-1.0 if winner == 'USSR' else 0.0)) * sign
                    vp_ret = (final_vp / 20.0) * sign
                    
                    flat_act = a['flat_action']

                    # Skip transitions the replay cannot reproduce. A desynchronised
                    # replay yields an empty mask or an illegal demonstrated action;
                    # training on those regresses cross-entropy against -1e9 masked
                    # logits and destroys the run.
                    if mask.sum() == 0 or flat_act < 0 or flat_act >= mask.shape[0] or mask[flat_act] == 0:
                        break

                    yield obs, mask, flat_act, win_ret, vp_ret

                    ma = ts.decode_flat_action(st, flat_act)
                    ts.Engine.step(st, ma)

                    # Datasets are generated through VectorizedBatchRunner::step_flat_all,
                    # which resolves ROLL_DIE chance nodes internally and therefore records
                    # only player decisions. Drain those same chance nodes here or the
                    # replay stalls at the first die roll, misapplies every subsequent
                    # recorded action, and diverges the RNG stream along with the hands.
                    while (
                        not ts.Engine.is_terminal(st)
                        and st.ctx().decision_player == ts.Player.NONE
                        and st.ctx().decision_type == ts.DecisionType.ROLL_DIE
                    ):
                        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

                game_count += 1



    def stream_with_plays(self, max_games: Optional[int] = None):
        """As `stream_transitions`, plus the id of the play each decision belongs to.

        A card spends its Operations one point at a time and the order the points were recorded
        in carries no decision, so agreement is scored over the play rather than the sequence
        (`ai/eval/agreement`). The grouping is recovered here rather than stored, because this
        format keeps only a seed and the actions and rebuilds everything else by replaying.
        Coups and realignments are excluded from grouping -- a die resolves between their points
        and the board the next one faces depends on the last.
        """
        from ai.eval.agreement import order_matters

        play_id = 0
        key = None
        for game in self._games(max_games):
            st = ts.GameState()
            ts.Engine.init_game(st, game["seed"])
            winner = game.get("winner", "DRAW")
            final_vp = game.get("final_vp", 0)

            for a in game["actions"]:
                p = st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE else st.phasing_player
                obs = np.array(ts.extract_observation(st, p), copy=True)
                mask = np.array(ts.get_flat_action_mask(st), copy=True)
                sign = 1.0 if (p == ts.Player.US or p == 1) else -1.0
                win_ret = (1.0 if winner == "US" else (-1.0 if winner == "USSR" else 0.0)) * sign
                vp_ret = (final_vp / 20.0) * sign
                flat_act = a["flat_action"]
                if (mask.sum() == 0 or flat_act < 0 or flat_act >= mask.shape[0]
                        or mask[flat_act] == 0):
                    break

                ctx = st.ctx()
                groupable = (ctx.decision_type == ts.DecisionType.POINT_NODE
                             and not order_matters(st))
                this_key = ((int(st.turn), int(st.action_round), int(p),
                             int(ctx.resolving_card), int(ctx.pending_op_card))
                            if groupable else None)
                if this_key is None or this_key != key:
                    play_id += 1
                key = this_key
                yield obs, mask, flat_act, win_ret, vp_ret, play_id

                ma = ts.decode_flat_action(st, flat_act)
                ts.Engine.step(st, ma)
                while (not ts.Engine.is_terminal(st)
                       and st.ctx().decision_player == ts.Player.NONE
                       and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                    ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

    def _games(self, max_games: Optional[int] = None):
        count = 0
        with gzip.open(self.filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if max_games is not None and count >= max_games:
                    break
                yield json.loads(line)
                count += 1

    def stream_policy_transitions(
        self, max_games: Optional[int] = None
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
        """
        with gzip.open(self.filepath, 'rt', encoding='utf-8') as f:
            games = 0
            for line in f:
                if max_games is not None and games >= max_games:
                    break
                game = json.loads(line)
                st = ts.GameState()
                ts.Engine.init_game(st, game['seed'])
                for a in game['actions']:
                    p_ = (st.ctx().decision_player
                          if st.ctx().decision_player != ts.Player.NONE else st.phasing_player)
                    mask = np.array(ts.get_flat_action_mask(st), copy=True)
                    flat_act = a['flat_action']
                    if (mask.sum() == 0 or flat_act < 0 or flat_act >= mask.shape[0]
                            or mask[flat_act] == 0):
                        break            # desynchronised replay; the rest of this game is junk

                    pi = a.get('search_pi')
                    if pi:
                        obs = np.array(ts.extract_observation(st, p_), copy=True)
                        target = np.zeros(mask.shape[0], dtype=np.float32)
                        for act, vis in zip(pi['a'], pi['v']):
                            if 0 <= act < target.shape[0] and mask[act]:
                                target[act] += float(vis)
                        tot = target.sum()
                        if tot > 0:
                            yield obs, mask, target / tot, int(st.ctx().decision_type)

                    ma = ts.decode_flat_action(st, flat_act)
                    ts.Engine.step(st, ma)
                    while (not ts.Engine.is_terminal(st)
                           and st.ctx().decision_player == ts.Player.NONE
                           and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                        ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                games += 1

    def stream_policy_batches(
        self,
        batch_size: int = 512,
        max_games: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        shuffle_buffer_size: int = 4096,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Bounded-memory shuffled batches of (obs, mask, policy_target, decision_type)."""
        b_obs: List[np.ndarray] = []
        b_mask: List[np.ndarray] = []
        b_pi: List[np.ndarray] = []
        b_dt: List[int] = []

        def emit(idx):
            return (
                torch.from_numpy(np.stack([b_obs[i] for i in idx])).float().to(device),
                torch.from_numpy(np.stack([b_mask[i] for i in idx])).to(device),
                torch.from_numpy(np.stack([b_pi[i] for i in idx])).float().to(device),
                torch.tensor([b_dt[i] for i in idx], dtype=torch.long, device=device),
            )

        for obs, mask, pi, dt in self.stream_policy_transitions(max_games=max_games):
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
        batch_size: int = 512,
        max_games: Optional[int] = None,
        device: torch.device = torch.device('cpu'),
        shuffle_buffer_size: int = 2048,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Streams mini-batches directly as PyTorch tensors using an online reservoir buffer.
        Peak memory is strictly bounded by shuffle_buffer_size (< 40 MB), completely eliminating OOM.
        """
        buffer_obs: List[np.ndarray] = []
        buffer_mask: List[np.ndarray] = []
        buffer_act: List[int] = []
        buffer_w: List[float] = []
        buffer_vp: List[float] = []

        for obs, mask, act, w_ret, vp_ret in self.stream_transitions(max_games=max_games):
            buffer_obs.append(obs)
            buffer_mask.append(mask)
            buffer_act.append(act)
            buffer_w.append(w_ret)
            buffer_vp.append(vp_ret)

            if len(buffer_obs) >= shuffle_buffer_size:
                indices = np.random.choice(len(buffer_obs), size=batch_size, replace=False)
                keep_mask = np.ones(len(buffer_obs), dtype=bool)
                keep_mask[indices] = False

                b_obs = [buffer_obs[i] for i in indices]
                b_mask = [buffer_mask[i] for i in indices]
                b_act = [buffer_act[i] for i in indices]
                b_w = [buffer_w[i] for i in indices]
                b_vp = [buffer_vp[i] for i in indices]

                buffer_obs = [buffer_obs[i] for i in range(len(buffer_obs)) if keep_mask[i]]
                buffer_mask = [buffer_mask[i] for i in range(len(buffer_mask)) if keep_mask[i]]
                buffer_act = [buffer_act[i] for i in range(len(buffer_act)) if keep_mask[i]]
                buffer_w = [buffer_w[i] for i in range(len(buffer_w)) if keep_mask[i]]
                buffer_vp = [buffer_vp[i] for i in range(len(buffer_vp)) if keep_mask[i]]

                yield (
                    torch.from_numpy(np.array(b_obs, dtype=np.float32)).to(device),
                    torch.from_numpy(np.array(b_mask, dtype=np.uint8)).to(device),
                    torch.tensor(b_act, dtype=torch.long, device=device),
                    torch.tensor(b_w, dtype=torch.float32, device=device),
                    torch.tensor(b_vp, dtype=torch.float32, device=device),
                )

        while len(buffer_obs) >= batch_size:
            indices = np.random.choice(len(buffer_obs), size=batch_size, replace=False)
            keep_mask = np.ones(len(buffer_obs), dtype=bool)
            keep_mask[indices] = False

            b_obs = [buffer_obs[i] for i in indices]
            b_mask = [buffer_mask[i] for i in indices]
            b_act = [buffer_act[i] for i in indices]
            b_w = [buffer_w[i] for i in indices]
            b_vp = [buffer_vp[i] for i in indices]

            buffer_obs = [buffer_obs[i] for i in range(len(buffer_obs)) if keep_mask[i]]
            buffer_mask = [buffer_mask[i] for i in range(len(buffer_mask)) if keep_mask[i]]
            buffer_act = [buffer_act[i] for i in range(len(buffer_act)) if keep_mask[i]]
            buffer_w = [buffer_w[i] for i in range(len(buffer_w)) if keep_mask[i]]
            buffer_vp = [buffer_vp[i] for i in range(len(buffer_vp)) if keep_mask[i]]

            yield (
                torch.from_numpy(np.array(b_obs, dtype=np.float32)).to(device),
                torch.from_numpy(np.array(b_mask, dtype=np.uint8)).to(device),
                torch.tensor(b_act, dtype=torch.long, device=device),
                torch.tensor(b_w, dtype=torch.float32, device=device),
                torch.tensor(b_vp, dtype=torch.float32, device=device),
            )

    def build_in_memory_tensors(self, max_games: Optional[int] = None, device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """Extracts and loads dataset directly into PyTorch tensors.
        WARNING: High RAM consumption. Use stream_batches() for training to prevent OOM.
        """
        if max_games is None or max_games > 500:
            raise ValueError(
                f"build_in_memory_tensors is restricted to max_games <= 500 to prevent system OOM. "
                f"Use stream_batches() for bounded streaming training on large datasets."
            )
        all_obs = []
        all_masks = []
        all_acts = []
        all_win_rets = []
        all_vp_rets = []
        
        for obs, mask, act, w_ret, vp_ret in self.stream_transitions(max_games=max_games):
            all_obs.append(obs)
            all_masks.append(mask)
            all_acts.append(act)
            all_win_rets.append(w_ret)
            all_vp_rets.append(vp_ret)
            
        return {
            'observations': torch.from_numpy(np.array(all_obs, dtype=np.float32)).to(device),
            'action_masks': torch.from_numpy(np.array(all_masks, dtype=np.uint8)).to(device),
            'actions': torch.tensor(all_acts, dtype=torch.long).to(device),
            'win_targets': torch.tensor(all_win_rets, dtype=torch.float32).to(device),
            'vp_targets': torch.tensor(all_vp_rets, dtype=torch.float32).to(device),
        }
