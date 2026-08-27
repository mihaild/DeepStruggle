"""Phase 0: Behavioral Cloning pre-training pipeline for ColdWarNet."""

import os
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from bindings.action_encoder import ActionEncoder
# ArenaEvaluator imported in train.py


class OldHeuristicPolicy:
    """Original baseline heuristic player without overcontrol prevention."""
    """Standard rule-based heuristic player for behavioral cloning bootstrap demonstrations."""

    @staticmethod
    def select_action(state: ts.GameState) -> int:
        ctx = state.ctx()
        mask = ActionEncoder.get_legal_mask(state)
        legal_indices = [int(i) for i in np.where(mask > 0)[0]]
        if not legal_indices:
            return ActionEncoder.CONFIRM_DONE_INDEX

        p = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
        opp = ts.Player.USSR if p == ts.Player.US else ts.Player.US
        d_type = ctx.decision_type

        # 1. SETUP Phase: secure European battlegrounds to stability
        if state.current_phase == ts.Phase.SETUP:
            if p == ts.Player.USSR:
                # USSR: 3 in East Germany (14), 3 in Poland (15)
                targets = [(14, 3), (15, 3), (12, 2), (13, 2)]
                for c_id, needed in targets:
                    if state.get_country(c_id).ussr_influence < needed:
                        action_idx = ActionEncoder.NODE_OFFSET + c_id
                        if action_idx in legal_indices:
                            return action_idx
            else:
                # US: Stage 0 Western Europe (WG to 4, Italy to 3, France) + Stage 1 Bonus (Iran to 2/3, Italy/WG)
                targets = [(25, 2), (7, 4), (10, 3), (8, 3), (25, 3), (10, 4), (9, 2)]
                for c_id, needed in targets:
                    if state.get_country(c_id).us_influence < needed:
                        action_idx = ActionEncoder.NODE_OFFSET + c_id
                        if action_idx in legal_indices:
                            return action_idx

        # 2. SELECT_CARD
        if d_type == ts.DecisionType.SELECT_CARD:
            # If Headline Phase: Prioritize friendly high-value events
            if state.current_phase == ts.Phase.HEADLINE:
                best_headline = None
                best_headline_ops = -1
                for idx in legal_indices:
                    if idx < ActionEncoder.PLAY_MODE_OFFSET:
                        card_id = idx + 1
                        try:
                            c_info = ts.CardData.get_card_info(card_id)
                            side = c_info.get("side", "NEUTRAL")
                            is_friendly = (side == ("US" if p == ts.Player.US else "USSR"))
                            if is_friendly and not c_info.get("is_scoring"):
                                ops = c_info.get("ops", 0)
                                if ops > best_headline_ops:
                                    best_headline_ops = ops
                                    best_headline = idx
                        except Exception:
                            pass
                if best_headline is not None:
                    return best_headline

            # Scoring cards if we hold advantage
            for idx in legal_indices:
                if idx < ActionEncoder.PLAY_MODE_OFFSET:
                    card_id = idx + 1
                    try:
                        c_info = ts.CardData.get_card_info(card_id)
                        if c_info.get("is_scoring"):
                            return idx
                    except Exception:
                        pass

            # Highest Ops card
            best_idx = legal_indices[0]
            best_ops = -1
            for idx in legal_indices:
                if idx < ActionEncoder.PLAY_MODE_OFFSET:
                    card_id = idx + 1
                    try:
                        c_info = ts.CardData.get_card_info(card_id)
                        ops = c_info.get("ops", 0)
                        if ops > best_ops:
                            best_ops = ops
                            best_idx = idx
                    except Exception:
                        pass
            return best_idx

        # 3. SELECT_PLAY_MODE: Prioritize OPS if opponent card, else EVENT or OPS
        if d_type == ts.DecisionType.SELECT_PLAY_MODE:
            ops_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.OPS)
            event_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.EVENT)
            space_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.SPACE)
            if ops_idx in legal_indices:
                return ops_idx
            if event_idx in legal_indices:
                return event_idx
            if space_idx in legal_indices:
                return space_idx

        # 4. SELECT_OP_MODE: Prefer COUP if DEFCON allows & target exists, else INFLUENCE
        if d_type == ts.DecisionType.SELECT_OP_MODE:
            inf_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.INFLUENCE)
            coup_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.COUP)
            realign_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.REALIGN)
            if inf_idx in legal_indices:
                return inf_idx
            if coup_idx in legal_indices:
                return coup_idx
            if realign_idx in legal_indices:
                return realign_idx

        # 5. POINT_NODE: Target low-stability battlegrounds with enemy presence (for Coups) or needed control (for Influence)
        if d_type == ts.DecisionType.POINT_NODE:
            bg_candidates = []
            for idx in legal_indices:
                if ActionEncoder.NODE_OFFSET <= idx < ActionEncoder.BRANCH_OFFSET:
                    country_id = idx - ActionEncoder.NODE_OFFSET
                    try:
                        info = ts.MapData.get_country_info(country_id)
                        if info.get("battleground"):
                            # Prefer battlegrounds
                            bg_candidates.append((idx, info.get("stability", 2)))
                    except Exception:
                        pass
            if bg_candidates:
                # Pick lowest stability battleground for max efficiency
                bg_candidates.sort(key=lambda x: x[1])
                return bg_candidates[0][0]

        # Default: first legal action
        return legal_indices[0]



class HeuristicPolicy:
    """Updated heuristic player that prioritizes controlling new battlegrounds and prevents overcontrol."""

    @staticmethod
    def select_action(state: ts.GameState) -> int:
        ctx = state.ctx()
        mask = ActionEncoder.get_legal_mask(state)
        legal_indices = [int(i) for i in np.where(mask > 0)[0]]
        if not legal_indices:
            return ActionEncoder.CONFIRM_DONE_INDEX

        p = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
        opp = ts.Player.USSR if p == ts.Player.US else ts.Player.US
        d_type = ctx.decision_type

        # 1. SETUP Phase: secure European battlegrounds to stability
        if state.current_phase == ts.Phase.SETUP:
            if p == ts.Player.USSR:
                # USSR: 3 in East Germany (14), 3 in Poland (15)
                targets = [(14, 3), (15, 3), (12, 2), (13, 2)]
                for c_id, needed in targets:
                    if state.get_country(c_id).ussr_influence < needed:
                        action_idx = ActionEncoder.NODE_OFFSET + c_id
                        if action_idx in legal_indices:
                            return action_idx
            else:
                # US: Stage 0 Western Europe (WG to 4, Italy to 3, France) + Stage 1 Bonus (Iran to 2/3, Italy/WG)
                targets = [(25, 2), (7, 4), (10, 3), (8, 3), (25, 3), (10, 4), (9, 2)]
                for c_id, needed in targets:
                    if state.get_country(c_id).us_influence < needed:
                        action_idx = ActionEncoder.NODE_OFFSET + c_id
                        if action_idx in legal_indices:
                            return action_idx

        # 2. SELECT_CARD
        if d_type == ts.DecisionType.SELECT_CARD:
            # If Headline Phase: Prioritize friendly high-value events
            if state.current_phase == ts.Phase.HEADLINE:
                best_headline = None
                best_headline_ops = -1
                for idx in legal_indices:
                    if idx < ActionEncoder.PLAY_MODE_OFFSET:
                        card_id = idx + 1
                        try:
                            c_info = ts.CardData.get_card_info(card_id)
                            side = c_info.get("side", "NEUTRAL")
                            is_friendly = (side == ("US" if p == ts.Player.US else "USSR"))
                            if is_friendly and not c_info.get("is_scoring"):
                                ops = c_info.get("ops", 0)
                                if ops > best_headline_ops:
                                    best_headline_ops = ops
                                    best_headline = idx
                        except Exception:
                            pass
                if best_headline is not None:
                    return best_headline

            # Scoring cards if we hold advantage
            for idx in legal_indices:
                if idx < ActionEncoder.PLAY_MODE_OFFSET:
                    card_id = idx + 1
                    try:
                        c_info = ts.CardData.get_card_info(card_id)
                        if c_info.get("is_scoring"):
                            return idx
                    except Exception:
                        pass

            # Highest Ops card
            best_idx = legal_indices[0]
            best_ops = -1
            for idx in legal_indices:
                if idx < ActionEncoder.PLAY_MODE_OFFSET:
                    card_id = idx + 1
                    try:
                        c_info = ts.CardData.get_card_info(card_id)
                        ops = c_info.get("ops", 0)
                        if ops > best_ops:
                            best_ops = ops
                            best_idx = idx
                    except Exception:
                        pass
            return best_idx

        # 3. SELECT_PLAY_MODE: Prioritize OPS if opponent card, else EVENT or OPS
        if d_type == ts.DecisionType.SELECT_PLAY_MODE:
            ops_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.OPS)
            event_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.EVENT)
            space_idx = ActionEncoder.PLAY_MODE_OFFSET + int(ts.PlayMode.SPACE)
            if ops_idx in legal_indices:
                return ops_idx
            if event_idx in legal_indices:
                return event_idx
            if space_idx in legal_indices:
                return space_idx

        # 4. SELECT_OP_MODE: Prefer COUP if DEFCON allows & target exists, else INFLUENCE
        if d_type == ts.DecisionType.SELECT_OP_MODE:
            inf_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.INFLUENCE)
            coup_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.COUP)
            realign_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.REALIGN)
            if inf_idx in legal_indices:
                return inf_idx
            if coup_idx in legal_indices:
                return coup_idx
            if realign_idx in legal_indices:
                return realign_idx

        # 5. POINT_NODE: Target new uncontrolled battlegrounds, then under-buffered battlegrounds, avoiding overcontrol
        if d_type == ts.DecisionType.POINT_NODE:
            best_action = None
            best_score = -999999

            for idx in legal_indices:
                if ActionEncoder.NODE_OFFSET <= idx < ActionEncoder.BRANCH_OFFSET:
                    country_id = idx - ActionEncoder.NODE_OFFSET
                    try:
                        c_state = state.get_country(country_id)
                        c_info = ts.MapData.get_country_info(country_id)
                        stab = c_info.get("stability", 2)
                        is_bg = c_info.get("battleground", False)

                        my_inf = c_state.us_influence if p == ts.Player.US else c_state.ussr_influence
                        opp_inf = c_state.ussr_influence if p == ts.Player.US else c_state.us_influence
                        my_ctrl = (my_inf >= opp_inf + stab)
                        deficit = max(stab - my_inf, opp_inf + stab - my_inf)

                        if is_bg:
                            if not my_ctrl:
                                # Top priority: uncontrolled battlegrounds (closest to control first)
                                score = 1000 - deficit * 10 - stab
                            elif my_inf == opp_inf + stab:
                                # Safe 1-point buffer on existing battlegrounds
                                score = 500 - stab
                            else:
                                # Overcontrolled battleground: heavily penalized
                                score = 10 - (my_inf - opp_inf - stab) * 5
                        else:
                            if not my_ctrl:
                                # Uncontrolled non-battlegrounds
                                score = 300 - deficit * 10 - stab
                            elif my_inf == opp_inf + stab:
                                score = 100
                            else:
                                score = 5 - (my_inf - opp_inf - stab) * 5

                        if score > best_score:
                            best_score = score
                            best_action = idx
                    except Exception:
                        pass

            if best_action is not None:
                return best_action

        # Default: first legal action
        return legal_indices[0]

class TrajectoryDataset(Dataset):
    """PyTorch Dataset of (observation, mask, action, target_value) tuples."""

    def __init__(self, obs_array: np.ndarray, mask_array: np.ndarray, action_array: np.ndarray, value_array: np.ndarray):
        self.obs = torch.from_numpy(obs_array).float()
        self.masks = torch.from_numpy(mask_array).float()
        self.actions = torch.from_numpy(action_array).long()
        self.values = torch.from_numpy(value_array).float()

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.masks[idx], self.actions[idx], self.values[idx]


class BehavioralCloningTrainer:
    """Supervised pre-training pipeline for ColdWarNet policy and value heads."""

    def __init__(
        self,
        model: ColdWarNet,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

    def generate_demonstration_dataset(
        self, num_games: int = 500, max_steps_per_game: int = 1000
    ) -> TrajectoryDataset:
        return self.collect_demonstrations(num_games, max_steps_per_game)

    def collect_demonstrations(self, num_games: int = 500, max_steps_per_game: int = 1000) -> TrajectoryDataset:
        """Simulates self-play games with HeuristicPolicy and records transitions."""
        print(f"Generating {num_games} heuristic demonstration games...")
        all_obs = []
        all_masks = []
        all_actions = []
        all_players = []
        all_game_indices = []

        total_steps = 0
        for g_idx in range(num_games):
            state = ts.GameState()
            seed = int(np.random.randint(1, 1_000_000_000))
            ts.Engine.init_game(state, seed)

            game_start_idx = len(all_obs)
            while not ts.Engine.is_terminal(state) and (len(all_obs) - game_start_idx) < 5000:
                p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
                obs = ts.extract_observation(state, p)
                mask = ActionEncoder.get_legal_mask(state)

                action_idx = HeuristicPolicy.select_action(state)

                all_obs.append(obs)
                all_masks.append(mask)
                all_actions.append(action_idx)
                all_players.append(p)
                all_game_indices.append(g_idx)

                ts.Engine.step_flat(state, action_idx)
                total_steps += 1

        obs_np = np.array(all_obs, dtype=np.float32)
        masks_np = np.array(all_masks, dtype=np.uint8)
        actions_np = np.array(all_actions, dtype=np.int64)

        # Compute terminal game outcomes for value target
        values_np = np.zeros(len(actions_np), dtype=np.float32)
        # Unique games
        unique_games = np.unique(all_game_indices)
        for g_idx in unique_games:
            idxs = np.where(np.array(all_game_indices) == g_idx)[0]
            if len(idxs) > 0:
                last_idx = idxs[-1]
                # Canonical win value: +1 if acting player wins, -1 if opponent wins
                for idx in idxs:
                    p = all_players[idx]
                    values_np[idx] = 1.0 if p == ts.Player.US else -1.0

        print(f"Collected {len(actions_np):,} transition steps from {num_games} games.")
        return TrajectoryDataset(obs_np, masks_np, actions_np, values_np)

    def train(self, dataset: TrajectoryDataset, epochs: int = 10, batch_size: int = 256, save_path: str = "data/checkpoints/coldwar_net_bc.pt") -> Dict[str, List[float]]:
        self.batch_size = batch_size
        history = self.train_epochs(dataset, epochs=epochs)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        return history

    def train_epochs(self, dataset: TrajectoryDataset, epochs: int = 10) -> Dict[str, List[float]]:
        """Trains ColdWarNet via supervised imitation learning."""
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.model.train()

        history = {"loss": [], "policy_loss": [], "value_loss": [], "accuracy": []}
        print(f"Starting Behavioral Cloning training for {epochs} epochs (Batch size: {self.batch_size})...")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            correct = 0
            total = 0

            for obs_b, mask_b, act_b, val_b in loader:
                obs_b = obs_b.to(self.device)
                mask_b = mask_b.to(self.device)
                act_b = act_b.to(self.device)
                val_b = val_b.to(self.device)

                self.optimizer.zero_grad()
                logits, v_win, v_vp = self.model(obs_b, mask_b)

                p_loss = self.policy_criterion(logits, act_b)
                v_loss = self.value_criterion(v_win.squeeze(-1), val_b)
                loss = p_loss + 0.5 * v_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item() * len(act_b)
                epoch_p_loss += p_loss.item() * len(act_b)
                epoch_v_loss += v_loss.item() * len(act_b)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == act_b).sum().item()
                total += len(act_b)

            avg_loss = epoch_loss / total
            avg_p_loss = epoch_p_loss / total
            avg_v_loss = epoch_v_loss / total
            acc = correct / total

            history["loss"].append(avg_loss)
            history["policy_loss"].append(avg_p_loss)
            history["value_loss"].append(avg_v_loss)
            history["accuracy"].append(acc)

            print(
                f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f}) | Action Acc: {acc*100:.2f}%"
            )

        return history
