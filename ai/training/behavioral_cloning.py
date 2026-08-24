"""Behavioral Cloning (BC) Pre-Trainer for Twilight Struggle ColdWarNet."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder


class HeuristicPolicy:
    """Fast native heuristic policy generator operating directly on ts.GameState."""

    @staticmethod
    def select_action(state: ts.GameState) -> int:
        ctx = state.ctx()
        mask = ActionEncoder.get_legal_mask(state)
        legal_indices = [int(i) for i in np.where(mask > 0)[0]]
        if not legal_indices:
            return ActionEncoder.CONFIRM_DONE_INDEX

        p = ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player
        d_type = ctx.decision_type

        # 1. SETUP Phase: prioritize key European battlegrounds
        if state.current_phase == ts.Phase.SETUP:
            preferred = [14, 15, 12, 13] if p == ts.Player.USSR else [6, 10, 7, 8]
            for c_id in preferred:
                action_idx = ActionEncoder.NODE_OFFSET + c_id
                if action_idx in legal_indices:
                    return action_idx

        # 2. SELECT_CARD: Prioritize scoring cards or high-ops cards
        if d_type == ts.DecisionType.SELECT_CARD:
            # Check for scoring cards first
            for idx in legal_indices:
                if idx < ActionEncoder.PLAY_MODE_OFFSET:
                    card_id = idx + 1
                    try:
                        c_info = ts.CardData.get_card_info(card_id)
                        if c_info.get("is_scoring"):
                            return idx
                    except Exception:
                        pass
            # Else pick highest Ops
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
            if ops_idx in legal_indices:
                return ops_idx
            if event_idx in legal_indices:
                return event_idx

        # 4. SELECT_OP_MODE: Prefer INFLUENCE if unplaced, else COUP on high stability/battleground
        if d_type == ts.DecisionType.SELECT_OP_MODE:
            inf_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.INFLUENCE)
            coup_idx = ActionEncoder.OP_MODE_OFFSET + int(ts.OpMode.COUP)
            if inf_idx in legal_indices:
                return inf_idx
            if coup_idx in legal_indices:
                return coup_idx

        # 5. POINT_NODE: Prioritize Battleground countries
        if d_type == ts.DecisionType.POINT_NODE:
            bg_candidates = []
            for idx in legal_indices:
                if ActionEncoder.NODE_OFFSET <= idx < ActionEncoder.BRANCH_OFFSET:
                    country_id = idx - ActionEncoder.NODE_OFFSET
                    try:
                        info = ts.MapData.get_country_info(country_id)
                        if info.get("battleground"):
                            bg_candidates.append(idx)
                    except Exception:
                        pass
            if bg_candidates:
                return int(np.random.choice(bg_candidates))

        # Default: sample uniformly from legal actions
        return int(np.random.choice(legal_indices))


class BehavioralCloningTrainer:
    """Generates demonstration games and pre-trains ColdWarNet via Supervised Learning."""

    def __init__(
        self,
        model: ColdWarNet,
        device: torch.device | str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def generate_demonstration_dataset(
        self, num_games: int = 500, max_steps_per_game: int = 400
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plays heuristic self-play games and records (obs, mask, action, y_win, y_vp)."""
        obs_list = []
        mask_list = []
        action_list = []
        win_list = []
        vp_list = []

        print(f"Generating {num_games} heuristic demonstration games...")
        for g in range(num_games):
            state = ts.GameState()
            seed = np.random.randint(1, 1_000_000_000)
            ts.Engine.init_game(state, seed)

            game_obs = []
            game_masks = []
            game_actions = []
            game_players = []

            step_count = 0
            while not ts.Engine.is_terminal(state) and step_count < max_steps_per_game:
                p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
                obs = ts.extract_observation(state, p)
                mask = ActionEncoder.get_legal_mask(state)

                action_idx = HeuristicPolicy.select_action(state)

                game_obs.append(obs)
                game_masks.append(mask)
                game_actions.append(action_idx)
                game_players.append(1 if p == ts.Player.US else -1)

                ts.Engine.step_flat(state, action_idx)
                step_count += 1

            term_util = ts.Engine.get_terminal_utility(state)
            final_vp = float(state.victory_points)

            # Assign targets aligned to acting player perspective
            for obs, mask, action, player in zip(game_obs, game_masks, game_actions, game_players):
                obs_list.append(obs)
                mask_list.append(mask)
                action_list.append(action)
                y_win = term_util if player == 1 else -term_util
                y_vp = final_vp if player == 1 else -final_vp
                win_list.append(y_win)
                vp_list.append(y_vp)

        print(f"Collected {len(obs_list):,} transition steps from {num_games} games.")

        t_obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        t_mask = torch.tensor(np.array(mask_list), dtype=torch.uint8)
        t_action = torch.tensor(np.array(action_list), dtype=torch.long)
        t_win = torch.tensor(np.array(win_list), dtype=torch.float32).unsqueeze(-1)
        t_vp = torch.tensor(np.array(vp_list), dtype=torch.float32).unsqueeze(-1)
        return t_obs, t_mask, t_action, t_win, t_vp

    def train(
        self,
        dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        epochs: int = 10,
        batch_size: int = 256,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Trains ColdWarNet using Cross-Entropy on actions and MSE on value targets."""
        t_obs, t_mask, t_action, t_win, t_vp = dataset
        ds = TensorDataset(t_obs, t_mask, t_action, t_win, t_vp)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        history: Dict[str, List[float]] = {"loss": [], "policy_loss": [], "value_loss": [], "accuracy": []}
        self.model.train()

        print(f"Starting Behavioral Cloning training for {epochs} epochs (Batch size: {batch_size})...")
        for ep in range(epochs):
            total_loss = 0.0
            total_p_loss = 0.0
            total_v_loss = 0.0
            correct = 0
            total = 0

            for b_obs, b_mask, b_act, b_win, b_vp in loader:
                b_obs = b_obs.to(self.device)
                b_mask = b_mask.to(self.device)
                b_act = b_act.to(self.device)
                b_win = b_win.to(self.device)
                b_vp = b_vp.to(self.device)

                logits, v_win, v_vp = self.model(b_obs, b_mask)

                # Policy Cross-Entropy Loss over legal actions
                policy_loss = F.cross_entropy(logits, b_act)

                # Dual Value Losses
                v_win_loss = F.mse_loss(v_win, b_win)
                v_vp_loss = F.mse_loss(v_vp, b_vp)
                value_loss = v_win_loss + 0.05 * v_vp_loss

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * len(b_act)
                total_p_loss += policy_loss.item() * len(b_act)
                total_v_loss += value_loss.item() * len(b_act)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == b_act).sum().item()
                total += len(b_act)

            avg_loss = total_loss / max(total, 1)
            avg_p_loss = total_p_loss / max(total, 1)
            avg_v_loss = total_v_loss / max(total, 1)
            acc = correct / max(total, 1)

            history["loss"].append(avg_loss)
            history["policy_loss"].append(avg_p_loss)
            history["value_loss"].append(avg_v_loss)
            history["accuracy"].append(acc)

            print(f"Epoch {ep+1:2d}/{epochs:2d} | Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f}) | Action Acc: {acc*100:.2f}%")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        return history
