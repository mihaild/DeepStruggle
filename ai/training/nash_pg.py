"""NashPG (Nash Policy Gradient) Trainer for Twilight Struggle Self-Play."""

import os
import time
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from bindings.ts_env import TsVectorizedEnv
from .rollout_buffer import RolloutBuffer


class NashPGTrainer:
    """NashPG Trainer: Iteratively regularized policy gradient for two-player zero-sum games."""

    def __init__(
        self,
        active_net: ColdWarNet,
        num_envs: int = 64,
        buffer_size: int = 128,
        lr: float = 3e-4,
        eta: float = 0.1,              # NashPG KL-regularization strength (fixed constant)
        clip_eps: float = 0.2,         # PPO clipping epsilon
        ent_coef: float = 0.01,        # Entropy exploration coefficient
        vf_coef: float = 0.5,          # Value loss coefficient
        vp_coef: float = 0.05,         # Auxiliary VP loss weight
        gamma: float = 0.999,          # Game outcome discount factor
        gae_lambda: float = 0.95,      # GAE lambda
        num_epochs: int = 4,           # Inner-loop optimization epochs
        batch_size: int = 256,         # Inner-loop mini-batch size
        ref_update_freq: int = 250_000,# Outer-loop reference update frequency in steps
        max_grad_norm: float = 0.5,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device)
        self.active_net = active_net.to(self.device)

        # Frozen reference network π_ref^(k)
        self.reference_net = create_coldwar_net(self.device)
        self.reference_net.load_state_dict(self.active_net.state_dict())
        self.reference_net.eval()

        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.lr = lr
        self.eta = eta
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vp_coef = vp_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.ref_update_freq = ref_update_freq
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(self.active_net.parameters(), lr=lr, eps=1e-5)
        self.env = TsVectorizedEnv(num_envs=num_envs, base_seed=int(time.time()))
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            num_envs=num_envs,
            obs_dim=ColdWarNet.TOTAL_OBS_SIZE,
            action_dim=ColdWarNet.ACTION_SPACE_SIZE,
            device=self.device,
        )

        self.total_env_steps = 0
        self.steps_since_ref_update = 0
        self.outer_loop_round = 0
        self.episode_history: List[Dict[str, Any]] = []

    def collect_rollouts(self) -> Dict[str, Any]:
        """Collects buffer_size steps from parallel vectorized environments using active_net."""
        self.buffer.reset()
        obs_np, masks_np, _ = self.env.reset_all()

        t0 = time.time()
        completed_episodes = []
        dones_np = np.zeros(self.env.num_envs, dtype=bool)

        for _ in range(self.buffer_size):
            obs_t = torch.from_numpy(obs_np).float().to(self.device)
            masks_t = torch.from_numpy(masks_np).to(self.device)

            with torch.no_grad():
                actions_t, log_probs_t, v_win_t, v_vp_t, _ = self.active_net.sample_action(
                    obs_t, masks_t, temperature=1.0
                )

            actions_np = actions_t.cpu().numpy()
            prev_players = self.env._get_batch_info()["decision_players"]

            next_obs_np, next_masks_np, rewards_np, dones_np, info = self.env.step(actions_np)

            self.buffer.add(
                obs=obs_t,
                masks=masks_t,
                actions=actions_t,
                log_probs=log_probs_t,
                rewards=rewards_np,
                dones=dones_np,
                values_win=v_win_t,
                values_vp=v_vp_t,
                players=prev_players,
            )

            if "completed_episodes" in info and info["completed_episodes"]:
                completed_episodes.extend(info["completed_episodes"])

            obs_np = next_obs_np
            masks_np = next_masks_np

        # Evaluate last state for GAE bootstrapping
        last_obs_t = torch.from_numpy(obs_np).float().to(self.device)
        last_masks_t = torch.from_numpy(masks_np).to(self.device)
        with torch.no_grad():
            _, last_v_win, last_v_vp = self.active_net(last_obs_t, last_masks_t)
            last_dones = torch.from_numpy(dones_np).to(self.device)

        last_players = torch.from_numpy(self.env._get_batch_info()["decision_players"]).to(self.device)
        self.buffer.compute_gae(
            last_v_win=last_v_win.squeeze(-1),
            last_v_vp=last_v_vp.squeeze(-1),
            last_dones=last_dones,
            last_players=last_players,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        steps_collected = self.buffer_size * self.num_envs
        self.total_env_steps += steps_collected
        self.steps_since_ref_update += steps_collected
        rollout_time = time.time() - t0

        return {
            "steps": steps_collected,
            "rollout_time": rollout_time,
            "fps": steps_collected / max(rollout_time, 1e-6),
            "completed_episodes": completed_episodes,
        }

    def train_iteration(self, **kwargs) -> Dict[str, Any]:
        rollout_info = self.collect_rollouts()
        train_metrics = self.train_step()
        train_metrics.update(rollout_info)
        return train_metrics

    def train_step(self) -> Dict[str, float]:
        """Runs the inner-loop NashPG update over the collected trajectory buffer."""
        self.active_net.train()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        kl_loss_accum = 0.0
        val_loss_accum = 0.0
        entropy_accum = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            for b_obs, b_mask, b_act, b_old_log_probs, b_adv, b_ret_win, b_ret_vp in self.buffer.get_batches(self.batch_size):
                cur_log_probs, cur_entropy, cur_v_win, cur_v_vp = self.active_net.evaluate_actions(
                    b_obs, b_mask, b_act
                )

                # 1. PPO Clipped Surrogate Loss
                ratio = torch.exp(cur_log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                # 2. NashPG Iterative KL Divergence Penalty vs Frozen Reference Policy π_ref^(k)
                if self.eta > 0.0:
                    with torch.no_grad():
                        ref_masked_logits, _, _ = self.reference_net(b_obs, b_mask)
                        ref_log_probs = F.log_softmax(ref_masked_logits, dim=-1)

                    cur_masked_logits, _, _ = self.active_net(b_obs, b_mask)
                    cur_probs = F.softmax(cur_masked_logits, dim=-1)
                    cur_log_p = F.log_softmax(cur_masked_logits, dim=-1)

                    # Compute KL divergence over legal actions: sum_a p(a) * (log p(a) - log ref(a))
                    kl_div = torch.sum(cur_probs * (cur_log_p - ref_log_probs), dim=-1).mean()
                    kl_loss = self.eta * kl_div
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    kl_div = torch.tensor(0.0, device=self.device)

                # Policy Loss = PPO Loss + KL Penalty - Entropy Bonus
                policy_loss = ppo_loss + kl_loss - self.ent_coef * cur_entropy.mean()

                # 3. Dual Value Loss (Win/Loss + Auxiliary VP)
                v_win_loss = F.mse_loss(cur_v_win, b_ret_win)
                v_vp_loss = F.mse_loss(cur_v_vp, b_ret_vp)
                val_loss = v_win_loss + self.vp_coef * v_vp_loss

                # Total Objective
                loss = policy_loss + self.vf_coef * val_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += loss.item()
                policy_loss_accum += ppo_loss.item()
                kl_loss_accum += kl_div.item() if isinstance(kl_div, torch.Tensor) else float(kl_div)
                val_loss_accum += val_loss.item()
                entropy_accum += cur_entropy.mean().item()
                num_updates += 1

        # 4. Check Outer-Loop Reference Update condition
        if self.steps_since_ref_update >= self.ref_update_freq:
            self.outer_loop_round += 1
            self.reference_net.load_state_dict(self.active_net.state_dict())
            self.steps_since_ref_update = 0
            print(f"\n[NashPG Outer Loop] Updated reference policy π_ref at {self.total_env_steps:,} total steps (Round {self.outer_loop_round})")

        return {
            "loss": total_loss_accum / max(num_updates, 1),
            "policy_loss": policy_loss_accum / max(num_updates, 1),
            "kl_div": kl_loss_accum / max(num_updates, 1),
            "val_loss": val_loss_accum / max(num_updates, 1),
            "entropy": entropy_accum / max(num_updates, 1),
        }

    def train_iterations(
        self, num_iterations: int = 100, log_interval: int = 10, save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Main training loop executing rollout collection + inner-loop updates."""
        history = []
        print(f"\n=======================================================")
        print(f"Starting NashPG Self-Play Training on {self.device}")
        print(f"Parallel Envs: {self.num_envs} | Buffer: {self.buffer_size} | Batch Size: {self.batch_size}")
        print(f"NashPG η: {self.eta} | Clip ε: {self.clip_eps} | Outer Update Freq: {self.ref_update_freq:,}")
        print(f"=======================================================\n")

        all_completed = []

        for it in range(1, num_iterations + 1):
            rollout_stats = self.collect_rollouts()
            train_stats = self.train_step()

            completed = rollout_stats["completed_episodes"]
            all_completed.extend(completed)

            combined_stats = {
                "iteration": it,
                "total_steps": self.total_env_steps,
                "fps": rollout_stats["fps"],
                **train_stats,
            }
            history.append(combined_stats)

            if it % log_interval == 0 or it == 1:
                recent_eps = all_completed[-100:] if all_completed else []
                if recent_eps:
                    us_wins = sum(1 for e in recent_eps if e.get("winner") == "US")
                    ussr_wins = sum(1 for e in recent_eps if e.get("winner") == "USSR")
                    avg_len = np.mean([e.get("length", 0) for e in recent_eps])
                    avg_vp = np.mean([e.get("victory_points", 0) for e in recent_eps])
                    win_str = f"US Win: {us_wins/len(recent_eps)*100:.1f}% | USSR Win: {ussr_wins/len(recent_eps)*100:.1f}% | Avg Len: {avg_len:.1f} | Avg VP: {avg_vp:+.1f}"
                else:
                    win_str = "No completed episodes yet"

                print(
                    f"Iter {it:4d}/{num_iterations:4d} | Steps: {self.total_env_steps:8,d} ({rollout_stats['fps']:6.0f} step/s) | "
                    f"Loss: {train_stats['loss']:.4f} (Pol: {train_stats['policy_loss']:.4f}, Val: {train_stats['val_loss']:.4f}, KL: {train_stats['kl_div']:.4f}, Ent: {train_stats['entropy']:.3f}) | "
                    f"{win_str}"
                )

            if save_path and (it % (log_interval * 5) == 0 or it == num_iterations):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.active_net.state_dict(), save_path)
                print(f"  --> Saved active policy checkpoint to {save_path}")

        return history
