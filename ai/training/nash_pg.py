"""NashPG (Nash Policy Gradient) and Oracle-Guided Multi-Task Trainers for Twilight Struggle Self-Play.

Implements:
1. BaseNashPGTrainer: Unified base class managing vectorized C++ rollouts, temperature scheduling,
   zero-sum GAE credit slicing, and outer-loop reference policy anchoring π_ref^(k).
2. NashPGTrainer: Standard Nash Policy Gradient trainer for ColdWarNet V1, V2, and V3.
3. OracleGuidedNashPGTrainer: Extended multi-task trainer for ColdWarNetV4 incorporating:
   - Suphx-style Oracle Critic supervision and Knowledge Distillation.
   - Auxiliary Opponent Hand Belief Head multi-label BCE loss.
"""

import copy
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bindings.ts_env import TsVectorizedEnv
from .rollout_buffer import RolloutBuffer


class BaseNashPGTrainer:
    """Abstract base class for Nash Policy Gradient self-play trainers."""

    def __init__(
        self,
        active_net: nn.Module,
        reference_net: Optional[nn.Module] = None,
        env: Optional[TsVectorizedEnv] = None,
        num_envs: int = 64,
        buffer_size: int = 128,
        batch_size: int = 4096,
        lr: float = 3e-4,
        eta: float = 0.1,              # NashPG KL-regularization strength
        clip_eps: float = 0.2,         # PPO clipping epsilon
        ent_coef: float = 0.01,        # Entropy exploration coefficient
        vf_coef: float = 0.5,          # Value loss coefficient
        vp_coef: float = 0.05,         # Auxiliary VP loss weight
        gamma: float = 0.999,          # Game outcome discount factor
        gae_lambda: float = 0.98,      # GAE lambda
        num_epochs: int = 4,           # Inner-loop optimization epochs per iteration
        ref_update_freq: int = 200_000,# Outer-loop reference update frequency in steps
        max_grad_norm: float = 1.0,
        slice_turn_boundaries: bool = False,
        temperature_schedule: bool = True,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device if (torch.cuda.is_available() and device == "cuda") else ("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
        self.active_net = active_net.to(self.device)

        if reference_net is not None:
            self.reference_net = reference_net.to(self.device)
        else:
            self.reference_net = copy.deepcopy(self.active_net).to(self.device)
        self.reference_net.eval()

        self.num_envs = num_envs if env is None else env.num_envs
        self.buffer_size = buffer_size
        self.batch_size = min(batch_size, self.num_envs * self.buffer_size)
        self.lr = lr
        self.eta = eta
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vp_coef = vp_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.ref_update_freq = ref_update_freq
        self.max_grad_norm = max_grad_norm
        self.slice_turn_boundaries = slice_turn_boundaries
        self.temperature_schedule = temperature_schedule

        self.optimizer = torch.optim.AdamW(self.active_net.parameters(), lr=lr, weight_decay=1e-4)

        self.env = env if env is not None else TsVectorizedEnv(num_envs=self.num_envs, base_seed=int(time.time()))
        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            num_envs=self.num_envs,
            obs_dim=4293,
            action_dim=212,
            device=self.device,
        )

        # Exploration temperature schedule
        if self.temperature_schedule and self.num_envs > 1:
            temps = np.zeros(self.num_envs, dtype=np.float32)
            temps[0 : self.num_envs // 4] = 0.15
            temps[self.num_envs // 4 : self.num_envs // 2] = 0.50
            temps[self.num_envs // 2 : 3 * self.num_envs // 4] = 0.10
            temps[3 * self.num_envs // 4 :] = 0.35
            self.env_temps = torch.from_numpy(temps).unsqueeze(1).to(self.device)
        else:
            self.env_temps = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)

        self._obs_np, self._masks_np, _ = self.env.reset_all()
        self._dones_np = np.zeros(self.num_envs, dtype=np.float32)
        self._info: Dict[str, Any] = {"acting_players": np.zeros(self.num_envs, dtype=np.int32)}

        self.total_env_steps = 0
        self.steps_since_ref_update = 0
        self.total_iterations = 0

    def set_reward_calculator(self, reward_calc: Any) -> None:
        self.env.reward_calc = reward_calc

    def set_slice_turn_boundaries(self, slice_boundaries: bool) -> None:
        self.slice_turn_boundaries = slice_boundaries

    def update_reference_policy(self) -> None:
        """Updates the frozen reference anchor: π_ref^(k+1) ← π_θ^(k)."""
        self.reference_net.load_state_dict(self.active_net.state_dict())
        self.reference_net.to(self.device)
        self.reference_net.eval()
        print(f"\n[NashPG Outer Loop] Updated reference policy π_ref at {self.total_env_steps:,} total steps (Round {self.total_iterations})", flush=True)

    def collect_rollouts(self) -> Dict[str, Any]:
        """Collects on-policy rollouts across all parallel environments."""
        t0 = time.time()
        self.active_net.eval()
        self.buffer.reset()
        completed_episodes: List[Dict[str, Any]] = []

        for _ in range(self.buffer_size):
            obs_t = torch.from_numpy(self._obs_np).float().to(self.device)
            masks_t = torch.from_numpy(self._masks_np).to(self.device)

            with torch.no_grad():
                logits, v_win_t, v_vp_t = self.active_net(obs_t, masks_t)
                v_win_t = v_win_t.squeeze(-1)
                v_vp_t = v_vp_t.squeeze(-1)

                if self.temperature_schedule and self.num_envs > 1:
                    scaled_logits = logits / self.env_temps
                    scaled_probs = F.softmax(scaled_logits, dim=-1)
                    actions_t = torch.multinomial(scaled_probs, 1).squeeze(1)
                else:
                    actions_t = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)

                unscaled_log_probs = F.log_softmax(logits, dim=-1)
                log_probs_t = unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            actions_np = actions_t.cpu().numpy()
            next_obs_np, next_masks_np, rewards_np, self._dones_np, self._info = self.env.step(actions_np)

            opp_hands = self._info.get("opponent_hands", None)

            self.buffer.add(
                obs=obs_t,
                masks=masks_t,
                actions=actions_t,
                log_probs=log_probs_t,
                rewards=rewards_np,
                dones=torch.from_numpy(self._dones_np).float().to(self.device),
                values_win=v_win_t,
                values_vp=v_vp_t,
                players=torch.from_numpy(self._info["acting_players"]).to(self.device),
                turns=torch.from_numpy(self._info["turns"]).to(self.device) if "turns" in self._info else None,
                vps=torch.from_numpy(self._info["victory_points"]).float().to(self.device) if "victory_points" in self._info else None,
                held_scoring_us=torch.from_numpy(self._info["held_scoring_us"]).to(self.device) if "held_scoring_us" in self._info else None,
                held_scoring_ussr=torch.from_numpy(self._info["held_scoring_ussr"]).to(self.device) if "held_scoring_ussr" in self._info else None,
                opp_hands=opp_hands,
            )

            if "completed_episodes" in self._info and self._info["completed_episodes"]:
                completed_episodes.extend(self._info["completed_episodes"])

            self._obs_np = next_obs_np
            self._masks_np = next_masks_np

        # Evaluate last state for GAE bootstrapping
        last_obs_t = torch.from_numpy(self._obs_np).float().to(self.device)
        last_masks_t = torch.from_numpy(self._masks_np).to(self.device)
        with torch.no_grad():
            _, last_v_win, last_v_vp = self.active_net(last_obs_t, last_masks_t)
            last_dones = torch.from_numpy(self._dones_np).to(self.device)

        last_players = torch.from_numpy(self.env._get_batch_info()["decision_players"]).to(self.device)
        self.buffer.compute_gae(
            last_v_win=last_v_win.squeeze(-1),
            last_v_vp=last_v_vp.squeeze(-1),
            last_dones=last_dones,
            last_players=last_players,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            slice_turn_boundaries=self.slice_turn_boundaries,
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

    def train_step(self) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement train_step")

    def train_iteration(self) -> Dict[str, float]:
        """Runs one full training iteration (rollout collection + inner SGD epochs + reference check)."""
        self.total_iterations += 1
        rollout_metrics = self.collect_rollouts()
        train_metrics = self.train_step()

        if self.steps_since_ref_update >= self.ref_update_freq:
            self.update_reference_policy()
            self.steps_since_ref_update = 0

        combined = {**rollout_metrics, **train_metrics}
        return combined


class NashPGTrainer(BaseNashPGTrainer):
    """Standard NashPG Trainer for ColdWarNet V1, V2, and V3."""

    def train_step(self) -> Dict[str, float]:
        self.active_net.train()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        val_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            for b_obs, b_mask, b_act, b_old_lp, b_adv, b_ret_win, b_ret_vp in self.buffer.get_batches(self.batch_size):
                cur_logits, cur_v_win, cur_v_vp = self.active_net(b_obs, b_mask)
                cur_v_win = cur_v_win.squeeze(-1)
                cur_v_vp = cur_v_vp.squeeze(-1)

                cur_dist = torch.distributions.Categorical(logits=cur_logits)
                cur_lp = cur_dist.log_prob(b_act)
                cur_entropy = cur_dist.entropy()

                ratio = torch.exp(cur_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                clip_frac = ((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)).float().mean().item()

                with torch.no_grad():
                    ref_logits, _, _ = self.reference_net(b_obs, b_mask)
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)

                cur_p = F.softmax(cur_logits, dim=-1)
                cur_log_p = F.log_softmax(cur_logits, dim=-1)
                kl_div = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1).mean()

                policy_loss = ppo_loss + self.eta * kl_div - self.ent_coef * cur_entropy.mean()
                val_loss = F.mse_loss(cur_v_win, b_ret_win) + self.vp_coef * F.mse_loss(cur_v_vp, b_ret_vp)
                loss = policy_loss + self.vf_coef * val_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += loss.item()
                policy_loss_accum += ppo_loss.item()
                val_loss_accum += val_loss.item()
                kl_accum += kl_div.item()
                entropy_accum += cur_entropy.mean().item()
                clip_frac_accum += clip_frac
                num_updates += 1

        return {
            "loss": total_loss_accum / max(1, num_updates),
            "policy_loss": policy_loss_accum / max(1, num_updates),
            "val_loss": val_loss_accum / max(1, num_updates),
            "kl_div": kl_accum / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
        }


class OracleGuidedNashPGTrainer(BaseNashPGTrainer):
    """Oracle-Guided Multi-Task NashPG Trainer for ColdWarNetV4.

    Adds:
    1. Opponent Belief Head Loss: Binary Cross-Entropy against true hidden opponent cards.
    2. Privileged Oracle Critic Head Loss: MSE against terminal returns.
    3. Public Critic Distillation Loss: MSE against Oracle Critic evaluation.
    """

    def __init__(
        self,
        active_net: nn.Module,
        reference_net: Optional[nn.Module] = None,
        env: Optional[TsVectorizedEnv] = None,
        belief_loss_coef: float = 0.10,    # Weight for auxiliary belief BCE loss
        oracle_loss_coef: float = 0.25,    # Weight for privileged oracle critic MSE
        distill_loss_coef: float = 0.25,   # Weight for public critic distillation MSE
        **kwargs: Any,
    ):
        if "batch_size" in kwargs and kwargs["batch_size"] > 1024:
            kwargs["batch_size"] = 1024
        super().__init__(active_net=active_net, reference_net=reference_net, env=env, **kwargs)
        self.batch_size = min(self.batch_size, 1024)
        self.belief_loss_coef = belief_loss_coef
        self.oracle_loss_coef = oracle_loss_coef
        self.distill_loss_coef = distill_loss_coef

    def train_step(self) -> Dict[str, float]:
        self.active_net.train()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        val_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        belief_loss_accum = 0.0
        oracle_loss_accum = 0.0
        distill_loss_accum = 0.0
        num_updates = 0

        for _ in range(self.num_epochs):
            for b_obs, b_mask, b_act, b_old_lp, b_adv, b_ret_win, b_ret_vp, b_opp_hands, _ in self.buffer.get_batches_with_oracle(self.batch_size):
                if hasattr(self.active_net, "forward_all"):
                    forward_all_fn = getattr(self.active_net, "forward_all")
                    cur_logits, cur_v_win, cur_v_vp, b_pred_belief, b_oracle_val = forward_all_fn(b_obs, b_mask, b_opp_hands)
                    cur_v_win = cur_v_win.squeeze(-1)
                    cur_v_vp = cur_v_vp.squeeze(-1)
                    b_oracle_val = b_oracle_val.squeeze(-1)
                else:
                    cur_logits, cur_v_win, cur_v_vp = self.active_net(b_obs, b_mask)
                    cur_v_win = cur_v_win.squeeze(-1)
                    cur_v_vp = cur_v_vp.squeeze(-1)
                    b_pred_belief = getattr(self.active_net, "predict_belief")(b_obs)
                    b_oracle_val = getattr(self.active_net, "evaluate_oracle")(b_obs, b_opp_hands).squeeze(-1)

                cur_dist = torch.distributions.Categorical(logits=cur_logits)
                cur_lp = cur_dist.log_prob(b_act)
                cur_entropy = cur_dist.entropy()

                ratio = torch.exp(cur_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                clip_frac = ((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)).float().mean().item()

                with torch.no_grad():
                    ref_logits, _, _ = self.reference_net(b_obs, b_mask)
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)

                cur_p = F.softmax(cur_logits, dim=-1)
                cur_log_p = F.log_softmax(cur_logits, dim=-1)
                kl_div = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1).mean()

                policy_loss = ppo_loss + self.eta * kl_div - self.ent_coef * cur_entropy.mean()
                val_loss = F.mse_loss(cur_v_win, b_ret_win) + self.vp_coef * F.mse_loss(cur_v_vp, b_ret_vp)

                # 1. Opponent Belief Head Loss (BCE against ground truth hidden cards)
                belief_loss = F.binary_cross_entropy(b_pred_belief, b_opp_hands.float())

                # 2. Privileged Oracle Critic Loss & Public Value Distillation
                oracle_loss = F.mse_loss(b_oracle_val, b_ret_win)
                distill_loss = F.mse_loss(cur_v_win, b_oracle_val.detach())

                # Multi-Task Combined Loss
                loss = (
                    policy_loss
                    + self.vf_coef * val_loss
                    + self.belief_loss_coef * belief_loss
                    + self.oracle_loss_coef * oracle_loss
                    + self.distill_loss_coef * distill_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += loss.item()
                policy_loss_accum += ppo_loss.item()
                val_loss_accum += val_loss.item()
                kl_accum += kl_div.item()
                entropy_accum += cur_entropy.mean().item()
                clip_frac_accum += clip_frac
                belief_loss_accum += belief_loss.item()
                oracle_loss_accum += oracle_loss.item()
                distill_loss_accum += distill_loss.item()
                num_updates += 1

        return {
            "loss": total_loss_accum / max(1, num_updates),
            "policy_loss": policy_loss_accum / max(1, num_updates),
            "val_loss": val_loss_accum / max(1, num_updates),
            "kl_div": kl_accum / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
            "belief_loss": belief_loss_accum / max(1, num_updates),
            "oracle_loss": oracle_loss_accum / max(1, num_updates),
            "distill_loss": distill_loss_accum / max(1, num_updates),
        }
