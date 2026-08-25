"""Vectorized Rollout Buffer for Masked Multi-Agent Twilight Struggle Training."""

from typing import Generator, Tuple
import torch
import numpy as np


class RolloutBuffer:
    """Stores trajectories from TsVectorizedEnv and computes GAE advantages with zero-sum perspective alignment."""

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_dim: int = 4293,
        action_dim: int = 212,
        device: torch.device | str = "cuda",
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Storage buffers allocated on device for fast GPU tensor operations
        self.obs = torch.zeros((buffer_size, num_envs, obs_dim), dtype=torch.float32, device=self.device)
        self.masks = torch.zeros((buffer_size, num_envs, action_dim), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.values_win = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.values_vp = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.players = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)

        # Computed targets
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.returns_win = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.returns_vp = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)

        self.step = 0
        self.full = False

    def reset(self) -> None:
        """Resets the buffer pointer."""
        self.step = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        masks: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        log_probs: torch.Tensor,
        rewards: np.ndarray | torch.Tensor,
        dones: np.ndarray | torch.Tensor,
        values_win: torch.Tensor,
        values_vp: torch.Tensor,
        players: np.ndarray | torch.Tensor,
    ) -> None:
        """Appends a single environment step across all parallel environments."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones)
        if isinstance(players, np.ndarray):
            players = torch.from_numpy(players)

        self.obs[self.step].copy_(obs)
        self.masks[self.step].copy_(masks)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values_win[self.step].copy_(values_win)
        self.values_vp[self.step].copy_(values_vp)
        self.players[self.step].copy_(players)

        self.step += 1
        if self.step >= self.buffer_size:
            self.full = True

    def compute_gae(
        self,
        last_v_win: torch.Tensor,
        last_v_vp: torch.Tensor,
        last_dones: torch.Tensor,
        last_players: torch.Tensor,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) with Zero-Sum Alternating Perspective Alignment."""
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(self.buffer_size)):
            curr_p = self.players[t]
            non_terminal = 1.0 - self.dones[t].float()

            if t == self.buffer_size - 1:
                # Sign alignment with bootstrap state player (+1 if same player, -1 if opponent)
                sign = torch.where(curr_p == last_players, 1.0, -1.0)
                next_val = sign * last_v_win
            else:
                # Sign alignment between step t and step t+1
                next_p = self.players[t + 1]
                sign = torch.where(curr_p == next_p, 1.0, -1.0)
                next_val = sign * self.values_win[t + 1]

            # TD error delta from perspective of acting player at step t
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values_win[t]
            last_gae = delta + gamma * gae_lambda * sign * non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns_win[t] = self.advantages[t] + self.values_win[t]
            self.returns_vp[t] = self.values_vp[t]

        # Normalize advantages per rollout batch
        flat_adv = self.advantages.view(-1)
        mean_adv = flat_adv.mean()
        std_adv = flat_adv.std() + 1e-8
        self.advantages = (self.advantages - mean_adv) / std_adv

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Yields randomized mini-batches for inner-loop SGD updates."""
        total_steps = self.buffer_size * self.num_envs
        indices = torch.randperm(total_steps, device=self.device)

        flat_obs = self.obs.view(total_steps, self.obs_dim)
        flat_masks = self.masks.view(total_steps, self.action_dim)
        flat_actions = self.actions.view(total_steps)
        flat_log_probs = self.log_probs.view(total_steps)
        flat_advantages = self.advantages.view(total_steps)
        flat_returns_win = self.returns_win.view(total_steps)
        flat_returns_vp = self.returns_vp.view(total_steps)

        for start_idx in range(0, total_steps, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                flat_obs[batch_idx],
                flat_masks[batch_idx],
                flat_actions[batch_idx],
                flat_log_probs[batch_idx],
                flat_advantages[batch_idx],
                flat_returns_win[batch_idx],
                flat_returns_vp[batch_idx],
            )
