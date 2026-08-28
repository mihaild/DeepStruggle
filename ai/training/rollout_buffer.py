"""Vectorized Rollout Buffer for Masked Multi-Agent Twilight Struggle Training with GAE Credit Slicing."""

from typing import Generator, Tuple, Optional
import torch
import numpy as np


class RolloutBuffer:
    """Stores trajectories from TsVectorizedEnv and computes GAE advantages with zero-sum perspective alignment and credit slicing."""

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
        self.turns = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)
        self.vps = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.held_scoring_us = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.held_scoring_ussr = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.opp_hands = torch.zeros((buffer_size, num_envs, 110), dtype=torch.float32, device=self.device)
        self.oracle_values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)

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
        turns: Optional[np.ndarray | torch.Tensor] = None,
        vps: Optional[np.ndarray | torch.Tensor] = None,
        held_scoring_us: Optional[np.ndarray | torch.Tensor] = None,
        held_scoring_ussr: Optional[np.ndarray | torch.Tensor] = None,
        opp_hands: Optional[np.ndarray | torch.Tensor] = None,
        oracle_values: Optional[torch.Tensor] = None,
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
        if isinstance(turns, np.ndarray):
            turns = torch.from_numpy(turns)

        self.obs[self.step].copy_(obs)
        self.masks[self.step].copy_(masks)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values_win[self.step].copy_(values_win)
        self.values_vp[self.step].copy_(values_vp)
        self.players[self.step].copy_(players)
        if turns is not None:
            if isinstance(turns, np.ndarray):
                turns = torch.from_numpy(turns)
            self.turns[self.step].copy_(turns)
        if vps is not None:
            if isinstance(vps, np.ndarray):
                vps = torch.from_numpy(vps)
            self.vps[self.step].copy_(vps)
        if held_scoring_us is not None:
            if isinstance(held_scoring_us, np.ndarray):
                held_scoring_us = torch.from_numpy(held_scoring_us)
            self.held_scoring_us[self.step].copy_(held_scoring_us)
        if held_scoring_ussr is not None:
            if isinstance(held_scoring_ussr, np.ndarray):
                held_scoring_ussr = torch.from_numpy(held_scoring_ussr)
            self.held_scoring_ussr[self.step].copy_(held_scoring_ussr)
        if opp_hands is not None:
            if isinstance(opp_hands, np.ndarray):
                opp_hands = torch.from_numpy(opp_hands)
            self.opp_hands[self.step].copy_(opp_hands)
        if oracle_values is not None:
            self.oracle_values[self.step].copy_(oracle_values)

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
        gae_lambda: float = 0.98,
        slice_turn_boundaries: bool = False,
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) with Zero-Sum Alternating Perspective Alignment."""
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        pending_hs_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_us = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_ussr = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_turn = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        last_ret_vp = last_v_vp.clone()

        for t in reversed(range(self.buffer_size)):
            curr_p = self.players[t]
            non_terminal = 1.0 - self.dones[t].float()

            if t == self.buffer_size - 1:
                sign = (curr_p * last_players).float()
                next_val = sign * last_v_win
                next_ret_vp = sign * last_ret_vp
            else:
                next_p = self.players[t + 1]
                sign = (curr_p * next_p).float()
                next_val = sign * self.values_win[t + 1]
                next_ret_vp = sign * self.returns_vp[t + 1]

                if slice_turn_boundaries:
                    turn_boundary_mask = (self.turns[t] == self.turns[t + 1]).float()
                    non_terminal = non_terminal * turn_boundary_mask

            # Update pending held scoring blunder states for terminal steps
            for e in range(self.num_envs):
                if self.dones[t, e]:
                    if self.held_scoring_us[t, e] or self.held_scoring_ussr[t, e]:
                        pending_hs_active[e] = True
                        pending_hs_us[e] = self.held_scoring_us[t, e]
                        pending_hs_ussr[e] = self.held_scoring_ussr[t, e]
                        pending_hs_turn[e] = self.turns[t, e]
                    else:
                        pending_hs_active[e] = False

            # Check if pending held-scoring blunder applies to step t
            for e in range(self.num_envs):
                if pending_hs_active[e]:
                    if self.turns[t, e] == pending_hs_turn[e]:
                        p = int(curr_p[e])
                        if p == 1:  # US
                            if pending_hs_us[e]:
                                self.returns_win[t, e] = -1.0
                                self.advantages[t, e] = -1.0 - self.values_win[t, e]
                            else:
                                self.returns_win[t, e] = self.values_win[t, e]
                                self.advantages[t, e] = 0.0
                        else:  # USSR
                            if pending_hs_ussr[e]:
                                self.returns_win[t, e] = -1.0
                                self.advantages[t, e] = -1.0 - self.values_win[t, e]
                            else:
                                self.returns_win[t, e] = self.values_win[t, e]
                                self.advantages[t, e] = 0.0
                        last_gae[e] = self.advantages[t, e]
                        continue
                    else:
                        # Turn boundary reached! Do not propagate held scoring blunder before current turn
                        pending_hs_active[e] = False
                        last_gae[e] = 0.0

                # Standard zero-sum Bellman TD error & GAE
                delta = self.rewards[t, e] + gamma * next_val[e] * non_terminal[e] - self.values_win[t, e]
                last_gae[e] = delta + gamma * gae_lambda * sign[e] * non_terminal[e] * last_gae[e]
                self.advantages[t, e] = last_gae[e]
                self.returns_win[t, e] = self.advantages[t, e] + self.values_win[t, e]

            # VP Return: real VP ground truth at terminal / delta VP
            # vps[t] is normalized VP in [-1.0, 1.0] from player perspective
            curr_vp_norm = self.vps[t] * curr_p.float() / 20.0
            self.returns_vp[t] = torch.where(
                self.dones[t],
                curr_vp_norm,
                curr_vp_norm * 0.1 + 0.9 * non_terminal * next_ret_vp
            )

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

    def get_batches_with_oracle(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Yields randomized mini-batches including opponent hands and oracle value targets for V4 training."""
        total_steps = self.buffer_size * self.num_envs
        indices = torch.randperm(total_steps, device=self.device)

        flat_obs = self.obs.view(total_steps, self.obs_dim)
        flat_masks = self.masks.view(total_steps, self.action_dim)
        flat_actions = self.actions.view(total_steps)
        flat_log_probs = self.log_probs.view(total_steps)
        flat_advantages = self.advantages.view(total_steps)
        flat_returns_win = self.returns_win.view(total_steps)
        flat_returns_vp = self.returns_vp.view(total_steps)
        flat_opp_hands = self.opp_hands.view(total_steps, 110)
        flat_oracle_values = self.oracle_values.view(total_steps)

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
                flat_opp_hands[batch_idx],
                flat_oracle_values[batch_idx],
            )
