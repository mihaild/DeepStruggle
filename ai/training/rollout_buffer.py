"""Vectorized Rollout Buffer for Masked Multi-Agent Twilight Struggle Training with GAE Credit Slicing."""

from typing import Dict, Generator, Tuple, Optional
import torch
import numpy as np

# Advantages below this magnitude (post-normalisation) carry effectively no learning
# signal for the decision they are attached to.
NEAR_ZERO_ADVANTAGE_EPS = 0.01


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    """Fraction of the return variance the value head accounts for: ``1 - Var(G - V) / Var(G)``.

    1.0 means the critic predicts returns exactly, 0.0 means it does no better than
    predicting the mean return, and negative means it is worse than that constant.
    Returns 0.0 when the returns are (near-)constant, where the ratio is undefined.
    """
    y = returns.detach().reshape(-1).float()
    v = values.detach().reshape(-1).float()
    if y.numel() == 0 or y.numel() != v.numel():
        return 0.0
    var_y = torch.var(y, unbiased=False)
    if float(var_y) < 1e-12:
        return 0.0
    return float(1.0 - torch.var(y - v, unbiased=False) / var_y)


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
        self.defcon_blunder = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)
        self.opp_hands = torch.zeros((buffer_size, num_envs, 110), dtype=torch.float32, device=self.device)

        # Computed targets
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.returns_win = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.returns_vp = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)

        self.step = 0
        self.full = False
        # Standard deviation of the advantages before per-rollout normalisation, kept
        # for diagnostics (post-normalisation std is ~1.0 by construction).
        self.raw_advantage_std = 0.0

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
        defcon_blunder: Optional[np.ndarray | torch.Tensor] = None,
        opp_hands: Optional[np.ndarray | torch.Tensor] = None,
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
        if defcon_blunder is not None:
            if isinstance(defcon_blunder, np.ndarray):
                defcon_blunder = torch.from_numpy(defcon_blunder)
            self.defcon_blunder[self.step].copy_(defcon_blunder)
        if opp_hands is not None:
            if isinstance(opp_hands, np.ndarray):
                opp_hands = torch.from_numpy(opp_hands)
            self.opp_hands[self.step].copy_(opp_hands)

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
        blunder_window: bool = True,
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) with Zero-Sum Alternating Perspective Alignment.

        Credit for an *unprovoked* blunder loss -- holding a scoring card past the end of a
        turn, or driving DEFCON to 1 by one's own choice -- is confined to the turn the
        blunder happened in, and the opponent is shielded from the resulting windfall. The
        play that preceded the blunder was not necessarily bad, and the opponent did not
        earn the win, so neither should be credited for it.

        This is deliberately conditional on the episode actually ending in a blunder.
        ``slice_turn_boundaries`` applies the same truncation to *every* episode including
        clean wins, which strips the outcome signal from all but the final turn of the
        ~80% of games that end normally; it is retained only as an ablation knob and
        defaults off.
        """
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Pending blunder window, armed at a terminal step and disarmed at the turn boundary.
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

            # Arm/disarm the blunder window at terminal steps. Held scoring can implicate
            # both players at once; an unprovoked DEFCON-1 suicide implicates exactly the
            # player who chose it.
            for e in range(self.num_envs):
                if self.dones[t, e]:
                    blunder_us = bool(self.held_scoring_us[t, e])
                    blunder_ussr = bool(self.held_scoring_ussr[t, e])
                    if blunder_window:
                        db = int(self.defcon_blunder[t, e])
                        if db == 1:
                            blunder_us = True
                        elif db == -1:
                            blunder_ussr = True
                    if blunder_window and (blunder_us or blunder_ussr):
                        pending_hs_active[e] = True
                        pending_hs_us[e] = blunder_us
                        pending_hs_ussr[e] = blunder_ussr
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
        self.raw_advantage_std = float(std_adv)
        self.advantages = (self.advantages - mean_adv) / std_adv

    def diagnostics(self) -> Dict[str, float]:
        """Value-head and advantage-distribution health metrics for the current rollout.

        Must be called after ``compute_gae``; the advantage figures describe the
        normalised advantages that the policy update actually consumes.
        """
        adv = self.advantages.detach().reshape(-1)
        near_zero = (adv.abs() < NEAR_ZERO_ADVANTAGE_EPS).float().mean() if adv.numel() > 0 else torch.zeros(())
        return {
            "explained_variance": explained_variance(self.returns_win, self.values_win),
            "adv_std": float(adv.std()) if adv.numel() > 1 else 0.0,
            "adv_std_raw": self.raw_advantage_std,
            "adv_frac_near_zero": float(near_zero),
        }

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
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
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
            )
