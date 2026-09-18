"""Vectorized Rollout Buffer for Masked Multi-Agent Twilight Struggle Training with GAE Credit Slicing."""

from typing import Dict, Generator, Tuple, Optional
import torch

from bindings.ts_env import obs_size
import numpy as np
from bindings.action_encoder import ActionEncoder

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
        obs_dim: Optional[int] = None,
        action_dim: int = ActionEncoder.FLAT_ACTION_SIZE,
        device: torch.device | str = "cuda",
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        # Resolved here, not in the signature: a default evaluated at import time reads an
        # engine constant while the module is loading, which is how an older build once took
        # down `import bindings` for every consumer.
        self.obs_dim = int(obs_size() if obs_dim is None else obs_dim)
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Storage buffers allocated on device for fast GPU tensor operations
        self.obs = torch.zeros((buffer_size, num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.masks = torch.zeros((buffer_size, num_envs, action_dim), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.values_win = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.values_vp = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.players = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)
        # 1 where the transition was chosen by the policy being trained. Under an
        # opponent pool the frozen opponent's transitions are still *stored* -- GAE is a
        # backward recursion over consecutive steps, and dropping them would leave the
        # learner's non-consecutive -- but they must not receive policy gradient.
        self.learner = torch.ones((buffer_size, num_envs), dtype=torch.float32,
                                  device=self.device)
        self.turns = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)
        self.vps = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.held_scoring_us = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.held_scoring_ussr = torch.zeros((buffer_size, num_envs), dtype=torch.bool, device=self.device)
        self.defcon_blunder = torch.zeros((buffer_size, num_envs), dtype=torch.int8, device=self.device)
        #: V(s_{t+1}, p_t) -- the state AFTER this step, evaluated from the perspective of the
        #: player who just moved. Filled only when the same-perspective bootstrap is enabled; the
        #: default path negates the next step's value instead, which crosses an information-set
        #: boundary. See `compute_gae`.
        self.next_values_own = torch.zeros((buffer_size, num_envs), dtype=torch.float32,
                                           device=self.device)
        self.has_next_values_own = False

        # Computed targets
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        #: P15-X4b. The searcher's visit distribution where a decision was searched, and the
        #: flag saying where that is. The flag is needed rather than inferred: an unsearched
        #: step is an all-zero row, which is not distinguishable from a degenerate target.
        self.search_pi = torch.zeros((buffer_size, num_envs, action_dim),
                                     dtype=torch.float32, device=self.device)
        self.has_search = torch.zeros((buffer_size, num_envs), dtype=torch.float32,
                                      device=self.device)
        self.returns_win = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.returns_vp = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)
        # Auxiliary target for the DEFCON-risk head: 1 where the acting player is about to
        # lose the game to its own DEFCON-1 choice within defcon_risk_horizon steps.
        self.defcon_risk_target = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=self.device)

        self.step = 0
        self.full = False
        #: Pre-normalisation advantage statistics split by acting side (US = +1, USSR = -1),
        #: filled by compute_gae. Empty until then.
        self.side_advantage: Dict[str, Dict[str, float]] = {}
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
        learner: Optional[np.ndarray | torch.Tensor] = None,
        turns: Optional[np.ndarray | torch.Tensor] = None,
        vps: Optional[np.ndarray | torch.Tensor] = None,
        held_scoring_us: Optional[np.ndarray | torch.Tensor] = None,
        held_scoring_ussr: Optional[np.ndarray | torch.Tensor] = None,
        defcon_blunder: Optional[np.ndarray | torch.Tensor] = None,
        next_values_own: Optional[torch.Tensor] = None,
        search_pi: Optional[torch.Tensor] = None,
        has_search: Optional[torch.Tensor] = None,
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
        if learner is None:
            self.learner[self.step].fill_(1.0)
        else:
            if isinstance(learner, np.ndarray):
                learner = torch.from_numpy(learner)
            self.learner[self.step].copy_(learner.to(self.learner.dtype))
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

        if next_values_own is not None:
            self.next_values_own[self.step].copy_(next_values_own)
            self.has_next_values_own = True

        if search_pi is not None and has_search is not None:
            self.search_pi[self.step] = search_pi.to(self.device)
            self.has_search[self.step] = has_search.to(self.device)
        self.step += 1
        if self.step >= self.buffer_size:
            self.full = True

    def compute_gae(
        self,
        last_v_win: torch.Tensor,
        last_v_vp: torch.Tensor,
        last_dones: torch.Tensor,
        last_players: torch.Tensor,
        gamma: float = 1.0,
        gae_lambda: float = 0.98,
        slice_turn_boundaries: bool = False,
        blunder_window: bool = True,
        defcon_risk_horizon: int = 4,
        same_perspective_bootstrap: bool = False,
        per_player_gae: bool = False,
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

        ``same_perspective_bootstrap`` replaces ``-V(s_{t+1}, p_{t+1})`` with
        ``V(s_{t+1}, p_t)``. The default negation assumes ``V(s, me) = -V(s, opponent)``, exact in
        a perfect-information game and false here: ``v_win`` is computed from a perspective-filtered
        observation, so the two evaluate different information sets. Measured over 227 positions,
        ``v_US + v_USSR`` has mean +0.051 and mean absolute 0.144 where the identity requires 0.
        The error lands hardest on the last decision before the side switches -- a card played for
        its event is scored by the opponent's opinion of the result, formed without seeing the
        deciding player's hand. Requires ``next_values_own`` to have been supplied to ``add``.
        """
        if same_perspective_bootstrap and not self.has_next_values_own:
            raise ValueError(
                "same_perspective_bootstrap needs V(s_{t+1}, p_t) for every step, but no "
                "next_values_own was passed to add(). Silently falling back to the negated "
                "bootstrap would make the arm measure nothing.")

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        window_mask = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.bool,
                                  device=self.device)

        # Pending blunder window, armed at a terminal step and disarmed at the turn boundary.
        pending_hs_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_us = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_ussr = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pending_hs_turn = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        last_ret_vp = last_v_vp.clone()

        # Backward-filled label for the auxiliary DEFCON-risk head. Armed at a terminal
        # step that the acting player brought on itself, then counted down over the
        # preceding steps so only the run-up is marked, not the whole episode.
        risk_us = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        risk_ussr = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        risk_left = torch.zeros(self.num_envs, dtype=torch.int16, device=self.device)

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

            if same_perspective_bootstrap:
                # The stored value is already the next state seen by the player who moved, so it
                # needs no sign: it is in the actor's frame by construction. `sign` is still used
                # below on `last_gae`, which flips the *following step's advantage* -- a different
                # correction, and still required.
                #
                # This overrides `next_val` for BOTH branches above, including the final step,
                # because next_values_own[t] is stored for every t -- so the last step needs no
                # `last_v_win` special case either.
                next_val = self.next_values_own[t]

            # Arm/disarm the blunder window at terminal steps, vectorized across envs.
            # The timestep loop must stay sequential because GAE is a backward recursion,
            # but every env at a given t is independent, so the whole inner loop collapses
            # into masked tensor ops. Held scoring can implicate both players at once; an
            # unprovoked DEFCON-1 suicide implicates exactly the player who chose it.
            done_t = self.dones[t]
            b_us = self.held_scoring_us[t].clone()
            b_ussr = self.held_scoring_ussr[t].clone()
            if blunder_window:
                db = self.defcon_blunder[t]
                b_us = b_us | (db == 1)
                b_ussr = b_ussr | (db == -1)
                arm = done_t & (b_us | b_ussr)
            else:
                arm = torch.zeros_like(done_t)

            # A terminal step that is not a blunder clears any window instead of leaving
            # a stale one armed across the episode boundary.
            pending_hs_active = torch.where(done_t, arm, pending_hs_active)
            pending_hs_us = torch.where(arm, b_us, pending_hs_us)
            pending_hs_ussr = torch.where(arm, b_ussr, pending_hs_ussr)
            pending_hs_turn = torch.where(arm, self.turns[t], pending_hs_turn)

            # The window covers only the turn the blunder happened in.
            in_window = pending_hs_active & (self.turns[t] == pending_hs_turn)
            expired = pending_hs_active & (~in_window)
            pending_hs_active = pending_hs_active & in_window
            last_gae = torch.where(expired, torch.zeros_like(last_gae), last_gae)

            # Inside the window: the blunderer eats -1, and the opponent is pinned to its
            # own value so its advantage is exactly zero -- it did not earn the windfall.
            is_blunderer = torch.where(curr_p == 1, pending_hs_us, pending_hs_ussr)
            v_t = self.values_win[t]
            win_ret = torch.where(is_blunderer, -torch.ones_like(v_t), v_t)
            win_adv = torch.where(is_blunderer, -1.0 - v_t, torch.zeros_like(v_t))

            # Outside it: the ordinary zero-sum Bellman TD error and GAE recursion.
            delta = self.rewards[t] + gamma * next_val * non_terminal - v_t
            std_gae = delta + gamma * gae_lambda * sign * non_terminal * last_gae

            last_gae = torch.where(in_window, win_adv, std_gae)
            self.advantages[t] = last_gae
            self.returns_win[t] = torch.where(in_window, win_ret, last_gae + v_t)
            # The blunder window is a deliberate override of both, and it outranks the estimator:
            # per-player GAE below must not undo it.
            window_mask[t] = in_window

            # DEFCON-risk label. defcon_blunder is non-zero only for a loss the losing
            # player chose (an unprovoked DEFCON-1, or a Cuban Missile Crisis coup), which
            # is exactly the class val_win_head is blind to.
            db_t = self.defcon_blunder[t]
            risk_arm = done_t & (db_t != 0)
            risk_us = torch.where(risk_arm, db_t == 1, risk_us)
            risk_ussr = torch.where(risk_arm, db_t == -1, risk_ussr)
            risk_left = torch.where(
                risk_arm,
                torch.full_like(risk_left, int(defcon_risk_horizon)),
                torch.where(done_t, torch.zeros_like(risk_left), risk_left),
            )
            doomed = torch.where(curr_p == 1, risk_us, risk_ussr)
            self.defcon_risk_target[t] = ((risk_left > 0) & doomed).float()
            risk_left = torch.clamp(risk_left - 1, min=0)

            # VP Return: real VP ground truth at terminal / delta VP
            # vps[t] is normalized VP in [-1.0, 1.0] from player perspective
            curr_vp_norm = self.vps[t] * curr_p.float() / 20.0
            self.returns_vp[t] = torch.where(
                self.dones[t],
                curr_vp_norm,
                curr_vp_norm * 0.1 + 0.9 * non_terminal * next_ret_vp
            )

        if per_player_gae:
            pp_adv, pp_ret = self._gae_per_player(gamma, gae_lambda, last_v_win, last_players)
            keep = ~window_mask
            self.advantages = torch.where(keep, pp_adv, self.advantages)
            self.returns_win = torch.where(keep, pp_ret, self.returns_win)

        # Normalize advantages per rollout batch.
        #
        # This is ONE mean and ONE std over both sides' transitions together. In a game whose
        # self-play win rate has gone lopsided that is the place to look first: if the value
        # head is well calibrated per role the two sides are already centred and mixing them
        # is harmless, but if one side's advantages carry a systematically different mean or a
        # much smaller spread, the shared statistics bias its updates or shrink its signal
        # relative to the other side's. Which of those is happening is an empirical question,
        # so the per-side figures are recorded below rather than assumed either way.
        flat_adv = self.advantages.view(-1)
        mean_adv = flat_adv.mean()
        std_adv = flat_adv.std() + 1e-8
        self.raw_advantage_std = float(std_adv)

        flat_players = self.players.view(-1)
        for tag, code in (("us", 1), ("ussr", -1)):
            sel = flat_adv[flat_players == code]
            self.side_advantage[tag] = {
                "n": int(sel.numel()),
                "mean": float(sel.mean()) if sel.numel() else 0.0,
                "std": float(sel.std()) if sel.numel() > 1 else 0.0,
            }

        self.advantages = (self.advantages - mean_adv) / std_adv

    def _gae_per_player(
        self,
        gamma: float,
        gae_lambda: float,
        last_v_win: torch.Tensor,
        last_players: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE within each player's own subsequence of decisions.

        Walks backward once, carrying per-player state: the advantage and value at that player's
        NEXT own decision, the rewards accrued to it since, and the gap in steps. Vectorised over
        environments; the loop is over timesteps only, as the interleaved version's is.

        Rewards are zero-sum and stored in the acting player's frame, so a step's reward accrues to
        player q as ``rewards[t] * players[t] * q`` -- which is how the opponent's moves enter q's
        return without any value being negated.
        """
        T, N = self.buffer_size, self.num_envs
        dev = self.device
        adv = torch.zeros((T, N), dtype=torch.float32, device=dev)
        ret = torch.zeros((T, N), dtype=torch.float32, device=dev)

        state = {}
        for q in (1, -1):
            # The next decision past the end of the buffer: value carried in q's own frame.
            state[q] = {
                "A": torch.zeros(N, dtype=torch.float32, device=dev),
                "V": last_v_win * last_players.float() * float(q),
                "R": torch.zeros(N, dtype=torch.float32, device=dev),
                "gap": torch.zeros(N, dtype=torch.float32, device=dev),
                "has": torch.ones(N, dtype=torch.bool, device=dev),
            }

        zero = torch.zeros(N, dtype=torch.float32, device=dev)
        for t in reversed(range(T)):
            p_t = self.players[t].float()
            done_t = self.dones[t].bool()
            r_t = self.rewards[t]

            # A terminal step ends the episode: everything later belongs to a different game, and
            # this step's own decision has nothing to bootstrap from.
            if bool(done_t.any()):
                for q in (1, -1):
                    st = state[q]
                    st["A"] = torch.where(done_t, zero, st["A"])
                    st["V"] = torch.where(done_t, zero, st["V"])
                    st["R"] = torch.where(done_t, zero, st["R"])
                    st["gap"] = torch.where(done_t, zero, st["gap"])
                    st["has"] = st["has"] & (~done_t)

            for q in (1, -1):
                st = state[q]
                st["R"] = st["R"] + r_t * p_t * float(q)
                st["gap"] = st["gap"] + 1.0

                is_mover = (self.players[t] == q)
                if not bool(is_mover.any()):
                    continue

                disc = torch.pow(torch.tensor(gamma, device=dev), st["gap"])
                carry = st["has"].float()
                v_t = self.values_win[t]
                delta = st["R"] + disc * st["V"] * carry - v_t
                a = delta + disc * gae_lambda * st["A"] * carry

                adv[t] = torch.where(is_mover, a, adv[t])
                ret[t] = torch.where(is_mover, a + v_t, ret[t])

                st["A"] = torch.where(is_mover, a, st["A"])
                st["V"] = torch.where(is_mover, v_t, st["V"])
                st["R"] = torch.where(is_mover, zero, st["R"])
                st["gap"] = torch.where(is_mover, zero, st["gap"])
                st["has"] = st["has"] | is_mover

        return adv, ret

    def diagnostics(self) -> Dict[str, float]:
        """Value-head and advantage-distribution health metrics for the current rollout.

        Must be called after ``compute_gae``; the advantage figures describe the
        normalised advantages that the policy update actually consumes.
        """
        adv = self.advantages.detach().reshape(-1)
        near_zero = (adv.abs() < NEAR_ZERO_ADVANTAGE_EPS).float().mean() if adv.numel() > 0 else torch.zeros(())
        out = {
            "explained_variance": explained_variance(self.returns_win, self.values_win),
            "adv_std": float(adv.std()) if adv.numel() > 1 else 0.0,
            "adv_std_raw": self.raw_advantage_std,
            "adv_frac_near_zero": float(near_zero),
        }
        # Per-side, pre-normalisation. `adv_mean_*` near zero means the value head has
        # already centred that role and the shared mean is harmless; a gap between the two
        # `adv_std_*` means the shared divisor is rescaling the two sides' signals unequally.
        for tag in ("us", "ussr"):
            side = self.side_advantage.get(tag)
            if side is None:
                continue
            out[f"adv_mean_{tag}"] = side["mean"]
            out[f"adv_std_{tag}"] = side["std"]
            out[f"adv_n_{tag}"] = float(side["n"])
        return out

    def priority_indices(self, alpha: float) -> torch.Tensor:
        """Sampling order weighted toward transitions whose outcome swung hardest.

        Decisive positions are rare: measured over 120 self-play games, an avoidable
        game-ending mistake was available at roughly 0.6% of decisions. Under uniform
        sampling those few hundred transitions are drowned by the tens of thousands that
        carry no such lesson, and the gradient never resolves them.

        Priority is |advantage|, which is generic -- it encodes no game rule, and makes no
        reference to DEFCON, scoring cards, or any card list. A step matters here exactly
        when the outcome differed sharply from what the critic expected, which is the same
        property that makes a blunder a blunder.

        alpha = 0 recovers uniform sampling; 1 samples in proportion to |advantage|.
        Sampling is with replacement, so the batch count is unchanged.

        This deliberately biases the gradient. Classic prioritised replay corrects for that
        with importance weights, which is right when the aim is an unbiased estimate of the
        same objective. Here the aim is the opposite: to stop a rare class of decisions
        being averaged away, so the reweighting IS the intervention and correcting it would
        undo it. The cost is that common positions are undertrained relative to their true
        frequency, which is why this defaults to off and wants a modest alpha when on.
        """
        total_steps = self.buffer_size * self.num_envs
        if alpha <= 0.0:
            return torch.randperm(total_steps, device=self.device)
        adv = self.advantages.view(-1).abs()
        weights = (adv + 1e-6) ** alpha
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            return torch.randperm(total_steps, device=self.device)
        return torch.multinomial(weights, total_steps, replacement=True)

    def get_batches(
        self, batch_size: int, priority_alpha: float = 0.0
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Yields mini-batches for inner-loop SGD updates.

        Element 9 is the learner mask: 1 where the action was chosen by the policy being
        trained, 0 where a frozen opponent chose it. It is all ones without an opponent pool.

        Elements 10 and 11 are P15-X4b's search target and the flag for where it exists. They are
        appended rather than inserted so that positional unpacking of the first nine is unchanged.
        """
        total_steps = self.buffer_size * self.num_envs
        indices = self.priority_indices(priority_alpha)

        flat_obs = self.obs.view(total_steps, self.obs_dim)
        flat_masks = self.masks.view(total_steps, self.action_dim)
        flat_actions = self.actions.view(total_steps)
        flat_log_probs = self.log_probs.view(total_steps)
        flat_advantages = self.advantages.view(total_steps)
        flat_returns_win = self.returns_win.view(total_steps)
        flat_returns_vp = self.returns_vp.view(total_steps)
        flat_defcon_risk = self.defcon_risk_target.view(total_steps)
        flat_learner = self.learner.view(total_steps)
        flat_search_pi = self.search_pi.view(total_steps, self.action_dim)
        flat_has_search = self.has_search.view(total_steps)

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
                flat_defcon_risk[batch_idx],
                flat_learner[batch_idx],
                flat_search_pi[batch_idx],
                flat_has_search[batch_idx],
            )

