"""NashPG (Nash Policy Gradient) and Oracle-Guided Multi-Task Trainers for Twilight Struggle Self-Play.

Implements:
1. BaseNashPGTrainer: Unified base class managing vectorized C++ rollouts, temperature scheduling,
   zero-sum GAE credit slicing, and outer-loop reference policy anchoring π_ref^(k).
2. NashPGTrainer: Standard Nash Policy Gradient trainer for ColdWarNet V1, V2, and V3.
3. The privileged oracle critic and opponent-belief head lived here, as
   OracleGuidedNashPGTrainer against ColdWarNetV4. Both are removed; P5 is queued to
   measure the idea and will rebuild it against the current critic (see the plan file for
   the commit to read the old implementation out of). What it did:
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

from ai.models.coldwar_net_v2 import VP_LIMIT

from bindings.ts_env import TsVectorizedEnv
from .rollout_buffer import RolloutBuffer
from .critic_tracker import CriticTracker

# Fixed-probe entropy: number of (observation, mask) pairs frozen at the start of
# training, and how often (in iterations) the probe is re-evaluated.
ENTROPY_PROBE_SIZE = 2000
ENTROPY_PROBE_INTERVAL = 10


class FixedEntropyProbe:
    """A frozen pool of (observation, action-mask) pairs for drift-free entropy tracking.

    On-policy entropy is measured on whatever states the current policy happens to visit,
    so it moves with the state distribution and can hide (or fake) policy collapse. This
    pool is sampled exactly once and never resampled, so its entropy is comparable across
    the whole run.
    """

    def __init__(self, size: int = ENTROPY_PROBE_SIZE, chunk_size: int = 256, seed: int = 20240517) -> None:
        self.size = size
        self.chunk_size = chunk_size
        self.seed = seed
        self.obs: Optional[torch.Tensor] = None
        self.masks: Optional[torch.Tensor] = None

    @property
    def is_filled(self) -> bool:
        return self.obs is not None and self.masks is not None

    def fill_from(self, obs: torch.Tensor, masks: torch.Tensor) -> bool:
        """Populates the pool from a flat (N, obs_dim) / (N, action_dim) sample.

        No-op once filled -- that permanence is the whole point of the probe.
        Returns True only on the call that actually filled it.
        """
        if self.is_filled:
            return False
        flat_obs = obs.reshape(obs.shape[0], -1)
        flat_masks = masks.reshape(masks.shape[0], -1)
        # Only states with a real choice to make: a forced single-action state has zero
        # entropy by construction and would just dilute the signal.
        eligible = torch.nonzero(flat_masks.sum(dim=-1) >= 2, as_tuple=False).reshape(-1)
        if eligible.numel() == 0:
            return False
        n_eligible = int(eligible.numel())
        n = min(self.size, n_eligible)
        rng = np.random.RandomState(self.seed)
        picked = rng.choice(n_eligible, size=n, replace=False)
        idx = eligible[torch.from_numpy(picked).long().to(eligible.device)]
        self.obs = flat_obs[idx].detach().clone().float()
        self.masks = flat_masks[idx].detach().clone()
        return True

    def mean_entropy(self, net: nn.Module) -> Optional[float]:
        """Mean masked policy entropy over the frozen pool, or None if not yet filled."""
        if self.obs is None or self.masks is None:
            return None
        was_training = net.training
        net.eval()
        total = 0.0
        count = 0
        try:
            with torch.no_grad():
                for start in range(0, self.obs.shape[0], self.chunk_size):
                    chunk_obs = self.obs[start : start + self.chunk_size]
                    chunk_masks = self.masks[start : start + self.chunk_size]
                    out = net(chunk_obs, chunk_masks)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    logits = logits.float().masked_fill(chunk_masks == 0, -1e9)
                    log_p = F.log_softmax(logits, dim=-1)
                    entropy = -(log_p.exp() * log_p).sum(dim=-1)
                    total += float(entropy.sum().item())
                    count += int(entropy.shape[0])
        finally:
            if was_training:
                net.train()
        return total / max(1, count)


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
        search_ce_coef: float = 0.0,   # P15-X4b: weight on CE toward the searcher. 0 = off
        search_sims: int = 32,
        search_subsample: float = 0.125,
        search_node_filter: str = "card_playmode",
        ent_coef: float = 0.01,        # Entropy exploration coefficient
        vf_coef: float = 0.5,          # Value loss coefficient
        vp_coef: float = 0.05,         # Auxiliary VP loss weight
        value_dist_coef: float = 0.02, # Categorical VP distribution loss weight
        adv_filter_quantile: float = 0.0,  # P1: drop the lowest-|A| share from the policy loss
        defcon_coef: float = 0.0,      # Auxiliary DEFCON-risk loss weight (0 disables the head)
        gamma: float = 1.0,            # Undiscounted: see note below
        # gamma must be exactly 1.0. Twilight Struggle is zero-sum and decided only at the
        # end, so any discount biases the agent against the endgame. At 0.999 over the
        # ~300 micro-decisions of a full game a terminal reward arrives attenuated by
        # ~26%, which is largest precisely where instant wins and losses live.
        gae_lambda: float = 0.98,      # GAE lambda
        num_epochs: int = 4,           # Inner-loop optimization epochs per iteration
        ref_update_freq: int = 200_000,# Outer-loop reference update frequency in steps
        max_grad_norm: float = 1.0,
        slice_turn_boundaries: bool = False,
        blunder_window: bool = True,
        same_perspective_bootstrap: bool = False,
        per_player_gae: bool = False,
        priority_alpha: float = 0.0,
        temperature_schedule: bool = True,
        setup_explore_frac: float = 0.0,
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
        # P15-X4b. A searcher used ONLY to produce cross-entropy targets: rollouts keep sampling
        # from the raw policy, so the state distribution is identical to the baseline's and the
        # arm stays one factor. Acting on the search policy is a different experiment.
        self.search_ce_coef = float(search_ce_coef)
        self._searcher = None
        if self.search_ce_coef > 0.0:
            from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
            self._searcher = BatchedMCTS(
                active_net, device=self.device,
                config=BatchedMCTSConfig(
                    simulations=search_sims, temperature=0.0, auto_advance=True,
                    advance_root=False, determinize=True,
                    node_filter=search_node_filter, subsample=search_subsample))
            print(f"[X4b] search CE on: coef {self.search_ce_coef}, {search_sims} sims, "
                  f"{search_node_filter}, subsample {search_subsample}", flush=True)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vp_coef = vp_coef
        self.value_dist_coef = value_dist_coef
        # P1 advantage filtering. Samples whose |advantage| falls below this quantile of the
        # minibatch are dropped from the *policy* term only; the value head still sees every
        # sample, which is the point -- the critic needs the uninformative states too.
        # 0.0 disables it. It changes the effective batch size, so it is screened as its own
        # factor rather than folded into the categorical change.
        self.adv_filter_quantile = float(adv_filter_quantile)
        self.defcon_coef = defcon_coef
        if defcon_coef > 0.0 and not hasattr(self.active_net, "forward_with_risk"):
            raise ValueError(
                f"--defcon-coef={defcon_coef} was requested but "
                f"{type(self.active_net).__name__} has no DEFCON-risk head. It is "
                f"implemented for v1 and v2; adding it to another architecture means "
                f"giving that class defcon_risk_head and forward_with_risk."
            )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.ref_update_freq = ref_update_freq
        self.max_grad_norm = max_grad_norm
        self.slice_turn_boundaries = slice_turn_boundaries
        self.blunder_window = blunder_window
        #: Bootstrap from V(s_{t+1}, p_t) instead of -V(s_{t+1}, p_{t+1}). Off by default: it
        #: changes what the critic is trained on, so it is an arm, not a correction applied
        #: silently. See `RolloutBuffer.compute_gae`.
        self.same_perspective_bootstrap = same_perspective_bootstrap
        #: Compute GAE within each player's own subsequence of decisions instead of over the
        #: interleaved one, which removes the cross-perspective bootstrap entirely rather than
        #: patching its sign. Off by default. See RolloutBuffer._gae_per_player.
        self.per_player_gae = per_player_gae
        self.priority_alpha = priority_alpha
        self.temperature_schedule = temperature_schedule

        # Forced opening exploration. The setup placement saturates: measured on E3-30-28 @240M
        # the chosen opening carries a logit 16.56 above the runner-up, so softmax gives it
        # p = 1 - 6e-8 even at tau = 1.0, and the rollout schedule samples at tau in [0.10, 0.50],
        # which is *sharper* still. No temperature reaches an alternative -- at tau=1 the second
        # choice comes up once per 16 million games -- so exploring the opening at all requires
        # overriding the action, not softening the distribution.
        #
        # The forced action is stored as the action taken, and its log-prob is the policy's own
        # log-prob of it, so the PPO ratio still starts at 1.0 and the transition trains normally.
        self.setup_explore_frac = float(setup_explore_frac)
        self._setup_explore_env = np.zeros(self.num_envs, dtype=bool)
        if self.setup_explore_frac > 0.0:
            n_expl = int(round(self.num_envs * self.setup_explore_frac))
            self._setup_explore_env[:n_expl] = True
            print(f"[setup explore] forcing a uniform legal opening in {n_expl}/{self.num_envs} "
                  f"envs ({self.setup_explore_frac:.0%}); those placements are trained on",
                  flush=True)
        #: Decisions since this env's episode began, so the SETUP check runs only where setup
        #: can still be live rather than on every step of a ~440-decision episode.
        self._ep_decisions = np.zeros(self.num_envs, dtype=np.int32)
        self._setup_rng = np.random.default_rng(12345)

        self.optimizer = torch.optim.AdamW(self.active_net.parameters(), lr=lr, weight_decay=1e-4)

        self.env = env if env is not None else TsVectorizedEnv(num_envs=self.num_envs, base_seed=int(time.time()))
        # Tier-1 critic discrimination. Holds one sample per game per turn until that
        # game's winner is known; costs an integer compare per env per step.
        self.critic_tracker = CriticTracker(num_envs=self.num_envs)
        #: Optional frozen-opponent pool. None means ordinary self-play, where the
        #: learner plays both sides and every transition receives policy gradient.
        self.opponent_pool: Any = None
        #: Which environments are pure self-play. Game statistics are reported over
        #: these only, so a pooled run stays comparable with an unpooled one.
        self._selfplay_mask = np.ones(self.num_envs, dtype=bool)
        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            num_envs=self.num_envs,
            # Taken from the env rather than fixed, so the buffer cannot disagree with the
            # layout the environment is actually producing. A mismatch here would be a reshape
            # error at best and a silently misaligned observation at worst.
            obs_dim=getattr(self.env, "observation_size", 4293),
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

        # Frozen entropy probe, filled from the very first rollout and never resampled.
        self.entropy_probe = FixedEntropyProbe(size=ENTROPY_PROBE_SIZE)
        self.entropy_probe_interval = ENTROPY_PROBE_INTERVAL

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
        if self.opponent_pool is not None:
            self.opponent_pool.start_iteration()
            self._selfplay_mask = ~self.opponent_pool.is_mixed
        else:
            self._selfplay_mask = np.ones(self.num_envs, dtype=bool)
        self.buffer.reset()
        completed_episodes: List[Dict[str, Any]] = []

        for _ in range(self.buffer_size):
            # copy=True is load-bearing. The env's observation and mask arrays are views into
            # buffers the runner reuses, so `env.step()` below overwrites them in place -- and
            # `buffer.add` is called *after* that step. On CUDA `.to(device)` already copies, so
            # this was invisible; on CPU `.to("cpu")` is a no-op and the buffer stored the
            # *post*-step observation and mask against the pre-step action. It hid during setup,
            # where consecutive masks are identical, and only showed on the step where the side
            # to move changed.
            obs_t = torch.from_numpy(self._obs_np).to(self.device, torch.float32, copy=True)
            masks_t = torch.from_numpy(self._masks_np).to(self.device, copy=True)

            # Who is to move in each environment, needed before acting so the batch can be
            # split between the learner and a frozen opponent.
            if self.opponent_pool is not None:
                _dp = np.asarray(self.env.runner.get_decision_players(), dtype=np.int8)
                learner_np = self.opponent_pool.learner_acts(_dp)
            else:
                learner_np = np.ones(self.num_envs, dtype=bool)
            learner_t = torch.from_numpy(learner_np).to(self.device)

            with torch.no_grad():
                if self.opponent_pool is None or bool(learner_np.all()):
                    logits, v_win_t, v_vp_t = self.active_net(obs_t, masks_t)
                else:
                    # Two passes on *complementary* subsets, so the total work is the same as
                    # the single full-batch pass it replaces -- one extra kernel launch, not
                    # twice the FLOPs. The opponent's logits are used only to act; its values
                    # are taken from the learner's critic, which is what GAE must bootstrap
                    # with (a frozen opponent's critic is a different function and mixing the
                    # two would corrupt the recursion).
                    logits = torch.empty((self.num_envs, 212), device=self.device,
                                         dtype=torch.float32)
                    opp_logits, _, _ = self.opponent_pool.current(
                        obs_t[~learner_t], masks_t[~learner_t])
                    logits[~learner_t] = opp_logits.float()
                    own_logits, _, _ = self.active_net(obs_t[learner_t], masks_t[learner_t])
                    logits[learner_t] = own_logits.float()
                    # One learner-critic pass over the whole batch: values must come from the
                    # policy being trained at every state, opponent-chosen ones included.
                    _, v_win_t, v_vp_t = self.active_net(obs_t, masks_t)
                v_win_t = v_win_t.squeeze(-1)
                v_vp_t = v_vp_t.squeeze(-1)

                # Multi-temperature exploration: Sample actions from temperature-scaled
                # behavior distribution beta(a|s) = softmax(logits / tau) with stratified per-env
                # temperatures (tau in [0.10, 0.50]) to encourage diverse trajectory exploration.
                if self.temperature_schedule and self.num_envs > 1:
                    scaled_logits = logits / self.env_temps
                    scaled_probs = F.softmax(scaled_logits, dim=-1)
                    actions_t = torch.multinomial(scaled_probs, 1).squeeze(1)
                else:
                    actions_t = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)

                # NOTE (Design Choice): Store canonical (tau=1.0) log-probabilities rather than
                # temperature-scaled log-probs log(beta(a|s)). This ensures the PPO importance ratio
                # r_t(theta) = exp(cur_lp - old_lp) = pi_theta(a) / pi_theta_old(a) initializes at exactly
                # 1.0 at step 0, allowing PPO clipping [1 - eps, 1 + eps] to operate smoothly without
                # immediately saturating the clip bounds on modal actions under low temperatures.
                # Stratified temperature sampling thus acts as pure exploration noise for optimizing
                # the underlying canonical policy parameterization pi_theta.
                if self.setup_explore_frac > 0.0:
                    self._force_setup_exploration(actions_t, masks_t)

                unscaled_log_probs = F.log_softmax(logits, dim=-1)
                log_probs_t = unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            actions_np = actions_t.cpu().numpy()

            # P15-X4b. MUST happen before the step: `_search_targets` reads the runner's current
            # state, and `buffer.add` below files the answer against `obs_t`, which is s_t. Taken
            # after the step the runner holds s_{t+1} (see the bootstrap note further down), so
            # every target would describe the position *after* the one it is stored with -- a
            # distribution over another state's actions, then masked against s_t's legal set, so
            # the mass lands wherever those indices happen to mean something here. It does not
            # fail loudly: it trains the policy toward noise with a real gradient behind it, which
            # collapsed two arms (entropy 1.03 -> 0.36, critic to chance) before it was found.
            search_pi_t, has_search_t = self._search_targets()

            next_obs_np, next_masks_np, rewards_np, self._dones_np, self._info = self.env.step(actions_np)

            if self.setup_explore_frac > 0.0:
                self._ep_decisions += 1
                self._ep_decisions[self._dones_np > 0.5] = 0

            # V(s_{t+1}, p_t): the resulting state seen by the player who just moved, rather than
            # by whoever moves next. The default bootstrap negates the next step's value, which
            # assumes V(s, me) = -V(s, opponent) -- exact only under perfect information, and
            # measurably false here (mean |v_US + v_USSR| = 0.144 where it should be 0). The gap
            # lands hardest on the last decision before the side switches: a card played for its
            # event is then scored by the opponent's opinion of the result, formed without seeing
            # the deciding player's hand.
            #
            # `_info["acting_players"]` is captured before the step (ts_env.py:288), so it names
            # p_t, and the runner now holds s_{t+1}. Terminal steps are included and cost nothing:
            # the env has auto-reset by now, so the value is of an unrelated fresh game, and
            # `non_terminal` zeroes it in compute_gae.
            next_own_t = None
            if self.same_perspective_bootstrap:
                own_obs = self.env.observations_for(
                    np.asarray(self._info["acting_players"], dtype=np.int8))
                with torch.no_grad():
                    _l, own_v, _vp = self.active_net(
                        torch.from_numpy(own_obs).to(self.device, torch.float32), None)
                next_own_t = own_v.squeeze(-1)

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
                learner=learner_t.float(),
                turns=torch.from_numpy(self._info["turns"]).to(self.device) if "turns" in self._info else None,
                vps=torch.from_numpy(self._info["victory_points"]).float().to(self.device) if "victory_points" in self._info else None,
                held_scoring_us=torch.from_numpy(self._info["held_scoring_us"]).to(self.device) if "held_scoring_us" in self._info else None,
                held_scoring_ussr=torch.from_numpy(self._info["held_scoring_ussr"]).to(self.device) if "held_scoring_ussr" in self._info else None,
                defcon_blunder=torch.from_numpy(self._info["defcon_blunder"]).to(self.device) if "defcon_blunder" in self._info else None,
                next_values_own=next_own_t,
                search_pi=search_pi_t,
                has_search=has_search_t,
            )

            # v_win is from the *acting* player's perspective; multiplying by the acting
            # player (US = +1, USSR = -1) puts every sample on one fixed perspective, so the
            # sign does not flip with whoever happens to be moving. `turns` is the pre-step
            # acting turn, which is the state v_win was computed on.
            acting = np.asarray(self._info["acting_players"])
            # Self-play environments only. A mixed environment's outcome is the learner against
            # a frozen snapshot, which says nothing about how balanced the learner is against
            # itself -- and the base rate, which AUC and Brier skill are measured against, would
            # be the pool's difficulty rather than the policy's own. Keeping this to self-play
            # is what makes the numbers comparable with runs that have no pool at all.
            self.critic_tracker.observe(
                v_win_t.detach().cpu().numpy() * acting,
                np.asarray(self._info["turns"]),
                self._selfplay_mask,
            )
            for _i, _reason in enumerate(self._info.get("ending_reasons", [])):
                if not _reason:
                    continue
                if self._selfplay_mask[_i]:
                    _vp = float(self._info["victory_points"][_i])
                    self.critic_tracker.resolve(_i, None if _vp == 0 else _vp > 0)
                else:
                    # Clear the pending samples without recording an outcome, or they would be
                    # attached to whatever the next episode in this slot produces.
                    self.critic_tracker.reset_env(_i)
                if self.opponent_pool is not None:
                    # The outcome goes with it: the pool keys its per-opponent win rates off
                    # this, and without them there is no in-run signal for whether pooling works
                    # and no basis for a PFSP draw.
                    self.opponent_pool.on_episode_end(
                        _i, victory_points=float(self._info["victory_points"][_i]))

            if "completed_episodes" in self._info and self._info["completed_episodes"]:
                # Win rate, ending mix and game length describe how the policy plays. Games
                # against a frozen pool opponent are a different question and would make a
                # pooled run incomparable with an unpooled one, so only self-play episodes are
                # summarised. (The blunder and position probes already run their own separate
                # self-play games, so they need no filtering.)
                completed_episodes.extend(
                    e for e in self._info["completed_episodes"]
                    if self._selfplay_mask[int(e.get("env_idx", 0))])

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
            blunder_window=self.blunder_window,
            same_perspective_bootstrap=self.same_perspective_bootstrap,
            per_player_gae=self.per_player_gae,
        )

        # Freeze the entropy probe pool from the first rollout only.
        if not self.entropy_probe.is_filled:
            self.entropy_probe.fill_from(
                self.buffer.obs.reshape(-1, self.buffer.obs_dim),
                self.buffer.masks.reshape(-1, self.buffer.action_dim),
            )

        steps_collected = self.buffer_size * self.num_envs
        self.total_env_steps += steps_collected
        self.steps_since_ref_update += steps_collected
        rollout_time = time.time() - t0

        metrics: Dict[str, Any] = {
            "steps": steps_collected,
            "rollout_time": rollout_time,
            "fps": steps_collected / max(rollout_time, 1e-6),
            "completed_episodes": completed_episodes,
        }
        metrics.update(self.buffer.diagnostics())
        metrics.update(self.critic_tracker.metrics())
        # Pool size and span, so a pool that silently stops growing is visible as a flat line
        # rather than being invisible. Without this the mechanism cannot be verified from a run.
        if self.opponent_pool is not None:
            metrics.update(self.opponent_pool.stats())
        return metrics

    def train_step(self) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement train_step")

    def _force_setup_exploration(self, actions_t: torch.Tensor, masks_t: torch.Tensor) -> None:
        """Replace the chosen opening with a uniform legal one, in the designated envs.

        Only while an env is still in `Phase.SETUP`, which is checked against the engine rather
        than guessed from the decision type: a `POINT_NODE` at turn 1 is also what an ordinary
        placement looks like, and overriding those would be a different experiment.

        The phase check reads the runner's state, so it is bounded to the opening window of each
        episode -- setup is about 15 decisions of roughly 440 -- rather than run every step.
        """
        import ts_engine as _ts

        cand = np.flatnonzero(self._setup_explore_env & (self._ep_decisions < 30))
        if cand.size == 0:
            return
        masks_np = masks_t.detach().cpu().numpy()
        forced = 0
        for i in cand:
            idx = int(i)
            try:
                st = self.env.runner.get_state(idx)
            except Exception:                      # a runner without per-env state: skip quietly
                return
            if st.current_phase != _ts.Phase.SETUP:
                continue
            legal = np.flatnonzero(masks_np[idx])
            if legal.size <= 1:
                continue
            actions_t[idx] = int(self._setup_rng.choice(legal))
            forced += 1
        self.setup_forced_actions = getattr(self, "setup_forced_actions", 0) + forced

    def _search_targets(self):
        """The searcher's visit distribution at the decisions this configuration searches.

        P15-X4b. Returns (targets, flag) over all environments, or (None, None) when search is
        off. The searcher only *answers* -- the actions actually played were already sampled from
        the raw policy above, so the state distribution is the baseline's and this arm varies one
        thing. Acting on the search policy would be a different experiment.

        A target is dropped when the searcher returns nothing, and normalised over the visits it
        did return. Legality is not re-checked here: `BatchedMCTS` already filters its answer
        against the caller's own mask, which is where the authority belongs.

        **Reads the runner's current state, so it must be called while that state is still the
        one the target will be stored against** -- before `env.step`, not after.
        """
        if self._searcher is None:
            return None, None
        import numpy as _np
        import ts_engine as ts

        n = self.num_envs
        pi = torch.zeros((n, self.buffer.action_dim), dtype=torch.float32, device=self.device)
        flag = torch.zeros(n, dtype=torch.float32, device=self.device)
        try:
            states = [self.env.runner.get_state(i) for i in range(n)]
            idx = [i for i, st in enumerate(states)
                   if not ts.Engine.is_terminal(st) and self._searcher.should_search(st)]
            if not idx:
                return pi, flag
            res = self._searcher.run([states[i].clone() for i in idx])
        except Exception as exc:                      # a broken teacher must not kill the run
            print(f"[X4b] search targets unavailable this step: {exc}", flush=True)
            return pi, flag

        for i, (acts, visits) in zip(idx, res):
            if not acts:
                continue
            v = _np.asarray(visits, dtype=_np.float32)
            tot = float(v.sum())
            if tot <= 0.0:
                continue
            for a, w in zip(acts, v):
                if 0 <= int(a) < pi.shape[1]:
                    pi[i, int(a)] = float(w) / tot
            flag[i] = 1.0
        self.search_targets_produced = getattr(self, "search_targets_produced", 0) + int(
            flag.sum().item())
        return pi, flag

    def train_iteration(self) -> Dict[str, Any]:
        """Runs one full training iteration (rollout collection + inner SGD epochs + reference check)."""
        self.total_iterations += 1
        rollout_metrics = self.collect_rollouts()
        train_metrics = self.train_step()

        if self.steps_since_ref_update >= self.ref_update_freq:
            self.update_reference_policy()
            self.steps_since_ref_update = 0

        combined: Dict[str, Any] = {**rollout_metrics, **train_metrics}

        # Fixed-probe entropy: evaluated on the frozen pool every N iterations, so it is
        # directly comparable across the run unlike the on-policy "entropy" above.
        if self.total_iterations == 1 or self.total_iterations % self.entropy_probe_interval == 0:
            probe_entropy = self.entropy_probe.mean_entropy(self.active_net)
            if probe_entropy is not None:
                combined["entropy_fixed_probe"] = probe_entropy

        return combined


class NashPGTrainer(BaseNashPGTrainer):
    """Standard NashPG Trainer for ColdWarNet V1, V2, and V3."""

    def _value_loss(self, cur_v_win, cur_v_vp, b_ret_win, b_ret_vp, value_logits):
        """MSE on the two scalars, or cross-entropy on the distribution.

        The categorical target is the lambda-return *in VP units* projected two-hot onto the
        atoms -- not the realised final VP. The buffer already computes that return for the
        auxiliary VP head, so this reuses the same bootstrapped target the scalar head was
        fitting, which keeps the change to the loss and not to what is being learned.

        `cur_v_win` keeps its own MSE against `b_ret_win`, exactly as in the scalar variant.
        The distribution replaces the auxiliary *VP* head, not the win head: v_win is the
        baseline GAE subtracts, and an earlier version that derived it from the distribution as
        P(VP>0) - P(VP<0) saturated -- see the note in coldwar_net_v2 where the heads are built.
        So the win objective here is bit-for-bit the control's, and the distribution is an
        additive term carrying its own coefficient.
        """
        if value_logits is None:
            return (F.mse_loss(cur_v_win, b_ret_win)
                    + self.vp_coef * F.mse_loss(cur_v_vp, b_ret_vp))
        # getattr for the same reason forward_with_risk uses it: active_net is typed
        # nn.Module, and these live on ColdWarNetV2.
        two_hot = getattr(self.active_net, "two_hot")
        # b_ret_vp is *normalised* VP in [-1, 1] -- rollout_buffer divides the final score by 20
        # -- while the atom support is real VP across [-20, +20]. Rescaling here is not cosmetic:
        # without it every target lands on the three middle atoms and the distribution never
        # learns its tails. The arm that ran without the rescale oscillated between the two sides
        # instead of learning both (80M, 40% against the anchor with 14% as the US, where the
        # scalar control reached 84%). v_win no longer reads this distribution, so that arm's
        # second failure mode is gone, but a degenerate distribution still makes v_vp useless.
        target = two_hot(b_ret_vp.detach() * float(VP_LIMIT))
        log_p = F.log_softmax(value_logits, dim=-1)
        cross_entropy = -(target * log_p).sum(dim=-1).mean()
        # Its own coefficient, not vp_coef: cross-entropy over 41 atoms starts near ln(41) and
        # settles around 1.6, where the MSE it replaces sits near 0.04, so reusing vp_coef would
        # weight the same sub-objective about forty times harder.
        return F.mse_loss(cur_v_win, b_ret_win) + self.value_dist_coef * cross_entropy

    def train_step(self) -> Dict[str, float]:
        self.active_net.train()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        val_loss_accum = 0.0
        risk_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        # P15-X4b. The CE term's own magnitude and how much of the update it claims. Without
        # these the arm that collapsed at ~20M gave no warning: the first visible symptom was
        # kl_div reaching 30, by which point the policy had already gone. The share matters more
        # than the value, because gradients are globally norm-clipped -- a CE term that dominates
        # the raw gradient does not produce a larger step, it produces a step that is almost
        # entirely CE, crowding the advantage and value signals out of the update.
        search_ce_accum = 0.0
        search_ce_frac_accum = 0.0
        search_rows_accum = 0
        num_updates = 0

        for _ in range(self.num_epochs):
            for (b_obs, b_mask, b_act, b_old_lp, b_adv, b_ret_win, b_ret_vp,
                 b_defcon_risk, b_learner, b_search_pi, b_has_search) in self.buffer.get_batches(
                     self.batch_size, self.priority_alpha):
                use_risk = self.defcon_coef > 0.0
                cur_value_logits = None
                if use_risk:
                    forward_with_risk = getattr(self.active_net, "forward_with_risk")
                    cur_logits, cur_v_win, cur_v_vp, cur_risk = forward_with_risk(b_obs, b_mask)
                    cur_risk = cur_risk.squeeze(-1)
                elif getattr(self.active_net, "categorical_value", False):
                    # One backbone pass for the policy, both scalars and the value logits the
                    # cross-entropy needs; re-running the trunk for the logits would double the
                    # cost of every update.
                    forward_with_value_logits = getattr(
                        self.active_net, "forward_with_value_logits")
                    cur_logits, cur_v_win, cur_v_vp, cur_value_logits = (
                        forward_with_value_logits(b_obs, b_mask))
                    cur_risk = None
                else:
                    cur_logits, cur_v_win, cur_v_vp = self.active_net(b_obs, b_mask)
                    cur_risk = None
                cur_v_win = cur_v_win.squeeze(-1)
                cur_v_vp = cur_v_vp.squeeze(-1)

                cur_dist = torch.distributions.Categorical(logits=cur_logits)
                cur_lp = cur_dist.log_prob(b_act)
                cur_entropy = cur_dist.entropy()
                # Masked with the policy loss below; the KL to pi_ref is deliberately
                # left over the whole batch, for broader state coverage.

                ratio = torch.exp(cur_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                surrogate = -torch.min(surr1, surr2)
                # A frozen opponent's actions are part of the environment, not of the policy
                # being trained: they are stored so GAE stays a recursion over consecutive
                # steps, but they must not pull on the policy. Without a pool this is all ones
                # and the expression reduces to the plain mean.
                keep = b_learner > 0.5
                if self.adv_filter_quantile > 0.0 and surrogate.numel() > 1:
                    # Keep the samples the policy can actually learn from. The threshold is a
                    # quantile of this minibatch rather than a fixed |A|, so it adapts as the
                    # advantage scale shrinks over training instead of silently dropping
                    # everything late on. Taken over the learner's own samples, so a mixed
                    # batch does not move the threshold with transitions it will discard.
                    own_adv = b_adv.abs()[keep]
                    if own_adv.numel() > 1:
                        thresh = torch.quantile(own_adv.float(), self.adv_filter_quantile)
                        # A searched decision is exempt: X4b exists to put gradient on
                        # exactly these states, and the filter would drop the ones whose
                        # outcome-advantage is small -- which late in a run is most of them.
                        keep = keep & ((b_adv.abs() >= thresh) | (b_has_search > 0.5))
                ppo_loss = (surrogate[keep].mean() if bool(keep.any())
                            else surrogate.sum() * 0.0)

                clip_frac = ((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)).float().mean().item()

                with torch.no_grad():
                    ref_logits, _, _ = self.reference_net(b_obs, b_mask)
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)

                cur_p = F.softmax(cur_logits, dim=-1)
                cur_log_p = F.log_softmax(cur_logits, dim=-1)
                kl_div = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1).mean()

                # Entropy over the learner's own actions only, for the same reason as the
                # surrogate: an entropy bonus on a frozen opponent's choices would push
                # the learner's policy toward states it did not choose to be in.
                own_entropy = (cur_entropy[b_learner > 0.5].mean()
                               if bool((b_learner > 0.5).any()) else cur_entropy.sum() * 0.0)
                # P15-X4b: pull the policy toward the searcher, on searched decisions only.
                # Soft cross-entropy against the visit distribution. The target is zero outside
                # the legal set, so masked logits contribute nothing and cannot produce a NaN.
                search_ce = cur_logits.sum() * 0.0
                if self.search_ce_coef > 0.0 and bool((b_has_search > 0.5).any()):
                    sel = b_has_search > 0.5
                    search_ce = -(b_search_pi[sel] * cur_log_p[sel]).sum(dim=-1).mean()

                policy_loss = (ppo_loss + self.eta * kl_div - self.ent_coef * own_entropy
                               + self.search_ce_coef * search_ce)
                val_loss = self._value_loss(cur_v_win, cur_v_vp, b_ret_win, b_ret_vp,
                                            cur_value_logits)
                loss = policy_loss + self.vf_coef * val_loss

                # Auxiliary DEFCON-risk objective. Positives are rare (a few percent of
                # steps), so the loss is weighted towards them rather than letting the
                # constant-zero solution dominate.
                if cur_risk is not None:
                    pos = b_defcon_risk.sum()
                    pos_weight = ((b_defcon_risk.numel() - pos) / pos.clamp(min=1.0)).clamp(1.0, 100.0)
                    risk_loss = F.binary_cross_entropy_with_logits(
                        cur_risk, b_defcon_risk, pos_weight=pos_weight
                    )
                    loss = loss + self.defcon_coef * risk_loss
                    risk_loss_accum += risk_loss.item()

                self.optimizer.zero_grad()
                if self.search_ce_coef > 0.0 and float(search_ce) != 0.0:
                    # One extra backward on the CE term alone, retaining the graph, to see what
                    # share of the update it is claiming. Only while the term is active.
                    ce_grads = torch.autograd.grad(
                        self.search_ce_coef * search_ce,
                        [q for q in self.active_net.parameters() if q.requires_grad],
                        retain_graph=True, allow_unused=True)
                    _ce_sq = [(g.detach() ** 2).sum() for g in ce_grads if g is not None]
                    ce_norm = (torch.sqrt(torch.stack(_ce_sq).sum()) if _ce_sq
                               else torch.zeros((), device=self.device))
                    self.optimizer.zero_grad()
                else:
                    ce_norm = None
                loss.backward()
                _tot_sq = [(q.grad.detach() ** 2).sum()
                           for q in self.active_net.parameters() if q.grad is not None]
                total_norm = (torch.sqrt(torch.stack(_tot_sq).sum()) if _tot_sq
                              else torch.zeros((), device=self.device))
                if ce_norm is not None:
                    search_ce_accum += float(search_ce)
                    search_ce_frac_accum += float(ce_norm / total_norm.clamp(min=1e-9))
                    search_rows_accum += 1
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
            "defcon_risk_loss": risk_loss_accum / max(1, num_updates),
            "policy_loss": policy_loss_accum / max(1, num_updates),
            "val_loss": val_loss_accum / max(1, num_updates),
            "kl_div": kl_accum / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
            # 0.0 when search CE is off, so the key is always present and a run without it is
            # still distinguishable from a run whose term silently produced nothing.
            "search_ce": search_ce_accum / max(1, search_rows_accum),
            "search_ce_grad_frac": search_ce_frac_accum / max(1, search_rows_accum),
        }
