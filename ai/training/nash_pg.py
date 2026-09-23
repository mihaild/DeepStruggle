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
from typing import Dict, List, Optional, Any, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.coldwar_net_v2 import VP_LIMIT

from bindings.ts_env import TsVectorizedEnv
from bindings.action_encoder import ActionEncoder
from .rollout_buffer import RolloutBuffer
from .critic_tracker import CriticTracker
from .graphed_forward import GraphCache

# Fixed-probe entropy: number of (observation, mask) pairs frozen at the start of
# training, and how often (in iterations) the probe is re-evaluated.
ENTROPY_PROBE_SIZE = 2000
ENTROPY_PROBE_INTERVAL = 10


def filter_search_visits(
    actions: Sequence[int],
    visits: "np.ndarray",
    legal_mask: "np.ndarray",
    width: int,
) -> Tuple[Optional[List[Tuple[int, float]]], float]:
    """Drop a determinized search's illegal recommendations and renormalise what survives.

    A determinized search may legitimately propose an action that cannot be played in the real
    state, because in this game the legal SET depends on hidden information -- `batched_mcts.py`
    says exactly this where it solves the same problem for an acting agent, letting the search
    propose and the true mask dispose. The training-target path did not dispose, so visits that
    were legal only under the sampled determinization became target mass on masked actions, which
    with the -1e9 mask fill showed up as `search_ce` above 1e5 on a third of iterations.

    Returns `(pairs, dropped_visits)` where `pairs` is `[(action, probability), ...]` over the
    surviving visits, normalised to sum to 1, or `None` when nothing legal survived -- in which
    case the caller must record no target rather than a guessed one.

    A free function so the dropping path is directly testable: the rate is ~1 row in 160, far too
    rare for a short smoke run to exercise, and an untested filter that silently stops filtering
    returns the bug.
    """
    keep = [
        0 <= int(a) < min(int(width), int(legal_mask.shape[0])) and bool(legal_mask[int(a)])
        for a in actions
    ]
    total = 0.0
    dropped = 0.0
    for k, w in zip(keep, visits):
        if k:
            total += float(w)
        else:
            dropped += float(w)
    if total <= 0.0:
        return None, dropped
    pairs = [(int(a), float(w) / total)
             for k, a, w in zip(keep, actions, visits) if k and float(w) > 0.0]
    return pairs, dropped


def wolf_seat_weights(sp_ussr: float, power: float = 1.0,
                      dead_zone: float = 0.0) -> Tuple[float, float]:
    """Per-seat policy-gradient weights (w_us, w_ussr) from the USSR's self-play win share.

    "Win or learn fast" (Bowling & Veloso, 2002): the seat that is winning learns slowly, and the
    seat that is losing learns fast. With x the USSR's share, the USSR's weight goes as
    (1 - x)^p and the US's as x^p, normalised so the two average 1:

        w_us = 2 x^p / (x^p + (1-x)^p),   w_ussr = 2 (1-x)^p / (x^p + (1-x)^p)

    At p = 1 this is exactly w_us = 2x and w_ussr = 2(1 - x). At an even split both are 1. The
    ratio between the seats is (x / (1-x))^p, so p < 1 softens it. x is clipped to [0.01, 0.99]
    only so that neither weight reaches exactly zero.

    `dead_zone` d leaves both weights at 1 while |x - 0.5| <= d, and outside it shifts x toward
    0.5 by d before applying the formula, so the weights leave 1 continuously at the zone's edge
    instead of jumping. d = 0 is the rule above exactly. E4-38/E4-39 showed the full rule keeps
    the self-play split near even but trades the lead back and forth (the USSR weight swung
    0.28-1.65); a dead zone acts only near a real imbalance.
    """
    x = float(sp_ussr)
    d = max(0.0, float(dead_zone))
    if d > 0.0:
        off = x - 0.5
        x = 0.5 + (1.0 if off > 0 else -1.0) * max(0.0, abs(off) - d)
    x = min(0.99, max(0.01, x))
    a = x ** float(power)
    b = (1.0 - x) ** float(power)
    return 2.0 * a / (a + b), 2.0 * b / (a + b)


def wolf_sample_weights(players: torch.Tensor, w_us: float, w_ussr: float) -> torch.Tensor:
    """Each sample's surrogate weight from its acting seat: +1 US, -1 USSR, anything else 1."""
    ones = torch.ones(players.shape, dtype=torch.float32, device=players.device)
    return torch.where(players == 1, ones * w_us, torch.where(players == -1, ones * w_ussr, ones))


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
        rollout_temps: Optional[Sequence[float]] = None,
        merged_influence: bool = False,
        per_seat_adv_norm: bool = False,
        wolf_seat_weight: bool = False,
        wolf_power: float = 1.0,
        wolf_ema_games: float = 2000.0,
        wolf_scope: str = "surrogate",
        wolf_dead_zone: float = 0.0,
        cuda_graphs: bool = True,
        device: torch.device | str = "cuda",
    ):
        self.device = torch.device(device if (torch.cuda.is_available() and device == "cuda") else ("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
        self.active_net = active_net.to(self.device)
        # P23 / E4.1: the learner decides in the merged-influence view. Pool opponents decide in
        # their own (OpponentPool.merged), so each env's two sides are set per iteration; see
        # `_apply_views`. The searcher builds E4 masks for its targets, so the two are not
        # combined until it learns the other view -- refused here rather than mis-targeted.
        self.merged_influence = bool(merged_influence)
        if self.merged_influence and search_ce_coef > 0.0:
            raise ValueError("--merged-influence with search CE is not supported: the searcher's "
                             "targets are built in the E4 view (P23)")

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
            action_dim=ActionEncoder.FLAT_ACTION_SIZE,
            device=self.device,
        )
        self.buffer.per_seat_adv_norm = bool(per_seat_adv_norm)
        #: --wolf-seat-weight. Each seat's PPO surrogate is scaled by `wolf_seat_weights`, driven by
        #: an exponential average of the USSR's win share in pure self-play games (both seats the
        #: current policy). The entropy bonus, the KL to pi_ref and the value loss are left
        #: unweighted, so on the losing seat the policy gradient gains on the entropy bonus, and
        #: the winning seat is held closer to pi_ref. Off by default, and off leaves the update
        #: untouched.
        self.wolf_seat_weight = bool(wolf_seat_weight)
        self.wolf_power = float(wolf_power)
        #: Games of memory in the average: each iteration's self-play games move it by
        #: alpha = min(1, games / wolf_ema_games).
        self.wolf_ema_games = float(wolf_ema_games)
        #: What the per-seat weights scale. "surrogate" (the first version, E4-38) scales the PPO
        #: surrogate alone; the unweighted entropy bonus then outweighs the down-weighted seat's
        #: policy gradient and pushes that seat toward uniform, and E4-38's entropy stayed ~1.75
        #: for 60M. "policy" scales the seat's whole policy objective -- surrogate, entropy bonus
        #: and KL to pi_ref together -- which is a per-seat learning rate, WoLF as published.
        if wolf_scope not in ("surrogate", "policy"):
            raise ValueError(f"wolf_scope must be 'surrogate' or 'policy', got {wolf_scope!r}")
        self.wolf_scope = str(wolf_scope)
        #: Half-width of the band around an even self-play split where the weights stay 1.
        self.wolf_dead_zone = float(wolf_dead_zone)
        #: Rollout forwards replayed as CUDA graphs (ai/training/graphed_forward.py): bitwise the
        #: same kernels as eager, one launch instead of ~317, and the learner and pool-opponent
        #: graphs overlap on two streams. CUDA only; --no-cuda-graphs falls back to eager.
        self._graphs: Optional[GraphCache] = None
        self._opp_stream: Optional[torch.cuda.Stream] = None
        if cuda_graphs and self.device.type == "cuda":
            self._graphs = GraphCache(self.num_envs, self.buffer.obs_dim,
                                      ActionEncoder.FLAT_ACTION_SIZE, self.device)
            self._opp_stream = torch.cuda.Stream(device=self.device)
        #: The USSR's smoothed self-play win share. Starts even, and is carried in the resume
        #: state so a resumed run does not relearn it.
        self.wolf_sp_ussr = 0.5

        # Rollout temperature bands. The default four are all BELOW 1.0, so sampling is
        # softmax(logits / tau) with tau < 1 -- sharper than the policy itself, in every band.
        # The comment this replaced called that "exploration"; relative to the policy's own
        # distribution it is the opposite, and nobody has measured whether sharpening rollouts
        # helps. `rollout_temps` makes the band an argument so that question can be asked.
        _bands = list(rollout_temps) if rollout_temps else [0.15, 0.50, 0.10, 0.35]
        if len(_bands) != 4:
            raise ValueError(f"rollout_temps needs exactly 4 values, got {len(_bands)}")
        if any(t <= 0.0 for t in _bands):
            raise ValueError(f"rollout temperatures must be positive, got {_bands}")
        if self.temperature_schedule and self.num_envs > 1:
            temps = np.zeros(self.num_envs, dtype=np.float32)
            temps[0 : self.num_envs // 4] = _bands[0]
            temps[self.num_envs // 4 : self.num_envs // 2] = _bands[1]
            temps[self.num_envs // 2 : 3 * self.num_envs // 4] = _bands[2]
            temps[3 * self.num_envs // 4 :] = _bands[3]
            if rollout_temps:
                print(f"[rollout temps] bands {_bands} (default is "
                      f"[0.15, 0.50, 0.10, 0.35], all sharpening)", flush=True)
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

    def _side_views(self, env_idx: int) -> Tuple[bool, bool]:
        """(US decides merged, USSR decides merged) for one env this iteration."""
        mine = self.merged_influence
        pool = self.opponent_pool
        if pool is None or not pool.is_mixed[env_idx]:
            return mine, mine
        theirs = bool(pool.current_merged)
        return (mine, theirs) if int(pool.learner_side[env_idx]) == 1 else (theirs, mine)

    def _uniform_view(self) -> Optional[bool]:
        """The single view every side of every env is in, or None when they differ."""
        pool = self.opponent_pool
        if pool is None or not pool.is_mixed.any() or bool(pool.current_merged) == self.merged_influence:
            return self.merged_influence
        return None

    def _apply_views(self) -> None:
        """P23: point every env's two sides at the right action view for this iteration.

        A no-op for an all-E4 run, so E4 training is untouched. The env skips a runner call when
        nothing changed, and a uniform view is set in one call.
        """
        if self.env is None:
            return
        uniform = self._uniform_view()
        if uniform is not None:
            if uniform or self.env.merged_us.any() or self.env.merged_ussr.any():
                if not (np.all(self.env.merged_us == uniform) and np.all(self.env.merged_ussr == uniform)):
                    self.env.set_merged_influence(uniform, uniform)
            return
        us = np.zeros(self.num_envs, dtype=bool)
        ussr = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            us[i], ussr[i] = self._side_views(i)
        self.env.set_merged_influence(us, ussr)

    def _apply_view_env(self, env_idx: int) -> None:
        if self.env is None or (not self.merged_influence and self.opponent_pool is not None
                                and not any(self.opponent_pool.merged)):
            return
        us, ussr = self._side_views(env_idx)
        self.env.set_merged_influence_env(env_idx, us, ussr)

    def _graphed_forward(self, obs_t: torch.Tensor, masks_t: torch.Tensor,
                         learner_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rollout forward as CUDA-graph replays: the learner over every env, and in a mixed
        batch the pool opponent over every env too, on its own stream so the two overlap. The
        opponent's rows are then copied into the learner's logits. Values always come from the
        learner (see the eager branch in collect_rollouts for why)."""
        assert self._graphs is not None and self._opp_stream is not None
        g_l = self._graphs.get(self.active_net)
        g_l.load(obs_t, masks_t)
        mixed = self.opponent_pool is not None and not bool(learner_np.all())
        if not mixed:
            g_l.replay()
            logits, v_win_t, v_vp_t = g_l.outputs()
            return logits, v_win_t, v_vp_t
        assert self.opponent_pool is not None
        g_o = self._graphs.get(self.opponent_pool.current)
        main = torch.cuda.current_stream(self.device)
        self._opp_stream.wait_stream(main)
        with torch.cuda.stream(self._opp_stream):
            g_o.load(obs_t, masks_t)
            g_o.replay()
        g_l.replay()
        main.wait_stream(self._opp_stream)
        logits, v_win_t, v_vp_t = g_l.outputs()
        logits = logits.float()
        opp_idx = torch.from_numpy(np.flatnonzero(~learner_np)).to(self.device)
        logits.index_copy_(0, opp_idx, g_o.static_out[0].index_select(0, opp_idx).float())
        return logits, v_win_t, v_vp_t

    def collect_rollouts(self) -> Dict[str, Any]:
        """Collects on-policy rollouts across all parallel environments."""
        t0 = time.time()
        self.active_net.eval()
        if self._graphs is not None:
            # Release the graphs of pool members evicted since the last rollout.
            self._graphs.retain([self.active_net] + (list(self.opponent_pool.nets)
                                                     if self.opponent_pool is not None else []))
        if self.opponent_pool is not None:
            self.opponent_pool.start_iteration()
            self._selfplay_mask = ~self.opponent_pool.is_mixed
        else:
            self._selfplay_mask = np.ones(self.num_envs, dtype=bool)
        self._apply_views()
        self.buffer.reset()
        seat_entropy_sum = {1: torch.zeros((), device=self.device), -1: torch.zeros((), device=self.device)}
        seat_entropy_n = {1: torch.zeros((), device=self.device), -1: torch.zeros((), device=self.device)}
        sp_games = 0.0
        sp_us_wins = 0.0
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
            _dp = np.asarray(self.env.runner.get_decision_players(), dtype=np.int8)
            if self.opponent_pool is not None:
                learner_np = self.opponent_pool.learner_acts(_dp)
            else:
                learner_np = np.ones(self.num_envs, dtype=bool)
            learner_t = torch.from_numpy(learner_np).to(self.device)

            with torch.no_grad():
                if self._graphs is not None:
                    logits, v_win_t, v_vp_t = self._graphed_forward(obs_t, masks_t, learner_np)
                elif self.opponent_pool is None or bool(learner_np.all()):
                    logits, v_win_t, v_vp_t = self.active_net(obs_t, masks_t)
                else:
                    # One learner pass over the whole batch gives the learner's logits AND the
                    # values at every state (GAE must bootstrap with the learner's critic, opponent-
                    # chosen states included), and the opponent runs on its own rows only. This
                    # used to be three passes -- opponent rows, learner rows, then the learner again
                    # over everything for values -- about twice the work of a full batch.
                    #
                    # The opponent's rows are selected by index, computed on the host from
                    # `learner_np`, which is already there: a boolean mask on the device needs a
                    # nonzero() and so a CPU-GPU sync each time it is used.
                    logits, v_win_t, v_vp_t = self.active_net(obs_t, masks_t)
                    logits = logits.float()
                    opp_idx = torch.from_numpy(np.flatnonzero(~learner_np)).to(self.device)
                    opp_logits, _, _ = self.opponent_pool.current(
                        obs_t.index_select(0, opp_idx), masks_t.index_select(0, opp_idx))
                    logits.index_copy_(0, opp_idx, opp_logits.float())
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

                # Per-seat policy entropy of the LEARNER's own decisions. Every collapse shows
                # entropy rising, and the pooled figure cannot say on which seat. Masked logits
                # are finite (-1e9), so p * log p is exactly zero there. Kept on the device: one
                # reduction per seat per step, no host sync.
                _ent = -(unscaled_log_probs.exp() * unscaled_log_probs).sum(dim=-1)
                _dp_t = torch.from_numpy(_dp).to(self.device)
                for _code in (1, -1):
                    _sel = learner_t & (_dp_t == _code)
                    seat_entropy_sum[_code] += (_ent * _sel).sum()
                    seat_entropy_n[_code] += _sel.sum()

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
                    sp_games += 1.0
                    sp_us_wins += 1.0 if _vp > 0 else (0.5 if _vp == 0 else 0.0)
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
                    if self.opponent_pool.seat_balance:
                        # Balancing may have turned this env mixed or back to self-play for its
                        # next episode; the self-play mask must follow, or the next game's result
                        # is filed under the wrong kind.
                        self._selfplay_mask[_i] = not bool(self.opponent_pool.is_mixed[_i])
                    # The learner's side was just redrawn, so this env's views may have swapped.
                    self._apply_view_env(_i)

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

        if self.opponent_pool is not None and self.opponent_pool.seat_balance:
            self.opponent_pool.observe_selfplay(sp_us_wins, sp_games)
        if self.wolf_seat_weight and sp_games > 0:
            _alpha = min(1.0, sp_games / self.wolf_ema_games)
            self.wolf_sp_ussr = ((1.0 - _alpha) * self.wolf_sp_ussr
                                 + _alpha * (1.0 - sp_us_wins / sp_games))

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
        if self.wolf_seat_weight:
            _w_us, _w_ussr = wolf_seat_weights(self.wolf_sp_ussr, self.wolf_power,
                                               self.wolf_dead_zone)
            metrics["wolf_sp_ussr"] = self.wolf_sp_ussr
            metrics["wolf_w_us"] = _w_us
            metrics["wolf_w_ussr"] = _w_ussr
        for _code, _tag in ((1, "us"), (-1, "ussr")):
            _n = float(seat_entropy_n[_code].item())
            if _n > 0:
                metrics[f"entropy_{_tag}"] = float(seat_entropy_sum[_code].item()) / _n
        metrics.update(self.critic_tracker.metrics())
        # Pool size and span, so a pool that silently stops growing is visible as a flat line
        # rather than being invisible. Without this the mechanism cannot be verified from a run.
        if self.opponent_pool is not None:
            metrics.update(self.opponent_pool.stats())
        # How much of the searcher's answer the real mask rejected. A determinized search can
        # legitimately propose an action that is illegal in the true state, so a small nonzero
        # rate is expected and healthy; it going to zero would mean the filter is not reaching
        # the targets, and it being large would mean the determinization has drifted far from
        # the real position.
        if self._searcher is not None:
            metrics["search_dropped_visit_frac"] = float(
                getattr(self, "search_dropped_visit_frac", 0.0))
            metrics["search_dropped_row_frac"] = float(
                getattr(self, "search_dropped_row_frac", 0.0))
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
        did return **that are legal in the real state**.

        That last clause used to read "legality is not re-checked here: BatchedMCTS already
        filters its answer against the caller's own mask". That is true of the *agent* path and
        not of `run()`, which is what this calls. The search is DETERMINIZED, and
        `batched_mcts.py` says so where it handles the same problem for an acting agent: "a
        determinized search can legitimately return an action that is illegal in the real state,
        because in this game the legal SET itself can depend on hidden information" -- so there
        the search proposes and the true mask disposes. Here nothing disposed, and visits on
        actions that are legal only under the sampled determinization were written straight into
        the target.

        The symptom was visible all along in `search_ce`, which exceeded 1e5 in about a third of
        iterations on every search arm. The mask fill is -1e9, so target mass e on a masked action
        costs e * 1e9; one stray visit out of 64 simulations is 0.016 of a row, and diluted across
        the searched rows in a batch that lands at ~1e5 of mean CE. Training was not visibly
        harmed -- the CE gradient is (pi - p_target), so the contribution is ~1e-4 -- but the
        target was wrong, and the same misalignment on a determinization-legal action that is
        also *really* legal would have been silent.

        Applies from the next launch: a running process has already imported this module, so
        E3-35-28 and anything else in flight keep the old behaviour.

        **Reads the runner's current state, so it must be called while that state is still the
        one the target will be stored against** -- before `env.step`, not after.
        """
        if self._searcher is None:
            return None, None
        import numpy as _np
        import ts_engine as ts

        from bindings.action_encoder import ActionEncoder

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

        dropped_visits = 0.0
        dropped_rows = 0
        for i, (acts, visits) in zip(idx, res):
            if not acts:
                continue
            v = _np.asarray(visits, dtype=_np.float32)
            # The real state's mask is the only authority on what may be played -- the same rule
            # batched_mcts.py applies for an acting agent. Drop visits the determinization made
            # look legal, then normalise over what survives, so the target stays a distribution.
            legal_mask = _np.asarray(ActionEncoder.get_legal_mask(states[i]))
            probs, dropped = filter_search_visits(acts, v, legal_mask, int(pi.shape[1]))
            if dropped > 0.0:
                dropped_visits += dropped
                dropped_rows += 1
            if probs is None:
                # Every visit was illegal here: no usable target rather than a guessed one.
                continue
            for a, p in probs:
                pi[i, a] = p
            flag[i] = 1.0
        self.search_dropped_visit_frac = (
            dropped_visits / max(1, len(idx)))
        self.search_dropped_row_frac = dropped_rows / max(1, len(idx))
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
        # The objective that is actually differentiated, and the KL regulariser's contribution to
        # it. `policy_loss` below reports only the PPO surrogate, so a regulariser that grows to
        # dominate the update leaves no trace in any logged series.
        policy_loss_total_accum = 0.0
        kl_term_accum = 0.0
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
        # How much of the search target lands OUTSIDE the legal mask. Inferred until now from the
        # size of search_ce (mask fill -1e9, so CE ~ 1e5 implies ~1e-4 of misplaced mass);
        # measured directly here, because the inference only works while the misplaced action is
        # illegal and says nothing about the legal-but-wrong case.
        search_illegal_mass_accum = 0.0      # worst single row in the iteration
        search_illegal_rows_accum = 0        # rows with any mass outside the mask
        search_rows_seen_accum = 0           # searched rows examined, for the rate
        # Entropy of the search target itself. Compare against `entropy`, the policy's own: the
        # decline's signature is policy entropy RISING, and whether the target's rises with it
        # decides whether the CE term is teaching that or merely failing to prevent it.
        search_target_entropy_accum = 0.0
        search_target_entropy_rows = 0
        # The two norms behind search_ce_grad_frac, kept separately. The ratio hides which side
        # moved, which is exactly what the open question turns on.
        ce_norm_accum = 0.0
        total_norm_accum = 0.0
        # Importance-ratio diagnostics. E3-32-30 went from healthy to NaN logits in one
        # iteration, and the stated cause -- exp() overflowing -- does not survive the
        # arithmetic: a log-prob is <= 0, so with the opening's measured -16.6 the exponent
        # reaches only ~16.6 and exp(16.6) is 1.6e7, far inside float32. The likelier path is a
        # large ratio meeting a NEGATIVE advantage, where PPO's min() selects the unclipped
        # branch and the gradient scales with the ratio; clip_grad_norm_ then turns an inf norm
        # into a zero scale and 0 * inf into NaN. These three separate the two stories.
        logratio_max_accum = -1e30      # before clamping, so the clamp cannot hide it
        old_lp_min_accum = 1e30         # how negative a stored log-prob actually gets
        ratio_negadv_max_accum = 0.0    # the dangerous combination, on its own
        num_updates = 0
        wolf_w_us, wolf_w_ussr = wolf_seat_weights(self.wolf_sp_ussr, self.wolf_power,
                                                   self.wolf_dead_zone)
        # Per-minibatch diagnostics accumulate ON THE DEVICE and are read once, after the loop.
        # Reading each with .item()/float()/bool() inside the loop forced ~14 CPU-GPU syncs per
        # minibatch, each stalling the queue until the GPU caught up
        # (research/log/training_throughput_cpu.md).
        def _z(v: float = 0.0) -> torch.Tensor:
            # float64, as the Python floats these replace were.
            return torch.full((), v, dtype=torch.float64, device=self.device)
        loss_t, policy_loss_t, policy_loss_total_t = _z(), _z(), _z()
        val_loss_t, kl_t, entropy_t, clip_frac_t, risk_loss_t = _z(), _z(), _z(), _z(), _z()
        logratio_max_t, old_lp_min_t, ratio_negadv_max_t = _z(-1e30), _z(1e30), _z()

        # pi_ref's log-probabilities over the whole buffer, once. pi_ref is frozen for the whole
        # update (it is refreshed after train_step, in train_iteration) and always in eval mode,
        # and the rollout data does not change across epochs -- so recomputing them for every
        # minibatch of every epoch did num_epochs times the work for the same numbers. Chunked at
        # the minibatch size, so each forward has the shape the per-minibatch one had.
        _n_all = self.buffer.buffer_size * self.num_envs
        _all_obs = self.buffer.obs.view(_n_all, self.buffer.obs_dim)
        _all_masks = self.buffer.masks.view(_n_all, self.buffer.action_dim)
        with torch.no_grad():
            ref_log_p_all = torch.cat([
                F.log_softmax(self.reference_net(_all_obs[i:i + self.batch_size],
                                                 _all_masks[i:i + self.batch_size])[0], dim=-1)
                for i in range(0, _n_all, self.batch_size)])

        for _ in range(self.num_epochs):
            for (b_obs, b_mask, b_act, b_old_lp, b_adv, b_ret_win, b_ret_vp,
                 b_defcon_risk, b_learner, b_search_pi, b_has_search,
                 b_players, b_idx) in self.buffer.get_batches(
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

                # validate_args=False: the logits come straight from the network, and validation ran a
                # support check that forced a CPU-GPU sync on every minibatch -- ~17% of the update
                # (research/log/training_throughput_cpu.md). The distribution itself is unchanged.
                cur_dist = torch.distributions.Categorical(logits=cur_logits, validate_args=False)
                cur_lp = cur_dist.log_prob(b_act)
                cur_entropy = cur_dist.entropy()
                # Masked with the policy loss below; the KL to pi_ref is deliberately
                # left over the whole batch, for broader state coverage.

                # Bound the exponent before exp(). The log-ratio is unbounded below by the
                # stored log-prob: an action the policy assigned ~6e-8 -- which is exactly what
                # --setup-explore-frac forces, since the opening carries a 16.56 logit gap --
                # gives old_lp near -16.6, and a modest shift then overflows exp() to inf.
                # Clipping the ratio afterwards does not help, because inf survives clamp and
                # inf * advantage is NaN. E3-32-30 died this way at 53.7M steps with every
                # instrument healthy in the iteration before: AUC 0.781, entropy 1.04, KL 0.126.
                #
                # +-20 is far outside the PPO clip range, so this changes nothing a healthy
                # update would have done -- it only stops an overflow becoming NaN.
                log_ratio = cur_lp - b_old_lp
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                with torch.no_grad():
                    logratio_max_t = torch.maximum(logratio_max_t, log_ratio.max().double())
                    old_lp_min_t = torch.minimum(old_lp_min_t, b_old_lp.min().double())
                    ratio_negadv_max_t = torch.maximum(
                        ratio_negadv_max_t,
                        torch.where(b_adv < 0, ratio, torch.zeros_like(ratio)).max().double())
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                surrogate = -torch.min(surr1, surr2)
                if self.wolf_seat_weight:
                    # WoLF: the winning seat's surrogate is scaled down and the losing seat's
                    # up (see wolf_seat_weights). Only the surrogate; see __init__.
                    surrogate = surrogate * wolf_sample_weights(
                        b_players, wolf_w_us, wolf_w_ussr).to(surrogate.dtype)
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
                # Mean over the kept samples as sum / count, which needs no host round trip; with
                # nothing kept it is 0, as before.
                _keep_f = keep.to(surrogate.dtype)
                ppo_loss = (surrogate * _keep_f).sum() / _keep_f.sum().clamp(min=1.0)

                with torch.no_grad():
                    clip_frac_t += ((ratio < 1.0 - self.clip_eps)
                                    | (ratio > 1.0 + self.clip_eps)).float().mean()

                ref_log_p = ref_log_p_all.index_select(0, b_idx)

                cur_p = F.softmax(cur_logits, dim=-1)
                cur_log_p = F.log_softmax(cur_logits, dim=-1)
                kl_per = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1)
                kl_div = kl_per.mean()

                # Entropy over the learner's own actions only, for the same reason as the
                # surrogate: an entropy bonus on a frozen opponent's choices would push
                # the learner's policy toward states it did not choose to be in.
                _own_f = (b_learner > 0.5).to(cur_entropy.dtype)
                _own_n = _own_f.sum().clamp(min=1.0)
                own_entropy = (cur_entropy * _own_f).sum() / _own_n
                # P15-X4b: pull the policy toward the searcher, on searched decisions only.
                # Soft cross-entropy against the visit distribution.
                #
                # The target is SUPPOSED to be zero outside the legal set, and this comment used
                # to assert it was. It is not. With the mask fill at -1e9, target mass e on a
                # masked action contributes e * 1e9 to the CE, and search_ce is observed above
                # 100 -- impossible for a 212-way softmax, whose maximum is log(212) = 5.36 -- in
                # about a third of iterations on every search arm measured, implying ~1e-4 of the
                # target sitting outside the legal set.
                #
                # Training is not visibly harmed: the CE gradient is (pi - p_target), so a masked
                # action contributes ~1e-4 and search_ce_grad_frac stays flat through the spikes.
                # What it costs is the METRIC -- a third of the rows are unusable -- and what it
                # warns about is worse: mass landing on the wrong action is only detectable when
                # the wrong action happens to be illegal. The same misalignment placing mass on a
                # LEGAL wrong action would be silent, which is exactly how the earlier
                # search-target off-by-one survived.
                #
                # So measure it rather than infer it from the loss magnitude. This is a
                # diagnostic only; the loss is deliberately left alone, because changing it would
                # change the experiment for arms in flight.
                search_ce = cur_logits.sum() * 0.0
                if self.search_ce_coef > 0.0 and bool((b_has_search > 0.5).any()):
                    sel = b_has_search > 0.5
                    search_ce = -(b_search_pi[sel] * cur_log_p[sel]).sum(dim=-1).mean()
                    with torch.no_grad():
                        # Entropy of the TARGET, which is the one quantity that separates two
                        # readings of the decline. Its signature is policy entropy RISING while
                        # strength falls, which is backwards for a policy that is merely
                        # over-sharpening. One explanation is a feedback loop: if the searcher's
                        # visit distribution flattens -- because the network guiding it has
                        # weakened, or because the positions reached are less decisive -- then
                        # the CE term is actively TEACHING the policy to be flatter, which
                        # weakens it further. If that is what happens, target entropy rises
                        # before or with policy entropy. If target entropy stays flat while the
                        # policy's rises, the CE term is not the thing doing it and the loop is
                        # dead as an explanation.
                        _t = b_search_pi[sel].clamp_min(1e-12)
                        _tent = -(b_search_pi[sel] * _t.log()).sum(dim=-1)
                        search_target_entropy_accum += float(_tent.mean())
                        search_target_entropy_rows += 1
                        _legal = b_mask[sel].bool()
                        _illegal_mass = (b_search_pi[sel] * (~_legal).to(b_search_pi.dtype))
                        _row = _illegal_mass.sum(dim=-1)
                        search_illegal_mass_accum = max(
                            search_illegal_mass_accum, float(_row.max()))
                        search_illegal_rows_accum += int((_row > 1e-9).sum())
                        search_rows_seen_accum += int(_row.numel())

                kl_term = kl_div
                ent_term = own_entropy
                if self.wolf_seat_weight and self.wolf_scope == "policy":
                    # WoLF as a per-seat learning rate: the entropy bonus and the KL to pi_ref are
                    # scaled with the surrogate, so a seat's own balance between them is unchanged
                    # and only its speed is. The logged kl_div and entropy stay unweighted.
                    _w = wolf_sample_weights(b_players, wolf_w_us, wolf_w_ussr).to(kl_per.dtype)
                    kl_term = (kl_per * _w).mean()
                    ent_term = (cur_entropy * _w * _own_f).sum() / _own_n
                policy_loss = (ppo_loss + self.eta * kl_term - self.ent_coef * ent_term
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
                    risk_loss_t += risk_loss.detach()

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
                if ce_norm is not None:
                    # Only the search-CE diagnostics read the whole-model gradient norm, so it is
                    # not computed (a few hundred small kernels) on every minibatch otherwise.
                    _tot_sq = [(q.grad.detach() ** 2).sum()
                               for q in self.active_net.parameters() if q.grad is not None]
                    total_norm = (torch.sqrt(torch.stack(_tot_sq).sum()) if _tot_sq
                                  else torch.zeros((), device=self.device))
                    search_ce_accum += float(search_ce)
                    search_ce_frac_accum += float(ce_norm / total_norm.clamp(min=1e-9))
                    search_rows_accum += 1
                    # The ratio alone cannot say WHICH side moved, and that is now the open
                    # question: on E3-35-28 search_ce_grad_frac falls 0.238 -> 0.025 while the
                    # search targets stay sharp (measured) and the KL is ruled out (measured).
                    # A policy drifting AWAY from a fixed target should give a LARGER
                    # (pi - p_target), so either the CE gradient is shrinking anyway or the other
                    # terms are growing past it. Logging both norms separates those in one look.
                    ce_norm_accum += float(ce_norm)
                    total_norm_accum += float(total_norm)
                nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                loss_t += loss.detach()
                # NOTE: this is the PPO surrogate ALONE, not the assembled policy_loss. Kept as
                # it is so the series stays comparable across every run ever logged; the
                # assembled objective and the regulariser's share are logged separately below,
                # because their absence is how a KL term reaching 300x the surrogate stayed
                # invisible for two days. See research/log/P15_kl_domination.md.
                policy_loss_t += ppo_loss.detach()
                policy_loss_total_t += policy_loss.detach()
                val_loss_t += val_loss.detach()
                kl_t += kl_div.detach()
                entropy_t += cur_entropy.mean().detach()
                num_updates += 1

        # One host read for everything accumulated on the device.
        (total_loss_accum, policy_loss_accum, policy_loss_total_accum, val_loss_accum, kl_accum,
         entropy_accum, clip_frac_accum, risk_loss_accum, logratio_max_accum, old_lp_min_accum,
         ratio_negadv_max_accum) = torch.stack([
            loss_t, policy_loss_t, policy_loss_total_t, val_loss_t, kl_t, entropy_t, clip_frac_t,
            risk_loss_t, logratio_max_t, old_lp_min_t, ratio_negadv_max_t]).tolist()
        kl_term_accum = float(self.eta) * kl_accum

        return {
            "loss": total_loss_accum / max(1, num_updates),
            "defcon_risk_loss": risk_loss_accum / max(1, num_updates),
            "policy_loss": policy_loss_accum / max(1, num_updates),
            # The assembled objective, and how much of it is the KL pull toward pi_ref. When
            # kl_term approaches or exceeds |policy_loss|, the update has stopped being policy
            # improvement and become regularisation.
            "policy_loss_total": policy_loss_total_accum / max(1, num_updates),
            "kl_term": kl_term_accum / max(1, num_updates),
            "val_loss": val_loss_accum / max(1, num_updates),
            "kl_div": kl_accum / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
            # 0.0 when search CE is off, so the key is always present and a run without it is
            # still distinguishable from a run whose term silently produced nothing.
            "search_ce": search_ce_accum / max(1, search_rows_accum),
            "search_ce_grad_frac": search_ce_frac_accum / max(1, search_rows_accum),
            # The numerator and denominator of that ratio, and how many minibatches carried a
            # searched row at all. Together they say whether a falling share means the CE
            # gradient shrank, the rest of the update grew, or simply that fewer rows were
            # searched -- three different problems with three different fixes.
            #
            # These will NOT divide to give search_ce_grad_frac, and neither is wrong:
            # grad_frac is the mean of per-minibatch RATIOS, while these are means of the norms
            # themselves, and E[a/b] != E[a]/E[b]. Measured on a smoke: 0.5567 / 4.3237 = 0.129
            # against a reported grad_frac of 0.191. Read each series for its own trend rather
            # than reconciling them against each other.
            "search_ce_grad_norm": ce_norm_accum / max(1, search_rows_accum),
            "total_grad_norm": total_norm_accum / max(1, search_rows_accum),
            "search_ce_rows": float(search_rows_accum),
            # Target/mask alignment. A nonzero rate means the searcher is recommending actions
            # the current mask forbids, which is a misalignment whose legal-action counterpart
            # would be invisible -- see the comment beside the CE term.
            # Read beside "entropy": if this rises with the policy's, the CE term is teaching the
            # policy to be flatter; if it stays put, the loop is not the explanation.
            "search_target_entropy": (
                search_target_entropy_accum / search_target_entropy_rows
                if search_target_entropy_rows else 0.0),
            "search_target_illegal_mass_max": search_illegal_mass_accum,
            "search_target_illegal_row_frac": (
                search_illegal_rows_accum / search_rows_seen_accum
                if search_rows_seen_accum else 0.0),
            # Maxima and a minimum over the iteration, not means: a single pathological sample
            # is what poisons a batch, and an average would bury it.
            "logratio_max": logratio_max_accum if num_updates else 0.0,
            "old_logprob_min": old_lp_min_accum if num_updates else 0.0,
            "ratio_negadv_max": ratio_negadv_max_accum,
        }
