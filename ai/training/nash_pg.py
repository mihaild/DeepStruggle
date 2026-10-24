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

import contextlib
import copy
import fcntl
import os
import time
from typing import Dict, List, Optional, Any, Sequence, Tuple, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.coldwar_net_v2 import VP_LIMIT

from bindings.ts_env import TsVectorizedEnv, model_obs_features
from bindings.action_encoder import ActionEncoder
from .rollout_buffer import RolloutBuffer, setup_phase_slot

from bindings.action_encoder import ActionEncoder as _AE

#: The play-mode block of the flat action space (EVENT, SPACE, OPS_INFLUENCE, OPS_COUP, OPS_REALIGN)
#: and the rows that choose among it, defined once in show_and_decide (P31 1a builds on them).
from .show_and_decide import (EVENT_SLOT, PLAY_MODE_HI, PLAY_MODE_LO,  # noqa: E402,F401
                              ApplicableEventForcer, ScenarioSeeder, apply_floor, floor_eps,
                              floor_rows, play_mode_rows)
from .mode_cf import ModeCounterfactual, mode_cf_loss  # noqa: E402
from .turn_credit import TurnCredit  # noqa: E402
from ai.search.rollout_root import ContinuationValues, RolloutConfig, RolloutTeacher, TeacherPolicyTargets  # noqa: E402
from .search_workers import SearchWorkers  # noqa: E402
from .searched_games import SearchedGames  # noqa: E402
from .forced_openings import OpeningSeeder  # noqa: E402
from .critic_tracker import CriticTracker
from .graphed_forward import GraphCache

# Fixed-probe entropy: number of (observation, mask) pairs frozen at the start of
# training, and how often (in iterations) the probe is re-evaluated.
ENTROPY_PROBE_SIZE = 2000
#: --compile-update choices and the torch.compile mode each one uses (P26).
COMPILE_UPDATE_MODES: Dict[str, str] = {"off": "", "default": "default",
                                        "max-autotune": "max-autotune-no-cudagraphs"}
ENTROPY_PROBE_INTERVAL = 10


#: Positions per call to the training searcher when a rollout's queued decisions are answered:
#: the batch the network runs at, and the size of the searcher's largest featuriser.
SEARCH_CHUNK = 1024


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


def wolf_seat_weights(sp_ussr: float, power: float = 1.0, dead_zone: float = 0.0,
                      dead_zone_mode: str = "shift") -> Tuple[float, float]:
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

    `dead_zone_mode="jump"` applies the plain rule unshifted once outside the zone: the weights jump
    from 1 at the edge, but a 0.90 split is braked 9:1 as without a zone, not 3:1. E4-41 (shift)
    sharpened and was stronger than its control on both seeds, but its seed 5 still drifted from
    0.67 to 0.96 against the softer brake.
    """
    if dead_zone_mode not in ("shift", "jump"):
        raise ValueError(f"dead_zone_mode must be 'shift' or 'jump', got {dead_zone_mode!r}")
    x = float(sp_ussr)
    d = max(0.0, float(dead_zone))
    if d > 0.0:
        off = x - 0.5
        if abs(off) <= d:
            return 1.0, 1.0
        if dead_zone_mode == "shift":
            x = 0.5 + (1.0 if off > 0 else -1.0) * (abs(off) - d)
    x = min(0.99, max(0.01, x))
    a = x ** float(power)
    b = (1.0 - x) ** float(power)
    return 2.0 * a / (a + b), 2.0 * b / (a + b)


#: Rollout temperature bands. The default samples at the policy itself (every band 1.0), so the
#: log-probabilities PPO records at temperature 1 are the ones the actions were drawn from.
#: Adopted 2026-09-27 (research/log/E5_06_rollout_temperature.md): from scratch it plateaus
#: +135/+170 above the old bands and 69-70% head to head, on two seeds.
DEFAULT_ROLLOUT_TEMPS: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
#: The bands every run used before then: all below 1.0, so every environment sampled sharper
#: than the policy while PPO recorded temperature-1 log-probabilities. A run whose metadata
#: records no `rollout_temps` used these.
LEGACY_ROLLOUT_TEMPS: Tuple[float, float, float, float] = (0.15, 0.50, 0.10, 0.35)


def opponent_actions(logits: torch.Tensor, actions: torch.Tensor, opp_rows: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """P24: re-choose the frozen opponent's actions at its own temperature.

    `logits` holds each row's deciding network's logits (the opponent's on `opp_rows`, masked to
    -1e9 off the legal set), and `actions` what the rollout sampled at the learner's temperature.
    At 0 the opponent plays its argmax -- the policy a greedy tournament measures -- otherwise it
    samples softmax(logits / temperature). Learner rows are returned unchanged.
    """
    if temperature <= 0.0:
        chosen = logits.argmax(dim=-1)
    else:
        chosen = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(1)
    return torch.where(opp_rows, chosen, actions)


def bonus_entropy(entropy: torch.Tensor, mask: torch.Tensor, normalize: bool) -> torch.Tensor:
    """The per-decision entropy the entropy bonus rewards.

    Raw by default. With `normalize` (--entropy-normalize) it is entropy / log(legal count), the
    fraction of that decision's maximum. A 50-option decision then carries no more room for the
    bonus than a 4-option one: E4.1's merged view grows the play- and op-mode decisions from ~4
    options to ~50, and a losing seat's policy there drifted to near-uniform (0.96 of its maximum)
    under the fixed bonus (research/log/P23_E4_1_ab.md). A forced decision (one legal action) has
    zero entropy and stays zero."""
    if not normalize:
        return entropy
    n = mask.to(entropy.dtype).sum(-1)
    return torch.where(n > 1.5, entropy / torch.log(n.clamp(min=2.0)), torch.zeros_like(entropy))


def wolf_sample_weights(players: torch.Tensor, w_us: float, w_ussr: float) -> torch.Tensor:
    """Each sample's surrogate weight from its acting seat: +1 US, -1 USSR, anything else 1."""
    ones = torch.ones(players.shape, dtype=torch.float32, device=players.device)
    return torch.where(players == 1, ones * w_us, torch.where(players == -1, ones * w_ussr, ones))


def teacher_kl(cur_log_p: torch.Tensor, teacher_log_p: torch.Tensor, rows: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """KL(teacher || policy) per row, averaged over the selected `rows`, and the row count.

    Both log-probabilities come from logits masked with the same legal set (fill -1e9), so a
    masked action has teacher probability exactly 0 and adds nothing; nothing here is clamped.
    No host sync: with no row selected the mean is 0.
    """
    t_p = teacher_log_p.exp()
    kl_row = (t_p * (teacher_log_p - cur_log_p)).sum(dim=-1)
    w = rows.to(kl_row.dtype)
    n = w.sum()
    return (kl_row * w).sum() / n.clamp(min=1.0), n


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


@contextlib.contextmanager
def phase_lock(path: str) -> Any:
    """An exclusive lock on `path` (created if absent) for the duration of the block; yields the
    seconds waited for it. flock is per open file, so it excludes other processes and other opens
    in this one; the kernel drops it if the holder dies."""
    t0 = time.time()
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield time.time() - t0
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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
        search_gumbel_k: int = 0,      # P32 T1: >0 targets the Gumbel root's improved policy
        search_prior_temperature: float = 1.0,
        search_teacher_net: Optional[nn.Module] = None,   # P32: a FROZEN network to search with
        search_value_beta: float = 0.0,   # P32 T2: the share of the search value in v_win's target
        search_rollout_horizon: int = 0,  # > 0: the search-CE targets come from a rollout root
        search_rollout_k: int = 4,
        search_rollout_worlds: int = 16,
        search_rollout_stop: str = "boundaries",   # where the rollout root's playouts stop (STOPS)
        search_rollout_leaf_side: str = "mover",   # whose side its leaves are read from (LEAF_SIDES)
        predeal_value_coef: float = 0.0,  # P33: the critic on both sides at the turn-end park (0 = off)
        searched_games_workers: int = 0,  # P33: worker processes playing whole games with the rollout root
        searched_games_envs: int = 32,    # games each of them plays at once
        searched_value_coef: float = 0.0,  # weight of the value loss on those games' decisions
        searched_value_buffer: int = 262144,  # their replay store, in rows (float16, on the CPU)
        searched_value_batch: int = 1024,  # rows of it sampled per minibatch
        search_workers: int = 0,          # > 0: the rollout-root targets priced by worker processes
        search_teacher_policy: bool = False,   # the frozen teacher's raw policy as the target (no search)
        teacher_net: Optional[nn.Module] = None,  # P32: a frozen policy to distil, one seat
        teacher_coef: float = 0.0,
        teacher_seat: int = 1,         # +1 US, -1 USSR, 0 both
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
        opponent_temperature: Optional[float] = None,
        adv_norm_learner_only: bool = False,
        wolf_seat_weight: bool = False,
        wolf_power: float = 1.0,
        wolf_ema_games: float = 2000.0,
        wolf_scope: str = "surrogate",
        wolf_dead_zone: float = 0.0,
        wolf_dead_zone_mode: str = "shift",
        adv_norm_floor: float = 0.0,
        entropy_ceiling: float = 0.0,
        entropy_ceiling_lr: float = 0.01,
        entropy_ceiling_min_coef: float = -0.02,
        entropy_ceiling_grace_steps: float = 5_000_000.0,
        target_kl: float = 0.0,
        entropy_normalize: bool = False,
        setup_entropy_floor: float = 0.0,
        setup_entropy_lr: float = 0.01,
        setup_entropy_max_coef: float = 1.0,
        setup_mc_credit: bool = False,
        setup_script_frac: float = 0.0,
        play_mode_temp: float = 1.0,
        setup_script_openings: Sequence[str] = (),
        play_mode_floor: float = 0.0,
        floor_scope: str = "all",
        floor_from: int = 0,
        floor_anneal_from: Optional[int] = None,
        floor_anneal_steps: int = 0,
        seed_scenarios: Sequence[str] = (),
        seed_frac: float = 0.0,
        seed_scenarios_from: int = 0,
        forced_openings_frac: float = 0.0,     # > 0: per side, that share of games start from a forced opening
        forced_openings_us: Sequence[str] = (),
        forced_openings_ussr: Sequence[str] = (),
        aux_opp_legality: float = 0.0,
        aux_opp_legality_frac: float = 0.1,
        aux_opp_legality_min_batch: int = 4096,
        mode_cf_coef: float = 0.0,
        mode_cf_subsample: int = 16,
        mode_cf_playouts: int = 1,
        mode_cf_from: int = 0,
        turn_credit_coef: float = 0.0,     # P32 B4': CE toward turn-end priced targets. 0 = off
        turn_credit_budget: int = 64,
        turn_credit_worlds: int = 16,
        turn_credit_beta: float = 10.0,
        turn_credit_k: float = 9.0,
        turn_credit_floor: float = 1e-3,
        turn_credit_nested_p: float = 0.0,
        turn_credit_nested_share: float = 0.33,
        turn_credit_narrow_share: float = 0.5,
        turn_credit_cap: int = 12,
        turn_credit_steps: int = 4,
        turn_credit_batch: int = 512,
        turn_credit_buffer: int = 8192,
        turn_credit_workers: int = 0,
        force_applicable_events: Sequence[str] = (),
        force_event_frac: float = 0.1,
        force_events_from: int = 0,
        force_event_credit: str = "own",
        floor_rows_kind: str = "play_mode_and_events",
        setup_mc_coef: float = 1.0,
        setup_mc_min_batch: int = 512,
        aux_own_coef: float = 0.0,
        aux_vp_coef: float = 0.0,
        aux_sample_frac: float = 0.1,
        aux_min_batch: int = 4096,
        aux_card_coef: float = 0.0,
        aux_card_sample_frac: float = 0.0005,
        aux_card_min_batch: int = 2048,
        aux_card_buffer: int = 65536,
        aux_card_steps: int = 4,
        aux_card_batch: int = 512,
        cuda_graphs: bool = True,
        compile_update: str = "off",
        z_loss_coef: float = 0.0,
        logit_gap_coef: float = 0.0,   # gap penalty: coef * mean relu(top - z - gap)^2 over legal moves
        logit_gap: float = 7.0,
        setup_block_lambda: bool = False,
        block_lambda: str = "off",
        phase_lock_path: Optional[str] = None,
        device: torch.device | str = "cuda",
    ):
        #: Two arms on one machine take turns: the rollout under `<path>.rollout`, the update under
        #: `<path>.update`, so while one collects the other updates (the rollout is CPU-bound, the
        #: update GPU-bound) instead of both doing the same at once. Each arm stays exactly serial
        #: -- the same data, order and numbers -- only the wall clock changes.
        self.phase_lock_path = phase_lock_path
        self.device = torch.device(device if (torch.cuda.is_available() and device == "cuda") else ("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"))
        self.active_net = active_net.to(self.device)
        # P23 / E4.1: the learner decides in the merged-influence view. Pool opponents decide in
        # their own (OpponentPool.merged), so each env's two sides are set per iteration; see
        # `_apply_views`. The searcher builds E4 masks for its targets, so the two are not
        # combined until it learns the other view -- refused here rather than mis-targeted.
        self.merged_influence = bool(merged_influence)
        # R49 (P33): the rollout root's greedy-continuation value as v_win's target, with no policy target.
        rollout_values = search_rollout_horizon > 0 and search_value_beta > 0.0 and search_ce_coef <= 0.0
        if rollout_values and (search_gumbel_k > 0 or search_workers > 0 or search_teacher_policy):
            raise ValueError("rollout-root value targets (--search-value-beta with --search-rollout-horizon) "
                             "take no --search-gumbel-k, --search-workers or --search-teacher-policy")
        if search_rollout_horizon > 0 and not rollout_values and (
                search_gumbel_k > 0 or search_ce_coef <= 0.0 or search_value_beta > 0.0):
            raise ValueError("rollout-root targets (--search-rollout-horizon) replace the Gumbel root: policy targets "
                             "need --search-ce-coef > 0 and neither --search-gumbel-k nor --search-value-beta; value "
                             "targets (R49) need --search-value-beta > 0 and --search-ce-coef 0")
        if search_teacher_policy and (search_teacher_net is None or search_ce_coef <= 0.0 or search_gumbel_k > 0
                                      or search_rollout_horizon > 0 or search_workers > 0):
            raise ValueError("--search-teacher-policy targets a frozen teacher's raw policy: it needs "
                             "--search-teacher-checkpoint and --search-ce-coef > 0, and no other searcher")
        if search_workers > 0 and (search_rollout_horizon <= 0 or search_teacher_net is None):
            raise ValueError("--search-workers prices rollout-root targets of a frozen teacher: it needs "
                             "--search-rollout-horizon > 0 and --search-teacher-checkpoint")
        if search_value_beta > 0.0 and search_gumbel_k <= 0 and not rollout_values:
            raise ValueError("search values into the critic need the Gumbel root (T2, --search-gumbel-k > 0) "
                             "or the rollout root's continuation (R49, --search-rollout-horizon > 0, no CE)")
        if self.merged_influence and (search_ce_coef > 0.0 or search_value_beta > 0.0):
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
        # P32: distil a frozen teacher's policy on the learner's own decisions in one seat -- a
        # seat-locked exploiter that plays that seat better than this network does. KL(teacher
        # || policy) beside the RL loss; rollouts and everything else are unchanged.
        self.teacher_coef = float(teacher_coef)
        self.teacher_seat = int(teacher_seat)
        self.teacher_net: Optional[nn.Module] = None
        if self.teacher_coef > 0.0:
            if teacher_net is None:
                raise ValueError("teacher_coef > 0 needs a teacher_net")
            self.teacher_net = teacher_net.to(self.device).eval()
            for _p in self.teacher_net.parameters():
                _p.requires_grad_(False)
        # P32 T2: the target for v_win at a searched decision is (1 - beta) * the lambda-return +
        # beta * the search value -- the improved policy's expected completed Q at the root. The
        # policy loss and the advantages are untouched, so it cannot flatten the policy.
        self.search_value_beta = float(search_value_beta)
        self._searcher: Optional[Any] = None   # BatchedMCTS (Gumbel / PUCT) or RolloutTeacher
        #: P36's gated active search (ai/training/active_search.py), set by the pipeline after
        #: construction: the root plays a fraction of the learner's entry decisions as environment
        #: once its gate is open, and its improved policy is the search-CE target there.
        self.active_search: Optional[Any] = None
        if self.search_ce_coef > 0.0 or self.search_value_beta > 0.0:
            from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
            # The network the searcher runs: the one being trained, or a frozen teacher. Searching
            # with the learner closes a loop -- targets built from the network follow it as it
            # changes (R29: target entropy 0.17 -> 0.53 behind the policy's 0.21 -> 0.93); a frozen
            # teacher's targets hold still, which is what standard distillation relies on.
            if search_teacher_net is not None:
                search_teacher_net = search_teacher_net.to(self.device).eval()
                for _p in search_teacher_net.parameters():
                    _p.requires_grad_(False)
            if rollout_values:
                # R49: greedy continuations from pi_ref -- the learner's own frozen snapshot, refreshed in
                # place with the reference policy (a frozen teacher if one is given).
                self._searcher = ContinuationValues(
                    search_teacher_net if search_teacher_net is not None else self.reference_net,
                    RolloutConfig(k=1, worlds=int(search_rollout_worlds), horizon=int(search_rollout_horizon),
                                  rule="argmax", stop=str(search_rollout_stop), cuda_graphs=bool(cuda_graphs),
                                  leaf_side=str(search_rollout_leaf_side)),
                    device=self.device, subsample=search_subsample, node_filter=search_node_filter)
            elif search_teacher_policy:
                # The control for the rollout teacher: the teacher's own policy, no search.
                self._searcher = TeacherPolicyTargets(search_teacher_net, device=self.device,
                                                      subsample=search_subsample, node_filter=search_node_filter)
            elif search_rollout_horizon > 0:
                # The rollout root as the teacher's search (R32's mechanics, another searcher): its
                # candidates' rollout values give the improved policy (ai/search/rollout_root.py).
                self._searcher = RolloutTeacher(
                    search_teacher_net if search_teacher_net is not None else active_net,
                    RolloutConfig(k=int(search_rollout_k), worlds=int(search_rollout_worlds),
                                  horizon=int(search_rollout_horizon), rule="z2", stop=str(search_rollout_stop),
                                  cuda_graphs=bool(cuda_graphs), leaf_side=str(search_rollout_leaf_side)),
                    device=self.device, subsample=search_subsample, node_filter=search_node_filter)
            else:
                self._searcher = BatchedMCTS(
                    search_teacher_net if search_teacher_net is not None else active_net, device=self.device,
                    config=BatchedMCTSConfig(
                        simulations=search_sims, temperature=0.0, auto_advance=True,
                        advance_root=False, determinize=True,
                        node_filter=search_node_filter, subsample=search_subsample,
                        # P32 T1: the Gumbel root's improved policy as the target instead of visit
                        # counts -- sharp where the search agrees, moved where its values differ.
                        gumbel_k=int(search_gumbel_k), gumbel_scale=1.0,
                        fpu_reduction=0.2 if search_gumbel_k > 0 else 0.0,
                        gumbel_prior_temperature=float(search_prior_temperature)),
                    featurise_capacity=SEARCH_CHUNK)
            if rollout_values:
                print(f"[R49] rollout-root value targets: beta {search_value_beta}, {search_rollout_worlds} worlds, "
                      f"horizon {search_rollout_horizon}, stop {search_rollout_stop}, leaves read from the "
                      f"{search_rollout_leaf_side}'s side, subsample {search_subsample}, from "
                      f"{'the frozen teacher' if search_teacher_net is not None else 'pi_ref'}", flush=True)
            print(f"[X4b] search CE on: coef {self.search_ce_coef}, {search_sims} sims, "
                  f"{search_node_filter}, subsample {search_subsample}"
                  + (f", rollout-root targets k={search_rollout_k}, {search_rollout_worlds} worlds, horizon "
                     f"{search_rollout_horizon}" if search_rollout_horizon > 0 else
                     f", Gumbel improved-policy targets k={search_gumbel_k}, root T={search_prior_temperature}"
                     if search_gumbel_k > 0 else ", visit-count targets"), flush=True)
        self.search_gumbel_k = int(search_gumbel_k)
        self.search_rollout_horizon = int(search_rollout_horizon)
        self.search_teacher_policy = bool(search_teacher_policy)
        #: With workers, the queued decisions are streamed to them during the rollout and collected
        #: at its end, so the targets land on this rollout's rows as before (ai/training/search_workers.py).
        self._search_pool: Optional[SearchWorkers] = None
        if search_workers > 0:
            self._search_pool = SearchWorkers(
                int(search_workers), search_teacher_net,
                RolloutConfig(k=int(search_rollout_k), worlds=int(search_rollout_worlds),
                              horizon=int(search_rollout_horizon), rule="z2", stop=str(search_rollout_stop),
                              leaf_side=str(search_rollout_leaf_side)), self.device)
            print(f"[X4b] the rollout-root targets are priced by {search_workers} worker processes "
                  f"beside the rollout", flush=True)
        self.search_saturated_frac = 0.0
        #: (buffer step, env, cloned state) of every decision to search this rollout; searched in
        #: one batch by `_flush_search_targets` when the rollout ends.
        self._search_queue: List[Tuple[int, int, Any]] = []
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        #: P33, the pre-deal critic: each env's turn-end park (after a turn's last action, before the
        #: deal; nobody's decision) is recorded from both sides by the runner, and joins the value loss
        #: with that side's next-decision lambda-return as its target (the result where the game ends
        #: first). _predeal_rows holds (buffer step, envs, US rows, USSR rows) until the update.
        self.predeal_value_coef = float(predeal_value_coef)
        self._predeal_on = False
        self._predeal_rows: List[Tuple[int, np.ndarray, torch.Tensor, torch.Tensor]] = []
        self._pd_obs: Optional[torch.Tensor] = None
        self._pd_tgt: Optional[torch.Tensor] = None
        #: P33, the critic on searched games (ai/training/searched_games.py): workers playing whole games
        #: with the rollout root (stop=earlier, leaf=own, the measured best) on pi_ref, both sides; each
        #: decision with the game's result from its decider's side trains the value head only.
        self.searched_games_workers = int(searched_games_workers)
        self.searched_games_envs = int(searched_games_envs)
        self.searched_value_coef = float(searched_value_coef)
        self.searched_value_batch = int(searched_value_batch)
        self._sg: Optional[SearchedGames] = None
        self._sg_cap = int(searched_value_buffer)
        self._sg_obs: Optional[torch.Tensor] = None
        self._sg_tgt: Optional[torch.Tensor] = None
        self._sg_size = 0
        self._sg_next = 0
        if self.searched_games_workers > 0 and self.searched_value_coef <= 0.0:
            raise ValueError("--searched-games-workers needs --searched-value-coef > 0 (the rows train nothing else)")
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

        # --setup-script-frac (owner, 2026-10-02): in a drawn fraction of games both sides' setup is
        # played by one of `setup_script_openings` (tools/lib/openings.py), and those placements are
        # TRAINED as if the policy had chosen them. The stored log-prob is the policy's own, so the
        # ratio starts at 1 and the clip bounds how fast a scripted placement can rise; the credit is
        # the game result (--setup-mc-credit, required), so a scripted opening gains probability only
        # where its games beat the critic's baseline. Drawn afresh for every game.
        # --play-mode-temp (owner, 2026-10-02; the expert review's first candidate): the learner
        # samples its play-mode decision -- event, space or Ops for a chosen card -- at this
        # temperature, so events the policy gives p ~ 0.01 still get tried. Exploration only: the
        # stored log-prob is the policy's own (tau = 1), as for the rollout temperatures.
        if play_mode_temp <= 0.0:
            raise ValueError("--play-mode-temp must be positive")
        self.play_mode_temp = float(play_mode_temp)
        self.setup_script_frac = float(setup_script_frac)
        self.setup_script_openings: Tuple[str, ...] = tuple(setup_script_openings)
        self._script_open = np.full(self.num_envs, -1, dtype=np.int32)      # -1: its own setup
        self._script_cursor = {1: np.zeros(self.num_envs, dtype=np.int32),
                               -1: np.zeros(self.num_envs, dtype=np.int32)}
        self._script_rng = np.random.default_rng(67890)
        self._script_games = np.zeros(2, dtype=np.int64)                   # (scripted, own) games drawn
        self._script_scripts: List[Dict[int, List[int]]] = []
        if self.setup_script_frac > 0.0:
            from tools.lib.openings import NODE_OFFSET as _NODE, OPENINGS
            if not 0.0 < self.setup_script_frac <= 1.0:
                raise ValueError("--setup-script-frac must be in (0, 1]")
            if not self.setup_script_openings:
                raise ValueError("--setup-script-frac needs at least one opening")
            if not setup_mc_credit:
                raise ValueError(
                    "--setup-script-frac needs --setup-mc-credit: the scripted placements are trained "
                    "on the game result; through the lambda-return their credit would be the critic's "
                    "view of openings it has barely seen")
            if self.setup_explore_frac > 0.0:
                raise ValueError("--setup-script-frac and --setup-explore-frac both override setup actions")
            for name in self.setup_script_openings:
                o = OPENINGS.get(name)
                if o is None or {"US", "USSR"} - set(o):
                    raise ValueError(f"opening {name!r} is unknown or does not script both sides")
                self._script_scripts.append({1: [_NODE + c for c in o["US"]],
                                             -1: [_NODE + c for c in o["USSR"]]})
            self._script_draw(np.arange(self.num_envs))
            print(f"[setup script] {self.setup_script_frac:.0%} of games set up by one of "
                  f"{list(self.setup_script_openings)} (both sides), trained on with the policy's own "
                  f"log-prob and the game result as credit", flush=True)

        # P31 1a (owner 2026-10-06): a uniform floor in the learner's BEHAVIOUR policy at play-mode
        # decisions and the non-country choices inside events, mu = (1 - eps) pi + eps uniform. The
        # stored log-prob stays log pi; each sample's surrogate is weighted by pi_old / mu (see
        # apply_floor for why the ratio must not be taken against mu). Learner rows only:
        # a frozen opponent's actions are the environment. eps follows floor_eps(): 0 before
        # floor_from, then constant, then linear to 0 over floor_anneal_steps from floor_anneal_from.
        if play_mode_floor < 0.0 or play_mode_floor >= 1.0:
            raise ValueError("--play-mode-floor must be in [0, 1)")
        if floor_scope not in ("all", "seeded"):
            raise ValueError("--floor-scope must be 'all' or 'seeded'")
        if floor_scope == "seeded" and play_mode_floor > 0.0 and not seed_scenarios:
            raise ValueError("--floor-scope seeded needs --seed-scenarios: there are no seeded games")
        self.play_mode_floor = float(play_mode_floor)
        self.floor_scope = str(floor_scope)
        if floor_rows_kind not in ("play_mode_and_events", "event_choices"):
            raise ValueError("--floor-rows must be play_mode_and_events or event_choices")
        self.floor_rows_kind = str(floor_rows_kind)
        self.floor_from = int(floor_from)
        self.floor_anneal_from = None if floor_anneal_from is None else int(floor_anneal_from)
        self.floor_anneal_steps = int(floor_anneal_steps)
        self._floor_stats = torch.zeros(3, dtype=torch.float64, device=self.device)  # rows, uniform draws, learner rows
        #: The floor's per-sample behaviour weight pi_old / mu over the buffer, flat [T * N]; 1 off
        #: the floor. Applied to the surrogate outside the clip (see apply_floor).
        self._behaviour_w: Optional[torch.Tensor] = None
        if self.play_mode_floor > 0.0:
            print(f"[floor] eps {self.play_mode_floor:g} at play-mode and in-event choices "
                  f"({self.floor_scope} games), from {self.floor_from:,} steps"
                  + (f", annealed to 0 over {self.floor_anneal_steps:,} from {self.floor_anneal_from:,}"
                     if self.floor_anneal_from is not None else "")
                  + "; corrected by the weight pi_old / mu on the surrogate", flush=True)
        # P31 1b (owner 2026-10-06): in a drawn fraction of games, the listed precursor events are
        # played for the US as ENVIRONMENT -- stored with learner = 0, so no policy gradient.
        self.seeder: Optional[ScenarioSeeder] = None
        self._seed_started = False
        if seed_scenarios:
            if seed_frac <= 0.0:
                raise ValueError("--seed-scenarios needs --seed-frac > 0")
            self.seeder = ScenarioSeeder(list(seed_scenarios), float(seed_frac), self.num_envs,
                                         start_step=int(seed_scenarios_from))
            print(f"[seed scenarios] {list(seed_scenarios)} forced as environment in "
                  f"{seed_frac:.0%} of games from {int(seed_scenarios_from):,} steps (no policy "
                  f"gradient on the forced plays)", flush=True)
        elif seed_frac > 0.0:
            raise ValueError("--seed-frac needs --seed-scenarios")
        # Owner 2026-10-13: openings forced as environment, independently per side
        # (ai/training/forced_openings.py) -- the network learns to play on from them.
        self.opening_seeder: Optional[OpeningSeeder] = None
        self._openings_started = False
        if forced_openings_frac > 0.0:
            self.opening_seeder = OpeningSeeder(float(forced_openings_frac), list(forced_openings_us),
                                                list(forced_openings_ussr), self.num_envs)
            print(f"[forced openings] each side, independently, in {forced_openings_frac:.0%} of games: US "
                  f"{list(forced_openings_us)}, USSR {list(forced_openings_ussr)} -- as environment (no policy "
                  f"gradient on the forced placements or the Marshall Plan headline)", flush=True)
        # Owner 2026-10-06: at the learner's play-mode decision for Wargames / Arms Race / One Small
        # Step with the event legal and applicable, play the event with probability
        # force_event_frac, trained as the policy's own (show_and_decide.ApplicableEventForcer).
        self.event_forcer: Optional[ApplicableEventForcer] = None
        if force_event_credit not in ("own", "environment"):
            raise ValueError("--force-event-credit must be own or environment")
        #: "own": the forced event is the learner's play (learner 1, log pi); "environment": as P31 1b,
        #: learner 0 -- no policy gradient on it, the decisions after it are the policy's own.
        self.force_event_credit = str(force_event_credit)
        if force_applicable_events:
            self.event_forcer = ApplicableEventForcer(list(force_applicable_events), float(force_event_frac),
                                                      start_step=int(force_events_from))
            print(f"[force events] {list(force_applicable_events)}: event played in "
                  f"{float(force_event_frac):.0%} of the learner's applicable plays from "
                  f"{int(force_events_from):,} steps, credited as "
                  f"{'the policy' + chr(39) + 's own' if force_event_credit == 'own' else 'environment'}",
                  flush=True)

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
        #: The view spec's observation half: the feature set the learner reads (recorded in its
        #: weights). Every seat gets it; pool members must use the same feature set.
        self.obs_features = model_obs_features(active_net)
        if self.env is not None:
            self.env.set_obs_features(self.obs_features, self.obs_features)
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
        self.buffer.adv_norm_learner_only = bool(adv_norm_learner_only)
        #: P24: the temperature pool opponents play at in rollouts; None samples them exactly as
        #: the learner is sampled, as every run before it did.
        self.opponent_temperature: Optional[float] = (
            None if opponent_temperature is None else float(opponent_temperature))
        # P25 3j-3l, the levers that act on the collapse loop's own closing points. Each is off at
        # 0 and off leaves the update bitwise unchanged. None is defined together with WoLF or
        # per-seat normalisation, so those combinations are refused rather than half-applied.
        if adv_norm_floor < 0.0 or entropy_ceiling < 0.0 or target_kl < 0.0:
            raise ValueError("adv_norm_floor, entropy_ceiling and target_kl must be >= 0")
        if (adv_norm_floor > 0.0 or entropy_ceiling > 0.0 or target_kl > 0.0) and wolf_seat_weight:
            raise ValueError("the P25 3j-3l levers are not defined together with --wolf-seat-weight")
        if adv_norm_floor > 0.0 and per_seat_adv_norm:
            raise ValueError("--adv-norm-floor floors the shared divisor; it is not defined with "
                             "--per-seat-adv-norm")
        #: 3j: see RolloutBuffer.adv_norm_floor.
        self.buffer.adv_norm_floor = float(adv_norm_floor)
        #: 3k, --entropy-ceiling: a one-sided, per-seat entropy coefficient. Once per iteration,
        #: after grace_steps, c_seat -= lr * (H_seat - ceiling), clipped to [min_coef, ent_coef],
        #: with H_seat the seat's mean rollout entropy. It never exceeds ent_coef, so below the
        #: ceiling it is the fixed bonus; above it the bonus shrinks and then becomes a penalty.
        #: One-sided because a two-sided target would also push a sharpening seat back up.
        self.entropy_ceiling = float(entropy_ceiling)
        self.entropy_ceiling_lr = float(entropy_ceiling_lr)
        self.entropy_ceiling_min_coef = float(entropy_ceiling_min_coef)
        self.entropy_ceiling_grace_steps = float(entropy_ceiling_grace_steps)
        self.ent_coef_seat: Dict[int, float] = {1: float(ent_coef), -1: float(ent_coef)}
        #: 3l, --target-kl: per-seat early stopping. In each minibatch a seat's approximate KL
        #: from the rollout policy, E[(r - 1) - log r] over its own decisions, is measured before
        #: the step; once it exceeds the target the seat's policy terms are masked out for the rest
        #: of the update (the value loss continues). The per-seat KL is logged whether or not the
        #: target is set, because the target is chosen from the controls' own distribution.
        self.target_kl = float(target_kl)
        #: --entropy-normalize: the bonus rewards entropy / log(legal) per decision; see bonus_entropy.
        self.entropy_normalize = bool(entropy_normalize)
        #: --setup-entropy-floor: a one-sided entropy FLOOR on the learner's setup placements, in
        #: nats. A setup placement pushed to p ~ 1 is never sampled differently again, and both the
        #: policy gradient and the ordinary entropy bonus vanish there, so the opening locks,
        #: good or bad (research/log/E5_11_setup_lock_and_critic_views.md). An extra bonus on setup
        #: decisions only, whose coefficient moves by lr x (floor - rollout setup entropy) each
        #: iteration within [0, max_coef]: zero while the setup is above the floor, so it acts
        #: only once the opening starts to lock. Normalised like the ordinary bonus (per learner
        #: decision), so a coefficient of 0.01 is the ordinary bonus again on those decisions.
        if setup_entropy_floor < 0.0 or setup_entropy_lr < 0.0 or setup_entropy_max_coef < 0.0:
            raise ValueError("setup_entropy_floor, setup_entropy_lr and setup_entropy_max_coef must be >= 0")
        self.setup_entropy_floor = float(setup_entropy_floor)
        self.setup_entropy_lr = float(setup_entropy_lr)
        self.setup_entropy_max_coef = float(setup_entropy_max_coef)
        self.setup_ent_coef = 0.0
        #: --setup-mc-credit: the learner's setup placements are credited with the GAME RESULT
        #: (Monte Carlo, lambda = 1 to the end), not the lambda-return. With gamma 1 and lambda
        #: 0.98 a setup placement's GAE weights the result by ~0.98^(decisions to the end), i.e.
        #: <= 0.01, so the setup learns only from the critic's values of the positions just after
        #: it -- and the critic over-rates unfamiliar openings 3-4x
        #: (research/log/E5_11_setup_lock_and_critic_views.md). Ataraxos trains its setup on MC
        #: returns for the same reason. A game outlasts a rollout, so each setup decision waits in
        #: a per-env pending list until its game ends, then joins a ready batch trained with the
        #: PPO clip against the log-prob it was sampled with; advantage = result - V(s), the critic
        #: only as a baseline. Setup rows are dropped from the ordinary surrogate; the entropy
        #: bonus (and --setup-entropy-floor) and the KL to pi_ref still apply to them there.
        self.setup_mc_credit = bool(setup_mc_credit)
        self.setup_mc_coef = float(setup_mc_coef)
        self.setup_mc_min_batch = int(setup_mc_min_batch)
        self._setup_pending: List[List[Tuple[Any, ...]]] = [[] for _ in range(self.num_envs)]
        self._setup_ready: List[Tuple[Any, ...]] = []
        #: P29 bet 2, --aux-ownership / --aux-vp-margin: auxiliary targets from each game's end --
        #: who controls every country (mine / the opponent's / neither, in the mover's frame) and
        #: the final VP margin -- read by extra heads on the trunk (LadderNet.forward_aux). A game
        #: outlasts a rollout, so a sampled fraction of positions (--aux-sample-frac, observations
        #: kept at half precision) waits per env until its game ends, as the setup credit does, and
        #: the labelled batch is trained in its own step once --aux-min-batch are ready.
        self.aux_own_coef = float(aux_own_coef)
        self.aux_vp_coef = float(aux_vp_coef)
        self.aux_sample_frac = float(aux_sample_frac)
        self.aux_min_batch = int(aux_min_batch)
        self.aux_targets = self.aux_own_coef > 0.0 or self.aux_vp_coef > 0.0
        self._aux_pending: List[List[Tuple[Any, int]]] = [[] for _ in range(self.num_envs)]
        self._aux_ready: List[Tuple[Any, np.ndarray, float]] = []
        if self.aux_targets:
            if not getattr(self.active_net, "aux_heads", False):
                raise ValueError("--aux-ownership / --aux-vp-margin need a network built with aux heads")
            self.env.record_final_control = True
        #: P30, --aux-card-events: for every card in the mover's hand, what its event would do on
        #: this board and what its Ops could take (ai.training.card_event_targets), predicted by a
        #: per-card head on the trunk (LadderNet.forward_card_aux). Labelled by the engine at the
        #: moment of the decision, so a sampled position (--aux-card-sample-frac) is ready at once.
        #: Labelling costs engine calls in Python, so few positions are labelled; they go into a
        #: FIFO buffer (--aux-card-buffer) and, once --aux-card-min-batch are in it, every iteration
        #: takes --aux-card-steps optimiser steps on minibatches of --aux-card-batch drawn from it.
        #: Reusing the buffer is the point: P29 bet 2 trained each labelled batch once, a few hundred
        #: steps in a run, and its head never reached the trunk.
        self.aux_card_coef = float(aux_card_coef)
        self.aux_card_sample_frac = float(aux_card_sample_frac)
        self.aux_card_min_batch = int(aux_card_min_batch)
        self.aux_card_buffer = int(aux_card_buffer)
        self.aux_card_steps = int(aux_card_steps)
        self.aux_card_batch = int(aux_card_batch)
        self._card_ready: List[Tuple[Any, np.ndarray, np.ndarray, np.ndarray]] = []
        self._card_labelled = 0
        self._card_label_seconds = 0.0
        if self.aux_card_coef > 0.0 and not getattr(self.active_net, "card_aux", False):
            raise ValueError("--aux-card-events needs a network built with the card-event head")
        # P31 1d: the opponent's legality at its next decision (see _opp_legal_record).
        self.aux_opp_legality = float(aux_opp_legality)
        self.aux_opp_legality_frac = float(aux_opp_legality_frac)
        self.aux_opp_legality_min_batch = int(aux_opp_legality_min_batch)
        if self.aux_opp_legality > 0.0 and not getattr(self.active_net, "opp_legal_aux", False):
            raise ValueError("--aux-opp-legality needs a network built with the opponent-legality head")
        if not 0.0 < self.aux_opp_legality_frac <= 1.0:
            raise ValueError("--aux-opp-legality-frac must be in (0, 1]")
        #: Per env, per side (0 = US, 1 = USSR): sampled positions waiting for the other side's next
        #: decision, as (observation fp16, that position's own can_opp flags).
        self._ol_pending: List[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = [
            [[], []] for _ in range(self.num_envs)]
        self._ol_npend = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._ol_ready: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # P31 1c: counterfactual mode credit (ai/training/mode_cf.py). Values per option are stored
        # flat over the buffer, [T * N, A], and read in the update by each minibatch's flat index.
        self.mode_cf_coef = float(mode_cf_coef)
        self.mode_cf_from = int(mode_cf_from)
        self.mode_cf: Optional[ModeCounterfactual] = None
        self._cf_adv: Optional[torch.Tensor] = None
        self._cf_has: Optional[torch.Tensor] = None
        self._cf_stats: Dict[str, float] = {}
        if self.mode_cf_coef > 0.0:
            if self.merged_influence:
                raise ValueError("--mode-cf-coef with --merged-influence is not supported: the "
                                 "playouts step E4 actions")
            self.mode_cf = ModeCounterfactual(int(mode_cf_subsample), int(mode_cf_playouts),
                                              auto_advance=bool(getattr(self.env, "_auto_advance", True)),
                                              obs_features=int(model_obs_features(self.active_net)))
            print(f"[mode cf] coef {self.mode_cf_coef:g}: every option of 1 in {int(mode_cf_subsample)} "
                  f"learner floor decisions played out ({int(mode_cf_playouts)} game(s) each), "
                  f"from {self.mode_cf_from:,} steps", flush=True)
        # P32 B4' (ai/training/turn_credit.py): sampled learner decisions priced to the end of the
        # turn at each rollout's end; their improved targets trained on after each PPO update.
        self.turn_credit: Optional[TurnCredit] = None
        if turn_credit_coef > 0.0:
            if self.merged_influence:
                raise ValueError("--turn-credit-coef with --merged-influence is not supported: the "
                                 "pricing steps E4 actions")
            self.turn_credit = TurnCredit(
                turn_credit_coef, budget=turn_credit_budget, worlds=turn_credit_worlds, beta=turn_credit_beta,
                k_se=turn_credit_k, floor=turn_credit_floor, nested_p=turn_credit_nested_p,
                nested_share=turn_credit_nested_share, narrow_share=turn_credit_narrow_share,
                cap=turn_credit_cap, buffer=turn_credit_buffer, steps=turn_credit_steps,
                batch=turn_credit_batch, obs_features=int(model_obs_features(self.active_net)),
                workers=turn_credit_workers)
            print(f"[turn credit] coef {turn_credit_coef:g}: {turn_credit_budget} learner decisions per rollout "
                  f"(2-{turn_credit_cap} options) priced over {turn_credit_worlds} worlds to the turn's end, "
                  f"beta {turn_credit_beta:g}, nested p {turn_credit_nested_p:g}, {turn_credit_steps} x "
                  f"{turn_credit_batch} CE steps per update"
                  + (f", priced by {turn_credit_workers} worker process(es) beside the next rollout"
                     if turn_credit_workers > 0 else ""), flush=True)
        #: P26: torch.compile for the update's forwards only (the learner's per-minibatch pass and
        #: the pi_ref log-probs). The rollout keeps its CUDA graphs of the eager network, so dynamo
        #: never runs inside a rollout; parameters are shared, so checkpoints are the eager
        #: module's. Compilation reorders arithmetic: off by default until an A/B passes.
        #: "max-autotune" is inductor's max-autotune-no-cudagraphs -- the cudagraph modes crashed
        #: with an illegal memory access beside this trainer's own graphs.
        if compile_update not in COMPILE_UPDATE_MODES:
            raise ValueError(f"compile_update must be one of {sorted(COMPILE_UPDATE_MODES)}, "
                             f"got {compile_update!r}")
        self.compile_update = str(compile_update)
        #: z-loss (PaLM): coef * mean(logsumexp(logits)^2) on the update's policy logits. The
        #: softmax is invariant to a common shift of the logits, so nothing else bounds their level,
        #: and it drifts upward without limit: ~350 by 240M in fp32, 2,300 median / 39,000 max by
        #: 590M in E4-57-44, and both 800M runs diverged through it (research/log/E4_long_runs.md).
        #: The penalty pulls the log-normaliser toward 0. 0 is off and leaves the update exactly as
        #: it was; the log-normaliser is logged either way (logit_lse_mean / _absmax).
        if z_loss_coef < 0.0:
            raise ValueError(f"z_loss_coef must be >= 0, got {z_loss_coef}")
        self.z_loss_coef = float(z_loss_coef)
        # A penalty on how far a legal move's logit falls below the top legal move, beyond
        # `logit_gap` (owner, 2026-10-09). A real loss term: its gradient is what it does. A
        # saturated softmax -- a losing coup led its alternatives by 28 nats, P = 1 in float32 --
        # gives policy gradient nothing to work with (it scales with 1 - P); every logit cap tried
        # either kept that (the tanh's own gradient vanishes there) or drifted the network (any
        # gradient past the cap is a signal with no effect on play, and it reaches the trunk).
        # Bounded gaps keep every move learnable at ~e^-gap.
        if logit_gap_coef < 0.0:
            raise ValueError(f"logit_gap_coef must be >= 0, got {logit_gap_coef}")
        self.logit_gap_coef = float(logit_gap_coef)
        self.logit_gap = float(logit_gap)
        self._compiled_nets: Dict[int, nn.Module] = {}
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
        #: "shift" (default, E4-41) or "jump": see wolf_seat_weights.
        self.wolf_dead_zone_mode = str(wolf_dead_zone_mode)
        wolf_seat_weights(0.5, 1.0, 0.0, self.wolf_dead_zone_mode)   # validates the mode
        #: Rollout forwards replayed as CUDA graphs (ai/training/graphed_forward.py): bitwise the
        #: same kernels as eager, one launch instead of ~317. The learner's and the pool
        #: opponent's graphs replay one after the other on the current stream -- see
        #: _graphed_forward for why never concurrently. CUDA only; --no-cuda-graphs is eager.
        self._graphs: Optional[GraphCache] = None
        if cuda_graphs and self.device.type == "cuda":
            self._graphs = GraphCache(self.num_envs, self.buffer.obs_dim,
                                      ActionEncoder.FLAT_ACTION_SIZE, self.device)
            # GAE's backward recursion too: ~5,000 elementwise launches per rollout, same kernels.
            self.buffer.graph_gae = True
        self.buffer.setup_block_lambda = bool(setup_block_lambda)
        self.buffer.block_lambda = str(block_lambda)
        self.buffer.block_mode()          # validates
        #: A2 needs each env's RNG state at every decision; read only when it is used.
        self._record_rngs = self.buffer.block_mode() == "same-side"
        #: The USSR's smoothed self-play win share. Starts even, and is carried in the resume
        #: state so a resumed run does not relearn it.
        self.wolf_sp_ussr = 0.5

        # Rollout temperature bands, one per quarter of the environments. The default samples at
        # the policy (DEFAULT_ROLLOUT_TEMPS); the pre-2026-09-27 bands, all below 1.0, sharpened
        # every environment and made the policy's argmax blunder universal (LEGACY_ROLLOUT_TEMPS).
        _bands = list(rollout_temps) if rollout_temps else list(DEFAULT_ROLLOUT_TEMPS)
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
            print(f"[rollout temps] bands {_bands} (default {list(DEFAULT_ROLLOUT_TEMPS)}; before "
                  f"2026-09-27 {list(LEGACY_ROLLOUT_TEMPS)})", flush=True)
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

    def _update_net(self, net: nn.Module) -> nn.Module:
        """`net` as the update calls it: compiled under --compile-update, else itself. Compiled
        lazily and keyed by identity, so a network swapped in later is compiled on its own."""
        if self.compile_update == "off":
            return net
        c = self._compiled_nets.get(id(net))
        if c is None or getattr(c, "_orig_mod", None) is not net:
            c = cast(nn.Module, torch.compile(net, mode=COMPILE_UPDATE_MODES[self.compile_update]))
            self._compiled_nets[id(net)] = c
        return c

    def update_reference_policy(self) -> None:
        """Updates the frozen reference anchor: π_ref^(k+1) ← π_θ^(k)."""
        self.reference_net.load_state_dict(self.active_net.state_dict())
        self.reference_net.to(self.device)
        self.reference_net.eval()
        if self._sg is not None:
            self._sg.refresh(self.reference_net)
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

    def _update_entropy_ceiling(self, metrics: Dict[str, Any]) -> None:
        """3k: move each seat's entropy coefficient toward keeping its rollout entropy at or below
        the ceiling. Nothing moves before the grace period, when entropy is still near uniform."""
        if self.total_env_steps < self.entropy_ceiling_grace_steps:
            return
        for code, tag in ((1, "us"), (-1, "ussr")):
            h = metrics.get(f"entropy_{tag}")
            if h is None:
                continue
            c = self.ent_coef_seat[code] - self.entropy_ceiling_lr * (float(h) - self.entropy_ceiling)
            self.ent_coef_seat[code] = min(self.ent_coef, max(self.entropy_ceiling_min_coef, c))

    def _setup_mc_record(self, obs_t: torch.Tensor, masks_t: torch.Tensor, actions_t: torch.Tensor,
                         log_probs_t: torch.Tensor, v_win_t: torch.Tensor, learner_np: np.ndarray,
                         dp: np.ndarray) -> None:
        """--setup-mc-credit: file this step's learner setup placements under their env."""
        setup = np.asarray(self._obs_np)[:, setup_phase_slot()] < (0.5 / 6.0)
        rows = np.flatnonzero(setup & learner_np)
        if rows.size == 0:
            return
        idx = torch.from_numpy(rows).to(obs_t.device)
        o, m = obs_t.index_select(0, idx), masks_t.index_select(0, idx)
        a, lp = actions_t.index_select(0, idx), log_probs_t.index_select(0, idx)
        v = v_win_t.reshape(-1).index_select(0, idx)
        for j, i in enumerate(rows):
            self._setup_pending[int(i)].append((o[j], m[j], a[j], lp[j].detach(), v[j].detach(), int(dp[i])))

    def _setup_mc_resolve(self, completed_episodes: Sequence[Dict[str, Any]]) -> None:
        """--setup-mc-credit: a finished game's setup placements get its result, seen by their mover."""
        for ep in completed_episodes:
            i = int(ep.get("env_idx", -1))
            if not 0 <= i < self.num_envs or not self._setup_pending[i]:
                continue
            util = float(ep.get("terminal_utility", 0.0))            # +1 US win, -1 USSR win
            for rec in self._setup_pending[i]:
                self._setup_ready.append(rec + (util * (1.0 if rec[5] == 1 else -1.0),))
            self._setup_pending[i] = []

    def _aux_record(self, obs_t: torch.Tensor, dp: np.ndarray) -> None:
        """Aux targets: file a sampled fraction of this step's decisions under their env."""
        pick = (dp != 0) & (np.random.random(dp.shape[0]) < self.aux_sample_frac)
        rows = np.flatnonzero(pick)
        if rows.size == 0:
            return
        o = obs_t.index_select(0, torch.from_numpy(rows).to(obs_t.device)).half()
        for j, i in enumerate(rows):
            self._aux_pending[int(i)].append((o[j], int(dp[i])))

    def _aux_resolve(self, completed_episodes: Sequence[Dict[str, Any]]) -> None:
        """Aux targets: a finished game labels its waiting positions, each in its mover's frame."""
        for ep in completed_episodes:
            i = int(ep.get("env_idx", -1))
            if not 0 <= i < self.num_envs or not self._aux_pending[i]:
                continue
            ctrl = ep.get("final_control")
            if ctrl is None:                         # not recorded: nothing to label them with
                self._aux_pending[i] = []
                continue
            vp = float(ep.get("victory_points", 0)) / 20.0
            for o, mover in self._aux_pending[i]:
                own = np.asarray(ctrl, dtype=np.int8) * np.int8(mover)     # +1 mine, -1 theirs, 0 neither
                cls = np.where(own == 1, 0, np.where(own == -1, 1, 2)).astype(np.int64)
                self._aux_ready.append((o, cls, vp * float(mover)))
            self._aux_pending[i] = []

    def _card_aux_record(self, obs_t: torch.Tensor, dp: np.ndarray) -> None:
        """Card-event target: label a sampled fraction of this step's decisions with the engine."""
        import ts_engine as ts
        from ai.training.card_event_targets import aux_label
        rows = np.flatnonzero((dp != 0) & (np.random.random(dp.shape[0]) < self.aux_card_sample_frac))
        if rows.size == 0:
            return
        t0 = time.perf_counter()
        o = obs_t.index_select(0, torch.from_numpy(rows).to(obs_t.device))
        o_np = o.float().cpu().numpy()
        o16 = o.half()
        for j, i in enumerate(rows):
            st = self.env.runner.get_state(int(i))
            lab = aux_label(st, ts.Player(int(dp[i])), o_np[j], int(np.random.randint(1, 1 << 30)))
            if lab is not None:
                self._card_ready.append((o16[j], *lab))
                self._card_labelled += 1
        if len(self._card_ready) > self.aux_card_buffer:
            del self._card_ready[:len(self._card_ready) - self.aux_card_buffer]
        self._card_label_seconds += time.perf_counter() - t0

    #: Board slots (26 per country, in the mover's frame): the mover's own legality and its
    #: opponent's, as the engine computes them (engine/src/observation.cpp).
    _CAN_MY_PLACE, _CAN_OPP_PLACE, _CAN_MY_COUP, _CAN_OPP_COUP = 19, 20, 21, 22
    _OL_READY_CAP = 65536

    def _opp_legal_record(self, obs_t: torch.Tensor, dp: np.ndarray) -> None:
        """P31 1d. A sampled position's label is its opponent's legality at the opponent's NEXT
        decision in that game -- read from the opponent's own observation then (its can_my_place /
        can_my_coup), after whatever this side played. So each decision first answers the other
        side's pending positions in its env, then (sampled) joins its own side's."""
        board = obs_t[:, :84 * 26].reshape(-1, 84, 26)
        own = torch.stack([board[:, :, self._CAN_MY_PLACE], board[:, :, self._CAN_MY_COUP]], -1) > 0.5
        side = np.where(dp == 1, 0, 1)
        movers = dp != 0
        answer = np.flatnonzero(movers & (self._ol_npend[np.arange(dp.shape[0]), 1 - side] > 0))
        for e in answer:
            e = int(e)
            other = 1 - int(side[e])
            for o16, cp in self._ol_pending[e][other]:
                self._ol_ready.append((o16, own[e], cp))
            self._ol_pending[e][other] = []
            self._ol_npend[e, other] = 0
        rows = np.flatnonzero(movers & (np.random.random(dp.shape[0]) < self.aux_opp_legality_frac))
        if rows.size:
            idx = torch.from_numpy(rows).to(obs_t.device)
            o16 = obs_t.index_select(0, idx).half()
            cp = torch.stack([board[:, :, self._CAN_OPP_PLACE], board[:, :, self._CAN_OPP_COUP]],
                             -1).index_select(0, idx) > 0.5
            for j, e in enumerate(rows):
                k = int(side[e])
                self._ol_pending[int(e)][k].append((o16[j], cp[j]))
                self._ol_npend[int(e), k] += 1
        if len(self._ol_ready) > self._OL_READY_CAP:
            del self._ol_ready[:len(self._ol_ready) - self._OL_READY_CAP]

    def _opp_legal_game_over(self, envs: np.ndarray) -> None:
        """P31 1d: a game ended; its pending positions were never answered and get no label."""
        for e in np.asarray(envs, dtype=np.int64):
            if self._ol_npend[int(e)].sum() > 0:
                self._ol_pending[int(e)] = [[], []]
                self._ol_npend[int(e)] = 0

    def _opp_legal_update(self) -> Dict[str, float]:
        """P31 1d: one optimiser step on the answered positions, BCE on the 84 x 2 flags.
        `opp_legal_copy_acc` is how often the position's own can_opp flags were already the answer
        and `opp_legal_changed_frac` how often they were not -- the part a head cannot copy."""
        out: Dict[str, float] = {"opp_legal_ready": float(len(self._ol_ready)),
                                 "opp_legal_pending": float(self._ol_npend.sum())}
        n = len(self._ol_ready)
        if n < max(1, self.aux_opp_legality_min_batch):
            return out
        recs, self._ol_ready = self._ol_ready, []
        obs = torch.stack([r[0] for r in recs]).float()
        y = torch.stack([r[1] for r in recs]).float()
        cp = torch.stack([r[2] for r in recs]).float()
        net = cast(Any, self.active_net)
        net.train()
        self.optimizer.zero_grad(set_to_none=True)
        bce_sum, correct = 0.0, 0.0
        for k in range(0, n, self.batch_size):
            sl = slice(k, k + self.batch_size)
            logits = net.forward_opp_legal(obs[sl]).float()
            bce = F.binary_cross_entropy_with_logits(logits, y[sl], reduction="sum") / 168.0
            (self.aux_opp_legality * bce / n).backward()
            with torch.no_grad():
                bce_sum += float(bce)
                correct += float(((logits > 0).float() == y[sl]).float().sum()) / 168.0
        nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        net.eval()
        with torch.no_grad():
            changed = float((cp != y).float().mean())
        out.update({"opp_legal_n": float(n), "opp_legal_loss": bce_sum / n,
                    "opp_legal_acc": correct / n, "opp_legal_copy_acc": 1.0 - changed,
                    "opp_legal_changed_frac": changed})
        return out

    def _card_aux_update(self) -> Dict[str, float]:
        """Card-event target: --aux-card-steps optimiser steps, each on a minibatch drawn from the
        buffer; masked MSE on the standardised targets at the held cards."""
        out: Dict[str, float] = {"card_aux_ready": float(len(self._card_ready)),
                                 "card_aux_labelled": float(self._card_labelled),
                                 "card_aux_label_s": float(self._card_label_seconds)}
        self._card_label_seconds = 0.0
        n = len(self._card_ready)
        if n < max(1, self.aux_card_min_batch) or self.aux_card_steps <= 0:
            return out
        net = cast(Any, self.active_net)
        net.train()
        se_sum, lab_sum = 0.0, 0.0
        for _ in range(self.aux_card_steps):
            pick = np.random.randint(0, n, size=min(self.aux_card_batch, n))
            recs = [self._card_ready[int(k)] for k in pick]
            dev = recs[0][0].device
            obs = torch.stack([r[0] for r in recs]).float()
            cards = torch.from_numpy(np.stack([r[1] for r in recs]).astype(np.int64)).to(dev)
            y = torch.from_numpy(np.stack([r[2] for r in recs])).to(dev)
            m = torch.from_numpy(np.stack([r[3] for r in recs])).to(dev)
            pred = net.forward_card_aux(obs).float()                           # (b, 110, D)
            idx = (cards - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, pred.shape[-1])
            g = torch.gather(pred, 1, idx)                                     # (b, 12, D)
            n_lab = float(m.sum())
            se = ((g - y) ** 2 * m).sum()
            self.optimizer.zero_grad(set_to_none=True)
            (self.aux_card_coef * se / max(1.0, n_lab)).backward()
            nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            se_sum += float(se.detach())
            lab_sum += n_lab
        net.eval()
        # the targets are standardised, so 1 - mse is the explained share of their variance
        out.update({"card_aux_n": float(self.aux_card_steps * min(self.aux_card_batch, n)),
                    "card_aux_loss": se_sum / max(1.0, lab_sum), "card_aux_r2": 1.0 - se_sum / max(1.0, lab_sum)})
        return out

    def _aux_update(self) -> Dict[str, float]:
        """Aux targets: one step on the labelled positions, ownership CE plus VP-margin MSE."""
        out: Dict[str, float] = {"aux_ready": float(len(self._aux_ready)),
                                 "aux_pending": float(sum(len(p) for p in self._aux_pending))}
        n = len(self._aux_ready)
        if n < max(1, self.aux_min_batch):
            return out
        recs, self._aux_ready = self._aux_ready, []
        dev = recs[0][0].device
        obs = torch.stack([r[0] for r in recs]).float()
        cls = torch.from_numpy(np.stack([r[1] for r in recs])).to(dev)
        vp = torch.tensor([r[2] for r in recs], dtype=torch.float32, device=dev)
        net = cast(Any, self.active_net)
        net.train()
        self.optimizer.zero_grad(set_to_none=True)
        own_loss_sum = 0.0
        vp_loss_sum = 0.0
        correct = 0.0
        for k in range(0, n, self.batch_size):
            sl = slice(k, k + self.batch_size)
            own_logits, vp_pred = net.forward_aux(obs[sl])
            own_logits = own_logits.float()
            own_ce = F.cross_entropy(own_logits.reshape(-1, 3), cls[sl].reshape(-1), reduction="sum") / 84.0
            vp_se = F.mse_loss(vp_pred.float().reshape(-1), vp[sl], reduction="sum")
            loss = (self.aux_own_coef * own_ce + self.aux_vp_coef * vp_se) / n
            loss.backward()
            with torch.no_grad():
                own_loss_sum += float(own_ce)
                vp_loss_sum += float(vp_se)
                correct += float((own_logits.argmax(-1) == cls[sl]).float().sum()) / 84.0
        nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        net.eval()
        out.update({"aux_n": float(n), "aux_own_loss": own_loss_sum / n, "aux_own_acc": correct / n,
                    "aux_vp_loss": vp_loss_sum / n})
        return out

    def _setup_mc_update(self) -> Dict[str, float]:
        """--setup-mc-credit: one clipped policy step on the setup placements whose games ended."""
        out: Dict[str, float] = {"setup_mc_ready": float(len(self._setup_ready)),
                                 "setup_mc_pending": float(sum(len(p) for p in self._setup_pending))}
        n = len(self._setup_ready)
        if n < max(1, self.setup_mc_min_batch):
            return out
        recs, self._setup_ready = self._setup_ready, []
        obs = torch.stack([r[0] for r in recs])
        masks = torch.stack([r[1] for r in recs])
        act = torch.stack([r[2] for r in recs]).long().reshape(-1)
        old_lp = torch.stack([r[3] for r in recs]).float().reshape(-1)
        v = torch.stack([r[4] for r in recs]).float().reshape(-1)
        g = torch.tensor([r[6] for r in recs], dtype=torch.float32, device=obs.device)
        adv = g - v
        if self.setup_script_frac > 0.0:
            # Centred per side. The credit result - V averaged +0.03..+0.18 in every update of
            # E7-10-44 and E7-11-44 (the learner beats its pool ~71% of the time and the critic
            # cannot see the opponent). A uniform positive credit barely moves the policy's own
            # placements (p ~ 1, gradient ~ 0) but pushes every scripted one up at full strength --
            # E7-11-44 adopted a -12.5 pp USSR opening that way. Centred, a scripted opening rises
            # only where its games beat the average of that side's placements in the batch.
            side = torch.tensor([int(r[5]) for r in recs], device=obs.device)
            for _code in (1, -1):
                _sel = side == _code
                if bool(_sel.any()):
                    adv[_sel] = adv[_sel] - adv[_sel].mean()
            out["setup_mc_adv_centre_us"] = float((g - v)[side == 1].mean()) if bool((side == 1).any()) else 0.0
            out["setup_mc_adv_centre_ussr"] = float((g - v)[side == -1].mean()) if bool((side == -1).any()) else 0.0
            # The rollout log-prob is stale by the time a game ends -- several PPO updates later.
            # For the policy's own placements (p ~ 1) that hardly matters; for a scripted one at
            # p ~ 1e-8 a few nats of drift puts the ratio far outside the clip, which then passes
            # the gradient of a LOST game at full weight x ratio and blocks that of a won one: the
            # update drove the scripted openings down and the setup entropy to 0 (E7-10-44's first
            # launch, voided). Measured from the current policy instead, every ratio starts at 1 and
            # the single step below is a plain result-minus-baseline policy gradient.
            with torch.no_grad():
                old_lp = torch.cat([
                    F.log_softmax(self.active_net(obs[i:i + self.batch_size], masks[i:i + self.batch_size])[0].float(),
                                  dim=-1).gather(1, act[i:i + self.batch_size].unsqueeze(1)).squeeze(1)
                    for i in range(0, n, self.batch_size)])
        # Scale only: the critic baseline already centres it, and a state-only baseline cannot
        # bias the gradient however wrong the critic is.
        adv_n = adv / adv.std().clamp(min=0.1) if n > 1 else adv
        self.active_net.train()
        self.optimizer.zero_grad(set_to_none=True)
        clip_n = 0.0
        ratio_dev = 0.0
        for i in range(0, n, self.batch_size):
            sl = slice(i, i + self.batch_size)
            logits = self.active_net(obs[sl], masks[sl])[0].float()
            lp = F.log_softmax(logits, dim=-1).gather(1, act[sl].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(torch.clamp(lp - old_lp[sl], -20.0, 20.0))
            a = adv_n[sl]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a)
            loss = -self.setup_mc_coef * surr.sum() / n
            loss.backward()
            with torch.no_grad():
                clip_n += float(((ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)).float().sum())
                ratio_dev += float((ratio - 1.0).abs().sum())
        nn.utils.clip_grad_norm_(self.active_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.active_net.eval()
        if self.setup_script_frac > 0.0:
            out["setup_script_lp_mean"] = float(old_lp.mean())
        out.update({"setup_mc_n": float(n), "setup_mc_result_mean": float(g.mean()),
                    "setup_mc_adv_mean": float(adv.mean()), "setup_mc_adv_std": float(adv.std()) if n > 1 else 0.0,
                    "setup_mc_clip_frac": clip_n / n, "setup_mc_ratio_dev": ratio_dev / n})
        return out

    def _update_setup_entropy(self, h: float) -> None:
        """--setup-entropy-floor: move the setup bonus toward holding setup entropy at the floor."""
        c = self.setup_ent_coef + self.setup_entropy_lr * (self.setup_entropy_floor - float(h))
        self.setup_ent_coef = min(self.setup_entropy_max_coef, max(0.0, c))

    def _graphed_forward(self, obs_t: torch.Tensor, masks_t: torch.Tensor,
                         learner_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rollout forward as CUDA-graph replays: the learner over every env, and in a mixed
        batch the pool opponent over every env too; the opponent's rows are then copied into the
        learner's logits. Values always come from the learner (see the eager branch in
        collect_rollouts for why).

        The two replays run one after the other on the current stream, NEVER concurrently on two
        streams. A captured graph bakes in the cuBLAS workspace of the stream it was captured on,
        and PyTorch hands out streams from a pool of 32 per device, reused round-robin. After ~16
        captures (two streams each; one capture per new pool member, i.e. per snapshot) an
        opponent's graph can share the learner's workspace, and replayed concurrently their split-K
        kernels, which coordinate through counters in that workspace, deadlocked the GPU: E4-42-03,
        E4-42-05 and E4-44-05 all hung here, in the first sync after the replays, with the GPU at
        100% and low power (research/log/training_throughput_cpu.md). Serialised replays cannot
        race whatever workspace they share.
        """
        assert self._graphs is not None
        g_l = self._graphs.get(self.active_net)
        g_l.load(obs_t, masks_t)
        mixed = self.opponent_pool is not None and not bool(learner_np.all())
        if not mixed:
            g_l.replay()
            logits, v_win_t, v_vp_t = g_l.outputs()
            return logits, v_win_t, v_vp_t
        assert self.opponent_pool is not None
        g_o = self._graphs.get(self.opponent_pool.current)
        g_l.replay()
        g_o.load(obs_t, masks_t)
        g_o.replay()
        logits, v_win_t, v_vp_t = g_l.outputs()
        logits = logits.float()
        opp_idx = torch.from_numpy(np.flatnonzero(~learner_np)).to(self.device)
        logits.index_copy_(0, opp_idx, g_o.static_out[0].index_select(0, opp_idx).float())
        return logits, v_win_t, v_vp_t

    def collect_rollouts(self) -> Dict[str, Any]:
        """Collects on-policy rollouts across all parallel environments."""
        if self.opponent_pool is not None and self.opponent_pool.obs_features != self.obs_features:
            raise ValueError("opponent pool observation features do not match the learner")
        t0 = time.time()
        if self._search_pool is not None:
            self._search_pool.begin()
        self.active_net.eval()
        if self._graphs is not None:
            self._graphs.begin_rollout()
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
        if self.active_search is not None:
            if self.active_search.maybe_refresh(self.active_net, int(self.total_env_steps)):
                print(f"[active search] the searched snapshot refreshed from the learner at "
                      f"{self.total_env_steps / 1e6:.0f}M", flush=True)
            self.active_search.begin_rollout(self.buffer)
        if self._behaviour_w is not None:
            self._behaviour_w.fill_(1.0)
        if self.opening_seeder is not None and not self._openings_started:
            self.opening_seeder.draw(np.arange(self.num_envs))
            self._openings_started = True
        if self.seeder is not None and not self._seed_started:
            # The first rollout's games are drawn here rather than in __init__, because a resumed
            # run only knows its step count (and so whether seeding has started) after loading.
            self.seeder.draw(np.arange(self.num_envs), self.total_env_steps)
            self._seed_started = True
        seat_entropy_sum = {1: torch.zeros((), device=self.device), -1: torch.zeros((), device=self.device)}
        seat_entropy_n = {1: torch.zeros((), device=self.device), -1: torch.zeros((), device=self.device)}
        setup_entropy_sum = torch.zeros((), device=self.device)
        setup_entropy_n = torch.zeros((), device=self.device)
        pm_n = torch.zeros((), device=self.device)
        pm_event = torch.zeros((), device=self.device)
        _setup_slot = setup_phase_slot()
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
                # A play-mode decision: every legal action is a play-mode slot (event, space, Ops),
                # and event or space is among them -- the event/Ops choice for a chosen card.
                _pm = play_mode_rows(masks_t)
                _pm_learner = _pm & learner_t
                if self.temperature_schedule and self.num_envs > 1:
                    row_temps = self.env_temps
                else:
                    row_temps = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
                if self.play_mode_temp != 1.0:
                    row_temps = torch.where(_pm_learner.unsqueeze(1), row_temps * self.play_mode_temp, row_temps)
                if (self.temperature_schedule and self.num_envs > 1) or self.play_mode_temp != 1.0:
                    actions_t = torch.multinomial(F.softmax(logits / row_temps, dim=-1), 1).squeeze(1)
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
                if self.opponent_temperature is not None and not bool(learner_np.all()):
                    actions_t = opponent_actions(logits, actions_t, ~learner_t,
                                                 self.opponent_temperature)

                if self.setup_script_frac > 0.0:
                    # After every other override, so a scripted game is scripted whoever moves;
                    # before the log-prob, so the stored one is the policy's own for that action.
                    self._apply_setup_script(actions_t, masks_t, _dp)

                unscaled_log_probs = F.log_softmax(logits, dim=-1)
                # P31 1a: the floor, on the learner's own floor rows. Off (eps 0), nothing here
                # draws from the RNG, so a run without the flag is bit-identical to before.
                _eps = self._current_floor_eps()
                if _eps > 0.0:
                    _fr = floor_rows(masks_t, self.floor_rows_kind) & learner_t
                    if self.floor_scope == "seeded":
                        assert self.seeder is not None
                        _fr = _fr & torch.from_numpy(self.seeder.seeded).to(self.device)
                    actions_t, log_probs_t, _bw, _took = apply_floor(actions_t, unscaled_log_probs,
                                                                     masks_t, _fr, _eps)
                    if self._behaviour_w is None:
                        self._behaviour_w = torch.ones(self.buffer_size * self.num_envs,
                                                       dtype=torch.float32, device=self.device)
                    _s = self.buffer.step * self.num_envs
                    self._behaviour_w[_s:_s + self.num_envs] = _bw
                    self._floor_stats += torch.stack(
                        [_fr.sum(), _took.sum(), learner_t.sum()]).to(torch.float64)
                else:
                    log_probs_t = unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                # P31 1b: the forced precursor plays, after every other override. They are the
                # environment's: learner = 0 from here on (no policy gradient, not in the learner's
                # statistics), and the stored log-prob is the policy's own for the forced action.
                if self.seeder is not None:
                    _forced = self.seeder.apply(actions_t, np.asarray(self._masks_np), _dp,
                                                self.env.runner, bool(self.env._auto_advance))
                    if _forced.any():
                        learner_np = np.asarray(learner_np, dtype=bool) & ~_forced
                        learner_t = torch.from_numpy(learner_np).to(self.device)
                        _pm_learner = _pm & learner_t
                        _forced_t = torch.from_numpy(_forced).to(self.device)
                        log_probs_t = torch.where(
                            _forced_t,
                            unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1),
                            log_probs_t)
                if self.opening_seeder is not None:
                    _fo = self.opening_seeder.apply(actions_t, np.asarray(self._masks_np), _dp, self.env.runner)
                    if _fo.any():
                        learner_np = np.asarray(learner_np, dtype=bool) & ~_fo
                        learner_t = torch.from_numpy(learner_np).to(self.device)
                        _pm_learner = _pm & learner_t
                        _fo_t = torch.from_numpy(_fo).to(self.device)
                        log_probs_t = torch.where(
                            _fo_t, unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1), log_probs_t)
                if self.event_forcer is not None:
                    # Applicable events forced as the policy's OWN play: learner stays 1, the stored
                    # log-prob is log pi of the event, and a floor weight on the row goes back to 1.
                    _ev = self.event_forcer.apply(actions_t, masks_t, np.asarray(learner_np, dtype=bool),
                                                  self.env.runner, int(self.total_env_steps))
                    if _ev.any():
                        _ev_t = torch.from_numpy(_ev).to(self.device)
                        log_probs_t = torch.where(
                            _ev_t, unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1),
                            log_probs_t)
                        if self.force_event_credit == "environment":
                            learner_np = np.asarray(learner_np, dtype=bool) & ~_ev
                            learner_t = torch.from_numpy(learner_np).to(self.device)
                            _pm_learner = _pm & learner_t
                        if self._behaviour_w is not None and _eps > 0.0:
                            _s = self.buffer.step * self.num_envs
                            _w = self._behaviour_w[_s:_s + self.num_envs]
                            _w[_ev_t] = 1.0
                if self.active_search is not None and self.active_search.joined:
                    # P36: the root's move at a fraction of the learner's entry decisions, played as
                    # ENVIRONMENT like a forced opening (learner 0, the stored log-prob the net's own of
                    # the move), with the root's improved policy filed as the row's search target.
                    _as = self.active_search.apply(actions_t, np.asarray(learner_np, dtype=bool),
                                                   self.env.runner, self.buffer, self.buffer.step)
                    if _as.any():
                        learner_np = np.asarray(learner_np, dtype=bool) & ~_as
                        learner_t = torch.from_numpy(learner_np).to(self.device)
                        _pm_learner = _pm & learner_t
                        _as_t = torch.from_numpy(_as).to(self.device)
                        log_probs_t = torch.where(
                            _as_t, unscaled_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1), log_probs_t)

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
                # The learner's setup placements (global phase slot exactly 0 in SETUP).
                pm_n += _pm_learner.sum()
                pm_event += (_pm_learner & (actions_t == EVENT_SLOT)).sum()
                _setup_sel = learner_t & (obs_t[:, _setup_slot] < (0.5 / 6.0))
                setup_entropy_sum += (_ent * _setup_sel).sum()
                setup_entropy_n += _setup_sel.sum()

            actions_np = actions_t.cpu().numpy()
            if self.mode_cf is not None and self.total_env_steps >= self.mode_cf_from:
                # P31 1c: the learner's floor decisions, 1 in k, cloned before the step.
                _cand = np.flatnonzero((floor_rows(masks_t) & learner_t).cpu().numpy())
                _pick = self.mode_cf.pick(_cand)
                if _pick.size:
                    self.mode_cf.enqueue(self.buffer.step, _pick, self.env.runner,
                                         np.asarray(self._masks_np), actions_np, _dp)
            if self.turn_credit is not None:
                # B4': cloned before the step, like the search and mode-cf queues.
                self.turn_credit.record(self.env.runner, np.asarray(self._masks_np),
                                        np.asarray(learner_np, dtype=bool))
            if self.setup_mc_credit:
                self._setup_mc_record(obs_t, masks_t, actions_t, log_probs_t, v_win_t,
                                      np.asarray(learner_np, dtype=bool), _dp)
            if self.aux_targets:
                self._aux_record(obs_t, np.asarray(_dp))
            if self.aux_card_coef > 0.0:
                self._card_aux_record(obs_t, np.asarray(_dp))
            if self.aux_opp_legality > 0.0:
                self._opp_legal_record(obs_t, np.asarray(_dp))
            # A2 (--block-lambda same-side): each env's RNG state at this decision, read before the
            # step. A change by the next decision means a chance node came between them.
            _rngs_np: Optional[np.ndarray] = None
            if self._record_rngs:
                _rngs_np = np.array([self.env.runner.get_state(i).rng_state
                                     for i in range(self.num_envs)], dtype=np.uint64).view(np.int64)

            # P15-X4b. MUST happen before the step: `_queue_search_targets` reads the runner's current
            # state, and `buffer.add` below files the answer against `obs_t`, which is s_t. Taken
            # after the step the runner holds s_{t+1} (see the bootstrap note further down), so
            # every target would describe the position *after* the one it is stored with -- a
            # distribution over another state's actions, then masked against s_t's legal set, so
            # the mass lands wherever those indices happen to mean something here. It does not
            # fail loudly: it trains the policy toward noise with a real gradient behind it, which
            # collapsed two arms (entropy 1.03 -> 0.36, critic to chance) before it was found.
            self._queue_search_targets(self.buffer.step)

            if self.predeal_value_coef > 0.0 and not self._predeal_on:
                self.env.runner.set_capture_predeal(True)
                self._predeal_on = True
                print(f"[pre-deal] the critic on both sides at the turn-end park, coef "
                      f"{self.predeal_value_coef}", flush=True)
            next_obs_np, next_masks_np, rewards_np, self._dones_np, self._info = self.env.step(actions_np)
            if self._predeal_on:
                _pf, _pus, _pussr = self.env.runner.get_predeal()
                _pe = np.flatnonzero(np.asarray(_pf))
                if _pe.size:
                    _w = self.buffer.obs_dim
                    self._predeal_rows.append((
                        self.buffer.step, _pe,
                        torch.from_numpy(np.array(np.asarray(_pus)[_pe, :_w])).to(self.device),
                        torch.from_numpy(np.array(np.asarray(_pussr)[_pe, :_w])).to(self.device)))

            if self.setup_explore_frac > 0.0:
                self._ep_decisions += 1
                self._ep_decisions[self._dones_np > 0.5] = 0
            if self.setup_script_frac > 0.0:
                self._script_draw(np.flatnonzero(self._dones_np > 0.5))
            if self.seeder is not None:
                self.seeder.draw(np.flatnonzero(self._dones_np > 0.5), self.total_env_steps)
            if self.opening_seeder is not None:
                self.opening_seeder.draw(np.flatnonzero(self._dones_np > 0.5))
            if self.aux_opp_legality > 0.0:
                self._opp_legal_game_over(np.flatnonzero(self._dones_np > 0.5))

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
                rngs=_rngs_np,
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

            if self.setup_mc_credit and self._info.get("completed_episodes"):
                self._setup_mc_resolve(self._info["completed_episodes"])
            if self.aux_targets and self._info.get("completed_episodes"):
                self._aux_resolve(self._info["completed_episodes"])
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

        self._flush_search_targets()
        if self.mode_cf is not None:
            self._flush_mode_cf()
        if self.turn_credit is not None:
            _was = self.active_net.training
            self.active_net.eval()
            self.turn_credit.flush(self.active_net, self.device)
            self.active_net.train(_was)

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
        if self.search_value_beta > 0.0:
            self._mix_search_values()

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
                                               self.wolf_dead_zone, self.wolf_dead_zone_mode)
            metrics["wolf_sp_ussr"] = self.wolf_sp_ussr
            metrics["wolf_w_us"] = _w_us
            metrics["wolf_w_ussr"] = _w_ussr
        for _code, _tag in ((1, "us"), (-1, "ussr")):
            _n = float(seat_entropy_n[_code].item())
            if _n > 0:
                metrics[f"entropy_{_tag}"] = float(seat_entropy_sum[_code].item()) / _n
        if self.entropy_ceiling > 0.0:
            self._update_entropy_ceiling(metrics)
            metrics["ent_coef_us"] = self.ent_coef_seat[1]
            metrics["ent_coef_ussr"] = self.ent_coef_seat[-1]
        _pmn = float(pm_n.item())
        if _pmn > 0:
            metrics["play_mode_event_frac"] = float(pm_event.item()) / _pmn
        _sn = float(setup_entropy_n.item())
        if _sn > 0:
            metrics["entropy_setup"] = float(setup_entropy_sum.item()) / _sn
            if self.setup_entropy_floor > 0.0:
                self._update_setup_entropy(metrics["entropy_setup"])
        if self.setup_entropy_floor > 0.0:
            metrics["setup_ent_coef"] = self.setup_ent_coef
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
            if self.search_value_beta > 0.0:
                metrics["search_value_rows"] = float(getattr(self, "search_value_rows", 0))
                metrics["search_value_vs_return"] = float(getattr(self, "search_value_vs_return", 0.0))
                metrics["search_value_vs_critic"] = float(getattr(self, "search_value_vs_critic", 0.0))
                metrics["search_value_bias"] = float(getattr(self, "search_value_bias", 0.0))
                metrics["search_value_bias_us"] = float(getattr(self, "search_value_bias_us", 0.0))
                metrics["search_value_bias_ussr"] = float(getattr(self, "search_value_bias_ussr", 0.0))
            metrics["search_dropped_visit_frac"] = float(
                getattr(self, "search_dropped_visit_frac", 0.0))
            metrics["search_dropped_row_frac"] = float(
                getattr(self, "search_dropped_row_frac", 0.0))
        return metrics

    def train_step(self) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement train_step")

    def _flush_mode_cf(self) -> None:
        """P31 1c: price the queued decisions' options and file them against their buffer rows."""
        assert self.mode_cf is not None
        n_rows = self.buffer_size * self.num_envs
        if self._cf_adv is None:
            self._cf_adv = torch.zeros((n_rows, ActionEncoder.FLAT_ACTION_SIZE), dtype=torch.float16,
                                       device=self.device)
            self._cf_has = torch.zeros(n_rows, dtype=torch.bool, device=self.device)
        assert self._cf_has is not None
        self._cf_adv.zero_()
        self._cf_has.zero_()
        t0 = time.perf_counter()
        plies0 = self.mode_cf.plies_played
        was_training = self.active_net.training
        self.active_net.eval()
        results = self.mode_cf.flush(self.active_net, self.device)
        self.active_net.train(was_training)
        spread, worse, n_opt = 0.0, 0, 0
        for q, qv in results:
            taken = np.flatnonzero(q.options == q.taken)
            if taken.size != 1:
                raise RuntimeError(f"the taken action {q.taken} is not among the decision's options")
            adv = qv - qv[taken[0]]
            idx = q.step * self.num_envs + q.env
            self._cf_adv[idx, torch.from_numpy(q.options).to(self.device)] = torch.from_numpy(
                adv).to(self.device, torch.float16)
            self._cf_has[idx] = True
            spread += float(qv.max() - qv.min())
            worse += int(adv.max() > 0)
            n_opt += int(q.options.size)
        n = len(results)
        self._cf_stats = {"mode_cf_n": float(n), "mode_cf_options": n_opt / max(n, 1),
                          "mode_cf_spread": spread / max(n, 1),
                          "mode_cf_taken_beaten_frac": worse / max(n, 1),
                          "mode_cf_plies": float(self.mode_cf.plies_played - plies0),
                          "mode_cf_seconds": time.perf_counter() - t0}

    def _current_floor_eps(self) -> float:
        """P31 1a: the floor's eps at the current step count."""
        return floor_eps(self.play_mode_floor, int(self.total_env_steps), self.floor_from,
                         self.floor_anneal_from, self.floor_anneal_steps)

    def _script_draw(self, envs: np.ndarray) -> None:
        """--setup-script-frac: decide afresh, for each of `envs` (at a game start), whether its
        setup is scripted and by which opening."""
        envs = np.asarray(envs, dtype=np.int64)
        if envs.size == 0:
            return
        pick = self._script_rng.random(envs.size) < self.setup_script_frac
        which = self._script_rng.integers(0, len(self._script_scripts), envs.size)
        self._script_open[envs] = np.where(pick, which, -1)
        self._script_cursor[1][envs] = 0
        self._script_cursor[-1][envs] = 0
        self._script_games += (int(pick.sum()), int((~pick).sum()))

    def _apply_setup_script(self, actions_t: torch.Tensor, masks_t: torch.Tensor, dp: np.ndarray) -> None:
        """--setup-script-frac: overwrite the setup placements of scripted games, for whichever agent
        is moving (a pool opponent's placements are scripted too; only the learner's are trained).
        Setup rows are recognised from the observation's phase slot (exactly 0 in SETUP)."""
        setup = np.asarray(self._obs_np)[:, setup_phase_slot()] < (0.5 / 6.0)
        rows = np.flatnonzero(setup & (self._script_open >= 0) & (dp != 0))
        if rows.size == 0:
            return
        masks_np = masks_t.detach().cpu().numpy()
        for r in rows:
            side = int(dp[r])
            script = self._script_scripts[int(self._script_open[r])][side]
            k = int(self._script_cursor[side][r])
            if k >= len(script):
                raise RuntimeError(f"env {r}: side {side} has no scripted placement {k}")
            idx = script[k]
            if not masks_np[r][idx]:
                raise RuntimeError(f"env {r}: scripted setup placement {k} (flat action {idx}) is "
                                   f"illegal here -- refusing a partly scripted opening")
            actions_t[r] = idx
            self._script_cursor[side][r] = k + 1

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

    def _mix_search_values(self) -> None:
        """P32 T2: blend the search value into v_win's target at the searched decisions, after
        GAE so the advantages -- and the policy loss -- are the control's. Logs how far the search
        value sits from the lambda-return it partly replaces, and from the critic's own estimate."""
        sel = self.buffer.has_search_v > 0.5
        n = int(sel.sum())
        self.search_value_rows = n
        if n == 0:
            return
        sv, ret = self.buffer.search_v[sel], self.buffer.returns_win[sel]
        self.search_value_vs_return = float((sv - ret).abs().mean())
        # Signed: a frozen teacher's search value is its value under its own improved policy, which
        # can sit uniformly above the student's return -- an optimism only the searched rows carry.
        self.search_value_bias = float((sv - ret).mean())
        # Per seat: the US critic is the suspect one (search on R32 scores 47% as US against its
        # own greedy policy), so an optimism would show there first.
        seat = self.buffer.players[sel]
        for name, side in (("us", 1), ("ussr", -1)):
            m = seat == side
            setattr(self, f"search_value_bias_{name}", float((sv - ret)[m].mean()) if bool(m.any()) else 0.0)
        self.search_value_vs_critic = float((sv - self.buffer.values_win[sel]).abs().mean())
        b = self.search_value_beta
        self.buffer.returns_win[sel] = (1.0 - b) * ret + b * sv

    def _queue_search_targets(self, step: int) -> None:
        """Note the decisions this configuration searches at buffer row `step`, for
        `_flush_search_targets` to answer when the rollout ends.

        P15-X4b. **Reads the runner's current state, so it must be called while that state is
        still the one the target will be stored against** -- before `env.step`, not after. Taken
        after the step the runner holds s_{t+1}, so every target would describe the position after
        the one it is filed with; that collapsed two arms (entropy 1.03 -> 0.36, critic to chance)
        before it was found.

        Deferring the search is exact. Search only *answers* -- the actions played were already
        sampled from the raw policy, so the state distribution is the baseline's -- and the weights
        do not change during a rollout, so a position searched at the end gets the answer it would
        have got at its step. What changes is the batch: searching each step's ~25 positions on its
        own ran the network at batch ~25 and featurised every leaf from Python, 8x off the
        unsearched throughput (research/log/search_performance_profile.md). The searcher's random
        draws come in a different order, so a run is not bit-identical to the per-step version.
        """
        if self._searcher is None:
            return
        import ts_engine as ts

        for i in range(self.num_envs):
            st = self.env.runner.get_state(i)
            if not ts.Engine.is_terminal(st) and self._searcher.should_search(st):
                self._search_queue.append((step, i, st.clone()))
        if self._search_pool is not None:
            self._search_pool.feed(self._search_queue)

    def _flush_search_targets(self) -> None:
        """Search every queued decision in one batch and write the visit distributions into the
        buffer rows they were queued against.

        A target is dropped when the searcher returns nothing, and normalised over the visits it
        did return **that are legal in the real state**. The search is DETERMINIZED, and in this
        game the legal SET itself can depend on hidden information (`batched_mcts.py`), so visits
        on actions legal only under the sampled world are removed here, as the acting agent's
        path does. Before that filter existed `search_ce` exceeded 1e5 in a third of iterations:
        the mask fill is -1e9, so a stray visit's mass cost 1e9 per unit.
        """
        if self._searcher is None:
            return
        import numpy as _np

        from bindings.action_encoder import ActionEncoder

        queue, self._search_queue = self._search_queue, []
        # Every row is rewritten each rollout: a row left alone would keep last rollout's target.
        self.buffer.search_pi.zero_()
        self.buffer.has_search.zero_()
        self.buffer.has_search_v.zero_()
        if not queue:
            return
        was_training = self.active_net.training
        self.active_net.eval()
        try:
            res: List[Tuple[List[int], Any]] = []
            root_values: List[float] = []
            sat_n = 0.0
            if self._search_pool is not None:
                self._search_pool.feed(queue, final=True)
                got = self._search_pool.collect()
                res = [got.get(j, ([], _np.zeros(0))) for j in range(len(queue))]
            for s0 in range(0, len(queue) if self._search_pool is None else 0, SEARCH_CHUNK):
                chunk = [st for _t, _i, st in queue[s0:s0 + SEARCH_CHUNK]]
                if self.search_gumbel_k > 0 or self.search_rollout_horizon > 0 or self.search_teacher_policy:
                    res.extend(self._searcher.improved_policies(chunk))
                    # The Gumbel root keeps its values on its root; ContinuationValues (R49) on itself.
                    root = getattr(self._searcher, "_gumbel", self._searcher)
                    sat_n += float(getattr(root, "last_saturated_frac", 0.0)) * len(chunk)
                    root_values.extend(float(x) for x in getattr(root, "last_values", []))
                else:
                    res.extend(self._searcher.run(chunk))
            self.search_saturated_frac = sat_n / max(1, len(queue))
        except Exception as exc:                      # a broken teacher must not kill the run
            print(f"[X4b] search targets unavailable this rollout: {exc}", flush=True)
            return
        finally:
            self.active_net.train(was_training)

        width = int(self.buffer.search_pi.shape[-1])
        t_idx: List[int] = []
        e_idx: List[int] = []
        a_idx: List[int] = []
        vals: List[float] = []
        flag_t: List[int] = []
        flag_e: List[int] = []
        dropped_visits = 0.0
        total_visits = 0.0
        dropped_rows = 0
        for (t, i, st), (acts, visits) in zip(queue, res):
            if not acts:
                continue
            v = _np.asarray(visits, dtype=_np.float32)
            total_visits += float(v.sum())
            # The real state's mask is the only authority on what may be played.
            legal_mask = _np.asarray(ActionEncoder.get_legal_mask(st))
            probs, dropped = filter_search_visits(acts, v, legal_mask, width)
            if dropped > 0.0:
                dropped_visits += dropped
                dropped_rows += 1
            if probs is None:
                # Every visit was illegal here: no usable target rather than a guessed one.
                continue
            for a, p in probs:
                t_idx.append(t)
                e_idx.append(i)
                a_idx.append(int(a))
                vals.append(float(p))
            flag_t.append(t)
            flag_e.append(i)
        if self.search_value_beta > 0.0 and len(root_values) == len(queue):
            sv = [(t, i, x) for (t, i, _st), x in zip(queue, root_values) if _np.isfinite(x)]
            if sv:
                dev = self.buffer.search_v.device
                ti = torch.tensor([a for a, _b, _c in sv], device=dev)
                ei = torch.tensor([b for _a, b, _c in sv], device=dev)
                self.buffer.search_v[ti, ei] = torch.tensor([c for _a, _b, c in sv],
                                                            dtype=self.buffer.search_v.dtype, device=dev)
                self.buffer.has_search_v[ti, ei] = 1.0
        if vals:
            dev = self.buffer.search_pi.device
            self.buffer.search_pi[torch.tensor(t_idx, device=dev), torch.tensor(e_idx, device=dev),
                                  torch.tensor(a_idx, device=dev)] = torch.tensor(
                                      vals, dtype=self.buffer.search_pi.dtype, device=dev)
            self.buffer.has_search[torch.tensor(flag_t, device=dev),
                                   torch.tensor(flag_e, device=dev)] = 1.0
        # The share of the searcher's visits the real mask rejected. (Before 2026-10-04 this was
        # dropped visits per searched row -- up to the simulation count -- despite its name.)
        self.search_dropped_visit_frac = dropped_visits / max(1.0, total_visits)
        self.search_dropped_row_frac = dropped_rows / max(1, len(queue))
        self.search_targets_produced = getattr(self, "search_targets_produced", 0) + len(flag_t)

    def train_iteration(self) -> Dict[str, Any]:
        """Runs one full training iteration (rollout collection + inner SGD epochs + reference check)."""
        self.total_iterations += 1
        if self.phase_lock_path:
            with phase_lock(self.phase_lock_path + ".rollout") as wait_rollout:
                rollout_metrics = self.collect_rollouts()
            with phase_lock(self.phase_lock_path + ".update") as wait_update:
                train_metrics = self.train_step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)   # the update's kernels finish under the lock
            rollout_metrics["phase_lock_wait_rollout"] = wait_rollout
            rollout_metrics["phase_lock_wait_update"] = wait_update
        else:
            rollout_metrics = self.collect_rollouts()
            train_metrics = self.train_step()

        if self.steps_since_ref_update >= self.ref_update_freq:
            self.update_reference_policy()
            self.steps_since_ref_update = 0

        combined: Dict[str, Any] = {**rollout_metrics, **train_metrics}
        if self.setup_mc_credit:
            combined.update(self._setup_mc_update())
        if self.setup_script_frac > 0.0:
            _tot = max(1, int(self._script_games.sum()))
            combined["setup_script_games_frac"] = float(self._script_games[0]) / _tot
        if self.play_mode_floor > 0.0:
            # Per iteration: eps, the share of the learner's decisions the floor covered, and the
            # share of those that took the uniform draw (~eps when it is on).
            _fs = self._floor_stats.tolist()
            self._floor_stats.zero_()
            combined["floor_eps"] = self._current_floor_eps()
            combined["floor_row_frac"] = _fs[0] / max(_fs[2], 1.0)
            combined["floor_draw_frac"] = _fs[1] / max(_fs[0], 1.0)
        if self.mode_cf is not None:
            combined.update(self._cf_stats)
        if self.event_forcer is not None:
            for _name in self.event_forcer.forced:
                combined[f"force_applicable_{_name}"] = float(self.event_forcer.applicable[_name])
                combined[f"force_forced_{_name}"] = float(self.event_forcer.forced[_name])
        if self.opening_seeder is not None:
            combined["forced_openings_moves"] = float(self.opening_seeder.forced_moves)
        if self.seeder is not None:
            _g = self.seeder.games
            combined["seed_games_frac"] = float(_g[0]) / max(float(_g.sum()), 1.0)
            for _name, _n in self.seeder.forced.items():
                combined[f"seed_forced_{_name}"] = float(_n)
        if self.aux_targets:
            combined.update(self._aux_update())
        if self.aux_card_coef > 0.0:
            combined.update(self._card_aux_update())
        if self.aux_opp_legality > 0.0:
            combined.update(self._opp_legal_update())
        if self.turn_credit is not None:
            combined.update(self.turn_credit.update(self.active_net, self.optimizer, self.max_grad_norm))

        # Fixed-probe entropy: evaluated on the frozen pool every N iterations, so it is
        # directly comparable across the run unlike the on-policy "entropy" above.
        if self.total_iterations == 1 or self.total_iterations % self.entropy_probe_interval == 0:
            probe_entropy = self.entropy_probe.mean_entropy(self.active_net)
            if probe_entropy is not None:
                combined["entropy_fixed_probe"] = probe_entropy

        return combined


class NashPGTrainer(BaseNashPGTrainer):
    """Standard NashPG Trainer for ColdWarNet V1, V2, and V3."""

    def _ingest_searched_games(self) -> int:
        """Start the searched-games workers on first use; move the decisions they have finished into the
        replay store. Returns the rows added."""
        if self.searched_games_workers <= 0:
            return 0
        if self._sg is None:
            self._sg = SearchedGames(self.searched_games_workers, self.reference_net,
                                     RolloutConfig(k=4, worlds=16, horizon=4, rule="z2", stop="earlier",
                                                   leaf_side="own"),
                                     self.device, envs=self.searched_games_envs)
            print(f"[searched games] {self.searched_games_workers} worker(s) x {self.searched_games_envs} games, "
                  f"rollout root k=4, 16 worlds, stop=earlier, leaf=own, on pi_ref; value coef "
                  f"{self.searched_value_coef}", flush=True)
            return 0
        got = self._sg.drain()
        if got is None:
            return 0
        obs, tgt, _games = got
        if self._sg_obs is None:
            self._sg_obs = torch.zeros((self._sg_cap, obs.shape[1]), dtype=torch.float16)
            self._sg_tgt = torch.zeros(self._sg_cap, dtype=torch.float32)
        assert self._sg_obs is not None and self._sg_tgt is not None
        w = min(obs.shape[1], self._sg_obs.shape[1])
        for lo in range(0, obs.shape[0], self._sg_cap):
            o, t = obs[lo:lo + self._sg_cap], tgt[lo:lo + self._sg_cap]
            idx = (self._sg_next + np.arange(o.shape[0])) % self._sg_cap
            ii = torch.from_numpy(idx)
            self._sg_obs[ii, :w] = torch.from_numpy(o[:, :w]).to(torch.float16)
            self._sg_tgt[ii] = torch.from_numpy(t)
            self._sg_next = int((self._sg_next + o.shape[0]) % self._sg_cap)
            self._sg_size = min(self._sg_cap, self._sg_size + o.shape[0])
        return int(obs.shape[0])

    def _build_predeal_targets(self) -> None:
        """The pre-deal rows of this rollout with their targets, in _pd_obs / _pd_tgt. A side's target
        is the lambda-return of its next decision in the same env (a learner row: the value targets
        are the learner's), or the game's result if it ends first (a held scoring card or final
        scoring at this turn end, or a win before that side moves again); a row whose next decision
        falls outside the buffer, or is an opponent-pool snapshot's, is dropped."""
        rows, self._predeal_rows = self._predeal_rows, []
        self._pd_obs = self._pd_tgt = None
        if not rows:
            return
        players = self.buffer.players.cpu().numpy()
        dones = self.buffer.dones.cpu().numpy() > 0.5
        rewards = self.buffer.rewards.float().cpu().numpy()
        rets = self.buffer.returns_win.float().cpu().numpy()
        learner = self.buffer.learner.float().cpu().numpy() > 0.5
        horizon = self.buffer.buffer_size
        obs: List[torch.Tensor] = []
        tgt: List[float] = []
        for t, envs, rows_us, rows_ussr in rows:
            for j, e in enumerate(envs):
                for side, row in ((1, rows_us[j]), (-1, rows_ussr[j])):
                    target: Optional[float] = None
                    for d in range(t, horizon):
                        if dones[d, e]:
                            target = float(np.clip(rewards[d, e] * players[d, e] * side, -1.0, 1.0))
                            break
                        if d + 1 < horizon and players[d + 1, e] == side:
                            if learner[d + 1, e]:
                                target = float(rets[d + 1, e])
                            break
                    if target is not None:
                        obs.append(row)
                        tgt.append(target)
        if obs:
            self._pd_obs = torch.stack(obs)
            self._pd_tgt = torch.tensor(tgt, dtype=torch.float32, device=self.device)

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
                                                   self.wolf_dead_zone, self.wolf_dead_zone_mode)
        # 3k: this update's per-seat entropy coefficients, as device scalars.
        ent_c_us = torch.tensor(self.ent_coef_seat[1], device=self.device)
        ent_c_ussr = torch.tensor(self.ent_coef_seat[-1], device=self.device)
        setup_c = self.setup_ent_coef
        _setup_slot_u = setup_phase_slot()
        # 3l: whether each seat is still updating its policy (1.0) or has been stopped (0.0), and
        # the per-seat approximate KL, accumulated on the device so the check costs no sync.
        seat_act = {1: torch.ones((), dtype=torch.float64, device=self.device),
                    -1: torch.ones((), dtype=torch.float64, device=self.device)}
        seat_kl_sum = {1: torch.zeros((), dtype=torch.float64, device=self.device),
                       -1: torch.zeros((), dtype=torch.float64, device=self.device)}
        seat_kl_cnt = {1: torch.zeros((), dtype=torch.float64, device=self.device),
                       -1: torch.zeros((), dtype=torch.float64, device=self.device)}
        seat_stopped = {1: torch.zeros((), dtype=torch.float64, device=self.device),
                        -1: torch.zeros((), dtype=torch.float64, device=self.device)}
        # Per-minibatch diagnostics accumulate ON THE DEVICE and are read once, after the loop.
        # Reading each with .item()/float()/bool() inside the loop forced ~14 CPU-GPU syncs per
        # minibatch, each stalling the queue until the GPU caught up
        # (research/log/training_throughput_cpu.md).
        def _z(v: float = 0.0) -> torch.Tensor:
            # float64, as the Python floats these replace were.
            return torch.full((), v, dtype=torch.float64, device=self.device)
        loss_t, policy_loss_t, policy_loss_total_t = _z(), _z(), _z()
        cf_loss_t = _z()
        val_loss_t, kl_t, entropy_t, clip_frac_t, risk_loss_t = _z(), _z(), _z(), _z(), _z()
        logratio_max_t, old_lp_min_t, ratio_negadv_max_t = _z(-1e30), _z(1e30), _z()
        lse_sum_t, lse_absmax_t, z_loss_t = _z(), _z(), _z()
        gap_loss_t, gap_frac_t = _z(), _z()

        # pi_ref's log-probabilities over the whole buffer, once. pi_ref is frozen for the whole
        # update (it is refreshed after train_step, in train_iteration) and always in eval mode,
        # and the rollout data does not change across epochs -- so recomputing them for every
        # minibatch of every epoch did num_epochs times the work for the same numbers. Chunked at
        # the minibatch size, so each forward has the shape the per-minibatch one had.
        _n_all = self.buffer.buffer_size * self.num_envs
        _all_obs = self.buffer.obs.view(_n_all, self.buffer.obs_dim)
        _all_masks = self.buffer.masks.view(_n_all, self.buffer.action_dim)
        _ref_fwd = self._update_net(self.reference_net)
        _act_fwd = self._update_net(self.active_net)
        with torch.no_grad():
            ref_log_p_all = torch.cat([
                F.log_softmax(_ref_fwd(_all_obs[i:i + self.batch_size],
                                                 _all_masks[i:i + self.batch_size])[0], dim=-1)
                for i in range(0, _n_all, self.batch_size)])

        teacher_log_p_all: Optional[torch.Tensor] = None
        if self.teacher_net is not None:
            _teach_fwd = self._update_net(self.teacher_net)
            with torch.no_grad():
                teacher_log_p_all = torch.cat([
                    F.log_softmax(_teach_fwd(_all_obs[i:i + self.batch_size],
                                             _all_masks[i:i + self.batch_size])[0], dim=-1)
                    for i in range(0, _n_all, self.batch_size)])
        teacher_kl_t, teacher_rows_t = _z(), _z()
        self._build_predeal_targets()
        pd_loss_t, pd_batches = _z(), 0
        sg_added = self._ingest_searched_games()
        sg_loss_t, sg_batches = _z(), 0
        _pd_rows = 0 if self._pd_tgt is None else int(self._pd_tgt.shape[0])

        for _ in range(self.num_epochs):
            # Each minibatch of an epoch takes its share of the pre-deal rows, each row once an epoch.
            _pd_chunks: List[torch.Tensor] = (
                list(torch.tensor_split(torch.randperm(_pd_rows, device=self.device),
                                        max(1, -(-_n_all // self.batch_size)))) if _pd_rows else [])
            _mb = 0
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
                    cur_logits, cur_v_win, cur_v_vp = _act_fwd(b_obs, b_mask)
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
                with torch.no_grad():
                    # 3l: each seat's approximate KL from the rollout policy on this minibatch,
                    # measured before the step; stopping applies from this minibatch on.
                    _k3 = ((ratio - 1.0) - torch.clamp(log_ratio, -20.0, 20.0)).double()
                    _own_b = b_learner > 0.5
                    if self.setup_mc_credit:
                        # Setup placements are not in this surrogate (they train in
                        # _setup_mc_update), so they are not in its KL either. A scripted one
                        # (--setup-script-frac) has log-prob near -18 and would swamp the statistic.
                        _own_b = _own_b & (b_obs[:, setup_phase_slot()] >= (0.5 / 6.0))
                    for _code in (1, -1):
                        _sel = (_own_b & (b_players == _code)).double()
                        _cnt = _sel.sum()
                        _has = (_cnt > 0).double()
                        _kl_s = (_k3 * _sel).sum() / _cnt.clamp(min=1.0)
                        seat_kl_sum[_code] += _kl_s * _has
                        seat_kl_cnt[_code] += _has
                        if self.target_kl > 0.0:
                            _over = ((_kl_s > self.target_kl) & (_cnt > 0)).double()
                            seat_act[_code] = seat_act[_code] * (1.0 - _over)
                            seat_stopped[_code] += 1.0 - seat_act[_code]
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                surrogate = -torch.min(surr1, surr2)
                if self._behaviour_w is not None:
                    # P31 1a: the floor's importance weight pi_old / mu, outside the clip.
                    surrogate = surrogate * self._behaviour_w.index_select(0, b_idx)
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
                if self.setup_mc_credit:
                    # Setup placements take their credit from the game result instead
                    # (_setup_mc_update); the lambda-return reaching them is the critic's.
                    keep = keep & (b_obs[:, setup_phase_slot()] >= (0.5 / 6.0))
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
                if self.target_kl > 0.0:
                    # A stopped seat's samples leave the numerator only: the denominator stays the
                    # full count, so stopping one seat does not speed up the other.
                    _seat_w = torch.where(b_players == 1, seat_act[1], seat_act[-1]).to(surrogate.dtype)
                    ppo_loss = ((surrogate * _keep_f * _seat_w).sum()
                                / _keep_f.sum().clamp(min=1.0))
                else:
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
                # The entropy the bonus rewards: raw, or as a fraction of each decision's maximum.
                # The logged entropy stays raw (cur_entropy.mean() below).
                ent_b = bonus_entropy(cur_entropy, b_mask, self.entropy_normalize)
                own_entropy = (ent_b * _own_f).sum() / _own_n
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
                    ent_term = (ent_b * _w * _own_f).sum() / _own_n
                ent_loss = self.ent_coef * ent_term
                if self.target_kl > 0.0:
                    _sw = torch.where(b_players == 1, seat_act[1], seat_act[-1]).to(kl_per.dtype)
                    kl_term = (kl_per * _sw).mean()
                    ent_term = (ent_b * _own_f * _sw).sum() / _own_n
                    ent_loss = self.ent_coef * ent_term
                if self.entropy_ceiling > 0.0:
                    _c = torch.where(b_players == 1, ent_c_us, ent_c_ussr).to(cur_entropy.dtype)
                    _ew = _own_f
                    if self.target_kl > 0.0:
                        _ew = _own_f * torch.where(b_players == 1, seat_act[1],
                                                   seat_act[-1]).to(cur_entropy.dtype)
                    ent_loss = (ent_b * _ew * _c).sum() / _own_n
                if setup_c > 0.0:
                    # The setup floor's bonus: raw entropy (the floor is in nats), learner's setup
                    # placements only, normalised per learner decision like the ordinary bonus.
                    _setup_f = (b_obs[:, _setup_slot_u] < (0.5 / 6.0)).to(cur_entropy.dtype) * _own_f
                    ent_loss = ent_loss + setup_c * (cur_entropy * _setup_f).sum() / _own_n
                policy_loss = (ppo_loss + self.eta * kl_term - ent_loss
                               + self.search_ce_coef * search_ce)
                if teacher_log_p_all is not None:
                    _t_rows = b_learner > 0.5
                    if self.teacher_seat != 0:
                        _t_rows = _t_rows & (b_players == self.teacher_seat)
                    _t_kl, _t_n = teacher_kl(cur_log_p, teacher_log_p_all.index_select(0, b_idx),
                                             _t_rows)
                    policy_loss = policy_loss + self.teacher_coef * _t_kl
                    teacher_kl_t += _t_kl.detach().double()
                    teacher_rows_t += _t_n.detach().double()
                if self._cf_has is not None and self._cf_adv is not None and self.mode_cf_coef > 0.0:
                    # P31 1c: the all-options term, -c * sum_a pi(a) * (Q(a) - Q(taken)), on the
                    # minibatch's priced rows only (no host sync: an empty selection adds zero).
                    _cf_loss = mode_cf_loss(cur_logits, self._cf_adv.index_select(0, b_idx),
                                            self._cf_has.index_select(0, b_idx))
                    policy_loss = policy_loss + self.mode_cf_coef * _cf_loss
                    cf_loss_t += _cf_loss.detach()
                # The log-normaliser over the legal actions (masked logits are -1e9, so they add
                # exactly nothing): the level the z-loss holds down.
                lse = torch.logsumexp(cur_logits.float(), dim=-1)
                if self.z_loss_coef > 0.0:
                    z_loss = self.z_loss_coef * (lse * lse).mean()
                    policy_loss = policy_loss + z_loss
                    z_loss_t += z_loss.detach()
                if self.logit_gap_coef > 0.0:
                    _legal = b_mask.bool()
                    _z_f = cur_logits.float()
                    _top = _z_f.masked_fill(~_legal, float("-inf")).amax(dim=-1, keepdim=True)
                    _top = torch.where(torch.isfinite(_top), _top, torch.zeros_like(_top))
                    _excess = torch.relu((_top - _z_f) - self.logit_gap) * _legal.to(_z_f.dtype)
                    _n_legal = _legal.sum(dim=-1).clamp(min=1).to(_z_f.dtype)
                    _gap_loss = ((_excess * _excess).sum(dim=-1) / _n_legal).mean()
                    policy_loss = policy_loss + self.logit_gap_coef * _gap_loss
                    with torch.no_grad():
                        gap_loss_t += _gap_loss.detach().double()
                        gap_frac_t += ((_excess > 0).sum().double()
                                       / _legal.sum().clamp(min=1).double())
                with torch.no_grad():
                    lse_sum_t += lse.mean().double()
                    lse_absmax_t = torch.maximum(lse_absmax_t, lse.abs().max().double())
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

                if _mb < len(_pd_chunks) and _pd_chunks[_mb].numel() and self._pd_obs is not None \
                        and self._pd_tgt is not None:
                    _sel = _pd_chunks[_mb]
                    _pd_v = self.active_net(self._pd_obs.index_select(0, _sel), None)[1].squeeze(-1)
                    _pd_loss = F.mse_loss(_pd_v.float(), self._pd_tgt.index_select(0, _sel))
                    loss = loss + self.vf_coef * self.predeal_value_coef * _pd_loss
                    pd_loss_t += _pd_loss.detach()
                    pd_batches += 1
                _mb += 1
                if self._sg_size and self._sg_obs is not None and self._sg_tgt is not None:
                    _si = torch.randint(self._sg_size, (self.searched_value_batch,))
                    _sv = self.active_net(self._sg_obs[_si].to(self.device, torch.float32), None)[1].squeeze(-1)
                    _sl = F.mse_loss(_sv.float(), self._sg_tgt[_si].to(self.device))
                    loss = loss + self.vf_coef * self.searched_value_coef * _sl
                    sg_loss_t += _sl.detach()
                    sg_batches += 1

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
                # invisible for two days. See research/archive/E3_ladder/log/P15_kl_domination.md.
                policy_loss_t += ppo_loss.detach()
                policy_loss_total_t += policy_loss.detach()
                val_loss_t += val_loss.detach()
                kl_t += kl_div.detach()
                entropy_t += cur_entropy.mean().detach()
                num_updates += 1

        # One host read for everything accumulated on the device.
        (total_loss_accum, policy_loss_accum, policy_loss_total_accum, val_loss_accum, kl_accum,
         entropy_accum, clip_frac_accum, risk_loss_accum, logratio_max_accum, old_lp_min_accum,
         ratio_negadv_max_accum, akl_us_s, akl_us_n, akl_ussr_s, akl_ussr_n, stop_us, stop_ussr,
         lse_sum_accum, lse_absmax_accum, z_loss_accum
         ) = torch.stack([
            loss_t, policy_loss_t, policy_loss_total_t, val_loss_t, kl_t, entropy_t, clip_frac_t,
            risk_loss_t, logratio_max_t, old_lp_min_t, ratio_negadv_max_t,
            seat_kl_sum[1], seat_kl_cnt[1], seat_kl_sum[-1], seat_kl_cnt[-1],
            seat_stopped[1], seat_stopped[-1],
            lse_sum_t, lse_absmax_t, z_loss_t]).tolist()
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
            **({"mode_cf_loss": float(cf_loss_t) / max(1, num_updates)} if self.mode_cf is not None else {}),
            # The policy logits' level (log-normaliser over legal actions), mean over minibatches
            # and largest |value| seen; and the z-loss term when it is on.
            "logit_lse_mean": lse_sum_accum / max(1, num_updates),
            "logit_lse_absmax": lse_absmax_accum,
            "z_loss": z_loss_accum / max(1, num_updates),
            # The gap penalty (before its coefficient) and the share of legal moves beyond the gap.
            "logit_gap_loss": float(gap_loss_t) / max(1, num_updates),
            "logit_gap_frac": float(gap_frac_t) / max(1, num_updates),
            "val_loss": val_loss_accum / max(1, num_updates),
            **({"predeal_value_loss": float(pd_loss_t) / max(1, pd_batches), "predeal_rows": _pd_rows}
               if self.predeal_value_coef > 0.0 else {}),
            **({"searched_value_loss": float(sg_loss_t) / max(1, sg_batches), "searched_rows": self._sg_size,
                "searched_rows_added": sg_added, "searched_games": 0 if self._sg is None else self._sg.games}
               if self.searched_games_workers > 0 else {}),
            "kl_div": kl_accum / max(1, num_updates),
            # 3l: per-seat approximate KL from the rollout policy (mean over minibatches), and
            # the share of minibatches in which the seat's policy terms were stopped.
            "approx_kl_us": akl_us_s / max(1.0, akl_us_n),
            "approx_kl_ussr": akl_ussr_s / max(1.0, akl_ussr_n),
            "kl_stop_frac_us": stop_us / max(1, num_updates),
            "kl_stop_frac_ussr": stop_ussr / max(1, num_updates),
            "entropy": entropy_accum / max(1, num_updates),
            "clip_frac": clip_frac_accum / max(1, num_updates),
            # 0.0 when search CE is off, so the key is always present and a run without it is
            # still distinguishable from a run whose term silently produced nothing.
            # P32: the mean per-minibatch KL(teacher || policy) on the distilled seat's own
            # decisions, and those decisions per minibatch; 0.0 without a teacher.
            "teacher_kl": float(teacher_kl_t) / max(1, num_updates),
            "teacher_rows": float(teacher_rows_t) / max(1, num_updates),
            # P32 T1: the share of searched decisions whose policy was saturated (max p > 1 - 1e-6)
            # at the last rollout -- where an improved-policy target cannot move it.
            "search_saturated_frac": float(getattr(self, "search_saturated_frac", 0.0)),
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
