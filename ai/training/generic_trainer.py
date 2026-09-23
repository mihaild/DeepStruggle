import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Generic Trainer: Configurable multi-stage training with live snapshot tournament evaluation.

import copy as _copy
import os
import re
import sys
import subprocess
import time
import collections
import json
import re
import argparse
from typing import List, Optional, Dict, Any, Final, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.models.coldwar_net_v2 import ColdWarNetV2, create_coldwar_net_v2, create_like
from ai.rewards.reward_calculator import ZeroSumTerminalReward, ShapedZeroSumReward, BlunderAwareRewardCalculator, UsefulActionsReward
from bindings.ts_env import OBS_LAYOUT_NAME, TsVectorizedEnv
from ai.training.rollout_buffer import RolloutBuffer
from ai.training.nash_pg import NashPGTrainer
from ai.training.start_pool import DEFAULT_TURN_MIX, StartPositionPool
from ai.eval.agreement import evaluate_dataset
from ai.training.human_corpus_dataset import HumanCorpusDataset
from ai.training.warmup_dataset_loader import WarmupDataset
from bindings.ts_env import ENDING_REASON_KEYS
from ai.eval.blunders import RULES as BLUNDER_RULES
from ai.itsc_reference import ITSC_GAMES, ITSC_REFERENCE, reference_for
from tools.lib.player_agent import PlayerAgent, NeuralAgent, load_agent, resolve_device
from tools.lib.action_view import checkpoint_merged_influence, run_view
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.tournament_evaluator import TournamentEvaluator
from bindings.action_encoder import ActionEncoder

# TensorBoard is optional: a missing (or broken) install must never take down a
# multi-hour training run, so the writer degrades to a no-op and JSONL logging carries on.
_SUMMARY_WRITER_CLS: Optional[Any] = None
_SUMMARY_WRITER_IMPORT_ERROR: Optional[str] = None
try:
    from torch.utils.tensorboard import SummaryWriter as _ImportedSummaryWriter
    _SUMMARY_WRITER_CLS = _ImportedSummaryWriter
except Exception as _tb_err:  # pragma: no cover - depends on the local install
    _SUMMARY_WRITER_IMPORT_ERROR = str(_tb_err)


# Maps training_metrics.jsonl keys to TensorBoard tags. Every JSONL metric is mirrored;
# anything not listed here lands under "misc/" so new metrics cannot silently go missing.
TB_TAGS: Dict[str, str] = {
    # --- progress: where the run is, and how fast ------------------------------------------
    "elapsed_seconds": "progress/elapsed_seconds",
    "iteration": "progress/iteration",
    "steps_per_sec": "progress/steps_per_sec",
    "steps_per_sec_avg": "progress/steps_per_sec_avg",

    # --- internal: the optimiser's own view. Nothing here says whether the agent plays well,
    # only whether the update is sane and the critic is learning. -------------------------
    "loss": "internal/loss",
    "policy_loss": "internal/policy_loss",
    "value_loss": "internal/value_loss",
    "kl_div": "internal/kl_div",
    "entropy": "internal/entropy",
    "entropy_fixed_probe": "internal/entropy_fixed_probe",
    "clip_frac": "internal/clip_frac",
    "explained_variance": "internal/explained_variance",
    "adv_std": "internal/adv_std",
    "adv_std_raw": "internal/adv_std_raw",

    # --- critic discrimination: does the value head still separate winners from losers? -----
    # Kept out of internal/ because these say something about the agent, not the optimiser:
    # when they fall, the advantage signal is going with them. See
    # research/method/measurement_tiers.md for why AUC rather than accuracy.
    "critic_auc": "critic/auc",
    "critic_auc_turn3": "critic/auc_turn3",
    "critic_brier_skill": "critic/brier_skill",
    "critic_base_rate": "critic/base_rate",
    "critic_samples": "critic/samples",

    # --- opponent pool: is it actually growing, and does it still span the run? ------------
    "opp_pool_size": "opponent/pool_size",
    "opp_win_rate_mean": "opponent/win_rate_mean",
    "opp_win_rate_min": "opponent/win_rate_min",
    "opp_win_rate_max": "opponent/win_rate_max",
    "opp_games_recorded": "opponent/games_recorded",
    "opp_pfsp_entropy": "opponent/pfsp_entropy",
    "opp_pfsp_max_prob": "opponent/pfsp_max_prob",
    "opp_pool_span_m": "opponent/pool_span_msteps",
    "opp_frac_mixed": "opponent/frac_mixed",
    "adv_frac_near_zero": "internal/adv_frac_near_zero",
    # Auxiliary heads. Emitted only when the corresponding option is on, so an absent series
    # here means "not enabled for this run", not "broken".
    "defcon_risk_loss": "internal/defcon_risk_loss",
    "belief_loss": "internal/belief_loss",
    "oracle_loss": "internal/oracle_loss",
    "distill_loss": "internal/distill_loss",

    # --- endgame: what the games themselves look like, and the only group with human
    # counterparts. Every chart here is shared by up to six lines -- pooled, the games the US
    # won, the games the USSR won, and a human line for each -- written from sibling runs.
    "episodes_completed": "endgame/episodes_completed",
    "us_win_rate": "endgame/us_win_rate",
    "ussr_win_rate": "endgame/ussr_win_rate",
    "draw_rate": "endgame/draw_rate",
    "mean_victory_points": "endgame/mean_victory_points",
    "mean_vp_margin": "endgame/mean_vp_margin",
    "mean_turn": "endgame/turn",
    "median_turn": "endgame/median_turn",
    "mean_ply": "endgame/ply",
    "median_ply": "endgame/median_ply",
    "ending_frac_20vp": "endgame/ending_20vp",
    "ending_frac_europe_control": "endgame/ending_europe_control",
    "ending_frac_final_scoring": "endgame/ending_final_scoring",
    "ending_frac_wargames": "endgame/ending_wargames",
    "ending_frac_held_scoring": "endgame/ending_held_scoring",
    "ending_frac_defcon1": "endgame/ending_defcon1",
    "ending_frac_defcon1_self": "endgame/ending_defcon1_self",
    "ending_frac_defcon1_provoked": "endgame/ending_defcon1_provoked",

    # --- strategy: is it playing the board well? Measured off probe games at snapshots, not
    # off the training rollouts, and none of it has a human counterpart yet.
    # The battleground diagnostics. Without an explicit mapping these fell to misc/ as
    # "misc/diag/empty_battlegrounds_turn8" -- nobody had added one because they were suppressed
    # from individual charts, so the gap was invisible until the suppression was lifted.
    "diag/empty_battlegrounds_turn5": "strategy/empty_battlegrounds_turn5",
    "diag/empty_battlegrounds_turn8": "strategy/empty_battlegrounds_turn8",
    "diag/uncontrolled_battlegrounds_turn5": "strategy/uncontrolled_battlegrounds_turn5",
    "diag/uncontrolled_battlegrounds_turn8": "strategy/uncontrolled_battlegrounds_turn8",
    "diag/salvageable_frac_turn6": "strategy/salvageable_frac_turn6",
    "diag/salvageable_given_reached_turn6": "strategy/salvageable_given_reached_turn6",
    "diag/mean_final_turn": "strategy/probe_mean_final_turn",
    "decisive_win_take_rate": "strategy/decisive_win_take",
    "decisive_loss_avoid_rate": "strategy/decisive_loss_avoid",
}

#: The sampling temperatures the blunder probe runs at, and the metric prefix each writes under.
#:
#: Evaluation has always run at 0.1 while rollouts are collected at a stratified 0.1-0.5 and the
#: PPO ratio is taken against canonical 1.0 log-probabilities. A low evaluation temperature was
#: introduced to stop the policy acting on moves it rates as unlikely-but-not-negligible -- a
#: battleground coup at DEFCON 2 and its relatives, which is exactly what these rules count. That
#: makes a single-temperature blunder rate unreadable: it cannot distinguish a policy that has
#: learned not to blunder from one whose blunders are merely being argmaxed away at evaluation
#: time. Both are reported so the gap between them is visible, and 0.1 keeps the unprefixed name
#: so existing runs stay comparable.
BLUNDER_PROBES: Tuple[Tuple[float, str], ...] = ((0.1, "blunder"), (1.0, "blunder_hot"))

#: Blunder rules from ai/eval/blunders.py, logged as `strategy/blunder_<rule>_rate` plus the
#: numerator and denominator. A rate with no denominator cannot be compared across runs: a
#: policy that never held Olympic Games at DEFCON 2 has demonstrated nothing by not misplaying
#: it, which is why `_chances` is logged beside `_rate`.
for _prefix in (_p for _t, _p in BLUNDER_PROBES):
    for _rule in BLUNDER_RULES:
        TB_TAGS[f"{_prefix}_{_rule}_rate"] = f"strategy/{_prefix}_{_rule}"

#: The metric stems that make up the endgame section. Each also exists per winning side, and
#: under mid-game start sampling per start turn.
GAME_STEMS: Tuple[str, ...] = (
    "episodes_completed",
    "mean_turn", "median_turn", "mean_ply", "median_ply",
    "mean_victory_points", "mean_vp_margin",
) + tuple(f"ending_frac_{k}" for k in tuple(ENDING_REASON_KEYS) + ("defcon1",))


#: A run's short name is `<engine>-<attempt>-<seed>`, e.g. `E9-99-01`. A step budget is
#: deliberately not part of it: one directory holds every budget of a lineage, and the budget is
#: already in each snapshot's filename.
#: Optional suffixes, from research/method/run_nomenclature.md:
#:   -<n>            a same-seed REPLICATE                   E4-08-01-2
#:   -<M>M.<seed>    a BRANCH resumed under a new seed,      E4-17-06-50M.11
#:                   repeatable, one per branch point        E4-17-06-50M.11-90M.07
#: A same-seed CONTINUATION takes no suffix: it keeps its lineage's name, and the budget is
#: in each snapshot's filename.
RUN_NAME_RE: Final = re.compile(r"^E\d+(?:\.\d+)?-\d{2}-\d{2}(?:-\d+)?(?:-\d+M\.\d{2})*$")


def _resolve_run_dir(output_dir: Optional[str], run_name: Optional[str],
                     arch: str, timestamp: str) -> str:
    """Where a run writes, with the short name in the directory itself.

    Giving `run_name` produces a directory called `<run_name>_<timestamp>`, so a listing of the
    checkpoint directory identifies each run without opening ten `metadata.json` files.

    Passing both is allowed only when they agree. They disagreeing is the failure this is for:
    the short name would then say one thing and the path another, and the path is what every
    later command quotes.
    """
    if run_name is not None and not RUN_NAME_RE.match(run_name):
        raise ValueError(
            f"run_name {run_name!r} is not <engine>-<attempt>-<seed> (e.g. 'E9-99-01'). "
            "Register the run under a conforming name before launching it.")
    if output_dir is not None:
        if run_name is not None and run_name not in os.path.basename(output_dir.rstrip("/")):
            raise ValueError(
                f"run_name {run_name!r} is not in output_dir {output_dir!r}. The directory name "
                "is what later commands quote, so it is the copy that has to carry the name.")
        return output_dir
    # The SHARED data tree, not `data/` relative to the current directory. A git worktree gets
    # its own empty data/ because the directory is git-ignored, so a run launched from one writes
    # where nothing else looks and loses everything when the worktree is removed.
    # tools/lib/data_root.py resolves the main checkout from git.
    from tools.lib.data_root import checkpoints_dir

    if run_name is not None:
        return os.path.join(checkpoints_dir(), f"{run_name}_{timestamp}")
    return os.path.join(checkpoints_dir(), f"run_{arch}_{timestamp}")


def _game_chart(stem: str) -> str:
    """Chart name for an endgame stem: `ending_frac_20vp` reads better as `ending_20vp`."""
    if stem.startswith("ending_frac_"):
        return "ending_" + stem[len("ending_frac_"):]
    return stem


#: Populations that share the endgame charts, as (metric-key suffix, run directory). Each is
#: written as its own TensorBoard *run* under the same tag, because TensorBoard draws one line
#: per run per chart -- so `endgame/turn` carries the pooled mean, the mean over games the US
#: won, the mean over games the USSR won, and a human line for each, in one chart rather than
#: six. "." is the run's own directory, i.e. the main writer.
POPULATIONS: Tuple[Tuple[str, str], ...] = (
    ("", "."),
    ("_won_us", "won_us"),
    ("_won_ussr", "won_ussr"),
)

#: Where each population's human reference line is written. Same tags again, one run each.
HUMAN_RUNS: Dict[str, str] = {
    "": "human_ITS",
    "_won_us": "human_won_us",
    "_won_ussr": "human_won_ussr",
}

#: Charts that combine metrics which are *different quantities*, not one quantity over
#: different populations -- the sibling-run trick cannot express those, so they go through
#: SummaryWriter.add_scalars. Values are metric keys, or floats for a fixed human line.
MULTILINE_CHARTS: Dict[str, Dict[str, Union[str, float]]] = {
    "endgame/win_rate": {
        "us": "us_win_rate",
        "ussr": "ussr_win_rate",
        "draw": "draw_rate",
        "human_us": ITSC_REFERENCE["us_win_rate"],
        "human_ussr": ITSC_REFERENCE["ussr_win_rate"],
        "human_draw": ITSC_REFERENCE["draw_rate"],
    },
    "endgame/ending_mix": {
        "20vp": "ending_frac_20vp",
        "europe_control": "ending_frac_europe_control",
        "final_scoring": "ending_frac_final_scoring",
        "wargames": "ending_frac_wargames",
        "defcon1": "ending_frac_defcon1",
        "held_scoring": "ending_frac_held_scoring",
    },
    # All four battleground series on one chart. Empty and uncontrolled answer different
    # questions -- an empty battleground is one nobody has touched, an uncontrolled one may be
    # heavily contested and still score for nobody -- and both are worth watching at the point
    # the board is set (turn 5) and the point it stops moving (turn 8).
    "strategy/battlegrounds": {
        "empty_turn5": "diag/empty_battlegrounds_turn5",
        "empty_turn8": "diag/empty_battlegrounds_turn8",
        "uncontrolled_turn5": "diag/uncontrolled_battlegrounds_turn5",
        "uncontrolled_turn8": "diag/uncontrolled_battlegrounds_turn8",
    },
}

# A rate and its 95% Wilson interval on one chart, instead of three series (rate, numerator,
# denominator) that the reader has to combine mentally. The band is what makes the rate safe to
# read alone: it encodes the sample size, so a rule with two chances is visibly a band across
# most of [0, 1] rather than a confident-looking 0.0.
for _prefix in (_p for _t, _p in BLUNDER_PROBES):
    for _rule in BLUNDER_RULES:
        MULTILINE_CHARTS[f"strategy/{_prefix}_{_rule}"] = {
            "rate": f"{_prefix}_{_rule}_rate",
            "ci_low": f"{_prefix}_{_rule}_ci_low",
            "ci_high": f"{_prefix}_{_rule}_ci_high",
        }
for _name in ("win_take", "loss_avoid"):
    MULTILINE_CHARTS[f"strategy/decisive_{_name}"] = {
        "rate": f"decisive_{_name}_rate",
        "ci_low": f"decisive_{_name}_ci_low",
        "ci_high": f"decisive_{_name}_ci_high",
    }

# Per-start-turn variants are NOT pre-registered. They exist only when mid-game start sampling
# is on, which is no run since it was settled negative (--start-pool-frac defaults to 0), and
# enumerating five turns x eleven stems put 55 dead entries -- 42% of the whole table -- in
# front of every reader. `_tb_tag` derives them by rule instead, so the series still appear
# correctly if anyone turns start sampling back on.

# Metrics that describe completed episodes; meaningless (and misleading as zeros) on an
# iteration where no game finished, so they are held back from TensorBoard then.
EPISODE_DEPENDENT_KEYS = frozenset(
    ["mean_turn", "median_turn", "mean_ply", "median_ply",
     "us_win_rate", "ussr_win_rate", "draw_rate", "mean_terminal_utility",
     "mean_victory_points", "mean_vp_margin"]
    + [f"ending_frac_{k}" for k in tuple(ENDING_REASON_KEYS) + ("defcon1",)]
)

#: Suffixes the episode-dependent stems appear under: the per-start-turn groups and the
#: per-winner ones. A group with no episodes this iteration must be held back exactly as the
#: pooled one is, or an iteration the USSR happened to lose every game in reports a US mean
#: turn of 0 rather than nothing.
_GROUP_SUFFIXES: Tuple[str, ...] = ("_start", "_won_us", "_won_ussr")

#: Matches the `<stem>_start<turn>` keys the start-pool breakdown emits.
_START_SUFFIX = re.compile(r"^(.*)_start(\d+)$")


def episode_dependent_in(stats: Dict[str, float]) -> frozenset:
    """Which keys of `stats` are episode-dependent, including per-start-turn variants.

    A per-start-turn group is only emitted when that start turn actually completed an
    episode, so the suffixed keys cannot be enumerated up front -- publishing zeros for
    absent groups would be the misleading thing this hold-back exists to prevent.
    """
    return frozenset(
        k for k in stats
        if k in EPISODE_DEPENDENT_KEYS
        or any(k.startswith(f"{stem}{sfx}")
               for stem in EPISODE_DEPENDENT_KEYS for sfx in _GROUP_SUFFIXES)
    )


#: Kept in the JSONL but not mirrored to TensorBoard. `total_steps` *is* the x-axis now, so a
#: chart of it would be the line y = x. Suppressed explicitly rather than left to fall into
#: misc/, where it reads as a metric someone forgot to name.
_TB_SUPPRESSED: frozenset = frozenset({
    # The x-axis itself; a chart of it would be the line y = x.
    "total_steps",
    # sign(final VP) averaged over episodes, i.e. exactly us_win_rate - ussr_win_rate, both of
    # which are charted. Kept in the JSONL because older analysis reads it.
    "mean_terminal_utility",
    # Numerators and denominators behind the banded rate charts. The Wilson interval drawn
    # around each rate already carries the sample size -- two chances give a band across most
    # of [0, 1], four hundred give a tight one -- so these no longer need a chart each. Still
    # written to the JSONL.
    "decisive_win_available",
    "decisive_loss_avoidable",
    # NOT suppressed any more. These were held back on the assumption that the combined
    # `strategy/battlegrounds` chart would show them -- but SummaryWriter.add_scalars does not
    # draw a chart in the run it is called on. It writes one CHILD RUN DIRECTORY per series
    # (tb/strategy_battlegrounds_empty_turn8/, ...), which TensorBoard lists in the left-hand
    # RUNS panel. So the four names appeared in the sidebar while the parent run had no tag for
    # them and selecting it showed nothing -- the data was there and unreachable without knowing
    # to tick four extra runs. They are now written as ordinary scalars as well; the combined
    # chart still exists for anyone who wants the four overlaid.
} | {
    f"{_p}_{_r}_{_part}" for _p in (_q for _t, _q in BLUNDER_PROBES)
    for _r in BLUNDER_RULES for _part in ("count", "chances")
} | {
    # Interval bounds belong to their rate's band chart and nowhere else; written as scalars of
    # their own they would each open a chart containing one edge of a band.
    f"{_p}_{_r}_{_part}" for _p in (_q for _t, _q in BLUNDER_PROBES)
    for _r in BLUNDER_RULES for _part in ("ci_low", "ci_high")
} | {
    f"decisive_{_n}_{_part}" for _n in ("win_take", "loss_avoid")
    for _part in ("ci_low", "ci_high")
})


def _tb_tag(key: str) -> str:
    """The TensorBoard tag for a metric key.

    Most are looked up. Per-opponent evaluation results cannot be: the opponent list is built at
    runtime, so those are matched by prefix. Anything unrecognised still lands under misc/ rather
    than being dropped, so a new metric cannot go missing silently.
    """
    if key in TB_TAGS:
        return TB_TAGS[key]
    if key.startswith("eval_win_rate_"):
        return "eval/win_rate_vs_" + key[len("eval_win_rate_"):]
    # Per-start-turn variants, derived rather than enumerated: they exist only under mid-game
    # start sampling, so pre-registering five turns x every game stem filled the table with
    # entries no ordinary run ever emits.
    match = _START_SUFFIX.match(key)
    if match:
        stem, turn = match.group(1), match.group(2)
        if stem in GAME_STEMS:
            return f"game_start{turn}/{_game_chart(stem)}"
    return f"misc/{key}"


class TensorBoardLogger:
    """Best-effort TensorBoard writer. Any failure disables it instead of raising.

    Charts are shared by several *runs* rather than split into several tags. TensorBoard draws
    one line per run per chart, so writing the same tag from sibling directories is what puts
    the pooled series, the per-winner splits and the human references together in one chart --
    `endgame/turn` carries six lines instead of occupying six charts. `add_scalars` covers the
    other case, where a chart combines genuinely different quantities (US / USSR / draw rate).
    """

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.log_dir = log_dir
        self.writer: Optional[Any] = None
        #: run directory -> writer, for every population and human reference line.
        self.side_writers: Dict[str, Any] = {}
        if not enabled:
            return
        if _SUMMARY_WRITER_CLS is None:
            print(
                f"Warning: TensorBoard logging unavailable ({_SUMMARY_WRITER_IMPORT_ERROR}); "
                f"continuing with JSONL metrics only. Install with: pip install tensorboard",
                flush=True,
            )
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = _SUMMARY_WRITER_CLS(log_dir=log_dir)
            for _suffix, run_dir in POPULATIONS:
                if run_dir != ".":
                    self.side_writers[run_dir] = self._open(run_dir)
            for run_dir in HUMAN_RUNS.values():
                self.side_writers[run_dir] = self._open(run_dir)
            print(f"TensorBoard logging enabled -> {log_dir}  (tensorboard --logdir {log_dir})",
                  flush=True)
            print(f"  lines per chart: pooled + won_us + won_ussr, each against a human "
                  f"reference from {ITSC_GAMES:,} ITS games", flush=True)
        except Exception as e:
            self.writer = None
            self.side_writers = {}
            print(f"Warning: Could not start TensorBoard writer at {log_dir}: {e}. "
                  f"Continuing without it.", flush=True)

    def _open(self, run_dir: str) -> Any:
        if _SUMMARY_WRITER_CLS is None:  # unreachable: __init__ returns early without it
            raise RuntimeError("no SummaryWriter available")
        path = os.path.join(self.log_dir, run_dir)
        os.makedirs(path, exist_ok=True)
        return _SUMMARY_WRITER_CLS(log_dir=path)

    @property
    def active(self) -> bool:
        return self.writer is not None

    def _writer_for(self, key: str) -> tuple[Any, str, str]:
        """(writer, base metric key, population suffix) for a metric key.

        A `_won_us` / `_won_ussr` metric is the same quantity over a subset of the games, so it
        belongs on the same chart as the pooled one, written from that population's run.
        """
        for suffix, run_dir in POPULATIONS:
            if suffix and key.endswith(suffix):
                writer = self.side_writers.get(run_dir)
                if writer is not None:
                    return writer, key[: -len(suffix)], suffix
        return self.writer, key, ""

    def log_metrics(self, metrics: Dict[str, Any], step: int,
                    skip_keys: Optional[frozenset[str]] = None) -> None:
        if self.writer is None:
            return
        try:
            for key, value in metrics.items():
                if key in _TB_SUPPRESSED or (skip_keys is not None and key in skip_keys):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                writer, base, suffix = self._writer_for(key)
                tag = _tb_tag(base)
                writer.add_scalar(tag, float(value), step)
                # The human counterpart for this population, re-emitted at the same step so the
                # flat line spans exactly the range the run covers rather than stopping at 0.
                human = reference_for(key)
                human_writer = self.side_writers.get(HUMAN_RUNS.get(suffix, ""))
                if human is not None and human_writer is not None:
                    human_writer.add_scalar(tag, float(human), step)
            self._log_multiline(metrics, step, skip_keys)
        except Exception as e:
            print(f"Warning: TensorBoard logging failed ({e}); disabling TensorBoard for the "
                  f"rest of the run.", flush=True)
            self.writer = None

    def _log_multiline(self, metrics: Dict[str, Any], step: int,
                       skip_keys: Optional[frozenset[str]]) -> None:
        """Charts combining different quantities, via add_scalars.

        A chart is emitted only when every one of its metric lines is present and not held
        back: a partial group would draw some lines and silently omit others, which reads as
        the missing ones being zero.
        """
        if self.writer is None:
            return
        for main_tag, lines in MULTILINE_CHARTS.items():
            values: Dict[str, float] = {}
            complete = True
            for label, source in lines.items():
                if isinstance(source, (int, float)):
                    values[label] = float(source)
                    continue
                if source not in metrics or (skip_keys is not None and source in skip_keys):
                    complete = False
                    break
                values[label] = float(metrics[source])
            if complete and values:
                self.writer.add_scalars(main_tag, values, step)

    def log_histogram(self, tag: str, values: Sequence[float], step: int) -> None:
        """A distribution rather than its mean -- the end-turn spread, chiefly.

        `endgame/turn` says games average 6.8; it cannot say whether that is most games ending
        near turn 7 or a mixture of turn-3 blowups and full-length games, which is the actual
        question when comparing against humans.
        """
        if self.writer is None or not len(values):
            return
        try:
            import numpy as np

            self.writer.add_histogram(tag, np.asarray(values, dtype=np.float32), step)
        except Exception:
            pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_text(tag, text, step)
        except Exception:
            pass

    def flush(self) -> None:
        for w in self.side_writers.values():
            try:
                w.flush()
            except Exception:
                pass
        if self.writer is None:
            return
        try:
            self.writer.flush()
        except Exception:
            pass

    def close(self) -> None:
        for w in self.side_writers.values():
            try:
                w.close()
            except Exception:
                pass
        self.side_writers = {}
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception:
            pass
        self.writer = None


def _episode_group_stats(episodes: List[Dict[str, Any]], suffix: str,
                         include_outcome: bool = True) -> Dict[str, float]:
    """Scalar summary of one group of finished episodes.

    `include_outcome` is False for the per-winner groups, where "who won" is the thing that
    defines the group: a ussr_win_rate of exactly 1.0 on every iteration is not a measurement.
    """
    stats: Dict[str, float] = {f"episodes_completed{suffix}": float(len(episodes))}
    turns = [float(ep["turn"]) for ep in episodes if "turn" in ep]
    stats[f"mean_turn{suffix}"] = float(np.mean(turns)) if turns else 0.0
    stats[f"median_turn{suffix}"] = float(np.median(turns)) if turns else 0.0
    # Length in plies: the finer, artefact-free companion to the turn counter. A turn
    # number cannot separate a game abandoned at turn 7 AR1 from one that ran to turn 7
    # AR7, and it reads 11 for a game that went the distance because finish_end_turn
    # increments before testing. 154 plies is a full game. See ai.game_length.
    plies = [float(ep["ply"]) for ep in episodes if "ply" in ep]
    stats[f"mean_ply{suffix}"] = float(np.mean(plies)) if plies else 0.0
    stats[f"median_ply{suffix}"] = float(np.median(plies)) if plies else 0.0

    # Which side won. Self-play carries a standing USSR imbalance that nothing was tracking
    # during a run. Free here: the episode records carry the winner.
    if include_outcome:
        # Which side won. Self-play carries a standing USSR imbalance that nothing was tracking
        # during a run. Free here: the episode records carry the
        # winner. Both sides are emitted rather than only the USSR: us_win_rate is derivable but
        # a dashboard should not make the reader do arithmetic to see the other half.
        winners = [str(ep.get("winner", "")) for ep in episodes]
        decided = [w for w in winners if w in ("US", "USSR")]
        stats[f"ussr_win_rate{suffix}"] = (
            float(sum(1 for w in decided if w == "USSR")) / float(len(decided)) if decided else 0.0)
        stats[f"us_win_rate{suffix}"] = (
            float(sum(1 for w in decided if w == "US")) / float(len(decided)) if decided else 0.0)
        stats[f"draw_rate{suffix}"] = (
            float(sum(1 for w in winners if w == "DRAW")) / float(len(winners)) if winners else 0.0)
        utils = [float(ep["terminal_utility"]) for ep in episodes if "terminal_utility" in ep]
        stats[f"mean_terminal_utility{suffix}"] = float(np.mean(utils)) if utils else 0.0

    # Final score, which the episode records have always carried and nothing displayed. It
    # separates a run that wins narrowly from one that wins by a mile, which the +/-1 terminal
    # utility cannot. US-positive, as everywhere else. ITS records no final score, so these
    # deliberately have no human reference line.
    vps = [float(ep["victory_points"]) for ep in episodes if "victory_points" in ep]
    stats[f"mean_victory_points{suffix}"] = float(np.mean(vps)) if vps else 0.0
    stats[f"mean_vp_margin{suffix}"] = float(np.mean(np.abs(vps))) if vps else 0.0

    reasons = [str(ep.get("ending_reason", "")) for ep in episodes]
    counted = [r for r in reasons if r]
    for key in ENDING_REASON_KEYS:
        stats[f"ending_frac_{key}{suffix}"] = (
            float(sum(1 for r in counted if r == key)) / float(len(counted)) if counted else 0.0
        )
    # DEFCON 1 as one number as well as split by whose decision caused it. The ITS results
    # database records the outcome without the cause, so the combined figure is the only one it
    # can be compared against -- see ai/itsc_reference.py.
    stats[f"ending_frac_defcon1{suffix}"] = (
        stats[f"ending_frac_defcon1_self{suffix}"] + stats[f"ending_frac_defcon1_provoked{suffix}"])
    return stats


def summarize_completed_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates the episodes that finished during one iteration into scalar metrics.

    Game length is reported at the game-turn granularity (mean and median terminal turn),
    and the ending-reason mix as a fraction of the episodes completed this iteration.

    With mid-game start sampling on, half the environments begin partway through a game, so a
    pooled mean turn or ending mix describes neither the real game nor the resumed one -- a run
    reads a mean turn of 8 while its turn-1 games still end at 6. The per-start-turn breakdown
    exists for that case and is emitted **only** in it.

    With the pool off, every episode starts at turn 1 and each suffixed series is an exact copy of
    its unsuffixed twin: a 320M run logged ten such copies for 1,220 iterations. Start sampling is
    settled negative (3.2, 3.3) and --start-pool-frac defaults to 0, so that is every ordinary run.
    """
    stats = _episode_group_stats(episodes, "")

    # Length and ending mix split by which side won. The two are not interchangeable in human
    # play and the difference is a real signal: in the ITS corpus a US win runs 8.62 turns and a
    # USSR win 8.00, the USSR takes half its wins on the VP track against the US's third, and the
    # US wins two thirds of the games that end at DEFCON 1. A pooled mix hides all of it.
    for side, tag in (("US", "_won_us"), ("USSR", "_won_ussr")):
        group = [ep for ep in episodes if str(ep.get("winner", "")) == side]
        stats.update(_episode_group_stats(group, tag, include_outcome=False))

    by_start: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for ep in episodes:
        by_start[int(ep.get("start_turn", 1))].append(ep)
    if len(by_start) > 1:
        for start_turn, group in by_start.items():
            stats.update(_episode_group_stats(group, f"_start{start_turn}"))
    return stats


#: How many decisions the per-epoch agreement pass reads. The self-play format
#: rebuilds its observations by replaying, so a full pass is minutes; this is enough
#: for a figure that is stable to a tenth of a point.
_AGREEMENT_SAMPLE = 20_000



class _HumanInjector:
    """Periodic supervised steps on human play, interleaved with RL.

    A behaviour-cloning warmup washes out early in RL: it is an initialisation, and RL walks away
    from it. This keeps the signal applied instead of applied once -- and applies it only on human
    positions, where the corpus actually has an opinion, which a KL term against a human policy
    would not: that would be evaluated on the states the RL policy visits, where a net trained on
    a few hundred games is extrapolating from nothing.

    `every` is in iterations. Small and often beats large and rare, because anything rarer than the
    washout it is fighting simply lets the policy drift back between doses.
    """

    def __init__(self, dataset_path: str, model: nn.Module, device: torch.device,
                 every: int, weight: float, batch_size: int = 512,
                 holdout_seed: int = 7, holdout_frac: float = 0.2) -> None:
        from ai.training.human_corpus_dataset import HumanCorpusDataset

        self.every = max(1, int(every))
        self.weight = float(weight)
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.ds = HumanCorpusDataset(dataset_path)
        self.opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.steps = 0

        # Injection trains on the same games the warm start did, and no others. Without this the
        # held-out games are fed to the policy a few thousand times over a long run, and the
        # agreement figure quietly stops measuring generalisation -- the same way the first
        # synth+human warm start read 65% "held out" because it had been fitted on all 280 games.
        game = np.asarray(self.ds._column("game"))
        games = np.unique(game)
        rng = np.random.default_rng(holdout_seed)
        rng.shuffle(games)
        held = set(games[:max(1, int(holdout_frac * len(games)))].tolist())
        self.train_idx = np.flatnonzero(
            np.fromiter((g not in held for g in game), dtype=bool, count=len(game)))
        self.held_out_games = len(held)

        self._obs = self.ds._column("obs")
        self._mask = self.ds._column("mask")
        self._act = self.ds._column("action")
        self._win = self.ds._column("win")
        self._vp = self.ds._column("vp")
        self._has = self.ds._column("has_outcome")
        self._rng = np.random.default_rng(20260907)

    def _next(self):
        idx = np.sort(self._rng.choice(self.train_idx, size=self.batch_size, replace=False))
        mask = np.unpackbits(np.asarray(self._mask[idx]), axis=1)[:, :ActionEncoder.FLAT_ACTION_SIZE]
        return (
            torch.from_numpy(np.asarray(self._obs[idx], dtype=np.float32)).to(self.device),
            torch.from_numpy(mask.astype(np.uint8)).to(self.device),
            torch.from_numpy(np.asarray(self._act[idx], dtype=np.int64)).to(self.device),
            torch.from_numpy(np.asarray(self._win[idx], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.asarray(self._vp[idx], dtype=np.float32)).to(self.device),
            torch.from_numpy(np.asarray(self._has[idx], dtype=np.float32)).to(self.device),
        )

    def maybe_step(self, iteration: int) -> float:
        if iteration % self.every:
            return 0.0
        b_obs, b_mask, b_act, b_val, b_vp, b_has = self._next()
        self.model.train()
        logits, v_win, v_vp = self.model(b_obs, b_mask)
        denom = b_has.sum().clamp(min=1.0)
        loss = self.weight * (
            F.cross_entropy(logits, b_act)
            + 0.5 * (((v_win.squeeze(-1) - b_val) ** 2 * b_has).sum() / denom)
            + 0.05 * (((v_vp.squeeze(-1) - b_vp) ** 2 * b_has).sum() / denom))
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()
        self.steps += 1
        return float(loss.item())


def run_search_distillation(
    model: nn.Module,
    dataset_path: str,
    output_checkpoint_path: str,
    epochs: int = 2,
    batch_size: int = 512,
    lr: float = 1e-4,
    max_games: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, float]:
    """P15-X4a step 3: pull the policy toward the searcher's visit distribution.

    Cross-entropy against a SOFT target -- the searcher's normalised visit counts -- on the
    decisions a searcher actually answered. Not the played action: the played action is the raw
    policy's own, and training on it is behaviour cloning of the thing being improved.

    **The value head is left alone.** No value term, because X4a asks one question -- is the
    search edge expressible as a policy? -- and a value loss would move a second thing. The trunk
    is shared, so the value head's *inputs* still shift; that is unavoidable without freezing the
    trunk, and it is why the run is rated rather than assumed.

    Small LR by default (1e-4 against BC's 1e-3): this starts from a trained policy and is meant
    to bend it, not retrain it.
    """
    dev = resolve_device(device)
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if os.path.isdir(dataset_path):
        raise ValueError(
            f"{dataset_path} is a directory. Search-target distillation reads a single "
            "jsonl.gz produced by tools/generate_search_targets.py.")

    print(f"=== Search-target distillation from {dataset_path} ===", flush=True)
    meta_path = dataset_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"    targets from {meta.get('searcher')} at commit "
              f"{str(meta.get('commit'))[:10]} (dirty={meta.get('git_dirty')})", flush=True)
    else:
        print("    WARNING: no sidecar metadata; the searcher's configuration is unrecorded",
              flush=True)

    stats = {"samples": 0.0, "final_loss": 0.0, "final_agreement": 0.0, "final_top1_kl": 0.0}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        seen = 0
        loss_sum = 0.0
        agree = 0
        for b_obs, b_mask, b_pi, _b_dt in WarmupDataset(dataset_path).stream_policy_batches(
                batch_size=batch_size, max_games=max_games, device=dev):
            logits, _v_win, _v_vp = model(b_obs, b_mask)
            logp = F.log_softmax(logits, dim=-1)
            # Soft cross-entropy. The target is zero outside the legal set, so masked logits
            # (-1e9) contribute nothing and cannot produce a NaN.
            loss = -(b_pi * logp).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n = b_obs.size(0)
            seen += n
            loss_sum += float(loss.item()) * n
            agree += int((logits.argmax(-1) == b_pi.argmax(-1)).sum().item())

        dt = max(1e-2, time.time() - t0)
        stats.update({"samples": float(seen),
                      "final_loss": loss_sum / max(1, seen),
                      "final_agreement": agree / max(1, seen)})
        print(f"  epoch {epoch}/{epochs} | {seen:,} targets ({seen/dt:.0f}/s) | "
              f"CE {stats['final_loss']:.4f} | top-1 agreement with the searcher "
              f"{stats['final_agreement']:.1%}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint_path)
    print(f"  wrote {output_checkpoint_path}", flush=True)
    return stats


def run_behavioral_cloning_warmup(
    model: nn.Module,
    dataset_path: str,
    output_checkpoint_path: str,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_games: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    dev = resolve_device(device)
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # A directory is the human corpus; a file is the self-play set. They differ in more than
    # storage: roughly half the human games stop mid-recording and have no outcome, so their
    # positions carry a policy target and no value target, and the batches say which.
    human = os.path.isdir(dataset_path)
    kind = "Human Corpus" if human else "Self-Play"
    print(f"=== Loading Warm-up Dataset ({kind}, streaming) from: {dataset_path} ===", flush=True)
    t0 = time.time()

    def _batches():
        if human:
            ds_h = HumanCorpusDataset(dataset_path)
            print(f"    {len(ds_h):,} samples, "
                  f"{ds_h.meta['samples_with_outcome']:,} with a value target "
                  f"({ds_h.meta['games']} games)", flush=True)
            for batch in ds_h.stream_batches(batch_size=batch_size, device=dev):
                yield batch
        else:
            for b in WarmupDataset(dataset_path).stream_batches(
                batch_size=batch_size, max_games=max_games, device=dev, shuffle_buffer_size=4096
            ):
                # Every self-play position has an outcome, so the value target always counts.
                yield (b[0], b[1], b[2], b[3], b[4], torch.ones_like(b[3]))

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        total_loss = 0.0
        correct_actions = 0
        samples_seen = 0

        for b_obs, b_mask, b_act, b_val, b_vp, b_has in _batches():
            logits, v_win, v_vp = model(b_obs, b_mask)
            policy_loss = F.cross_entropy(logits, b_act)
            # Masked, not dropped: a position from a game whose recording stopped still shows
            # what the human played, so it belongs in the policy loss and not the value loss.
            # Averaging over the whole batch instead would quietly scale the value term by
            # whatever fraction of it happened to be settled.
            denom = b_has.sum().clamp(min=1.0)
            val_win_loss = ((v_win.squeeze(-1) - b_val) ** 2 * b_has).sum() / denom
            val_vp_loss = ((v_vp.squeeze(-1) - b_vp) ** 2 * b_has).sum() / denom

            loss = policy_loss + 0.5 * val_win_loss + 0.05 * val_vp_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            correct_actions += (preds == b_act).sum().item()
            cur_b_size = b_obs.size(0)
            total_loss += loss.item() * cur_b_size
            samples_seen += cur_b_size

            if samples_seen % 100000 < batch_size:
                cur_l = total_loss / max(1, samples_seen)
                cur_acc = (correct_actions / max(1, samples_seen)) * 100.0
                dt = max(1e-2, time.time() - t_epoch)
                print(f"  Epoch {epoch:2d}/{epochs:2d} | {samples_seen:,} samples ({samples_seen/dt:.0f} samples/s) | Loss: {cur_l:.4f} | Acc: {cur_acc:.2f}%", flush=True)

        avg_loss = total_loss / max(1, samples_seen)
        acc = (correct_actions / max(1, samples_seen)) * 100.0
        dt = max(1e-2, time.time() - t_epoch)
        # The in-batch figure scores every point of a play against the exact index the
        # demonstration happened to record, which marks a reordered-but-identical placement wrong.
        # Agreement is the measure that does not: it scores a play on the multiset of countries,
        # excluding coups and realignments where a die between points makes the order real. It
        # needs an unshuffled pass, since a play's points have to stay together.
        agree = evaluate_dataset(model, dataset_path, dev, max_samples=_AGREEMENT_SAMPLE)
        model.train()
        print(f"  Epoch {epoch:2d}/{epochs:2d} COMPLETED in {dt:.1f}s | Loss: {avg_loss:.4f} | "
              f"Strict Acc: {acc:.2f}% | Agreement: {agree.unordered:.2f}% "
              f"(ordered {agree.ordered:.2f}%, {agree.decisions:,} decisions) | "
              f"Total Samples: {samples_seen:,}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint_path)
    print(f"=== Warm-up Complete in {time.time() - t0:.1f}s. Saved to: {output_checkpoint_path} ===", flush=True)


def evaluate_and_log_snapshot(
    model: nn.Module,
    opponents: List[PlayerAgent],
    elapsed_seconds: int,
    out_dir: str,
    report_path: str,
    games_per_side: int = 25,
    device: Optional[Union[torch.device, str]] = None,
    add_to_opponents_after: bool = True,
    arch: str = "v2",
    decisive_games: int = 128,
    position_games: int = 128,
    blunder_games: int = 32,
    num_baselines: int = 0,
    max_snapshot_opponents: int = 4,
    merged_influence: bool = False,
) -> Dict[str, float]:
    dev = resolve_device(device)
    snap_name = f"snapshot_{elapsed_seconds}s"
    # The model is evaluated in the action view it trains in (P23), probes included.
    current_agent = NeuralAgent(model=model, device=dev, name=snap_name,
                                merged_influence=merged_influence)

    # Decisive-decision rates. These move long before win rate does, because a forced win
    # or avoidable forced loss arises at well under 1% of decisions -- rare enough to be
    # invisible in aggregate results, decisive enough that each one is a whole game.
    decisive_metrics: Dict[str, float] = {}
    try:
        from ai.eval.decisive_probe import measure_decisive_batched
        # Batched: the single-state loop spends 96.9% of its time in the policy forward,
        # so handing the GPU one state at a time was the whole cost.
        stats = measure_decisive_batched(model, num_envs=decisive_games,
                                         merged_influence=merged_influence)
        decisive_metrics = stats.as_metrics()
        print(f"  decisive: takes {stats.win_take_rate * 100:.0f}% of {stats.win_available} forced wins | "
              f"avoids {stats.loss_avoid_rate * 100:.0f}% of {stats.loss_avoidable} avoidable losses",
              flush=True)
    except Exception as exc:
        print(f"  (decisive probe unavailable: {exc})", flush=True)

    # Position diagnostics. Win rate plateaus while the play underneath stays incoherent;
    # these read the board instead. empty_battlegrounds_turn8 is the sharpest -- roughly a
    # quarter of battlegrounds sit untouched from turn 8 on, always the same ones, and the
    # count stops falling rather than slowly improving.
    # Named mistakes, with their denominators. These ran only in tools/play_match.py before,
    # so nothing was tracking them during training -- which is where they matter, because a
    # policy can gain Elo while learning the mistakes better rather than fewer.
    # Run at both evaluation temperatures (see BLUNDER_PROBES): 0.1, which is how tournaments
    # and replays sample, and 1.0, which is the distribution the PPO ratio actually optimises.
    # The probe is the cheap part of a snapshot evaluation -- 1.6s of ~20s measured, against 9.4s
    # for the decisive probe -- so the second pass buys the comparison for well under a percent.
    blunder_metrics: Dict[str, float] = {}
    for _temperature, _prefix in BLUNDER_PROBES:
        try:
            from ai.eval.blunders import measure_blunders_batched

            counts = measure_blunders_batched(
                model, num_games=blunder_games, temperature=_temperature,
                merged_influence=merged_influence)
            blunder_metrics.update(counts.metrics(prefix=_prefix))
            print(f"  blunders (tau={_temperature}):\n" + counts.summary(), flush=True)
        except Exception as e:
            print(f"  blunder probe at tau={_temperature} failed ({e}); continuing", flush=True)

    position_metrics: Dict[str, float] = {}
    try:
        from ai.eval.position_diagnostics import format_report, profile_self_play_batched
        # Batched: ~106k decisions/sec against ~890 for the one-state-at-a-time loop, so
        # this costs seconds rather than minutes of every snapshot evaluation.
        profile = profile_self_play_batched(model, num_envs=position_games,
                                            merged_influence=merged_influence)
        position_metrics = profile["scalars"]
        print(f"  positions: {profile['scalars']['diag/empty_battlegrounds_turn8']:.1f} empty / "
              f"{profile['scalars']['diag/uncontrolled_battlegrounds_turn8']:.1f} uncontrolled "
              f"battlegrounds at turn 8 | "
              f"{100 * profile['scalars']['diag/salvageable_frac_turn6']:.0f}% salvageable at turn 6",
              flush=True)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n<details><summary>Positions @ {elapsed_seconds}s</summary>\n\n```\n"
                    f"{format_report(profile)}\n```\n</details>\n\n")
    except Exception as exc:
        print(f"  (position diagnostics unavailable: {exc})", flush=True)

    if games_per_side <= 0:
        # A zero budget means "do not evaluate", which is a reasonable thing to ask for on a
        # smoke run. It used to reach BatchMatchRunner and divide by a chunk size of zero.
        print(f"\n--- Skipping snapshot evaluation @ {elapsed_seconds}s "
              f"(--eval-games-per-side is {games_per_side}) ---", flush=True)
        return {}

    print(f"\n--- Evaluating Newest Snapshot @ {elapsed_seconds}s against {len(opponents)} Opponents ({games_per_side*2} games each) ---", flush=True)
    report_entry = [f"### Snapshot @ {elapsed_seconds}s (Evaluated against {len(opponents)} baselines / past snapshots)\n\n"]
    report_entry.append("| Opponent | Overall Win Rate | As US Win Rate | As USSR Win Rate | Top Loss Causes (US) | Top Loss Causes (USSR) |\n")
    report_entry.append("|:---|:---:|:---:|:---:|:---|:---|\n")

    # Win rate against each fixed baseline, which is the curve this whole evaluation exists to
    # produce and which used to reach only the markdown table. Keyed by opponent name, so only the
    # baselines qualify: the snapshot opponents are renamed every interval and would each start a
    # new series that stops one interval later.
    eval_metrics: Dict[str, float] = {}

    for idx, opp in enumerate(opponents):
        res = BatchMatchRunner.play_parallel_matchup(
            current_agent, opp, games_per_side=games_per_side, device=dev, temperature=0.1)
        wr_tot = res["win_rate_a"] * 100.0
        wr_us = res["win_rate_a_as_us"] * 100.0
        wr_ussr = res["win_rate_a_as_ussr"] * 100.0

        top_us = ", ".join([f"{k} ({v})" for k, v in sorted(res["causes_loss_us"].items(), key=lambda x: x[1], reverse=True)]) or "None (0 losses)"
        top_ussr = ", ".join([f"{k} ({v})" for k, v in sorted(res["causes_loss_ussr"].items(), key=lambda x: x[1], reverse=True)]) or "None (0 losses)"

        print(f"  vs {opp.name:<25s} -> Overall: {wr_tot:5.1f}% ({res['a_wins']}W-{res['b_wins']}L) | US: {wr_us:5.1f}% ({res['a_wins_as_us']}W-{res['a_losses_as_us']}L) | USSR: {wr_ussr:5.1f}% ({res['a_wins_as_ussr']}W-{res['a_losses_as_ussr']}L)", flush=True)
        print(f"       Losses as US:   {top_us}", flush=True)
        print(f"       Losses as USSR: {top_ussr}", flush=True)

        if idx < num_baselines:
            eval_metrics[f"eval_win_rate_{opp.name}"] = float(res["win_rate_a"])
            eval_metrics[f"eval_win_rate_{opp.name}_as_us"] = float(res["win_rate_a_as_us"])
            eval_metrics[f"eval_win_rate_{opp.name}_as_ussr"] = float(res["win_rate_a_as_ussr"])

        report_entry.append(f"| **{opp.name}** | **{wr_tot:.1f}%** ({res['a_wins']}W-{res['b_wins']}L) | {wr_us:.1f}% ({res['a_wins_as_us']}W-{res['a_losses_as_us']}L) | {wr_ussr:.1f}% ({res['a_wins_as_ussr']}W-{res['a_losses_as_ussr']}L) | {top_us} | {top_ussr} |\n")

    report_entry.append("\n---\n\n")
    print("-" * 80 + "\n", flush=True)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("".join(report_entry))

    if add_to_opponents_after:
        if arch == "v1":
            frozen_net = create_coldwar_net(dev)
        else:
            # Shaped from the model being evaluated, not from the factory defaults. The card
            # block width and whether the history branch exists are both configurable now, and a
            # frozen copy built at defaults simply fails to load a configured policy -- and it
            # fails at the first snapshot rather than at startup, hours in.
            # Every dimension read off the model. Listing them by hand has failed twice.
            #
            # This was an ALLOW-LIST -- `arch in ("v2", "mlp")` -- and adding `ladder` to the CLI
            # without adding it here sent a 3.18M-parameter ladder model down the v1 branch,
            # which built a 512-wide trunk for a 480-wide checkpoint and died at the first
            # snapshot: exactly the failure the paragraph above describes, for the third time.
            # Inverted, so a new architecture gets the model-derived copy by default and only
            # the legacy v1 backbone is special-cased.
            frozen_net = create_like(model, dev)
        frozen_net.load_state_dict(model.state_dict())
        frozen_net.to(dev)
        frozen_net.eval()
        opponents.append(NeuralAgent(model=frozen_net, device=dev, name=f"Snapshot_{elapsed_seconds}s"))
        # Evaluate against the baselines plus only the most recent snapshots. The opponent
        # list otherwise grows by one every interval, so eval cost is quadratic in run
        # length: the last evaluation of a 3-hour run faced 14 opponents and took 957s
        # against a 900s snapshot interval, which starved training entirely. Older
        # snapshots are also the least informative comparison for a policy that has moved
        # well past them.
        if max_snapshot_opponents > 0:
            excess = len(opponents) - num_baselines - max_snapshot_opponents
            if excess > 0:
                del opponents[num_baselines:num_baselines + excess]

    return {**decisive_metrics, **position_metrics, **blunder_metrics, **eval_metrics}


def run_post_training_tournament(
    checkpoint_dir: str,
    additional_models: Optional[List[str]] = None,
    games_per_side: int = 500,
    device: Optional[Union[torch.device, str]] = None,
) -> None:
    from tools.tournament import run_massive_tournament
    from tools.lib.checkpoint_utils import discover_checkpoints

    dev = resolve_device(device)
    print("\n" + "=" * 80, flush=True)
    print(" LAUNCHING POST-TRAINING MASSIVE TOURNAMENT BENCHMARK", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Snapshots only. `discover_checkpoints` lists every .pt, which is right for
    # `tools/inspect_checkpoints.py` but wrong here: a run directory also holds `resume_*.pt` and
    # `resume_state.pt`, which carry optimizer and numpy RNG state rather than a policy. Those
    # cannot be loaded under `weights_only=True`, so entering one as a tournament model raises
    # UnpicklingError and takes the whole post-training tournament down with it -- which is how
    # E4-02-01 lost its tournament after completing all 240M steps.
    ckpts = discover_checkpoints(checkpoint_dir)
    models_to_evaluate = [c["path"] for c in ckpts
                          if os.path.basename(c["path"]).startswith("snapshot_")]
    if not models_to_evaluate:
        print("  no snapshot_*.pt in the run directory; skipping the tournament", flush=True)
        return

    baselines = additional_models or ["heuristic", "random"]
    for b in baselines:
        if b not in models_to_evaluate:
            models_to_evaluate.append(b)

    report_out = os.path.join(checkpoint_dir, "final_tournament_report.md")
    json_out = os.path.join(checkpoint_dir, "final_tournament_results.json")

    run_massive_tournament(
        model_specs=models_to_evaluate,
        games_per_side=games_per_side,
        device=str(dev),
        anchor_model="HeuristicBot",
        anchor_elo=1500.0,
        output_report=report_out,
        output_json=json_out,
    )



#: Written beside the snapshots and overwritten each time one is taken. Separate from
#: `snapshot_*.pt` on purpose: those are bare state dicts that every evaluation tool, the
#: tournament runner and `load_agent` all read directly, and widening them into a dict of
#: dicts would break each of those readers silently.
RESUME_FILENAME = "resume_state.pt"

#: Per-snapshot resume states sit beside their snapshot under this name, so any snapshot can be
#: branched from and not only the run's end. resume_state.pt remains the newest one, which is what
#: an ordinary continuation wants and what --resume <dir> resolves to.
RESUME_AT_STEPS = "resume_{steps}steps.pt"


def resume_states_in(run_dir: str) -> Dict[int, str]:
    """Step count -> path, for every branch point a run directory offers."""
    found: Dict[int, str] = {}
    try:
        names = os.listdir(run_dir)
    except OSError:
        return found
    for name in names:
        m = re.fullmatch(r"resume_(\d+)steps\.pt", name)
        if m:
            found[int(m.group(1))] = os.path.join(run_dir, name)
    return found


def resolve_resume(resume: str) -> str:
    """The resume state a --resume argument names.

    Accepts a file, a run directory (its newest state), or `<run_dir>:<steps>` / `<run_dir>@<steps>`
    to branch from a particular snapshot. A step count that does not exist is an error listing what
    the directory has, rather than a silent fall back to the newest -- resuming from the wrong point
    produces a run that looks entirely normal and answers a different question.
    """
    for sep in (":", "@"):
        if sep in resume:
            head, _, tail = resume.rpartition(sep)
            if head and tail.isdigit():
                want = int(tail)
                states = resume_states_in(head)
                if want not in states:
                    have = ", ".join(f"{k:,}" for k in sorted(states)) or "none"
                    raise FileNotFoundError(
                        f"{head} has no resume state at {want:,} steps. Available: {have}. "
                        f"(Per-snapshot states are only written by runs since they were added; "
                        f"an older run has just its final {RESUME_FILENAME}.)")
                return states[want]
    if os.path.isfile(resume):
        return resume
    return os.path.join(resume, RESUME_FILENAME)


def save_resume_state(path: str, model: nn.Module, trainer: Any, iteration: int,
                      total_env_steps: int, elapsed_seconds: float,
                      seed: Optional[int] = None) -> None:
    """Everything needed to pick a run back up, except the environment.

    The environment is deliberately not saved: `VectorizedBatchRunner` holds 512 live games and
    has no serialisation, and restoring it is not worth building because it does not matter --
    the games are an i.i.d. stream, so continuing with fresh deals is the same experiment. What
    does matter is the optimiser moments, the reference policy, and the step counter, because
    those are what make a resumed run continue rather than restart.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "reference_state_dict": trainer.reference_net.state_dict(),
        "total_env_steps": int(total_env_steps),
        "total_iterations": int(getattr(trainer, "total_iterations", 0)),
        "iteration": int(iteration),
        "elapsed_seconds": float(elapsed_seconds),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        # Recorded so a later resume can tell "continue this run" from "branch it": restoring the
        # RNG is right for the first and wrong for the second.
        "seed": None if seed is None else int(seed),
        # Which snapshots the opponent pool held and how each has fared, but not their weights --
        # those are the snapshot files already beside this state. Carrying it makes a resumed pool
        # the pool the run actually had, rather than a plausible reconstruction of one. Losing it
        # is a real change to the experiment, so it should be asked for (--reset-opponent-pool),
        # not suffered.
        "opponent_pool": (trainer.opponent_pool.state_dict()
                          if getattr(trainer, "opponent_pool", None) is not None else None),
        # --wolf-seat-weight's running self-play share; a resumed run would otherwise restart it
        # at an even split and weight both seats equally until it relearned the imbalance.
        "wolf_sp_ussr": float(getattr(trainer, "wolf_sp_ussr", 0.5)),
    }, path)


def load_resume_state(path: str, model: nn.Module, trainer: Any,
                      seed: Optional[int] = None) -> Dict[str, Any]:
    """Restore a run in place and return where it left off.

    `seed` is the seed the *resuming* run was given. Where it differs from the one the state was
    written under, the restored RNG is replaced by it: the caller asked for a different stream, and
    a restore would otherwise hand back the original run's action sampling and minibatch order,
    leaving only the environment deals to differ. Where it matches, or where neither is given, the
    RNG is restored and the continuation is the run it would have been.

    A state written before the seed was recorded has none, and an explicit `seed` then wins --
    honouring the flag is less surprising than silently ignoring it.
    """
    blob = torch.load(path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    trainer.optimizer.load_state_dict(blob["optimizer_state_dict"])
    trainer.reference_net.load_state_dict(blob["reference_state_dict"])
    trainer.total_env_steps = int(blob["total_env_steps"])
    if hasattr(trainer, "total_iterations"):
        trainer.total_iterations = int(blob.get("total_iterations", 0))
    if hasattr(trainer, "wolf_sp_ussr"):
        trainer.wolf_sp_ussr = float(blob.get("wolf_sp_ussr", 0.5))
    recorded_seed = blob.get("seed", None)
    reseed = seed is not None and (recorded_seed is None or int(recorded_seed) != int(seed))
    if reseed:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)
        print(f"Reseeded to {seed} (state was written under {recorded_seed}); the RNG in the "
              f"resume file is deliberately not restored, so this continuation diverges.",
              flush=True)
    else:
        try:
            torch.set_rng_state(blob["torch_rng_state"].cpu().to(torch.uint8))
            np.random.set_state(blob["numpy_rng_state"])
        except Exception as exc:                  # a resumed run is still valid without these
            print(f"Warning: could not restore RNG state ({exc}); continuing with the current one.",
                  flush=True)
    return {
        "iteration": int(blob.get("iteration", 0)),
        "total_env_steps": int(blob["total_env_steps"]),
        "elapsed_seconds": float(blob.get("elapsed_seconds", 0.0)),
        # None for a state written before pools were carried, or for a run that had no pool.
        "opponent_pool": blob.get("opponent_pool"),
    }


def train_pipeline(
    arch: str = "v2",
    seed: Optional[int] = None,
    resume: Optional[str] = None,
    resume_every_snapshot: bool = True,
    resume_every_steps: int = 40_000_000,
    warmup_checkpoint: Optional[str] = None,
    warmup_dataset: Optional[str] = None,
    inject_dataset: Optional[str] = None,
    inject_every: int = 0,
    inject_weight: float = 1.0,
    bc_epochs: int = 5,
    train_steps: int = 80_000_000,
    decisiveness_turns: float = 0.0,
    max_snapshot_opponents: int = 4,
    snapshot_every_steps: int = 5_000_000,
    eval_opponents: Optional[List[str]] = None,
    eval_games_per_side: int = 50,
    num_envs: int = 512,
    buffer_size: int = 128,
    batch_size: int = 4096,
    lr: float = 3e-4,
    eta: float = 0.1,
    vf_coef: float = 0.5,
    value_dist_coef: float = 0.02,
    categorical_value: bool = False,
    adv_filter_quantile: float = 0.0,
    window_provoked_defcon: bool = False,
    identity_dim: int = 0,
    self_transform: bool = False,
    attn_readout: int = 0,
    per_entity_heads: int = 0,
    graph_layers: int = 2,
    drop_static: bool = False,
    ladder_config: Optional[Dict[str, Any]] = None,
    seed_init: Optional[int] = None,
    seed_sampling: Optional[int] = None,
    seed_env: Optional[int] = None,
    seed_pool: Optional[int] = None,
    entropy_coef: float = 0.01,
    reward_scheme: str = "blunder_aware",
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
    post_tournament: bool = False,
    post_tournament_models: Optional[List[str]] = None,
    post_tournament_games: int = 500,
    curriculum_switch_steps: Optional[int] = None,
    curriculum_switch_fraction: float = 0.5,
    slice_turn_boundaries: Optional[bool] = None,
    same_perspective_bootstrap: bool = False,
    per_player_gae: bool = False,
    blunder_window: bool = True,
    gamma: float = 1.0,
    priority_alpha: float = 0.0,
    defcon_coef: float = 0.0,
    opponent_checkpoints: Optional[List[str]] = None,
    opponent_frac: float = 0.0,
    opponent_lock_side: Optional[str] = None,
    opponent_self_pool: bool = False,
    opponent_pool_size: int = 12,
    reset_opponent_pool: bool = False,
    setup_explore_frac: float = 0.0,
    rollout_temps: Optional[List[float]] = None,
    search_ce_coef: float = 0.0,
    search_sims: int = 32,
    search_subsample: float = 0.125,
    search_node_filter: str = "card_playmode",
    opponent_pfsp: bool = False,
    opponent_pfsp_weighting: str = "var",
    opponent_pfsp_uniform_mix: float = 0.25,
    seat_balance: bool = False,
    seat_balance_max_frac: float = 0.8,
    per_seat_adv_norm: bool = False,
    wolf_seat_weight: bool = False,
    wolf_power: float = 1.0,
    wolf_ema_games: float = 2000.0,
    start_pool_frac: float = 0.0,
    start_pool_capacity: int = 512,
    start_pool_episodes: int = 600,
    ref_update_freq: int = 200_000,
    gae_lambda: float = 0.98,
    merged_influence: bool = False,
    tensorboard: bool = True,
) -> None:
    # Checked first, before a device is resolved or a directory is made: a run whose budget is
    # nonsense should fail having built nothing. Both are in env steps and there is no time flag
    # to fall back to, which is the point -- the snapshot cadence used to be derived from
    # --duration-seconds and --snapshot-interval-seconds, and E3-22-28's first attempt thereby
    # snapshotted every 26.7M steps against its baseline's ~5M. Snapshots feed the self-play
    # opponent pool, so that quietly made it a two-factor experiment and it was discarded.
    if int(train_steps) <= 0:
        raise ValueError(
            f"train_steps must be positive, got {train_steps}. A run is budgeted in env steps; "
            "there is no wall-clock budget, because steps/sec depends on the policy and a time "
            "budget gives two arms different amounts of training.")
    if int(snapshot_every_steps) <= 0:
        raise ValueError(
            f"snapshot_every_steps must be positive, got {snapshot_every_steps}. It sets both the "
            "snapshot cadence and the rate the self-play opponent pool grows, so two arms that "
            "differ in it are not a one-factor comparison.")

    dev = resolve_device(device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Seeding both the environment stream and torch, or neither. Left None the run behaves as
    # it always has: a fixed environment seed and an unseeded network, which means two runs of
    # the same configuration differ only in initialisation and sampling. That understates
    # run-to-run variance, because every run sees the same deals and dice -- so a variance
    # estimate has to vary this, and an arm that wants to be paired with another has to share it.
    # Each source defaults to `seed`, so behaviour is unchanged unless an override is given.
    # They are separable because a single seed makes "this seed collapses" unattributable: the
    # same number picks the starting weights, the rollout sampling, the card deals and dice, and
    # the opponent draw.
    _seed_init = seed if seed_init is None else int(seed_init)
    _seed_sampling = seed if seed_sampling is None else int(seed_sampling)
    _seed_env = seed if seed_env is None else int(seed_env)
    _seed_pool = seed if seed_pool is None else int(seed_pool)
    if _seed_init is not None:
        torch.manual_seed(int(_seed_init))
        np.random.seed(int(_seed_init) & 0xFFFFFFFF)
    env_base_seed = 12345 if _seed_env is None else int(_seed_env)

    # The run's short name goes in the directory name, not only in metadata. A free-form
    # directory name does not say which engine the run was trained on or which seed it used --
    # and that is how a set of cross-engine comparisons came to be written up as same-engine
    # ones.
    out_dir = _resolve_run_dir(output_dir, run_name, arch, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    # This run's own PID, so a watcher can tell THIS run from any other training process on the
    # box. tools/scripts/watch_run.py otherwise matches `pgrep -f tools/train.py`, which is true
    # whenever any run is alive: a dead run then reports STALL instead of CRASH, which is the
    # confusion that script exists to prevent. Written before anything can fail, and left behind
    # deliberately -- a stale pid file whose process is gone reads as "dead", which is correct.
    try:
        with open(os.path.join(out_dir, "run.pid"), "w", encoding="utf-8") as _pf:
            _pf.write(f"{os.getpid()}\n")
    except OSError:
        pass  # a watcher that cannot read it falls back to the pattern match

    log_path = os.path.join(out_dir, "training_metrics.jsonl")
    report_path = os.path.join(out_dir, "tournament_report.md")
    tb = TensorBoardLogger(os.path.join(out_dir, "tb"), enabled=tensorboard)

    # Write metadata.json recording git commit, training mode, and description
    git_commit = "unknown"
    git_message = "unknown"
    git_dirty = False
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True).strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_dirty = bool(status_out)
    except Exception:
        pass

    # P23 / E4.1: the step from which this run decides in the merged-influence view. Recorded so
    # every later consumer can tell, per snapshot, which view the weights were trained in
    # (tools/lib/action_view.py). An E4 lineage continued as E4.1 switches at its resume step;
    # an E4.1 run resumed as E4.1 keeps its source's switch step.
    merged_from_step = 0
    if resume:
        _res_file = resolve_resume(resume)
        _src_merged, _src_from = run_view(os.path.dirname(os.path.abspath(_res_file)))
        if _src_merged and not merged_influence:
            raise ValueError(
                "resuming an E4.1 (--merged-influence) run without --merged-influence is not "
                "supported: its composed actions would be absent from every mask it is shown")
        if merged_influence:
            if _src_merged:
                merged_from_step = _src_from
            else:
                _rb = torch.load(_res_file, map_location="cpu", weights_only=False)
                merged_from_step = int(_rb["total_env_steps"])
                del _rb

    metadata_path = os.path.join(out_dir, "metadata.json")
    metadata_info = {
        "run_id": os.path.basename(out_dir),
        # The run's registered short name. None for a run launched before the field existed, or
        # launched without it -- which is itself worth being able to see.
        "run_name": run_name,
        "arch": arch,
        # Recorded, not chosen. There is one observation layout; the field stays so a run's
        # metadata still says which, and so older runs stay readable beside newer ones.
        "obs_layout": OBS_LAYOUT_NAME,
        "seed": seed,
        # The ladder's whole architecture lives here and nowhere else in metadata: `arch` says
        # only "ladder", and every structural axis -- input mode, aggregation, widths, which
        # entities get a per-entity head -- is carried in this dict. Without it
        # `tools/scripts/launch_flags.py --diff` is blind to exactly the kind of drift invariant
        # 15 exists to catch, which is how 34 M2d arms were launched with no record of the shape
        # they trained. None for non-ladder runs.
        "ladder_config": (dict(ladder_config) if ladder_config else None),
        # Each defaults to `seed`; recorded resolved so a run says which streams it actually used.
        "seed_init": (seed if seed_init is None else int(seed_init)),
        "seed_sampling": (seed if seed_sampling is None else int(seed_sampling)),
        "seed_env": (seed if seed_env is None else int(seed_env)),
        "seed_pool": (seed if seed_pool is None else int(seed_pool)),
        "resumed_from": resume,
        "base_commit": git_commit,
        "commit_message": git_message,
        "git_dirty": git_dirty,
        "training_mode": reward_scheme,
        "reward_scheme": reward_scheme,
        "same_perspective_bootstrap": same_perspective_bootstrap,
        "per_player_gae": per_player_gae,
        "resume_every_snapshot": bool(resume_every_snapshot),
        "resume_every_steps": int(resume_every_steps),
        "decisiveness_turns": decisiveness_turns,
        "train_steps": int(train_steps),
        "snapshot_every_steps": int(snapshot_every_steps),
        "search_ce_coef": float(search_ce_coef),
        "search_sims": int(search_sims),
        "search_subsample": float(search_subsample),
        "search_node_filter": search_node_filter,
        "opponent_pfsp": bool(opponent_pfsp),
        "opponent_pfsp_weighting": opponent_pfsp_weighting,
        "opponent_pfsp_uniform_mix": float(opponent_pfsp_uniform_mix),
        "curriculum_switch_steps": (None if curriculum_switch_steps is None
                                    else int(curriculum_switch_steps)),
        "num_envs": num_envs,
        # The hyperparameters that distinguish one run from another. Without these an ablation is
        # indistinguishable from its control in the record -- an `--eta 0` run once wrote metadata
        # identical to its control's apart from the free-text description, and prose is not
        # something a later query can filter on.
        "train_steps": train_steps,
        "eta": eta,
        "vf_coef": vf_coef,
        "value_dist_coef": value_dist_coef,
        "categorical_value": bool(categorical_value),
        "adv_filter_quantile": adv_filter_quantile,
        "window_provoked_defcon": bool(window_provoked_defcon),
        "identity_dim": int(identity_dim),
        "self_transform": bool(self_transform),
        "attn_readout": int(attn_readout),
        "per_entity_heads": int(per_entity_heads),
        "graph_layers": int(graph_layers),
        "drop_static": bool(drop_static),
        # The opponent-pool configuration: the difference between a self-play run and one trained
        # against its own history, which is otherwise recorded nowhere. `opponent_frac` is the
        # operative one -- a pool with frac 0 is not in use however it is configured.
        "opponent_frac": float(opponent_frac),
        "opponent_self_pool": bool(opponent_self_pool),
        "opponent_pool_size": int(opponent_pool_size),
        "reset_opponent_pool": bool(reset_opponent_pool),
        "setup_explore_frac": float(setup_explore_frac),
        "rollout_temps": list(rollout_temps) if rollout_temps else None,
        "opponent_checkpoints": list(opponent_checkpoints or []),
        # Recorded because they change what the arm IS, and an unrecorded flag is how a
        # two-factor experiment stays invisible -- snapshot_every_steps was missing for
        # exactly this reason and cost a 47M-step run. The side lock decides which seat
        # ever receives a gradient; the warm start decides what the run began from; the
        # eval opponents decide whether the in-run trace can rate the arm at all.
        "opponent_lock_side": opponent_lock_side,
        "warmup_checkpoint": warmup_checkpoint,
        "eval_opponents": list(eval_opponents or []),
        "ent_coef": entropy_coef,
        "ref_update_freq": ref_update_freq,
        "seat_balance": bool(seat_balance),
        "seat_balance_max_frac": float(seat_balance_max_frac),
        "per_seat_adv_norm": bool(per_seat_adv_norm),
        "wolf_seat_weight": bool(wolf_seat_weight),
        "wolf_power": float(wolf_power),
        "wolf_ema_games": float(wolf_ema_games),
        "gae_lambda": gae_lambda,
        "merged_influence": bool(merged_influence),
        "merged_influence_from_step": int(merged_from_step),
        "description": description or f"Self-play RL training with arch={arch}, reward={reward_scheme}, budget={train_steps:,} steps.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2)
    tb.log_text("run/metadata", "```json\n" + json.dumps(metadata_info, indent=2) + "\n```", 0)

    # 1. Initialize Model
    if arch == "ladder":
        # The whole structure is in `ladder_config`, which train.py refuses to build unless
        # every axis was named. Nothing here supplies a default.
        from ai.models.ladder_net import create_ladder_net
        if not ladder_config:
            raise ValueError("--arch ladder requires a ladder configuration; see "
                             "research/plans/P21_architecture_ladder.md")
        model = create_ladder_net(dev, categorical_value=categorical_value, **ladder_config)
    elif arch == "mlp":
        from ai.models.coldwar_net_v2 import create_coldwar_net_mlp
        model = create_coldwar_net_mlp(dev, categorical_value=categorical_value,
                                       drop_static=drop_static)
    elif arch == "v2":
        model = create_coldwar_net_v2(dev, categorical_value=categorical_value,
                                      identity_dim=identity_dim,
                                      self_transform=self_transform,
                                      attn_readout=attn_readout,
                                      per_entity_heads=per_entity_heads,
                                      graph_layers=graph_layers)
    else:
        model = create_coldwar_net(dev)

    # Everything above consumed the INITIALISATION stream; everything below -- action sampling
    # via torch.multinomial, minibatch shuffling via torch.randperm -- draws from here. Reseeding
    # at exactly this point is what separates "unlucky starting weights" from "unlucky rollouts".
    if _seed_sampling is not None and _seed_sampling != _seed_init:
        torch.manual_seed(int(_seed_sampling))
        print(f"[seed] sampling stream reseeded to {_seed_sampling} after initialisation "
              f"(init used {_seed_init})", flush=True)

    # 2. Handle Warm-up
    if warmup_checkpoint and os.path.exists(warmup_checkpoint):
        print(f"Loading Warm-up Checkpoint from: {warmup_checkpoint}", flush=True)
        from tools.lib.player_agent import load_checkpoint_into
        load_checkpoint_into(model, torch.load(warmup_checkpoint, map_location=dev, weights_only=True))
        model.to(dev)
    elif warmup_dataset and os.path.exists(warmup_dataset):
        warmup_save_path = os.path.join(out_dir, f"coldwar_net_{arch}_warmup.pt")
        run_behavioral_cloning_warmup(
            model=model,
            dataset_path=warmup_dataset,
            output_checkpoint_path=warmup_save_path,
            epochs=bc_epochs,
            device=dev,
        )
    else:
        print("No warmup checkpoint or dataset specified. Starting from fresh weights.", flush=True)

    # 3. Setup Reward Calculator & Vectorized Env
    is_curriculum = (reward_scheme == "curriculum")
    if is_curriculum or reward_scheme == "useful_actions":
        reward_calc = UsefulActionsReward()
    elif reward_scheme == "blunder_aware":
        reward_calc = BlunderAwareRewardCalculator(decisiveness_turns=decisiveness_turns)
    elif reward_scheme == "shaped":
        reward_calc = ShapedZeroSumReward()
    else:
        reward_calc = ZeroSumTerminalReward()
    # Mid-game start positions. Self-play from turn 1 reaches the late game rarely and
    # plays it badly, so a share of environments resume from saved turn-boundary positions
    # instead. The pool is rebuilt as the policy moves on; see ai/training/start_pool.py.
    start_pool: Optional[StartPositionPool] = None
    env_start_turns: List[Optional[int]] = [None] * num_envs
    if start_pool_frac > 0.0:
        mix = dict(DEFAULT_TURN_MIX)
        pool_turns = tuple(t for t in mix if t != 1)
        start_pool = StartPositionPool(turns=pool_turns,
                                       capacity_per_turn=start_pool_capacity)

        def _start_provider(env_idx: int) -> Optional[Any]:
            turn = env_start_turns[env_idx]
            if turn is None or start_pool is None:
                return None
            return start_pool.sample(turn)

        env = TsVectorizedEnv(num_envs=num_envs, base_seed=env_base_seed,
                              reward_calculator=reward_calc,
                              start_provider=_start_provider,
                              window_provoked_defcon=window_provoked_defcon)
    else:
        env = TsVectorizedEnv(num_envs=num_envs, base_seed=env_base_seed,
                              reward_calculator=reward_calc,
                              window_provoked_defcon=window_provoked_defcon)

    # Curriculum timing configuration
    if is_curriculum:
        if curriculum_switch_steps is not None:
            curriculum_switch_at = float(curriculum_switch_steps)
        else:
            curriculum_switch_at = float(train_steps) * float(curriculum_switch_fraction)
        curriculum_switched = False
        print(f"[CURRICULUM] Active: Stage 1 = UsefulActionsReward (first "
              f"{curriculum_switch_at:,.0f} steps), Stage 2 = BlunderAwareRewardCalculator",
              flush=True)
    else:
        curriculum_switch_at = float("inf")
        curriculum_switched = False

    # 4. Instantiate the NashPG trainer
    trainer = NashPGTrainer(
        active_net=model,
        env=env,
        num_envs=num_envs,
        buffer_size=buffer_size,
        batch_size=batch_size,
        lr=lr,
        eta=eta,
        vf_coef=vf_coef,
        value_dist_coef=value_dist_coef,
        adv_filter_quantile=adv_filter_quantile,
        ent_coef=entropy_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        num_epochs=4,
        ref_update_freq=ref_update_freq,
        max_grad_norm=1.0,
        # "auto" (None) means: rely on per-episode blunder windowing, not the blunt
        # global truncation. Global slicing measurably degrades the policy -- it strips
        # the outcome signal from clean wins too -- so it is opt-in for ablations only.
        slice_turn_boundaries=(False if slice_turn_boundaries is None else slice_turn_boundaries),
        blunder_window=blunder_window,
        same_perspective_bootstrap=same_perspective_bootstrap,
        per_player_gae=per_player_gae,
        search_ce_coef=search_ce_coef,
        search_sims=search_sims,
        search_subsample=search_subsample,
        search_node_filter=search_node_filter,
        priority_alpha=priority_alpha,
        defcon_coef=defcon_coef,
        temperature_schedule=True,
        setup_explore_frac=setup_explore_frac,
        rollout_temps=rollout_temps,
        merged_influence=merged_influence,
        per_seat_adv_norm=per_seat_adv_norm,
        wolf_seat_weight=wolf_seat_weight,
        wolf_power=wolf_power,
        wolf_ema_games=wolf_ema_games,
        device=dev,
    )

    # Frozen-opponent sampling. A share of environments plays the learner against a past
    # snapshot instead of against itself, so the outcome depends on the learner's actions
    # again and the advantage signal has something to be non-zero about. See
    # ai/training/opponent_pool.py and research/plans/P10_opponent_sampling.md.
    if opponent_frac > 0.0 and (opponent_checkpoints or opponent_self_pool):
        from ai.training.opponent_pool import OpponentPool, load_pool

        _lock = {"us": 1, "ussr": -1, None: None}[opponent_lock_side]
        # Defined for both branches: a fixed-checkpoint pool has no recorded state to restore,
        # and the code after the branch reads these unconditionally.
        _saved_pool: Optional[Dict[str, Any]] = None
        _seed_steps: List[int] = []
        if opponent_checkpoints:
            _seed_nets = load_pool(opponent_checkpoints, dev)
            _seed_merged = [checkpoint_merged_influence(p) for p in opponent_checkpoints]
            _src = f"{len(opponent_checkpoints)} fixed snapshot(s)"
        else:
            # On RESUME, rebuild the pool from the snapshots this run already wrote. The resume
            # state carries the model, optimiser, pi_ref, step counts and RNG -- not the pool --
            # so without this a resumed self-pool run restarts with one copy of its current self.
            # That is nearly plain self-play, and it would take ~60M steps to return to capacity:
            # an "extension" of a pooled run would spend its first third not really pooled, and
            # nothing would report it. Spread the seeds across the run's history rather than
            # taking the most recent, which is what the pool's eviction rule aims for too.
            _seed_paths: List[str] = []
            _run_dir = ""
            if resume:
                _src_file = resolve_resume(resume)
                _run_dir = os.path.dirname(os.path.abspath(_src_file))
                _snaps = []
                for _f in os.listdir(_run_dir):
                    _m = re.match(r"snapshot_(\d+)steps\.pt$", _f)
                    if _m:
                        _snaps.append((int(_m.group(1)), os.path.join(_run_dir, _f)))
                _snaps.sort()

                # The pool the run actually held, recorded in its resume state. Preferred over
                # any reconstruction: eviction is by spacing and depends on the order snapshots
                # arrived, so an evenly-sampled rebuild lands on a *different* pool that merely
                # looks similar -- and it zeroes the win/game record, which a PFSP draw needs.
                if not reset_opponent_pool:
                    try:
                        _b = torch.load(_src_file, map_location="cpu", weights_only=False)
                        _saved_pool = _b.get("opponent_pool")
                        del _b
                    except Exception as _exc:     # a resume is still valid without it
                        print(f"[opponent pool] could not read pool state from {_src_file} "
                              f"({_exc}); falling back to a rebuild.", flush=True)

                _have = {n: q for n, q in _snaps}
                _want = [int(x) for x in (_saved_pool or {}).get("steps", [])]
                if _want:
                    _seed_steps = [w for w in _want if w in _have]
                    _seed_paths = [_have[w] for w in _seed_steps]
                    _missing = len(_want) - len(_seed_steps)
                    if _missing:
                        print(f"[opponent pool] {_missing} of {len(_want)} recorded members have "
                              f"no snapshot on disk and are dropped.", flush=True)
                if not _seed_paths:
                    # No record, or none of it survives: spread evenly over the run's history,
                    # which is what the pool's own eviction rule aims for.
                    if len(_snaps) > opponent_pool_size:
                        _idx = np.linspace(0, len(_snaps) - 1,
                                           opponent_pool_size).round().astype(int)
                        _snaps = [_snaps[int(i)] for i in sorted(set(_idx.tolist()))]
                    _seed_paths = [q for _, q in _snaps]
                    _seed_steps = [n for n, _ in _snaps]
                    _saved_pool = None

            if _seed_paths:
                _seed_nets = load_pool(_seed_paths, dev)
                # Each member keeps the action view it was trained in (P23): E4-era snapshots of
                # a lineage continued as E4.1 go on being shown E4 masks.
                _seed_merged = [checkpoint_merged_influence(p) for p in _seed_paths]
                _span = _seed_steps[-1] - _seed_steps[0]
                _how = "restored" if _saved_pool else "rebuilt"
                _src = (f"{len(_seed_nets)} snapshot(s) of this run, spanning "
                        f"{_span / 1e6:.0f}M steps ({_how} on resume)")
            else:
                # A fresh self-growing pool has nothing to play against at step 0, so it is
                # seeded with a frozen copy of the starting policy -- the run's own past self,
                # which is exactly what the pool is made of thereafter.
                _seed_nets = [_copy.deepcopy(model).to(dev)]
                _seed_merged = [bool(merged_influence)]
                _src = "self (seeded from the initial policy)"
                if resume and not reset_opponent_pool:
                    # For a RESUMED run this branch is never what anyone wanted: the run has a
                    # history of snapshots and should be playing them. Reaching here means none
                    # were found, and the last time that happened nothing said so --
                    # E3-29-28_20260917_074357 trained 5M steps against a single frozen copy of
                    # itself, winning 99.7%, and the resulting "collapse" cost two days of
                    # investigation aimed at the CE coefficient and the reference schedule.
                    # See research/log/P15_X4b_collapse_is_pool_starvation.md.
                    print("[opponent pool] WARNING: resuming a self-growing pool but found no "
                          f"snapshots in {_run_dir!r}, so the pool starts at ONE frozen copy of "
                          "the current policy. This is nearly plain self-play and the arm will "
                          "not be comparable to a pooled one. Pass --reset-opponent-pool if that "
                          "is deliberate; otherwise check that the resume path names a run "
                          "directory that holds snapshot_<n>steps.pt files.", flush=True)
        trainer.opponent_pool = OpponentPool(
            _seed_nets,
            num_envs=num_envs,
            frac=opponent_frac,
            seed=(_seed_pool or 0),
            lock_learner_side=_lock,
            capacity=opponent_pool_size,
            pfsp=bool(opponent_pfsp),
            pfsp_weighting=opponent_pfsp_weighting,
            pfsp_uniform_mix=opponent_pfsp_uniform_mix,
            merged=_seed_merged,
            seat_balance=seat_balance,
            seat_balance_max_frac=seat_balance_max_frac,
        )
        if any(_seed_merged) or merged_influence:
            print(f"[opponent pool] action views (P23): learner "
                  f"{'E4.1' if merged_influence else 'E4'}, members "
                  f"{sum(_seed_merged)} E4.1 / {len(_seed_merged) - sum(_seed_merged)} E4", flush=True)
        if opponent_self_pool and not opponent_checkpoints and _saved_pool:
            trainer.opponent_pool.load_state_dict(_saved_pool, _seed_steps)
        if reset_opponent_pool and resume:
            print("[opponent pool] --reset-opponent-pool: the recorded pool was discarded "
                  "deliberately.", flush=True)
        print(f"[opponent pool] {_src}, frac={opponent_frac}, capacity={opponent_pool_size}, "
              f"self-growing={bool(opponent_self_pool)}, learner side="
              f"{opponent_lock_side or 'alternating'}, "
              f"draw={'PFSP-' + opponent_pfsp_weighting if opponent_pfsp else 'uniform'}"
              + (f" (uniform floor {opponent_pfsp_uniform_mix})" if opponent_pfsp else "")
              + (f", seat-balance=on (max frac {seat_balance_max_frac})" if seat_balance else ""))
    if per_seat_adv_norm:
        print("[advantages] normalised per seat (--per-seat-adv-norm)", flush=True)
    if wolf_seat_weight:
        print(f"[wolf] per-seat surrogate weights from the self-play USSR share: power={wolf_power}, "
              f"memory={wolf_ema_games:g} games (--wolf-seat-weight)", flush=True)

    # Opponent agents for evaluation (starts with baselines, dynamically appends past snapshots)
    opp_specs = eval_opponents or ["random", "heuristic"]
    opponents: List[PlayerAgent] = []
    for spec in opp_specs:
        try:
            opponents.append(load_agent(spec, device=dev))
        except Exception as e:
            print(f"Warning: Could not load opponent \"{spec}\": {e}", flush=True)

    t_start = time.time()
    # Evaluation and pool refreshes are charged to their own clock, not to the training
    # budget. Previously they came out of the same wall clock as training: one 3-hour run
    # spent 61% of its budget evaluating and completed 473 iterations while its A/B partner
    # completed 1024, purely because the arm that plays longer games has costlier
    # evaluations. That made a wall-clock budget silently policy-dependent.
    overhead_seconds = 0.0
    # Everything about a run's schedule is stated in env steps. A wall-clock budget cannot make
    # two arms comparable, because steps/sec depends on the policy -- one 3-hour A/B ended 1024
    # iterations against 473 for exactly that reason. The snapshot interval used to be DERIVED
    # from two time flags as `train_steps // (duration_seconds // snapshot_interval_seconds)`,
    # which is how E3-22-28's first attempt came to snapshot every 26.7M steps where its baseline
    # snapshotted every ~5M. Snapshots feed the opponent pool, so that silently made the arm a
    # two-factor experiment. There is no time flag left to get wrong.
    step_budget = int(train_steps)
    eval_every_steps = int(snapshot_every_steps)
    next_eval_steps = eval_every_steps
    it = 0

    injector = None
    if inject_dataset and inject_every > 0:
        injector = _HumanInjector(inject_dataset, model, dev, inject_every, inject_weight)
        print(f"Injecting human data from {inject_dataset} every {inject_every} "
              f"iterations at weight {inject_weight}; "
              f"{len(injector.train_idx):,} train samples, "
              f"{injector.held_out_games} games held out and never injected",
              flush=True)

    print("=" * 80, flush=True)
    print(f"STARTING GENERIC TRAINING PIPELINE ({train_steps:,} steps, "
          f"snapshot every {eval_every_steps:,} steps)", flush=True)
    num_baselines = len(opponents)
    budget_desc = f"{train_steps:,} steps"

    print(f"Arch: {arch} | Envs: {num_envs} | Budget: {budget_desc} | "
          f"Opponents to evaluate: {[o.name for o in opponents]} "
          f"(+ up to {max_snapshot_opponents} recent snapshots)", flush=True)
    print("=" * 80, flush=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Snapshot Tournament Evaluation Report ({arch.upper()})\n\n")

    # Initial Snapshot (0s / start)
    resume_path = os.path.join(out_dir, RESUME_FILENAME)
    resumed_elapsed = 0.0
    if resume:
        src = resolve_resume(resume)
        state = load_resume_state(src, model, trainer, seed=seed)
        it = state["iteration"]
        resumed_elapsed = state["elapsed_seconds"]
        # Rewind the clock so elapsed keeps counting from where the run stopped rather than from
        # zero. Nothing is scheduled on it any more -- it is reporting and ETA only.
        t_start -= resumed_elapsed
        # Snapshots are due by step count, and those steps already happened.
        next_eval_steps = ((state["total_env_steps"] // eval_every_steps) + 1) * eval_every_steps
        print(f"Resumed from {src}: {state['total_env_steps']:,} steps, iteration {it}, "
              f"{resumed_elapsed:.0f}s of training already done", flush=True)

    # Restored before anything reads or writes the weights: snapshot_0s.pt claims to be
    # this run's starting point, and an eval of it costs real time, so both have to see
    # the resumed policy rather than a fresh initialisation.
    snap_0_path = os.path.join(out_dir, "snapshot_0s.pt")
    torch.save(model.state_dict(), snap_0_path)
    evaluate_and_log_snapshot(
        model=model,
        merged_influence=merged_influence,
        opponents=opponents,
        elapsed_seconds=0,
        out_dir=out_dir,
        report_path=report_path,
        games_per_side=eval_games_per_side,
        device=dev,
        add_to_opponents_after=True,
        arch=arch,
        num_baselines=num_baselines,
        max_snapshot_opponents=max_snapshot_opponents,
    )

    def _refresh_start_pool(tag: str) -> Dict[str, float]:
        """Rebuild the pool from the current policy and re-draw which env starts where."""
        if start_pool is None:
            return {}
        stats = start_pool.harvest(model, num_envs=min(256, max(32, num_envs)),
                                   num_episodes=start_pool_episodes)
        for i in range(num_envs):
            env_start_turns[i] = None
        assignment = start_pool.assign_starts(num_envs, DEFAULT_TURN_MIX)
        for i, turn in enumerate(assignment):
            env_start_turns[i] = turn
        sizes = {t: start_pool.size(t) for t in sorted(start_pool.buckets)}
        resumed = sum(1 for t in assignment if t is not None)
        print(f"  start pool ({tag}): {sizes} | {resumed}/{num_envs} envs resume mid-game",
              flush=True)
        return stats.as_metrics()

    def _progress_label(train_elapsed: float, steps_done: int) -> str:
        """Show progress against the active budget, with a projected wall-clock finish.

        A step budget is what makes two arms comparable, but it says nothing about how long
        the run will take, so project the finish from the rate observed so far and include
        evaluation overhead measured to date.
        """
        rate = steps_done / max(train_elapsed, 1e-6)
        remaining = max(0, step_budget - steps_done) / max(rate, 1e-6)
        eta = int(train_elapsed + overhead_seconds + remaining)
        return f"{steps_done:,}/{step_budget:,} steps, ETA {eta}s"

    # Which auxiliary loss terms this configuration actually trains. Everything else would be a
    # constant zero series.
    active_aux_losses: List[str] = []
    if defcon_coef > 0.0:
        active_aux_losses.append("defcon_risk_loss")
    if injector is not None:
        active_aux_losses.append("inject_loss")
    if search_ce_coef > 0.0:
        # P15-X4b. The CE term's magnitude and its share of the raw gradient. Registering them
        # here rather than unconditionally keeps them off every run that has no searcher, and
        # registering them at ALL is the point: the arm that collapsed at ~20M logged neither,
        # so the first visible symptom was kl_div reaching 30 with the policy already gone.
        active_aux_losses.append("search_ce")
        active_aux_losses.append("search_ce_grad_frac")
        # Target/mask alignment, which search_ce only reports by accident: mass on a masked
        # action shows up as an absurd CE because the mask fill is -1e9, while the same
        # misalignment on a legal action is silent. These two measure it directly.
        # Entropy of the search target, to be read beside the policy's own "entropy". The decline
        # signature is policy entropy RISING; whether the target's rises with it decides whether
        # the CE term is teaching that flattening or merely failing to prevent it.
        # The two norms behind search_ce_grad_frac plus the searched-row count. A falling share
        # can mean the CE gradient shrank, the rest of the update grew, or fewer rows were
        # searched; the ratio alone cannot tell those apart and they need different fixes.
        active_aux_losses.append("search_ce_grad_norm")
        active_aux_losses.append("total_grad_norm")
        active_aux_losses.append("search_ce_rows")
        active_aux_losses.append("search_target_entropy")
        active_aux_losses.append("search_target_illegal_mass_max")
        active_aux_losses.append("search_target_illegal_row_frac")
        # What the real mask rejected from the searcher's answer before it became a target. A
        # small nonzero rate is correct -- determinization makes some illegal actions look legal;
        # zero would mean the filter is not running.
        active_aux_losses.append("search_dropped_visit_frac")
        active_aux_losses.append("search_dropped_row_frac")

    # Seeded from the trainer, not from the loop variable, which does not exist yet -- and from
    # the clock as it stands, not from zero: a resumed run's elapsed already includes the previous
    # leg, so a zero baseline would divide the first iteration's steps by hours and report ~0.
    prev_steps = int(trainer.total_env_steps)
    prev_elapsed = time.time() - t_start - overhead_seconds

    # Step count of the most recent step-tagged resume file, so the interval is
    # measured from what was actually written rather than from the loop counter.
    last_tagged_resume: Optional[int] = None

    _refresh_start_pool("initial")

    while True:
        elapsed = time.time() - t_start - overhead_seconds
        if trainer.total_env_steps >= step_budget:
            break

        # Curriculum stage switch from UsefulActionsReward to BlunderAwareRewardCalculator
        if (is_curriculum and not curriculum_switched
                and trainer.total_env_steps >= curriculum_switch_at):
            curriculum_switched = True
            trainer.set_reward_calculator(
                BlunderAwareRewardCalculator(decisiveness_turns=decisiveness_turns))
            trainer.set_slice_turn_boundaries(False if slice_turn_boundaries is None else slice_turn_boundaries)
            print(f"\n{'=' * 80}", flush=True)
            print(f"[CURRICULUM] STAGE 2 SWITCH: Replaced UsefulActionsReward with "
                  f"BlunderAwareRewardCalculator at {trainer.total_env_steps:,} / "
                  f"{step_budget:,} steps", flush=True)
            print(f"{'=' * 80}\n", flush=True)

        it += 1
        iteration_metrics = trainer.train_iteration()
        if injector is not None:
            iteration_metrics["inject_loss"] = injector.maybe_step(it)
        total_env_steps = trainer.total_env_steps
        episode_stats = summarize_completed_episodes(iteration_metrics.get("completed_episodes", []))

        # Log training step metrics
        step_metrics: Dict[str, Any] = {
            "iteration": it,
            "elapsed_seconds": int(elapsed),
            "total_steps": total_env_steps,
            # Rate since the previous iteration, not total/elapsed. The lifetime average is
            # meaningless on a resumed run -- its clock is rewound to include the previous leg, so
            # a run actually doing 7,123 steps/s reported 9,930 -- and it is what a reader checking
            # throughput is looking at. The lifetime figure is kept beside it, named for what it is.
            "steps_per_sec": int((total_env_steps - prev_steps) / max(1e-6, elapsed - prev_elapsed)),
            "steps_per_sec_avg": int(total_env_steps / max(1.0, elapsed)),
            "loss": iteration_metrics["loss"],
            "policy_loss": iteration_metrics["policy_loss"],
            "value_loss": iteration_metrics["val_loss"],
            "kl_div": iteration_metrics["kl_div"],
            "entropy": iteration_metrics["entropy"],
            "clip_frac": iteration_metrics.get("clip_frac", 0.0),
            "explained_variance": float(iteration_metrics.get("explained_variance", 0.0)),
            "adv_std": float(iteration_metrics.get("adv_std", 0.0)),
            "adv_std_raw": float(iteration_metrics.get("adv_std_raw", 0.0)),
            "adv_frac_near_zero": float(iteration_metrics.get("adv_frac_near_zero", 0.0)),
            # Importance-ratio health, on every run rather than behind a flag: the failure they
            # exist to catch took one iteration and left every other instrument looking fine.
            "logratio_max": float(iteration_metrics.get("logratio_max", 0.0)),
            "old_logprob_min": float(iteration_metrics.get("old_logprob_min", 0.0)),
            "ratio_negadv_max": float(iteration_metrics.get("ratio_negadv_max", 0.0)),
            # The objective actually differentiated, and the KL regulariser's share of it.
            # "policy_loss" above is the PPO surrogate alone, which is why E3-31-28 could run
            # with the KL term at 300x the surrogate and no logged series showing it.
            "policy_loss_total": float(iteration_metrics.get("policy_loss_total", 0.0)),
            "kl_term": float(iteration_metrics.get("kl_term", 0.0)),
        }
        # Only once enough games have finished for the tracker to report; logging a
        # placeholder 0.0 before then would draw a line that looks like a collapse.
        for _ck in ("critic_auc", "critic_auc_turn3", "critic_brier_skill",
                    "critic_base_rate", "critic_samples"):
            if _ck in iteration_metrics:
                step_metrics[_ck] = float(iteration_metrics[_ck])
        # Every opponent-pool metric, by prefix rather than by name. This was a fixed list of
        # three, so the per-opponent win rates added for PFSP were computed each iteration and
        # silently dropped here -- caught by a smoke run, not by any test, because stats() was
        # correct and only the forwarding was not. A prefix cannot go stale the same way.
        for _ok, _ov in iteration_metrics.items():
            if _ok.startswith("opp_") and isinstance(_ov, (int, float)):
                step_metrics[_ok] = float(_ov)
        # Per-seat learning signal, by name pattern for the same reason: the buffer computed
        # adv_mean/std/n per seat on every iteration and this list dropped all six, so the
        # question every collapse raises -- is the losing seat still getting a signal? -- could
        # not be answered from any run's log.
        for _sk in ("adv_mean_us", "adv_mean_ussr", "adv_std_us", "adv_std_ussr",
                    "adv_n_us", "adv_n_ussr", "entropy_us", "entropy_ussr",
                    "wolf_sp_ussr", "wolf_w_us", "wolf_w_ussr"):
            if _sk in iteration_metrics:
                step_metrics[_sk] = float(iteration_metrics[_sk])
        # Auxiliary losses only where the term that produces them is switched on. Logged
        # unconditionally they are a flat zero line for the whole run -- five of them on an
        # ordinary v2 run -- which reads as "trained and converged" rather than "not present".
        for _aux in active_aux_losses:
            step_metrics[_aux] = float(iteration_metrics.get(_aux, 0.0))

        step_metrics.update(episode_stats)
        if "entropy_fixed_probe" in iteration_metrics:
            step_metrics["entropy_fixed_probe"] = float(iteration_metrics["entropy_fixed_probe"])

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_metrics) + "\n")

        # Indexed by environment steps, not iteration. Runs are budgeted and compared by
        # --train-steps, and iteration count depends on --num-envs and rollout length, so two
        # otherwise comparable runs would sit on different x-axes.
        # The end-turn distribution, not just its mean: one mean end turn is either most games
        # ending near it or a mixture of early blowups and full-length games, and only the
        # second is what the human corpus looks like.
        _finished = iteration_metrics.get("completed_episodes", [])
        if _finished:
            tb.log_histogram("endgame/turn_distribution",
                             [float(e["turn"]) for e in _finished if "turn" in e],
                             total_env_steps)
            tb.log_histogram("endgame/ply_distribution",
                             [float(e["ply"]) for e in _finished if "ply" in e],
                             total_env_steps)

        tb.log_metrics(
            step_metrics,
            step=total_env_steps,
            skip_keys=(episode_dependent_in(episode_stats)
                       if episode_stats["episodes_completed"] == 0.0 else None),
        )
        prev_steps, prev_elapsed = total_env_steps, elapsed

        if it % 10 == 0:
            tb.flush()

        if it % 10 == 0:
            print(
                f"[{_progress_label(elapsed, total_env_steps)}] It {it:4d} | Steps: {total_env_steps:,} ({step_metrics['steps_per_sec']:,} st/s) | "
                f"Loss: {step_metrics['loss']:.3f} | KL: {step_metrics['kl_div']:.4f} | Ent: {step_metrics['entropy']:.3f} | Clip: {step_metrics['clip_frac']*100:.1f}% | "
                f"EV: {step_metrics['explained_variance']:+.3f} | Turn: {step_metrics['mean_turn']:.1f} "
                f"| Ply: {step_metrics['mean_ply']:.0f}/154",
                flush=True,
            )

        # Snapshot Evaluation
        due = total_env_steps >= next_eval_steps
        if due:
            t_eval0 = time.time()
            snap_path = os.path.join(
                out_dir,
                # The exact step count, not millions: an interval below 1M would round every
                # snapshot to the same name and they would overwrite each other in silence.
                # Sort these numerically, not lexicographically.
                f"snapshot_{total_env_steps}steps.pt")
            torch.save(model.state_dict(), snap_path)
            # Hand this snapshot to the opponent pool, if it is growing from the run's own
            # history. A *copy* is loaded from what was just written rather than the live model:
            # adding the model under training would give the learner an opponent whose weights
            # move with it, which is ordinary self-play wearing a costume.
            if opponent_self_pool and trainer.opponent_pool is not None:
                try:
                    # deepcopy for the architecture, then overwrite with the snapshot's
                    # weights. There is no model factory that reconstructs an arbitrary
                    # configuration from metadata, and guessing one would be a way to build a
                    # subtly different opponent.
                    _opp = _copy.deepcopy(model)
                    _opp.load_state_dict(torch.load(snap_path, map_location=dev,
                                                    weights_only=True))
                    trainer.opponent_pool.add(_opp.to(dev), total_env_steps,
                                              merged=bool(merged_influence))
                except Exception as _e:
                    # A pool that fails to grow is a degraded experiment, not a dead one --
                    # say so loudly and keep training rather than losing the run.
                    print(f"[opponent pool] FAILED to add snapshot at {total_env_steps}: {_e}",
                          flush=True)
            # Beside the snapshot, not inside it: snapshot_*.pt stays a bare state dict because
            # load_agent, the tournament runner and every eval module read it as one.
            save_resume_state(resume_path, model, trainer, it, total_env_steps, elapsed,
                              seed=seed)
            # A step-tagged copy, so this point stays branchable after the next snapshot
            # overwrites resume_state.pt. Gated on its own interval rather than written at
            # every snapshot: a resume file is 48 MB against a snapshot's 13 MB, so at a 5M
            # snapshot interval writing one each time is 1.5 GB per 160M-step run and the
            # resume files become 80% of the directory. Branch points are wanted every tens of
            # millions of steps; snapshots are wanted far more often than that, because they
            # are what tournaments and probes read.
            if resume_every_snapshot and (
                    last_tagged_resume is None
                    or total_env_steps - last_tagged_resume >= resume_every_steps):
                save_resume_state(
                    os.path.join(out_dir, RESUME_AT_STEPS.format(steps=total_env_steps)),
                    model, trainer, it, total_env_steps, elapsed, seed=seed)
                last_tagged_resume = total_env_steps
            decisive = evaluate_and_log_snapshot(
                model=model,
                merged_influence=merged_influence,
                opponents=opponents,
                elapsed_seconds=int(elapsed),
                out_dir=out_dir,
                report_path=report_path,
                games_per_side=eval_games_per_side,
                device=dev,
                add_to_opponents_after=True,
                arch=arch,
                num_baselines=num_baselines,
                max_snapshot_opponents=max_snapshot_opponents,
            )
            pool_metrics = _refresh_start_pool(f"@{int(elapsed)}s")
            if pool_metrics:
                decisive = {**(decisive or {}), **pool_metrics}
            if decisive:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iteration": it, "elapsed_seconds": int(elapsed), **decisive}) + "\n")
                if tb is not None:
                    tb.log_metrics(decisive, total_env_steps)
            next_eval_steps += eval_every_steps
            overhead_seconds += time.time() - t_eval0

    # Final Snapshot
    final_snap_path = os.path.join(out_dir, "snapshot_final.pt")
    torch.save(model.state_dict(), final_snap_path)
    # And the resume state, which is otherwise only written at snapshot boundaries -- so a run
    # that ends between them leaves a resume point up to one interval behind its own final
    # weights, and picking it back up would silently repeat those steps.
    save_resume_state(resume_path, model, trainer, it, trainer.total_env_steps,
                      time.time() - t_start - overhead_seconds, seed=seed)
    evaluate_and_log_snapshot(
        model=model,
        merged_influence=merged_influence,
        opponents=opponents,
        elapsed_seconds=int(time.time() - t_start),
        out_dir=out_dir,
        report_path=report_path,
        games_per_side=eval_games_per_side,
        device=dev,
        add_to_opponents_after=False,
        arch=arch,
        num_baselines=num_baselines,
        max_snapshot_opponents=max_snapshot_opponents,
    )

    tb.flush()
    tb.close()

    print(f"\n=== Training Complete. Final Checkpoint: {final_snap_path} ===", flush=True)

    # Post-training massive tournament
    if post_tournament:
        run_post_training_tournament(
            checkpoint_dir=out_dir,
            additional_models=post_tournament_models or ["random", "heuristic"],
            games_per_side=post_tournament_games,
            device=dev,
        )
