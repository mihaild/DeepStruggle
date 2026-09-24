import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Unified Generic CLI Training & Evaluation Runner for Twilight Struggle AI.

import argparse
import os
import sys
import torch

from ai.training.generic_trainer import train_pipeline, run_behavioral_cloning_warmup
from ai.training.generic_trainer import run_search_distillation
from tools.lib.data_root import data_path
from tools.lib.player_agent import load_agent
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.tournament_evaluator import TournamentEvaluator
from ai.models.coldwar_net import create_coldwar_net
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


#: The ladder axes that `--arch ladder` requires. Listed once so the error names all of them.
_LADDER_REQUIRED = ("ladder_input_mode", "ladder_aggregation", "ladder_entity_dim",
                    "ladder_entity_proj_dim", "ladder_hidden_dim", "ladder_res_blocks")
#: Required in addition, but only when the rung actually has per-entity heads.
_LADDER_HEAD_REQUIRED = ("ladder_head_context", "ladder_head_static", "ladder_head_entities")
#: Required in addition, but only when the P22 card lookup is switched on.
_LADDER_LOOKUP_REQUIRED = ("ladder_card_lookup_heads", "ladder_card_lookup_dim",
                           "ladder_card_lookup_identity_dim")


def _ladder_config(args: argparse.Namespace) -> "dict[str, object] | None":
    """The P21 backbone configuration, or None when another architecture was asked for.

    Every axis is required rather than defaulted. A defaulted variant argument is how a model
    silently becomes a different architecture from the one the run's metadata claims -- the
    failure this repository has already had five times with a defaulted observation layout.
    """
    if args.arch != "ladder":
        for name in (_LADDER_REQUIRED + _LADDER_HEAD_REQUIRED
                     + ("ladder_card_self_attention", "ladder_cross_attention",
                        "ladder_card_lookup", "ladder_card_lookup_heads",
                        "ladder_card_lookup_dim", "ladder_card_lookup_identity_dim")):
            if getattr(args, name, None) is not None:
                raise SystemExit(
                    f"--{name.replace('_', '-')} was given but --arch is {args.arch!r}. "
                    f"The ladder flags only apply to --arch ladder; refused rather than "
                    f"silently ignored.")
        return None
    required = list(_LADDER_REQUIRED)
    if int(args.per_entity_heads):
        required += list(_LADDER_HEAD_REQUIRED)
    if args.ladder_card_lookup:
        required += list(_LADDER_LOOKUP_REQUIRED)
    missing = [n for n in required if getattr(args, n, None) is None]
    if missing:
        raise SystemExit(
            "--arch ladder requires every structural axis to be named explicitly. Missing: "
            + ", ".join("--" + m.replace("_", "-") for m in missing)
            + ".\nSee research/plans/P21_architecture_ladder.md for the rung definitions.")
    return dict(
        input_mode=args.ladder_input_mode,
        aggregation=args.ladder_aggregation,
        entity_dim=int(args.ladder_entity_dim),
        entity_proj_dim=int(args.ladder_entity_proj_dim),
        card_self_attention=bool(args.ladder_card_self_attention),
        cross_attention=bool(args.ladder_cross_attention),
        per_entity_heads=int(args.per_entity_heads),
        # Defaulted only when there is no head to configure, where the values are inert and
        # LadderNet refuses anything but these.
        head_context=bool(args.ladder_head_context) if args.per_entity_heads else True,
        head_static=bool(args.ladder_head_static) if args.per_entity_heads else True,
        head_entities=(args.ladder_head_entities if args.per_entity_heads else "both"),
        identity_dim=int(args.identity_dim),
        drop_static=bool(args.drop_static),
        hidden_dim=int(args.ladder_hidden_dim),
        num_res_blocks=int(args.ladder_res_blocks),
        num_attn_heads=4,
        card_lookup=bool(args.ladder_card_lookup),
        # Inert when the lookup is off, and LadderNet never reads them in that case.
        card_lookup_heads=int(args.ladder_card_lookup_heads or 0),
        card_lookup_dim=int(args.ladder_card_lookup_dim or 0),
        card_lookup_identity_dim=int(args.ladder_card_lookup_identity_dim or 0),
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from main() so tests can assert on it.

    Every budget here is in env steps. The time flags this parser used to carry
    (--duration-seconds, --snapshot-interval-seconds, --curriculum-switch-seconds) are
    gone rather than deprecated: the snapshot cadence was derived from the first two,
    and a silently-accepted time flag is exactly how E3-22-28's first attempt came to
    snapshot five times slower than the baseline it was being compared against.
    """
    parser = argparse.ArgumentParser(
        description="Generic Twilight Struggle Neural AI Training Pipeline",
        # No prefix matching. `--snapshot-every` used to be an alias for
        # --snapshot-interval-seconds; with abbreviation on, a stale command passing
        # `--snapshot-every 600` is silently accepted as --snapshot-every-steps 600 --
        # a 600-STEP cadence instead of 600 seconds. A removed flag has to fail.
        allow_abbrev=False)
    parser.add_argument("--arch", type=str, default="v2", choices=["v1", "v2", "mlp", "ladder"],
                        help="Model architecture. v2 is the baseline; v1 is the original\n"
                             "network, kept because checkpoints that predate v2 still name it;\n"
                             "`ladder` is the P21 configurable backbone, whose structural flags\n"
                             "are all REQUIRED (see --ladder-*).")

    lad = parser.add_argument_group(
        "P21 ladder backbone (--arch ladder)",
        "Every axis is required with --arch ladder. There are no defaults on purpose: a "
        "defaulted variant argument was the mechanism behind five instances of one bug in this "
        "repository, because handing a model the wrong variant returns a number instead of "
        "raising. See research/plans/P21_architecture_ladder.md.")
    lad.add_argument("--ladder-input-mode", type=str, default=None,
                     choices=["flat", "grouped", "entity"],
                     help="flat = M0 MLP; grouped = M1 per-block dense; entity = M2+ tokens.")
    lad.add_argument("--ladder-aggregation", type=str, default=None,
                     choices=["flatten", "pool"],
                     help="How entity tokens become a fixed vector. flatten keeps position; "
                          "pool is the mean+max the current architecture uses, and is the "
                          "removal M5 measures.")
    lad.add_argument("--ladder-entity-dim", type=int, default=None,
                     help="Per-entity token width d.")
    lad.add_argument("--ladder-entity-proj-dim", type=int, default=None,
                     help="Width each block is projected to before the trunk.")
    lad.add_argument("--ladder-card-self-attention", action="store_true", default=None,
                     help="M3: cards attend over cards. The current architecture has no "
                          "attention between cards of any kind.")
    lad.add_argument("--ladder-cross-attention", action="store_true", default=None,
                     help="M4: cards attend over countries.")
    lad.add_argument("--ladder-hidden-dim", type=int, default=None, help="Trunk width.")
    lad.add_argument("--ladder-res-blocks", type=int, default=None,
                     help="Number of residual blocks in the trunk.")
    lad.add_argument("--ladder-head-context", dest="ladder_head_context",
                     action="store_true", default=None,
                     help="Feed pe_trunk(h) into the per-entity head. The only path from the "
                          "trunk into the correction; without it the head is a pure per-entity "
                          "map (P21 arm M2a).")
    lad.add_argument("--no-ladder-head-context", dest="ladder_head_context",
                     action="store_false",
                     help="Per-entity head reads its entity only, never the trunk.")
    lad.add_argument("--ladder-head-static", dest="ladder_head_static",
                     action="store_true", default=None,
                     help="Include the constant per-entity slots in the head's input. Alone they "
                          "would be a fixed per-TYPE bias; dropping them leaves the dynamic "
                          "slots, which is where the mechanism's content must be (arm M2b).")
    lad.add_argument("--no-ladder-head-static", dest="ladder_head_static",
                     action="store_false",
                     help="Per-entity head sees only the dynamic slots.")
    lad.add_argument("--ladder-card-lookup", dest="ladder_card_lookup",
                     action="store_true", default=None,
                     help="P22: identity-keyed lookup over the raw card rows. Keys are a learned\n"
                          "per-card identity plus the six property slots; values are the full row,\n"
                          "so what is retrieved is the card's LOCATION. The query comes from the\n"
                          "pre-fusion vector, which makes the lookup conditional -- 'Europe is\n"
                          "negative, therefore check Europe Scoring' -- where the dense card\n"
                          "projection computes every lookup unconditionally.")
    lad.add_argument("--no-ladder-card-lookup", dest="ladder_card_lookup",
                     action="store_false", help="Disable the card lookup (the default).")
    lad.add_argument("--ladder-card-lookup-heads", type=int, default=None,
                     help="Attention heads for the card lookup.")
    lad.add_argument("--ladder-card-lookup-dim", type=int, default=None,
                     help="Per-head key/value width for the card lookup.")
    lad.add_argument("--ladder-card-lookup-identity-dim", type=int, default=None,
                     help="Learned identity width in the lookup KEYS. 0 addresses cards by\n"
                          "property alone, which cannot separate Europe Scoring from Asia\n"
                          "Scoring -- identical ops, era and is_scoring -- so a query returns an\n"
                          "average over the cards it needed to tell apart. Kept reachable as the\n"
                          "ablation that attributes the gain, not as a variant expected to work.")
    lad.add_argument("--ladder-head-entities", type=str, default=None,
                     choices=["both", "country", "card"],
                     help="Which per-entity heads exist. The card-collision finding predicts "
                          "`country` carries most of M2's gain (arms M2d/M2e).")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "warmup", "eval", "curriculum", "distill"],
                        help="Execution mode. `distill` is P15-X4a: soft cross-entropy "
                             "toward a searcher's visit distribution, from a dataset made "
                             "by tools/generate_search_targets.py.")

    # Warm-up / Checkpoint options
    parser.add_argument("--warmup-checkpoint", "--load-path", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--warmup-dataset", type=str, default=None, help="Path to dataset file for supervised BC warmup")
    parser.add_argument("--distill-dataset", type=str, default=None,
                        help="Search-target dataset for --mode distill. Must carry search_pi "
                             "records; a plain self-play set has no targets and would train "
                             "on nothing.")
    parser.add_argument("--distill-epochs", type=int, default=2)
    parser.add_argument("--distill-lr", type=float, default=1e-4,
                        help="Deliberately below BC's 1e-3: this bends a trained policy "
                             "rather than training one.")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Number of epochs for BC warmup")

    # Budget & Snapshot parameters. Both are in env steps, deliberately: a wall-clock budget
    # cannot make two arms comparable, because steps/sec depends on the policy.
    parser.add_argument("--decisiveness-turns", type=float, default=0.0,
                        help="Scale the terminal reward by (1 - turn/K), so a result on turn T is "
                             "worth 1 - T/K instead of 1 (0 = off). With gamma=1 and terminal-only "
                             "rewards the objective is indifferent to *when* you win: taking a forced "
                             "win now and winning three turns later both return +1, so nothing in the "
                             "gradient prefers the former, and trained policies duly leave "
                             "engine-verified forced wins on the table. The scale applies to "
                             "losses too, so a self-inflicted defeat on turn 3 costs more than on "
                             "turn 9, which is the intended 'prolong a lost game' incentive. K is "
                             "a slope: at 40, turn 3 is worth 0.925 and turn 10 is 0.75. Too steep "
                             "biases against build-to-final-scoring play, so watch the ending mix.")
    # A step budget is not a controlled comparison across engine versions: a rebuilt engine plays
    # a different game, so two runs with matched step counts are only broadly comparable. The
    # comparison that holds is a head-to-head tournament on the current engine, which does not
    # need matched steps at all.
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a run from a resume state. Pass the file, the run\n"
                             "directory (its newest state), or '<run_dir>:<steps>' to branch\n"
                             "from a particular snapshot. Restores the weights, the optimiser\n"
                             "moments, the reference policy and the step counter, so training\n"
                             "continues rather than restarting. The environment is not\n"
                             "restored -- the games are an i.i.d. stream, so a resumed run\n"
                             "simply deals fresh ones. Giving a --seed that differs from the\n"
                             "one the state was written under also re-seeds torch and numpy,\n"
                             "so the continuation genuinely diverges instead of replaying the\n"
                             "original sampling.")
    parser.add_argument("--resume-every-snapshot", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Write a resume state beside every snapshot, so any snapshot can\n"
                             "be branched from later and not just the run's end. Costs 38 MB a\n"
                             "snapshot on top of the 13 MB snapshot itself. With this off, a\n"
                             "run keeps only its newest state, and a stretch of it can never\n"
                             "be re-run from -- so a run that turns out to need branching\n"
                             "partway through has nowhere to branch.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the environment stream and torch together. Left unset,\n"
                             "the environment seed is fixed at 12345 and the network is\n"
                             "unseeded, so two runs of one configuration differ only in\n"
                             "initialisation -- which understates run-to-run variance.\n"
                             "Give distinct seeds to measure that variance; give the same\n"
                             "seed to two runs that differ in one thing, to pair them.")
    grp = parser.add_argument_group(
        "seed decomposition",
        "A single --seed drives four independent sources at once: weight initialisation, "
        "rollout sampling, the engine's card deals and dice, and the opponent-pool draw. Each "
        "can be overridden separately so that a seed-dependent outcome can be attributed to one "
        "of them. Every override defaults to --seed.")
    grp.add_argument("--seed-init", type=int, default=None,
                     help="Weight initialisation (and the numpy global stream).")
    grp.add_argument("--seed-sampling", type=int, default=None,
                     help="Action sampling and minibatch shuffling. Applied by re-seeding torch "
                          "immediately after the model is built, so it is independent of "
                          "--seed-init.")
    grp.add_argument("--seed-env", type=int, default=None,
                     help="The engine's per-game streams: card deals and dice.")
    grp.add_argument("--seed-pool", type=int, default=None,
                     help="Opponent-pool draws and which side the learner takes.")
    parser.add_argument("--resume-every-steps", type=int, default=40_000_000,
                        help="Write a step-tagged resume_<steps>.pt at most this often. Resume files are 48MB against a snapshot's 13MB, so this is deliberately much coarser than the snapshot interval.")
    parser.add_argument("--snapshot-every-steps", type=int, default=10_000_000,
                        help="Save and evaluate a snapshot every N env steps (10M since "
                             "2026-09-24; 5M before, when evaluating one cost ~17%% of a run's "
                             "wall time: research/log/P26_quick_screen.md). Reporting only: "
                             "evaluation restores the RNG streams it draws from, and the "
                             "opponent pool grows on --pool-every-steps, so this does not change "
                             "what a run learns.")
    parser.add_argument("--pool-every-steps", type=int, default=5_000_000,
                        help="Add the current policy to the self-play opponent pool every N env "
                             "steps (as pool_<N>steps.pt between snapshots). This sets the rate "
                             "the pool grows: two arms that differ in it train against different "
                             "opponent distributions and are not a one-factor comparison. 5M is "
                             "the lineage's rate; it used to be tied to the snapshot interval, "
                             "and E3-22-28's first attempt snapshotted every 26.7M steps against "
                             "its baseline's 5M and had to be thrown away.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="TF32 matmuls (P26; default on since 2026-09-24): +15%% steps/s on "
                             "M2d, solo or paired, with mean KL 1e-7 to fp32 at inference and no "
                             "strength cost in a 3-seed A/B (research/log/P26_quick_screen.md). "
                             "--no-tf32 for fp32, as every run before E4-57.")
    parser.add_argument("--inject-dataset", type=str, default=None,
                        help="Human corpus directory to interleave supervised steps from during "
                             "RL. A BC warmup washes out early in training; this keeps the "
                             "signal applied rather than applied once.")
    parser.add_argument("--inject-every", type=int, default=0,
                        help="Iterations between injected batches (0 = off). Small and often beats "
                             "large and rare: anything rarer than the washout lets the policy drift "
                             "back between doses.")
    parser.add_argument("--inject-weight", type=float, default=1.0,
                        help="Scale on the injected supervised loss.")
    parser.add_argument("--train-steps", type=int, default=80_000_000,
                        help="The run's budget, in env steps. Must be positive. "
                             "Defaults to the standard 80,000,000. "
                             "Use this for A/B arms: steps/sec depends on the policy, so a wall-clock "
                             "budget gives the two arms different amounts of training. One 3-hour A/B "
                             "ended 1024 iterations against 473 for exactly that reason. The progress "
                             "line projects a wall-clock ETA so a step budget can still be aimed at a "
                             "target duration.")
    parser.add_argument("--eval-max-snapshot-opponents", type=int, default=4,
                        help="Evaluate each snapshot against the baselines plus at most this many recent "
                             "snapshots (0 = unlimited). Unlimited makes evaluation cost quadratic in run "
                             "length: the final evaluation of a 3-hour run faced 14 opponents and took "
                             "957s against a 900s snapshot interval.")

    # Tournament & Evaluation parameters
    parser.add_argument("--eval-opponents", nargs="+", default=["random", "heuristic"], help="List of opponent models/bots to evaluate on snapshots")
    parser.add_argument("--eval-games-per-side", type=int, default=50, help="Games per side per opponent (total 2x games per matchup)")

    # Post-Training Tournament parameters
    parser.add_argument("--post-tournament", action="store_true", help="Automatically launch massive tournament benchmark across all snapshots upon training completion")
    parser.add_argument("--post-tournament-models", nargs="+", default=["heuristic", "random"], help="Additional benchmark models or baselines to include in post-training tournament")
    parser.add_argument("--post-tournament-games", type=int, default=500, help="Games per side in post-training tournament")

    # RL Hyperparameters
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel vectorized game environments")
    parser.add_argument("--buffer-size", type=int, default=128, help="Rollout buffer size per environment")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size for SGD updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eta", type=float, default=0.1, help="NashPG reference KL penalty weight")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Weight on the value loss in the shared-trunk objective. 0.5 suits\n"
                             "the win-value MSE, which sits near 0.04 against a policy loss near\n"
                             "0.04, and it stays 0.5 under --categorical-value: the distribution\n"
                             "carries its own --value-dist-coef instead, so the win objective is\n"
                             "the control's either way. Retuning this to compensate for a\n"
                             "differently-scaled value term was measured not to work -- the\n"
                             "defect was the saturated baseline, not the weighting.")
    parser.add_argument("--value-dist-coef", type=float, default=0.02,
                        help="Weight on the categorical VP cross-entropy, inside the value loss.\n"
                             "Not vp_coef: cross-entropy over 41 atoms starts near ln(41) and\n"
                             "settles near 1.6, where the VP MSE it replaces sits near 0.04, so\n"
                             "sharing a coefficient would weight it about forty times harder.\n"
                             "0.02 puts the distribution objective on par with the win MSE.")
    parser.add_argument("--categorical-value", action="store_true", default=False,
                        help="P1: replace the two scalar value heads with one categorical\n"
                             "distribution over final VP (41 atoms across [-20, +20]), trained\n"
                             "by cross-entropy against the two-hot-projected lambda-return.\n"
                             "It replaces the auxiliary *VP* head only: v_win keeps its own\n"
                             "regressed scalar head, because it is the baseline GAE subtracts\n"
                             "and deriving it from the distribution's sign mass saturates it.\n"
                             "v_vp is exposed as E[VP]/20 from the distribution, so every\n"
                             "consumer runs unchanged. Outcomes here are multimodal and\n"
                             "MSE on a scalar regresses to the mean of the modes -- a value that\n"
                             "is never observed. Changes the checkpoint shape: a categorical\n"
                             "checkpoint cannot be loaded as a scalar one or the reverse.")
    parser.add_argument("--adv-filter-quantile", type=float, default=0.0,
                        help="P1: drop samples whose |advantage| falls below this quantile of\n"
                             "the minibatch from the POLICY loss only; the value head still sees\n"
                             "every sample. 0 disables it. Reported at 2.5x wall-clock in\n"
                             "Ataraxos. It changes the effective batch size, so screen it as its\n"
                             "own factor rather than folding it in with --categorical-value.")
    parser.add_argument("--window-provoked-defcon", action="store_true", default=False,
                        help="Credit a *provoked* DEFCON-1 loss to the player who played the\n"
                             "card, the same way an unprovoked one is credited, instead of\n"
                             "letting it propagate back as an ordinary loss.\n"
                             "The case for it: the fatal card play sits a handful of\n"
                             "micro-actions from the loss and inside the same turn, which the\n"
                             "turn-scoped window already covers, and the critic the -1 would\n"
                             "otherwise propagate through barely moves at the deciding choice\n"
                             "-- so there is nothing for it to attach to. A window's advantage\n"
                             "is -1 - v_t and never consults it. Changes the returns, so a run\n"
                             "with it on is not comparable to one without except as its own\n"
                             "A/B.")
    parser.add_argument("--drop-static", action="store_true", default=False,
                        help="With --arch mlp, drop the per-entity observation slots that never\n"
                             "change: stability, battleground, region, Ops, era, one-time,\n"
                             "is-scoring. 1,364 of 3,824 dims, 35.7%%.\n"
                             "A shared-weight encoder needs them to tell its tokens apart; a\n"
                             "positional reader gets identity from the offset, so those weights\n"
                             "only add a constant the bias already supplies.")
    parser.add_argument("--identity-dim", type=int, default=0,
                        help="Width of a learned identity embedding added to each card and\n"
                             "country token (0 disables it).\n"
                             "The card block encodes a card's properties and never which card\n"
                             "it is; identity exists only as position, and v2's shared per-card\n"
                             "MLP plus pooling discards position. Measured: 110 cards collapse\n"
                             "to 46 signatures, 86%% of them colliding, in groups of up to six --\n"
                             "Tear Down this Wall is the same vector as Chernobyl. Countries are\n"
                             "better off, the graph convolution seeing real adjacency.\n"
                             "Model-side only: the observation is untouched.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--reward-scheme", type=str, default="blunder_aware", choices=["blunder_aware", "terminal", "shaped", "useful_actions", "curriculum"], help="Reward calculation scheme")
    parser.add_argument("--curriculum-switch-steps", type=int, default=None,
                        help="Env step at which the curriculum switches to the BlunderAware "
                             "reward (default: --curriculum-switch-fraction of --train-steps)")
    parser.add_argument("--curriculum-switch-fraction", type=float, default=0.5, help="Fraction of --train-steps at which the curriculum switches to the BlunderAware reward (default: 0.5)")
    parser.add_argument("--per-player-gae", action="store_true", default=False,
                        help="Compute GAE within each player's own subsequence of decisions -- "
                             "bootstrapping from that player's next OWN decision, with the "
                             "opponent's intervening rewards folded in -- instead of over the "
                             "interleaved sequence with a sign flip. Removes the cross-perspective "
                             "bootstrap rather than patching it: the interleaved form assumes "
                             "V(s,me) = -V(s,opponent), which is a perfect-information identity "
                             "and false here (0.144 mean absolute). Measured offline: return RMSE "
                             "0.551 -> 0.481 at lambda 0.98, and exact telescoping at lambda 1. "
                             "No extra forward passes.")
    parser.add_argument("--same-perspective-bootstrap", action="store_true", default=False,
                        help="Bootstrap the value target from V(s_{t+1}, p_t) -- the resulting "
                             "state as seen by the player who just moved -- instead of negating "
                             "the next step's value. The negation assumes V(s,me) = -V(s,opponent), "
                             "which is exact only under perfect information; here the observation "
                             "hides the opponent's hand, and v_US + v_USSR was measured at 0.144 "
                             "mean absolute where it should be 0. Costs one extra forward per "
                             "step and changes what the critic is trained on, so it is an arm.")
    parser.add_argument("--slice-turn-boundaries", type=str, default="auto", choices=["auto", "on", "off"],
                        help="Truncate GAE bootstrapping at game-turn boundaries for EVERY episode, clean wins included. Measurably degrades the policy; retained for ablations. 'auto' (default) leaves it off and relies on per-episode blunder windowing instead.")
    parser.add_argument("--no-blunder-window", action="store_true",
                        help="Disable per-episode blunder windowing. By default an unprovoked blunder loss (held scoring card, or self-inflicted DEFCON 1) only penalises the blunderer within that turn and shields the opponent from the windfall.")
    parser.add_argument("--ref-update-freq", type=int, default=200_000,
                        help="Env steps between NashPG reference-policy refreshes. At 512 envs x 128 buffer one iteration is 65,536 steps, so the default refreshes pi_ref every 4 iterations; raise it for a genuinely frozen anchor.")
    parser.add_argument("--seat-balance", action="store_true", default=False,
                        help="Keep both seats' games winnable. The further self-play drifts from an "
                             "even split, the more envs play the learner against the pool, the "
                             "more often the learner takes the WEAK seat there, and the more the "
                             "opponent draw favours members that seat beats about half the time. "
                             "Aimed at the side-collapse loop: a seat losing every game gets no "
                             "gradient. Needs --opponent-frac > 0 and a pool.")
    parser.add_argument("--seat-balance-max-frac", type=float, default=0.8,
                        help="Upper bound on the mixed-env fraction under full seat-balance pressure.")
    parser.add_argument("--per-seat-adv-norm", action="store_true", default=False,
                        help="Normalise advantages per seat instead of over both, so a losing "
                             "seat's small spread is not scaled away by the winning seat's.")
    parser.add_argument("--wolf-seat-weight", action="store_true", default=False,
                        help="WoLF-style per-seat learning rates: scale each seat's PPO surrogate "
                             "by w_us = 2x^p/(x^p+(1-x)^p) and w_ussr = 2(1-x)^p/(x^p+(1-x)^p), with "
                             "x the USSR's smoothed win share in pure self-play. The winning seat "
                             "learns slowly and the losing seat fast. The entropy bonus, the KL to "
                             "pi_ref and the value loss are not weighted.")
    parser.add_argument("--wolf-power", type=float, default=1.0,
                        help="p in --wolf-seat-weight. 1.0 gives w_us = 2x, w_ussr = 2(1-x); "
                             "below 1 softens the ratio (x/(1-x))^p between the seats.")
    parser.add_argument("--wolf-scope", choices=["surrogate", "policy"], default="surrogate",
                        help="What --wolf-seat-weight scales. 'surrogate' (the first version, "
                             "E4-38) weights the PPO surrogate only; the unweighted entropy bonus "
                             "then pushes the down-weighted seat toward uniform. 'policy' weights "
                             "the seat's whole policy objective -- surrogate, entropy bonus and KL "
                             "to pi_ref -- a per-seat learning rate, as WoLF is defined.")
    parser.add_argument("--wolf-dead-zone", type=float, default=0.0,
                        help="--wolf-seat-weight leaves both weights at 1 while the USSR's self-play "
                             "share is within 0.5 +- this, and outside it shifts the share toward 0.5 "
                             "by this much before weighting, so the weights leave 1 continuously. "
                             "0 is the plain rule.")
    parser.add_argument("--wolf-dead-zone-mode", choices=["shift", "jump"], default="shift",
                        help="Outside --wolf-dead-zone: 'shift' moves the share toward 0.5 by the zone "
                             "width before weighting (continuous, softer); 'jump' applies the plain "
                             "rule to the unshifted share (full brake once outside the zone).")
    parser.add_argument("--adv-norm-floor", type=float, default=0.0,
                        help="P25 3j: divide advantages by max(batch std, c x EMA of the batch std) "
                             "instead of the batch std, so a faded signal stays small rather than "
                             "being rescaled to unit noise. c is this value; 0 is off. EMA memory "
                             "20M steps, no floor for the first 2M.")
    parser.add_argument("--entropy-ceiling", type=float, default=0.0,
                        help="P25 3k: a one-sided per-seat entropy ceiling, in nats. Each seat's "
                             "entropy coefficient moves by -0.01 x (rollout entropy - ceiling) per "
                             "iteration, clipped to [-0.02, --entropy-coef], from 5M steps; below the "
                             "ceiling it is the fixed bonus. 0 is off.")
    parser.add_argument("--target-kl", type=float, default=0.0,
                        help="P25 3l: per-seat early stopping of the PPO epochs. Once a seat's "
                             "approximate KL from the rollout policy on a minibatch exceeds this, its "
                             "policy terms are masked for the rest of the update. 0 is off; the "
                             "per-seat KL is logged either way (approx_kl_us / approx_kl_ussr).")
    parser.add_argument("--entropy-normalize", action="store_true", default=False,
                        help="The entropy bonus rewards entropy / log(legal count) per decision -- the "
                             "fraction of that decision's maximum -- so a many-option decision gets no "
                             "more room than a few-option one (P23: E4.1's ~50-option op-mode nodes). "
                             "Same average bonus as the raw one at ~2.1x --entropy-coef on E4's "
                             "decision mix. Logged entropy stays raw.")
    parser.add_argument("--no-cuda-graphs", action="store_true", default=False,
                        help="Run the rollout forwards eagerly instead of as CUDA-graph replays. The "
                             "replays execute the same kernels (bitwise-identical outputs per network); "
                             "this is a fallback for debugging, and for devices where capture fails.")
    parser.add_argument("--wolf-ema-games", type=float, default=2000.0,
                        help="Memory of --wolf-seat-weight's average, in self-play games: each "
                             "iteration moves it by alpha = min(1, games / this).")
    parser.add_argument("--merged-influence", action="store_true", default=False,
                        help="P23 / E4.1: decide in the merged-influence view, where 'ops for "
                             "influence, first point in X' is one decision (the two E4 steps it "
                             "names, applied together). ~11%% fewer decisions per game; rules, "
                             "state and observation are unchanged. Recorded in metadata.json with "
                             "the step it starts from, so pool members and later evaluations use "
                             "each checkpoint's own view.")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="GAE lambda. Credit decays by lambda per decision over the joint stream of both players' decisions (unless --per-player-gae), so at 0.98 the effective horizon is ~50 decisions, about 1.3 game turns.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor. Keep at 1.0: the game is zero-sum and decided at the end, so any discount biases against the endgame (0.999 attenuates a terminal reward by ~26%% over a full game).")
    parser.add_argument("--defcon-coef", type=float, default=0.0,
                        help="Weight of the auxiliary DEFCON-risk head, which predicts whether the player to move is about to lose the game to its own DEFCON-1 choice (0 = head disabled). Added because val_win_head largely restates the VP margin -- corr(v_win, v_vp) = 0.86 -- so it reads self-inflicted DEFCON-1 deaths as roughly even positions while pricing ordinary losing positions correctly. Try 0.1.")
    parser.add_argument("--opponent-checkpoints", nargs="+", default=None,
                        help="Frozen snapshots to play a share of environments against, instead of pure self-play")
    parser.add_argument("--opponent-frac", type=float, default=0.0,
                        help="Fraction of environments facing a frozen opponent (0 disables)")
    parser.add_argument("--opponent-self-pool", action="store_true",
                        help="Grow the opponent pool from this run's own snapshots, seeded with the initial policy. Use instead of --opponent-checkpoints.")
    parser.add_argument("--opponent-pool-size", type=int, default=12,
                        help="Maximum snapshots held in the pool. Eviction keeps the endpoints and drops the most redundant interior point, so the pool stays spread across the run rather than becoming all-recent.")
    parser.add_argument("--search-ce-coef", type=float, default=0.0,
                        help="P15-X4b. Weight on a cross-entropy term pulling the policy toward "
                             "an honest searcher's visit distribution, on searched decisions "
                             "only. 0 disables it and the run is bit-identical to the baseline. "
                             "Search produces TARGETS and never acts, so the state distribution "
                             "is unchanged and the arm varies one thing.")
    parser.add_argument("--search-sims", type=int, default=32,
                        help="Simulations per searched decision during training.")
    parser.add_argument("--search-subsample", type=float, default=0.125,
                        help="Fraction of eligible decisions searched (0.125 = 1 in 8).")
    parser.add_argument("--search-node-filter", type=str, default="card_playmode",
                        choices=["card_playmode", "all"],
                        help="Which decisions are eligible.")
    parser.add_argument("--rollout-temps", type=float, nargs=4, default=None,
                        metavar=("T1", "T2", "T3", "T4"),
                        help="The four per-environment rollout sampling temperatures. Default is "
                             "0.15 0.50 0.10 0.35 -- every band BELOW 1.0, so sampling is sharper "
                             "than the policy itself despite the schedule being described as "
                             "exploration. Pass values around 1.0 to sample from the policy as "
                             "trained, which is the textbook PPO choice and has never been "
                             "measured here.")
    parser.add_argument("--setup-explore-frac", type=float, default=0.0,
                        help="Fraction of environments whose OPENING placement is replaced by a "
                             "uniform legal choice, and trained on. Temperature cannot reach these "
                             "decisions: measured on E3-30-28 the preferred opening carries a "
                             "logit 16.56 above the runner-up, so it is chosen with probability "
                             "1-6e-8 even at tau=1, while rollouts sample at tau in [0.10, 0.50], "
                             "sharper still. 0 disables it and the run is the baseline.")
    parser.add_argument("--reset-opponent-pool", action="store_true", default=False,
                        help="On resume, discard the opponent pool recorded in the resume state "
                             "and rebuild it by sampling this run's snapshots evenly. Without "
                             "this the pool is restored exactly -- same members, same win/game "
                             "record -- because losing it changes the experiment and should be "
                             "asked for rather than suffered.")
    parser.add_argument("--opponent-pfsp", action="store_true", default=False,
                        help="Draw the pool opponent by prioritised fictitious self-play instead "
                             "of uniformly. NOTE: ai/training/opponent_pool.py argues this should "
                             "HURT -- the pool's documented mechanism is that weak old snapshots "
                             "give the learner's trailing side winnable games, and prioritisation "
                             "removes exactly those. This flag exists to test that argument.")
    parser.add_argument("--opponent-pfsp-weighting", type=str, default="var",
                        choices=["var", "hard"],
                        help="var: x(1-x), peaks on evenly matched opponents. hard: (1-x)^2, "
                             "peaks on opponents beating the learner. Default var, because among "
                             "past selves there is usually nothing beating the learner and 'hard' "
                             "then degenerates to 'most recent'.")
    parser.add_argument("--opponent-pfsp-uniform-mix", type=float, default=0.25,
                        help="Floor of uniform probability mixed under the PFSP weights, so no "
                             "pool member can be driven to zero. The pool's value is its spread.")
    parser.add_argument("--opponent-lock-side", type=str, default=None,
                        choices=["us", "ussr"],
                        help="Pin the learner to one side against the frozen opponent; default alternates")
    parser.add_argument("--start-pool-frac", type=float, default=0.0,
                        help="Enable mid-game start sampling (>0 turns it on). A share of environments resume from saved turn-boundary positions instead of the real opening, with the split set by DEFAULT_TURN_MIX: 50%% turn 1, 15%% turn 4, 15%% turn 6, 10%% turn 8, 10%% turn 10. Self-play from turn 1 reaches turn 10 in only ~20%% of games and leaves the same eight battlegrounds untouched from turn 8 on, so those states are otherwise barely sampled.")
    parser.add_argument("--start-pool-capacity", type=int, default=512,
                        help="Positions kept per turn bucket.")
    parser.add_argument("--start-pool-episodes", type=int, default=600,
                        help="Self-play episodes per pool refresh. Refreshing costs a few seconds -- batched rollout runs ~13x faster than training -- and happens at every snapshot evaluation so the pool tracks the policy rather than preserving an older one.")
    parser.add_argument("--priority-alpha", type=float, default=0.0,
                        help="Sample minibatches weighted by |advantage|^alpha so rare decisive transitions are not drowned by routine ones (0 = uniform). Deliberately biases the gradient toward high-swing states; try 0.5.")
    parser.add_argument("--output-dir", "--save-path", type=str, default=None, help="Output directory for checkpoints (default: data/checkpoints/run_[version]_[start date]_[start time])")
    parser.add_argument("--self-transform", action="store_true",
                        help="Give each GraphConv layer a second weight matrix applied to the node itself, so a country can be held at full strength instead of averaged with its neighbours. A linear probe recovers a country's exact influence far more often from its raw observation slots than from its post-GraphConv token, and the loss tracks neighbour count.")
    parser.add_argument("--graph-layers", type=int, default=2, choices=[0, 1, 2],
                        help="How many graph convolutions over the map adjacency (default 2). Adjacency's mechanical uses -- placement legality, coup legality, the realignment modifier -- are already precomputed per country in the observation, so what the graph adds is strategic reasoning about neighbourhoods. With --self-transform the second layer consistently loses per-country influence without a measured gain, so 1 is worth testing; 0 keeps a per-country encoder and drops adjacency entirely, isolating the relation itself.")
    parser.add_argument("--per-entity-heads", type=int, default=0,
                        help="Width of per-entity policy heads (0 = off; try 64). A card's logit is computed from that card's own token and raw slots, and a country's from that country's, each conditioned on a projection of the trunk; the 18 actions that name no entity stay dense. A country's exact influence is almost entirely recoverable from its own token and almost entirely absent from the pooled trunk, and no read-out ending in one fixed-size summary closed that gap.")
    parser.add_argument("--attn-readout", type=int, default=0,
                        help="Width of an end-of-trunk attention read-out (0 = off; try 64). After the residual trunk, the state vector queries the 84 country and 110 card tokens -- each concatenated with its raw observation slots -- and the result is folded back in. Targets the other half of the same finding: the pre-pooling token holds two thirds of the recoverable per-country influence and the pooled trunk holds none.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run short name as <engine>-<attempt>-<seed>, e.g. E9-99-01. Becomes the checkpoint directory prefix (<run-name>_[date]_[time]) and is recorded in metadata.json. Omit any step budget: one directory holds every budget of a lineage and each snapshot's filename already carries its own.")
    parser.add_argument("--description", type=str, default=None, help="Short description of what was changed and training objective to save in checkpoint metadata.json")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                        help="Mirror every training_metrics.jsonl metric to TensorBoard event files in <output-dir>/tb (default: on). Use --no-tensorboard to disable.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode in ["train", "curriculum"]:
        eff_reward_scheme = "curriculum" if args.mode == "curriculum" else args.reward_scheme
        train_pipeline(
            arch=args.arch,
            warmup_checkpoint=args.warmup_checkpoint,
            warmup_dataset=args.warmup_dataset,
            bc_epochs=args.bc_epochs,
            eval_opponents=args.eval_opponents,
            eval_games_per_side=args.eval_games_per_side,
            num_envs=args.num_envs,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            eta=args.eta,
            vf_coef=args.vf_coef,
            value_dist_coef=args.value_dist_coef,
            categorical_value=args.categorical_value,
            adv_filter_quantile=args.adv_filter_quantile,
            window_provoked_defcon=args.window_provoked_defcon,
            identity_dim=args.identity_dim,
            drop_static=args.drop_static,
            ladder_config=_ladder_config(args),
            seed_init=args.seed_init,
            seed_sampling=args.seed_sampling,
            seed_env=args.seed_env,
            seed_pool=args.seed_pool,
            defcon_coef=args.defcon_coef,
            train_steps=args.train_steps,
            seed=args.seed,
            resume=args.resume,
            resume_every_snapshot=args.resume_every_snapshot,
            resume_every_steps=args.resume_every_steps,
            snapshot_every_steps=args.snapshot_every_steps,
            pool_every_steps=args.pool_every_steps,
            tf32=args.tf32,
            inject_dataset=args.inject_dataset,
            inject_every=args.inject_every,
            inject_weight=args.inject_weight,
            decisiveness_turns=args.decisiveness_turns,
            max_snapshot_opponents=args.eval_max_snapshot_opponents,
            opponent_checkpoints=args.opponent_checkpoints,
            opponent_frac=args.opponent_frac,
            opponent_lock_side=args.opponent_lock_side,
            opponent_self_pool=args.opponent_self_pool,
            opponent_pool_size=args.opponent_pool_size,
            reset_opponent_pool=args.reset_opponent_pool,
            setup_explore_frac=args.setup_explore_frac,
            rollout_temps=args.rollout_temps,
            search_ce_coef=args.search_ce_coef,
            search_sims=args.search_sims,
            search_subsample=args.search_subsample,
            search_node_filter=args.search_node_filter,
            opponent_pfsp=args.opponent_pfsp,
            opponent_pfsp_weighting=args.opponent_pfsp_weighting,
            opponent_pfsp_uniform_mix=args.opponent_pfsp_uniform_mix,
            start_pool_frac=args.start_pool_frac,
            start_pool_capacity=args.start_pool_capacity,
            start_pool_episodes=args.start_pool_episodes,
            entropy_coef=args.entropy_coef,
            reward_scheme=eff_reward_scheme,
            output_dir=args.output_dir,
            run_name=args.run_name,
            self_transform=args.self_transform,
            attn_readout=args.attn_readout,
            per_entity_heads=args.per_entity_heads,
            graph_layers=args.graph_layers,
            description=args.description,
            device=args.device,
            post_tournament=args.post_tournament,
            post_tournament_models=args.post_tournament_models,
            post_tournament_games=args.post_tournament_games,
            curriculum_switch_steps=args.curriculum_switch_steps,
            curriculum_switch_fraction=args.curriculum_switch_fraction,
            slice_turn_boundaries=(None if args.slice_turn_boundaries == "auto" else args.slice_turn_boundaries == "on"),
            same_perspective_bootstrap=args.same_perspective_bootstrap,
            per_player_gae=args.per_player_gae,
            ref_update_freq=args.ref_update_freq,
            gae_lambda=args.gae_lambda,
            merged_influence=args.merged_influence,
            seat_balance=args.seat_balance,
            seat_balance_max_frac=args.seat_balance_max_frac,
            per_seat_adv_norm=args.per_seat_adv_norm,
            wolf_seat_weight=args.wolf_seat_weight,
            wolf_power=args.wolf_power,
            wolf_ema_games=args.wolf_ema_games,
            wolf_scope=args.wolf_scope,
            wolf_dead_zone=args.wolf_dead_zone,
            wolf_dead_zone_mode=args.wolf_dead_zone_mode,
            adv_norm_floor=args.adv_norm_floor,
            entropy_ceiling=args.entropy_ceiling,
            target_kl=args.target_kl,
            entropy_normalize=args.entropy_normalize,
            cuda_graphs=not args.no_cuda_graphs,
            blunder_window=not args.no_blunder_window,
            gamma=args.gamma,
            priority_alpha=args.priority_alpha,
            tensorboard=args.tensorboard,
        )
    elif args.mode == "warmup":
        if not args.warmup_dataset:
            print("Error: --warmup-dataset required for mode=warmup")
            sys.exit(1)
        dev = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
        if args.arch == "ladder":
            # Refused rather than silently building a v1: the P21 ladder runs cold starts only,
            # and this branch is the same allow-list shape that sent a ladder model down the v1
            # path in generic_trainer and killed a run at its first snapshot.
            raise SystemExit(
                "--mode warmup does not support --arch ladder. The P21 ladder is cold-start "
                "only; see research/plans/P21_architecture_ladder.md.")
        if args.arch == "v2":
            model = create_coldwar_net_v2(dev)
        else:
            model = create_coldwar_net(dev)
        out_save = args.output_dir or f"data/checkpoints/coldwar_net_{args.arch}_warmup.pt"
        run_behavioral_cloning_warmup(
            model=model,
            dataset_path=args.warmup_dataset,
            output_checkpoint_path=out_save,
            epochs=args.bc_epochs,
            batch_size=args.batch_size if args.batch_size <= 2048 else 1024,
            lr=args.lr,
            device=dev,
        )
    elif args.mode == "distill":
        if not args.distill_dataset:
            print("Error: --distill-dataset required for mode=distill")
            sys.exit(1)
        if not args.warmup_checkpoint:
            print("Error: --warmup-checkpoint required for mode=distill -- X4a distils FROM a "
                  "trained policy; starting from random weights would answer a different "
                  "question.")
            sys.exit(1)
        dev = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
        agent_d = load_agent(args.warmup_checkpoint, device=dev)
        model_d = getattr(agent_d, "model", None)
        if model_d is None:
            print(f"Error: {args.warmup_checkpoint} did not load as a neural agent")
            sys.exit(1)
        out_save = args.output_dir or data_path("checkpoints", "distilled_search.pt")
        run_search_distillation(
            model=model_d,
            dataset_path=args.distill_dataset,
            output_checkpoint_path=out_save,
            epochs=args.distill_epochs,
            batch_size=args.batch_size if args.batch_size <= 2048 else 512,
            lr=args.distill_lr,
            device=dev,
        )
    elif args.mode == "eval":
        if not args.warmup_checkpoint:
            print("Error: --warmup-checkpoint (or model path) required for mode=eval")
            sys.exit(1)
        agent_main = load_agent(args.warmup_checkpoint, device=args.device)
        opponents = [load_agent(spec, device=args.device) for spec in args.eval_opponents]

        print(f"=== Evaluating Agent \"{agent_main.name}\" against {len(opponents)} Opponents ({args.eval_games_per_side*2} games each) ===")
        for opp in opponents:
            res = BatchMatchRunner.play_parallel_matchup(
                agent_main, opp, games_per_side=args.eval_games_per_side, temperature=0.1)
            print(f"\nMatchup: {agent_main.name} vs {opp.name}")
            print(f"  Win Rate: {res['win_rate_a']*100:.1f}% ({res['a_wins']}W - {res['b_wins']}L - {res['draws']}D)")
            print(f"  Avg Turn: {res['avg_turn']:.1f} | Avg Steps: {res['avg_steps']:.0f} | Avg VP Margin: {res['avg_vp_margin_a']:+.1f}")
            print(f"  Ending Causes: {res['causes']}")


if __name__ == "__main__":
    main()
