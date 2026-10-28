import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Unified Generic CLI Training & Evaluation Runner for Twilight Struggle AI.

import argparse
import os
import sys
import torch

from ai.training.generic_trainer import (train_pipeline, run_behavioral_cloning_warmup, obs_feature_bits,
                                         DEFAULT_OBS_FEATURES, OBS_FEATURE_BITS)
from ai.training.generic_trainer import run_search_distillation
from ai.training.generic_trainer import resolve_resume
from ai.training.policy_fit import PolicyFitConfig, run_policy_fit
from ai.training.q_fit import QFitConfig, run_q_fit
from ai.training.value_fit import ValueFitConfig, parse_arch, run_value_fit
from tools.lib.data_root import data_path
from tools.lib.openings import FORCED_OPENINGS_US, FORCED_OPENINGS_USSR
from tools.lib.player_agent import load_agent
from tools.lib.batch_tournament import BatchMatchRunner
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


def _resolve_head_center(args: argparse.Namespace) -> bool:
    """--ladder-head-center, with its default ("auto") resolved.

    On since 2026-09-25 for every ladder network with per-entity heads in the E4 view: centring
    removes the country-logit shift the E4 policy cannot see, which otherwise drifts until a long
    TF32 run diverges, at no measured cost (research/log/E4_long_runs.md). Auto follows the
    checkpoint a run starts from -- a resume or a warm start keeps the setting its weights were
    trained with, since the buffer that records it is part of the state dict -- and stays off in
    the merged (E4.1) view, where countries do compete with play modes. An explicit
    --ladder-head-center / --no-ladder-head-center always wins.
    """
    if args.ladder_head_center is not None:
        return bool(args.ladder_head_center)
    if not getattr(args, "per_entity_heads", 0):
        return False
    src = getattr(args, "resume", None) or getattr(args, "warmup_checkpoint", None)
    if src:
        path = resolve_resume(src) if getattr(args, "resume", None) else src
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        return "pe_center" in sd
    return not bool(getattr(args, "merged_influence", False))


def _ladder_config(args: argparse.Namespace) -> "dict[str, object] | None":
    """The P21 backbone configuration, or None when another architecture was asked for.

    Every axis is required rather than defaulted. A defaulted variant argument is how a model
    silently becomes a different architecture from the one the run's metadata claims -- the
    failure this repository has already had five times with a defaulted observation layout.
    """
    if args.arch != "ladder":
        if getattr(args, "ladder_token_layers", 0):
            raise SystemExit("--ladder-token-layers only applies to --arch ladder; refused rather "
                             "than silently ignored.")
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
        head_center=_resolve_head_center(args),
        identity_dim=int(args.identity_dim),
        drop_static=bool(args.drop_static),
        hidden_dim=int(args.ladder_hidden_dim),
        num_res_blocks=int(args.ladder_res_blocks),
        # Recorded only off its historical 0.05, so the configs of earlier runs stay comparable.
        **({"res_dropout": float(args.ladder_res_dropout)} if float(args.ladder_res_dropout) != 0.05 else {}),
        # Recorded only when on, so the configs of earlier runs stay comparable.
        **({"trunk_norm": True} if args.ladder_trunk_norm else {}),
        **({"res_post_norm": True} if args.ladder_res_post_norm else {}),
        **({"sphere_blocks": int(args.ladder_sphere_blocks), "sphere_expansion": int(args.ladder_sphere_expansion)}
           if int(args.ladder_sphere_blocks) > 0 else {}),
        **({"sphere_out_scale": float(args.ladder_sphere_out_scale)}
           if int(args.ladder_sphere_blocks) > 0 and float(args.ladder_sphere_out_scale) != 1.0 else {}),
        **({"head_norm": True} if args.ladder_head_norm else {}),
        num_attn_heads=4,
        card_lookup=bool(args.ladder_card_lookup),
        # Inert when the lookup is off, and LadderNet never reads them in that case.
        card_lookup_heads=int(args.ladder_card_lookup_heads or 0),
        card_lookup_dim=int(args.ladder_card_lookup_dim or 0),
        card_lookup_identity_dim=int(args.ladder_card_lookup_identity_dim or 0),
        # P30 C1. Recorded only when on, so the configs of earlier runs stay comparable.
        **({"token_layers": int(args.ladder_token_layers), "token_dim": int(args.ladder_token_dim)}
           if args.ladder_token_layers else {}),
        # Owner 2026-10-06. Recorded only when on, as the token path.
        **({"branch_head": True} if getattr(args, "ladder_branch_head", False) else {}),
        # Owner 2026-10-16 (Cell A). Recorded only when on.
        **({"region_attn_dim": int(args.ladder_region_attn_dim),
            "region_attn_heads": int(args.ladder_region_attn_heads)}
           if getattr(args, "ladder_region_attn_dim", 0) else {}),
        **({"play_mode_head": True} if getattr(args, "ladder_play_mode_head", False) else {}),
        # Owner 2026-10-09. Recorded only when on.
        **({"logit_cap": float(args.ladder_logit_cap)} if getattr(args, "ladder_logit_cap", 0.0) else {}),
        **({"logit_cap_grad": str(args.ladder_logit_cap_grad)}
           if getattr(args, "ladder_logit_cap", 0.0) and getattr(args, "ladder_logit_cap_grad", "tanh") != "tanh" else {}),
        **({"logit_cap_leak": float(args.ladder_logit_cap_leak)}
           if getattr(args, "ladder_logit_cap", 0.0) and getattr(args, "ladder_logit_cap_leak", 0.0) else {}),
        **({"logit_cap_leak_up": float(args.ladder_logit_cap_leak_up)}
           if getattr(args, "ladder_logit_cap", 0.0) and getattr(args, "ladder_logit_cap_leak_up", 0.0) else {}),
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
    lad.add_argument("--ladder-res-dropout", type=float, default=0.05,
                     help="Dropout inside each residual block. 0.05 is the historical default; use 0 for RL: the "
                          "rollouts act in eval mode and the update in train mode, so dropout makes the update's "
                          "log-probabilities and KL those of a policy that never played (2026-10-22).")
    lad.add_argument("--ladder-trunk-norm", action="store_true",
                     help="A LayerNorm on the trunk output after the residual blocks, so the heads read a "
                          "bounded input. Without it the pre-LN blocks' sum grows unbounded (A22: RMS 28.9 by "
                          "400M, A8 0.37) and the play-mode, Ops-mode and branch heads' logits reach spreads "
                          "of thousands (2026-10-23). Needs --ladder-res-blocks > 0.")
    lad.add_argument("--ladder-sphere-blocks", type=int, default=0,
                     help="SimbaV2's hyperspherical trunk after the fusion (P38 rung 2): this many blocks with the "
                          "stream on the unit sphere, a learned interpolation residual and unit-norm weight rows "
                          "re-projected after every optimiser step (ai/models/hyperspherical.py). 0 is off. "
                          "Replaces --ladder-res-blocks and --ladder-trunk-norm.")
    lad.add_argument("--ladder-sphere-expansion", type=int, default=4,
                     help="Hidden width of each sphere block as a multiple of the fusion width (SimbaV2: 4).")
    lad.add_argument("--ladder-sphere-out-scale", type=float, default=1.0,
                     help="The per-element RMS of the sphere trunk's output, the heads' input (1 by default; A8's "
                          "heads read ~0.4-0.55). At 1 the per-update KL sat at ~3x the control's whatever the "
                          "learning rate (P38 rung 2b).")
    lad.add_argument("--ladder-head-norm", action="store_true",
                     help="A LayerNorm between the first Linear and the GELU of the per-entity, branch and play-mode "
                          "heads, as the base policy head has. Only those heads go dormant (the plateaued control, "
                          "R48 and the sphere trunk alike; 2026-10-26). Registered last, so it can be added at a "
                          "branch point with --resume.")
    lad.add_argument("--ladder-res-post-norm", action="store_true",
                     help="A LayerNorm after each residual block's add (post-LN), so the residual stream is "
                          "renormalised at every block and cannot grow. With pre-LN blocks alone it grew in a few "
                          "dimensions even under --ladder-trunk-norm (A25: RMS 37, effective rank of the heads' "
                          "input ~22 of 768 by 2,400M; research/method/network_stability.md). Needs "
                          "--ladder-res-blocks > 0.")
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
    lad.add_argument("--ladder-region-attn-dim", type=int, default=0,
                     help="Cell A (P33): width of one masked attention layer over the per-country head's inputs, "
                          "each country reading its region and map neighbours, zero-initialised so a resumed "
                          "state plays as before. 0 = off.")
    lad.add_argument("--ladder-region-attn-heads", type=int, default=4,
                     help="Heads of the --ladder-region-attn-dim layer.")
    lad.add_argument("--ladder-token-layers", type=int, default=0,
                     help="P30 C1: transformer layers over card and country tokens (plus one\n"
                          "global token), beside the grouped projections; the global token joins\n"
                          "the fusion input and each per-entity head reads its own token.\n"
                          "0 (default) = off. --ladder-input-mode grouped only.")
    lad.add_argument("--ladder-branch-head", action="store_true", default=False,
                     help="A head on the branch block (event branches, DEFCON values, regions) reading "
                          "the trunk plus a one-hot of the resolving card, zero-initialised; a --resume "
                          "from a state without it adds it. Off by default.")
    lad.add_argument("--ladder-play-mode-head", action="store_true", default=False,
                     help="A head on the play-mode block (EVENT, SPACE, the three Ops modes) reading the "
                          "trunk plus a one-hot of the card being played, zero-initialised; addable on "
                          "--resume. Off by default.")
    lad.add_argument("--ladder-logit-cap", type=float, default=0.0,
                     help="Bound each legal move's logit deficit to the top legal move to C through "
                          "C*tanh(deficit/C), so no policy saturates (owner 2026-10-09). Greedy play "
                          "unchanged; addable on --resume. 0 (default): off.")
    lad.add_argument("--ladder-logit-cap-grad", type=str, default="tanh", choices=["tanh", "straight"],
                     help="How the gradient passes the logit cap: through the tanh (default), or "
                          "straight-through as if uncapped -- the tanh's derivative is ~0.001 at a "
                          "saturated move, so it barely learns. Play is identical either way.")
    lad.add_argument("--ladder-logit-cap-leak", type=float, default=0.0,
                     help="With the tanh cap gradient: add this times the raw gradient (forward unchanged), "
                          "so a saturated move learns at ~this share of the full gradient. 0 (default): off.")
    lad.add_argument("--ladder-logit-cap-leak-up", type=float, default=0.0,
                     help="With the tanh cap gradient: add this times the raw gradient only where it "
                          "raises a logit -- a saturated move can be pulled back, never pushed further "
                          "down (an unsatisfiable push that drifts the network). 0 (default): off.")
    lad.add_argument("--ladder-token-dim", type=int, default=128,
                     help="Token width for --ladder-token-layers (default 128).")
    lad.add_argument("--ladder-head-center", action=argparse.BooleanOptionalAction, default=None,
                     help="Centre the per-entity heads' hidden features across entities before "
                          "their final projection, removing the country-logit shift the E4 policy "
                          "cannot see and that otherwise drifts until a long TF32 run diverges "
                          "(research/log/E4_long_runs.md). Default (auto): on for per-entity heads in "
                          "the E4 view; a resume or warm start follows its checkpoint; off with "
                          "--merged-influence."),
    lad.add_argument("--ladder-head-entities", type=str, default=None,
                     choices=["both", "country", "card"],
                     help="Which per-entity heads exist. The card-collision finding predicts "
                          "`country` carries most of M2's gain (arms M2d/M2e).")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "warmup", "eval", "curriculum", "distill", "value-fit", "policy-fit", "q-fit"],
                        help="Execution mode. `distill` is P15-X4a: soft cross-entropy "
                             "toward a searcher's visit distribution, from a dataset made "
                             "by tools/generate_search_targets.py. `value-fit` is P34's "
                             "supervised evaluator on a tools/value_labels.py dataset "
                             "(ai/training/value_fit.py), from --warmup-checkpoint's weights "
                             "or, with --value-reinit, its shape.")

    # Warm-up / Checkpoint options
    parser.add_argument("--warmup-checkpoint", "--load-path", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--resume-weights", type=str, default=None,
                        help="With --resume: after restoring the run's state (steps, optimizer, opponent pool, RNG), load "
                             "these weights (same architecture) into the network and pi_ref -- e.g. RL continued from a soup.")
    parser.add_argument("--resume-widen", action="store_true", default=False,
                        help="With --resume: the model's grouped input projections (--ladder-entity-proj-dim) are an "
                             "integer multiple of the state's; widen the network, pi_ref and Adam's moments without "
                             "changing what it plays (ai/models/grow.py) and continue.")
    parser.add_argument("--resume-reset-policy-head", action="store_true", default=False,
                        help="With --resume: re-initialise the policy head (policy_head, the per-entity heads and their "
                             "trunk context, the branch and play-mode heads) with fresh weights and Adam moments, keeping "
                             "the trunk and the value heads -- P37's last-layer reset, once at the branch point, aimed at "
                             "the policy head's measured plasticity loss (P37a).")
    parser.add_argument("--resume-grow-tokens", action="store_true", default=False,
                        help="With --resume: the model has --ladder-token-layers the state lacks. The token path "
                             "starts fresh and the fusion layer and per-entity heads read it through zero weights, "
                             "so the network plays as saved and the path learns from the first step "
                             "(ai/models/grow.py; P35 item 1's grown read).")
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
    pf = parser.add_argument_group("policy fit (--mode policy-fit, P36c)")
    pf.add_argument("--policy-dataset", type=str, default=None,
                    help="A tools/distill_targets.py .npz: the keyed root's improved policy on entry decisions.")
    pf.add_argument("--policy-worlds", type=int, default=128, help="Which world count's targets to fit.")
    pf.add_argument("--policy-target", type=str, default=None,
                    help="Which teacher's targets to fit, by its distill_targets.py --teacher name (overrides --policy-worlds).")
    pf.add_argument("--policy-reinit-heads", action="store_true", default=False,
                    help="Re-initialise the policy heads, keeping the trained trunk (plasticity).")
    pf.add_argument("--policy-reinit", action="store_true", default=False,
                    help="Train --warmup-checkpoint's architecture from fresh weights instead of fine-tuning it.")
    pf.add_argument("--policy-arch", nargs="+", default=None, metavar="KEY=VALUE",
                    help="With --policy-reinit: change the network's ladder settings (a larger student).")
    pf.add_argument("--policy-epochs", type=float, default=60.0, help="At most this many passes.")
    pf.add_argument("--policy-lr", type=float, default=3e-4)
    pf.add_argument("--policy-batch", type=int, default=4096)
    pf.add_argument("--policy-eval-every", type=int, default=200)
    pf.add_argument("--policy-head-only", action="store_true", default=False,
                    help="Freeze the network; train only a correction head on its trunk features added to its logits.")
    pf.add_argument("--policy-kl-anchor", type=float, default=0.0,
                    help="Weight of KL(initial policy || student) on every position: a fine-tune held near its start.")
    pf.add_argument("--policy-train-fraction", type=float, default=1.0,
                    help="Use this share of the training games (whole games): the data-scaling cells.")
    pf.add_argument("--policy-patience", type=int, default=10,
                    help="Stop when the training fit has not risen by 0.2 points in this many evaluations (convergence).")
    qf = parser.add_argument_group("action-value fit (--mode q-fit): the root's per-candidate leads as values")
    qf.add_argument("--q-dataset", type=str, default=None,
                    help="A tools/distill_targets.py .npz carrying a root teacher's candidate statistics (lead_T, lead_se_T).")
    qf.add_argument("--q-target", type=str, default=None, help="Which teacher's leads to fit, by its --teacher name.")
    qf.add_argument("--q-head", choices=["mlp", "policy"], default="mlp",
                    help="mlp: a fresh head on the trunk's features; policy: the network's logits times a learned scale.")
    qf.add_argument("--q-train-trunk", action="store_true", default=False,
                    help="Train the copy's trunk too (a GPU); otherwise the trunk is frozen and its features cached.")
    qf.add_argument("--q-weighting", choices=["inverse_variance", "unit"], default="inverse_variance")
    qf.add_argument("--q-se-floor", type=float, default=0.01,
                    help="Added in quadrature to every lead's SE: no pair weighs more than 1 / floor^2.")
    qf.add_argument("--q-baseline", type=str, default="h0", help="A second teacher's leads read as the bar.")
    qf.add_argument("--q-judge", type=str, default=None,
                    help="The teacher whose two seeds judge the held-out rows (default --q-target).")
    qf.add_argument("--q-epochs", type=float, default=40.0)
    qf.add_argument("--q-lr", type=float, default=3e-4)
    qf.add_argument("--q-batch", type=int, default=1024)
    qf.add_argument("--q-eval-every", type=int, default=200)
    qf.add_argument("--q-patience", type=int, default=10)
    qf.add_argument("--q-train-fraction", type=float, default=1.0,
                    help="Fit on this share of the training games (whole games): the data-scaling cells.")
    vf = parser.add_argument_group("value fit (--mode value-fit, P34)")
    vf.add_argument("--value-dataset", type=str, default=None,
                    help="A tools/value_labels.py dataset directory (train.npz, holdout.npz, siblings.npz, manifest.json).")
    vf.add_argument("--value-reinit", action="store_true", default=False,
                    help="Train --warmup-checkpoint's architecture from fresh weights instead of fine-tuning its weights.")
    vf.add_argument("--value-arch", nargs="+", default=None, metavar="KEY=VALUE",
                    help="With --value-reinit: change the network's ladder settings, e.g. hidden_dim=768 "
                         "num_res_blocks=8 (A3's deep shape) or token_layers=2 token_dim=128 (A6's tokens).")
    vf.add_argument("--value-epochs", type=float, default=20.0, help="At most this many passes over the training positions.")
    vf.add_argument("--value-lr", type=float, default=2e-4)
    vf.add_argument("--value-batch", type=int, default=4096, help="Positions per step.")
    vf.add_argument("--value-sibling-groups", type=int, default=256,
                    help="Sibling (position, world) groups per step for the paired loss.")
    vf.add_argument("--value-sibling-coef", type=float, default=10.0, help="Weight of the paired sibling loss.")
    vf.add_argument("--value-vp-coef", type=float, default=1.0, help="Weight of v_vp's squared error to the final VP (/20).")
    vf.add_argument("--value-aux-coef", type=float, default=0.1,
                    help="Weight of the auxiliary head (turn-end VP and DEFCON, region swings).")
    vf.add_argument("--value-eval-every", type=int, default=250, help="Steps between held-out evaluations.")
    vf.add_argument("--value-siblings", type=str, default=None,
                    help="The sibling slice from this dataset directory (e.g. a keyed relabel) instead of --value-dataset's.")
    vf.add_argument("--value-sibling-loss", choices=["world", "decision"], default="world",
                    help="world: the paired loss per (decision, world); decision: per decision, the mean over its shared "
                         "worlds of each candidate's leaf value fitted to the mean of its results (P34 revision 2).")
    vf.add_argument("--value-sibling-decisions", type=int, default=32, help="Decisions a step under --value-sibling-loss decision.")
    vf.add_argument("--value-leaf-coef", type=float, default=0.0,
                    help="Weight of each sibling leaf's own cross-entropy to its world's result.")
    vf.add_argument("--value-leaf-weights", action="store_true", default=False,
                    help="Weight the main rows by how often the rollout root reads their kind of decision at a leaf.")
    vf.add_argument("--value-patience", type=int, default=6,
                    help="Stop after this many evaluations without a better held-out loss.")

    # Budget & Snapshot parameters. Both are in env steps, deliberately: a wall-clock budget
    # cannot make two arms comparable, because steps/sec depends on the policy.
    parser.add_argument("--vp-potential", type=float, default=0.0,
                        help="Potential-based VP shaping on top of --reward-scheme (0 = off): "
                             "Phi = this x VP from the US side, Phi(terminal) = 0, and each step "
                             "adds Phi(s') - Phi(s) for the mover. Over a game it sums to a "
                             "constant, so the optimal policy is unchanged; it only pays VP "
                             "gains when they happen and takes them back at the end.")
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
                             "biases against build-to-final-scoring play, so watch the ending mix. "
                             "Applies to --reward-scheme blunder_aware, useful_actions and "
                             "curriculum (both stages); refused with terminal and shaped.")
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
    parser.add_argument("--inject-setup-only", action="store_true", default=False,
                        help="P4 arm B: inject only the corpus's setup placements (phase SETUP), policy "
                             "loss only -- a human anchor for the opening and nothing else.")
    parser.add_argument("--block-lambda", choices=["off", "setup", "setup-side", "same-side"],
                        default=None,
                        help="P4: GAE lambda = 1 inside decision blocks. setup: every consecutive "
                             "SETUP-phase pair, across both sides (= --setup-block-lambda, arm A); "
                             "setup-side: the same, cut where the mover changes (A1); same-side: any "
                             "consecutive pair of one game with the same mover and the engine's RNG "
                             "unchanged -- no die roll, card draw or hidden chance between (A2). "
                             "Default: off (setup-side was the default 2026-09-26..27; at flat rollout "
                             "temperature it added no strength and froze the setup, "
                             "research/log/E5_12_setup_credit_at_flat_temperature.md), or setup with "
                             "--setup-block-lambda.")
    parser.add_argument("--setup-block-lambda", action="store_true", default=False,
                        help="P4 arm A: GAE lambda = 1 between consecutive setup placements, so the "
                             "setup's credit telescopes to the turn-1 headline position instead of "
                             "bootstrapping from the critic at half-placed boards.")
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
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW's decoupled weight decay in the RL update (1e-4, every run before P38; SimBa uses "
                             "1e-2 -- P38 rung 1).")
    parser.add_argument("--no-decay-norms-biases", dest="decay_norms_biases", action="store_false",
                        help="Leave LayerNorm gains and biases (every parameter of fewer than two dimensions) "
                             "undecayed, in a second optimiser group. Default: one group, everything decayed, as "
                             "before P38.")
    # P39 arm 5 (research/plans/P39_deep_trunk_optimisation.md): the optimiser's other parameters.
    parser.add_argument("--adam-beta2", type=float, default=0.999,
                        help="AdamW's beta2 (P39 arm 5 tries 0.95: a shorter second-moment memory).")
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="AdamW's epsilon (P39 arm 5 tries 1e-6: the sphere trunk's unit-norm rows give tiny "
                             "gradient entries).")
    parser.add_argument("--policy-head-lr-mult", type=float, default=1.0,
                        help="P39 arm 5: the policy heads' learning rate as a multiple of --lr (the heads saturate "
                             "first and drag the trunk). 1, the default, is one rate for every parameter.")
    # P39 arm 4: muP against the A8 control's widths.
    parser.add_argument("--mup", action="store_true",
                        help="muP (ai/models/mup.py): each matrix's rate scaled by the base fan-in over its own, the "
                             "readouts zeroed at initialisation; at the base widths the rates are --lr unchanged.")
    parser.add_argument("--mup-base-proj-dim", type=int, default=256,
                        help="--mup's base entity-projection width (A8's).")
    parser.add_argument("--mup-base-hidden-dim", type=int, default=480,
                        help="--mup's base fusion width (A8's); a parameter the base lacks is scaled against it.")
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
    parser.add_argument("--adv-norm-learner-only", action="store_true", default=False,
                        help="Take the advantage normalisation's mean and spread over the "
                             "learner's own transitions, not a frozen pool opponent's too. The "
                             "opponent's advantages are roughly the negative of the learner's, so "
                             "averaged in they leave the learner's own uncentred; against a "
                             "near-equal frozen opponent that collapses the learner (P24).")
    parser.add_argument("--opponent-temperature", type=float, default=None,
                        help="P24: the temperature pool opponents play at in rollouts; 0 is "
                             "greedy, the policy a greedy tournament rates. Default: sampled "
                             "exactly as the learner is.")
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
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="P39 arm 3: PPO's clipping epsilon (the ratio is clipped to 1 +- this).")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="P39 arm 3: optimisation epochs over each iteration's rollout.")
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
    parser.add_argument("--setup-entropy-floor", type=float, default=0.0,
                        help="An entropy FLOOR, in nats, on the learner's setup placements: an extra "
                             "bonus on setup decisions only, whose coefficient moves by "
                             "--setup-entropy-lr x (floor - rollout setup entropy) per iteration within "
                             "[0, --setup-entropy-max-coef]. A setup placement at p ~ 1 is never "
                             "resampled and its opening locks, good or bad "
                             "(research/log/E5_11_setup_lock_and_critic_views.md). 0 is off.")
    parser.add_argument("--setup-entropy-lr", type=float, default=0.01)
    parser.add_argument("--setup-mc-credit", action="store_true", default=False,
                        help="Credit the learner's setup placements with the game result (Monte "
                             "Carlo, advantage = result - V) instead of the lambda-return, which at "
                             "gamma 1 / lambda 0.98 reaches the setup only through the critic's "
                             "values of the next positions -- and the critic over-rates unfamiliar "
                             "openings (research/log/E5_11_setup_lock_and_critic_views.md). As "
                             "Ataraxos trains its setup. Pair with --setup-entropy-floor: a setup "
                             "played at p ~ 1 is never compared with anything.")
    parser.add_argument("--setup-mc-coef", type=float, default=1.0)
    parser.add_argument("--setup-script-frac", type=float, default=0.0,
                        help="Fraction of games (drawn per game) whose setup is played, for BOTH sides, by "
                             "one of --setup-script-openings, and trained on as if the policy had chosen "
                             "it: the stored log-prob is the policy's own, and the credit is the game result "
                             "(requires --setup-mc-credit). Lets the policy play, and be credited for, "
                             "openings it never samples itself (research/log/E7_shallow_setup_lock.md). "
                             "0 = off.")
    parser.add_argument("--play-mode-temp", type=float, default=1.0,
                        help="Sampling temperature for the learner's play-mode decisions only (event, "
                             "space or Ops for a chosen card), so events the policy rarely plays still get "
                             "tried; the stored log-prob stays the policy's own. 1.0 (default) = off. The "
                             "expert review's first candidate (research/log/expert_review_E7.md).")
    # P31 (research/log/P31_branch_arms_B1_B2.md): show the learner rare situations, let training
    # decide what they are worth. Every flag is off by default.
    parser.add_argument("--play-mode-floor", type=float, default=0.0,
                        help="P31 1a: the learner's behaviour policy at play-mode decisions and the "
                             "non-country choices inside events is mu = (1-eps)*pi + eps*uniform(legal). "
                             "The stored log-prob stays the policy's own, and each floor sample's "
                             "surrogate is weighted by pi_old/mu (at most 1/(1-eps)) outside PPO's "
                             "clip. 0 is off.")
    parser.add_argument("--floor-scope", choices=["all", "seeded"], default="all",
                        help="P31 1a: apply the floor in every game, or only in --seed-scenarios games.")
    parser.add_argument("--floor-from", type=int, default=0,
                        help="P31 1a: the floor is 0 before this many total env steps.")
    parser.add_argument("--floor-anneal-from", type=int, default=None,
                        help="P31 1a: from this many steps the floor falls linearly to 0 over "
                             "--floor-anneal-steps. Unset: no anneal.")
    parser.add_argument("--floor-anneal-steps", type=int, default=0,
                        help="P31 1a: the length of the anneal.")
    parser.add_argument("--seed-scenarios", nargs="+", default=None,
                        choices=["subs", "chernobyl", "cmc_combo"],
                        help="P31 1b: in --seed-frac of games, the listed card is played for its event "
                             "for the US as ENVIRONMENT the first time the US holds it at an action "
                             "round's card play (Chernobyl's region drawn uniformly); the forced plays "
                             "get no policy gradient. cmc_combo instead forces the US's whole scripted "
                             "Cuban Missile Crisis combo (tools/lib/scripted_rules.py) wherever it applies.")
    parser.add_argument("--seed-frac", type=float, default=0.0,
                        help="P31 1b: the fraction of games seeded, drawn afresh at every game start.")
    parser.add_argument("--seed-scenarios-from", type=int, default=0,
                        help="P31 1b: no game is seeded before this many total env steps.")
    parser.add_argument("--aux-opp-legality", type=float, default=0.0,
                        help="P31 1d: weight of an auxiliary head predicting, per country, whether the "
                             "opponent may place influence and may coup there at its NEXT decision "
                             "(read from its own observation then). Builds the head (opp_legal_head.*); "
                             "a --resume from a state without it adds it fresh. 0 is off.")
    parser.add_argument("--aux-opp-legality-frac", type=float, default=0.1,
                        help="P31 1d: the fraction of decisions sampled as positions to label.")
    parser.add_argument("--aux-opp-legality-min-batch", type=int, default=4096,
                        help="P31 1d: labelled positions needed before an update step.")
    parser.add_argument("--mode-cf-coef", type=float, default=0.0,
                        help="P31 1c: weight of the counterfactual mode credit. At 1 in "
                             "--mode-cf-subsample of the learner's floor decisions (play mode and the "
                             "non-country choices inside events), every legal option is played out to the "
                             "end by the current network on both sides from a clone of the state, and the "
                             "loss gets -c * sum_a pi(a) * (Q(a) - Q(taken)). 0 is off.")
    parser.add_argument("--mode-cf-subsample", type=int, default=16,
                        help="P31 1c: price 1 in this many of the learner's floor decisions.")
    parser.add_argument("--mode-cf-playouts", type=int, default=1,
                        help="P31 1c: games played per option; the option's value is their mean result.")
    parser.add_argument("--mode-cf-from", type=int, default=0,
                        help="P31 1c: no decision is priced before this many total env steps.")
    parser.add_argument("--turn-credit-coef", type=float, default=0.0,
                        help="P32 B4': weight of the turn-credit cross-entropy. Sampled learner decisions with 2..--turn-credit-cap options are priced at each rollout's end -- every option over --turn-credit-worlds worlds (a shared redeal of the hidden cards, independent dice), the current network greedy on both sides to the end of the turn, valued by its critic there or the result -- and trained toward pi' ∝ max(pi, floor) * exp(beta * Q), weighted by evidence (ai/training/turn_credit.py). 0 is off.")
    parser.add_argument("--turn-credit-budget", type=int, default=64,
                        help="P32 B4': decisions priced per rollout (nested ones included).")
    parser.add_argument("--turn-credit-worlds", type=int, default=16,
                        help="P32 B4': worlds per priced decision.")
    parser.add_argument("--turn-credit-beta", type=float, default=10.0,
                        help="P32 B4': nats per unit of turn-end value in the target.")
    parser.add_argument("--turn-credit-k", type=float, default=9.0,
                        help="P32 B4': a row's weight is d^2 / (d^2 + k * se^2) -- half weight at z = sqrt(k).")
    parser.add_argument("--turn-credit-floor", type=float, default=1e-3,
                        help="P32 B4': the policy probability floor in the target, so a saturated option stays reachable.")
    parser.add_argument("--turn-credit-nested-p", type=float, default=0.0,
                        help="P32 B4': share of the decider's small decisions met in the branches of options the policy does not prefer that are queued for pricing too (a pair learned from its end). 0 is off.")
    parser.add_argument("--turn-credit-nested-share", type=float, default=0.33,
                        help="P32 B4': at most this share of the budget for nested decisions.")
    parser.add_argument("--turn-credit-narrow-share", type=float, default=0.5,
                        help="P32 B4': share of the budget for narrow spots (max p > 0.9, some option below 0.02).")
    parser.add_argument("--turn-credit-cap", type=int, default=12,
                        help="P32 B4': the largest option set priced.")
    parser.add_argument("--turn-credit-steps", type=int, default=4,
                        help="P32 B4': CE optimiser steps after each PPO update.")
    parser.add_argument("--turn-credit-batch", type=int, default=512,
                        help="P32 B4': rows per CE step.")
    parser.add_argument("--turn-credit-buffer", type=int, default=8192,
                        help="P32 B4': priced rows kept (FIFO).")
    parser.add_argument("--turn-credit-workers", type=int, default=0,
                        help="P32 B4': price in this many worker processes, each with its own copy of the network, "
                             "beside the next rollout -- a rollout's rows arrive one iteration later. 0: price in the "
                             "trainer after each rollout.")
    parser.add_argument("--forced-openings-frac", type=float, default=0.0,
                        help="Owner 2026-10-13: each side, independently, in this share of games has its setup played by one of "
                             "--forced-openings-us / -ussr (tools/lib/openings.py), as ENVIRONMENT -- no policy gradient on the "
                             "forced placements -- so the network learns to play on from them without being taught to choose "
                             "them. us_marshall also headlines Marshall Plan on turn 1 when it is in hand (else another US "
                             "opening is played). 0 = off.")
    parser.add_argument("--forced-openings-us", nargs="*", default=list(FORCED_OPENINGS_US),
                        help="US openings for --forced-openings-frac (default: the owner's three).")
    parser.add_argument("--forced-openings-ussr", nargs="*", default=list(FORCED_OPENINGS_USSR),
                        help="USSR openings for --forced-openings-frac (default: the owner's two).")
    parser.add_argument("--force-applicable-events", nargs="+", default=None,
                        choices=["wargames", "arms_race", "one_small_step", "wargames_branch"],
                        help="Owner 2026-10-06: at the learner's play-mode decision for the listed card "
                             "with its event legal and applicable (Wargames: DEFCON 2 and 7+ VP ahead; "
                             "Arms Race: ahead in military Ops; One Small Step: behind in space), play the "
                             "event with probability --force-event-frac, trained as the policy's own play.")
    parser.add_argument("--force-event-frac", type=float, default=0.1,
                        help="The probability an applicable play is forced to the event.")
    parser.add_argument("--force-events-from", type=int, default=0,
                        help="No event is forced before this many total env steps.")
    parser.add_argument("--force-event-credit", choices=["own", "environment"], default="own",
                        help="own: the forced event is trained as the policy's own play; environment: "
                             "learner = 0 on it (no policy gradient), as --seed-scenarios. "
                             "wargames_branch (Wargames at DEFCON 2, any lead) is meant for environment: "
                             "it supplies visits to Wargames' branch, which the policy then decides.")
    parser.add_argument("--floor-rows", choices=["play_mode_and_events", "event_choices"],
                        default="play_mode_and_events",
                        help="Which decisions --play-mode-floor covers: play mode plus the non-country "
                             "choices inside events (default), or only the latter (event branches, "
                             "DEFCON values, regions).")
    parser.add_argument("--setup-script-openings", nargs="+", default=None,
                        help="Opening names from tools/lib/openings.py for --setup-script-frac, drawn "
                             "uniformly per scripted game. Default: the four human variants "
                             "(HUMAN_OPENING_MIX).")
    parser.add_argument("--setup-mc-min-batch", type=int, default=512,
                        help="Finished-game setup placements to gather before one MC setup step.")
    # P28 step 2 (research/log/P28_step2a_lr_schedule.md): optimiser levers at the plateau.
    parser.add_argument("--lr-schedule", choices=["constant", "step", "cosine"], default="constant",
                        help="Learning-rate schedule, in env steps from --lr-schedule-start. constant "
                             "(the default) keeps --lr throughout; step lowers it to each of "
                             "--lr-schedule-values in turn every --lr-schedule-every steps; cosine "
                             "decays it to --lr-min over --lr-schedule-span steps.")
    parser.add_argument("--lr-schedule-start", type=int, default=None,
                        help="Env step the schedule counts from. Default: where this run starts "
                             "(the resume point), so the steps already trained keep their rate.")
    parser.add_argument("--lr-schedule-every", type=int, default=60_000_000,
                        help="step schedule: steps between rate changes.")
    parser.add_argument("--lr-schedule-values", type=float, nargs="+", default=[1e-4, 3e-5],
                        help="step schedule: the rates after each interval; the last one holds.")
    parser.add_argument("--lr-schedule-span", type=int, default=200_000_000,
                        help="cosine schedule: steps from --lr to --lr-min.")
    parser.add_argument("--lr-min", type=float, default=3e-5, help="cosine schedule: the floor.")
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
                        help="P39 arm 1: ramp the learning rate linearly from 0 to its scheduled value over "
                             "the first N env steps (counted from step 0, so a resume mid-ramp continues it). "
                             "0, the default, is no warmup.")
    # P29 bet 2 (research/log/P29_bet2_ownership_probes.md): auxiliary targets from each game's end.
    parser.add_argument("--aux-ownership", type=float, default=0.0,
                        help="Weight of an auxiliary head predicting who controls each of the 84 "
                             "countries at game end (mine / opponent's / neither, in the mover's "
                             "frame; cross-entropy). 0, the default, builds no aux heads. Ladder "
                             "architecture only.")
    parser.add_argument("--aux-vp-margin", type=float, default=0.0,
                        help="Weight of an auxiliary head predicting the final VP margin (mover's "
                             "frame, VP/20; squared error). 0 is off.")
    parser.add_argument("--aux-sample-frac", type=float, default=0.1,
                        help="Fraction of decisions kept (at half precision) until their game ends "
                             "and labelled for the aux targets.")
    parser.add_argument("--aux-min-batch", type=int, default=4096,
                        help="Labelled positions to gather before one aux training step.")
    # The view spec (owner, 2026-10-01): optional observation blocks appended to the base layout.
    parser.add_argument("--obs-features", nargs="*", default=list(DEFAULT_OBS_FEATURES),
                        choices=sorted(OBS_FEATURE_BITS),
                        help="Optional observation features the network reads. ops_budget: a block "
                             "appended after the base v2.3 layout, the Ops the card at play will be "
                             "granted (at SELECT_PLAY_MODE) and each side's per-card Ops modifier "
                             "(Containment / Brezhnev / Red Scare). headline_reveal (no width): while "
                             "the side holding Space Race box 4 alone chooses its headline, the "
                             "opponent's, chosen first and revealed, shows in its card row. The list "
                             "replaces the default, which since E8 is headline_reveal: name both to "
                             "have both, and give the flag with no names for none (what every run "
                             "before E8 read). Recorded in the run's metadata and in the weights; "
                             "every evaluator then builds this view for the model. Ladder only.")
    # P30 (research/log/P30_card_board_targets.md): what each held card's event and Ops would do.
    parser.add_argument("--aux-card-events", type=float, default=0.0,
                        help="Weight of an auxiliary per-card head predicting, for every card in the "
                             "mover's hand, what its event would do on this board (VP, DEFCON, regional "
                             "margins, battlegrounds, influence) and what its Ops could take -- labelled "
                             "by the engine at the decision (ai.training.card_event_targets); masked MSE "
                             "on standardised targets. 0, the default, builds no head. Ladder only.")
    parser.add_argument("--aux-card-sample-frac", type=float, default=0.0005,
                        help="Fraction of decisions labelled for --aux-card-events. A label costs ~2 ms "
                             "of engine calls in Python, so 0.0005 is ~50 a second at 96k steps/s, ~10%% "
                             "of the time; watch card_aux_label_s against the iteration time.")
    parser.add_argument("--aux-card-min-batch", type=int, default=2048,
                        help="Labelled positions in the buffer before --aux-card-events steps begin.")
    parser.add_argument("--aux-card-buffer", type=int, default=65536,
                        help="FIFO buffer of labelled positions that the --aux-card-events steps draw from.")
    parser.add_argument("--aux-card-steps", type=int, default=4,
                        help="--aux-card-events optimiser steps per iteration, each on one minibatch "
                             "drawn from the buffer.")
    parser.add_argument("--aux-card-batch", type=int, default=512,
                        help="Positions per --aux-card-events minibatch.")
    parser.add_argument("--ema-weights", type=float, default=0.0,
                        help="Keep an exponential moving average of the weights with this time "
                             "constant in env steps (0, the default, keeps none). Snapshots, pool "
                             "members and evaluations then use the average; the live weights keep "
                             "training, and the resume state holds both.")
    parser.add_argument("--forced-opening", type=str, default=None,
                        help="Start every training game after a scripted setup (a name from "
                             "tools/lib/openings.py, e.g. 'human': USSR East Germany 1, Poland 4, "
                             "Yugoslavia 1; US West Germany 4, Italy 3, Iran 2). The setup is never "
                             "learned, so rate the result with tools/tournament.py --opening or an "
                             "opening:<name>: spec prefix.")
    parser.add_argument("--setup-entropy-max-coef", type=float, default=1.0)
    parser.add_argument("--logit-gap-coef", type=float, default=0.0,
                        help="Penalty coef * mean over legal moves of relu(top - logit - --logit-gap)^2 "
                             "(owner 2026-10-09): keeps a saturated policy learnable. 0 (default): off.")
    parser.add_argument("--logit-gap", type=float, default=7.0,
                        help="The logit gap below the top legal move that --logit-gap-coef starts at.")
    parser.add_argument("--phase-lock", type=str, default=None, metavar="PATH",
                        help="Two arms on one machine take turns: the rollout under PATH.rollout, the update under "
                             "PATH.update (flock), so one collects while the other updates. Each arm stays exactly "
                             "serial; only the wall clock changes. Give both arms the same PATH.")
    parser.add_argument("--ladder-token-chunk", type=int, default=0, metavar="ROWS",
                        help="Run the token path in passes of ROWS rows, each checkpointed under autograd "
                             "(recomputed in the backward, not held): the same function at a fraction of the "
                             "memory, for --ladder-token-layers at a large --batch-size (A18 at 16,384 needs "
                             "~20 GB without it). 0 (default): one pass.")
    parser.add_argument("--z-loss-coef", type=float, default=0.0,
                        help="z-loss: coef * mean(logsumexp(policy logits)^2) in the update (PaLM uses "
                             "1e-4). Bounds the logits' level, which the softmax leaves free and which "
                             "drifts upward without limit; both 800M runs diverged through it "
                             "(research/log/E4_long_runs.md). 0 is off.")
    parser.add_argument("--compile-update", choices=["off", "default", "max-autotune"],
                        default="off",
                        help="torch.compile the PPO update's forwards (P26); the rollout keeps "
                             "its CUDA graphs of the eager network. Reorders arithmetic, so off "
                             "until an A/B passes. max-autotune is inductor's "
                             "max-autotune-no-cudagraphs: longer first compile, faster kernels.")
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
    parser.add_argument("--search-gumbel-k", type=int, default=0,
                        help="P32 T1: >0 makes the search CE target the Gumbel root's improved policy "
                             "(softmax of logit/T + sigma(completed Q) over every legal move, k candidates "
                             "halved over --search-sims evaluations) instead of PUCT visit counts. 0 (default): "
                             "visit counts. Meant for a capped policy (--ladder-logit-cap): a saturated "
                             "policy is beyond sigma's reach.")
    parser.add_argument("--search-prior-temperature", type=float, default=1.0,
                        help="P32 T1: the Gumbel root's logits are divided by this (1 = the paper's rule).")
    parser.add_argument("--search-value-beta", type=float, default=0.0,
                        help="P32 T2: at a searched decision, v_win's target becomes (1 - beta) * the "
                             "lambda-return + beta * the search value (the improved policy's expected "
                             "completed Q at the Gumbel root). Needs --search-gumbel-k > 0; runs the "
                             "searcher with --search-ce-coef 0 too. The advantages are untouched. 0 = off.")
    parser.add_argument("--search-rollout-horizon", type=int, default=0,
                        help="Rollout-root search: playouts of this many action-round boundaries (0 = off; where they stop is --search-rollout-stop). With --search-ce-coef > 0 it replaces the Gumbel root of the search-CE targets (ai/search/rollout_root.py): the target is softmax(logit + sigma(completed Q)) with the candidates' rollout values, and needs no --search-gumbel-k. With --search-value-beta > 0 and no CE it gives value targets instead (P33's critic track): the greedy continuation's mean leaf value, blended into v_win's target.")
    parser.add_argument("--search-rollout-stop", choices=["boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal"], default="boundaries",
                        help="Where the rollout root's playouts stop: after --search-rollout-horizon boundaries (the default, the measured root), at the end of the decision's turn, or at the later or the earlier of the two.")
    parser.add_argument("--searched-games-workers", type=int, default=0,
                        help="P33: worker processes playing whole games with the rollout root (k=4, 16 worlds, stop=earlier, leaf=own) on pi_ref for both sides; each decision, with the game's result from its decider's side, trains the value head only (--searched-value-coef). 0 = off.")
    parser.add_argument("--searched-games-envs", type=int, default=32,
                        help="Games each --searched-games-workers process plays at once.")
    parser.add_argument("--searched-value-coef", type=float, default=0.0,
                        help="Weight (times --vf-coef) of the value loss on searched games' decisions, a sample of --searched-value-batch from their replay store per minibatch.")
    parser.add_argument("--searched-value-batch", type=int, default=1024,
                        help="Searched-game rows sampled from the replay store per minibatch.")
    parser.add_argument("--predeal-value-coef", type=float, default=0.0,
                        help="P33, the pre-deal critic: weight (times --vf-coef) of a value loss on each env's turn-end park -- after a turn's last action, before the deal, nobody's decision -- read from both sides, each with its own next decision's lambda-return as the target (the result where the game ends first). The rollout root's stop=predeal reads the critic there. 0 = off.")
    parser.add_argument("--search-rollout-leaf-side", choices=["mover", "decider", "own"], default="mover",
                        help="Whose side the rollout root's leaves are read from: the root mover's at every leaf (the default, the measured root); each leaf's decider's -- the side the critic is trained on -- negated into the mover's frame where that is the opponent; or own: past the stop, play on to the mover's next decision and read it there (P33).")
    parser.add_argument("--search-rollout-k", type=int, default=4,
                        help="Rollout-root search targets: candidates (the teacher's most probable moves).")
    parser.add_argument("--search-rollout-worlds", type=int, default=16,
                        help='Rollout-root search targets: worlds per candidate.')
    parser.add_argument("--search-workers", type=int, default=0,
                        help="Rollout-root search targets: price them in this many worker processes, each with its own "
                             "copy of the frozen teacher, streamed the queued decisions during the rollout and collected "
                             "at its end -- the targets land on the same rows as without workers. 0: the trainer prices them.")
    parser.add_argument("--search-teacher-policy", action="store_true",
                        help="Search-CE targets are the frozen --search-teacher-checkpoint's own policy, with no search: the "
                             "control for the rollout-root teacher (same rows, same weight).")
    # P36's gated active search (ai/training/active_search.py; owner, 2026-10-20).
    parser.add_argument("--active-search-spec", type=str, default=None,
                        help="P36: the rollout root that acts in this run, as a spec tail without its checkpoint "
                             "(e.g. 4:16:4:z2:earlier:own:keyed:aligned:box4). Off by default. It joins only once its "
                             "gate opens: a check every --active-search-gate-every-steps plays the raw snapshot "
                             "under the root against itself raw, and the first at --active-search-gate-threshold "
                             "or more opens it. Then the root plays --active-search-frac of the learner's entry "
                             "decisions (headline and action-round card choices) as environment -- no policy "
                             "gradient on them -- and its improved policy is their search-CE target, searching a "
                             "lagged snapshot of the learner.")
    parser.add_argument("--active-search-frac", type=float, default=1.0 / 64.0,
                        help="P36: the share of the learner's entry decisions the root plays once its gate is open.")
    parser.add_argument("--active-search-ce-coef", type=float, default=0.1,
                        help="P36: weight of the cross-entropy toward the root's improved policy on the rows it played.")
    parser.add_argument("--active-search-gate-every-steps", type=int, default=40_000_000,
                        help="P36: steps between gate checks (run in a separate tools/tournament.py process).")
    parser.add_argument("--active-search-gate-games", type=int, default=200,
                        help="P36: games per gate check (half a side).")
    parser.add_argument("--active-search-gate-threshold", type=float, default=0.55,
                        help="P36: the root's score against its raw snapshot that opens the gate.")
    parser.add_argument("--active-search-gate-only", action="store_true",
                        help="P36: run the gate checks and record when the gate opened, but never join -- so a "
                             "later resume without this flag switches search on at once if the gate is open.")
    parser.add_argument("--active-search-refresh-steps", type=int, default=10_000_000,
                        help="P36: steps between refreshes of the searched snapshot from the learner, early.")
    parser.add_argument("--active-search-refresh-steps-late", type=int, default=50_000_000,
                        help="P36: the same from --active-search-late-from on.")
    parser.add_argument("--active-search-late-from", type=int, default=400_000_000,
                        help="P36: the step the refresh cadence slows at.")
    parser.add_argument("--search-teacher-checkpoint", type=str, default=None,
                        help="P32: search with this FROZEN network for the search CE targets, instead of the "
                             "network being trained -- whose targets follow it as it changes. Same "
                             "observation features and action view as the learner.")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="P32: a frozen checkpoint whose policy is distilled, with "
                             "--teacher-coef, on the learner's own decisions in --teacher-seat "
                             "(e.g. a seat-locked exploiter that plays that seat better). Same "
                             "observation features and action view as the learner.")
    parser.add_argument("--teacher-coef", type=float, default=0.0,
                        help="Weight of KL(teacher || policy) in the policy loss. 0 = off.")
    parser.add_argument("--teacher-seat", type=str, default="us", choices=["us", "ussr", "both"],
                        help="The seat whose decisions are distilled.")
    parser.add_argument("--rollout-temps", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                        metavar=("T1", "T2", "T3", "T4"),
                        help="The four per-environment rollout sampling temperatures. Default "
                             "1.0 1.0 1.0 1.0: sample at the policy, so PPO's temperature-1 "
                             "log-probabilities are the ones the actions came from (adopted "
                             "2026-09-27, research/log/E5_06_rollout_temperature.md). Runs before "
                             "then used 0.15 0.50 0.10 0.35, every band sharper than the policy; "
                             "pass those to reproduce one.")
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
    parser.add_argument("--league-dirs", nargs="+", default=None,
                        help="P24: other runs' directories whose snapshots join the opponent pool "
                             "as a separate 'league' group, re-scanned every --pool-every-steps. "
                             "With --opponent-self-pool the run's own history stays in the pool "
                             "beside them (the main agent); without it the pool is the league "
                             "alone (a main exploiter, with --opponent-frac 1.0). A file named "
                             "snapshot_<n>steps.pt, or <prefix>_snapshot_<n>steps.pt, is a member.")
    parser.add_argument("--league-pool-size", type=int, default=4,
                        help="Most league members kept; the oldest-added is evicted first.")
    parser.add_argument("--league-frac", type=float, default=None,
                        help="Probability that an iteration's pool opponent is a league member, "
                             "when the pool also holds the run's own history. Default: every "
                             "member drawn alike, so the league's share follows the group sizes.")
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
                        help="Run name, E<n>-A<n>-R<n>-S<n> plus its branch points (research/method/run_nomenclature.md), e.g. E7-A4-R1-S44@4390M+S45. Becomes the checkpoint directory prefix (<run-name>_[date]_[time]) and is recorded in metadata.json; resuming the run under its own name from its newest state continues in that directory (--new-directory to opt out). Omit any step budget: each snapshot's filename carries its own.")
    parser.add_argument("--new-directory", action="store_true",
                        help="Write an unchanged continuation (--resume of this --run-name's own directory from its newest state) into a new <run-name>_<timestamp> directory instead of continuing in the run's own directory, which is the default.")
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
            weight_decay=args.weight_decay,
            decay_norms_biases=args.decay_norms_biases,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            policy_head_lr_mult=args.policy_head_lr_mult,
            mup=args.mup,
            mup_base_proj_dim=args.mup_base_proj_dim,
            mup_base_hidden_dim=args.mup_base_hidden_dim,
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
            inject_setup_only=args.inject_setup_only,
            setup_block_lambda=args.setup_block_lambda,
            block_lambda=(args.block_lambda if args.block_lambda is not None
                          else ("setup" if args.setup_block_lambda else "off")),
            decisiveness_turns=args.decisiveness_turns,
            vp_potential=args.vp_potential,
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
            search_gumbel_k=args.search_gumbel_k,
            search_prior_temperature=args.search_prior_temperature,
            search_teacher_checkpoint=args.search_teacher_checkpoint,
            search_value_beta=args.search_value_beta,
            search_rollout_horizon=args.search_rollout_horizon,
            search_rollout_k=args.search_rollout_k,
            search_rollout_worlds=args.search_rollout_worlds,
            search_rollout_stop=args.search_rollout_stop,
            search_rollout_leaf_side=args.search_rollout_leaf_side,
            predeal_value_coef=args.predeal_value_coef,
            searched_games_workers=args.searched_games_workers,
            searched_games_envs=args.searched_games_envs,
            searched_value_coef=args.searched_value_coef,
            searched_value_batch=args.searched_value_batch,
            search_workers=args.search_workers,
            search_teacher_policy=args.search_teacher_policy,
            active_search_spec=args.active_search_spec,
            active_search_frac=args.active_search_frac,
            active_search_ce_coef=args.active_search_ce_coef,
            active_search_gate_every_steps=args.active_search_gate_every_steps,
            active_search_gate_games=args.active_search_gate_games,
            active_search_gate_threshold=args.active_search_gate_threshold,
            active_search_gate_only=args.active_search_gate_only,
            active_search_refresh_steps=args.active_search_refresh_steps,
            active_search_refresh_steps_late=args.active_search_refresh_steps_late,
            active_search_late_from=args.active_search_late_from,
            resume_weights=args.resume_weights,
            resume_widen=args.resume_widen,
            resume_grow_tokens=args.resume_grow_tokens,
            resume_reset_policy_head=args.resume_reset_policy_head,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_coef=args.teacher_coef,
            teacher_seat=args.teacher_seat,
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
            new_directory=args.new_directory,
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
            league_dirs=args.league_dirs,
            league_pool_size=args.league_pool_size,
            league_frac=args.league_frac,
            per_seat_adv_norm=args.per_seat_adv_norm,
            adv_norm_learner_only=args.adv_norm_learner_only,
            opponent_temperature=args.opponent_temperature,
            wolf_seat_weight=args.wolf_seat_weight,
            wolf_power=args.wolf_power,
            wolf_ema_games=args.wolf_ema_games,
            wolf_scope=args.wolf_scope,
            wolf_dead_zone=args.wolf_dead_zone,
            wolf_dead_zone_mode=args.wolf_dead_zone_mode,
            adv_norm_floor=args.adv_norm_floor,
            entropy_ceiling=args.entropy_ceiling,
            target_kl=args.target_kl,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            entropy_normalize=args.entropy_normalize,
            setup_entropy_floor=args.setup_entropy_floor,
            setup_entropy_lr=args.setup_entropy_lr,
            setup_entropy_max_coef=args.setup_entropy_max_coef,
            forced_opening=args.forced_opening,
            setup_mc_credit=args.setup_mc_credit,
            setup_script_frac=args.setup_script_frac,
            setup_script_openings=args.setup_script_openings,
            play_mode_temp=args.play_mode_temp,
            play_mode_floor=args.play_mode_floor,
            floor_scope=args.floor_scope,
            floor_from=args.floor_from,
            floor_anneal_from=args.floor_anneal_from,
            floor_anneal_steps=args.floor_anneal_steps,
            seed_scenarios=args.seed_scenarios,
            seed_frac=args.seed_frac,
            seed_scenarios_from=args.seed_scenarios_from,
            aux_opp_legality=args.aux_opp_legality,
            aux_opp_legality_frac=args.aux_opp_legality_frac,
            aux_opp_legality_min_batch=args.aux_opp_legality_min_batch,
            mode_cf_coef=args.mode_cf_coef,
            mode_cf_subsample=args.mode_cf_subsample,
            mode_cf_playouts=args.mode_cf_playouts,
            mode_cf_from=args.mode_cf_from,
            turn_credit_coef=args.turn_credit_coef,
            turn_credit_budget=args.turn_credit_budget,
            turn_credit_worlds=args.turn_credit_worlds,
            turn_credit_beta=args.turn_credit_beta,
            turn_credit_k=args.turn_credit_k,
            turn_credit_floor=args.turn_credit_floor,
            turn_credit_nested_p=args.turn_credit_nested_p,
            turn_credit_nested_share=args.turn_credit_nested_share,
            turn_credit_narrow_share=args.turn_credit_narrow_share,
            turn_credit_cap=args.turn_credit_cap,
            turn_credit_steps=args.turn_credit_steps,
            turn_credit_batch=args.turn_credit_batch,
            turn_credit_buffer=args.turn_credit_buffer,
            turn_credit_workers=args.turn_credit_workers,
            forced_openings_frac=args.forced_openings_frac,
            forced_openings_us=args.forced_openings_us,
            forced_openings_ussr=args.forced_openings_ussr,
            force_applicable_events=args.force_applicable_events,
            force_event_frac=args.force_event_frac,
            force_events_from=args.force_events_from,
            force_event_credit=args.force_event_credit,
            floor_rows_kind=args.floor_rows,
            setup_mc_coef=args.setup_mc_coef,
            setup_mc_min_batch=args.setup_mc_min_batch,
            aux_own_coef=args.aux_ownership,
            aux_vp_coef=args.aux_vp_margin,
            aux_sample_frac=args.aux_sample_frac,
            aux_min_batch=args.aux_min_batch,
            # Ladder only: the other architectures read the base layout, and record that they do.
            obs_features=obs_feature_bits(args.obs_features) if args.arch == "ladder" else 0,
            aux_card_coef=args.aux_card_events,
            aux_card_sample_frac=args.aux_card_sample_frac,
            aux_card_min_batch=args.aux_card_min_batch,
            aux_card_buffer=args.aux_card_buffer,
            aux_card_steps=args.aux_card_steps,
            aux_card_batch=args.aux_card_batch,
            lr_schedule=args.lr_schedule,
            lr_schedule_start=args.lr_schedule_start,
            lr_schedule_every=args.lr_schedule_every,
            lr_schedule_values=args.lr_schedule_values,
            lr_schedule_span=args.lr_schedule_span,
            lr_min=args.lr_min,
            lr_warmup_steps=args.lr_warmup_steps,
            ema_weights=args.ema_weights,
            compile_update=args.compile_update,
            z_loss_coef=args.z_loss_coef,
            logit_gap_coef=args.logit_gap_coef,
            logit_gap=args.logit_gap,
            phase_lock=args.phase_lock,
            token_chunk=args.ladder_token_chunk,
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
    elif args.mode == "policy-fit":
        if not args.policy_dataset or not args.warmup_checkpoint:
            raise SystemExit("--mode policy-fit needs --policy-dataset and --warmup-checkpoint (the weights to "
                             "fine-tune, or with --policy-reinit the shape to train from scratch)")
        run_policy_fit(
            PolicyFitConfig(dataset=args.policy_dataset, worlds=args.policy_worlds, init=args.warmup_checkpoint,
                            reinit=args.policy_reinit, arch=parse_arch(args.policy_arch), batch=args.policy_batch,
                            lr=args.policy_lr, max_epochs=args.policy_epochs, eval_every=args.policy_eval_every,
                            patience=args.policy_patience, train_fraction=args.policy_train_fraction,
                            head_only=args.policy_head_only, kl_anchor=args.policy_kl_anchor,
                            target=args.policy_target, reinit_heads=args.policy_reinit_heads,
                            seed=args.seed if args.seed is not None else 44,
                            device=args.device),
            run_name=args.run_name, description=args.description, output_dir=args.output_dir,
            cli={k: v for k, v in vars(args).items()
                 if k.startswith("policy_") or k in ("mode", "warmup_checkpoint", "seed")})
    elif args.mode == "q-fit":
        if not args.q_dataset or not args.q_target or not args.warmup_checkpoint:
            raise SystemExit("--mode q-fit needs --q-dataset, --q-target and --warmup-checkpoint (the network copied)")
        run_q_fit(
            QFitConfig(dataset=args.q_dataset, target=args.q_target, init=args.warmup_checkpoint, head=args.q_head,
                       train_trunk=args.q_train_trunk, weighting=args.q_weighting, se_floor=args.q_se_floor,
                       baseline=args.q_baseline or None, judge=args.q_judge, batch=args.q_batch, lr=args.q_lr, max_epochs=args.q_epochs,
                       eval_every=args.q_eval_every, patience=args.q_patience, train_fraction=args.q_train_fraction,
                       seed=args.seed if args.seed is not None else 44, device=args.device),
            run_name=args.run_name, description=args.description, output_dir=args.output_dir,
            cli={k: v for k, v in vars(args).items()
                 if k.startswith("q_") or k in ("mode", "warmup_checkpoint", "seed")})
    elif args.mode == "value-fit":
        if not args.value_dataset or not args.warmup_checkpoint:
            raise SystemExit("--mode value-fit needs --value-dataset and --warmup-checkpoint (the weights to "
                             "fine-tune, or with --value-reinit the shape to train from scratch)")
        run_value_fit(
            ValueFitConfig(dataset=args.value_dataset, init=args.warmup_checkpoint, reinit=args.value_reinit,
                           arch=parse_arch(args.value_arch),
                           batch=args.value_batch, sibling_groups=args.value_sibling_groups,
                           sibling_coef=args.value_sibling_coef, vp_coef=args.value_vp_coef,
                           aux_coef=args.value_aux_coef, lr=args.value_lr, max_epochs=args.value_epochs,
                           eval_every=args.value_eval_every, patience=args.value_patience,
                           siblings=args.value_siblings, sibling_loss=args.value_sibling_loss,
                           sibling_decisions=args.value_sibling_decisions, leaf_coef=args.value_leaf_coef,
                           leaf_weights=args.value_leaf_weights,
                           seed=args.seed if args.seed is not None else 44, device=args.device),
            run_name=args.run_name, description=args.description, output_dir=args.output_dir,
            cli={k: v for k, v in vars(args).items()
                 if k.startswith("value_") or k in ("mode", "warmup_checkpoint", "seed")})
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
            print(f"  Ending Causes: {res['causes_all']}")


if __name__ == "__main__":
    main()
