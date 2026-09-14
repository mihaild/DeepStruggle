import os
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = os.path.abspath('.triton_cache')
# Unified Generic CLI Training & Evaluation Runner for Twilight Struggle AI.

import argparse
import os
import sys
import torch

from ai.training.generic_trainer import train_pipeline, run_behavioral_cloning_warmup
from tools.lib.player_agent import load_agent
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.tournament_evaluator import TournamentEvaluator
from ai.models.coldwar_net import create_coldwar_net
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


def main():
    parser = argparse.ArgumentParser(description="Generic Twilight Struggle Neural AI Training Pipeline")
    parser.add_argument("--arch", type=str, default="v2", choices=["v1", "v2", "mlp"],
                        help="Model architecture. v2 is the baseline; v1 is the original\n"
                             "network, kept because checkpoints that predate v2 still name it.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "warmup", "eval", "curriculum"], help="Execution mode")

    # Warm-up / Checkpoint options
    parser.add_argument("--warmup-checkpoint", "--load-path", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--warmup-dataset", type=str, default=None, help="Path to dataset file for supervised BC warmup")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Number of epochs for BC warmup")

    # Time & Snapshot parameters
    parser.add_argument("--duration-seconds", "--seconds-to-train", type=int, default=3600,
                        help="RL training duration in seconds. Counts training only -- snapshot "
                             "evaluation and start-pool refreshes are excluded, so the budget is "
                             "not eaten by evaluation cost that varies with the policy.")
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
    parser.add_argument("--snapshot-every-steps", type=int, default=0,
                        help="Take a snapshot every N env steps (0 = derive the "
                             "interval from --duration-seconds and "
                             "--snapshot-interval-seconds, which is indirect when "
                             "the budget is already in steps). Snapshots are then "
                             "named by step count rather than by elapsed seconds.")
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
                        help="Budget the run by env steps instead of by time (0 = use --duration-seconds). "
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
    parser.add_argument("--snapshot-interval-seconds", "--snapshot-every", type=int, default=600, help="Snapshot and tournament evaluation interval in seconds")

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
    parser.add_argument("--curriculum-switch-seconds", type=int, default=None, help="Elapsed training seconds at which curriculum switches to BlunderAware reward (default: 50%% of duration)")
    parser.add_argument("--curriculum-switch-fraction", type=float, default=0.5, help="Fraction of training duration at which curriculum switches to BlunderAware reward (default: 0.5)")
    parser.add_argument("--slice-turn-boundaries", type=str, default="auto", choices=["auto", "on", "off"],
                        help="Truncate GAE bootstrapping at game-turn boundaries for EVERY episode, clean wins included. Measurably degrades the policy; retained for ablations. 'auto' (default) leaves it off and relies on per-episode blunder windowing instead.")
    parser.add_argument("--no-blunder-window", action="store_true",
                        help="Disable per-episode blunder windowing. By default an unprovoked blunder loss (held scoring card, or self-inflicted DEFCON 1) only penalises the blunderer within that turn and shields the opponent from the windfall.")
    parser.add_argument("--ref-update-freq", type=int, default=200_000,
                        help="Env steps between NashPG reference-policy refreshes. At 512 envs x 128 buffer one iteration is 65,536 steps, so the default refreshes pi_ref every 4 iterations; raise it for a genuinely frozen anchor.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor. Keep at 1.0: the game is zero-sum and decided at the end, so any discount biases against the endgame (0.999 attenuates a terminal reward by ~26%% over a full game).")
    parser.add_argument("--defcon-coef", type=float, default=0.0,
                        help="Weight of the auxiliary DEFCON-risk head, which predicts whether the player to move is about to lose the game to its own DEFCON-1 choice (0 = head disabled). Added because val_win_head largely restates the VP margin -- corr(v_win, v_vp) = 0.86 -- so it reads self-inflicted DEFCON-1 deaths as roughly even positions while pricing ordinary losing positions correctly. Try 0.1.")
    parser.add_argument("--opponent-checkpoints", nargs="+", default=None,
                        help="Frozen snapshots to play a share of environments against, instead of pure self-play")
    parser.add_argument("--opponent-frac", type=float, default=0.0,
                        help="Fraction of environments facing a frozen opponent (0 disables)")
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

    args = parser.parse_args()

    if args.mode in ["train", "curriculum"]:
        eff_reward_scheme = "curriculum" if args.mode == "curriculum" else args.reward_scheme
        train_pipeline(
            arch=args.arch,
            warmup_checkpoint=args.warmup_checkpoint,
            warmup_dataset=args.warmup_dataset,
            bc_epochs=args.bc_epochs,
            duration_seconds=args.duration_seconds,
            snapshot_interval_seconds=args.snapshot_interval_seconds,
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
            defcon_coef=args.defcon_coef,
            train_steps=args.train_steps,
            seed=args.seed,
            resume=args.resume,
            resume_every_snapshot=args.resume_every_snapshot,
            snapshot_every_steps=args.snapshot_every_steps,
            inject_dataset=args.inject_dataset,
            inject_every=args.inject_every,
            inject_weight=args.inject_weight,
            decisiveness_turns=args.decisiveness_turns,
            max_snapshot_opponents=args.eval_max_snapshot_opponents,
            opponent_checkpoints=args.opponent_checkpoints,
            opponent_frac=args.opponent_frac,
            opponent_lock_side=args.opponent_lock_side,
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
            curriculum_switch_seconds=args.curriculum_switch_seconds,
            curriculum_switch_fraction=args.curriculum_switch_fraction,
            slice_turn_boundaries=(None if args.slice_turn_boundaries == "auto" else args.slice_turn_boundaries == "on"),
            ref_update_freq=args.ref_update_freq,
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
