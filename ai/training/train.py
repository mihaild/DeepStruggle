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
from tools.lib.tournament_evaluator import TournamentEvaluator
from ai.models.coldwar_net import create_coldwar_net
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.models.coldwar_net_v3 import create_coldwar_net_v3


def main():
    parser = argparse.ArgumentParser(description="Generic Twilight Struggle Neural AI Training Pipeline")
    parser.add_argument("--arch", type=str, default="v4", choices=["v1", "v2", "v3", "v4"], help="Model architecture: v1, v2, v3, or v4 (ColdWarNetV4 Deep Card-Transformer + Belief Head)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "warmup", "eval", "curriculum"], help="Execution mode")

    # Warm-up / Checkpoint options
    parser.add_argument("--warmup-checkpoint", "--load-path", type=str, default=None, help="Path to pre-trained checkpoint")
    parser.add_argument("--warmup-dataset", type=str, default=None, help="Path to dataset file for supervised BC warmup")
    parser.add_argument("--bc-epochs", type=int, default=5, help="Number of epochs for BC warmup")

    # Time & Snapshot parameters
    parser.add_argument("--duration-seconds", "--seconds-to-train", type=int, default=3600, help="Total RL training duration in seconds")
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
    parser.add_argument("--priority-alpha", type=float, default=0.0,
                        help="Sample minibatches weighted by |advantage|^alpha so rare decisive transitions are not drowned by routine ones (0 = uniform). Deliberately biases the gradient toward high-swing states; try 0.5.")
    parser.add_argument("--output-dir", "--save-path", type=str, default=None, help="Output directory for checkpoints (default: data/checkpoints/run_[version]_[start date]_[start time])")
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
            entropy_coef=args.entropy_coef,
            reward_scheme=eff_reward_scheme,
            output_dir=args.output_dir,
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
        if args.arch == "v3":
            model = create_coldwar_net_v3(dev)
        elif args.arch == "v2":
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
            res = TournamentEvaluator.play_matchup(agent_main, opp, games_per_side=args.eval_games_per_side)
            print(f"\nMatchup: {agent_main.name} vs {opp.name}")
            print(f"  Win Rate: {res['win_rate_a']*100:.1f}% ({res['a_wins']}W - {res['b_wins']}L - {res['draws']}D)")
            print(f"  Avg Turn: {res['avg_turn']:.1f} | Avg Steps: {res['avg_steps']:.0f} | Avg VP Margin: {res['avg_vp_margin_a']:+.1f}")
            print(f"  Ending Causes: {res['causes']}")


if __name__ == "__main__":
    main()
