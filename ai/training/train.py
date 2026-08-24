"""Unified CLI Training and Evaluation Runner for Twilight Struggle AI (ColdWarNet / NashPG)."""

import argparse
import os
import sys
import time
import torch

from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.training.behavioral_cloning import BehavioralCloningTrainer
from ai.training.nash_pg import NashPGTrainer
from ai.eval.arena import ArenaEvaluator


def main():
    parser = argparse.ArgumentParser(description="Twilight Struggle ColdWarNet / NashPG Trainer")
    parser.add_argument(
        "--mode",
        type=str,
        default="nashpg",
        choices=["bc", "nashpg", "eval"],
        help="Training mode: 'bc' (Behavioral Cloning), 'nashpg' (Nash Policy Gradient), 'eval' (Tournament Arena)",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel C++ game environments")
    parser.add_argument("--buffer-size", type=int, default=128, help="Rollout buffer size per environment")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations (for nashpg)")
    parser.add_argument("--bc-games", type=int, default=500, help="Number of demonstration games for Behavioral Cloning")
    parser.add_argument("--bc-epochs", type=int, default=10, help="Epochs for Behavioral Cloning")
    parser.add_argument("--eval-games", type=int, default=40, help="Number of tournament evaluation games")
    parser.add_argument("--eval-opponent", type=str, default="heuristic", choices=["heuristic", "random"], help="Evaluation opponent")
    parser.add_argument("--eta", type=float, default=0.1, help="NashPG KL regularization coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ref-update-freq", type=int, default=250_000, help="Outer-loop reference update frequency in steps")
    parser.add_argument("--save-path", type=str, default="checkpoints/coldwar_net.pt", help="Path to save model checkpoint")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[ColdWarNet Trainer] Active Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = create_coldwar_net(device)

    if args.load_path and os.path.exists(args.load_path):
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print(f"[ColdWarNet Trainer] Loaded weights from {args.load_path}")

    if args.mode == "bc":
        # Phase 0: Behavioral Cloning Pre-Training
        bc_trainer = BehavioralCloningTrainer(model, device=device, lr=args.lr)
        dataset = bc_trainer.generate_demonstration_dataset(num_games=args.bc_games)
        bc_trainer.train(dataset, epochs=args.bc_epochs, batch_size=256, save_path=args.save_path)

        # Quick evaluation
        evaluator = ArenaEvaluator(model, device=device)
        evaluator.run_tournament(opponent_type="heuristic", num_games=20)

    elif args.mode == "nashpg":
        # Phase 1: NashPG Self-Play Reinforcement Learning
        nash_trainer = NashPGTrainer(
            active_net=model,
            num_envs=args.num_envs,
            buffer_size=args.buffer_size,
            lr=args.lr,
            eta=args.eta,
            ref_update_freq=args.ref_update_freq,
            device=device,
        )
        nash_trainer.train_iterations(num_iterations=args.iterations, log_interval=5, save_path=args.save_path)

        # Post-training evaluation
        evaluator = ArenaEvaluator(model, device=device)
        evaluator.run_tournament(opponent_type="heuristic", num_games=args.eval_games)

    elif args.mode == "eval":
        # Tournament Evaluation
        evaluator = ArenaEvaluator(model, device=device)
        evaluator.run_tournament(opponent_type=args.eval_opponent, num_games=args.eval_games)


if __name__ == "__main__":
    main()
