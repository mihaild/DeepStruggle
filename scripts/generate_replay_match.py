"""Generates a full self-play game replay (.tslog.json) using trained ColdWarNet checkpoints."""

import argparse
import json
import os
import sys
import time
from typing import Optional
import numpy as np
import torch

# Ensure repository root is on sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder
from server.replay import ReplayLogger, REPLAYS_DIR


def generate_self_play_replay(
    model_path: str = "checkpoints/snapshot_20m.pt",
    seed: int = 42,
    temperature: float = 0.3,
    game_id: str = "snapshot_20m_self_play",
    output_filename: Optional[str] = None,
    device_str: str = "cuda",
) -> str:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"\n{'='*70}")
    print(f" Generating Self-Play Game: {game_id}")
    print(f" Model: {model_path} | Device: {device} | Seed: {seed} | Temp: {temperature}")
    print(f"{'='*70}\n")

    model = create_coldwar_net(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state = ts.GameState()
    ts.Engine.init_game(state, seed)

    replay_logger = ReplayLogger(
        game_id=game_id,
        seed=seed,
        us_player=f"ColdWarNet (20m) [US]",
        ussr_player=f"ColdWarNet (20m) [USSR]",
    )

    # Initial state snapshot
    initial_dict = ts.state_to_dict(state)
    step_index = 0
    max_steps = 5000  # Generous safety limit, plays naturally to full game completion

    print(f"{'Step':>4s} | {'Turn':>4s} | {'AR':>2s} | {'Player':>5s} | {'DEFCON':>6s} | {'VP':>4s} | {'Action Description'}")
    print(f"{'-'*75}")

    while not ts.Engine.is_terminal(state) and step_index < max_steps:
        step_index += 1
        p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
        player_name = "US" if p == ts.Player.US else ("USSR" if p == ts.Player.USSR else "NONE")

        obs = ts.extract_observation(state, p)
        mask = ActionEncoder.get_legal_mask(state)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            act_t, _, _, _, _ = model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=False)

        action_idx = int(act_t.item())
        action_desc = ActionEncoder.get_action_name(state, action_idx)

        # MicroAction info
        ma = ts.ActionMask.decode_flat_action(state, action_idx)
        action_dict = {
            "flat_action_idx": action_idx,
            "decision_type": int(ma.decision_type),
            "primary_id": int(ma.primary_id),
            "secondary_id": int(ma.secondary_id),
            "flags": int(ma.flags),
        }

        turn_before = state.turn
        ar_before = state.action_round
        phase_before = str(state.current_phase).replace("Phase.", "")

        # Execute in C++ engine
        ts.Engine.step_flat(state, action_idx)

        state_after_dict = ts.state_to_dict(state)

        # Log step to replay
        replay_logger.log_step(
            step_index=step_index,
            turn=turn_before,
            ar=ar_before,
            phase=phase_before,
            player=player_name,
            action=action_dict,
            description=action_desc,
            state_snapshot=state_after_dict,
        )

        if step_index % 10 == 0 or state.current_phase == ts.Phase.GAME_OVER or "Scoring" in action_desc:
            print(f"{step_index:4d} | {turn_before:4d} | {ar_before:2d} | {player_name:>5s} | {state.defcon:6d} | {state.victory_points:+4d} | {action_desc}")

    # Determine game conclusion
    term_util = ts.Engine.get_terminal_utility(state)
    if term_util > 0:
        winner = "US"
        margin = state.victory_points
        reason = "Victory Point Threshold (+20 VP) or DEFCON Inversion" if state.defcon > 1 else "USSR triggered DEFCON 1"
    elif term_util < 0:
        winner = "USSR"
        margin = state.victory_points
        reason = "Victory Point Threshold (-20 VP) or DEFCON Inversion" if state.defcon > 1 else "US triggered DEFCON 1"
    else:
        winner = "DRAW"
        margin = 0
        reason = "Final Scoring Draw or Maximum Steps Limit"

    replay_logger.set_result(
        winner=winner,
        margin=margin,
        end_turn=state.turn,
        reason=reason,
    )

    if not output_filename:
        output_filename = os.path.join(REPLAYS_DIR, f"{game_id}.tslog.json")

    saved_path = replay_logger.save(output_filename)

    print(f"{'-'*75}")
    print(f" Game Ended on Turn {state.turn} (Step {step_index})")
    print(f" Winner: {winner} (VP: {state.victory_points:+d}, DEFCON: {state.defcon})")
    print(f" Reason: {reason}")
    print(f" Replay saved to: {saved_path}\n")

    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Self-Play Game Replay")
    parser.add_argument("--model", type=str, default="checkpoints/snapshot_20m.pt", help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for game initialization")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--game-id", type=str, default="snapshot_20m_self_play", help="Replay Game ID")
    parser.add_argument("--output", type=str, default=None, help="Custom output path")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")

    args = parser.parse_args()
    generate_self_play_replay(
        model_path=args.model,
        seed=args.seed,
        temperature=args.temp,
        game_id=args.game_id,
        output_filename=args.output,
        device_str=args.device,
    )
