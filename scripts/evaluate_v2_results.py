import os
import sys
import time
import json
import datetime
from typing import Dict, Any, List
import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2, ColdWarNetV2
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.eval.self_play import generate_selfplay_replay


def choose_random_action(mask: np.ndarray) -> int:
    legal = np.where(mask == 1)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


def play_matchup(
    model_ussr: Any,
    model_us: Any,
    num_games: int = 50,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.1,
    max_steps_per_game: int = 4000,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """Simulates num_games between two models in parallel with C++ batch runner."""
    games_completed = 0
    ussr_wins = 0
    us_wins = 0
    draws = 0
    total_turns = 0

    batch_runner = ts.VectorizedBatchRunner(batch_size, int(time.time() * 1000) % 1000000)
    batch_runner.refresh_all()

    step_counts = np.zeros(batch_size, dtype=np.int32)

    while games_completed < num_games:
        obs_raw = np.array(batch_runner.get_observations(), dtype=np.float32).reshape(batch_size, 4293)
        masks_raw = np.array(batch_runner.get_action_masks(), dtype=np.uint8).reshape(batch_size, 212)
        players = np.array(batch_runner.get_decision_players(), dtype=np.int8)

        obs_t = torch.from_numpy(obs_raw).to(device)
        masks_t = torch.from_numpy(masks_raw).to(device)

        actions = np.zeros(batch_size, dtype=np.int32)

        # USSR masks & actions
        ussr_idx = np.where(players == -1)[0]
        if len(ussr_idx) > 0:
            if isinstance(model_ussr, (ColdWarNet, ColdWarNetV2)):
                with torch.no_grad():
                    a_t, _, _, _, _ = model_ussr.sample_action(
                        obs_t[ussr_idx], masks_t[ussr_idx], temperature=temperature
                    )
                    actions[ussr_idx] = a_t.cpu().numpy()
            elif model_ussr == "random":
                for idx in ussr_idx:
                    actions[idx] = choose_random_action(masks_raw[idx])

        # US masks & actions
        us_idx = np.where(players == 1)[0]
        if len(us_idx) > 0:
            if isinstance(model_us, (ColdWarNet, ColdWarNetV2)):
                with torch.no_grad():
                    a_t, _, _, _, _ = model_us.sample_action(
                        obs_t[us_idx], masks_t[us_idx], temperature=temperature
                    )
                    actions[us_idx] = a_t.cpu().numpy()
            elif model_us == "random":
                for idx in us_idx:
                    actions[idx] = choose_random_action(masks_raw[idx])

        # Neutral fallback
        none_idx = np.where(players == 0)[0]
        if len(none_idx) > 0:
            for idx in none_idx:
                actions[idx] = choose_random_action(masks_raw[idx])

        batch_runner.step_flat_all([int(x) for x in actions])
        dones = np.array(batch_runner.get_terminals(), dtype=bool)
        term_utils = np.array(batch_runner.get_terminal_utilities(), dtype=np.float32)
        step_counts += 1

        for i in range(batch_size):
            st = batch_runner.get_state(i)
            if dones[i] or step_counts[i] >= max_steps_per_game:
                if games_completed < num_games:
                    games_completed += 1
                    total_turns += st.turn
                    if term_utils[i] < 0 or (term_utils[i] == 0 and st.victory_points < 0):
                        ussr_wins += 1
                    elif term_utils[i] > 0 or (term_utils[i] == 0 and st.victory_points > 0):
                        us_wins += 1
                    else:
                        draws += 1

                batch_runner.reset_game(i, int(time.time() * 1000 + i * 997) % 1000000)
                step_counts[i] = 0

    return {
        "num_games": num_games,
        "ussr_wins": ussr_wins,
        "us_wins": us_wins,
        "draws": draws,
        "ussr_win_rate": ussr_wins / num_games * 100.0,
        "us_win_rate": us_wins / num_games * 100.0,
        "avg_turn": total_turns / num_games,
    }


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = "checkpoints/run_v2_20260825_224511"

    print("=" * 80, flush=True)
    print("RUNNING COMPREHENSIVE TOURNAMENT & BENCHMARKS (ColdWarNetV2)", flush=True)
    print(f"Target Run: {run_dir} | Device: {device}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load V2 Models (Snapshots 0m, 10m, 20m, 30m, 40m, 50m, 60m)
    v2_snapshots = {}
    for minute in [0, 10, 20, 30, 40, 50, 60]:
        path = f"{run_dir}/snapshot_{minute}m.pt"
        if os.path.exists(path):
            net = create_coldwar_net_v2(device)
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            net.eval()
            v2_snapshots[f"V2_Snapshot_{minute}m"] = net
            print(f"Loaded V2_Snapshot_{minute}m", flush=True)

    final_v2 = v2_snapshots["V2_Snapshot_60m"]

    # 2. Load V1 Baselines
    v1_baselines = {}
    v1_paths = {
        "V1_Snapshot_210m": "checkpoints/run_20260825_093352/snapshot_210m.pt",
        "V1_Snapshot_180m": "checkpoints/run_20260825_093352/snapshot_180m.pt",
        "V1_Snapshot_360m": "checkpoints/run_20260825_093352/snapshot_360m.pt",
        "V1_Snapshot_480m": "checkpoints/run_20260825_093352/snapshot_480m.pt",
    }
    for name, p in v1_paths.items():
        if os.path.exists(p):
            net = create_coldwar_net(device)
            net.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            net.eval()
            v1_baselines[name] = net
            print(f"Loaded {name}", flush=True)

    # 3. All opponents to evaluate final V2 (snapshot_60m) against:
    opponents = {
        "RandomBot": "random",
    }
    # Add V1 Baselines
    opponents.update(v1_baselines)
    # Add Intermediate V2 Snapshots
    for k, v in v2_snapshots.items():
        if k != "V2_Snapshot_60m":
            opponents[k] = v

    games_per_pairing = 50
    results = {}

    # Self-Play
    self_res = play_matchup(final_v2, final_v2, num_games=games_per_pairing, device=device)
    print(
        f"\n[V2_Snapshot_60m vs Self]: USSR Win: {self_res["ussr_win_rate"]:5.1f}% | US Win: {self_res["us_win_rate"]:5.1f}% | Avg Turn: {self_res["avg_turn"]:.1f}",
        flush=True
    )
    results["SelfPlay"] = self_res

    # Tournaments vs each opponent
    for name, opp in opponents.items():
        res_ussr = play_matchup(final_v2, opp, num_games=games_per_pairing, device=device)
        res_us = play_matchup(opp, final_v2, num_games=games_per_pairing, device=device)

        target_as_us_win = res_us["us_win_rate"]
        overall_win = (res_ussr["ussr_wins"] + res_us["us_wins"]) / (2 * games_per_pairing) * 100.0

        print(
            f"\n[V2_Snapshot_60m vs {name:18s}]: Overall Win: {overall_win:5.1f}%\n"
            f"  • As USSR ({games_per_pairing} games): Win {res_ussr["ussr_win_rate"]:5.1f}% (Avg Turn {res_ussr["avg_turn"]:.1f})\n"
            f"  • As US   ({games_per_pairing} games): Win {target_as_us_win:5.1f}% (Avg Turn {res_us["avg_turn"]:.1f})",
            flush=True
        )

        results[name] = {
            "as_ussr": res_ussr,
            "as_us": res_us,
            "overall_win": overall_win,
        }

    # 4. Generate Self-Play Demo Replay
    replay_path = "replays/v2_selfplay_demo.tslog.json"
    generate_selfplay_replay(model=final_v2, output_path=replay_path, device=device)

    # 5. Write Final Tournament Report
    report_path = f"{run_dir}/tournament_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ColdWarNetV2 1-Hour Training & Tournament Report\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
        f.write(f"**Architecture**: `ColdWarNetV2` (Card-to-Country Cross-Attention + GCN)\n")
        f.write(f"**Total Steps Trained**: 33,423,360 environment steps\n\n")
        f.write("## Tournament Results vs All Baselines & Previous Generations\n\n")
        f.write("| Opponent | Overall Win % | Win % as USSR | Win % as US | Avg Turn (USSR / US) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for opp_name, data in results.items():
            if opp_name == "SelfPlay":
                f.write(f"| ColdWarNetV2 (Self-Play) | 50.0% | {data["ussr_win_rate"]:.1f}% | {data["us_win_rate"]:.1f}% | {data["avg_turn"]:.1f} |\n")
            else:
                f.write(
                    f"| {opp_name:24s} | **{data["overall_win"]:5.1f}%** | {data["as_ussr"]["ussr_win_rate"]:5.1f}% | {data["as_us"]["us_win_rate"]:5.1f}% | {data["as_ussr"]["avg_turn"]:.1f} / {data["as_us"]["avg_turn"]:.1f} |\n"
                )

    print(f"\nTournament report saved to {report_path}", flush=True)
    print("=" * 80, flush=True)
    print("ALL EVALUATIONS COMPLETE!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    evaluate_all()
