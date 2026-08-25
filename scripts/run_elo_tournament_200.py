import os
import sys
import time
import json
import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2, ColdWarNetV2
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net


def choose_random_action(mask: np.ndarray) -> int:
    legal = np.where(mask == 1)[0]
    return int(np.random.choice(legal)) if len(legal) > 0 else 0


def choose_heuristic_action(state: ts.GameState, mask: np.ndarray, role: str) -> int:
    legal_indices = np.where(mask == 1)[0]
    if len(legal_indices) == 0:
        return 0
    best_act = legal_indices[0]
    best_score = -1e9
    is_ussr = (role == "USSR")

    for idx in legal_indices:
        ma = ts.decode_flat_action(state, int(idx))
        score = 0.0
        # 1. SETUP: Prefer key battlegrounds
        if state.current_phase == ts.Phase.SETUP:
            if is_ussr and ma.primary_id in [14, 15]:  # East Germany, Poland
                score += 100.0
            elif not is_ussr and ma.primary_id in [6, 10]:  # West Germany, Italy
                score += 100.0

        # 2. SELECT_CARD: Scoring cards or high Ops
        elif ma.decision_type == ts.DecisionType.SELECT_CARD:
            cid = ma.primary_id
            if 1 <= cid <= 110:
                cinfo = ts.CardData.get_card_info(cid)
                if cinfo.get("is_scoring"):
                    score += 50.0
                score += cinfo.get("ops", 0) * 10.0

        # 3. POINT_NODE: Prefer battlegrounds & Europe
        elif ma.decision_type == ts.DecisionType.POINT_NODE:
            nid = ma.primary_id
            if 0 <= nid < 84:
                cinfo = ts.MapData.get_country_info(nid)
                if cinfo.get("battleground"):
                    score += 30.0
                if cinfo.get("region") == ts.Region.EUROPE:
                    score += 20.0
                elif cinfo.get("region") == ts.Region.ASIA:
                    score += 10.0

        # 4. SELECT_OP_MODE: Prefer Influence
        elif ma.decision_type == ts.DecisionType.SELECT_OP_MODE:
            if ma.primary_id == 0:  # INFLUENCE
                score += 10.0

        if score > best_score:
            best_score = score
            best_act = idx

    return int(best_act)


def play_directed_matchup(
    model_ussr: Any,
    model_us: Any,
    num_games: int = 100,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.1,
    batch_size: int = 100,
    max_steps: int = 4000,
) -> Dict[str, Any]:
    """Runs num_games where model_ussr is USSR and model_us is US."""
    games_completed = 0
    ussr_wins = 0
    us_wins = 0
    draws = 0
    total_turns = 0
    defcon_losses_ussr = 0
    defcon_losses_us = 0
    vp_wins_ussr = 0
    vp_wins_us = 0

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

        # USSR Decisions
        ussr_idx = np.where(players == -1)[0]
        if len(ussr_idx) > 0:
            if isinstance(model_ussr, (ColdWarNet, ColdWarNetV2)):
                with torch.no_grad():
                    a_t, _, _, _, _ = model_ussr.sample_action(
                        obs_t[ussr_idx], masks_t[ussr_idx], temperature=temperature
                    )
                    actions[ussr_idx] = a_t.cpu().numpy()
            elif model_ussr == "heuristic":
                for idx in ussr_idx:
                    st = batch_runner.get_state(idx)
                    actions[idx] = choose_heuristic_action(st, masks_raw[idx], "USSR")
            elif model_ussr == "random":
                for idx in ussr_idx:
                    actions[idx] = choose_random_action(masks_raw[idx])

        # US Decisions
        us_idx = np.where(players == 1)[0]
        if len(us_idx) > 0:
            if isinstance(model_us, (ColdWarNet, ColdWarNetV2)):
                with torch.no_grad():
                    a_t, _, _, _, _ = model_us.sample_action(
                        obs_t[us_idx], masks_t[us_idx], temperature=temperature
                    )
                    actions[us_idx] = a_t.cpu().numpy()
            elif model_us == "heuristic":
                for idx in us_idx:
                    st = batch_runner.get_state(idx)
                    actions[idx] = choose_heuristic_action(st, masks_raw[idx], "US")
            elif model_us == "random":
                for idx in us_idx:
                    actions[idx] = choose_random_action(masks_raw[idx])

        # Neutral / Non-phasing fallback
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
            if dones[i] or step_counts[i] >= max_steps:
                if games_completed < num_games:
                    games_completed += 1
                    total_turns += st.turn
                    if term_utils[i] < 0 or (term_utils[i] == 0 and st.victory_points < 0):
                        ussr_wins += 1
                        if st.defcon <= 1:
                            defcon_losses_us += 1
                        else:
                            vp_wins_ussr += 1
                    elif term_utils[i] > 0 or (term_utils[i] == 0 and st.victory_points > 0):
                        us_wins += 1
                        if st.defcon <= 1:
                            defcon_losses_ussr += 1
                        else:
                            vp_wins_us += 1
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
        "defcon_losses_ussr": defcon_losses_ussr,
        "defcon_losses_us": defcon_losses_us,
        "vp_wins_ussr": vp_wins_ussr,
        "vp_wins_us": vp_wins_us,
    }


def compute_bradley_terry_elo(win_matrix: np.ndarray, model_names: List[str], anchor_name: str = "RandomBot", anchor_elo: float = 1000.0) -> Dict[str, float]:
    n = len(model_names)
    gamma = np.ones(n, dtype=np.float64)
    W = np.sum(win_matrix, axis=1)
    N = win_matrix + win_matrix.T

    for _ in range(2000):
        gamma_prev = gamma.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and N[i, j] > 0:
                    denom += N[i, j] / (gamma[i] + gamma[j])
            if denom > 0:
                gamma[i] = W[i] / denom
        gamma = gamma / np.mean(gamma)
        if np.max(np.abs(gamma - gamma_prev)) < 1e-7:
            break

    raw_elo = 400.0 * np.log10(gamma + 1e-9)
    if anchor_name in model_names:
        anchor_idx = model_names.index(anchor_name)
        shift = anchor_elo - raw_elo[anchor_idx]
    else:
        shift = 1500.0 - np.mean(raw_elo)

    final_elo = raw_elo + shift
    return {model_names[i]: round(float(final_elo[i]), 1) for i in range(n)}


def run_full_tournament():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90, flush=True)
    print("STARTING 200-GAME PAIRWISE TOURNAMENT WITH ELO & WIN/LOSS MATRIX", flush=True)
    print(f"Device: {device} | Games per Pairing: 200 (100 as USSR, 100 as US)", flush=True)
    print("=" * 90, flush=True)

    models = {}

    # 1. ColdWarNetV2 Checkpoints
    v2_60m = create_coldwar_net_v2(device)
    v2_60m.load_state_dict(torch.load("checkpoints/run_v2_20260825_224511/snapshot_60m.pt", map_location=device, weights_only=True))
    v2_60m.eval()
    models["ColdWarNetV2_60m"] = v2_60m

    v2_50m = create_coldwar_net_v2(device)
    v2_50m.load_state_dict(torch.load("checkpoints/run_v2_20260825_224511/snapshot_50m.pt", map_location=device, weights_only=True))
    v2_50m.eval()
    models["ColdWarNetV2_50m"] = v2_50m

    v2_0m = create_coldwar_net_v2(device)
    v2_0m.load_state_dict(torch.load("checkpoints/run_v2_20260825_224511/snapshot_0m.pt", map_location=device, weights_only=True))
    v2_0m.eval()
    models["ColdWarNetV2_Warmup_0m"] = v2_0m

    # 2. ColdWarNetV1 Checkpoints
    v1_480m = create_coldwar_net(device)
    v1_480m.load_state_dict(torch.load("checkpoints/run_20260825_093352/snapshot_480m.pt", map_location=device, weights_only=True))
    v1_480m.eval()
    models["ColdWarNetV1_480m"] = v1_480m

    v1_210m = create_coldwar_net(device)
    v1_210m.load_state_dict(torch.load("checkpoints/run_20260825_093352/snapshot_210m.pt", map_location=device, weights_only=True))
    v1_210m.eval()
    models["ColdWarNetV1_210m"] = v1_210m

    # 3. Rule-Based Baselines
    models["HeuristicBot"] = "heuristic"
    models["RandomBot"] = "random"

    model_names = list(models.keys())
    n = len(model_names)
    print(f"Loaded {n} competitors: {model_names}", flush=True)

    # Win Matrix: win_matrix[i, j] = score of model_names[i] against model_names[j]
    win_matrix = np.zeros((n, n), dtype=np.float64)
    detailed_matchups = {}

    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            name_a = model_names[i]
            name_b = model_names[j]
            mod_a = models[name_a]
            mod_b = models[name_b]

            print(f"\n>>> Matchup [{name_a}] vs [{name_b}] (200 Games)...", flush=True)

            # Leg 1: A is USSR (100 games), B is US (100 games)
            res1 = play_directed_matchup(mod_a, mod_b, num_games=100, device=device)
            # Leg 2: B is USSR (100 games), A is US (100 games)
            res2 = play_directed_matchup(mod_b, mod_a, num_games=100, device=device)

            wins_a = res1["ussr_wins"] + res2["us_wins"] + 0.5 * (res1["draws"] + res2["draws"])
            wins_b = res1["us_wins"] + res2["ussr_wins"] + 0.5 * (res1["draws"] + res2["draws"])

            win_matrix[i, j] = wins_a
            win_matrix[j, i] = wins_b

            avg_turn_a_ussr = res1["avg_turn"]
            avg_turn_a_us = res2["avg_turn"]
            win_pct_a = wins_a / 200.0 * 100.0
            win_pct_b = wins_b / 200.0 * 100.0

            print(
                f"  Result: [{name_a}] {wins_a:.0f} - {wins_b:.0f} [{name_b}] "
                f"({win_pct_a:.1f}% vs {win_pct_b:.1f}%) | Avg Turn: {avg_turn_a_ussr:.1f} / {avg_turn_a_us:.1f}",
                flush=True
            )

            detailed_matchups[f"{name_a}__vs__{name_b}"] = {
                "leg1_a_ussr": res1,
                "leg2_a_us": res2,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "win_pct_a": win_pct_a,
                "win_pct_b": win_pct_b,
            }

    # Compute Bradley-Terry Elo
    elo_ratings = compute_bradley_terry_elo(win_matrix, model_names, anchor_name="RandomBot", anchor_elo=1000.0)

    # Sort competitors by Elo
    sorted_models = sorted(model_names, key=lambda m: elo_ratings[m], reverse=True)

    print("\n" + "=" * 90, flush=True)
    print("TOURNAMENT RESULTS & BRADLEY-TERRY ELO RATINGS", flush=True)
    print("=" * 90, flush=True)
    for rank, m in enumerate(sorted_models, 1):
        total_wins = np.sum(win_matrix[model_names.index(m)])
        total_played = 200 * (n - 1)
        win_pct = total_wins / total_played * 100.0
        print(f"#{rank} | {m:24s} | Elo: {elo_ratings[m]:6.1f} | Win Rate: {win_pct:5.1f}% ({total_wins:.0f}/{total_played})", flush=True)

    # Dump Markdown Report
    report_path = "checkpoints/run_v2_20260825_224511/elo_tournament_200_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Twilight Struggle Grand Tournament: Bradley-Terry Elo & Win/Loss Matrix\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
        f.write(f"**Games Per Pairing**: 200 (100 as USSR, 100 as US)\n")
        f.write(f"**Total Games Simulated**: {200 * n * (n - 1) // 2:,}\n\n")

        f.write("## 1. Bradley-Terry Elo Leaderboard\n\n")
        f.write("| Rank | Model / Competitor | Elo Rating | Total Win % | Total Score |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|\n")
        for rank, m in enumerate(sorted_models, 1):
            total_wins = np.sum(win_matrix[model_names.index(m)])
            total_played = 200 * (n - 1)
            win_pct = total_wins / total_played * 100.0
            f.write(f"| {rank} | **{m}** | **{elo_ratings[m]:.1f}** | {win_pct:.1f}% | {total_wins:.0f} / {total_played} |\n")

        f.write("\n## 2. Pairwise Win/Loss Matrix (Rows vs Columns)\n\n")
        f.write("Each cell `(Row, Col)` shows: **Row Model Wins - Col Model Wins (Row Win %)**\n\n")
        header = "| Competitor | " + " | ".join([f"**{m[:12]}**" for m in sorted_models]) + " |"
        f.write(header + "\n")
        f.write("|:---|" + ":---:|"*len(sorted_models) + "\n")
        for r_m in sorted_models:
            r_idx = model_names.index(r_m)
            row_str = f"| **{r_m}** |"
            for c_m in sorted_models:
                c_idx = model_names.index(c_m)
                if r_idx == c_idx:
                    row_str += " — |"
                else:
                    w_r = win_matrix[r_idx, c_idx]
                    w_c = win_matrix[c_idx, r_idx]
                    pct = w_r / (w_r + w_c) * 100.0
                    row_str += f" {w_r:.0f}-{w_c:.0f} ({pct:.0f}%) |"
            f.write(row_str + "\n")

    print(f"\nTournament report saved to: {report_path}", flush=True)


if __name__ == "__main__":
    run_full_tournament()
