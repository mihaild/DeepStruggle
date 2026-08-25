import os
import sys
import time
import gzip
import json
import math
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2, ColdWarNetV2
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.eval.self_play import generate_selfplay_replay
from server.replay_types import TrainingMetricEntryDict
from ai.training.warmup_dataset_loader import WarmupDataset
from ai.env.ts_env import TsVectorizedEnv as TsVectorEnv
from ai.training.rollout_buffer import RolloutBuffer
from bot.bot_client import HeuristicBot, RandomBot


class StreamingWarmupDataset(IterableDataset):
    """Zero-memory-overhead streaming dataset directly from compressed game logs."""
    def __init__(self, filepath: str = "data/warmup_5k_games.jsonl.gz", max_transitions: int = 500_000):
        super().__init__()
        self.filepath = filepath
        self.max_transitions = max_transitions

    def __iter__(self):
        ds = WarmupDataset(self.filepath)
        count = 0
        for obs, mask, act, win_ret, vp_ret in ds.stream_transitions():
            if self.max_transitions is not None and count >= self.max_transitions:
                break
            yield (
                torch.from_numpy(obs).float(),
                torch.from_numpy(mask).to(torch.uint8),
                torch.tensor(act, dtype=torch.long),
                torch.tensor(win_ret, dtype=torch.float32),
                torch.tensor(vp_ret, dtype=torch.float32)
            )
            count += 1


def run_warmup_pretraining(
    model: ColdWarNetV2,
    dataset_path: str = "data/warmup_5k_games.jsonl.gz",
    epochs: int = 5,
    samples_per_epoch: int = 500_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    save_path: str = "checkpoints/coldwar_net_v2_warmup.pt",
):
    print("\n" + "=" * 80, flush=True)
    print("PHASE 1: SUPERVISED WARM-UP PRE-TRAINING (ColdWarNetV2)", flush=True)
    print(f"Dataset: {dataset_path} | Epochs: {epochs} | Transitions/Epoch: {samples_per_epoch:,} | Batch Size: {batch_size} | Device: {device}", flush=True)
    print("=" * 80, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * (samples_per_epoch // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    model.train()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        sds = StreamingWarmupDataset(dataset_path, max_transitions=samples_per_epoch)
        loader = DataLoader(sds, batch_size=batch_size, drop_last=True)

        total_loss = 0.0
        total_pol_loss = 0.0
        total_win_loss = 0.0
        total_vp_loss = 0.0
        correct_actions = 0
        total_acts = 0

        for b_obs, b_mask, b_act, b_win, b_vp in loader:
            b_obs = b_obs.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_act = b_act.to(device, non_blocking=True)
            b_win = b_win.to(device, non_blocking=True)
            b_vp = b_vp.to(device, non_blocking=True)

            logits, v_win, v_vp = model(b_obs, b_mask)
            pol_loss = F.cross_entropy(logits, b_act)
            win_loss = F.mse_loss(v_win.squeeze(-1), b_win)
            vp_loss = F.mse_loss(v_vp.squeeze(-1), b_vp)

            loss = pol_loss + 0.5 * win_loss + 0.05 * vp_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = len(b_act)
            total_loss += loss.item() * bs
            total_pol_loss += pol_loss.item() * bs
            total_win_loss += win_loss.item() * bs
            total_vp_loss += vp_loss.item() * bs

            preds = torch.argmax(logits, dim=-1)
            correct_actions += (preds == b_act).sum().item()
            total_acts += bs

        avg_loss = total_loss / max(total_acts, 1)
        avg_pol = total_pol_loss / max(total_acts, 1)
        avg_win = total_win_loss / max(total_acts, 1)
        avg_vp = total_vp_loss / max(total_acts, 1)
        acc = correct_actions / max(total_acts, 1) * 100.0
        elapsed = time.time() - ep_t0

        print(
            f"Epoch {ep:2d}/{epochs:2d} ({elapsed:5.1f}s) | Loss: {avg_loss:.4f} (Pol: {avg_pol:.4f}, WinMSE: {avg_win:.4f}, VpMSE: {avg_vp:.4f}) | Action Accuracy: {acc:5.2f}%",
            flush=True
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved warm-up model checkpoint to {save_path}", flush=True)
    return model


def play_matchup(
    model_ussr: Any,
    model_us: Any,
    num_games: int = 50,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.1,
    max_steps_per_game: int = 4000,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """Simulates num_games between two models in parallel."""
    games_completed = 0
    ussr_wins = 0
    us_wins = 0
    draws = 0
    total_turns = 0

    batch_runner = ts.VectorizedBatchRunner(batch_size, int(time.time() * 1000) % 1000000)
    batch_runner.reset_all()

    step_counts = np.zeros(batch_size, dtype=np.int32)

    while games_completed < num_games:
        obs_raw = batch_runner.get_observations()
        masks_raw = batch_runner.get_action_masks()
        players = batch_runner.get_decision_players()

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
            elif model_ussr == "heuristic":
                for idx in ussr_idx:
                    st = batch_runner.get_game_state(idx)
                    m = HeuristicBot.choose_action(st)
                    actions[idx] = ts.encode_micro_action(st, m)
            elif model_ussr == "random":
                for idx in ussr_idx:
                    st = batch_runner.get_game_state(idx)
                    m = RandomBot.choose_action(st)
                    actions[idx] = ts.encode_micro_action(st, m)

        # US masks & actions
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
                    st = batch_runner.get_game_state(idx)
                    m = HeuristicBot.choose_action(st)
                    actions[idx] = ts.encode_micro_action(st, m)
            elif model_us == "random":
                for idx in us_idx:
                    st = batch_runner.get_game_state(idx)
                    m = RandomBot.choose_action(st)
                    actions[idx] = ts.encode_micro_action(st, m)

        # Neutral fallback
        none_idx = np.where(players == 0)[0]
        if len(none_idx) > 0:
            for idx in none_idx:
                legal = np.where(masks_raw[idx] == 1)[0]
                actions[idx] = legal[0] if len(legal) > 0 else 0

        dones = batch_runner.step_flat_all(actions)
        step_counts += 1

        for i in range(batch_size):
            st = batch_runner.get_game_state(i)
            if dones[i] or step_counts[i] >= max_steps_per_game:
                if games_completed < num_games:
                    games_completed += 1
                    total_turns += st.turn
                    if st.winner == ts.Player.USSR or (
                        st.winner == ts.Player.NONE and st.victory_points <= -20
                    ):
                        ussr_wins += 1
                    elif st.winner == ts.Player.US or (
                        st.winner == ts.Player.NONE and st.victory_points >= 20
                    ):
                        us_wins += 1
                    else:
                        draws += 1

                batch_runner.reset_env(i, int(time.time() * 1000 + i * 997) % 1000000)
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


def evaluate_snapshot_matchups(
    snapshot_model: ColdWarNetV2,
    snapshot_name: str,
    all_known_models: Dict[str, Any],
    device: torch.device = torch.device("cuda"),
    games_per_side: int = 50,
) -> Dict[str, Any]:
    """Evaluates a new snapshot against all known baselines and prior snapshots."""
    print(f"\n--- Running Intermediate Tournament for [{snapshot_name}] ---", flush=True)
    snapshot_model.eval()
    results = {}

    for opp_name, opp_model in all_known_models.items():
        if opp_name == snapshot_name:
            continue
        # Snapshot as USSR
        res_ussr = play_matchup(snapshot_model, opp_model, num_games=games_per_side, device=device)
        # Snapshot as US
        res_us = play_matchup(opp_model, snapshot_model, num_games=games_per_side, device=device)

        overall_win = (res_ussr["ussr_wins"] + res_us["us_wins"]) / (2 * games_per_side) * 100.0
        print(
            f"  vs {opp_name:18s} | Overall Win: {overall_win:5.1f}% | USSR Win: {res_ussr[ussr_win_rate]:5.1f}% | US Win: {res_us[us_win_rate]:5.1f}% (Avg Turn {res_ussr[avg_turn]:.1f}/{res_us[avg_turn]:.1f})",
            flush=True
        )
        results[opp_name] = {
            "as_ussr": res_ussr,
            "as_us": res_us,
            "overall_win": overall_win,
        }
    return results


def run_comprehensive_tournament(
    target_model: ColdWarNetV2,
    prev_checkpoints: Dict[str, str],
    device: torch.device = torch.device("cuda"),
    games_per_pairing: int = 50,
) -> Dict[str, Any]:
    print("\n" + "=" * 80, flush=True)
    print(
        f"COMPREHENSIVE TOURNAMENT EVALUATION ({games_per_pairing} games as US, {games_per_pairing} as USSR per opponent)",
        flush=True
    )
    print("=" * 80, flush=True)

    opponents = {}
    opponents["RandomBot"] = "random"
    opponents["HeuristicBot"] = "heuristic"

    for name, path in prev_checkpoints.items():
        if os.path.exists(path):
            net = create_coldwar_net(device)
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            net.eval()
            opponents[name] = net

    results = {}
    target_model.eval()

    # 1. Target vs Itself
    self_res = play_matchup(target_model, target_model, num_games=games_per_pairing, device=device)
    print(
        f"\n[ColdWarNetV2 vs Self]: USSR Win: {self_res["ussr_win_rate"]:5.1f}% | US Win: {self_res["us_win_rate"]:5.1f}% | Avg Turn: {self_res["avg_turn"]:.1f}",
        flush=True
    )
    results["SelfPlay"] = self_res

    # 2. Target vs Each Opponent
    for name, opp in opponents.items():
        res_ussr = play_matchup(target_model, opp, num_games=games_per_pairing, device=device)
        res_us = play_matchup(opp, target_model, num_games=games_per_pairing, device=device)

        target_as_us_win_rate = res_us["us_win_rate"]
        overall_win_rate = (
            (res_ussr["ussr_wins"] + res_us["us_wins"]) / (2 * games_per_pairing) * 100.0
        )

        print(f"\n[ColdWarNetV2 vs {name:15s}]: Overall Win: {overall_win_rate:5.1f}%", flush=True)
        print(
            f"  • As USSR ({games_per_pairing} games): Win {res_ussr["ussr_win_rate"]:5.1f}% (Avg Turn {res_ussr["avg_turn"]:.1f})",
            flush=True
        )
        print(
            f"  • As US   ({games_per_pairing} games): Win {target_as_us_win_rate:5.1f}% (Avg Turn {res_us["avg_turn"]:.1f})",
            flush=True
        )

        results[name] = {
            "as_ussr": res_ussr,
            "as_us": res_us,
            "overall_win_rate": overall_win_rate,
        }

    return results


def run_1hr_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("STARTING FULL 1-HOUR NASHPG TRAINING & EVALUATION PIPELINE (ColdWarNetV2)", flush=True)
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    print("=" * 80, flush=True)

    # 1. Create Model V2
    model = create_coldwar_net_v2(device)

    # 2. Phase 1: Warm-Up on 5,000 games (Streaming with zero memory overhead)
    warmup_path = "checkpoints/coldwar_net_v2_warmup.pt"
    if os.path.exists(warmup_path):
        print(f"Loading existing warm-up model from {warmup_path}...", flush=True)
        model.load_state_dict(torch.load(warmup_path, map_location=device, weights_only=True))
    else:
        run_warmup_pretraining(
            model=model,
            dataset_path="data/warmup_5k_games.jsonl.gz",
            epochs=5,
            samples_per_epoch=500_000,
            batch_size=2048,
            lr=1e-3,
            device=device,
            save_path=warmup_path,
        )

    # 3. Setup Timestamped Output Run Directory
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"checkpoints/run_v2_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutput Run Directory: {run_dir}", flush=True)

    metrics_log_path = f"{run_dir}/training_metrics.jsonl"
    intermediate_report_path = f"{run_dir}/intermediate_tournaments.md"

    # Save initial 0m snapshot
    snap_0m_path = f"{run_dir}/snapshot_0m.pt"
    torch.save(model.state_dict(), snap_0m_path)
    print(f"Saved Snapshot 0m to {snap_0m_path}", flush=True)

    # Active model registry for intermediate comparisons
    known_models = {
        "RandomBot": "random",
        "HeuristicBot": "heuristic",
    }
    prev_anchors = {
        "Snapshot_210m_v1": "checkpoints/run_20260825_093352/snapshot_210m.pt",
        "Snapshot_180m_v1": "checkpoints/run_20260825_093352/snapshot_180m.pt",
    }
    for name, path in prev_anchors.items():
        if os.path.exists(path):
            net = create_coldwar_net(device)
            net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            net.eval()
            known_models[name] = net

    # Register initial Snapshot 0m
    snap_0m_net = create_coldwar_net_v2(device)
    snap_0m_net.load_state_dict(model.state_dict())
    snap_0m_net.eval()
    known_models["Snapshot_0m"] = snap_0m_net

    # 4. Setup NashPG Vectorized Environment
    num_envs = 512
    buffer_size = 128
    batch_size = 4096
    env = TsVectorEnv(num_envs=num_envs, base_seed=12345)

    ref_model = create_coldwar_net_v2(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()

    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        num_envs=num_envs,
        obs_dim=4293,
        action_dim=212,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 4-Profile Stratified Temperature Assignments across 512 Envs
    env_temps = np.zeros(num_envs, dtype=np.float32)
    # Profile A (0..127): Sharp Opening -> Mid-War Exp
    env_temps[0:128] = 0.15
    # Profile B (128..255): Opening Diversity -> Sharp Recovery
    env_temps[128:256] = 0.50
    # Profile C (256..383): Crisp Grandmaster
    env_temps[256:384] = 0.10
    # Profile D (384..511): General Exploration
    env_temps[384:512] = 0.35

    obs_np, masks_np, _ = env.reset_all()

    total_training_duration = 3600  # 60 minutes (1 hour)
    eval_interval = 600  # 10 minutes
    next_eval_time = eval_interval

    t_start = time.time()
    it = 0
    total_env_steps = 0
    steps_since_ref_update = 0
    ref_update_freq = 200_000

    print("\n" + "=" * 80, flush=True)
    print(f"PHASE 2: 1-HOUR NASHPG REINFORCEMENT LEARNING ({num_envs} Parallel Envs)", flush=True)
    print("=" * 80, flush=True)

    with open(intermediate_report_path, "w", encoding="utf-8") as f:
        f.write("# Intermediate Snapshot Tournament Tracker (ColdWarNetV2)\n\n")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= total_training_duration:
            break

        it += 1
        model.eval()
        buffer.reset()

        # Rollout Collection
        t_roll = time.time()
        for step in range(buffer_size):
            obs_t = torch.from_numpy(obs_np).float().to(device)
            masks_t = torch.from_numpy(masks_np).to(device)

            with torch.no_grad():
                logits, v_win, v_vp = model(obs_t, masks_t)
                scaled_logits = logits / torch.from_numpy(env_temps).unsqueeze(-1).to(device)
                dist = torch.distributions.Categorical(logits=scaled_logits)
                actions_t = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

            actions_np = actions_t.cpu().numpy()
            next_obs_np, next_masks_np, rewards_np, dones_np, info = env.step(actions_np)

            players_np = info["decision_players"]
            buffer.add(
                obs=obs_t,
                masks=masks_t,
                actions=actions_t,
                log_probs=log_probs_t,
                values_win=v_win.squeeze(-1),
                values_vp=v_vp.squeeze(-1),
                rewards=torch.from_numpy(rewards_np).float().to(device),
                dones=torch.from_numpy(dones_np).float().to(device),
                players=torch.from_numpy(players_np).to(device),
            )

            obs_np = next_obs_np
            masks_np = next_masks_np

        # GAE Bootstrap
        last_obs_t = torch.from_numpy(obs_np).float().to(device)
        last_masks_t = torch.from_numpy(masks_np).to(device)
        with torch.no_grad():
            _, last_v_win, last_v_vp = model(last_obs_t, last_masks_t)
            last_dones = torch.from_numpy(dones_np).to(device)
            last_players = torch.from_numpy(info["decision_players"]).to(device)

        buffer.compute_gae(
            last_v_win=last_v_win.squeeze(-1),
            last_v_vp=last_v_vp.squeeze(-1),
            last_dones=last_dones,
            last_players=last_players,
            gamma=0.995,
            gae_lambda=0.95,
        )

        steps_collected = buffer_size * num_envs
        total_env_steps += steps_collected
        steps_since_ref_update += steps_collected
        roll_fps = steps_collected / max(time.time() - t_roll, 1e-6)

        # Policy Update
        model.train()
        total_loss_accum = 0.0
        pol_loss_accum = 0.0
        val_loss_accum = 0.0
        kl_accum = 0.0
        entropy_accum = 0.0
        clip_frac_accum = 0.0
        num_updates = 0

        for _ in range(4):
            for b_obs, b_mask, b_act, b_old_log_probs, b_adv, b_ret_win, b_ret_vp in buffer.get_batches(batch_size):
                cur_log_probs, cur_entropy, cur_v_win, cur_v_vp = model.evaluate_actions(
                    b_obs, b_mask, b_act
                )

                ratio = torch.exp(cur_log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                clipped = (ratio < 0.8) | (ratio > 1.2)
                clip_frac = clipped.float().mean().item()

                with torch.no_grad():
                    ref_logits, _, _ = ref_model(b_obs, b_mask)
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)

                cur_logits, _, _ = model(b_obs, b_mask)
                cur_p = F.softmax(cur_logits, dim=-1)
                cur_lp = F.log_softmax(cur_logits, dim=-1)
                kl_div = torch.sum(cur_p * (cur_lp - ref_log_p), dim=-1).mean()

                policy_loss = ppo_loss + 0.1 * kl_div - 0.01 * cur_entropy.mean()
                val_loss = F.mse_loss(cur_v_win, b_ret_win) + 0.05 * F.mse_loss(cur_v_vp, b_ret_vp)
                loss = policy_loss + 0.5 * val_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_accum += loss.item()
                pol_loss_accum += ppo_loss.item()
                val_loss_accum += val_loss.item()
                kl_accum += kl_div.item()
                entropy_accum += cur_entropy.mean().item()
                clip_frac_accum += clip_frac
                num_updates += 1

        if steps_since_ref_update >= ref_update_freq:
            ref_model.load_state_dict(model.state_dict())
            steps_since_ref_update = 0

        # Log Metrics
        mean_loss = total_loss_accum / max(num_updates, 1)
        mean_pol = pol_loss_accum / max(num_updates, 1)
        mean_val = val_loss_accum / max(num_updates, 1)
        mean_kl = kl_accum / max(num_updates, 1)
        mean_ent = entropy_accum / max(num_updates, 1)
        mean_clip = clip_frac_accum / max(num_updates, 1)

        # Write to JSONL
        metric_entry: TrainingMetricEntryDict = {
            "it": it,
            "elapsed_sec": round(elapsed, 1),
            "env_steps": total_env_steps,
            "fps": round(roll_fps, 1),
            "loss": round(mean_loss, 4),
            "pol_loss": round(mean_pol, 4),
            "val_loss": round(mean_val, 4),
            "kl_div": round(mean_kl, 4),
            "entropy": round(mean_ent, 3),
            "clip_frac": round(mean_clip, 3),
        }
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_entry) + "\n")

        if it % 10 == 0:
            print(
                f"Iter {it:4d} | Elapsed: {elapsed/60:4.1f}m/60m | Steps: {total_env_steps:8,d} ({roll_fps:5.0f} st/s) | "
                f"Loss: {mean_loss:.4f} (Pol: {mean_pol:.4f}, Val: {mean_val:.4f}, KL: {mean_kl:.4f}, Entropy: {mean_ent:.3f}, ClipFrac: {mean_clip:.2f})",
                flush=True
            )

        # Check 10-Minute Snapshot Condition
        if elapsed >= next_eval_time:
            snap_min = int(round(elapsed / 60.0 / 10.0) * 10)
            snap_name = f"Snapshot_{snap_min}m"
            snap_path = f"{run_dir}/snapshot_{snap_min}m.pt"
            torch.save(model.state_dict(), snap_path)
            print(f"\n>>> [SNAPSHOT {snap_min}m] Saved checkpoint: {snap_path}", flush=True)

            # Intermediate Benchmark Tournament
            snap_eval_net = create_coldwar_net_v2(device)
            snap_eval_net.load_state_dict(model.state_dict())
            snap_eval_net.eval()

            bench_res = evaluate_snapshot_matchups(
                snapshot_model=snap_eval_net,
                snapshot_name=snap_name,
                all_known_models=known_models,
                device=device,
                games_per_side=50
            )
            known_models[snap_name] = snap_eval_net

            # Append to intermediate report
            with open(intermediate_report_path, "a", encoding="utf-8") as f:
                f.write(f"\n### {snap_name} (Elapsed: {elapsed/60:.1f}m, Steps: {total_env_steps:,})\n\n")
                f.write("| Opponent | Overall Win % | Win % as USSR | Win % as US | Avg Turn (USSR/US) |\n")
                f.write("|:---|:---:|:---:|:---:|:---:|\n")
                for opp_k, res_v in bench_res.items():
                    f.write(
                        f"| {opp_k:18s} | **{res_v[overall_win]:5.1f}%** | {res_v[as_ussr][ussr_win_rate]:5.1f}% | {res_v[as_us][us_win_rate]:5.1f}% | {res_v[as_ussr][avg_turn]:.1f} / {res_v[as_us][avg_turn]:.1f} |\n"
                    )

            next_eval_time += eval_interval

    # Save final 60m model
    final_path = f"{run_dir}/snapshot_60m.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinished 1-Hour RL training! Final model saved to {final_path}", flush=True)

    # 5. Comprehensive Tournament
    prev_models = {
        "Snapshot_210m_v1": "checkpoints/run_20260825_093352/snapshot_210m.pt",
        "Snapshot_180m_v1": "checkpoints/run_20260825_093352/snapshot_180m.pt",
        "Snapshot_360m_v1": "checkpoints/run_20260825_093352/snapshot_360m.pt",
        "Snapshot_480m_v1": "checkpoints/run_20260825_093352/snapshot_480m.pt",
    }
    tourney_results = run_comprehensive_tournament(
        target_model=model,
        prev_checkpoints=prev_models,
        device=device,
        games_per_pairing=50,
    )

    # 6. Generate Self-Play Demo Replay
    replay_path = "replays/v2_selfplay_demo.tslog.json"
    generate_selfplay_replay(model=model, output_path=replay_path, device=device)

    # 7. Write Final Report
    report_path = f"{run_dir}/tournament_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ColdWarNetV2 1-Hour Training & Tournament Report\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
        f.write(f"**Model Architecture**: `ColdWarNetV2` (Card-to-Country Cross-Attention + GCN)\n")
        f.write(f"**Total Steps Trained**: {total_env_steps:,}\n\n")
        f.write("## Tournament Results vs All Baselines\n\n")
        f.write("| Opponent | Overall Win % | Win % as USSR | Win % as US |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for opp_name, data in tourney_results.items():
            if opp_name == "SelfPlay":
                f.write(
                    f"| ColdWarNetV2 (Self) | 50.0% | {data["ussr_win_rate"]:.1f}% | {data["us_win_rate"]:.1f}% |\n"
                )
            else:
                f.write(
                    f"| {opp_name} | **{data["overall_win_rate"]:.1f}%** | {data["as_ussr"]["ussr_win_rate"]:.1f}% | {data["as_us"]["us_win_rate"]:.1f}% |\n"
                )

    print(f"\nTournament report saved to {report_path}", flush=True)
    print("\n=======================================================", flush=True)
    print("ALL TASKS COMPLETED SUCCESSFULLY!", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    run_1hr_training_pipeline()
