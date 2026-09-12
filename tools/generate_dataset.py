import os
import time
import json
import gzip
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, cast
import ts_engine as ts
from tools.lib.player_agent import load_agent, NeuralAgent
from bindings.action_encoder import ActionEncoder

def get_temperature_for_profile(profile_idx: int, turn: int) -> float:
    if profile_idx == 0:  # Profile 1: Grandmaster Crisp
        return 0.1
    elif profile_idx == 1:  # Profile 2: Opening Diversity -> Sharp Mid/Late
        return 0.6 if turn <= 2 else 0.15
    elif profile_idx == 2:  # Profile 3: Sharp Opening -> Mid-War Exploration
        return 0.1 if turn <= 3 else (0.5 if turn <= 7 else 0.15)
    else:  # Profile 4: Diverse Exploration
        return 0.4

def generate_warmup_dataset(
    checkpoints: Dict[str, str],
    total_games: int = 5000,
    batch_size: int = 500,
    output_path: str = "data/datasets/warmup_regenerated.jsonl.gz",
    device_str: str = "cuda"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    print("Loading top models:")
    models = {}
    for name, path in checkpoints.items():
        print(f"  {name}: {path}")
        agent = load_agent(path, device=device)
        if isinstance(agent, NeuralAgent):
            agent.model.eval()
            models[name] = agent.model
        else:
            raise ValueError(f"Expected neural agent for {name}, got {type(agent)}")
        
    model_names = list(models.keys())
    
    print("\n" + "="*80)
    print(f" Generating {total_games:,} Warmup Games with Multi-Temperature Profiles")
    print(f" Output: {output_path}")
    print(f" Parallel Envs: {batch_size} | Device: {device}")
    print("="*80 + "\n")
    
    t_start = time.time()
    games_completed = 0
    total_steps_recorded = 0
    ussr_total_wins = 0
    us_total_wins = 0
    draws_total = 0
    
    profile_names = [
        "Grandmaster_Crisp",
        "Opening_Diversity_To_Sharp_Endgame",
        "Sharp_Opening_To_Midwar_Exploration",
        "Diverse_Exploration"
    ]
    
    with gzip.open(output_path, "wt", encoding="utf-8") as gz_out:
        batch_idx = 0
        while games_completed < total_games:
            current_batch = min(batch_size, total_games - games_completed)
            base_seed = int(time.time() * 1000) % 1_000_000_000 + games_completed * 1007 + 1
            runner = ts.VectorizedBatchRunner(current_batch, base_seed)
            
            # Randomly pair models and assign profiles for each env in batch
            batch_ussr_models = [model_names[np.random.randint(len(model_names))] for _ in range(current_batch)]
            batch_us_models = [model_names[np.random.randint(len(model_names))] for _ in range(current_batch)]
            batch_profiles = [games_completed % 4 for _ in range(current_batch)]
            
            # Step-by-step action recordings per game
            game_histories = [
                {
                    "game_id": f"warmup_{games_completed + i:05d}",
                    "seed": base_seed + i * 10007 + 1,
                    "profile_id": batch_profiles[i],
                    "profile_name": profile_names[batch_profiles[i]],
                    "ussr_model": batch_ussr_models[i],
                    "us_model": batch_us_models[i],
                    "actions": []
                }
                for i in range(current_batch)
            ]
            
            completed = [False] * current_batch
            steps = 0
            
            while not all(completed) and steps < 4000:
                steps += 1
                obs_all = np.array(runner.get_observations(), copy=False)
                masks_all = np.array(runner.get_action_masks(), copy=False)
                players = runner.get_decision_players()
                terminals = runner.get_terminals()
                term_utils = runner.get_terminal_utilities()
                
                # Check for completed games in batch
                for i in range(current_batch):
                    if not completed[i] and terminals[i]:
                        completed[i] = True
                        st = runner.get_state(i)
                        winner = "USSR" if term_utils[i] < 0 else ("US" if term_utils[i] > 0 else "DRAW")
                        if winner == "USSR": ussr_total_wins += 1
                        elif winner == "US": us_total_wins += 1
                        else: draws_total += 1
                        
                        game_histories[i]["winner"] = winner
                        game_histories[i]["final_vp"] = int(st.victory_points)
                        game_histories[i]["end_turn"] = int(st.turn)
                        game_histories[i]["total_steps"] = len(game_histories[i]["actions"])
                        
                if all(completed):
                    break
                    
                active = [i for i in range(current_batch) if not completed[i]]
                actions = [0] * current_batch
                
                # Group active environments by active model
                model_to_envs: Dict[str, List[int]] = {name: [] for name in model_names}
                
                for i in active:
                    p = players[i]
                    m_name = batch_ussr_models[i] if p == -1 else batch_us_models[i]
                    model_to_envs[m_name].append(i)
                    
                for m_name, env_indices in model_to_envs.items():
                    if not env_indices:
                        continue
                    m = models[m_name]
                    obs_sub = torch.from_numpy(obs_all[env_indices]).float().to(device)
                    mask_sub = torch.from_numpy(masks_all[env_indices]).to(device)
                    
                    # Compute per-env temperature based on profile & turn
                    # For batched sampling efficiency, average temp in sub-batch
                    turns = [runner.get_state(i).turn for i in env_indices]
                    temps = [get_temperature_for_profile(batch_profiles[i], turns[idx]) for idx, i in enumerate(env_indices)]
                    avg_temp = float(np.mean(temps))
                    
                    with torch.no_grad():
                        act_t, _, _, _, _ = cast(Any, m).sample_action(obs_sub, mask_sub, temperature=avg_temp, deterministic=(avg_temp <= 0.05))
                    act_list = act_t.cpu().numpy().tolist()
                    
                    for sub_i, env_i in enumerate(env_indices):
                        a_flat = int(act_list[sub_i])
                        actions[env_i] = a_flat
                        
                        st = runner.get_state(env_i)
                        game_histories[env_i]["actions"].append({
                            "step": len(game_histories[env_i]["actions"]),
                            "turn": int(st.turn),
                            "player": "USSR" if players[env_i] == -1 else "US",
                            "flat_action": a_flat
                        })
                        
                runner.step_flat_all(actions)
                
            # Write batch of finished games to compressed JSONL
            for g in game_histories:
                gz_out.write(json.dumps(g) + chr(10))
                steps_val = g.get("total_steps")
                total_steps_recorded += int(steps_val) if isinstance(steps_val, int) else len(cast(list, g.get("actions", [])))
                
            games_completed += current_batch
            batch_idx += 1
            elapsed = time.time() - t_start
            speed_gps = games_completed / max(elapsed, 0.001)
            print(f"  [{games_completed:5d}/{total_games:5d}] Generated in {elapsed:5.1f}s ({speed_gps:4.0f} games/s) | Total Steps: {total_steps_recorded:,} | USSR Wins: {ussr_total_wins} | US Wins: {us_total_wins}")
            
    total_time = time.time() - t_start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "="*80)
    print(f" WARMUP DATASET GENERATION COMPLETE!")
    print(f" - Total Games: {games_completed:,}")
    print(f" - Total Transitions: {total_steps_recorded:,}")
    print(f" - Dataset File: {output_path} ({file_size_mb:.2f} MB)")
    print(f" - Total Elapsed: {total_time:.1f}s (Average {games_completed/total_time:.0f} games/s)")
    print(f" - Win Balance: USSR {ussr_total_wins/games_completed*100:.1f}% | US {us_total_wins/games_completed*100:.1f}% | Draws {draws_total/games_completed*100:.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate high-quality demonstration datasets from top neural snapshots")
    parser.add_argument("--models", nargs="+", default=None, help="List of model checkpoint paths")
    parser.add_argument("--total-games", type=int, default=5000, help="Total games to generate")
    parser.add_argument("--batch-size", type=int, default=500, help="Parallel batch size")
    parser.add_argument("--output-path", type=str, default="data/datasets/warmup_regenerated.jsonl.gz",
                        help="Output .jsonl.gz path. Name it after the generating checkpoint and the "
                             "engine it was built against -- a demonstration set in the (seed, actions) "
                             "format is only valid for the engine that produced it, and silently "
                             "truncates against any other.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Named outright, never guessed. A demonstration set carries the policy that generated it,
    # so which checkpoints were used is part of what the dataset is -- falling back to whichever
    # snapshot happens to be on disk would make two invocations of this command mean different
    # things.
    if not args.models:
        parser.error("--models is required: give the checkpoint path(s) to generate from.")
    top_checkpoints = {os.path.splitext(os.path.basename(m))[0]: m for m in args.models}
    missing = [m for m in args.models if not os.path.exists(m)]
    if missing:
        parser.error("checkpoint(s) not found: " + ", ".join(missing))
    
    generate_warmup_dataset(
        checkpoints=top_checkpoints,
        total_games=args.total_games,
        batch_size=args.batch_size,
        output_path=args.output_path,
        device_str=args.device
    )
