"""Self-play demonstration dataset: (seed, actions) traces from chosen checkpoints.

    PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \\
        --models <checkpoint.pt> [<checkpoint.pt> ...] --total-games 5000 --output-path <dataset.jsonl.gz>

Each game pairs two of the given networks at random, one per seat, and plays it under one of four
temperature profiles (assigned round-robin over the games). Every network decides in its own view:
observations in its feature set (`set_obs_features`) and masks in its action view
(`set_merged_influence`); each game records both seats' views (`us_merged_influence`,
`ussr_merged_influence`) so `WarmupDataset` replays a merged-view seat's composed actions in that
view, and both seats' feature sets (`us_obs_features`, `ussr_obs_features`), whose rule bits are the
rules each seat played (`ts.OBS_FEATURES_RULES`).
"""
import argparse
import gzip
import json
import os
import time
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch

import ts_engine as ts
from tools.lib.checkpoint_id import checkpoint_label
from tools.lib.data_root import data_path
from tools.lib.player_agent import NeuralAgent, load_agent

#: Runner steps a batch may take. A game is a few hundred decisions; one still running here is a
#: stuck agent or an engine loop, never a long game.
MAX_STEPS = 4000

PROFILE_NAMES = (
    "Grandmaster_Crisp",
    "Opening_Diversity_To_Sharp_Endgame",
    "Sharp_Opening_To_Midwar_Exploration",
    "Diverse_Exploration",
)


def get_temperature_for_profile(profile_idx: int, turn: int) -> float:
    if profile_idx == 0:  # Profile 1: Grandmaster Crisp
        return 0.1
    elif profile_idx == 1:  # Profile 2: Opening Diversity -> Sharp Mid/Late
        return 0.6 if turn <= 2 else 0.15
    elif profile_idx == 2:  # Profile 3: Sharp Opening -> Mid-War Exploration
        return 0.1 if turn <= 3 else (0.5 if turn <= 7 else 0.15)
    else:  # Profile 4: Diverse Exploration
        return 0.4


class UnfinishedGames(RuntimeError):
    """Games still running after MAX_STEPS runner steps: never written as finished."""


def generate_warmup_dataset(
    checkpoints: Dict[str, str],
    output_path: str,
    total_games: int = 5000,
    batch_size: int = 500,
    device_str: str = "cuda",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("Loading top models:")
    agents: Dict[str, NeuralAgent] = {}
    for name, path in checkpoints.items():
        print(f"  {name}: {path}")
        agent = load_agent(path, device=device)
        if not isinstance(agent, NeuralAgent):
            raise ValueError(f"Expected neural agent for {name}, got {type(agent)}")
        agent.model.eval()
        agents[name] = agent
        print(f"    obs features {agent.obs_features:#x}, "
              f"{'merged-influence (E4.1)' if agent.merged_influence else 'E4'} view")

    model_names = list(agents.keys())

    print("\n" + "=" * 80)
    print(f" Generating {total_games:,} Warmup Games with Multi-Temperature Profiles")
    print(f" Output: {output_path}")
    print(f" Parallel Envs: {batch_size} | Device: {device}")
    print("=" * 80 + "\n")

    t_start = time.time()
    games_completed = 0
    total_steps_recorded = 0
    ussr_total_wins = 0
    us_total_wins = 0
    draws_total = 0

    with gzip.open(output_path, "wt", encoding="utf-8") as gz_out:
        while games_completed < total_games:
            current_batch = min(batch_size, total_games - games_completed)
            base_seed = int(time.time() * 1000) % 1_000_000_000 + games_completed * 1007 + 1
            runner = ts.VectorizedBatchRunner(current_batch, base_seed)

            # Randomly pair models; profiles round-robin over the games, so every batch -- and so
            # every dataset of at least four games -- holds all four.
            batch_ussr_models = [model_names[np.random.randint(len(model_names))] for _ in range(current_batch)]
            batch_us_models = [model_names[np.random.randint(len(model_names))] for _ in range(current_batch)]
            batch_profiles = [(games_completed + i) % len(PROFILE_NAMES) for i in range(current_batch)]

            # Each seat in its own network's view. Set before the first observation is read:
            # set_obs_features rebuilds every env's rows.
            runner.set_obs_features([agents[m].obs_features for m in batch_us_models],
                                    [agents[m].obs_features for m in batch_ussr_models])
            runner.set_merged_influence([agents[m].merged_influence for m in batch_us_models],
                                        [agents[m].merged_influence for m in batch_ussr_models])

            # Step-by-step action recordings per game. The seed is the runner's own per-env seed
            # (VectorizedBatchRunner: base_seed + i * 10007 + 1).
            game_histories: List[Dict[str, Any]] = [
                {
                    "game_id": f"warmup_{games_completed + i:05d}",
                    "seed": base_seed + i * 10007 + 1,
                    "profile_id": batch_profiles[i],
                    "profile_name": PROFILE_NAMES[batch_profiles[i]],
                    "ussr_model": batch_ussr_models[i],
                    "us_model": batch_us_models[i],
                    "us_merged_influence": agents[batch_us_models[i]].merged_influence,
                    "ussr_merged_influence": agents[batch_ussr_models[i]].merged_influence,
                    # Each seat's feature set, whose rule bits are the rules it played
                    # (WarmupDataset seats them on replay).
                    "us_obs_features": agents[batch_us_models[i]].obs_features,
                    "ussr_obs_features": agents[batch_ussr_models[i]].obs_features,
                    "actions": [],
                }
                for i in range(current_batch)
            ]

            completed = [False] * current_batch
            steps = 0

            def absorb_finished() -> None:
                nonlocal ussr_total_wins, us_total_wins, draws_total
                terminals = runner.get_terminals()
                term_utils = runner.get_terminal_utilities()
                for i in range(current_batch):
                    if not completed[i] and terminals[i]:
                        completed[i] = True
                        st = runner.get_state(i)
                        winner = "USSR" if term_utils[i] < 0 else ("US" if term_utils[i] > 0 else "DRAW")
                        if winner == "USSR":
                            ussr_total_wins += 1
                        elif winner == "US":
                            us_total_wins += 1
                        else:
                            draws_total += 1
                        game_histories[i]["winner"] = winner
                        game_histories[i]["final_vp"] = int(st.victory_points)
                        game_histories[i]["end_turn"] = int(st.turn)
                        game_histories[i]["total_steps"] = len(game_histories[i]["actions"])

            absorb_finished()
            while not all(completed):
                if steps >= MAX_STEPS:
                    stuck = [i for i in range(current_batch) if not completed[i]]
                    detail = "; ".join(
                        f"seed {game_histories[i]['seed']} ({game_histories[i]['us_model']} vs "
                        f"{game_histories[i]['ussr_model']}) at turn {int(runner.get_state(i).turn)}, "
                        f"{len(game_histories[i]['actions'])} decisions"
                        for i in stuck[:5])
                    raise UnfinishedGames(
                        f"{len(stuck)} of {current_batch} games in this batch did not finish in "
                        f"{MAX_STEPS} steps (a stuck agent or an engine loop); nothing of this batch "
                        f"was written. First: {detail}")
                steps += 1
                obs_all = np.array(runner.get_observations(), copy=False)
                masks_all = np.array(runner.get_action_masks(), copy=False)
                players = runner.get_decision_players()

                active = [i for i in range(current_batch) if not completed[i]]
                actions = [0] * current_batch

                # Group active environments by deciding model and temperature: each env samples at
                # its own profile's temperature.
                groups: Dict[Tuple[str, float], List[int]] = {}
                for i in active:
                    m_name = batch_ussr_models[i] if players[i] == -1 else batch_us_models[i]
                    temp = get_temperature_for_profile(batch_profiles[i], int(runner.get_state(i).turn))
                    groups.setdefault((m_name, temp), []).append(i)

                for (m_name, temp), env_indices in groups.items():
                    agent = agents[m_name]
                    # Rows are zero-padded to the widest set in the batch; this network reads the
                    # first obs_size floats of its own.
                    obs_sub = torch.from_numpy(obs_all[env_indices, :agent.obs_size]).float().to(device)
                    mask_sub = torch.from_numpy(masks_all[env_indices]).to(device)
                    with torch.no_grad():
                        act_t, _, _, _, _ = cast(Any, agent.model).sample_action(
                            obs_sub, mask_sub, temperature=temp, deterministic=(temp <= 0.05))
                    act_list = act_t.cpu().numpy().tolist()

                    for sub_i, env_i in enumerate(env_indices):
                        a_flat = int(act_list[sub_i])
                        actions[env_i] = a_flat
                        st = runner.get_state(env_i)
                        game_histories[env_i]["actions"].append({
                            "step": len(game_histories[env_i]["actions"]),
                            "turn": int(st.turn),
                            "player": "USSR" if players[env_i] == -1 else "US",
                            "flat_action": a_flat,
                        })

                results = runner.step_flat_all(actions)
                refused = [i for i in active if not results[i]]
                if refused:
                    i = refused[0]
                    raise RuntimeError(
                        f"the engine refused flat action {actions[i]} in game seed "
                        f"{game_histories[i]['seed']} ({len(refused)} refusals this step): the "
                        f"recorded trace would no longer replay")
                absorb_finished()

            # Write batch of finished games to compressed JSONL
            for g in game_histories:
                gz_out.write(json.dumps(g) + chr(10))
                total_steps_recorded += int(g["total_steps"])

            games_completed += current_batch
            elapsed = time.time() - t_start
            speed_gps = games_completed / max(elapsed, 0.001)
            print(f"  [{games_completed:5d}/{total_games:5d}] Generated in {elapsed:5.1f}s ({speed_gps:4.0f} games/s) | Total Steps: {total_steps_recorded:,} | USSR Wins: {ussr_total_wins} | US Wins: {us_total_wins}")

    total_time = time.time() - t_start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 80)
    print(" WARMUP DATASET GENERATION COMPLETE!")
    print(f" - Total Games: {games_completed:,}")
    print(f" - Total Transitions: {total_steps_recorded:,}")
    print(f" - Dataset File: {output_path} ({file_size_mb:.2f} MB)")
    print(f" - Total Elapsed: {total_time:.1f}s (Average {games_completed/total_time:.0f} games/s)")
    print(f" - Win Balance: USSR {ussr_total_wins/games_completed*100:.1f}% | US {us_total_wins/games_completed*100:.1f}% | Draws {draws_total/games_completed*100:.1f}%")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-quality demonstration datasets from top neural snapshots")
    parser.add_argument("--models", nargs="+", default=None, help="List of model checkpoint paths")
    parser.add_argument("--total-games", type=int, default=5000, help="Total games to generate")
    parser.add_argument("--batch-size", type=int, default=500, help="Parallel batch size")
    parser.add_argument("--output-path", type=str, default=data_path("datasets", "warmup_regenerated.jsonl.gz"),
                        help="Output .jsonl.gz path (default: the shared data tree's datasets/). Name it "
                             "after the generating checkpoint and the engine it was built against -- a "
                             "demonstration set in the (seed, actions) format is only valid for the engine "
                             "that produced it, and silently truncates against any other.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")

    args = parser.parse_args()

    # Named outright, never guessed. A demonstration set carries the policy that generated it,
    # so which checkpoints were used is part of what the dataset is -- falling back to whichever
    # snapshot happens to be on disk would make two invocations of this command mean different
    # things.
    if not args.models:
        parser.error("--models is required: give the checkpoint path(s) to generate from.")
    # Keyed by the run-attributed label, not the file stem: snapshots of different runs share
    # file names (every run has a snapshot_final.pt), and a stem key silently kept only the last.
    top_checkpoints = {checkpoint_label(m): m for m in args.models}
    if len(top_checkpoints) != len(args.models):
        parser.error("two --models resolve to the same checkpoint label: "
                     + ", ".join(f"{m} -> {checkpoint_label(m)}" for m in args.models))
    missing = [m for m in args.models if not os.path.exists(m)]
    if missing:
        parser.error("checkpoint(s) not found: " + ", ".join(missing))

    generate_warmup_dataset(
        checkpoints=top_checkpoints,
        output_path=args.output_path,
        total_games=args.total_games,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
