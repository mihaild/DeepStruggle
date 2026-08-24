"""Arena Evaluator: Tournament benchmarking for Twilight Struggle AI Agents."""

import time
from typing import Dict, List, Optional, Any
import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder
from ai.training.behavioral_cloning import HeuristicPolicy


class ArenaEvaluator:
    """Runs automated tournament matches between agents and collects statistical metrics."""

    def __init__(self, model: Optional[ColdWarNet] = None, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device) if model is not None else None
        if self.model is not None:
            self.model.eval()

    def play_single_game(
        self,
        us_agent_type: str = "neural",
        ussr_agent_type: str = "heuristic",
        seed: Optional[int] = None,
        max_steps: int = 5000,
    ) -> Dict[str, Any]:
        """Plays a single game between two specified agents."""
        state = ts.GameState()
        s = seed if seed is not None else int(np.random.randint(1, 1_000_000_000))
        ts.Engine.init_game(state, s)

        step_count = 0
        t0 = time.time()

        while not ts.Engine.is_terminal(state) and step_count < max_steps:
            p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
            agent_type = us_agent_type if p == ts.Player.US else ussr_agent_type

            if agent_type == "neural" and self.model is not None:
                obs = ts.extract_observation(state, p)
                mask = ActionEncoder.get_legal_mask(state)
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    actions_t, _, _, _, _ = self.model.sample_action(obs_t, mask_t, temperature=0.3, deterministic=False)
                action_idx = int(actions_t.item())
            elif agent_type == "heuristic":
                action_idx = HeuristicPolicy.select_action(state)
            else: # "random"
                mask = ActionEncoder.get_legal_mask(state)
                legal_indices = [int(i) for i in np.where(mask > 0)[0]]
                action_idx = int(np.random.choice(legal_indices)) if legal_indices else ActionEncoder.CONFIRM_DONE_INDEX

            ts.Engine.step_flat(state, action_idx)
            step_count += 1

        elapsed = time.time() - t0
        term_util = ts.Engine.get_terminal_utility(state)
        final_vp = int(state.victory_points)

        winner = "US" if term_util > 0 else ("USSR" if term_util < 0 else "DRAW")
        return {
            "winner": winner,
            "terminal_utility": term_util,
            "victory_points": final_vp,
            "steps": step_count,
            "turn": int(state.turn),
            "defcon": int(state.defcon),
            "elapsed_seconds": elapsed,
            "us_agent": us_agent_type,
            "ussr_agent": ussr_agent_type,
        }

    def run_tournament(
        self,
        opponent_type: str = "heuristic",
        num_games: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Runs a tournament playing 50% games as US and 50% as USSR."""
        results = []
        neural_wins = 0
        opponent_wins = 0
        draws = 0
        vp_margins = []

        half = num_games // 2
        matchups = [("neural", opponent_type)] * half + [(opponent_type, "neural")] * (num_games - half)

        if verbose:
            print(f"\n=======================================================")
            print(f"Running Tournament: NeuralBot vs {opponent_type.capitalize()}Bot ({num_games} games)")
            print(f"=======================================================\n")

        for g_idx, (us_type, ussr_type) in enumerate(matchups):
            res = self.play_single_game(us_agent_type=us_type, ussr_agent_type=ussr_type)
            results.append(res)

            neural_role = "US" if us_type == "neural" else "USSR"
            neural_won = res["winner"] == neural_role
            opp_won = res["winner"] != neural_role and res["winner"] != "DRAW"

            if neural_won:
                neural_wins += 1
            elif opp_won:
                opponent_wins += 1
            else:
                draws += 1

            vp_for_neural = res["victory_points"] if neural_role == "US" else -res["victory_points"]
            vp_margins.append(vp_for_neural)

            if verbose and (g_idx + 1) % max(1, num_games // 10) == 0:
                print(
                    f"Game {g_idx+1:3d}/{num_games:3d} | Neural as {neural_role:4s} -> Winner: {res['winner']:4s} | "
                    f"VP: {res['victory_points']:+2d} | Turn: {res['turn']:2d} | Defcon: {res['defcon']} | "
                    f"Current Score: {neural_wins}W - {opponent_wins}L - {draws}D"
                )

        win_rate = neural_wins / max(num_games, 1)
        summary = {
            "total_games": num_games,
            "neural_wins": neural_wins,
            "opponent_wins": opponent_wins,
            "draws": draws,
            "win_rate": win_rate,
            "avg_vp_margin": float(np.mean(vp_margins)),
            "avg_steps": float(np.mean([r["steps"] for r in results])),
            "avg_turn": float(np.mean([r["turn"] for r in results])),
            "opponent_type": opponent_type,
        }

        if verbose:
            print(f"\n--- Tournament Results Summary ---")
            print(f"Total Games: {num_games}")
            print(f"NeuralBot Win Rate: {win_rate*100:.1f}% ({neural_wins} Wins, {opponent_wins} Losses, {draws} Draws)")
            print(f"Average VP Margin for NeuralBot: {summary['avg_vp_margin']:+.2f}")
            print(f"Average Game Length: {summary['avg_steps']:.1f} steps (Turn {summary['avg_turn']:.1f})")
            print(f"-----------------------------------\n")

        return summary
