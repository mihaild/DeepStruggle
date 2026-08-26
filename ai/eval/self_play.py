"""Unified self-play game simulation and .tslog.json replay generation."""

import os
import sys
import time
from typing import Optional, Union, List, Tuple, Any, Dict, cast
import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from ai.env.action_encoder import ActionEncoder
from server.replay import ReplayLogger, REPLAYS_DIR
from server.replay_types import ReplayLogDict, ReplayActionDict, GameStateDict


def generate_self_play_replay(
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    model_name: str = "ColdWarNet",
    seed: int = 2026,
    temperature: float = 0.3,
    game_id: Optional[str] = None,
    output_path: Optional[Union[str, List[str]]] = None,
    us_player_name: Optional[str] = None,
    ussr_player_name: Optional[str] = None,
    device: Union[torch.device, str] = "cuda",
    max_steps: int = 4000,
    verbose: bool = True,
) -> Tuple[ReplayLogDict, str]:
    """Simulates a complete self-play game between neural policies and saves standardized .tslog.json replay."""
    dev: torch.device = torch.device(device if torch.cuda.is_available() and str(device) == "cuda" else "cpu")

    if model is None:
        from ai.eval.player_agent import load_agent, NeuralAgent
        candidates = [
            model_path,
            "checkpoints/snapshot_20m.pt",
            "checkpoints/coldwar_net.pt",
            "checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt",
        ]
        found_path = None
        for c in candidates:
            if c and os.path.exists(c):
                found_path = c
                break

        if found_path is not None:
            agent = load_agent(found_path, device=dev)
            if isinstance(agent, NeuralAgent):
                active_model: Any = agent.model
            else:
                raise ValueError(f"Model path {found_path} did not produce a NeuralAgent")
        else:
            from ai.models.coldwar_net import create_coldwar_net
            active_model = create_coldwar_net(dev)
    else:
        active_model: Any = model.to(dev) if hasattr(model, "to") else model
        if hasattr(active_model, "eval"):
            active_model.eval()

    gid = game_id
    if not gid:
        if isinstance(output_path, str):
            gid = os.path.splitext(os.path.basename(output_path))[0].replace(".tslog", "")
        else:
            gid = f"{model_name.lower().replace(" ", "_")}_self_play"

    us_label = us_player_name or f"{model_name} [US]"
    ussr_label = ussr_player_name or f"{model_name} [USSR]"

    if verbose:
        sep = "=" * 70
        print()
        print(sep)
        print(f" Generating Self-Play Game Replay: {gid}")
        print(f" Model: {model_name} | Device: {dev} | Seed: {seed} | Temp: {temperature}")
        print(sep)
        print()

    state = ts.GameState()
    ts.Engine.init_game(state, seed)

    replay_logger = ReplayLogger(
        game_id=gid,
        seed=seed,
        us_player=us_label,
        ussr_player=ussr_label,
    )

    step_index = 0
    if verbose:
        print("Step | Turn | AR | Player | DEFCON | VP | Action Description")
        print("-" * 75)

    while not ts.Engine.is_terminal(state) and step_index < max_steps:
        step_index += 1
        p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
        player_name = "US" if p == ts.Player.US else ("USSR" if p == ts.Player.USSR else "NONE")

        obs = np.array(ts.extract_observation(state, p), copy=False).reshape(1, -1)
        mask = np.array(ActionEncoder.get_legal_mask(state), copy=False).reshape(1, -1)

        obs_t = torch.from_numpy(obs).float().to(dev)
        mask_t = torch.from_numpy(mask).to(dev)

        with torch.no_grad():
            if hasattr(active_model, "sample_action"):
                act_t, _, _, _, _ = active_model.sample_action(obs_t, mask_t, temperature=temperature, deterministic=False)
                action_idx = int(act_t.item())
            else:
                logits, _ = active_model(obs_t, mask_t)
                action_idx = int(torch.argmax(logits, dim=-1).item())

        action_desc = ActionEncoder.get_action_name(state, action_idx)
        ma = ts.ActionMask.decode_flat_action(state, action_idx)

        action_dict: ReplayActionDict = {
            "flat_action_idx": action_idx,
            "decision_type": int(ma.decision_type),
            "primary_id": int(ma.primary_id),
            "secondary_id": int(ma.secondary_id),
            "flags": int(ma.flags),
            "card_id": int(ma.primary_id) if int(ma.decision_type) in (1, 2, 3, 4) else None,
            "target_id": int(ma.secondary_id) if int(ma.decision_type) in (5, 6, 7) else None,
        }

        turn_before = int(state.turn)
        ar_before = int(state.action_round)
        phase_before = str(state.current_phase).replace("Phase.", "")

        ok = ts.Engine.step_flat(state, action_idx)
        state_after_dict: GameStateDict = cast(GameStateDict, ts.state_to_dict(state))

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

        if verbose and (step_index % 10 == 0 or state.current_phase == ts.Phase.GAME_OVER or "Scoring" in action_desc):
            print(f"{step_index:4d} | {turn_before:4d} | {ar_before:2d} | {player_name:>5s} | {state.defcon:6d} | {state.victory_points:+4d} | {action_desc}")

        if not ok:
            break

    term_util = ts.Engine.get_terminal_utility(state)
    if term_util > 0:
        winner = "US"
        margin = int(state.victory_points)
        reason = "Victory Point Threshold (+20 VP) or DEFCON Inversion" if state.defcon > 1 else "USSR triggered DEFCON 1"
    elif term_util < 0:
        winner = "USSR"
        margin = int(state.victory_points)
        reason = "Victory Point Threshold (-20 VP) or DEFCON Inversion" if state.defcon > 1 else "US triggered DEFCON 1"
    else:
        winner = "DRAW"
        margin = int(state.victory_points)
        reason = "Final Scoring Draw or Maximum Steps Limit"

    replay_logger.set_result(
        winner=winner,
        margin=margin,
        end_turn=int(state.turn),
        reason=reason,
    )

    paths_to_save: List[str] = []
    if output_path is None:
        paths_to_save = [os.path.join(REPLAYS_DIR, f"{gid}.tslog.json")]
    elif isinstance(output_path, str):
        paths_to_save = [output_path]
    elif isinstance(output_path, (list, tuple)):
        paths_to_save = list(output_path)

    saved_primary_path = paths_to_save[0]
    for p_path in paths_to_save:
        os.makedirs(os.path.dirname(os.path.abspath(p_path)), exist_ok=True)
        replay_logger.save(p_path)

    if verbose:
        print("-" * 75)
        print(f" Game Ended on Turn {state.turn} (Step {step_index})")
        print(f" Winner: {winner} (VP: {state.victory_points:+d}, DEFCON: {state.defcon})")
        print(f" Reason: {reason}")
        print(f" Replay saved to: {saved_primary_path}")
        print()

    return replay_logger.to_dict(), saved_primary_path


generate_selfplay_replay = generate_self_play_replay
