"""Unified self-play game simulation and .tslog.json replay generation."""

import os
import sys
import time
from typing import Optional, Union, List, Tuple, Any, Dict, cast
import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net import ColdWarNet, create_coldwar_net
from bindings.action_encoder import ActionEncoder
from ai.eval.blunders import BlunderCounts, check_play
from web.server.replay import ReplayLogger, replays_dir
from web.server.replay_types import ReplayLogDict, ReplayActionDict, GameStateDict
from tools.lib.tournament_evaluator import classify_game_ending_reason


def generate_self_play_replay(
    model: Union[torch.nn.Module, str, Any],
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

    active_model: Any
    if isinstance(model, str):
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model checkpoint path not found: {model}")
        from tools.lib.player_agent import load_agent, NeuralAgent
        agent = load_agent(model, device=dev)
        if isinstance(agent, NeuralAgent):
            active_model = agent.model
        else:
            raise ValueError(f"Model path {model} did not produce a NeuralAgent")
    elif isinstance(model, torch.nn.Module):
        active_model = model.to(dev)
    elif hasattr(model, "model"):
        active_model = model.model.to(dev) if hasattr(model.model, "to") else model.model
    else:
        active_model = model

    if hasattr(active_model, "eval"):
        active_model.eval()

    # The observation layout the model was trained for, read off the model. Left to the default
    # this extracted the 4,293-wide legacy block for every model, and a network reads fixed slices
    # -- so a v2.1 or v2.2 policy was silently handed the wrong regions and played accordingly,
    # without raising. Every self-play replay generated for a non-legacy checkpoint before this
    # was produced by a model reading scrambled input.
    _obs_width = int(getattr(active_model, "TOTAL_OBS_SIZE", ts.OBS_SIZE_LEGACY))
    obs_layout = {int(ts.OBS_SIZE_LEGACY): "legacy",
                  int(ts.OBS_SIZE_V21): "v2.1",
                  int(ts.OBS_SIZE_V22): "v2.2"}.get(_obs_width)
    if obs_layout is None:
        raise ValueError(
            f"model expects an observation of width {_obs_width}, which matches no known layout "
            f"({ts.OBS_SIZE_LEGACY} legacy, {ts.OBS_SIZE_V21} v2.1, {ts.OBS_SIZE_V22} v2.2)")

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
    blunders = BlunderCounts()
    last_card: dict[str, int] = {}
    if verbose:
        print("Step | Turn | AR | Player | DEFCON | VP | Action Description")
        print("-" * 75)

    while not ts.Engine.is_terminal(state) and step_index < max_steps:
        step_index += 1
        p = state.ctx().decision_player if state.ctx().decision_player != ts.Player.NONE else state.phasing_player
        player_name = "US" if p == ts.Player.US else ("USSR" if p == ts.Player.USSR else "NONE")

        obs = np.array(ts.extract_observation(state, p, layout=obs_layout),
                       copy=False).reshape(1, -1)
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

        # Blunder check. SELECT_PLAY_MODE is where both halves are known -- which card and what it
        # is being spent on -- and the card is still in its owner's hand there, which the rules
        # read. Missile Envy forces a card on its recipient, so that play is not their error.
        _dt = int(ma.decision_type)
        if _dt == 1:
            last_card[player_name] = int(ma.primary_id)
        elif _dt == 2:
            _mode = {0: "EVENT", 1: "OPS", 2: "SPACE"}.get(int(ma.primary_id))
            _card = last_card.get(player_name, 0)
            if _mode and 1 <= _card <= 110:
                _forced = (int(getattr(state, "forced_card_id", 0)) == _card
                           and getattr(state, "forced_card_player", None) == p)
                for _b in check_play(state, p, _card, _mode, forced=_forced, counts=blunders):
                    if verbose:
                        print(f"     !! BLUNDER {_b}")

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

        # Resolve any chance nodes the action lands on, exactly as the vectorized runner
        # does in VectorizedBatchRunner::step_flat_all. Asking the agent to choose at a
        # ROLL_DIE node instead is not equivalent: for Summit (#45) it leaves the engine
        # with a different phasing_player, which changes who acts next and who loses a
        # DEFCON-1 ending, and games collapse to a fraction of their true length.
        while (not ts.Engine.is_terminal(state)
               and state.ctx().decision_player == ts.Player.NONE
               and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

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
    winner = "US" if term_util > 0 else ("USSR" if term_util < 0 else "DRAW")
    margin = int(state.victory_points)
    reason = classify_game_ending_reason(state)

    replay_logger.set_result(
        winner=winner,
        margin=margin,
        end_turn=int(state.turn),
        reason=reason,
    )

    paths_to_save: List[str] = []
    if output_path is None:
        paths_to_save = [os.path.join(replays_dir(), f"{gid}.tslog.json")]
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
        if blunders.opportunities:
            print(" Blunders (committed / chances):")
            print(blunders.summary())
        print(f" Winner: {winner} (VP: {state.victory_points:+d}, DEFCON: {state.defcon})")
        print(f" Reason: {reason}")
        print(f" Replay saved to: {saved_primary_path}")
        print()

    return replay_logger.to_dict(), saved_primary_path


generate_selfplay_replay = generate_self_play_replay
