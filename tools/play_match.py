#!/usr/bin/env python3
"""
Unified Match Runner & Replay Generator for Twilight Struggle AI.

Plays a match between any pair of agents (supporting distinct checkpoints,
heuristic baselines, strategic agents, or interactive human terminal play)
and records standardized .tslog.json replays for the Web Workbench.
"""

import argparse
import os
import sys
import time
import random
from typing import Optional, Dict, Any, Tuple, cast
from web.server.replay_types import ReplayLogDict

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts
from bot import (
    BaseBot,
    RandomBot,
    HeuristicBot,
    ExploratoryBot,
    StrategicBot,
    EventHeavyBot,
    HumanBot,
)
try:
    from bot import NeuralBot
except ImportError:
    NeuralBot = None

from web.server.replay import ReplayLogger, ReplayManager, REPLAYS_DIR
from web.server.replay_types import ReplayActionDict, GameStateDict
from tools.lib.tournament_evaluator import classify_game_ending_reason
from tools.lib.scoring_formatter import format_regional_scoring_breakdown
from tools.lib.checkpoint_utils import discover_checkpoints


def resolve_agent(agent_spec: str, role: str, temperature: float = 0.1, device: str = "cpu") -> Tuple[BaseBot, str]:
    """Resolves an agent specification into a BaseBot instance and human-readable name.

    Specification can be:
    - 'heuristic' -> HeuristicBot
    - 'random' -> RandomBot
    - 'strategic' -> StrategicBot
    - 'event_heavy' -> EventHeavyBot
    - 'exploratory' -> ExploratoryBot
    - 'human' -> HumanBot (interactive CLI terminal player)
    - 'neural' -> NeuralBot (latest checkpoint)
    - Path to a .pt file or 'neural:<path>' -> NeuralBot loaded with that checkpoint
    """
    clean_spec = agent_spec.strip()

    if clean_spec == "human":
        return HumanBot(role), f"Human ({role})"

    if clean_spec == "heuristic":
        return HeuristicBot(role), "HeuristicBot"

    if clean_spec == "random":
        return RandomBot(role), "RandomBot"

    if clean_spec == "strategic":
        return StrategicBot(role), "StrategicBot"

    if clean_spec == "event_heavy":
        return EventHeavyBot(role), "EventHeavyBot"

    if clean_spec == "exploratory":
        return ExploratoryBot(role), "ExploratoryBot"

    # Neural bot resolution
    model_path: Optional[str] = None
    if clean_spec.startswith("neural:"):
        model_path = clean_spec.split(":", 1)[1]
    elif clean_spec.endswith(".pt") or os.path.exists(clean_spec):
        model_path = clean_spec
    elif clean_spec == "neural":
        # Discover latest checkpoint
        chkpt_dir = os.path.join(_root, "data", "checkpoints") if os.path.exists(os.path.join(_root, "data", "checkpoints")) else os.path.join(_root, "checkpoints")
        discovered = discover_checkpoints(chkpt_dir)
        if discovered:
            discovered.sort(key=lambda x: x["mtime"], reverse=True)
            model_path = discovered[0]["path"]

    if model_path is not None:
        if NeuralBot is None:
            raise ImportError("NeuralBot is not available. Please ensure PyTorch is installed.")
        bot_name = f"Neural ({os.path.basename(model_path)})"
        return NeuralBot(role, model_path=model_path, temperature=temperature, device=device), bot_name

    raise ValueError(
        f"Unknown agent specification '{agent_spec}'. "
        f"Choose from: heuristic, random, strategic, event_heavy, exploratory, human, or a .pt checkpoint path."
    )


def format_action_description(action_dict: Dict[str, Any], state_dict: Dict[str, Any]) -> str:
    """Formats a concise human-readable description for an action."""
    d_type = action_dict.get("decision_type", 0)
    p_id = action_dict.get("primary_id", 0)
    flags = action_dict.get("flags", 0)

    if d_type == 1:  # SELECT_CARD
        c_name = ts.CardData.get_card_name(p_id) if 1 <= p_id <= 110 else f"Card #{p_id}"
        return f"Plays card #{p_id} '{c_name}'"
    elif d_type == 2:  # SELECT_PLAY_MODE
        modes = {0: "Event", 1: "Operations", 2: "Space Race"}
        return f"Plays as {modes.get(p_id, f'Mode {p_id}')}"
    elif d_type == 3:  # CHOOSE_TIMING_BRANCH
        branches = {0: "Operations First", 1: "Event First"}
        return f"Timing: {branches.get(p_id, f'Branch {p_id}')}"
    elif d_type == 4:  # SELECT_OP_MODE
        op_modes = {0: "Influence", 1: "Coup", 2: "Realignment"}
        return f"Op Mode: {op_modes.get(p_id, f'Mode {p_id}')}"
    elif d_type == 5:  # POINT_NODE
        if flags == 128 or p_id == 255:
            return "Pass / Confirm Done"
        if 0 <= p_id < 84:
            c_name = ts.MapData.get_country_name(p_id)
            return f"Targets {c_name} (ID {p_id})"
        return f"Node #{p_id}"
    return f"Action (type={d_type}, id={p_id})"


def run_match(
    agent_us_spec: str,
    agent_ussr_spec: str,
    seed: int = 2026,
    game_id: Optional[str] = None,
    output_path: Optional[str] = None,
    temperature: float = 0.1,
    commentary: bool = False,
    device: str = "cpu",
    max_steps: int = 4000,
    verbose: bool = True,
) -> Tuple[ReplayLogDict, str]:
    """Runs a full Twilight Struggle game between two specified agents and saves standardized replay."""
    if game_id is None:
        game_id = f"match_{int(time.time())}"

    bot_us, name_us = resolve_agent(agent_us_spec, "US", temperature=temperature, device=device)
    bot_ussr, name_ussr = resolve_agent(agent_ussr_spec, "USSR", temperature=temperature, device=device)

    state = ts.GameState()
    ts.Engine.init_game(state, seed)

    logger = ReplayLogger(game_id=game_id, seed=seed, us_player=name_us, ussr_player=name_ussr)

    if verbose:
        print("=" * 80)
        print(f"★ TWILIGHT STRUGGLE MATCH: {name_us} (US) vs {name_ussr} (USSR)")
        print(f"★ Game ID: {game_id} | Seed: {seed} | Commentary: {commentary}")
        print("=" * 80)

    step_count = 0
    while not ts.Engine.is_terminal(state) and step_count < max_steps:
        p_enum = state.ctx().decision_player
        p_str = "US" if p_enum == ts.Player.US else "USSR"
        active_bot = bot_us if p_str == "US" else bot_ussr

        d = state.to_dict()
        legal = d.get("legal_actions", {})

        action_dict = active_bot.select_action(d, legal)
        if action_dict is None:
            if verbose:
                print(f"[{p_str}] Forfeited or passed.")
            break

        action = ts.MicroAction(
            ts.DecisionType(int(action_dict["decision_type"])),
            int(action_dict["primary_id"]),
            int(action_dict["secondary_id"]),
            int(action_dict["flags"]),
        )

        # Extract commentary and scoring breakdowns if requested
        strat_text = getattr(active_bot, "last_strategy", "")
        comm_text = getattr(active_bot, "last_commentary", "")
        desc = format_action_description(action_dict, d)

        # If a scoring card was selected, generate regional scoring breakdown
        if commentary and action_dict.get("decision_type") == 1:
            cid = action_dict.get("primary_id", 0)
            if 1 <= cid <= 110 and ts.CardData.get_card_info(cid).get("is_scoring", False):
                breakdown = format_regional_scoring_breakdown(d, cid)
                if breakdown:
                    comm_text = f"{comm_text}\n{breakdown}" if comm_text else breakdown

        replay_action: ReplayActionDict = {
            "decision_type": ts.DecisionType(int(action_dict["decision_type"])),
            "primary_id": int(action_dict["primary_id"]),
            "secondary_id": int(action_dict["secondary_id"]),
            "flags": int(action_dict["flags"]),
        }

        logger.log_step(
            step_index=step_count,
            turn=state.turn,
            ar=state.action_round,
            phase=d.get("current_phase_name", "ACTION"),
            player=p_str,
            action=replay_action,
            description=desc,
            state_snapshot=cast(GameStateDict, d),
        )

        if verbose and (commentary or isinstance(active_bot, HumanBot)):
            print(f"[Step {step_count:3d} | Turn {state.turn} AR {state.action_round} | {p_str}]: {desc}")
            if commentary and strat_text:
                print(f"    • Rationale: {strat_text}")
            if commentary and comm_text:
                print(f"    • Commentary: {comm_text}")

        success = ts.Engine.step(state, action)
        if not success:
            if verbose:
                print(f"❌ Engine rejected legal action: {action_dict}")
            break

        step_count += 1

    # Classify outcome
    reason = classify_game_ending_reason(state)
    vp = state.victory_points
    winner = "US" if vp > 0 else ("USSR" if vp < 0 else "DRAW")
    if "US Win" in reason or "Europe Control (US)" in reason:
        winner = "US"
    elif "USSR Win" in reason or "Europe Control (USSR)" in reason:
        winner = "USSR"

    logger.set_result(winner=winner, margin=vp, end_turn=state.turn, reason=reason)
    replay_data = logger.to_dict()

    # Determine save path
    if output_path is None:
        replays_dir = os.path.join(_root, "data", "replays") if os.path.exists(os.path.join(_root, "data", "replays")) else os.path.join(_root, "replays")
        os.makedirs(replays_dir, exist_ok=True)
        final_output_path = os.path.join(replays_dir, f"{game_id}.tslog.json")
    else:
        final_output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)

    logger.save(final_output_path)

    if verbose:
        print("\n" + "=" * 80)
        print("★ MATCH COMPLETED")
        print(f"★ Winner: {winner} | Final VP: {vp:+d} | End Turn: {state.turn} | Steps: {step_count}")
        print(f"★ Ending Reason: {reason}")
        print(f"★ Replay File: {final_output_path}")
        replay_filename = os.path.basename(final_output_path)
        print(f"★ Web Workbench URL: http://localhost:8000/?replay={replay_filename}")
        print("=" * 80 + "\n")

    return replay_data, final_output_path


def main():
    parser = argparse.ArgumentParser(description="Twilight Struggle Match Runner & Replay Generator")
    parser.add_argument("--us", type=str, default="heuristic", help="US agent (bot name, checkpoint path, or 'human')")
    parser.add_argument("--ussr", type=str, default="random", help="USSR agent (bot name, checkpoint path, or 'human')")
    parser.add_argument("--agent", type=str, default=None, help="Self-play agent (sets both --us and --ussr)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to randomized)")
    parser.add_argument("--game-id", type=str, default=None, help="Custom game session ID")
    parser.add_argument("--output", type=str, default=None, help="Custom output .tslog.json path")
    parser.add_argument("--temperature", type=float, default=0.1, help="Neural sampling temperature")
    parser.add_argument("--commentary", action="store_true", help="Embed strategic commentary and scoring audits")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch inference device (cpu or cuda)")

    args = parser.parse_args()

    us_agent = args.agent if args.agent is not None else args.us
    ussr_agent = args.agent if args.agent is not None else args.ussr
    match_seed = args.seed if args.seed is not None else random.randint(1, 1000000)

    # Neural self-play goes through the canonical generator, which reads observations from
    # Observation::extract and resolves chance nodes exactly as the training env does. The
    # bot loop below reaches the network through NeuralBot, which rebuilds the 4293-dim
    # observation in Python; that duplicate has drifted from the engine, and replays made
    # with it do not reproduce the games training actually plays.
    if args.agent is not None and os.path.exists(args.agent):
        from tools.lib.self_play import generate_self_play_replay
        generate_self_play_replay(
            model=args.agent,
            seed=match_seed,
            temperature=args.temperature,
            game_id=args.game_id,
            output_path=args.output,
            device=args.device,
            verbose=True,
        )
        return

    run_match(
        agent_us_spec=us_agent,
        agent_ussr_spec=ussr_agent,
        seed=match_seed,
        game_id=args.game_id,
        output_path=args.output,
        temperature=args.temperature,
        commentary=args.commentary,
        device=args.device,
    )


if __name__ == "__main__":
    main()
