#!/usr/bin/env python3
"""
Unified Match Runner & Replay Generator for Twilight Struggle AI.

Plays a match between any pair of agents (supporting distinct checkpoints,
heuristic baselines, strategic agents, or interactive human terminal play)
and records standardized .tslog.json replays for the Web Workbench.
"""

import argparse
import base64
import os
import sys
import time
import random
from typing import Optional, Dict, Any, Tuple, cast
from web.server.replay_types import ReplayLogDict

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
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

from web.server.replay import ReplayLogger, ReplayManager, replays_dir
from web.server.replay_types import ReplayActionDict, GameStateDict
from tools.lib.tournament_evaluator import classify_game_ending_reason
from tools.lib.scoring_formatter import format_regional_scoring_breakdown
from tools.lib.checkpoint_utils import discover_checkpoints
from tools.lib.openings import OPENINGS, acting_side, scripted_setup_index


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

    # search:<checkpoint>[:sims[:determinize]] -- MCTS over a checkpoint, played through the
    # standard match loop so the replay goes through the same writer as every other game.
    if clean_spec.startswith("search:"):
        parts = clean_spec.split(":")
        path = parts[1]
        sims = int(parts[2]) if len(parts) > 2 and parts[2] else 64
        determinize = len(parts) > 3 and parts[3].lower().startswith("determin")
        from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
        from bindings.action_encoder import ActionEncoder
        from tools.lib.player_agent import NeuralAgent

        base = NeuralAgent.from_checkpoint(path, device=device)
        # advance_root=False: this match loop does NOT settle the state before asking a bot, so
        # the tree must root exactly where the loop is. With it True the searcher answered about a
        # later decision and returned actions the engine rejected -- 1,355 times in one game.
        cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                                advance_root=False, determinize=determinize)
        searcher = BatchedMCTS(base.model, device=device, config=cfg, featurise_capacity=1)

        class _SearchBot(BaseBot):
            """Bot-shaped adapter that needs the engine state, not the JSON view.

            A searcher clones and steps the real GameState, so BaseBot's dict interface cannot
            serve it. `wants_game_state` tells the match loop to pass the state instead, and
            `select_action` raises rather than silently playing an unsearched move.
            """

            wants_game_state = True

            def __init__(self, role):
                super().__init__(role, name=f"search{sims}{'-det' if determinize else ''}")

            def select_action(self, state, legal_actions):
                raise NotImplementedError(
                    "search bots need the engine GameState; the match loop must route through "
                    "select_from_state (see wants_game_state)")

            def select_from_state(self, state) -> Dict[str, Any]:
                flat = int(searcher.best_actions([state])[0])
                legal = np.asarray(ActionEncoder.get_legal_mask(state))
                if not legal[flat]:
                    raise RuntimeError(
                        f"search returned flat action {flat}, illegal at decision_type="
                        f"{int(state.ctx().decision_type)}; the engine would reject it and the "
                        f"loop would re-offer the same node forever")
                ma = ts.decode_flat_action(state, flat)
                return {
                    "decision_type": int(ma.decision_type),
                    "primary_id": int(ma.primary_id),
                    "secondary_id": int(ma.secondary_id),
                    "flags": int(ma.flags),
                }

            def reset(self):
                searcher.reset()

        label = f"Search({sims}{'/det' if determinize else ''}, {os.path.basename(path)})"
        return _SearchBot(role), label

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



def forced_setup_action(state: "ts.GameState", opening: str,
                        cursor: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """The next scripted placement as a bot-shaped action dict, or None to let the agent decide.

    The flat index comes from the shared opening registry and is turned into a MicroAction by the
    engine's own decoder, rather than assembled field by field here.
    """
    idx = scripted_setup_index(state, acting_side(state), opening, cursor)
    if idx is None:
        return None
    micro = ts.decode_flat_action(state, idx)
    return {
        "decision_type": int(micro.decision_type),
        "primary_id": int(micro.primary_id),
        "secondary_id": int(micro.secondary_id),
        "flags": int(micro.flags),
    }


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
    opening: Optional[str] = None,
) -> Tuple[ReplayLogDict, str]:
    """Runs a full Twilight Struggle game between two specified agents and saves standardized replay."""
    # The id and the filename must agree. With --output the file used the caller's name while the
    # id stayed a timestamp, so a replay called s240_1.tslog.json identified itself as
    # match_1789505117 -- a number appearing nowhere else, which made tracing a reported problem
    # back to its file a matter of opening every one. An explicit --game-id still wins; otherwise
    # the id is taken from the output filename, which callers already name descriptively.
    if game_id is None and output_path is not None:
        base = os.path.basename(output_path)
        for suffix in (".tslog.json", ".json"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        game_id = base or f"match_{int(time.time())}"
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

    setup_cursor: Dict[str, int] = {"US": 0, "USSR": 0}

    step_count = 0
    while not ts.Engine.is_terminal(state) and step_count < max_steps:
        p_enum = state.ctx().decision_player
        p_str = "US" if p_enum == ts.Player.US else "USSR"
        active_bot = bot_us if p_str == "US" else bot_ussr

        d = state.to_dict()
        legal = d.get("legal_actions", {})
        # The engine's own observation, exactly as web/server/session.py hands it to a bot over
        # the wire. A neural bot needs it and has no other source: the Python reconstruction it
        # used to fall back on rebuilt the observation field by field, drifted from the engine,
        # and produced replays that did not reproduce the games training plays. It is gone, so
        # this is where the observation comes from.
        d["observation_b64"] = base64.b64encode(
            np.asarray(ts.extract_observation(state, p_enum), dtype=np.float32).tobytes()
        ).decode("ascii")

        action_dict = (forced_setup_action(state, opening, setup_cursor)
                       if opening else None)
        if action_dict is None:
            # A searcher needs the engine state to clone and step; every other bot reads the
            # JSON view. The state is already in scope here, so no plumbing is required.
            if getattr(active_bot, "wants_game_state", False):
                action_dict = getattr(active_bot, "select_from_state")(state)
            else:
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

    # Determine save path. Through the server's own resolver, never a second guess at it: this
    # used to prefer data/replays "if it exists" and fall back to replays/, which in the main
    # checkout are the same directory -- replays/ is a symlink -- so the two could not disagree.
    # In a fresh copy without that symlink they do, and a match wrote its replay where the
    # viewer does not look. One resolver, and it honours TS_REPLAYS_DIR.
    if output_path is None:
        target_dir = replays_dir()
        os.makedirs(target_dir, exist_ok=True)
        final_output_path = os.path.join(target_dir, f"{game_id}.tslog.json")
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
    parser.add_argument("--opening", type=str, default=None, choices=sorted(OPENINGS),
                        help="Force a named setup on both sides instead of letting the agents place (e.g. 'human')")

    args = parser.parse_args()

    us_agent = args.agent if args.agent is not None else args.us
    ussr_agent = args.agent if args.agent is not None else args.ussr
    match_seed = args.seed if args.seed is not None else random.randint(1, 1000000)

    # Neural self-play goes through the canonical generator, which resolves chance nodes exactly
    # as the training env does and writes the replay through the one schema. The mixed path below
    # (a network against a rule-based bot) drives NeuralBot instead, and hands it the engine's own
    # observation rather than a reconstruction -- see the loop.
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
            opening=args.opening,
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
        opening=args.opening,
    )


if __name__ == "__main__":
    main()
