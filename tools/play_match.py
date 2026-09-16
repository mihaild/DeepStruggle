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
from web.server.replay_types import (
    ReplayActionDict,
    GameStateDict,
    ReplayCriticDict,
    ReplayPolicyDict,
    ReplayTraceMetaDict,
)
from ai.eval.policy_readout import (
    TRACE_P_FLOOR,
    TRACE_TOP_K,
    name_actions,
    read_critic,
    unasked_readout,
)
from tools.lib.engine_fingerprint import fingerprint as engine_fingerprint
from tools.lib.tournament_evaluator import classify_game_ending_reason
from tools.lib.scoring_formatter import format_regional_scoring_breakdown
from tools.lib.checkpoint_utils import discover_checkpoints
from tools.lib.game_loop import GameLoop, SettlePolicy, StepRecord
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


class _BotSource:
    """Adapts a bot to the loop's ActionSource protocol.

    Three kinds of bot reach here and they want different things. A searcher needs the engine
    state (`wants_game_state`); every other bot reads the JSON view plus the legal-action block;
    and a scripted opening overrides both while it is still placing influence. Keeping that here
    means the loop itself never learns about bots.
    """

    def __init__(self, bot: BaseBot, role: str, opening: Optional[str],
                 cursor: Dict[str, int], commentary: bool) -> None:
        self.bot = bot
        self.role = role
        self.opening = opening
        self.cursor = cursor
        self.commentary = commentary
        self.last_view: Dict[str, Any] = {}
        self.last_description = ""

    def choose(self, state: "ts.GameState") -> Optional["ts.MicroAction"]:
        action_dict = (forced_setup_action(state, self.opening, self.cursor)
                       if self.opening else None)

        d = cast(Dict[str, Any], state.to_dict())
        # The engine's own observation, exactly as web/server/session.py hands it to a bot over
        # the wire. A neural bot needs it and has no other source.
        p_enum = state.ctx().decision_player
        d["observation_b64"] = base64.b64encode(
            np.asarray(ts.extract_observation(state, p_enum), dtype=np.float32).tobytes()
        ).decode("ascii")
        self.last_view = d

        if action_dict is None:
            if getattr(self.bot, "wants_game_state", False):
                action_dict = getattr(self.bot, "select_from_state")(state)
            else:
                action_dict = self.bot.select_action(d, d.get("legal_actions", {}))
        if action_dict is None:
            return None

        # The bot works from the JSON view and cannot name a flat action; here the engine state
        # is still in hand, so the names are filled in at the only point that has both.
        readout = getattr(self.bot, "last_readout", None)
        if readout is not None:
            name_actions(readout, state)

        self.last_description = format_action_description(action_dict, d)
        return ts.MicroAction(
            ts.DecisionType(int(action_dict["decision_type"])),
            int(action_dict["primary_id"]),
            int(action_dict["secondary_id"]),
            int(action_dict["flags"]),
        )


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
    trace: bool = False,
    trace_top_k: int = TRACE_TOP_K,
    trace_critic_every: str = "step",
) -> Tuple[ReplayLogDict, str]:
    """Runs a full Twilight Struggle game between two specified agents and saves standardized replay.

    With `trace`, a neural side records what it believed at each node it chose, and one of the
    neural models answers the critic for every step -- see `ai/eval/policy_readout.py`. Off by
    default here, unlike self-play: most matches in this CLI are bot-vs-bot baselines where
    there is no distribution to record and the extra forwards would be pure cost.
    """
    if trace_critic_every not in ("step", "decision", "off"):
        raise ValueError(
            f"trace_critic_every must be 'step', 'decision' or 'off'; got {trace_critic_every!r}")
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

    if trace:
        for _b in (bot_us, bot_ussr):
            if hasattr(_b, "trace"):
                setattr(_b, "trace", True)
    # ONE critic for the whole game, even when both sides are neural. A ribbon that switched
    # networks halfway would plot two different opinions on one axis and read as a change of
    # mind, so the model that supplies it is fixed here and named in the trace metadata.
    critic_model: Optional[Any] = None
    critic_model_name = ""
    if trace:
        for _b, _n in ((bot_us, name_us), (bot_ussr, name_ussr)):
            if getattr(_b, "model", None) is not None:
                critic_model, critic_model_name = getattr(_b, "model"), _n
                break

    state = ts.GameState()
    ts.Engine.init_game(state, seed)

    logger = ReplayLogger(game_id=game_id, seed=seed, us_player=name_us, ussr_player=name_ussr)

    if trace:
        trace_meta: ReplayTraceMetaDict = {
            "mode": "inline",
            "model_us": name_us,
            "model_ussr": name_ussr,
            "temperature": float(temperature),
            "top_k": int(trace_top_k),
            "p_floor": float(TRACE_P_FLOOR),
            "engine_fingerprint": engine_fingerprint(),
        }
        if critic_model is not None:
            trace_meta["arch"] = type(critic_model).__name__
            trace_meta["critic_model"] = critic_model_name
        logger.set_trace_meta(trace_meta)

    if verbose:
        print("=" * 80)
        print(f"★ TWILIGHT STRUGGLE MATCH: {name_us} (US) vs {name_ussr} (USSR)")
        print(f"★ Game ID: {game_id} | Seed: {seed} | Commentary: {commentary}")
        print("=" * 80)

    setup_cursor: Dict[str, int] = {"US": 0, "USSR": 0}

    sources: Dict[Any, Any] = {
        ts.Player.US: _BotSource(bot_us, "US", opening, setup_cursor, commentary),
        ts.Player.USSR: _BotSource(bot_ussr, "USSR", opening, setup_cursor, commentary),
    }

    def _record(rec: StepRecord, after: "ts.GameState") -> None:
        """Called only after the engine ACCEPTED the action -- refused ones never reach a replay."""
        src = sources[ts.Player.US if rec.player == "US" else ts.Player.USSR]
        view = cast(Dict[str, Any], rec.state_before or {})
        dt = int(rec.action.decision_type)
        replay_action: ReplayActionDict = {
            # Without flat_action_idx the fidelity test SKIPS the replay instead of checking it,
            # so omitting it would retire the check rather than fail it.
            "flat_action_idx": int(rec.flat) if rec.flat is not None else -1,
            "decision_type": dt,
            "primary_id": int(rec.action.primary_id),
            "secondary_id": int(rec.action.secondary_id),
            "flags": int(rec.action.flags),
            "card_id": int(rec.action.primary_id) if dt in (1, 2, 3, 4) else None,
            "target_id": int(rec.action.secondary_id) if dt in (5, 6, 7) else None,
        }
        action_dict = {
            "decision_type": dt,
            "primary_id": int(rec.action.primary_id),
            "secondary_id": int(rec.action.secondary_id),
            "flags": int(rec.action.flags),
        }
        # A forced action was played by the loop, not chosen by a bot, so it has no rationale of
        # its own and the bot's last description belongs to a different move.
        desc = (format_action_description(action_dict, view) if rec.forced
                else (src.last_description or format_action_description(action_dict, view)))

        policy: Optional[ReplayPolicyDict] = None
        critic: Optional[ReplayCriticDict] = None
        if trace:
            flat = int(rec.flat) if rec.flat is not None else 0
            if rec.forced:
                policy = unasked_readout(flat, after)
            else:
                # Only a neural bot has a distribution; a heuristic one leaves the block off
                # rather than reporting a certainty it never computed.
                policy = getattr(src.bot, "last_readout", None)
            if critic_model is not None and (
                    trace_critic_every == "step" or not rec.forced):
                critic = read_critic(critic_model, after)

        logger.log_step(
            step_index=rec.index,
            turn=rec.turn,
            ar=rec.action_round,
            phase=str(view.get("current_phase_name", "ACTION")),
            player=rec.player,
            action=replay_action,
            description=desc,
            state_snapshot=cast(GameStateDict, rec.state_after or {}),
            policy=policy,
            critic=critic,
        )
        if verbose and (commentary or isinstance(getattr(src, "bot", None), HumanBot)):
            print(f"[Step {rec.index:3d} | Turn {rec.turn} AR {rec.action_round} | "
                  f"{rec.player}]: {desc}")
            if commentary and not rec.forced:
                strat_text = getattr(src.bot, "last_strategy", "")
                comm_text = getattr(src.bot, "last_commentary", "")
                if strat_text:
                    print(f"    • Rationale: {strat_text}")
                if comm_text:
                    print(f"    • Commentary: {comm_text}")

    loop = GameLoop(
        state, sources,
        # RECORD_FORCED so every step reaches the recorder: a replay must re-drive by applying its
        # actions in order, without the reader knowing how it was produced.
        settle=SettlePolicy.RECORD_FORCED,
        recorder=_record,
        snapshot=lambda st: cast(Dict[str, Any], st.to_dict()),
        max_steps=max_steps,
    )
    outcome = loop.run()
    step_count = outcome.steps
    if verbose and outcome.forfeited_by:
        print(f"[{outcome.forfeited_by}] Forfeited or passed.")
    if verbose and outcome.hit_step_cap:
        print(f"⚠ Hit the {max_steps}-step cap without finishing -- this is not a completed game.")

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
    # Tri-state on purpose: the default differs by path. Neural self-play traces unless told
    # not to; a bot-vs-bot match has nothing to trace unless a neural side is in it and the
    # caller asks.
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None,
                        help="Record the policy's distribution at each node it chose and the "
                             "critic's reading of every position, into the replay "
                             "(default: on for neural self-play, off for a match)")
    parser.add_argument("--trace-top-k", type=int, default=TRACE_TOP_K,
                        help="Legal actions listed per node, by descending probability")
    parser.add_argument("--trace-critic-every", type=str, default="step",
                        choices=("step", "decision", "off"),
                        help="'step' for a gap-free value curve, 'decision' for chosen nodes only")

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
            trace=True if args.trace is None else args.trace,
            trace_top_k=args.trace_top_k,
            trace_critic_every=args.trace_critic_every,
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
        trace=bool(args.trace),
        trace_top_k=args.trace_top_k,
        trace_critic_every=args.trace_critic_every,
    )


if __name__ == "__main__":
    main()
