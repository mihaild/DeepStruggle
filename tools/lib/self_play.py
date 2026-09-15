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
from bindings.ts_env import check_obs_width
from web.server.replay import ReplayLogger, replays_dir
from web.server.replay_types import ReplayLogDict, ReplayActionDict, GameStateDict
from tools.lib.tournament_evaluator import classify_game_ending_reason
from tools.lib.game_loop import GameLoop, SettlePolicy, StepRecord
from tools.lib.openings import acting_side, scripted_setup_index


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
    opening: Optional[str] = None,
) -> Tuple[ReplayLogDict, str]:
    """Simulates a complete self-play game between neural policies and saves standardized .tslog.json replay.

    `opening` names a setup from `tools.lib.openings` to force on both sides in place of their own
    placements (currently only "human"). The policy plays everything after setup, so the replay
    shows what it does with a board it did not choose.
    """
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

    # There is one observation layout, so nothing here selects one. What is still worth
    # checking is that this model reads the width the engine emits: a mismatch means a
    # checkpoint from a retired layout, and a network reads fixed slices, so it would be
    # misread rather than rejected.
    check_obs_width(active_model)

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

    setup_cursor: Dict[str, int] = {"US": 0, "USSR": 0}

    blunders = BlunderCounts()
    last_card: dict[str, int] = {}
    if verbose:
        print("Step | Turn | AR | Player | DEFCON | VP | Action Description")
        print("-" * 75)

    class PolicySource:
        """The network's move, plus the bookkeeping that needs the pre-action position.

        The blunder check has to run here rather than in the recorder: it reads the card while it
        is still in its owner's hand, which is only true before the action is applied.
        """

        def __init__(self) -> None:
            self.last_desc = ""

        def choose(self, st: ts.GameState) -> Optional[ts.MicroAction]:
            p_enum = (st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE
                      else st.phasing_player)
            player_name = "US" if p_enum == ts.Player.US else (
                "USSR" if p_enum == ts.Player.USSR else "NONE")

            forced_idx = (scripted_setup_index(st, acting_side(st), opening, setup_cursor)
                          if opening else None)
            if forced_idx is not None:
                action_idx = forced_idx
            else:
                obs = np.array(ts.extract_observation(st, p_enum), copy=False).reshape(1, -1)
                mask = np.array(ActionEncoder.get_legal_mask(st), copy=False).reshape(1, -1)
                obs_t = torch.from_numpy(obs).float().to(dev)
                mask_t = torch.from_numpy(mask).to(dev)
                with torch.no_grad():
                    if hasattr(active_model, "sample_action"):
                        act_t, _, _, _, _ = active_model.sample_action(
                            obs_t, mask_t, temperature=temperature, deterministic=False)
                        action_idx = int(act_t.item())
                    else:
                        logits, _ = active_model(obs_t, mask_t)
                        action_idx = int(torch.argmax(logits, dim=-1).item())

            self.last_desc = ActionEncoder.get_action_name(st, action_idx)
            ma = ts.ActionMask.decode_flat_action(st, action_idx)

            # Blunder check. SELECT_PLAY_MODE is where both halves are known -- which card and
            # what it is being spent on -- and the card is still in its owner's hand there, which
            # the rules read. Missile Envy forces a card on its recipient, so that play is not
            # their error.
            _dt = int(ma.decision_type)
            if _dt == 1:
                last_card[player_name] = int(ma.primary_id)
            elif _dt == 2:
                _mode = {0: "EVENT", 1: "OPS", 2: "SPACE"}.get(int(ma.primary_id))
                _card = last_card.get(player_name, 0)
                if _mode and 1 <= _card <= 110:
                    _forced = (int(getattr(st, "forced_card_id", 0)) == _card
                               and getattr(st, "forced_card_player", None) == p_enum)
                    for _b in check_play(st, p_enum, _card, _mode, forced=_forced, counts=blunders):
                        if verbose:
                            print(f"     !! BLUNDER {_b}")
            return ma

    source = PolicySource()

    def _record(rec: StepRecord, after: ts.GameState) -> None:
        """Runs only after the engine accepted the action -- a refusal raises out of the loop."""
        dt = int(rec.action.decision_type)
        action_dict: ReplayActionDict = {
            "flat_action_idx": int(rec.flat) if rec.flat is not None else -1,
            "decision_type": dt,
            "primary_id": int(rec.action.primary_id),
            "secondary_id": int(rec.action.secondary_id),
            "flags": int(rec.action.flags),
            "card_id": int(rec.action.primary_id) if dt in (1, 2, 3, 4) else None,
            "target_id": int(rec.action.secondary_id) if dt in (5, 6, 7) else None,
        }
        desc = source.last_desc if not rec.forced else ActionEncoder.get_action_name(
            after, int(rec.flat) if rec.flat is not None else 0)
        replay_logger.log_step(
            step_index=rec.index + 1,
            turn=rec.turn,
            ar=rec.action_round,
            phase=rec.phase_name,
            player=rec.player,
            action=action_dict,
            description=desc,
            state_snapshot=cast(GameStateDict, rec.state_after),
        )
        if verbose and (rec.index % 10 == 0 or after.current_phase == ts.Phase.GAME_OVER
                        or "Scoring" in desc):
            print(f"{rec.index + 1:4d} | {rec.turn:4d} | {rec.action_round:2d} | "
                  f"{rec.player:>5s} | {after.defcon:6d} | {after.victory_points:+4d} | {desc}")

    loop = GameLoop(
        state, {ts.Player.US: source, ts.Player.USSR: source},
        settle=SettlePolicy.RECORD_FORCED,
        recorder=_record,
        snapshot=lambda st: cast(GameStateDict, ts.state_to_dict(st)),
        max_steps=max_steps,
    )
    outcome = loop.run()
    step_index = outcome.steps

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
