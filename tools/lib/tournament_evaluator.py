# Tournament Evaluator: Match benchmarking and detailed game outcome statistics.

import time
from typing import Dict, List, Any, Optional
import numpy as np
import ts_engine as ts
from tools.lib.player_agent import PlayerAgent
from ai.game_length import ply as game_ply


def classify_game_ending_reason(state: ts.GameState) -> str:
    """Accurately determines the exact cause of game termination from the canonical list:
    - 20 VP
    - Europe Control
    - DEFCON 1 (own decision)
    - DEFCON 1 (opponent decision)
    - final scoring
    - wargames
    """
    # 0. Cuban Missile Crisis suicide: couping while CMC is active without the influence
    # to cancel it. The engine ends the game at +/-20 VP and deliberately leaves DEFCON
    # alone, so without this check the loss is indistinguishable from a legitimate 20 VP
    # win -- and it is a self-inflicted loss, not a win by anyone's play.
    if state.has_flag(ts.EffectBits.CMC_SUICIDE_LOSS):
        return "DEFCON 1 (own decision)"

    # 1. DEFCON 1 (Takes absolute precedence over VP)
    if state.defcon <= 1:
        is_provoked = state.has_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED)
        return "DEFCON 1 (opponent decision)" if is_provoked else "DEFCON 1 (own decision)"

    # 2. Wargames (#100): the game is over, before final scoring, without 20 VP.
    #
    # The bound is `<= 10`, not `< 10`. A game that goes the distance terminates holding
    # turn *11*: finish_end_turn increments the turn and only then tests `turn <= 10` before
    # calling execute_final_scoring. So turn 10 is not final scoring -- it is a game that
    # ended during the last turn -- and with `< 10` a Wargames played in turn 10 fell past
    # this test to rule 4 and was reported as final scoring. 3 of the 119 finished games in
    # the human corpus end exactly that way.
    if state.turn <= 10 and abs(state.victory_points) < 20 and state.current_phase == ts.Phase.GAME_OVER:
        return "wargames"

    # 3. Europe Control: controlling Europe when Europe is scored ends the game at +/-20 VP,
    # which is indistinguishable from any other 20 VP win without the flag the engine sets.
    # It is 1.4% of the ITS corpus and was being counted as an ordinary 20 VP win here.
    if state.has_flag(ts.EffectBits.EUROPE_CONTROL_WIN):
        return "Europe Control"

    # 4. 20 VP Milestone or Held Scoring
    if abs(state.victory_points) >= 20:
        return "20 VP"

    # 5. Final Scoring
    if state.turn >= 10:
        return "final scoring"

    if state.current_phase == ts.Phase.GAME_OVER:
        return "wargames"

    return "20 VP"


def _drain_chance_nodes(state: "ts.GameState") -> None:
    """Resolve pending die rolls before asking a policy to move.

    At a ROLL_DIE node ctx().decision_player is NONE. The loop below falls back to
    phasing_player when that happens, which handed the die roll to whichever agent was
    phasing and let a policy network pick its own dice through step_flat -- 134 such nodes
    per 20 games, about 6.7 a game. The vectorized runner resolves them inside the engine
    and never exposes them, which is why the two paths disagreed by more than 25 points on
    the same matchup. Draining here makes this path agree with the batched one, and is the
    same fix already applied to the decisive probe.
    """
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))


class TournamentEvaluator:
    """Runs head-to-head match evaluations between agents with full statistical profiling."""

    @staticmethod
    def play_matchup(
        agent_a: PlayerAgent,
        agent_b: PlayerAgent,
        games_per_side: int = 50,
        base_seed: int = 10000,
        max_steps: int = 2000,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        total_games = games_per_side * 2
        a_wins = 0
        b_wins = 0
        draws = 0

        a_us_wins = 0
        a_us_losses = 0
        a_us_draws = 0

        a_ussr_wins = 0
        a_ussr_losses = 0
        a_ussr_draws = 0

        steps_list: List[int] = []
        turns_list: List[int] = []
        # Length in plies as well as turns: the turn counter answers "which of the ten" and
        # nothing finer, so a game abandoned at turn 7 AR1 and one that runs to turn 7 AR7 are
        # the same number, and a completed game reads 11 rather than the 10 a human log calls
        # it. See ai.eval.game_length.
        plies_list: List[int] = []
        vp_margins: List[int] = []

        causes_loss_us: Dict[str, int] = {}
        causes_loss_ussr: Dict[str, int] = {}
        causes_all: Dict[str, int] = {}

        t0 = time.time()

        for g_idx in range(total_games):
            a_is_ussr = (g_idx < games_per_side)
            # Paired deals: game g and game g + games_per_side share a seed and so share
            # an identical shuffle, with the sides swapped. Deal luck then cancels between
            # the two halves instead of contributing variance to the difference, which is
            # what makes small effect sizes measurable at a fixed game count.
            seed = base_seed + (g_idx % games_per_side)

            st = ts.GameState()
            ts.Engine.init_game(st, seed)
            _drain_chance_nodes(st)

            step = 0
            while not ts.Engine.is_terminal(st) and step < max_steps:
                p = st.ctx().decision_player if st.ctx().decision_player != ts.Player.NONE else st.phasing_player
                if (p == ts.Player.USSR and a_is_ussr) or (p == ts.Player.US and not a_is_ussr):
                    act_idx = agent_a.select_action(st, p, temperature=temperature)
                else:
                    act_idx = agent_b.select_action(st, p, temperature=temperature)

                ts.Engine.step_flat(st, act_idx)
                _drain_chance_nodes(st)
                step += 1

            term_util = float(ts.Engine.get_terminal_utility(st))
            vp = int(st.victory_points)
            turn = int(st.turn)
            reason = classify_game_ending_reason(st)

            causes_all[reason] = causes_all.get(reason, 0) + 1
            steps_list.append(step)
            turns_list.append(turn)
            plies_list.append(game_ply(turn, int(st.action_round),
                                       st.phasing_player == ts.Player.US,
                                       headline_stage=int(st.headline_stage)))

            a_won = (term_util > 0 and not a_is_ussr) or (term_util < 0 and a_is_ussr)
            b_won = (term_util < 0 and not a_is_ussr) or (term_util > 0 and a_is_ussr)

            if a_won:
                a_wins += 1
                if a_is_ussr:
                    a_ussr_wins += 1
                else:
                    a_us_wins += 1
            elif b_won:
                b_wins += 1
                clean_reason = reason
                if a_is_ussr:
                    a_ussr_losses += 1
                    causes_loss_ussr[clean_reason] = causes_loss_ussr.get(clean_reason, 0) + 1
                else:
                    a_us_losses += 1
                    causes_loss_us[clean_reason] = causes_loss_us.get(clean_reason, 0) + 1
            else:
                draws += 1
                if a_is_ussr:
                    a_ussr_draws += 1
                else:
                    a_us_draws += 1

            vp_for_a = -vp if a_is_ussr else vp
            vp_margins.append(vp_for_a)

        elapsed = time.time() - t0

        return {
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "total_games": total_games,
            "games_per_side": games_per_side,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "win_rate_a": float(a_wins / max(1, total_games)),
            "win_rate_b": float(b_wins / max(1, total_games)),
            "a_wins_as_us": a_us_wins,
            "a_losses_as_us": a_us_losses,
            "a_draws_as_us": a_us_draws,
            "win_rate_a_as_us": float(a_us_wins / max(1, games_per_side)),
            "a_wins_as_ussr": a_ussr_wins,
            "a_losses_as_ussr": a_ussr_losses,
            "a_draws_as_ussr": a_ussr_draws,
            "win_rate_a_as_ussr": float(a_ussr_wins / max(1, games_per_side)),
            "avg_steps": float(np.mean(steps_list)),
            "avg_turn": float(np.mean(turns_list)),
            "avg_ply": float(np.mean(plies_list)),
            "avg_vp_margin_a": float(np.mean(vp_margins)),
            "causes_loss_us": causes_loss_us,
            "causes_loss_ussr": causes_loss_ussr,
            "causes_all": causes_all,
            "elapsed_seconds": elapsed,
        }
