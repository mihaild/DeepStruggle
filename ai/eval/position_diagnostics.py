"""Diagnostics for the *positions* self-play produces, not just the results.

Win rate is a poor progress signal for this agent: it plateaus while the underlying play
stays incoherent. These metrics look at the board instead, and they moved first when
anything was actually wrong.

Two failure modes motivate them, both seen in trained agents:

* A group of battlegrounds sits completely untouched from the mid-game onwards, and the
  count stops falling -- the agent gives up contesting the board rather than slowly getting
  to it. Crucially the *same* battlegrounds are always the empty ones, and the
  always-contested ones are exactly those seeded by the opening setup: the agent fights
  where it was placed and never opens a new front.

* Games end early, so the Late War deck is nearly unplayed.

A "salvageable" position is one worth resuming from: neither side is far enough ahead that
the game is decided, and there is something left to play for.
"""

from __future__ import annotations

import collections
import statistics
from typing import Any, Callable, Dict, List, Optional

import ts_engine as ts
from bindings.settle import SettleMode, settle

REGIONS = (
    ts.Region.EUROPE, ts.Region.ASIA, ts.Region.MIDDLE_EAST,
    ts.Region.AFRICA, ts.Region.CENTRAL_AMERICA, ts.Region.SOUTH_AMERICA,
)

BATTLEGROUNDS: List[int] = [
    cid for cid in range(84) if ts.MapData.get_country_info(cid)["battleground"]
]

# Defaults for is_salvageable. The regional bar is on the *net*, not on either side's raw
# total: it is entirely normal for one side to be worth 20 VP across Africa and Central
# America while the other is worth 20 across South America and Asia. That is a balanced
# board, and scoring both would move the VP track by nothing. What disqualifies a position
# is the imbalance -- how far scoring everything right now would swing the game.
MAX_ABS_VP = 10
MAX_REGION_NET = 20


def region_score_sums(state: ts.GameState) -> tuple[int, int]:
    """(US, USSR) totals if every region were scored right now.

    Scoring.evaluate_region does not mutate, so this reuses the engine's own scoring
    rather than reimplementing presence/domination/control in Python.
    """
    us = ussr = 0
    for region in REGIONS:
        summary = ts.Scoring.evaluate_region(state, region, False)
        us += int(summary.us_score)
        ussr += int(summary.ussr_score)
    return us, ussr


def empty_battlegrounds(state: ts.GameState) -> List[int]:
    """Battlegrounds with no influence from either side."""
    out = []
    for cid in BATTLEGROUNDS:
        c = state.get_country(cid)
        if int(c.us_influence) == 0 and int(c.ussr_influence) == 0:
            out.append(cid)
    return out


def uncontrolled_battlegrounds(state: ts.GameState) -> List[int]:
    """Battlegrounds neither side controls.

    A strictly weaker condition than `empty_battlegrounds` and a different question. An empty
    battleground is one nobody has touched; an uncontrolled one may be heavily contested and
    still score for nobody. Both matter: the first says the agent is not showing up, the second
    says it is showing up and not finishing, and a policy can improve on one while the other
    stands still.
    """
    return [cid for cid in BATTLEGROUNDS
            if ts.Scoring.get_country_control(state, cid) == ts.Player.NONE]


def region_score_net(state: ts.GameState) -> int:
    """US minus USSR if every region were scored now; the swing the board is holding."""
    us, ussr = region_score_sums(state)
    return us - ussr


def is_salvageable(
    state: ts.GameState,
    max_abs_vp: int = MAX_ABS_VP,
    max_region_net: int = MAX_REGION_NET,
) -> bool:
    """Is this position worth resuming from -- undecided, with game left to play?

    Two ways a position can already be settled: the VP track has run away, or the board
    has, so that scoring the regions would immediately end it. Both are checked on the
    balance between the sides rather than on either side's absolute holdings.

    A decided position is worse than useless as a start state: with terminal-only reward
    every action there returns the same value, so it contributes no gradient at all while
    still consuming rollout budget.
    """
    if ts.Engine.is_terminal(state):
        return False
    if abs(int(state.victory_points)) > max_abs_vp:
        return False
    return abs(region_score_net(state)) <= max_region_net


def _drain(state: ts.GameState) -> None:
    settle(state, SettleMode.FORCED)


def profile_self_play(
    select_action: Callable[[ts.GameState, Any], Optional[int]],
    num_games: int = 100,
    base_seed: int = 810_000,
    max_steps: int = 3000,
) -> Dict[str, Any]:
    """Play games and describe the positions they pass through.

    select_action(state, player) -> flat action index, matching PlayerAgent.select_action.
    """
    reach: collections.Counter = collections.Counter()
    salvageable: collections.Counter = collections.Counter()
    empty_bgs: Dict[int, List[int]] = collections.defaultdict(list)
    unctrl_bgs: Dict[int, List[int]] = collections.defaultdict(list)
    per_bg_empty: collections.Counter = collections.Counter()
    late_samples = 0
    us_scores: Dict[int, List[int]] = collections.defaultdict(list)
    ussr_scores: Dict[int, List[int]] = collections.defaultdict(list)
    nets: Dict[int, List[int]] = collections.defaultdict(list)
    final_turns: List[int] = []

    for i in range(num_games):
        state = ts.GameState()
        ts.Engine.init_game(state, base_seed + i)
        seen: set = set()
        for _ in range(max_steps):
            _drain(state)
            if ts.Engine.is_terminal(state):
                break
            ctx = state.ctx()
            turn = int(state.turn)
            if turn not in seen:
                seen.add(turn)
                reach[turn] += 1
                if is_salvageable(state):
                    salvageable[turn] += 1
                empty = empty_battlegrounds(state)
                empty_bgs[turn].append(len(empty))
                unctrl_bgs[turn].append(len(uncontrolled_battlegrounds(state)))
                us, ussr = region_score_sums(state)
                us_scores[turn].append(us)
                ussr_scores[turn].append(ussr)
                nets[turn].append(us - ussr)
                if turn >= 6:
                    late_samples += 1
                    for cid in empty:
                        per_bg_empty[cid] += 1
            action = select_action(state, ctx.decision_player)
            if action is None:
                break
            ts.Engine.step_flat(state, int(action))
        final_turns.append(int(state.turn))

    def mean(xs: List[int]) -> float:
        return statistics.mean(xs) if xs else 0.0

    per_turn = {
        t: {
            "reached_frac": reach[t] / num_games,
            "salvageable_frac": salvageable[t] / num_games,
            "salvageable_given_reached": salvageable[t] / reach[t] if reach[t] else 0.0,
            "mean_empty_battlegrounds": mean(empty_bgs[t]),
            "mean_uncontrolled_battlegrounds": mean(unctrl_bgs[t]),
            "mean_us_region_score": mean(us_scores[t]),
            "mean_ussr_region_score": mean(ussr_scores[t]),
            "mean_abs_region_net": mean([abs(v) for v in nets[t]]),
        }
        for t in sorted(reach)
    }

    return {
        "num_games": num_games,
        "mean_final_turn": mean(final_turns),
        "per_turn": per_turn,
        "empty_battleground_rate_late": {
            ts.MapData.get_country_info(cid)["name"]: per_bg_empty[cid] / late_samples
            for cid in sorted(per_bg_empty, key=lambda c: -per_bg_empty[c])
        } if late_samples else {},
        "scalars": scalar_metrics(per_turn, mean(final_turns)),
    }


def scalar_metrics(per_turn: Dict[int, Dict[str, float]], mean_final_turn: float) -> Dict[str, float]:
    """The handful worth logging every evaluation.

    empty_battlegrounds_turn8 is the sharpest of these: it is a direct read on whether the
    agent has started contesting the board, and it moves long before win rate does.
    """
    def at(turn: int, key: str) -> float:
        return per_turn.get(turn, {}).get(key, 0.0)

    # frac_reaching_turn9 is deliberately absent: game length is measured on every training
    # iteration by game/mean_ply and the end-turn histogram, over every episode rather than
    # this probe's sample, so a second sparse copy of the same fact was only another chart.
    return {
        "diag/mean_final_turn": mean_final_turn,
        "diag/empty_battlegrounds_turn8": at(8, "mean_empty_battlegrounds"),
        "diag/empty_battlegrounds_turn5": at(5, "mean_empty_battlegrounds"),
        "diag/uncontrolled_battlegrounds_turn8": at(8, "mean_uncontrolled_battlegrounds"),
        "diag/uncontrolled_battlegrounds_turn5": at(5, "mean_uncontrolled_battlegrounds"),
        "diag/salvageable_frac_turn6": at(6, "salvageable_frac"),
        "diag/salvageable_given_reached_turn6": at(6, "salvageable_given_reached"),
    }


def format_report(profile: Dict[str, Any], top_battlegrounds: int = 10) -> str:
    """Human-readable summary for the snapshot report."""
    lines = [
        f"positions over {profile['num_games']} self-play games "
        f"(mean final turn {profile['mean_final_turn']:.2f})",
        "",
        f"{'turn':>4} {'reached':>9} {'salvageable':>12} {'empty BGs':>10} "
        f"{'US score':>9} {'USSR score':>11} {'|net|':>7}",
    ]
    for turn, row in profile["per_turn"].items():
        lines.append(
            f"{turn:>4} {100*row['reached_frac']:>8.1f}% {100*row['salvageable_frac']:>11.1f}% "
            f"{row['mean_empty_battlegrounds']:>10.2f} {row['mean_us_region_score']:>9.1f} "
            f"{row['mean_ussr_region_score']:>11.1f} {row['mean_abs_region_net']:>7.1f}"
        )
    rates = profile.get("empty_battleground_rate_late") or {}
    if rates:
        lines += ["", "most-neglected battlegrounds (turn >= 6):"]
        for name, rate in list(rates.items())[:top_battlegrounds]:
            lines.append(f"  {name:<20} empty in {100*rate:5.1f}% of positions")
    return "\n".join(lines)

def profile_self_play_batched(
    model: Any,
    num_envs: int = 256,
    base_seed: int = 820_000,
    temperature: float = 0.1,
    max_iters: int = 20_000,
) -> Dict[str, Any]:
    """Same profile, driven through the vectorized runner instead of one state at a time.

    Measured on this machine: the single-state loop manages ~890 decisions/sec, while 512
    batched envs reach ~106,000 -- the engine step is the same, the difference is entirely
    how much work each GPU call is given. Training itself runs at ~7,900 env-steps/sec
    because of the PPO epochs, so batched sampling is cheaper than the training it
    instruments, and the single-state version was nine times more expensive.

    Exactly one episode is profiled per environment -- the first -- and the loop runs until
    every env has finished it. Sample size is therefore num_envs, and which games land in
    the sample cannot depend on how long they take.

    Both halves of that matter, and an earlier version got only the first half right. It
    buffered positions per env and folded them in on completion, but still stopped at N
    *completed* episodes with N well below num_envs. Short games finish first, so the
    sample was the fastest N of num_envs: at N=30 over 128 envs it reported a mean final
    turn of 3.30 and claimed 0% of games reach turn 9, while draining the same policy
    unbiased gave 6.14 and 21%. Every late-game metric -- empty battlegrounds at turn 8,
    salvageability at turn 6 -- was pinned near zero by construction, because the games
    that get that far are exactly the ones still in flight when the budget ran out.
    """
    import numpy as np
    import torch

    from bindings.ts_env import TsVectorizedEnv, check_obs_width

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # There is one layout now, so this probe cannot be handed the wrong one -- which is what
    # every number it reported before that was true. What is still checked is the model's own
    # width, because a mismatch means a checkpoint from a retired layout.
    check_obs_width(model)
    env = TsVectorizedEnv(num_envs=num_envs, base_seed=base_seed)
    obs, masks, _ = env.reset_all()

    reach: collections.Counter = collections.Counter()
    salvageable: collections.Counter = collections.Counter()
    empty_bgs: Dict[int, List[int]] = collections.defaultdict(list)
    unctrl_bgs: Dict[int, List[int]] = collections.defaultdict(list)
    us_scores: Dict[int, List[int]] = collections.defaultdict(list)
    ussr_scores: Dict[int, List[int]] = collections.defaultdict(list)
    nets: Dict[int, List[int]] = collections.defaultdict(list)
    per_bg_empty: collections.Counter = collections.Counter()
    late_samples = 0
    final_turns: List[int] = []
    episodes = 0

    pending: List[List[tuple]] = [[] for _ in range(num_envs)]
    seen: List[set] = [set() for _ in range(num_envs)]

    def flush(i: int) -> None:
        nonlocal late_samples, episodes
        if not pending[i]:
            pending[i], seen[i] = [], set()
            return
        for (turn, salv, empty_ids, us, ussr, n_unctrl) in pending[i]:
            reach[turn] += 1
            if salv:
                salvageable[turn] += 1
            empty_bgs[turn].append(len(empty_ids))
            unctrl_bgs[turn].append(n_unctrl)
            us_scores[turn].append(us)
            ussr_scores[turn].append(ussr)
            nets[turn].append(us - ussr)
            if turn >= 6:
                late_samples += 1
                for cid in empty_ids:
                    per_bg_empty[cid] += 1
        final_turns.append(max(t for (t, *_rest) in pending[i]))
        episodes += 1
        pending[i], seen[i] = [], set()

    counted = [False] * num_envs

    try:
        for _ in range(max_iters):
            if all(counted):
                break
            for i in range(num_envs):
                if counted[i]:
                    continue
                state = env.runner.get_state(i)
                if ts.Engine.is_terminal(state):
                    continue
                turn = int(state.turn)
                if turn in seen[i]:
                    continue
                seen[i].add(turn)
                empty = empty_battlegrounds(state)
                unctrl = uncontrolled_battlegrounds(state)
                us, ussr = region_score_sums(state)
                pending[i].append((turn, is_salvageable(state), empty, us, ussr, len(unctrl)))

            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            mask_t = torch.from_numpy(np.asarray(masks)).to(device)
            with torch.no_grad():
                actions, _, _, _, _ = model.sample_action(obs_t, mask_t, temperature=temperature)
            obs, masks, _, dones, _ = env.step(actions.cpu().numpy())
            for i, done in enumerate(dones):
                if done and not counted[i]:
                    flush(i)
                    counted[i] = True
    finally:
        if was_training:
            model.train()

    # Envs whose first episode never finished inside max_iters are dropped rather than
    # counted half-played. That is a bounded, symmetric loss, not a length filter.
    return _assemble(episodes, final_turns, reach, salvageable, empty_bgs, unctrl_bgs,
                     us_scores, ussr_scores, nets, per_bg_empty, late_samples)


def _assemble(num_games, final_turns, reach, salvageable, empty_bgs, unctrl_bgs,
              us_scores, ussr_scores, nets, per_bg_empty, late_samples) -> Dict[str, Any]:
    def mean(xs) -> float:
        xs = list(xs)
        return statistics.mean(xs) if xs else 0.0

    # Normalise by episodes *started*, not episodes completed: envs still mid-game when
    # the budget runs out have contributed positions but no completion, which would push
    # reached_frac above 1. Every episode passes through turn 1, so reach[1] counts starts.
    started = reach[1] if reach.get(1) else max(1, num_games)
    per_turn = {
        t: {
            "reached_frac": reach[t] / started,
            "salvageable_frac": salvageable[t] / started,
            "salvageable_given_reached": salvageable[t] / reach[t] if reach[t] else 0.0,
            "mean_empty_battlegrounds": mean(empty_bgs[t]),
            "mean_uncontrolled_battlegrounds": mean(unctrl_bgs[t]),
            "mean_us_region_score": mean(us_scores[t]),
            "mean_ussr_region_score": mean(ussr_scores[t]),
            "mean_abs_region_net": mean(abs(v) for v in nets[t]),
        }
        for t in sorted(reach)
    }
    return {
        "num_games": num_games,
        "mean_final_turn": mean(final_turns),
        "per_turn": per_turn,
        "empty_battleground_rate_late": {
            ts.MapData.get_country_info(cid)["name"]: per_bg_empty[cid] / late_samples
            for cid in sorted(per_bg_empty, key=lambda c: -per_bg_empty[c])
        } if late_samples else {},
        "scalars": scalar_metrics(per_turn, mean(final_turns)),
    }
