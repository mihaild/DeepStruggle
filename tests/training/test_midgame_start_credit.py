"""Credit assignment must not leak across an episode boundary when starts are mid-game.

With start-position sampling, an episode can begin at turn 8 immediately after one that
ended at turn 6, in the same buffer slot. Two mechanisms scan backwards over that boundary
and both had to be checked rather than assumed:

* GAE bootstraps from t+1 into t. If it carried across a reset, a resumed episode's value
  would be credited to the previous game's final move.
* The blunder window arms at a terminal step and walks back over "the turn the blunder
  happened in". Consecutive episodes can now sit on the same turn number, so turn equality
  alone could carry the window into a game that never blundered.

Neither needs the previous trajectory's advantage stored with the position: a resumed
episode is a genuinely new episode whose returns come from its own outcome. Carrying an
advantage over would import a different game's result.
"""

import torch

from ai.training.rollout_buffer import RolloutBuffer

OBS_DIM, ACTION_DIM = 4293, 212


def _buffer(size: int) -> RolloutBuffer:
    return RolloutBuffer(buffer_size=size, num_envs=1, obs_dim=OBS_DIM,
                         action_dim=ACTION_DIM, device="cpu")


def _add(buf: RolloutBuffer, *, player: int, turn: int, done: bool,
         value: float, reward: float = 0.0, blunder: int = 0) -> None:
    buf.add(
        obs=torch.zeros(1, OBS_DIM),
        masks=torch.ones(1, ACTION_DIM, dtype=torch.uint8),
        actions=torch.zeros(1, dtype=torch.long),
        log_probs=torch.zeros(1),
        rewards=torch.full((1,), reward),
        dones=torch.full((1,), float(done)),
        values_win=torch.full((1,), value),
        values_vp=torch.zeros(1),
        players=torch.full((1,), float(player)),
        turns=torch.full((1,), turn, dtype=torch.int8),
        vps=torch.zeros(1),
        defcon_blunder=torch.full((1,), blunder, dtype=torch.int8),
    )


def test_gae_does_not_bootstrap_across_a_resumed_episode() -> None:
    """The step that ends a game must not borrow value from the game that follows it."""
    buf = _buffer(4)
    # Episode A: two steps, ending at turn 6. Episode B resumes at turn 8.
    _add(buf, player=1, turn=6, done=False, value=0.0)
    _add(buf, player=-1, turn=6, done=True, value=0.0, reward=1.0)
    _add(buf, player=1, turn=8, done=False, value=0.9)   # a strong resumed position
    _add(buf, player=-1, turn=8, done=False, value=0.9)

    buf.compute_gae(last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
                    last_dones=torch.zeros(1), last_players=torch.ones(1),
                    blunder_window=False)

    # The terminal step's return is its own reward, not the resumed episode's value.
    assert abs(float(buf.returns_win[1, 0]) - 1.0) < 1e-5, (
        f"terminal return {float(buf.returns_win[1, 0])} was contaminated by the "
        f"episode that follows it in the buffer"
    )


def test_blunder_window_stays_inside_the_episode_that_blundered() -> None:
    """Turn equality alone must not carry the window into a neighbouring episode.

    Both episodes here sit on turn 6, which mid-game starts make common; only the later
    one ends in a self-inflicted DEFCON-1 loss.
    """
    buf = _buffer(4)
    # Episode A on turn 6, ending cleanly (no blunder).
    _add(buf, player=1, turn=6, done=False, value=0.5)
    _add(buf, player=1, turn=6, done=True, value=0.5, reward=1.0, blunder=0)
    # Episode B, also turn 6, ending in the US's own DEFCON-1 loss.
    _add(buf, player=1, turn=6, done=False, value=0.5)
    _add(buf, player=1, turn=6, done=True, value=0.5, reward=-1.0, blunder=1)

    buf.compute_gae(last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
                    last_dones=torch.zeros(1), last_players=torch.ones(1),
                    blunder_window=True)

    # Inside the window the blunderer's return is pinned to -1.
    assert abs(float(buf.returns_win[3, 0]) + 1.0) < 1e-5, "blunderer was not penalised"
    assert abs(float(buf.returns_win[2, 0]) + 1.0) < 1e-5, "window did not cover the run-up"
    # The clean episode on the same turn must be untouched by it.
    assert float(buf.returns_win[1, 0]) > 0.0, (
        f"the window leaked into the previous episode: its terminal return is "
        f"{float(buf.returns_win[1, 0])}, but that game ended cleanly"
    )


def test_a_resumed_episode_is_scored_by_its_own_outcome() -> None:
    """No need to carry an advantage across with the position: the new game earns its own."""
    buf = _buffer(2)
    _add(buf, player=1, turn=8, done=False, value=0.0)
    _add(buf, player=-1, turn=8, done=True, value=0.0, reward=-1.0)

    buf.compute_gae(last_v_win=torch.zeros(1), last_v_vp=torch.zeros(1),
                    last_dones=torch.zeros(1), last_players=torch.ones(1),
                    blunder_window=False)

    assert abs(float(buf.returns_win[1, 0]) + 1.0) < 1e-5


def test_completed_episodes_report_the_turn_they_started_on() -> None:
    """The env must tag each episode, or metrics cannot be split by start turn.

    Without this the pooled mean turn silently mixes real games with resumed ones: a run
    reads a mean turn of 8 while its turn-1 games are still ending at 6.
    """
    import numpy as np
    import ts_engine as ts
    from bindings.action_encoder import ActionEncoder
    from bindings.ts_env import TsVectorizedEnv

    def midgame_start(env_idx: int):
        if env_idx % 2:
            return None                      # odd envs play the real opening
        state = ts.GameState()
        ts.Engine.init_game(state, 4242 + env_idx)
        state.turn = 5                       # stored pre-deal, so it targets turn 6
        return state

    env = TsVectorizedEnv(num_envs=8, base_seed=4242, start_provider=midgame_start)
    _, masks, _ = env.reset_all()

    assert list(env.env_start_turn[:4]) == [6, 1, 6, 1], (
        f"start turns not tracked: {list(env.env_start_turn[:4])}"
    )

    seen: set = set()
    rng = np.random.RandomState(0)
    for _ in range(2000):
        actions = []
        for i in range(env.num_envs):
            legal = np.flatnonzero(np.asarray(masks)[i])
            actions.append(int(rng.choice(legal)) if len(legal) else 211)
        _, masks, _, _, info = env.step(np.array(actions))
        for ep in info.get("completed_episodes", []):
            assert "start_turn" in ep, "completed episode is missing start_turn"
            seen.add(ep["start_turn"])
        if len(seen) >= 2:
            break

    assert seen, "no episode completed; the fixture proves nothing"
    assert seen <= {1, 6}, f"unexpected start turns reported: {seen}"
