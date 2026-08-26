"""TsVectorizedEnv: High-performance vectorized wrapper for ts::VectorizedBatchRunner."""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import ts_engine as ts
from ai.env.action_encoder import ActionEncoder
from ai.env.reward_calculator import RewardCalculator, ZeroSumTerminalReward, BlunderAwareRewardCalculator


class TsEnv:
    """Gymnasium-like single-game environment wrapper for ts::Engine."""

    OBSERVATION_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(self, seed: Optional[int] = None, reward_calculator: Optional[RewardCalculator] = None):
        self.state = ts.GameState()
        self.seed = seed or 42
        self.reward_calc: RewardCalculator = reward_calculator or BlunderAwareRewardCalculator()
        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial game state."""
        if seed is not None:
            self.seed = seed
        ts.Engine.init_game(self.state, self.seed)
        p = self.current_player
        obs = ts.extract_observation(self.state, p)
        mask = ActionEncoder.get_legal_mask(self.state)
        return obs, mask, self._get_info()

    @property
    def current_player(self) -> ts.Player:
        """Returns the decision player if one is required, otherwise phasing player."""
        p = self.state.ctx().decision_player
        if p != ts.Player.NONE:
            return p
        return self.state.phasing_player

    @property
    def is_done(self) -> bool:
        return ts.Engine.is_terminal(self.state)

    def step(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """Executes a single micro-action in the environment."""
        if self.is_done:
            p = self.current_player
            obs = ts.extract_observation(self.state, p)
            mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.uint8)
            reward = float(ts.Engine.get_terminal_utility(self.state))
            return obs, mask, reward, True, self._get_info()

        acting_player = int(self.current_player)
        prev_vp = np.array([self.state.victory_points], dtype=np.int8)

        ok = ts.Engine.step_flat(self.state, int(action_idx))

        done = self.is_done
        term_util = np.array([ts.Engine.get_terminal_utility(self.state) if done else 0.0], dtype=np.float32)
        curr_vp = np.array([self.state.victory_points], dtype=np.int8)

        rewards = self.reward_calc.compute_step_rewards(
            acting_players=np.array([acting_player], dtype=np.int8),
            dones=np.array([done], dtype=bool),
            terminal_utilities=term_util,
            prev_victory_points=prev_vp,
            curr_victory_points=curr_vp,
            states=[self.state],
        )
        reward = float(rewards[0])

        p = self.current_player
        obs = ts.extract_observation(self.state, p)
        mask = ActionEncoder.get_legal_mask(self.state) if not done else np.zeros(self.ACTION_SPACE_SIZE, dtype=np.uint8)
        info = self._get_info()
        info["step_ok"] = ok
        info["acting_player"] = acting_player
        info["prev_player"] = acting_player
        return obs, mask, reward, done, info

    def _get_info(self) -> Dict[str, Any]:
        return {
            "turn": int(self.state.turn),
            "action_round": int(self.state.action_round),
            "phase": int(self.state.current_phase),
            "defcon": int(self.state.defcon),
            "victory_points": int(self.state.victory_points),
            "current_player": int(self.current_player),
            "terminal_utility": float(ts.Engine.get_terminal_utility(self.state)),
            "is_terminal": self.is_done,
        }


class TsVectorizedEnv:
    """High-throughput C++ vectorized batch environment executing N parallel games."""

    OBSERVATION_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(
        self,
        num_envs: int = 64,
        base_seed: int = 42,
        auto_reset: bool = True,
        reward_calculator: Optional[RewardCalculator] = None,
    ):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.auto_reset = auto_reset
        self.reward_calc: RewardCalculator = reward_calculator or BlunderAwareRewardCalculator()
        self.runner = ts.VectorizedBatchRunner(num_envs, base_seed)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self.ep_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset_all(self, base_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        if base_seed is not None:
            self.base_seed = base_seed
            self.runner = ts.VectorizedBatchRunner(self.num_envs, self.base_seed)
        else:
            self.runner.refresh_all()
        self.ep_lengths.fill(0)
        self.ep_rewards.fill(0.0)
        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)
        return obs, masks, self._get_batch_info()

    def reset_env(self, env_idx: int, seed: Optional[int] = None) -> None:
        """Reset a single environment by index."""
        s = seed if seed is not None else int(np.random.randint(1, 1_000_000_000))
        self.runner.reset_game(env_idx, s)
        self.ep_lengths[env_idx] = 0
        self.ep_rewards[env_idx] = 0.0

    def step(self, actions: np.ndarray | List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all N environments in parallel with exact acting player attribution."""
        # 1. Capture exact acting players BEFORE stepping simulation
        acting_players = np.array(self.runner.get_decision_players(), dtype=np.int8)
        prev_vp = np.array(self.runner.get_victory_points(), dtype=np.int8)

        # 2. Step C++ simulation in parallel
        action_list = [int(a) for a in actions]
        self.runner.step_flat_all(action_list)

        # 3. Read post-step metrics BEFORE auto-resetting
        dones = np.array(self.runner.get_terminals(), dtype=bool)
        term_utils = np.array(self.runner.get_terminal_utilities(), dtype=np.float32)
        curr_vp = np.array(self.runner.get_victory_points(), dtype=np.int8)

        # Retrieve state pointers for any terminal environments
        states: List[Optional[ts.GameState]] = []
        if np.any(dones):
            for i in range(self.num_envs):
                states.append(self.runner.get_state(i) if dones[i] else None)
        else:
            states = [None] * self.num_envs

        # 4. Pure algebraic reward computation via RewardCalculator
        rewards = self.reward_calc.compute_step_rewards(
            acting_players=acting_players,
            dones=dones,
            terminal_utilities=term_utils,
            prev_victory_points=prev_vp,
            curr_victory_points=curr_vp,
            states=states,
        )

        self.ep_lengths += 1
        self.ep_rewards += rewards

        completed_episodes = []
        if self.auto_reset:
            for i in range(self.num_envs):
                if dones[i]:
                    completed_episodes.append({
                        "env_idx": i,
                        "length": int(self.ep_lengths[i]),
                        "reward": float(self.ep_rewards[i]),
                        "terminal_utility": float(term_utils[i]),
                    })
                    new_seed = int(np.random.randint(1, 1_000_000_000))
                    self.runner.reset_game(i, new_seed)
                    self.ep_lengths[i] = 0
                    self.ep_rewards[i] = 0.0

            if completed_episodes:
                self.runner.refresh_all()

        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)

        info = self._get_batch_info()
        info["completed_episodes"] = completed_episodes
        info["acting_players"] = acting_players
        info["dones"] = dones

        return obs, masks, rewards, dones, info

    def _get_batch_info(self) -> Dict[str, Any]:
        return {
            "decision_players": np.array(self.runner.get_decision_players(), dtype=np.int8),
            "victory_points": np.array(self.runner.get_victory_points(), dtype=np.int8),
            "terminals": np.array(self.runner.get_terminals(), dtype=bool),
            "terminal_utilities": np.array(self.runner.get_terminal_utilities(), dtype=np.float32),
        }

# Backward compatibility alias
TsSingleEnv = TsEnv
