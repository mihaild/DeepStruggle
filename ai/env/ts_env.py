"""Twilight Struggle Vectorized and Single Environment Wrappers."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import ts_engine as ts
from .action_encoder import ActionEncoder


class TsSingleEnv:
    """Single environment wrapper for ts.GameState."""

    OBSERVATION_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(self, seed: Optional[int] = None):
        self.state = ts.GameState()
        self.seed = seed or np.random.randint(1, 1_000_000_000)
        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Resets the game state with given or random seed."""
        if seed is not None:
            self.seed = seed
        ts.Engine.init_game(self.state, self.seed)
        p = self.current_player
        obs = ts.extract_observation(self.state, p)
        mask = ActionEncoder.get_legal_mask(self.state)
        info = self._get_info()
        return obs, mask, info

    @property
    def current_player(self) -> ts.Player:
        """Returns the active decision player (or phasing player)."""
        ctx_player = self.state.ctx().decision_player
        if ctx_player != ts.Player.NONE:
            return ctx_player
        return self.state.phasing_player

    @property
    def is_done(self) -> bool:
        """Checks if the game has concluded."""
        return bool(ts.Engine.is_terminal(self.state))

    def step(self, action_idx: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """Executes a flat action index [0..211]."""
        if self.is_done:
            p = self.current_player
            obs = ts.extract_observation(self.state, p)
            mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.uint8)
            reward = ts.Engine.get_terminal_utility(self.state)
            return obs, mask, reward, True, self._get_info()

        prev_player = self.current_player
        ok = ts.Engine.step_flat(self.state, int(action_idx))

        done = self.is_done
        terminal_util = ts.Engine.get_terminal_utility(self.state) if done else 0.0

        # Perspective reward: positive if favourable to prev_player
        reward = float(terminal_util if prev_player == ts.Player.US else -terminal_util)

        p = self.current_player
        obs = ts.extract_observation(self.state, p)
        mask = ActionEncoder.get_legal_mask(self.state) if not done else np.zeros(self.ACTION_SPACE_SIZE, dtype=np.uint8)
        info = self._get_info()
        info["step_ok"] = ok
        info["prev_player"] = int(prev_player)
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
    """High-throughput C++ vectorized batch environment executing N parallel games.

    Leverages ts.VectorizedBatchRunner in contiguous C++ memory for peak rollout speed.
    """

    OBSERVATION_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(self, num_envs: int = 64, base_seed: int = 42, auto_reset: bool = True):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.auto_reset = auto_reset
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
        """Steps all N environments in parallel.

        Args:
            actions: Array/list of flat action indices of length num_envs.

        Returns:
            obs: np.ndarray shape (num_envs, 4293)
            masks: np.ndarray shape (num_envs, 212)
            rewards: np.ndarray shape (num_envs,) perspective-aligned rewards
            dones: np.ndarray shape (num_envs,) boolean flags
            info: Dict containing episode statistics and terminal outcomes
        """
        prev_players = np.array(self.runner.get_decision_players(), dtype=np.int8)

        # Convert to list of ints for C++ runner
        action_list = [int(a) for a in actions]
        self.runner.step_flat_all(action_list)

        dones = np.array(self.runner.get_terminals(), dtype=bool)
        term_utils = np.array(self.runner.get_terminal_utilities(), dtype=np.float32)

        # Reward calculation: terminal utility (+1 US / -1 USSR) aligned to acting player perspective
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            if dones[i]:
                # terminal reward from perspective of the player that acted
                rewards[i] = term_utils[i] if prev_players[i] == 1 else -term_utils[i]

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
                        "victory_points": int(self.runner.get_victory_points()[i]),
                        "winner": "US" if term_utils[i] > 0 else ("USSR" if term_utils[i] < 0 else "DRAW"),
                    })
                    # Auto-reset environment with fresh random seed
                    self.reset_env(i)

        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)

        info = self._get_batch_info()
        info["completed_episodes"] = completed_episodes
        info["prev_players"] = prev_players
        return obs, masks, rewards, dones, info

    def _get_batch_info(self) -> Dict[str, Any]:
        return {
            "decision_players": np.array(self.runner.get_decision_players(), dtype=np.int8),
            "terminals": np.array(self.runner.get_terminals(), dtype=bool),
            "victory_points": np.array(self.runner.get_victory_points(), dtype=np.int8),
        }
