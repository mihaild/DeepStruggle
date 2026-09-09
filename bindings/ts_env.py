"""TsVectorizedEnv: High-performance vectorized wrapper for ts::VectorizedBatchRunner."""

from typing import Tuple, Dict, Any, List, Optional, Callable
import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from ai.rewards.reward_calculator import RewardCalculator, ZeroSumTerminalReward, BlunderAwareRewardCalculator

# Canonical short keys for the game-ending reasons reported per completed episode.
# The long-form strings come from tools.lib.tournament_evaluator.classify_game_ending_reason,
# which stays the single source of truth for the classification itself.
ENDING_REASON_KEYS: Tuple[str, ...] = (
    "20vp",
    "final_scoring",
    "defcon1_self",
    "defcon1_provoked",
    "held_scoring",
    "wargames",
)

_ENDING_REASON_MAP: Dict[str, str] = {
    "20 VP": "20vp",
    "final scoring": "final_scoring",
    "DEFCON 1 (own decision)": "defcon1_self",
    "DEFCON 1 (opponent decision)": "defcon1_provoked",
    "wargames": "wargames",
}

_ending_classifier: Optional[Callable[[ts.GameState], str]] = None


def _classify_ending(state: ts.GameState, held_scoring: bool) -> str:
    """Maps a terminal state to one of ENDING_REASON_KEYS.

    Imported lazily: tools.lib.tournament_evaluator pulls in torch and the model zoo,
    which the environment itself has no need for.
    """
    global _ending_classifier
    if _ending_classifier is None:
        from tools.lib.tournament_evaluator import classify_game_ending_reason
        _ending_classifier = classify_game_ending_reason
    reason = _ending_classifier(state)
    key = _ENDING_REASON_MAP.get(reason, "20vp")
    # DEFCON 1 keeps absolute precedence (as in the classifier); otherwise an
    # unplayed scoring card is the more specific cause than the VP total it produced.
    if held_scoring and not key.startswith("defcon1"):
        key = "held_scoring"
    return key


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
        while not self.is_done and self.state.ctx().decision_player == ts.Player.NONE and self.state.ctx().decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(self.state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))

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

    # The legacy width. An instance built with legacy_obs=False reports the v2 width instead,
    # so callers should read `self.observation_size` rather than the class attribute.
    OBSERVATION_SIZE = 4293
    ACTION_SPACE_SIZE = 212

    def __init__(
        self,
        num_envs: int = 64,
        base_seed: int = 42,
        auto_reset: bool = True,
        start_provider: Optional[Callable[[int], Optional["ts.GameState"]]] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        layout: str = "legacy",
    ):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.auto_reset = auto_reset
        # Fixed for the environment's lifetime: the model's input width is built from it, so an
        # env that changed layout mid-run would simply be a way to feed a network garbage.
        self.layout = str(layout)
        self.observation_size = int({"legacy": ts.OBS_SIZE_LEGACY,
                                     "v2.1": ts.OBS_SIZE_V21,
                                     "v2.2": ts.OBS_SIZE_V22}[self.layout])
        # Optional source of mid-game start positions. Called with an env index after that
        # env resets; returning a GameState starts it there instead of from a fresh deal,
        # returning None leaves the real opening. The provider owns cloning and reseeding:
        # handing back a stored object would let training step it, and reusing its
        # rng_state would give every rollout from that position the same deal and dice.
        self.start_provider = start_provider
        # Which game turn each env's current episode began on. 1 unless a mid-game start
        # position was injected. Reported with completed episodes so training metrics can
        # be split by start turn: with half the envs resuming mid-game, a pooled mean turn
        # or explained variance describes neither the real game nor the resumed one.
        self.env_start_turn = np.ones(num_envs, dtype=np.int16)
        self.reward_calc: RewardCalculator = reward_calculator or BlunderAwareRewardCalculator()
        self.runner = ts.VectorizedBatchRunner(num_envs, base_seed, self.layout)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self.ep_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset_all(self, base_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        if hasattr(self.reward_calc, "reset"):
            self.reward_calc.reset()
        if base_seed is not None:
            self.base_seed = base_seed
            self.runner = ts.VectorizedBatchRunner(self.num_envs, self.base_seed,
                                                   self.layout)
        else:
            self.runner.refresh_all()
        for _i in range(self.num_envs):
            self._apply_start_position(_i)
        self.ep_lengths.fill(0)
        self.ep_rewards.fill(0.0)
        if hasattr(self.reward_calc, "on_all_reset"):
            self.reward_calc.on_all_reset([self.runner.get_state(i) for i in range(self.num_envs)])
        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)
        return obs, masks, self._get_batch_info()

    def _apply_start_position(self, env_idx: int) -> None:
        self.env_start_turn[env_idx] = 1
        if self.start_provider is None:
            return
        state = self.start_provider(env_idx)
        if state is not None:
            self.runner.set_state(env_idx, state)
            # Positions are stored pre-deal, one turn before the turn they target, so the
            # episode's effective start turn is the next one.
            self.env_start_turn[env_idx] = int(state.turn) + 1

    def reset_env(self, env_idx: int, seed: Optional[int] = None) -> None:
        """Reset a single environment by index."""
        s = seed if seed is not None else int(np.random.randint(1, 1_000_000_000))
        self.runner.reset_game(env_idx, s)
        self._apply_start_position(env_idx)
        self.ep_lengths[env_idx] = 0
        self.ep_rewards[env_idx] = 0.0

    def step(self, actions: np.ndarray | List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all N environments in parallel with exact acting player attribution."""
        # 1. Capture exact acting players & privileged opponent hands BEFORE stepping simulation
        acting_players = np.array(self.runner.get_decision_players(), dtype=np.int8)
        opp_hands = np.array(self.runner.get_opponent_hands(acting_players.tolist()), dtype=np.float32).reshape(self.num_envs, 110)
        acting_turns = np.array(self.runner.get_turns(), dtype=np.int8)
        prev_vp = np.array(self.runner.get_victory_points(), dtype=np.int8)

        # 2. Step C++ simulation in parallel
        action_list = [int(a) for a in actions]
        self.runner.step_flat_all(action_list)

        # 3. Read post-step metrics BEFORE auto-resetting
        dones = np.array(self.runner.get_terminals(), dtype=bool)
        term_utils = np.array(self.runner.get_terminal_utilities(), dtype=np.float32)
        curr_vp = np.array(self.runner.get_victory_points(), dtype=np.int8)
        # Post-step turn, read before any auto-reset rewinds the game to turn 1.
        curr_turns = np.array(self.runner.get_turns(), dtype=np.int8)

        # Rule 4.4 Held scoring card detection for terminal states
        held_scoring_us = np.zeros(self.num_envs, dtype=bool)
        held_scoring_ussr = np.zeros(self.num_envs, dtype=bool)
        # Unprovoked DEFCON-1 suicide: the player who chose the losing action, else 0.
        # A provoked suicide stays 0 on purpose. There the phasing player was forced to fire
        # an opponent-associated event, so the mistake lies in the earlier card management
        # rather than the final move, and its credit must keep propagating backwards.
        defcon_blunder = np.zeros(self.num_envs, dtype=np.int8)
        ending_reasons: List[str] = [""] * self.num_envs
        done_idx = np.flatnonzero(dones)
        if len(done_idx):
            # Only a handful of the envs finish on any given step, so walk the terminal
            # ones directly rather than testing all num_envs in Python.
            for raw in done_idx:
                i = int(raw)
                st = self.runner.get_state(i)
                if ts.Engine.is_held_scoring_game_over(st):
                    held_scoring_us[i] = ts.Engine.is_held_scoring_loss(st, ts.Player.US)
                    held_scoring_ussr[i] = ts.Engine.is_held_scoring_loss(st, ts.Player.USSR)
                elif st.defcon <= 1 and not st.has_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED):
                    defcon_blunder[i] = int(st.phasing_player)
                elif st.has_flag(ts.EffectBits.CMC_SUICIDE_LOSS):
                    # Couping under Cuban Missile Crisis. The engine ends the game without
                    # touching DEFCON, so the check above cannot see it, and these
                    # self-inflicted losses were being credited as ordinary outcomes
                    # instead of windowed as blunders.
                    defcon_blunder[i] = int(st.phasing_player)
                ending_reasons[i] = _classify_ending(
                    st, bool(held_scoring_us[i] or held_scoring_ussr[i])
                )

        # Retrieve state pointers for any terminal environments (or all environments if reward calculator requires it)
        states: List[Optional[ts.GameState]] = []
        needs_all_states = getattr(self.reward_calc, "needs_all_states", False)
        if needs_all_states:
            for i in range(self.num_envs):
                states.append(self.runner.get_state(i))
        elif len(done_idx):
            states = [None] * self.num_envs
            for i in done_idx:
                states[int(i)] = self.runner.get_state(int(i))
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

        completed_episodes: List[Dict[str, Any]] = []
        if self.auto_reset:
            for i in range(self.num_envs):
                if dones[i]:
                    completed_episodes.append({
                        "env_idx": i,
                        "length": int(self.ep_lengths[i]),
                        "reward": float(self.ep_rewards[i]),
                        "terminal_utility": float(term_utils[i]),
                        "winner": "US" if term_utils[i] > 0 else ("USSR" if term_utils[i] < 0 else "DRAW"),
                        "victory_points": int(curr_vp[i]),
                        "turn": int(curr_turns[i]),
                        "ending_reason": ending_reasons[i],
                        "start_turn": int(self.env_start_turn[i]),
                    })
                    new_seed = int(np.random.randint(1, 1_000_000_000))
                    self.runner.reset_game(i, new_seed)
                    # Inject before on_env_reset so the reward calculator sees the position
                    # the episode actually begins from, not the discarded fresh deal.
                    self._apply_start_position(i)
                    if hasattr(self.reward_calc, "on_env_reset"):
                        self.reward_calc.on_env_reset(i, self.runner.get_state(i))
                    self.ep_lengths[i] = 0
                    self.ep_rewards[i] = 0.0

            if completed_episodes:
                self.runner.refresh_all()

        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)

        info = self._get_batch_info()
        info["victory_points"] = curr_vp
        info["completed_episodes"] = completed_episodes
        info["acting_players"] = acting_players
        info["opponent_hands"] = opp_hands
        info["turns"] = acting_turns
        info["held_scoring_us"] = held_scoring_us
        info["held_scoring_ussr"] = held_scoring_ussr
        info["defcon_blunder"] = defcon_blunder
        info["ending_reasons"] = ending_reasons
        info["terminal_turns"] = curr_turns
        info["dones"] = dones

        return obs, masks, rewards, dones, info

    def _get_batch_info(self) -> Dict[str, Any]:
        return {
            "decision_players": np.array(self.runner.get_decision_players(), dtype=np.int8),
            "turns": np.array(self.runner.get_turns(), dtype=np.int8),
            "victory_points": np.array(self.runner.get_victory_points(), dtype=np.int8),
            "terminals": np.array(self.runner.get_terminals(), dtype=bool),
            "terminal_utilities": np.array(self.runner.get_terminal_utilities(), dtype=np.float32),
        }

# Backward compatibility alias
TsSingleEnv = TsEnv
