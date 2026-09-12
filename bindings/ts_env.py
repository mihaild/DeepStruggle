"""TsVectorizedEnv: High-performance vectorized wrapper for ts::VectorizedBatchRunner."""

from typing import Tuple, Dict, Any, List, Optional, Callable
import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from ai.rewards.reward_calculator import RewardCalculator, ZeroSumTerminalReward, BlunderAwareRewardCalculator
from ai.game_length import ply as _game_ply

# Canonical short keys for the game-ending reasons reported per completed episode.
# The long-form strings come from tools.lib.tournament_evaluator.classify_game_ending_reason,
# which stays the single source of truth for the classification itself.
ENDING_REASON_KEYS: Tuple[str, ...] = (
    "20vp",
    "europe_control",
    "final_scoring",
    "defcon1_self",
    "defcon1_provoked",
    "held_scoring",
    "wargames",
)

_ENDING_REASON_MAP: Dict[str, str] = {
    "20 VP": "20vp",
    "Europe Control": "europe_control",
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


#: The engine's one observation layout, by the name runs record in their metadata. A plain
#: string, deliberately: nothing here reads an engine constant at import time -- see obs_size().
OBS_LAYOUT_NAME: str = "v2.3"


def obs_size() -> int:
    """The observation width this ts_engine build emits, or a rebuild instruction.

    Read through getattr at *call* time, never at import time. Reaching straight for a constant
    at module scope meant a build that predated it raised AttributeError from inside
    `import bindings`, which took down every consumer of the package -- a web server included,
    over a layout it never used. A stale build is a real problem and still gets a loud error,
    but from whatever actually needs the engine rather than from the import.
    """
    width = getattr(ts, "OBS_SIZE", None)
    if width is None:
        raise RuntimeError(
            "this ts_engine build does not define OBS_SIZE, so it predates the single-layout "
            "refactor. Rebuild the engine:\n"
            "    tools/scripts/check_engine_fresh.sh")
    return int(width)


def check_obs_width(model: Any) -> int:
    """The observation width a model reads, checked against the engine's one layout.

    There used to be three layouts and a `layout_for_model` that mapped a width to a name. The
    names are gone -- `ts.extract_observation` and `ts.VectorizedBatchRunner` take no layout, so
    there is nothing left to select wrongly. What remains worth checking is the other half of the
    old failure: a model whose input width is not the engine's, which does not raise on its own
    because a network reads fixed slices and a mismatched vector simply gets misread. Four
    separate probes did exactly that, one of them reporting a mean final turn of 1-2 against an
    actual 6.8 (`research/metrics.md` 1.4.1).

    A model of the wrong width is a checkpoint from a retired layout: legacy (4293) or v2.1
    (3891) or v2.2 (3825). Those cannot be run and are not being converted -- they predate the
    starred-card fix, so they were trained against a different game.
    """
    width = int(getattr(model, "TOTAL_OBS_SIZE", 0) or 0)
    if width != obs_size():
        raise ValueError(
            f"this model reads {width} floats; the engine emits {obs_size()} (layout "
            f"v2.3). A checkpoint from a retired layout cannot be run: it would load cleanly "
            f"and misread every input.")
    return width


class TsEnv:
    """Gymnasium-like single-game environment wrapper for ts::Engine."""

    ACTION_SPACE_SIZE = 212

    @property
    def OBSERVATION_SIZE(self) -> int:
        return obs_size()

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


def credits_defcon_blunder(state: "ts.GameState", window_provoked: bool) -> bool:
    """Should this DEFCON-1 terminal be credited to the phasing player as a blunder?

    Split out of the step loop so it can be tested on a constructed state: a provoked ending is
    rare enough that random play never produces one, and the only policies that do reliably are
    checkpoints, which are git-ignored.
    """
    if int(state.defcon) > 1:
        return False
    return window_provoked or not state.has_flag(ts.EffectBits.DEFCON_SUICIDE_PROVOKED)


class TsVectorizedEnv:
    """High-throughput C++ vectorized batch environment executing N parallel games."""

    ACTION_SPACE_SIZE = 212

    @property
    def OBSERVATION_SIZE(self) -> int:
        return obs_size()

    def __init__(
        self,
        num_envs: int = 64,
        base_seed: int = 42,
        auto_reset: bool = True,
        start_provider: Optional[Callable[[int], Optional["ts.GameState"]]] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        window_provoked_defcon: bool = False,
    ):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.auto_reset = auto_reset
        self.observation_size = obs_size()
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
        # Credit a *provoked* DEFCON-1 to the player who played the card, the same way an
        # unprovoked one is credited. Off by default, because it changes the returns and so
        # every arm's numbers with it on are on a different footing from those without.
        # See metrics.md 21.1 for why it is worth trying: the critic registers at most 0.014
        # when the fatal card is chosen, so the -1 has nothing to attach to unless it is
        # windowed, and the window's advantage (-1 - v_t) never consults the critic.
        self.window_provoked_defcon = bool(window_provoked_defcon)
        self.reward_calc: RewardCalculator = reward_calculator or BlunderAwareRewardCalculator()
        self.runner = ts.VectorizedBatchRunner(num_envs, base_seed)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self.ep_rewards = np.zeros(num_envs, dtype=np.float32)

    def reset_all(self, base_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        if hasattr(self.reward_calc, "reset"):
            self.reward_calc.reset()
        if base_seed is not None:
            self.base_seed = base_seed
            self.runner = ts.VectorizedBatchRunner(self.num_envs, self.base_seed)
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
        # A provoked suicide stays 0 unless `window_provoked_defcon` is set. The argument for
        # leaving it at 0 was that the mistake lies in earlier card management rather than the
        # final move, so its credit should keep propagating backwards. Measured, that premise
        # does not hold: the fatal card play sits 3-9 micro-actions from the loss and inside the
        # same turn, which the turn-scoped window already covers, and the critic it would have
        # to propagate through moves by at most 0.014 at the deciding choice (metrics.md 21.1).
        defcon_blunder = np.zeros(self.num_envs, dtype=np.int8)
        ending_reasons: List[str] = [""] * self.num_envs
        # Terminal length in plies. The turn alone cannot express it: a game abandoned at
        # turn 7 AR1 and one that ran to turn 7 AR7 share a turn number, and a game that
        # went the distance reads 11 because finish_end_turn increments before testing.
        terminal_plies = np.zeros(self.num_envs, dtype=np.int16)
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
                elif credits_defcon_blunder(st, self.window_provoked_defcon):
                    # phasing_player is the side that played the card even when the opponent
                    # is the one acting -- verified on all five h2_480M_provoked_* replays.
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
                terminal_plies[i] = _game_ply(
                    int(st.turn), int(st.action_round), st.phasing_player == ts.Player.US)

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
                        "ply": int(terminal_plies[i]),
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
        info["turns"] = acting_turns
        info["held_scoring_us"] = held_scoring_us
        info["held_scoring_ussr"] = held_scoring_ussr
        info["defcon_blunder"] = defcon_blunder
        info["ending_reasons"] = ending_reasons
        info["terminal_turns"] = curr_turns
        info["terminal_plies"] = terminal_plies
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
