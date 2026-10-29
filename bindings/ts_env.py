"""TsVectorizedEnv: High-performance vectorized wrapper for ts::VectorizedBatchRunner."""

from typing import Tuple, Dict, Any, List, NamedTuple, Optional, Callable, Sequence
import numpy as np
import ts_engine as ts
from bindings.settle import SettleMode
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError
from ai.rewards.reward_calculator import RewardCalculator, BlunderAwareRewardCalculator
from ai.game_length import ply as _game_ply

# Canonical short keys for the game-ending reasons reported per completed episode.
# The long-form strings come from ts.game_ending_reason (ts::state_json::ending_reason, the
# one classifier, which tools.lib.game_ending.classify_game_ending_reason wraps).
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


def _classify_ending(state: ts.GameState, held_scoring: bool) -> str:
    """Maps a terminal state to one of ENDING_REASON_KEYS.

    Asks the engine directly; tools.lib.game_ending.classify_game_ending_reason is the same call.
    """
    reason = ts.game_ending_reason(state)
    if reason not in _ENDING_REASON_MAP:
        raise ValueError(f"the engine reported an ending reason this table does not know: {reason!r}; "
                         f"add it to _ENDING_REASON_MAP (and ENDING_REASON_KEYS)")
    key = _ENDING_REASON_MAP[reason]
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
    separate probes did exactly that, one of them reporting a mean final turn several times off
    the truth while looking entirely plausible.

    A model of the wrong width is a checkpoint from a retired layout: legacy (4293) or v2.1
    (3891) or v2.2 (3825). Those cannot be run and are not being converted -- they predate the
    starred-card fix, so they were trained against a different game.
    """
    width = int(getattr(model, "TOTAL_OBS_SIZE", 0) or 0)
    features = model_obs_features(model)
    expected = int(ts.obs_size_for(features))
    if width != expected:
        raise ValueError(
            f"this model reads {width} floats; the engine emits {expected} for its observation "
            f"feature set {features:#x} (layout v2.3 + appended blocks). A checkpoint from a "
            f"retired layout cannot be run: it would load cleanly and misread every input.")
    return width


def model_obs_features(model: Any) -> int:
    """The observation feature set (ts.OBS_FEATURE_* bits) a model reads: 0 for the base layout.

    Recorded in the weights (LadderNet's `obs_features` buffer), so it travels with the checkpoint.
    Every caller that builds observations for a model must ask for exactly this set -- `observe`
    below (`ts.extract_observation_features(state, player, model_obs_features(model))`), or
    `TsVectorizedEnv.set_obs_features` / `VectorizedBatchRunner.set_obs_features` for batches.

    The base functions (`ts.extract_observation`, a runner with no features set) are the set 0,
    and **nothing refuses them for a featured model**. A block that appends floats changes the
    width, which `check_obs_width` catches; `HEADLINE_REVEAL` appends none -- it changes the
    *content* of the base card rows (the box-4 holder sees the opponent's revealed headline), so
    a model trained with it and fed the base observation loads, runs and silently misreads.
    """
    return int(getattr(model, "obs_feature_bits", 0) or 0)


class AgentView(NamedTuple):
    """What an agent decides in: the observation it reads and the action view it masks in.

    `features` is its ts.OBS_FEATURE_* set (0 = the base layout), `merged_influence` whether it
    decides in the E4.1 merged-influence view (P23, tools/lib/action_view.py) rather than E4's.
    """
    features: int
    merged_influence: bool


def agent_view(agent: Any) -> AgentView:
    """The view an agent (or a bare network) decides in.

    An agent that declares `obs_features` (NeuralAgent, OnnxAgent) is taken at its word; one that
    carries a network in `.model` reads that network's set; a bare network (an `nn.Module`) its
    own. `merged_influence` is the agent's attribute. An agent with neither a declared set nor a
    model -- a rule-based bot -- acts on the state, never reads the observation, and decides in the
    E4 view: (0, False).

    A bare network cannot carry its action view (a snapshot file does not record it; the run
    directory does), so it always reports E4. A harness handed a bare network for a merged-view
    checkpoint must be told so by its caller (`tools.lib.action_view.checkpoint_merged_influence`).
    """
    declared = getattr(agent, "obs_features", None)
    if declared is not None:
        features = int(declared)
    else:
        model = getattr(agent, "model", None)
        features = model_obs_features(model if model is not None else agent)
    return AgentView(features, bool(getattr(agent, "merged_influence", False)))


def observe(state: "ts.GameState", player: "ts.Player", features: int) -> np.ndarray:
    """The observation `player` has of `state` in the feature set `features`, as float32.

    The one entry point harnesses use for a single position: always pass the deciding model's set
    (`model_obs_features(model)` / `agent_view(agent).features`), never the base layout by habit.
    """
    return np.asarray(ts.extract_observation_features(state, player, int(features)), dtype=np.float32)


def legal_mask(state: "ts.GameState", merged_influence: bool) -> np.ndarray:
    """The flat legal mask (uint8) of `state` in the given action view. The view is a required
    argument so that no harness builds an E4 mask for a merged-view network by default."""
    return np.asarray(ts.Engine.get_flat_action_mask(state, bool(merged_influence)), dtype=np.uint8)


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

    ACTION_SPACE_SIZE = ActionEncoder.FLAT_ACTION_SIZE

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
        settle: SettleMode = SettleMode.FORCED,
        game_seed: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.base_seed = base_seed
        # The seed of every game after each env's first. None (the default, every evaluation tool) draws it from the
        # global numpy RNG as before; a trainer passes its --seed-env, so the deals and dice follow that seed alone
        # rather than whatever else consumes the global stream, and saves the generator with its resume state.
        self._game_rng: Optional[np.random.Generator] = (
            None if game_seed is None else np.random.default_rng(int(game_seed)))
        self.auto_reset = auto_reset
        # Off unless a trainer wants the ownership target (P29 bet 2); see step().
        self.record_final_control = False
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
        # unprovoked one is credited. Off by default, because it changes the returns and so a
        # run's numbers with it on are on a different footing from those without.
        # Why it is worth trying: the critic barely moves when the fatal card is chosen, so the
        # -1 has nothing to attach to unless it is windowed, and the window's advantage
        # (-1 - v_t) never consults the critic.
        self.window_provoked_defcon = bool(window_provoked_defcon)
        # How far past a decision the player has no say in each step advances. FORCED is the
        # default: a decision with one legal action is not a decision, and handing it back costs
        # a Python round trip and a crossing of the binding boundary for nothing. Measured at
        # 31.2% fewer decisions per game, forced decisions from 6.1% to 0.0%, and 0.76x the
        # engine time.
        #
        # It is a flag rather than a constant because it moves the decision stream, so a run with
        # it is not comparable with a run without -- the same footing as window_provoked_defcon
        # above. The batched and single-state blunder probes must agree on it in particular:
        # ai/eval/blunders.py settles to match, and test_the_two_probes_agree_on_the_same_games
        # is what catches them drifting apart.
        #
        # NONE is not expressible here. The C++ runner always resolves chance nodes -- it never
        # exposes a ROLL_DIE at all -- so the only two reachable depths are CHANCE and FORCED.
        if settle is SettleMode.NONE:
            raise ValueError(
                "TsVectorizedEnv cannot present chance nodes: the C++ runner resolves them "
                "internally, so SettleMode.NONE has no meaning here. Use CHANCE or FORCED.")
        self.settle = settle
        self._auto_advance = settle is SettleMode.FORCED
        self.reward_calc: RewardCalculator = reward_calculator or BlunderAwareRewardCalculator()
        self.runner = ts.VectorizedBatchRunner(num_envs, base_seed)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self.ep_rewards = np.zeros(num_envs, dtype=np.float32)
        # P23 / E4.1: per env and side, whether that side decides in the merged-influence view.
        # Held here as well as in the runner because `reset_all(base_seed)` builds a new runner,
        # which would otherwise silently return every side to the E4 view.
        self.merged_us = np.zeros(num_envs, dtype=bool)
        self.merged_ussr = np.zeros(num_envs, dtype=bool)
        # The view spec's observation half: per env and side, the ts.OBS_FEATURE_* bits that
        # side's model reads (0 = the base layout). Held here for the same reason as the merged
        # flags -- a new runner on `reset_all(base_seed)` would otherwise drop them.
        self.obs_features_us = np.zeros(num_envs, dtype=np.uint32)
        self.obs_features_ussr = np.zeros(num_envs, dtype=np.uint32)
        # Whether the runner records the turn-end park observations (`runner.get_predeal`), held
        # for the same reason: `reset_all(base_seed)` builds a new runner.
        self.capture_predeal = False

    def set_capture_predeal(self, on: bool) -> None:
        """Record every env's turn-end park from both sides (`runner.get_predeal`); survives a
        `reset_all(base_seed)`, which a direct `runner.set_capture_predeal` call does not."""
        self.capture_predeal = bool(on)
        self.runner.set_capture_predeal(self.capture_predeal)

    def set_obs_features(self, us: np.ndarray | int, ussr: np.ndarray | int) -> None:
        """Every env's per-side observation feature set (an int applies to all envs). Each row of
        the observation is built in its decider's set and zero-padded to the widest set in use,
        so `observation_size` becomes that width; a model reads the first
        `ts.obs_size_for(its set)` floats of the rows where it decides."""
        self.obs_features_us[:] = us
        self.obs_features_ussr[:] = ussr
        self.runner.set_obs_features([int(x) for x in self.obs_features_us],
                                     [int(x) for x in self.obs_features_ussr])
        self.observation_size = int(self.runner.obs_width)

    def set_merged_influence(self, us: np.ndarray | bool, ussr: np.ndarray | bool) -> None:
        """Set every env's per-side view at once (a bool applies to all envs)."""
        self.merged_us[:] = us
        self.merged_ussr[:] = ussr
        self.runner.set_merged_influence([bool(x) for x in self.merged_us],
                                         [bool(x) for x in self.merged_ussr])

    def set_merged_influence_env(self, env_idx: int, us: bool, ussr: bool) -> None:
        """One env's per-side view; only that env's cached mask is rebuilt."""
        if bool(self.merged_us[env_idx]) == us and bool(self.merged_ussr[env_idx]) == ussr:
            return
        self.merged_us[env_idx] = us
        self.merged_ussr[env_idx] = ussr
        self.runner.set_merged_influence_env(env_idx, us, ussr)

    def reset_all(self, base_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        if hasattr(self.reward_calc, "reset"):
            self.reward_calc.reset()
        if base_seed is not None:
            self.base_seed = base_seed
            self.runner = ts.VectorizedBatchRunner(self.num_envs, self.base_seed)
            if self.merged_us.any() or self.merged_ussr.any():
                self.runner.set_merged_influence([bool(x) for x in self.merged_us],
                                                 [bool(x) for x in self.merged_ussr])
            if self.obs_features_us.any() or self.obs_features_ussr.any():
                self.runner.set_obs_features([int(x) for x in self.obs_features_us],
                                             [int(x) for x in self.obs_features_ussr])
            if self.capture_predeal:
                self.runner.set_capture_predeal(True)
        else:
            self.runner.refresh_all()
        injected = False
        for _i in range(self.num_envs):
            injected |= self._apply_start_position(_i)
        if injected:
            # Injected states replaced the ones the runner had just refreshed. The refresh above
            # used to run before the injection and left those envs with stale buffers.
            self.runner.refresh_all()
        self.ep_lengths.fill(0)
        self.ep_rewards.fill(0.0)
        if hasattr(self.reward_calc, "on_all_reset"):
            self.reward_calc.on_all_reset([self.runner.get_state(i) for i in range(self.num_envs)])
        obs = np.array(self.runner.get_observations(), copy=False)
        masks = np.array(self.runner.get_action_masks(), copy=False)
        return obs, masks, self._get_batch_info()

    def _apply_start_position(self, env_idx: int) -> bool:
        """Inject a start position into a freshly reset env. True when the state was replaced.

        `set_state` does not rebuild the runner's cached observation and mask (`reset_game` and
        `step_flat_all` do, per env), so a caller that gets True must refresh before reading them.
        """
        self.env_start_turn[env_idx] = 1
        if self.start_provider is None:
            return False
        state = self.start_provider(env_idx)
        if state is None:
            return False
        self.runner.set_state(env_idx, state)
        # Pool positions are stored pre-deal, one turn before the turn they target, so the
        # episode's effective start turn is the next one. A turn-1 state is a scripted opening
        # (--forced-opening): the pool never stores one, and that episode starts on turn 1.
        self.env_start_turn[env_idx] = 1 if int(state.turn) <= 1 else int(state.turn) + 1
        return True

    def next_game_seed(self) -> int:
        """The seed of the next game: from the env's own generator when it has one, else the global numpy RNG."""
        if self._game_rng is not None:
            return int(self._game_rng.integers(1, 1_000_000_000))
        return int(np.random.randint(1, 1_000_000_000))

    def game_rng_state(self) -> Optional[Dict[str, Any]]:
        """The game-seed generator's state, for a resume file; None without one."""
        return None if self._game_rng is None else dict(self._game_rng.bit_generator.state)

    def set_game_rng_state(self, state: Dict[str, Any]) -> None:
        if self._game_rng is None:
            raise ValueError("this env draws game seeds from the global numpy RNG; it has no generator to restore")
        self._game_rng.bit_generator.state = state

    def reseed_games(self, seed: Sequence[int] | int) -> None:
        """A fresh game-seed generator (a resume without a saved one keys it by the seed and the step)."""
        self._game_rng = np.random.default_rng(seed)

    def reset_env(self, env_idx: int, seed: Optional[int] = None) -> None:
        """Reset a single environment by index."""
        s = seed if seed is not None else self.next_game_seed()
        self.runner.reset_game(env_idx, s)
        if self._apply_start_position(env_idx):
            self.runner.refresh_all()
        self.ep_lengths[env_idx] = 0
        self.ep_rewards[env_idx] = 0.0

    def step(self, actions: np.ndarray | List[int],
             hold: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all N environments in parallel with exact acting player attribution.

        `hold` (bool per env) leaves those games exactly where they are for this step -- P40 Arm S's
        searched opponent, whose games wait at a searched decision until enough are queued for one
        root call. A held env is sent a card slot its mask refuses, which the engine rejects before
        touching the game -- at a SELECT_CARD decision by the mask, anywhere else by the decision-type
        check (`StateMachine::step`; the one card-shaped exception, UN Intervention at Grain Sales'
        play-mode node, is still decided by the mask). The observation, the RNG and the mask are
        unchanged; only the display-only last-roll record is cleared, as every step clears it. A held
        env that is NOT refused, or a refusal anywhere else, raises. Its reward is 0, it is never
        done, and the step does not count toward its episode length."""
        # 1. Capture exact acting players & privileged opponent hands BEFORE stepping simulation
        # The masks the caller sampled from, captured before the step overwrites the buffer --
        # needed to tell a mask/engine disagreement from a sampler that ignored the mask.
        #
        # This is the CACHED buffer, which is what a batched caller actually sampled from. It can
        # be stale if someone replaced a state with `runner.set_state` without refreshing (writing
        # through `get_state` once did it too; that returns a copy now), and then the diagnostic
        # below would claim an action "WAS legal in the mask" on the strength of a mask that no
        # longer describes the position -- which cost real time once. The failure path re-derives
        # a fresh mask and reports both.
        prev_masks = np.array(self.runner.get_action_masks(), copy=True)
        acting_players = np.array(self.runner.get_decision_players(), dtype=np.int8)
        acting_turns = np.array(self.runner.get_turns(), dtype=np.int8)
        prev_vp = np.array(self.runner.get_victory_points(), dtype=np.int8)

        # 2. Step C++ simulation in parallel
        action_list = [int(a) for a in actions]
        held_idx = np.flatnonzero(hold) if hold is not None else np.zeros(0, dtype=np.int64)
        for raw in held_idx:
            action_list[int(raw)] = self._refused_card_slot(int(raw), prev_masks[int(raw)])

        step_results = self.runner.step_flat_all(action_list, self._auto_advance)
        if len(held_idx):
            not_refused = [int(i) for i in held_idx if step_results[int(i)] != 0]
            if not_refused:
                raise IllegalActionError(
                    f"held envs {not_refused[:8]} were stepped: the engine accepted the refused card slot "
                    f"{[action_list[i] for i in not_refused[:8]]}, so the hold moved the game")
            step_results = list(step_results)
            for raw in held_idx:
                step_results[int(raw)] = 1
        # 0 = the engine refused the action. Never observed (0 refusals in 5,052 batched steps),
        # but if it happens the rollout is being built from a game that did not advance, so the
        # data is wrong rather than merely late. `0 in list` is a C-level scan with no allocation;
        # converting to numpy to ask the same question cost ~29% of the step call.
        if 0 in step_results:
            bad = step_results.index(0)
            act = int(action_list[bad])
            st = self.runner.get_state(bad)
            # Which of the two failures this is, because they need opposite fixes:
            #   in the mask  -> the mask and the engine disagree, an engine defect;
            #   not in mask  -> the sampler picked a masked-out action, a trainer defect.
            # Without this the message names neither, and the first occurrence cost a run.
            was_legal = bool(0 <= act < len(prev_masks[bad]) and prev_masks[bad][act])
            n_legal = int(prev_masks[bad].sum())
            # In the view of whoever acted: a composed E4.1 action is not in the E4 mask.
            actor_merged = bool(self.merged_us[bad] if int(acting_players[bad]) == int(ts.Player.US)
                                else self.merged_ussr[bad])
            fresh = np.asarray(ActionEncoder.get_legal_mask(st, actor_merged))
            fresh_legal = bool(0 <= act < len(fresh) and fresh[act])
            stale = int((fresh != prev_masks[bad]).sum())
            raise IllegalActionError(
                f"engine refused flat action {act} in env {bad} of {self.num_envs} "
                f"(decision {ts.DecisionType(int(st.ctx().decision_type))}); "
                f"{step_results.count(0)} of the batch refused. "
                f"The action was legal in the CACHED mask; freshly generated it is "
                f"{'legal' if fresh_legal else 'NOT legal'}"
                + (f", and the cache is stale on {stale} entries -- a caller replaced the state "
                   f"(runner.set_state) without refreshing" if stale else "")
                if was_legal else
                f"engine refused flat action {act} in env {bad} of {self.num_envs} "
                f"(decision {ts.DecisionType(int(st.ctx().decision_type))}); "
                f"{step_results.count(0)} of the batch refused. "
                f"The action was NOT in the mask ({n_legal} legal there), so the sampler "
                f"chose outside it -- a uniform softmax over an all -1e9 row does exactly "
                f"this. phase={st.current_phase} vp={int(st.victory_points)} "
                f"terminal={ts.Engine.is_terminal(st)}")

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
        # does not hold: the fatal card play sits a handful of micro-actions from the loss and
        # inside the same turn, which the turn-scoped window already covers, and the critic it
        # would have to propagate through barely moves at the deciding choice.
        defcon_blunder = np.zeros(self.num_envs, dtype=np.int8)
        ending_reasons: List[str] = [""] * self.num_envs
        # Terminal length in plies. The turn alone cannot express it: a game abandoned at
        # turn 7 AR1 and one that ran to turn 7 AR7 share a turn number, and a game that
        # went the distance reads 11 because finish_end_turn increments before testing.
        terminal_plies = np.zeros(self.num_envs, dtype=np.int16)
        # P29 bet 2's ownership target: who controls each country when the game ends, read here
        # because the auto-reset below discards the terminal state. Only when a trainer asks for
        # it (record_final_control), since it costs 84 engine calls per finished game.
        final_control: Dict[int, np.ndarray] = {}
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
                    # is the one acting -- verified against self-play games that ended this way.
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
                    int(st.turn), int(st.action_round), st.phasing_player == ts.Player.US,
                    headline_stage=int(st.headline_stage))
                if self.record_final_control:
                    # +1 US, -1 USSR, 0 neither (ts.Player's own values).
                    final_control[i] = np.array(
                        [int(ts.Scoring.get_country_control(st, c)) for c in range(84)], dtype=np.int8)

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

        if len(held_idx):
            rewards[held_idx] = 0.0
        self.ep_lengths += 1
        self.ep_lengths[held_idx] -= 1
        self.ep_rewards += rewards

        completed_episodes: List[Dict[str, Any]] = []
        injected = False
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
                    if i in final_control:
                        completed_episodes[-1]["final_control"] = final_control[i]
                    new_seed = self.next_game_seed()
                    self.runner.reset_game(i, new_seed)
                    # Inject before on_env_reset so the reward calculator sees the position
                    # the episode actually begins from, not the discarded fresh deal.
                    injected |= self._apply_start_position(i)
                    if hasattr(self.reward_calc, "on_env_reset"):
                        self.reward_calc.on_env_reset(i, self.runner.get_state(i))
                    self.ep_lengths[i] = 0
                    self.ep_rewards[i] = 0.0

            # `reset_game` and `step_flat_all` already rebuild every env they touch, so a refresh is
            # needed only where `set_state` injected a start position. Refreshing all 512 envs
            # whenever any game ended cost ~4.4 ms of single-threaded observation building per
            # step (research/log/training_throughput_cpu.md).
            if injected:
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

    def _refused_card_slot(self, env_idx: int, mask: np.ndarray) -> int:
        """A card slot the engine will refuse at env `env_idx`'s current decision: the first card its
        mask leaves out (a hand never holds all 110, and outside a card choice none is legal)."""
        off = np.flatnonzero(np.asarray(mask[ActionEncoder.CARD_OFFSET:ActionEncoder.PLAY_MODE_OFFSET]) == 0)
        if not off.size:
            raise ValueError(f"env {env_idx}: every card slot is legal, so none can be refused")
        return int(ActionEncoder.CARD_OFFSET + off[0])

    def observations_for(self, players: np.ndarray) -> np.ndarray:
        """Observations of the CURRENT states, each from the perspective named in `players`.

        `get_observations()` always uses the acting player, which is the only perspective the C++
        runner knows how to produce: `refresh_single` reads `ctx().decision_player`, else
        `phasing_player`. Evaluating a state from the side that just moved therefore has to go
        through `extract_observation` per environment.

        Measured at 512 environments: 4.61 ms here against 0.84 ms for the batched C++ call, and
        about 7% of the 61.7 ms an end-to-end training step takes -- roughly half that in practice,
        since it is only needed where the mover changed. A batched C++ entry point taking a
        per-environment perspective is the version worth having if this proves out.

        `players` uses the engine's coding: +1 for US, -1 for USSR. Each row is built in the
        observation feature set `set_obs_features` gave that env and side, and is
        `observation_size` wide like the batched rows.
        """
        # Each row in the feature set of the side it is read for (the view spec), zero-padded to
        # the runner's width exactly as `get_observations` pads: a featured model fed the base
        # layout misreads without failing (HEADLINE_REVEAL changes content, not width).
        out = np.zeros((self.num_envs, int(self.observation_size)), dtype=np.float32)
        for i in range(self.num_envs):
            us = int(players[i]) > 0
            p = ts.Player.US if us else ts.Player.USSR
            feats = int(self.obs_features_us[i] if us else self.obs_features_ussr[i])
            row = observe(self.runner.get_state(i), p, feats)
            out[i, :row.shape[0]] = row
        return out

    def _get_batch_info(self) -> Dict[str, Any]:
        return {
            "decision_players": np.array(self.runner.get_decision_players(), dtype=np.int8),
            "turns": np.array(self.runner.get_turns(), dtype=np.int8),
            "victory_points": np.array(self.runner.get_victory_points(), dtype=np.int8),
            "terminals": np.array(self.runner.get_terminals(), dtype=bool),
            "terminal_utilities": np.array(self.runner.get_terminal_utilities(), dtype=np.float32),
        }
