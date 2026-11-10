"""Twilight Struggle C++ Simulation Engine Python Bindings (nanobind)"""

from collections.abc import Sequence
import enum
from typing import Annotated, Final, TypedDict, overload

import numpy
from numpy.typing import NDArray

from ts_engine import EffectBits as EffectBits



class CountryInfo(TypedDict):
    """Static map data for one country, as `MapData.get_country_info` returns it.

    A TypedDict rather than a bare `dict` so a checker sees the keys and their types: a
    misspelled key and a field used at the wrong type are both errors against this.

    Note `region` is the `Region` enum, not an int. It is bound with nb::is_arithmetic(),
    so int(), `== 0`, numpy conversion and sorting all still work -- but comparing it to a
    `Region` member works too, which against a bare int was silently always False.
    """

    id: int
    name: str
    stability: int
    battleground: bool
    region: Region
    in_western_europe: bool
    in_eastern_europe: bool
    in_southeast_asia: bool
    superpower_adjacent: str
    neighbors: list[int]

class RollType(enum.IntEnum):
    NONE = 0

    COUP = 1

    REALIGNMENT = 2

    SPACE_RACE = 3

    WAR_EVENT = 4

    OLYMPIC_GAMES = 5

    SUMMIT = 6

    TRAP_ESCAPE = 7

    TURN_CLEANUP = 8

NONE: DecisionType = DecisionType.NONE

COUP: OpMode = OpMode.COUP

REALIGNMENT: RollType = RollType.REALIGNMENT

SPACE_RACE: RollType = RollType.SPACE_RACE

WAR_EVENT: RollType = RollType.WAR_EVENT

OLYMPIC_GAMES: RollType = RollType.OLYMPIC_GAMES

SUMMIT: RollType = RollType.SUMMIT

TRAP_ESCAPE: RollType = RollType.TRAP_ESCAPE

TURN_CLEANUP: RollType = RollType.TURN_CLEANUP

class Player(enum.IntEnum):
    NONE = 0

    US = 1

    USSR = -1

US: Player = Player.US

USSR: Player = Player.USSR

class Phase(enum.IntEnum):
    SETUP = 0

    HEADLINE = 1

    ACTION_ROUND = 2

    INTERRUPT = 3

    DISCARD = 4

    END_TURN = 5

    GAME_OVER = 6

SETUP: Phase = Phase.SETUP

HEADLINE: Phase = Phase.HEADLINE

ACTION_ROUND: Phase = Phase.ACTION_ROUND

INTERRUPT: Phase = Phase.INTERRUPT

DISCARD: Phase = Phase.DISCARD

END_TURN: Phase = Phase.END_TURN

GAME_OVER: Phase = Phase.GAME_OVER

class DecisionType(enum.IntEnum):
    NONE = 0

    SELECT_CARD = 1

    SELECT_PLAY_MODE = 2

    CHOOSE_TIMING_BRANCH = 3

    SELECT_OP_MODE = 4

    POINT_NODE = 5

    CHOOSE_BRANCH = 6

    ROLL_DIE = 7

SELECT_CARD: DecisionType = DecisionType.SELECT_CARD

SELECT_PLAY_MODE: DecisionType = DecisionType.SELECT_PLAY_MODE

CHOOSE_TIMING_BRANCH: DecisionType = DecisionType.CHOOSE_TIMING_BRANCH

SELECT_OP_MODE: DecisionType = DecisionType.SELECT_OP_MODE

POINT_NODE: DecisionType = DecisionType.POINT_NODE

CHOOSE_BRANCH: DecisionType = DecisionType.CHOOSE_BRANCH

ROLL_DIE: DecisionType = DecisionType.ROLL_DIE

class Resolution(enum.IntEnum):
    EVENT = 0

    SPACE = 1

    OPS_INFLUENCE = 2

    OPS_COUP = 3

    OPS_REALIGN = 4

    COUNT = 5

EVENT: Resolution = Resolution.EVENT

SPACE: Resolution = Resolution.SPACE

OPS_INFLUENCE: Resolution = Resolution.OPS_INFLUENCE

OPS_COUP: Resolution = Resolution.OPS_COUP

OPS_REALIGN: Resolution = Resolution.OPS_REALIGN

COUNT: Resolution = Resolution.COUNT

class TimingBranch(enum.IntEnum):
    OPS_FIRST = 0

    EVENT_FIRST = 1

OPS_FIRST: TimingBranch = TimingBranch.OPS_FIRST

EVENT_FIRST: TimingBranch = TimingBranch.EVENT_FIRST

class OpMode(enum.IntEnum):
    INFLUENCE = 0

    COUP = 1

    REALIGN = 2

INFLUENCE: OpMode = OpMode.INFLUENCE

REALIGN: OpMode = OpMode.REALIGN

def anomalies_reported() -> int:
    """
    How many anomalies (ts::report_anomaly: a state that should not arise, logged to stderr) this thread has reported. A caller that must not continue past one reads it before and after a step.
    """

def set_mask_cache_checks(on: bool) -> None:
    """
    Check every cached legal mask against a fresh one (aborting on a difference).
    """

def mask_cache_checks() -> bool: ...

def reveal_hand(state: GameState, player: Player) -> None:
    """
    Mark every card that player is holding right now as public to the opponent. Cards drawn afterwards are hidden again -- knowledge attaches to cards, not to players.
    """

def reveal_both_hands(state: GameState) -> None:
    """
    Mark both hands public, as when the draw deck runs out and each side can name the other's hand as the complement of what it can see.
    """

def in_hand_of(location: CardLocation, player: Player) -> bool:
    """
    True when the card is in that player's hand, known to the opponent or not.
    """

def known_to_opponent(location: CardLocation) -> bool:
    """
    True when the player who is not holding the card knows it is in that hand.
    """

def hand_of(player: Player, known: bool = False) -> CardLocation:
    """
    The hand location for a player; hidden from the opponent unless known=True.
    """

def hand_holder(location: CardLocation) -> Player:
    """Whose hand it is, or Player.NONE when the card is not in one."""

def revealed(location: CardLocation) -> CardLocation:
    """The same hand, marked public. Identity for anything not in a hand."""

class CardLocation(enum.IntEnum):
    UNAVAILABLE = 0

    DRAW_DECK = 1

    HAND_US_UNKNOWN = 2

    HAND_US_KNOWN = 3

    HAND_USSR_UNKNOWN = 4

    HAND_USSR_KNOWN = 5

    DISCARD_PILE = 6

    REMOVED_FROM_GAME = 7

    ONGOING_EVENT = 8

    PEEKED_TEMP = 9

    HEADLINE_COMMITTED = 10

UNAVAILABLE: CardLocation = CardLocation.UNAVAILABLE

DRAW_DECK: CardLocation = CardLocation.DRAW_DECK

HAND_US_UNKNOWN: CardLocation = CardLocation.HAND_US_UNKNOWN

HAND_US_KNOWN: CardLocation = CardLocation.HAND_US_KNOWN

HAND_USSR_UNKNOWN: CardLocation = CardLocation.HAND_USSR_UNKNOWN

HAND_USSR_KNOWN: CardLocation = CardLocation.HAND_USSR_KNOWN

DISCARD_PILE: CardLocation = CardLocation.DISCARD_PILE

REMOVED_FROM_GAME: CardLocation = CardLocation.REMOVED_FROM_GAME

ONGOING_EVENT: CardLocation = CardLocation.ONGOING_EVENT

PEEKED_TEMP: CardLocation = CardLocation.PEEKED_TEMP

HEADLINE_COMMITTED: CardLocation = CardLocation.HEADLINE_COMMITTED

class WarEra(enum.IntEnum):
    EARLY = 0

    MID = 1

    LATE = 2

EARLY: WarEra = WarEra.EARLY

MID: WarEra = WarEra.MID

LATE: WarEra = WarEra.LATE

class Region(enum.IntEnum):
    EUROPE = 0

    ASIA = 1

    MIDDLE_EAST = 2

    AFRICA = 3

    CENTRAL_AMERICA = 4

    SOUTH_AMERICA = 5

    NONE_REGION = 255

EUROPE: Region = Region.EUROPE

ASIA: Region = Region.ASIA

MIDDLE_EAST: Region = Region.MIDDLE_EAST

AFRICA: Region = Region.AFRICA

CENTRAL_AMERICA: Region = Region.CENTRAL_AMERICA

SOUTH_AMERICA: Region = Region.SOUTH_AMERICA

NONE_REGION: Region = Region.NONE_REGION

class MicroAction:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, decision_type: DecisionType, primary_id: int = 0, secondary_id: int = 0, flags: int = 0) -> None: ...

    @property
    def decision_type(self) -> DecisionType: ...

    @decision_type.setter
    def decision_type(self, arg: DecisionType, /) -> None: ...

    @property
    def primary_id(self) -> int: ...

    @primary_id.setter
    def primary_id(self, arg: int, /) -> None: ...

    @property
    def secondary_id(self) -> int: ...

    @secondary_id.setter
    def secondary_id(self, arg: int, /) -> None: ...

    @property
    def flags(self) -> int: ...

    @flags.setter
    def flags(self, arg: int, /) -> None: ...

    def is_confirm_done(self) -> bool: ...

    def __repr__(self) -> str: ...

class CountryState:
    @property
    def us_influence(self) -> int: ...

    @property
    def ussr_influence(self) -> int: ...

class DecisionContext:
    @property
    def decision_player(self) -> Player: ...

    @decision_player.setter
    def decision_player(self, arg: Player, /) -> None: ...

    @property
    def decision_type(self) -> DecisionType: ...

    @decision_type.setter
    def decision_type(self, arg: DecisionType, /) -> None: ...

    @property
    def pending_op_card(self) -> int: ...

    @pending_op_card.setter
    def pending_op_card(self, arg: int, /) -> None: ...

    @property
    def pending_ops_value(self) -> int: ...

    @pending_ops_value.setter
    def pending_ops_value(self, arg: int, /) -> None: ...

    @property
    def remaining_steps(self) -> int: ...

    @remaining_steps.setter
    def remaining_steps(self, arg: int, /) -> None: ...

    @property
    def max_per_country(self) -> int: ...

    @max_per_country.setter
    def max_per_country(self, arg: int, /) -> None: ...

    @property
    def allow_early_stop(self) -> int: ...

    @allow_early_stop.setter
    def allow_early_stop(self, arg: int, /) -> None: ...

    @property
    def resolving_card(self) -> int: ...

    @resolving_card.setter
    def resolving_card(self, arg: int, /) -> None: ...

    @property
    def op_mode(self) -> OpMode: ...

    @op_mode.setter
    def op_mode(self, arg: OpMode, /) -> None: ...

    @property
    def pending_roll_type(self) -> RollType: ...

    @property
    def roll_target(self) -> int: ...

    @property
    def roll_actor(self) -> Player: ...

    @property
    def event_stage(self) -> int: ...

    @event_stage.setter
    def event_stage(self, arg: int, /) -> None: ...

    def is_visited(self, arg: int, /) -> bool: ...

    def node_count(self, arg: int, /) -> int: ...

    NODE_COUNT_MAX: Final[int] = ...
    """(arg: object, /) -> int"""

class GameState:
    def __init__(self) -> None: ...

    def clone(self) -> GameState: ...

    @property
    def victory_points(self) -> int: ...

    @victory_points.setter
    def victory_points(self, arg: int, /) -> None: ...

    @property
    def defcon(self) -> int: ...

    @defcon.setter
    def defcon(self, arg: int, /) -> None: ...

    @property
    def us_mil_ops(self) -> int: ...

    @us_mil_ops.setter
    def us_mil_ops(self, arg: int, /) -> None: ...

    @property
    def ussr_mil_ops(self) -> int: ...

    @ussr_mil_ops.setter
    def ussr_mil_ops(self, arg: int, /) -> None: ...

    @property
    def us_space_track(self) -> int: ...

    @us_space_track.setter
    def us_space_track(self, arg: int, /) -> None: ...

    @property
    def ussr_space_track(self) -> int: ...

    @ussr_space_track.setter
    def ussr_space_track(self, arg: int, /) -> None: ...

    @property
    def turn(self) -> int: ...

    @turn.setter
    def turn(self, arg: int, /) -> None: ...

    @property
    def action_round(self) -> int: ...

    @action_round.setter
    def action_round(self, arg: int, /) -> None: ...

    @property
    def defcon_dropped_to_2(self) -> int: ...

    @defcon_dropped_to_2.setter
    def defcon_dropped_to_2(self, arg: int, /) -> None: ...

    @property
    def phasing_player(self) -> Player: ...

    @phasing_player.setter
    def phasing_player(self, arg: Player, /) -> None: ...

    @property
    def headline_us_card(self) -> int: ...

    @headline_us_card.setter
    def headline_us_card(self, arg: int, /) -> None: ...

    @property
    def headline_ussr_card(self) -> int: ...

    @headline_ussr_card.setter
    def headline_ussr_card(self, arg: int, /) -> None: ...

    @property
    def headline_first_card(self) -> int: ...

    @headline_first_card.setter
    def headline_first_card(self, arg: int, /) -> None: ...

    @property
    def headline_second_card(self) -> int: ...

    @headline_second_card.setter
    def headline_second_card(self, arg: int, /) -> None: ...

    @property
    def headline_stage(self) -> int: ...

    @headline_stage.setter
    def headline_stage(self, arg: int, /) -> None: ...

    @property
    def current_phase(self) -> Phase: ...

    @current_phase.setter
    def current_phase(self, arg: Phase, /) -> None: ...

    @property
    def forced_card_player(self) -> Player: ...

    @forced_card_player.setter
    def forced_card_player(self, arg: Player, /) -> None: ...

    @property
    def forced_card_id(self) -> int: ...

    @forced_card_id.setter
    def forced_card_id(self, arg: int, /) -> None: ...

    @property
    def china_card_holder(self) -> Player: ...

    @china_card_holder.setter
    def china_card_holder(self, arg: Player, /) -> None: ...

    @property
    def china_card_playable(self) -> int: ...

    @china_card_playable.setter
    def china_card_playable(self, arg: int, /) -> None: ...

    @property
    def persistent_effects(self) -> int: ...

    @persistent_effects.setter
    def persistent_effects(self, arg: int, /) -> None: ...

    @property
    def ctx_stack_depth(self) -> int: ...

    @ctx_stack_depth.setter
    def ctx_stack_depth(self, arg: int, /) -> None: ...

    @property
    def rng_state(self) -> int: ...

    @rng_state.setter
    def rng_state(self, arg: int, /) -> None: ...

    def seat_obs_features(self, player: Player, features: int) -> None:
        """
        Seat a model of feature set `features` at `player`: that seat plays the rule bits the set names and the pre-fix rule of each it lacks. Observation bits are ignored.
        """

    def seat_plays_rule(self, player: Player, rule: int) -> bool:
        """
        Whether the seat `player` plays rule bit `rule` (the fix) rather than the rule before it.
        """

    def set_flag(self, arg: int, /) -> None: ...

    def clear_flag(self, arg: int, /) -> None: ...

    def ctx(self) -> DecisionContext: ...

    def has_flag(self, arg: int, /) -> bool: ...

    def get_country(self, arg: int, /) -> CountryState: ...

    def set_country(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    def get_space_turns_used(self, arg: Player, /) -> int: ...

    def set_space_turns_used(self, arg0: Player, arg1: int, /) -> None: ...

    def set_forced_deal(self, player: Player, cards: Sequence[int]) -> None:
        """
        Name the cards the NEXT deal gives this player, for re-driving a recording.

        A die roll can already be forced through its own action; a deal cannot, because a
        deal is not a decision. Without this a replay is not self-contained: re-driving it
        on an engine whose shuffle or draw order changed diverges at the first deal,
        silently, producing a different game rather than an error.

        Per player on purpose -- what is recorded is WHICH CARDS each side received, so a
        recording survives a change to the deal algorithm and not merely to the shuffle.
        Consumed by one deal and then cleared; a source driving several refills before
        each. Empty in normal play, which costs one comparison per draw. A named card that
        is not in the draw deck when the deal reaches it is reported as an anomaly and that
        draw falls back to the RNG, rather than being silently mis-dealt.
        """

    def get_forced_deal_remaining(self, player: Player) -> int:
        """
        How many named cards this player's queue still holds. 0 in normal play.
        """

    def record_space_attempt(self, arg: Player, /) -> None: ...

    def get_card_location(self, arg: int, /) -> CardLocation: ...

    def set_card_location(self, arg0: int, arg1: CardLocation, /) -> None: ...

    def to_dict(self) -> dict: ...

    def to_save_dict(self) -> dict:
        """
        Named-field save of a position, mid-game included: scalars, RNG, card locations, board influence, the headline owners, the die-roll record AND the decision-context stack (ctx_stack + depth). Round-trips with state_from_save_dict -- verified over every position of whole games, comparing the pending decision, the legal mask and the observation. Does NOT carry action_history or turn_aggregates, which nothing reads for rules or for the observation; they are diagnostics.
        """

    def to_save_json(self) -> str:
        """
        The save as canonical JSON text: keys sorted, compact -- the same bytes as json.dumps(self.to_save_dict(), sort_keys=True, separators=(',', ':')). Written in C++ and shared with the WebAssembly engine, so a position saved by either opens in the other.
        """

    def to_display_json(self) -> str:
        """to_dict() as JSON text, as the browser workbench's engine produces it."""

    def to_json(self) -> str: ...

    def raw_bytes(self) -> bytes:
        """
        The GameState's memory, for exact equality tests. It is trivially copyable, so two states reached from copies of one origin by the same steps compare equal byte for byte (padding included) -- which is how P23 checks a composed step against the two E4 steps.
        """

class StateMachine:
    @staticmethod
    def advance_headline_step(arg: GameState, /) -> None: ...

    @staticmethod
    def advance_after_action_round(arg: GameState, /) -> None: ...

    @staticmethod
    def deal_cards_to_hands(arg: GameState, /) -> None: ...

    @staticmethod
    def reshuffle_discard_into_draw(arg: GameState, /) -> None: ...

class Engine:
    @staticmethod
    def init_game(arg0: GameState, arg1: int, /) -> None: ...

    @staticmethod
    def try_step(state: GameState, action: MicroAction, auto_advance: bool = False) -> bool:
        """
        Advance one action, returning False if the engine refuses it. For probing only.
        """

    @staticmethod
    def step(state: GameState, action: MicroAction, auto_advance: bool = False) -> None:
        """Advance one action, raising RuntimeError if the engine refuses it."""

    @staticmethod
    def is_terminal(arg: GameState, /) -> bool: ...

    @staticmethod
    def get_terminal_utility(arg: GameState, /) -> float: ...

    @staticmethod
    def has_held_scoring_card(arg0: GameState, arg1: Player, /) -> bool: ...

    @staticmethod
    def is_held_scoring_game_over(arg: GameState, /) -> bool: ...

    @staticmethod
    def is_held_scoring_loss(arg0: GameState, arg1: Player, /) -> bool: ...

    @staticmethod
    def get_legal_action_mask(arg: GameState, /) -> list: ...

    @staticmethod
    def get_legal_action_indices(arg: GameState, /) -> list: ...

    @staticmethod
    def get_flat_action_mask(state: GameState, merged_influence: bool = False, macro_influence: bool = False) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]:
        """
        The flat legal mask. merged_influence=True is the E4.1 view (P23): at an op-choice node, NODE slot X means 'influence, first point in X'. macro_influence=True is P42's: at a POINT_NODE, MACRO_NODE + c means 'place in c until control', offered where it differs from one point.
        """

    @staticmethod
    def try_step_flat(state: GameState, action_idx: int, auto_advance: bool = False, merged_influence: bool = False, macro_influence: bool = False) -> bool:
        """
        Advance one flat action, returning False if the engine refuses it. For probing only.
        """

    @staticmethod
    def step_flat(state: GameState, action_idx: int, auto_advance: bool = False, merged_influence: bool = False, macro_influence: bool = False) -> None:
        """
        Advance one flat action, raising RuntimeError if the engine refuses it. merged_influence=True applies a composed E4.1 action as the two E4 steps it names; macro_influence=True applies an until-control slot as the one-point steps that define it.
        """

    @staticmethod
    def auto_advance_step(state: GameState, max_steps: int = 128) -> int: ...

class MapData:
    @staticmethod
    def get_country_name(arg: int, /) -> str: ...

    @staticmethod
    def get_country_by_name(arg: str, /) -> int: ...

    @staticmethod
    def placement_rank(arg: int, /) -> int:
        """
        P43: the rank a canonical placement run (CANONICAL_SETUP) follows, a permutation of 0..83, most important last.
        """

    @staticmethod
    def get_country_info(arg: int, /) -> CountryInfo:
        """Static map data for one country. See `CountryInfo`."""

class CardHandlers:
    @staticmethod
    def trigger_event(state: GameState, card_id: int, player: Player) -> bool: ...

    @staticmethod
    def can_trigger_event(state: GameState, card_id: int, player: Player) -> bool: ...

    @staticmethod
    def handle_event_step(arg0: GameState, arg1: MicroAction, /) -> bool: ...

class RegionalStatus(enum.IntEnum):
    NONE = 0

    PRESENCE = 1

    DOMINATION = 2

    CONTROL = 3

class RegionScoreSummary:
    @property
    def us_status(self) -> RegionalStatus: ...

    @property
    def ussr_status(self) -> RegionalStatus: ...

    @property
    def us_countries(self) -> int: ...

    @property
    def ussr_countries(self) -> int: ...

    @property
    def us_battlegrounds(self) -> int: ...

    @property
    def ussr_battlegrounds(self) -> int: ...

    @property
    def us_superpower_adjacent(self) -> int: ...

    @property
    def ussr_superpower_adjacent(self) -> int: ...

    @property
    def us_score(self) -> int: ...

    @property
    def ussr_score(self) -> int: ...

    @property
    def net_delta(self) -> int: ...

    @property
    def ussr_countries_scored(self) -> int: ...

    @property
    def ussr_battlegrounds_scored(self) -> int: ...

    @property
    def ussr_superpower_adjacent_scored(self) -> int: ...

    @property
    def shuttle_diplomacy_applied(self) -> bool: ...

class Operations:
    @staticmethod
    def can_place_influence(arg0: GameState, arg1: Player, arg2: int, /) -> bool: ...

class Scoring:
    @staticmethod
    def score_region(arg0: GameState, arg1: Region, /) -> None: ...

    @staticmethod
    def score_southeast_asia(arg: GameState, /) -> None: ...

    @staticmethod
    def execute_final_scoring(arg: GameState, /) -> None: ...

    @staticmethod
    def evaluate_military_ops(arg: GameState, /) -> None: ...

    @staticmethod
    def evaluate_region(state: GameState, region: Region, is_final_scoring: bool = False) -> RegionScoreSummary: ...

    @staticmethod
    def get_country_control(arg0: GameState, arg1: int, /) -> Player: ...

    @staticmethod
    def is_controlled_by(arg0: GameState, arg1: int, arg2: Player, /) -> bool: ...

    @staticmethod
    def compute_useful_actions_potential(arg0: GameState, arg1: Player, /) -> float: ...

class CardData:
    @staticmethod
    def get_card_name(arg: int, /) -> str: ...

    @staticmethod
    def get_card_by_name(arg: str, /) -> int: ...

    @staticmethod
    def get_card_info(arg: int, /) -> dict: ...

def state_to_dict(arg: GameState, /) -> dict:
    """Convert GameState to Python dictionary"""

def state_from_save_dict(arg: dict, /) -> GameState:
    """
    Rebuild a position from to_save_dict output, mid-game included. Missing keys take their default, so a save written before a field existed still loads, and an unknown key is ignored, so a save from a newer build opens minus what it cannot use. Restores the decision-context stack; does not restore action_history or turn_aggregates, which are diagnostics that no rule and no observation reads.
    """

def selftest_digest(games: int) -> tuple[int, int]:
    """
    (digest, steps) over `games` whole games (bindings/selftest.hpp). The WebAssembly build exports the same function, so equal numbers mean the two engines play the same games, position, observation and legal mask alike.
    """

def keyed_key(state: GameState, world_key: int, keyed2: bool = False) -> GameState:
    """
    A copy of `state` keyed for its next engine step (bindings/keyed_chance.hpp): rng_state set to hash(world_key, situation) and, at the turn's end, the next deal forced from the world's deck order. What the batches' keyed=True does before every step (keyed2=True: keyed2's packing); a reference playout calls it the same way.
    """

def state_size() -> int:
    """Bytes of one GameState: the row width of raw_bytes() arrays."""

def state_from_raw_bytes(data: bytes) -> GameState:
    """
    The inverse of GameState.raw_bytes(), for the SAME engine build only: the bytes are the struct's memory, so a dataset of them records the engine fingerprint it was written by.
    """

class ObsCodebook:
    """
    Observation floats as exact one-byte codes (bindings/playout_labels.hpp): encode_raw() writes stored states' observations, each from its acting side, as codes; values()[codes] decodes them. One codebook per dataset: codes only mean something with their own table.
    """

    def __init__(self) -> None: ...

    def encode_raw(self, states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], features: int = 0) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]:
        """
        (n, obs width) uint8 codes of raw states' observations, each from its acting side.
        """

    def values(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]:
        """code -> value (code 0 = 0.0); it only grows."""

    def size(self) -> int: ...

class PlayoutLabels:
    """
    P34's playout labeller: queued positions dealt into `worlds` worlds from the mover's view, fresh dice each, played greedily to the game's end by the caller's network under the safe-playout rule, with outcome capture (bindings/playout_labels.hpp). Batching is continuous: fill_live(obs) -> logits -> step(logits) until idle(), collecting take_positions() / take_siblings() as they finish.
    """

    def __init__(self, features: int = 0, worlds: int = 32, capacity: int = 16384, seed: int = 0, safe: bool = True, leaf_horizon: int = 0, max_plies: int = 8000, keyed: bool = False, keyed2: bool = False) -> None:
        """
        keyed=True ties every draw to the situation it is made in (bindings/keyed_chance.hpp): a world's candidates draw the same dice and cards wherever they reach the same situation. keyed2=True is keyed chance with the player fields packed so a USSR side cannot hide the turn; keyed alone keeps the original packing.
        """

    def enqueue_raw(self, states: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], ids: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)], candidates: Annotated[NDArray[numpy.int32], dict(shape=(None, None), order='C', device='cpu', writable=False)] | None = None) -> None:
        """
        Queue positions: `states` is (n, state_size()) uint8 of GameState.raw_bytes() from this engine build, `ids` (n,) int64, and `candidates`, if given, (n, k) int32 flat actions legal at each position -- the sibling slice. Nothing is queued on an error.
        """

    def enqueue(self, states: Sequence[GameState], ids: Sequence[int], candidates: Sequence[Sequence[int]] = []) -> None:
        """enqueue_raw for GameState objects."""

    def world(self, base: GameState, id: int, w: int) -> GameState:
        """
        World `w` of position `id` exactly as the labeller deals it: the mover's unseen cards redealt and the world's dice, from the (seed, id) stream.
        """

    def forced(self) -> int:
        """
        Decisions of finished rows with one legal option, played without the network.
        """

    def chosen(self) -> int:
        """Decisions of finished rows the network chose."""

    def queued(self) -> int: ...

    def occupied(self) -> int: ...

    def capacity(self) -> int: ...

    def obs_width(self) -> int: ...

    def idle(self) -> bool: ...

    def fill_live(self, obs: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')]) -> int:
        """
        Refill free rows from the queue and write the live rows' observations, each from its acting side, into the first n rows of `obs`. Returns n, step()'s row count.
        """

    def fill_live_codes(self, codes: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu')]) -> int:
        """
        fill_live with each observation float written as its uint8 code in codebook(): decode as codebook()[codes]. Exact.
        """

    def codebook(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]:
        """
        The values the codes of fill_live_codes stand for (code 0 = 0.0); it only grows.
        """

    def step(self, logits: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], forced: Annotated[NDArray[numpy.int32], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> None:
        """
        Advance the listed rows by the masked argmax of their logits rows (the first n), under the safe rule, and harvest the worlds that ended. `forced` (int32, -1 = the argmax) gives a row another player's action, taken unless the safe rule refuses it.
        """

    def listed_info(self) -> Annotated[NDArray[numpy.int32], dict(shape=(None, None))]:
        """
        Per listed row: decision type, 1 if a headline card choice, its turn, its position's turn.
        """

    def listed_state(self, j: int) -> GameState:
        """A copy of listed row j's position (rows in fill_live's order)."""

    def take_positions(self) -> dict:
        """
        The main-label positions finished since the last call: id (n,); count (n, 2) = worlds finished, worlds failed; sums (n, 5) over the finished worlds of score, score^2, final VP, turn-end VP, turn-end DEFCON (mover's side); swing_sum / swing_n (n, 7) per region (Europe, Asia, Middle East, Africa, Central America, South America, Southeast Asia): summed swing at the next scoring and worlds that scored it.
        """

    def take_siblings(self) -> dict:
        """
        The sibling rows finished since the last call: id (n,); meta (n, 5) = candidate index, world, failed, has_leaf, scored-region bits; cols (n, 4) = score, final VP, turn-end VP, turn-end DEFCON; swing (n, 7); leaf (n, state_size()) raw states, zeros where has_leaf is 0.
        """

def determinize(state: GameState, me: Player, seed: int, keep_hand: bool = False, keep_revealed: bool = False) -> GameState:
    """
    A clone with the cards `me` cannot see -- the opponent's unknown hand and the draw deck -- reshuffled between those two places from SplitMix64 state `seed`, every count kept, every seen card left; an unrevealed opposing headline is forgotten. The C++ port of ai.search.dmcts.determinize that the rollout root uses in both builds. keep_hand=True keeps the opponent's true hand and reshuffles only the draw deck (the oracle's world). keep_revealed=True keeps an opposing headline `me` has seen under Space Race box 4 (revealed_headline), chosen and in the hand.
    """

def revealed_headline(state: GameState, me: Player) -> int:
    """
    The opposing headline `me` has seen, else 0: with Man in Earth Orbit (Space Race box 4) held by `me` alone the opponent chooses first and reveals it.
    """

def place_hand(state: GameState, me: Player, held: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> GameState:
    """
    A clone with `me`'s hidden pool (the opponent's unknown hand and the deck) split by `held` (110 bytes, card c at c - 1): set -> the opponent's unknown hand, else the deck.
    """

def hypothesis_rows(state: GameState, me: Player, held: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)], features: int = 0) -> tuple:
    """
    A past decision of `me`'s opponent at `state` under n hypothesised hands: for each row of `held` (n x 110) the position with `me`'s hidden pool split by it (place_hand), its observation from the opponent's side and its legal mask -> (obs n x width, masks n x 220). The policy-likelihood belief's inner loop (ai/search/belief.py).
    """

def hypothesis_masks(state: GameState, me: Player, held: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]:
    """
    hypothesis_rows' legal masks alone (n x 220): the observations differ only in each hidden card's own-hand / deck slot, which the belief sets on the device (ai/search/belief.py).
    """

class BeliefTracker:
    """
    The policy-likelihood belief over the opponent's unseen hand (bindings/belief.hpp, shared with the browser): observe() every decision of the games; at root decisions prepare(), evaluate the rows it lists (rows_patch or rows_full) at every temperature, set_log_likelihoods(), worlds().
    """

    def __init__(self, particles: int = 2048, all_decisions: bool = True, seed: int = 0, reveal: bool = False) -> None: ...

    @staticmethod
    def temperatures() -> list[float]: ...

    @property
    def particles(self) -> int: ...

    def reseed(self, seed: int) -> None: ...

    def new_games(self) -> None: ...

    def observe(self, game: int, state: GameState, decider: Player, action: int) -> None: ...

    def prepare(self, games: Sequence[int], states: Sequence[GameState]) -> list[bool]: ...

    def job_count(self) -> int: ...

    def job_decisions(self, job: int) -> int: ...

    def job_position(self, job: int) -> int:
        """The index, among the last prepare() call's positions, of job j."""

    def job_hands(self, job: int) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]: ...

    def rows_patch(self, features: int = 0) -> tuple:
        """
        (bases R x W, pools R x 110, lefts R x 110, job_of R, actions R, hands J x N x 110, packed masks R x N x 28): every job's recorded decisions, for the caller to build the rows on its device.
        """

    def set_log_likelihoods(self, log_likelihoods: Annotated[NDArray[numpy.float32], dict(shape=(None, None, None), order='C', device='cpu', writable=False)]) -> None: ...

    def worlds(self, position: int, n: int) -> object:
        """
        n hands (n x 110 uint8) resampled from the prepared position's belief, or None where it has none.
        """

    def temperature_weights(self, position: int) -> list[float]: ...

class RolloutBatch:
    """
    The rollout root's playouts for one or more decisions: the sampled worlds of each candidate move, their greedy playouts to a horizon of action-round boundaries (stop='boundaries', the default), to the end of the decision's turn ('turn'), or to the later of the two ('later'), and the leaves the critic reads. The network is the caller's: fill_live() -> logits -> step(), until all_done(); fill_leaves() -> critic -> set_leaf_values(); then values(). Written once for the Python searcher and the browser (bindings/rollout_batch.hpp).
    """

    def __init__(self, horizon: int = 4, shared_dice: bool = True, guard: bool = True, features: int = 0, seed: int = 0, stop: str = 'boundaries', leaf: str = 'mover', keyed: bool = False, oracle: bool = False, aligned: bool = False, box4: bool = False, keyed2: bool = False) -> None: ...

    def add(self, base: GameState, candidates: Sequence[int], worlds: int, hands: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)] | None = None) -> int:
        """
        One decision: `base` with the mover's flat `candidates` (all legal there), in `worlds` worlds. Returns the rows added -- len(candidates) times the worlds kept. `hands` (uint8, worlds x 110), when given, is each world's opposing unknown hand (a belief's worlds, ai/search/belief.py) instead of a random redeal.
        """

    def race(self, position: int, worlds: int, hands: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu', writable=False)] | None = None) -> int:
        """
        The second stage of a raced decision, once its rows are valued: every challenger whose paired lead over candidate 0 is below 0 leaves the race, and `worlds` fresh worlds are added for candidate 0 and the challengers left (none when none is). Returns the rows added; `hands` as add's.
        """

    def in_race(self, position: int, candidate: int) -> bool:
        """
        Whether candidate `candidate` (an index into the position's list) is still in the race.
        """

    def positions(self) -> int: ...

    def mover(self, position: int) -> Player: ...

    def worlds_used(self, position: int) -> int: ...

    def rows(self) -> int: ...

    def live(self) -> int: ...

    def all_done(self) -> bool: ...

    def obs_width(self) -> int: ...

    def refused(self) -> int: ...

    def fill_live(self) -> tuple:
        """
        (observations, masks) of the live rows, each row from its acting side, in the order step() expects.
        """

    def step(self, logits: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)]) -> None:
        """
        Advance the rows the last fill_live listed by the masked argmax of each logits row.
        """

    def fill_leaves(self) -> tuple:
        """
        (observations, masks) of the rows at their horizon and not yet valued, each from the MOVER's side, in the order set_leaf_values() expects.
        """

    def set_leaf_values(self, values: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None: ...

    def values(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]:
        """Every row's value from its mover's side; NaN while live or unvalued."""

    def row_position(self) -> list[int]: ...

    def row_world(self) -> list[int]: ...

    def row_candidate(self) -> list[int]: ...

    def row_done(self) -> list[bool]: ...

    def row_terminal(self) -> list[bool]: ...

    def row_crossed(self) -> list[int]: ...

    def row_state(self, row: int) -> GameState:
        """A copy of the row's position, for tests."""

def rollout_parse_rule(rule: str) -> tuple[str, float]:
    """
    (kind, parameter) of a rollout rule -- argmax, z<x> or kl<t>; ValueError otherwise.
    """

def rollout_candidates(logits: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], mask: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], k: int) -> list[int]:
    """
    The k most probable legal moves, best first (fewer where fewer are legal).
    """

def rollout_decide(batch: RolloutBatch, position: int, candidates: Sequence[int], logits: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)], mask: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu', writable=False)], rule: str) -> dict:
    """
    One position's decision from a valued batch: {pick, worlds, q, lead, lead_se} -- each candidate's mean value from the mover's side, its paired lead over the network's own move (candidate 0) with the standard error, and the pick under `rule`.
    """

def game_ending_reason(state: GameState) -> str:
    """
    Why a finished game ended: '20 VP', 'Europe Control', 'DEFCON 1 (own decision)', 'DEFCON 1 (opponent decision)', 'final scoring' or 'wargames'. Shared with the browser workbench's engine; tools/lib/tournament_evaluator delegates to it.
    """

def state_from_save_json(text: str) -> GameState:
    """
    state_from_save_dict for the JSON text to_save_json writes (or any JSON of that shape). Raises ValueError for text that is not JSON or not a save.
    """

def get_flat_action_mask(state: GameState, merged_influence: bool = False, macro_influence: bool = False) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

def decode_flat_action(arg0: GameState, arg1: int, /) -> MicroAction:
    """
    The MicroAction a flat index stands for here. Raises ValueError outside the space.
    """

def encode_micro_action(arg0: GameState, arg1: MicroAction, /) -> int:
    """
    The flat index of a MicroAction. Raises ValueError for one no slot encodes.
    """

def extract_observation(state: GameState, perspective: Player) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]: ...

def obs_size_for(features: int) -> int:
    """
    Observation width for a feature set: the base 3,824 plus each appended block.
    """

def extract_observation_features(state: GameState, perspective: Player, features: int) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]:
    """
    The base observation followed by the blocks `features` appends. features=0 is exactly extract_observation(state, perspective).
    """

OBS_FEATURE_OPS_BUDGET: int = 2

OBS_FEATURE_HEADLINE_REVEAL: int = 4

OBS_FEATURE_OPP_EVENT_GATED: int = 8

OBS_FEATURE_CANONICAL_SETUP: int = 16

OBS_FEATURES_RULES: int = 24

OBS_FEATURES_PERSON_RULES: int = 8

OBS_FEATURES_ALL: int = 30

OBS_FLAG_STAGED_CARDS: int = 1

OBS_SIZE_V23: int = 3824

OBS_SIZE: int = 3824

FLAT_ACTION_SPACE_SIZE: int = 304

FLAT_CARD: int = 0

FLAT_RESOLUTION: int = 110

FLAT_OP_MODE: int = 112

FLAT_ROLL_DIE: int = 115

FLAT_NODE: int = 116

FLAT_BRANCH: int = 200

FLAT_CONFIRM_DONE: int = 208

FLAT_DEFCON_VALUE: int = 209

FLAT_REGION: int = 214

FLAT_BASE_SIZE: int = 220

FLAT_MACRO_NODE: int = 220

BUILD_COMPILER: str

class ActionMask:
    @staticmethod
    def generate_flat_mask(state: GameState, merged_influence: bool = False, macro_influence: bool = False) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

    @staticmethod
    def is_merged_influence_action(arg0: GameState, arg1: int, /) -> bool: ...

    @staticmethod
    def is_macro_action(arg: int, /) -> bool: ...

    @staticmethod
    def decode_flat_action(arg0: GameState, arg1: int, /) -> MicroAction: ...

    @staticmethod
    def encode_micro_action(arg0: GameState, arg1: MicroAction, /) -> int: ...

class BatchedSearch:
    def __init__(self, capacity: int, c_puct: float = 1.5, auto_advance: bool = True, obs_features: int = 0, merged_influence: bool = False, fpu_reduction: float = 0.0) -> None:
        """
        Up to `capacity` independent searches, stepped together. obs_features and merged_influence are the evaluating model's view; fpu_reduction values an unvisited move at its node's value less this much, from the mover's side (0: the node's value).
        """

    @property
    def capacity(self) -> int: ...

    @property
    def obs_width(self) -> int: ...

    @property
    def num_trees(self) -> int: ...

    @property
    def num_groups(self) -> int: ...

    @property
    def roots_phase(self) -> bool: ...

    @overload
    def reset(self, roots: Sequence[GameState], simulations: int, rng_state: Sequence[int], num_groups: int = 1) -> None:
        """
        Start one search per root, each with its own budget of `simulations`, the trees split into `num_groups` contiguous groups evaluated separately. `rng_state` is random.Random.getstate()[1]: the children's chance seeds are drawn from it exactly as the Python tree draws them.
        """

    @overload
    def reset(self, roots: Sequence[GameState], simulations: Sequence[int], rng_state: Sequence[int], num_groups: int = 1) -> None:
        """
        As above with one budget per root, so searches of different sizes share their rounds.
        """

    def set_buffers(self, obs: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], masks: Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', device='cpu')]) -> None:
        """
        Write leaves into these caller-owned arrays (e.g. page-locked) instead of this object's own. The caller keeps them alive.
        """

    def group_offset(self, group: int) -> int: ...

    def group_has_budget(self, group: int) -> bool: ...

    def mt_state(self) -> list[int]:
        """The random.Random state after this search's draws, for setstate()."""

    def select_leaves(self, group: int) -> int:
        """
        Write group `group`'s leaves awaiting evaluation into rows [group_offset, +k) of observations() and masks() and return k: its roots on the first call, then one simulation round per call. Take the groups' rounds in turn, 0, 1, ...
        """

    def expand_and_backup(self, group: int, probs: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu', writable=False)], values: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None:
        """
        The network's answer for rows [0, k): softmax over masked logits, and v_win from each leaf mover's side. Fills the leaves in and backs their values up.
        """

    def observations(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None))]: ...

    def masks(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]: ...

    def set_root_priors(self, index: int, priors: Sequence[float]) -> None: ...

    def root(self, index: int) -> tuple:
        """
        (terminal, mover, value_us, actions, priors, visits, value sums) of a root.
        """

    def tree_size(self, index: int) -> int: ...

class VectorizedBatchRunner:
    def __init__(self, num_envs: int, base_seed: int = 12345) -> None: ...

    @property
    def obs_width(self) -> int: ...

    def set_obs_features(self, us: Sequence[int], ussr: Sequence[int]) -> None:
        """
        Per env and side, the observation feature set (obs_features bits) that side's model reads. Rows are written in the decider's set and zero-padded to obs_width.
        """

    def set_obs_features_env(self, env_index: int, us: int, ussr: int) -> None: ...

    def reset_game(self, arg0: int, arg1: int, /) -> None: ...

    def refresh_all(self) -> None: ...

    def set_merged_influence(self, us: Sequence[bool], ussr: Sequence[bool]) -> None:
        """
        P23 / E4.1: per env and side, whether that side decides in the merged-influence view. Rebuilds the cached masks.
        """

    def set_action_views(self, us: Sequence[int], ussr: Sequence[int]) -> None:
        """
        Per env and side, the action view that side decides in: 0 E4, 1 merged (E4.1, P23), 2 macro (P42: the until-control slots). Masks are rebuilt; steps apply each view's composed actions.
        """

    def set_action_views_env(self, idx: int, us: int, ussr: int) -> None: ...

    def set_merged_influence_env(self, env_index: int, us: bool, ussr: bool) -> None:
        """
        P23 / E4.1: the same for one env; rebuilds only that env's cached mask.
        """

    def step_flat_all(self, actions: Sequence[int], auto_advance: bool = False) -> list[int]: ...

    def set_capture_predeal(self, on: bool) -> None:
        """
        P33: record each env's turn-end park (after a turn's last action, before the deal) from both sides during step_flat_all's auto-advance; read it with get_predeal after the step.
        """

    def get_predeal(self) -> tuple:
        """
        (flags, obs_us, obs_ussr) of the last step: per env, whether it passed its turn-end park, and that position's observation from each side. Views into the runner: copy before the next step.
        """

    def get_observations(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None))]: ...

    def get_action_masks(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]: ...

    def get_decision_players(self) -> list[int]: ...

    def get_terminals(self) -> list[bool]: ...

    def get_terminal_utilities(self) -> list[float]: ...

    def get_victory_points(self) -> list[int]: ...

    def get_opponent_hands(self, arg: Sequence[int], /) -> list[float]: ...

    def get_turns(self) -> list[int]: ...

    def get_state(self, idx: int) -> GameState: ...

    def set_state(self, idx: int, state: GameState) -> None: ...

    def compute_useful_actions_potentials(self, arg: Sequence[int], /) -> list[float]: ...
