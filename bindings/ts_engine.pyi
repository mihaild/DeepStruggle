"""Twilight Struggle C++ Simulation Engine Python Bindings (nanobind)"""

from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

class EffectBits:
    ALDRICH_AMES_ACTIVE: int
    AWACS_PLAYED: int
    BEAR_TRAP_ACTIVE: int
    BREZHNEV_DOCTRINE_ACTIVE: int
    CAMP_DAVID_PLAYED: int
    CHERNOBYL_ACTIVE: int
    CHERNOBYL_REGION_MASK: int
    CHERNOBYL_REGION_SHIFT: int
    CMC_ACTIVE_US: int
    CMC_ACTIVE_USSR: int
    CONTAINMENT_ACTIVE: int
    DEATH_SQUADS_US: int
    DEATH_SQUADS_USSR: int
    DEFCON_SUICIDE_PROVOKED: int
    CMC_SUICIDE_LOSS: int
    CMC_SUICIDE_LOSS: int
    EVIL_EMPIRE_PLAYED: int
    FLOWER_POWER_ACTIVE: int
    FORMOSAN_RESOLUTION_ACTIVE: int
    IRANIAN_HOSTAGE_CRISIS_PLAY: int
    IRAN_CONTRA_ACTIVE: int
    IRON_LADY_PLAYED: int
    JOHN_PAUL_II_PLAYED: int
    MARSHALL_PLAN_PLAYED: int
    NATO_ACTIVE: int
    NATO_CANCELED_FRANCE: int
    NATO_CANCELED_WEST_GERMANY: int
    NORAD_ACTIVE: int
    NORTH_SEA_OIL_ACTIVE: int
    NORTH_SEA_OIL_PLAYED: int
    NUCLEAR_SUBS_ACTIVE: int
    PURGE_USSR_ACTIVE: int
    PURGE_US_ACTIVE: int
    QUAGMIRE_ACTIVE: int
    SALT_ACTIVE: int
    SHUTTLE_DIPLOMACY_ACTIVE: int
    SPACE_USSR_ATTEMPT_1: int
    SPACE_USSR_ATTEMPT_2: int
    SPACE_US_ATTEMPT_1: int
    SPACE_US_ATTEMPT_2: int
    TEAR_DOWN_THIS_WALL_PLAYED: int
    THE_REFORMER_PLAYED: int
    U2_INCIDENT_ACTIVE: int
    US_JAPAN_PACT_ACTIVE: int
    VIETNAM_REVOLTS_ACTIVE: int
    WARSAW_PACT_PLAYED: int
    WE_WILL_BURY_YOU_PENDING: int
    WILLY_BRANDT_PLAYED: int
    YURI_AND_SAMANTHA_ACTIVE: int



class RollType(enum.IntEnum):
    NONE = 0

    COUP = 1

    REALIGNMENT = 2

    SPACE_RACE = 3

    WAR_EVENT = 4

    OLYMPIC_GAMES = 5

    SUMMIT = 6

    TRAP_ESCAPE = 7

NONE: DecisionType = DecisionType.NONE

COUP: OpMode = OpMode.COUP

REALIGNMENT: RollType = RollType.REALIGNMENT

SPACE_RACE: RollType = RollType.SPACE_RACE

WAR_EVENT: RollType = RollType.WAR_EVENT

OLYMPIC_GAMES: RollType = RollType.OLYMPIC_GAMES

SUMMIT: RollType = RollType.SUMMIT

TRAP_ESCAPE: RollType = RollType.TRAP_ESCAPE

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

class PlayMode(enum.IntEnum):
    EVENT = 0

    OPS = 1

    SPACE = 2

    PASS = 3

EVENT: PlayMode = PlayMode.EVENT

OPS: PlayMode = PlayMode.OPS

SPACE: PlayMode = PlayMode.SPACE

PASS: PlayMode = PlayMode.PASS

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

class CardLocation(enum.IntEnum):
    UNAVAILABLE = 0

    DRAW_DECK = 1

    HAND_US = 2

    HAND_USSR = 3

    DISCARD_PILE = 4

    REMOVED_FROM_GAME = 5

    ONGOING_EVENT = 6

    PEEKED_TEMP = 7

UNAVAILABLE: CardLocation = CardLocation.UNAVAILABLE

DRAW_DECK: CardLocation = CardLocation.DRAW_DECK

HAND_US: CardLocation = CardLocation.HAND_US

HAND_USSR: CardLocation = CardLocation.HAND_USSR

DISCARD_PILE: CardLocation = CardLocation.DISCARD_PILE

REMOVED_FROM_GAME: CardLocation = CardLocation.REMOVED_FROM_GAME

ONGOING_EVENT: CardLocation = CardLocation.ONGOING_EVENT

PEEKED_TEMP: CardLocation = CardLocation.PEEKED_TEMP

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

    def clone(self) -> GameState: ...

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

    @us_influence.setter
    def us_influence(self, arg: int, /) -> None: ...

    @property
    def ussr_influence(self) -> int: ...

    @ussr_influence.setter
    def ussr_influence(self, arg: int, /) -> None: ...

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
    def temp_card_cnt(self) -> int: ...

    @temp_card_cnt.setter
    def temp_card_cnt(self, arg: int, /) -> None: ...

    @property
    def temp_cards(self) -> list[int]: ...

    @temp_cards.setter
    def temp_cards(self, arg: Sequence[int], /) -> None: ...

    def is_visited(self, arg: int, /) -> bool: ...

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

    def set_flag(self, arg: int, /) -> None: ...

    def clear_flag(self, arg: int, /) -> None: ...

    def ctx(self) -> DecisionContext: ...

    def has_flag(self, arg: int, /) -> bool: ...

    def get_country(self, arg: int, /) -> CountryState: ...

    def set_country(self, arg0: int, arg1: int, arg2: int, /) -> None: ...

    def get_space_turns_used(self, arg: Player, /) -> int: ...

    def set_space_turns_used(self, arg0: Player, arg1: int, /) -> None: ...

    def record_space_attempt(self, arg: Player, /) -> None: ...

    def get_card_location(self, arg: int, /) -> CardLocation: ...

    def set_card_location(self, arg0: int, arg1: CardLocation, /) -> None: ...

    def to_dict(self) -> dict: ...

    def to_json(self) -> str: ...

class StateMachine:
    @staticmethod
    def advance_headline_step(state: GameState, /) -> None: ...

    @staticmethod
    def advance_after_action_round(state: GameState, /) -> None: ...

class Engine:
    @staticmethod
    def init_game(arg0: GameState, arg1: int, /) -> None: ...

    @staticmethod
    def step(state: GameState, action: MicroAction, auto_advance: bool = False) -> bool: ...

    @staticmethod
    def is_terminal(arg: GameState, /) -> bool: ...

    @staticmethod
    def get_terminal_utility(arg: GameState, /) -> float: ...

    @staticmethod
    def get_legal_action_mask(arg: GameState, /) -> list: ...

    @staticmethod
    def get_legal_action_indices(arg: GameState, /) -> list: ...

    @staticmethod
    def get_flat_action_mask(arg: GameState, /) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

    @staticmethod
    def step_flat(state: GameState, action_idx: int, auto_advance: bool = False) -> bool: ...

    @staticmethod
    def auto_advance_step(state: GameState, max_steps: int = 128) -> int: ...

    @staticmethod
    def has_held_scoring_card(arg0: GameState, arg1: Player, /) -> bool: ...

    @staticmethod
    def is_held_scoring_game_over(arg: GameState, /) -> bool: ...

    @staticmethod
    def is_held_scoring_loss(arg0: GameState, arg1: Player, /) -> bool: ...

class MapData:
    @staticmethod
    def get_country_name(arg: int, /) -> str: ...

    @staticmethod
    def get_country_by_name(arg: str, /) -> int: ...

    @staticmethod
    def get_country_info(arg: int, /) -> dict: ...

class CardHandlers:
    @staticmethod
    def can_trigger_event(state: GameState, card_id: int, player: Player) -> bool: ...

    @staticmethod
    def trigger_event(state: GameState, card_id: int, player: Player, forced_roll: int = 0) -> bool: ...

    @staticmethod
    def handle_event_step(arg0: GameState, arg1: MicroAction, /) -> bool: ...

class RegionalStatus:
    NONE: int
    PRESENCE: int
    DOMINATION: int
    CONTROL: int

class RegionScoreSummary:
    us_status: RegionalStatus
    ussr_status: RegionalStatus
    us_countries: int
    ussr_countries: int
    us_battlegrounds: int
    ussr_battlegrounds: int
    us_superpower_adjacent: int
    ussr_superpower_adjacent: int
    us_score: int
    ussr_score: int
    net_delta: int

class Operations:
    @staticmethod
    def can_place_influence(state: GameState, player: Player, country_id: int, /) -> bool: ...

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
    def evaluate_region(state: GameState, region: Region, is_final_scoring: bool = False, /) -> RegionScoreSummary: ...

    @staticmethod
    def get_country_control(state: GameState, country_id: int, /) -> Player: ...

    @staticmethod
    def is_controlled_by(state: GameState, country_id: int, player: Player, /) -> bool: ...

    @staticmethod
    def compute_useful_actions_potential(state: GameState, player: Player, /) -> float: ...

class CardData:
    @staticmethod
    def get_card_name(arg: int, /) -> str: ...

    @staticmethod
    def get_card_by_name(arg: str, /) -> int: ...

    @staticmethod
    def get_card_info(arg: int, /) -> dict: ...

def state_to_dict(arg: GameState, /) -> dict:
    """Convert GameState to Python dictionary"""

def get_flat_action_mask(arg: GameState, /) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

def decode_flat_action(arg0: GameState, arg1: int, /) -> MicroAction: ...

def encode_micro_action(arg0: GameState, arg1: MicroAction, /) -> int: ...

def extract_observation(arg0: GameState, arg1: Player, /) -> Annotated[NDArray[numpy.float32], dict(shape=(None,))]: ...

class ActionMask:
    @staticmethod
    def generate_flat_mask(arg: GameState, /) -> Annotated[NDArray[numpy.uint8], dict(shape=(None,))]: ...

    @staticmethod
    def decode_flat_action(arg0: GameState, arg1: int, /) -> MicroAction: ...

    @staticmethod
    def encode_micro_action(arg0: GameState, arg1: MicroAction, /) -> int: ...

class VectorizedBatchRunner:
    def __init__(self, num_envs: int, base_seed: int = 12345) -> None: ...

    def reset_game(self, arg0: int, arg1: int, /) -> None: ...

    def refresh_all(self) -> None: ...

    def step_flat_all(self, actions: Sequence[int], auto_advance: bool = False) -> list[int]: ...

    def get_observations(self) -> Annotated[NDArray[numpy.float32], dict(shape=(None, None))]: ...

    def get_action_masks(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None))]: ...

    def get_decision_players(self) -> list[int]: ...

    def get_terminals(self) -> list[bool]: ...

    def get_terminal_utilities(self) -> list[float]: ...

    def get_victory_points(self) -> list[int]: ...

    def get_opponent_hands(self, acting_players: list[int], /) -> list[float]: ...
    def get_turns(self) -> list[int]: ...

    def get_state(self, arg: int, /) -> GameState: ...

    def set_state(self, idx: int, state: GameState) -> None: ...

    def set_state(self, idx: int, state: GameState) -> None: ...
    def compute_useful_actions_potentials(self, acting_players: list[int], /) -> list[float]: ...
