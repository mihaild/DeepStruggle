"""TypedDict definitions for Twilight Struggle state serialization, replays, logs, and metrics."""

from typing import TypedDict, Optional, List, Dict, Any, Union


class CountryStateDict(TypedDict, total=False):
    """Serialization format for a single country in GameState."""
    id: int
    name: str
    stability: int
    battleground: bool
    region: int
    us_influence: int
    ussr_influence: int
    controlled_by: str


class DieRollDict(TypedDict, total=False):
    """Serialization format for die roll outcome."""
    type: str
    type_id: int
    roller: str
    card_id: int
    target_country: int
    modified_roll: int
    success: bool


class ChinaCardDict(TypedDict, total=False):
    """Serialization format for China card status."""
    holder: str
    playable: bool


class DecisionContextDict(TypedDict, total=False):
    """Serialization format for active decision context."""
    decision_player: str
    decision_type: int
    decision_type_name: str
    card_id: int
    target_id: int
    ops: int
    flags: int


class LegalActionsDict(TypedDict, total=False):
    """Serialization format for legal actions representation."""
    decision_type: int
    decision_type_name: str
    decision_player: str
    legal_cards: List[int]
    legal_countries: List[int]
    legal_space: bool


class ActionLogEntryDict(TypedDict, total=False):
    """Individual action log entry."""
    turn: int
    ar: int
    player: str
    text: str
    details: Optional[List[str]]


class GameStateDict(TypedDict, total=False):
    """Complete GameState dictionary serialized by ts.state_to_dict."""
    victory_points: int
    defcon: int
    mil_ops: Dict[str, int]
    space: Dict[str, int]
    turn: int
    action_round: int
    phasing_player: str
    current_phase: int
    current_phase_name: str
    phase_name: str
    headline_us_card: int
    headline_ussr_card: int
    headline_first_card: int
    headline_second_card: int
    headline_stage: int
    forced_card_player: str
    forced_card_id: int
    last_die_roll: int
    last_opp_die_roll: int
    die_roll: DieRollDict
    space_turns_used: Dict[str, int]
    china_card: ChinaCardDict
    flags: List[str]
    persistent_effects: int
    countries: Dict[str, CountryStateDict]
    hands: Dict[str, List[int]]
    discard_pile: List[int]
    removed_pile: List[int]
    unavailable_cards: List[int]
    draw_deck_count: int
    card_locations: Dict[Union[int, str], str]
    decision_context: DecisionContextDict
    legal_actions: LegalActionsDict
    is_terminal: bool
    terminal_utility: float
    action_logs: List[ActionLogEntryDict]


class ReplayActionDict(TypedDict, total=False):
    """Action payload stored inside a replay step."""
    flat_action_idx: int
    decision_type: int
    primary_id: int
    secondary_id: int
    flags: int
    card_id: Optional[int]
    target_id: Optional[int]
    action_type: Optional[int]
    type: Optional[str]


class ReplayStepDict(TypedDict, total=False):
    """Single step in a standard .tslog.json replay file."""
    step_index: int
    turn: int
    ar: int
    phase: str
    player: str
    action: ReplayActionDict
    description: str
    state_snapshot: GameStateDict


class ReplayResultDict(TypedDict, total=False):
    """Outcome summary of a completed game in replay metadata."""
    winner: str
    margin: int
    end_turn: int
    reason: str


class ReplayPlayersDict(TypedDict, total=False):
    """Player labels in replay metadata."""
    US: str
    USSR: str


class ReplayMetadataDict(TypedDict, total=False):
    """Metadata block in standard .tslog.json replay files."""
    game_id: str
    created_at: str
    seed: int
    players: Dict[str, str]
    total_steps: int
    result: Optional[ReplayResultDict]


class ReplayInitialStateDict(TypedDict, total=False):
    """Initial state block in replay files."""
    seed: int


class ReplayLogDict(TypedDict, total=False):
    """Standard .tslog.json top-level document structure."""
    version: str
    metadata: ReplayMetadataDict
    initial_state: ReplayInitialStateDict
    steps: List[ReplayStepDict]
    # Backward compatibility attributes:
    game_id: str
    seed: int
    total_steps: int
    winner: str
    final_vp: int
    final_turn: int
    final_state: GameStateDict
    events: List[Dict[str, Any]]
    logs: List[Dict[str, Any]]


class ReplaySummaryDict(TypedDict, total=False):
    """Summary item returned by ReplayManager.list_replays."""
    filename: str
    game_id: str
    created_at: str
    players: Dict[str, str]
    total_steps: int
    result: Optional[ReplayResultDict]


class TrainingMetricEntryDict(TypedDict, total=False):
    """Training loop metrics entry serialized per iteration."""
    iteration: int
    step: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    win_rate_vs_prev: float
    elapsed_seconds: float
    timestamp: str


class EloBenchmarkReportDict(TypedDict, total=False):
    """Output summary of ELO evaluation benchmark."""
    elo_ratings: Dict[str, float]
    iterations: List[int]
    elapsed_seconds: float


class TournamentReportDict(TypedDict, total=False):
    """Output summary of Arena tournament evaluation."""
    total_games: int
    neural_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    avg_vp_margin: float
    avg_steps: float
    avg_turn: float
    opponent_type: str
    details: Optional[List[Dict[str, Any]]]


class AuditLogEntryDict(TypedDict, total=False):
    """Audit entry for LLM/bot game verification logs."""
    step: int
    turn: int
    phase: str
    ar: int
    phasing_player: str
    defcon: int
    victory_points: int
    mil_ops: Dict[str, int]
    space: Dict[str, int]
    decision: Dict[str, Any]
    action_taken: Dict[str, Any]
    action_executed: Dict[str, Any]
    violations: List[str]
    audit_passed: bool
    strategy: Optional[str]
    commentary: Optional[str]
    chain_of_thought: Optional[str]


class AuditGameReportDict(TypedDict, total=False):
    """Top-level audit report written by bot audit scripts."""
    total_steps: int
    violations_count: int
    violations: List[str]
    logs: List[AuditLogEntryDict]
    meta: Optional[Dict[str, Any]]


class WebSocketActionPayloadDict(TypedDict, total=False):
    """MicroAction payload sent over WebSocket or HTTP."""
    type: int
    card_id: int
    target_id: int
    ops: int
    flags: int
    action_type: int
    primary_id: int
    secondary_id: int


class WebSocketMessageDict(TypedDict, total=False):
    """Top-level WebSocket packet exchanged with client."""
    type: str
    game_id: Optional[str]
    player: Optional[str]
    state: Optional[GameStateDict]
    action: Optional[WebSocketActionPayloadDict]
    error: Optional[str]
    winner: Optional[str]
    message: Optional[str]


class SavedGameSessionDict(TypedDict, total=False):
    """Serialization format for local game state checkpoint in CLI agent player."""
    seed: int
    actions: List[Dict[str, Any]]
    action_logs: List[Dict[str, Any]]
