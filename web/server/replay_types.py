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
    step_index: int
    turn: int
    ar: int
    phase: str
    player: str
    text: str
    details: Optional[List[str]]
    vp_delta: int


class GameStateDict(TypedDict, total=False):
    """Complete GameState dictionary serialized by ts.state_to_dict."""
    # The engine's own observation for the player whose move it is, base64 float32.
    # Present only for that player: it encodes their hand, which the opponent must not see.
    observation_b64: str
    # The position as a URL-safe token (web/server/analysis.py encode_position), so the
    # workbench can put it in the address bar and a link reopens exactly this board.
    position: str
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


class PolicyChoiceDict(TypedDict, total=False):
    """One legal action and the probability the policy put on it."""
    idx: int
    p: float
    name: str


class ReplayPolicyDict(TypedDict, total=False):
    """What the policy believed at one decision node, before the action was applied.

    `p` values are the model's OWN distribution -- softmax of the masked logits at temperature
    1 -- not the tempered distribution that was sampled from. The two answer different
    questions ("what did it believe" vs "how likely was this sample"), so both are recorded:
    `p_chosen` is the belief and `p_chosen_sampled` is the sampling probability at
    `temperature`. `top` holds the legal actions by descending probability, capped and floored;
    `p_tail` is the mass that did not fit.
    """
    source: str
    temperature: float
    n_legal: int
    chosen_idx: int
    p_chosen: float
    p_chosen_sampled: float
    argmax_idx: int
    p_max: float
    entropy: float
    top: List[PolicyChoiceDict]
    p_tail: float
    v_win: float
    v_vp: float


class ReplayCriticDict(TypedDict, total=False):
    """Both value heads from both perspectives, read on the state the step's snapshot shows.

    A zero-sum critic must satisfy v(US) = -v(USSR), so the residuals are pure model error.
    """
    v_win_us: float
    v_win_ussr: float
    v_vp_us: float
    v_vp_ussr: float
    win_residual: float
    vp_residual: float
    at: str


class ReplayTraceMetaDict(TypedDict, total=False):
    """Provenance for the policy/critic trace: which model, which build, which settings.

    `engine_fingerprint` matters for an annotated trace: it is produced by re-driving the game
    from (seed, actions), and a rebuilt engine can change the decision stream with no Python
    change, so the numbers belong to the build that produced them.
    """
    mode: str
    model_us: str
    model_ussr: str
    #: Which side's network answered the critic. One model answers for the whole game even when
    #: two different checkpoints are playing, so the value curve stays one opinion.
    critic_model: str
    checkpoint_sha256_12: str
    arch: str
    temperature: float
    top_k: int
    p_floor: float
    engine_fingerprint: str


class ReplayStepDict(TypedDict, total=False):
    """Single step in a standard .tslog.json replay file.

    `policy` and `critic` are optional: a heuristic bot has no distribution, a human game has no
    model, and replays written before the trace existed have neither. Readers must treat both as
    absent by default.
    """
    step_index: int
    turn: int
    ar: int
    phase: str
    player: str
    action: ReplayActionDict
    description: str
    state_snapshot: GameStateDict
    policy: ReplayPolicyDict
    critic: ReplayCriticDict


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
    trace: ReplayTraceMetaDict
    # Present only for a game begun from a loaded position rather than from the seed.
    start_position: str


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


class AnalysisChoiceDict(TypedDict, total=False):
    """One legal action in a live analysis, with the MicroAction a click would send.

    `decision_type`/`primary_id`/`flags` let the client attach the probability to the button,
    card or country that sends exactly that action, without re-deriving flat offsets. A
    `composed` entry is an E4.1 merged-view action ("Ops for influence, first point in
    `country_id`", or with no `country_id` "place nothing"); its MicroAction fields name the
    commit half, which is the influence button.
    """
    idx: int
    p: float
    name: str
    decision_type: int
    primary_id: int
    flags: int
    composed: bool
    country_id: int


class LiveAnalysisDict(TypedDict, total=False):
    """What a chosen checkpoint thinks of the live position (web/server/analysis.py).

    `policy` is the summary of `read_policy` (argmax, entropy, n_legal, ...) without `top`;
    the per-action probabilities are `choices`, every legal action, by descending p. Absent
    `policy` means no decision is open (terminal game); the critic is always present.
    """
    model: str
    label: str
    merged_influence: bool
    decision_player: str
    decision_type: int
    critic: ReplayCriticDict
    policy: ReplayPolicyDict
    choices: List[AnalysisChoiceDict]


class AnalysisRunDict(TypedDict):
    """One run directory's checkpoints, as offered by the workbench's model picker."""
    run: str
    snapshots: List[str]


class AnalysisModelListDict(TypedDict):
    """`GET /api/analysis/models`: every checkpoint under the shared checkpoints tree."""
    root: str
    runs: List[AnalysisRunDict]
    loose: List[str]


class WebSocketMessageDict(TypedDict, total=False):
    """Top-level WebSocket packet exchanged with client."""
    type: str
    game_id: Optional[str]
    player: Optional[str]
    state: Optional[GameStateDict]
    # Only on STATE_UPDATEs to a socket that asked for one with SET_ANALYSIS_MODEL.
    analysis: Optional[LiveAnalysisDict]
    analysis_model: Optional[str]
    action: Optional[WebSocketActionPayloadDict]
    error: Optional[str]
    winner: Optional[str]
    message: Optional[str]


class SavedGameSessionDict(TypedDict, total=False):
    """Serialization format for local game state checkpoint in CLI agent player."""
    seed: int
    actions: List[Dict[str, Any]]
    action_logs: List[Dict[str, Any]]
