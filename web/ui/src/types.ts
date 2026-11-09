/**
 * The display state (bindings/state_json.cpp `display_state`, what Python's `GameState.to_dict()`
 * returns and web/server/replay_types.py `GameStateDict` types), plus the fields the page attaches.
 */
import type { Message } from "./i18n";

export interface CountryNode {
  id: number;
  name: string;
  stability: number;
  battleground: boolean;
  region: number;
  us_influence: number;
  ussr_influence: number;
  controlled_by: 'US' | 'USSR' | 'NONE';
}

export interface DecisionContext {
  decision_player: 'US' | 'USSR' | 'NONE';
  /** The engine's DecisionType (its names: WasmEngine.layout.decision_types). */
  decision_type: number;
  decision_type_name?: string;
  /** OpMode of an operation in progress (WasmEngine.layout.op_modes), and its name. */
  op_mode?: number;
  op_mode_name?: string;
  pending_op_card: number;
  pending_op_card_name?: string;
  pending_ops_value: number;
  remaining_steps: number;
  max_per_country: number;
  allow_early_stop: boolean;
  resolving_card: number;
  resolving_card_name?: string;
  stack_depth: number;
}

export interface LegalActions {
  decision_type: number;
  decision_type_name?: string;
  decision_player: 'US' | 'USSR' | 'NONE';
  valid_ids: number[];
  valid_action_labels?: Record<string, string>;
  allow_early_stop: boolean;
  /** The engine's flat mask offers CONFIRM_DONE (session.ts). For a target decision that is
   * either an early stop or the way out once no eligible target is left -- the display's own
   * per-decision mask has no slot for the second. */
  can_confirm_done?: boolean;
}

export interface CardInHand {
  id: number;
  name: string;
  ops: number;
}

/** What a click is, for grouping the log into plays (game/log_view.ts). */
export type LogKind = "setup" | "headline" | "play" | "mode" | "point" | "step";

export interface ActionLogItem {
  step_index: number;
  turn: number;
  ar: number;
  phase: string;
  player: string;
  text: string;
  details?: string[];
  vp_delta?: number;
  /** Set on a click's entry; the lines the session writes itself (turn end, undo...) have none. */
  kind?: LogKind;
  /** The heading of the play the click starts (never set for a headline still face down). */
  title?: string;
  /** A play's mode, for its heading. */
  mode?: string;
  /** Cards whose ordinary discard the heading implies. */
  cards?: number[];
  /** The Event the click answered (`resolving_card` before it), 0 for none, and its name. */
  resolving?: number;
  resolving_name?: string;
}

/** What one Event of a step did to VP and DEFCON (GameState::last_event_tracks). */
export interface EventTrackState {
  card_id: number;
  card_name?: string;
  /** Positive toward the US, as victory_points. */
  vp_delta: number;
  defcon_delta: number;
}

export interface DieRollState {
  type: string;
  type_id?: number;
  roller?: string;
  card_id?: number;
  card_name?: string;
  country_id?: number;
  country_name?: string;
  roll1?: number;
  mod1?: number;
  total1?: number;
  roll2?: number;
  mod2?: number;
  total2?: number;
  success?: boolean;
  net_delta?: number;
}

export interface GameState {
  victory_points: number;
  defcon: number;
  mil_ops: { US: number; USSR: number };
  space: { US: number; USSR: number };
  turn: number;
  action_round: number;
  phasing_player: 'US' | 'USSR' | 'NONE';
  current_phase: number;
  current_phase_name?: string;
  phase_name?: string;
  headline_us_card: number;
  headline_ussr_card: number;
  /** Once both headlines are in: the card that resolves first and the one second (0 before). */
  headline_first_card?: number;
  headline_second_card?: number;
  /** 0 while choosing, 1 resolving the first headline, 2 the second. */
  headline_stage?: number;
  forced_card_player: 'US' | 'USSR' | 'NONE';
  forced_card_id: number;
  space_turns_used: { US: number; USSR: number };
  china_card: { holder: 'US' | 'USSR'; playable: boolean };
  die_roll?: DieRollState;
  event_tracks?: EventTrackState[];
  last_die_roll?: number;
  last_opp_die_roll?: number;
  flags: string[];
  persistent_effects: number;
  countries: Record<string, CountryNode>;
  hands: {
    US: number[];
    USSR: number[];
    US_cards?: CardInHand[];
    USSR_cards?: CardInHand[];
  };
  discard_pile: number[];
  removed_pile: number[];
  unavailable_cards?: number[];
  card_locations?: Record<string, string>;
  draw_deck_count: number;
  decision_context: DecisionContext;
  legal_actions: LegalActions;
  is_terminal: boolean;
  terminal_utility: number;
  game_id?: string;
  seed?: number;
  step_index?: number;
  can_undo?: boolean;
  action_logs?: ActionLogItem[];
  players?: { US: string; USSR: string };
  /** URL-safe token for this exact position (the server's encode_position). */
  position?: string;
  /** Whose view the state was redacted for (game/view.ts); absent or "DEV" means everything. */
  view?: "DEV" | "US" | "USSR";
  /** A player's view: how many of each hand's cards it may not see (only the other side's). */
  hidden_cards?: { US: number; USSR: number };
  /** A player's view: the draw deck plus the other side's unseen cards. */
  unseen_count?: number;
  /** A player's view: the other side is deciding, and its options are withheld. */
  view_hidden_decision?: boolean;
  /** A player's view: what the other side's latest turn did to the map (game/last_turn.ts). */
  last_turn?: LastTurn | null;
  /** The latest action round's die rolls while still recent (game/rolls.ts, set by main.ts). */
  recent_rolls?: RecentRoll[];
  /** What each scoring card would score here, by the engine (WasmEngine.regionScores, set by main.ts). */
  region_scores?: RegionScores | null;
}

export type RegionalStatus = "NONE" | "PRESENCE" | "DOMINATION" | "CONTROL";

/** One side of a region's scoring, as the engine scores it (ts_region_scores_json). */
export interface RegionSideScore {
  /** The side's standing; null for Southeast Asia, which has none. */
  status: RegionalStatus | null;
  /** The countries, battlegrounds and countries next to the enemy superpower the scoring counted. */
  countries: number;
  battlegrounds: number;
  adjacent: number;
  /** What the standing alone is worth; null where it has none (Southeast Asia). */
  base_vp: number | null;
  vp: number;
  /** The countries the side controls there: `battleground` as this scoring counts it, and in
   *  Southeast Asia the `vp` each is worth. */
  controlled: Array<{ id: number; battleground?: boolean; vp?: number }>;
  /** USSR only: whether Shuttle Diplomacy took one of its battlegrounds out of this count. */
  shuttle_diplomacy?: boolean;
}

export interface RegionScore {
  /** The Region (Europe 0 .. South America 5), or 6 for Southeast Asia. */
  slot: number;
  /** The battlegrounds in the region, those a scoring counts besides the map's (Taiwan) included. */
  battlegrounds: number;
  /** US VP minus USSR VP. */
  net: number;
  /** The side the scoring would hand the game (Control of Europe), else null. */
  wins: "US" | "USSR" | null;
  us: RegionSideScore;
  ussr: RegionSideScore;
}

export interface RegionScores {
  regions: RegionScore[];
  /** Flags in the state this engine build does not know (an older or newer replay). */
  ignored_flags: string[];
}

/** A die roll and when it was made ("T3 AR4", "T2 Headline"). */
export interface RecentRoll extends DieRollState {
  /** In English, as data. */
  when: string;
  /** The same as a Message (game/rolls.ts `rollWhen`), which the tray shows; `when` without it. */
  whenMessage?: Message;
}

/** One country the other side's latest turn changed (influence deltas) or targeted. */
export interface LastTurnCountry {
  id: number;
  name: string;
  us: number;
  ussr: number;
  /** Chosen as a target (influence, coup, realignment, an event) -- a failed coup changes nothing. */
  targeted: boolean;
}

export interface LastTurn {
  player: "US" | "USSR";
  /** "Setup", "Headline" or "T3 AR4", in English, as data. */
  label: string;
  /** The same as a Message (game/last_turn.ts `turnLabel`), which the map shows. */
  labelMessage: Message;
  countries: LastTurnCountry[];
}

export interface CardMetadata {
  id: number;
  name: string;
  ops: number;
  side: 'us' | 'ussr' | 'neutral';
  age: string; // 'early war' | 'mid war' | 'late war'
  description: string;
  one_time: boolean;
}

export interface MapMetadata {
  countries: Array<{
    id: number;
    name: string;
    stability: number;
    battleground: boolean;
    region: string;
    subregions?: string[];
    superpower_adjacent: string | null;
    neighbours: string[];
    pos: [number, number];
  }>;
  regions: Record<string, {
    region_id: number;
    color: string;
    battlegrounds: string[];
  }>;
}

export interface MicroAction {
  decision_type: number;
  primary_id: number;
  secondary_id?: number;
  flags?: number;
}
