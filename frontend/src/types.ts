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
  decision_type: number; // 0..6
  decision_type_name?: string;
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
}

export interface CardInHand {
  id: number;
  name: string;
  ops: number;
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
  forced_card_player: 'US' | 'USSR' | 'NONE';
  forced_card_id: number;
  space_turns_used: { US: number; USSR: number };
  china_card: { holder: 'US' | 'USSR'; playable: boolean };
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
  draw_deck_count: number;
  decision_context: DecisionContext;
  legal_actions: LegalActions;
  is_terminal: boolean;
  terminal_utility: number;
  game_id?: string;
  seed?: number;
  step_index?: number;
  action_logs?: Array<{ step_index: number; turn: number; ar: number; phase: string; player: string; text: string }>;
  players?: { US: string; USSR: string };
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
