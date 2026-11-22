# Engine Core, Observations & Action Masks (`engine/`)

Audit at commit `fb9b817` (2026-11-22). Per [AGENTS.md](../../AGENTS.md) Invariant 11, none of these may be modified without the owner's prior approval.

---

## High-Severity Rules, State Machine & Card-Location Bugs

### E-1. `OLYMPIC_GAMES` uses `state.phasing_player` as sponsor on `ROLL_DIE`
- **Locations:** `engine/src/events/early_war.cpp:203-210`, `engine/src/card_dispatcher.cpp:686-749`
- **Cause:** On `CHOOSE_BRANCH` (`primary_id == 0`, Participate), `card_dispatcher.cpp:687-693` clears `decision_player = Player::NONE` and stages `RollType::OLYMPIC_GAMES` without saving `roll_actor = get_opponent(decision_player)`. On `ROLL_DIE` (line 714), it reads `Player sponsor = state.phasing_player;`.
- **Impact:** Whenever the USSR is phasing (`phasing_player == USSR`) and triggers a US event that fires Olympic Games as `Player::US` (e.g., Grain Sales, Star Wars, FYP → Grain Sales/Star Wars, or Missile Envy → Olympic Games), USSR is asked Participate vs Boycott; if USSR participates, USSR gets the `+2` sponsor modifier (winning 13/16 instead of 3/16). This also inverts P56's `roll_outcomes` expectation at that node.
- **Fix direction:** Set `state.ctx().roll_actor = get_opponent(state.ctx().decision_player);` before clearing `decision_player` at line 690, and read `roll_actor` (falling back to `phasing_player`) at line 714.

### E-2. `STAR_WARS` passes `Player::US` unconditionally to `trigger_event`
- **Locations:** `engine/src/card_dispatcher.cpp:1234-1253, 2004-2015`, `engine/src/events/early_war.cpp:139-141, 149-151`, `engine/include/ts/war_events.hpp:51-67, 105-175`
- **Cause:** Star Wars' mask offers any triggerable non-scoring card in the discard pile (including USSR cards), and `card_dispatcher.cpp:1244-1248` calls `trigger_event(state, card_id, Player::US)` unconditionally rather than computing `exec_player = (c_info.side == Player::USSR) ? Player::USSR : Player::US` (as headline dispatch does in `state_machine.cpp:270, 1135`).
- **Impact:** Picking **Korean War** or **Arab-Israeli War** via Star Wars forwards `Player::US` to `war_helpers::trigger_war`, setting `roll_actor = Player::US` — awarding US +2 Mil Ops, subtracting *USSR*-controlled neighbors, and on 4–6 giving **US** +2 VP and replacing **USSR** influence in South Korea / Israel with **US** influence.
- **Fix direction:** Compute `exec_player` from `c_info.side` (USSR for USSR cards, US otherwise) for `decision_player`, `event_has_effect`, and `trigger_event`.

### E-3. `settle_we_will_bury_you` at `SELECT_PLAY_MODE` misses `phasing_player == US` and `card == UN_INTERVENTION`
- **Location:** `engine/src/state_machine.cpp:1369-1373`
- **Cause:**
  ```cpp
  if (p == Player::US && state.current_phase == Phase::ACTION_ROUND &&
      settle_we_will_bury_you(state, mode == Resolution::EVENT)) {
      return true;
  }
  ```
  `trigger_grain_sales` (`mid_war.cpp:473-476`) enters `SELECT_PLAY_MODE` directly with `decision_player = Player::US` during USSR's Action Round, bypassing `SELECT_CARD`.
- **Impact:** When `WE_WILL_BURY_YOU_PENDING` is set and USSR plays Grain Sales on USSR's AR:
  - US playing the drawn card as `Resolution::EVENT` (any card, even Event-first on a USSR card) clears `WE_WILL_BURY_YOU_PENDING` for **0 VP** without playing UN Intervention.
  - US playing the drawn card for Ops or Space **immediately pays USSR 3 VP on USSR's AR**, even if US holds UN Intervention for US's upcoming AR (and can end the game at -20 VP before US's AR).
- **Fix direction:** Gate on `state.phasing_player == Player::US && state.ctx_stack_depth == 0` and pass `mode == Resolution::EVENT && card == card_ids::UN_INTERVENTION`.

### E-4. `clear_force_for_card` retains the `forced_card_id == MISSILE_ENVY` tautology
- **Location:** `engine/src/state_machine.cpp:827-834` (compare fixed guards at `action_mask.cpp:300-310`, `state_machine.cpp:1360-1367`)
- **Cause:** `forced_card_id` is always `MISSILE_ENVY` whenever `forced_card_player != NONE`, so `(state.forced_card_id == card || state.forced_card_id == card_ids::MISSILE_ENVY || state.forced_card_id == 0)` evaluates true for *any* `card`.
- **Impact:** When US owes Missile Envy and plays *any* card drawn by **Grain Sales** for Ops (or Event-first on a USSR card) — on USSR's AR1 or via a US Grain Sales headline — `commit_card_to_ops` clears `forced_card_player = NONE`, erasing the Missile Envy obligation.
- **Fix direction:** Remove the `|| state.forced_card_id == card_ids::MISSILE_ENVY` disjunct so only playing `MISSILE_ENVY` itself (or `forced_card_id == 0`) clears it.

### E-5. `trigger_five_year_plan` calls `relocate_played_card` before `trigger_event`, misplacing `SHUTTLE_DIPLOMACY`
- **Locations:** `engine/src/events/early_war.cpp:60-69`, `engine/src/card_dispatcher.cpp:307-313`
- **Cause:** Unlike every other event dispatch site, `trigger_five_year_plan` calls `relocate_played_card` *before* `trigger_event`.
- **Impact:** When FYP discards **Shuttle Diplomacy** (#73), `SHUTTLE_DIPLOMACY_ACTIVE` is not set yet, so `relocate_played_card` puts it in `DISCARD_PILE` instead of `ONGOING_EVENT`; `trigger_shuttle_diplomacy` then sets the active flag. Shuttle Diplomacy becomes simultaneously active and in `DISCARD_PILE` — retrievable by SALT, replayable by Star Wars, reshuffleable into deck/hands (where a later Asia/ME scoring at `scoring.cpp:248` moves it out of a player's hand/deck), and shown as `DISCARD` in observations.
- **Fix direction:** Compute `const bool fired = CardHandlers::event_has_effect(...)`, run `trigger_event`, and call `relocate_played_card(..., fired, Handover::Respect)` afterward.

### E-6. In-play card skip checks only top-frame `ctx().resolving_card`
- **Locations:** `engine/src/events/mid_war.cpp:409-416, 461`, `engine/src/events/early_war.cpp:42-73`, `engine/src/events/late_war.cpp:198-205`, `engine/src/state_machine.cpp:381-382, 1426-1427`
- **Cause:**
  1. When USSR plays `FIVE_YEAR_PLAN` for Ops (Event-first or Ops-first), FYP stays in `HAND_USSR_*` until `trigger_five_year_plan` returns. If FYP discards `GRAIN_SALES`, FYP pushes a child frame (`resolving_card = GRAIN_SALES`), and `trigger_grain_sales` skips only `state.ctx().resolving_card` (`GRAIN_SALES`), not `FIVE_YEAR_PLAN` on the parent frame.
  2. `trigger_latin_debt_crisis` (`late_war.cpp:198-205`) does not skip `state.ctx().resolving_card` at all.
- **Impact:**
  1. Grain Sales can draw `FIVE_YEAR_PLAN` itself into `HAND_US_KNOWN` (100% if USSR held no other cards); immediately afterward `relocate_played_card(FIVE_YEAR_PLAN)` moves it to `DISCARD_PILE` while Grain Sales' `SELECT_PLAY_MODE` holds `pending_op_card = FIVE_YEAR_PLAN`, letting US replay FYP from discard or decline Grain Sales (`card_dispatcher.cpp:1210`) and resurrect FYP back into `HAND_USSR_KNOWN`.
  2. US playing `LATIN_AMERICAN_DEBT_CRISIS` (2 Ops) for Ops under `CONTAINMENT_ACTIVE` (3 effective Ops) while holding no other 3-Op card triggers an empty `SELECT_CARD` prompt (forced `CONFIRM_DONE`) even without `ALWAYS_ASK_DISCARD`.
- **Fix direction:** Provide a helper checking all active stack frames (`0..ctx_stack_depth`) for `resolving_card` and `pending_op_card`, and use it in every hand-scanning event.

### E-7. `headline_first_card` / `headline_second_card` stay set all turn, blocking pushed-frame settling
- **Location:** `engine/src/state_machine.cpp:294-311, 397-403, 465-466, 499-511`
- **Cause:** `advance_headline_step` never clears `headline_first_card`/`headline_second_card` when the Headline phase ends. Both `settle_card_played_on_pushed_frame` (line 400) and headline `advance_after_ops` (line 501) skip any card matching either ID.
- **Impact:**
  - **In `Phase::ACTION_ROUND`:** If any non-starred card `C` headlined this turn re-enters USSR's hand (via **SALT Negotiations**, or **Missile Envy** headlined by US while USSR is trapped under Bear Trap + Purge) and is later drawn by **Grain Sales** on a pushed frame (USSR Event-first Grain Sales, Star Wars → Grain Sales, or FYP → Grain Sales) and played by US for Ops, `settle_card_played_on_pushed_frame` returns `false` — **dropping `C`'s owed USSR Event and leaving `C` permanently in US's hand**.
  - **In `Phase::HEADLINE`:** If Headline 1 is US's `MISSILE_ENVY` (moves `MISSILE_ENVY` to USSR's hand) and Headline 2 is USSR's `GRAIN_SALES` (draws `MISSILE_ENVY` to US's hand and US plays it for Ops), `op_card != state.headline_first_card` skips relocation and leaves `MISSILE_ENVY` in US's hand.
- **Fix direction:** Clear `headline_first_card`/`headline_second_card` when exiting `Phase::HEADLINE`, and in `Phase::HEADLINE` only exclude the *currently resolving* headline card when it is still `HEADLINE_COMMITTED` (not after it has been handed into a hand).

### E-8. Grain Sales leaks `allow_early_stop = 1` onto drawn US/Neutral Events
- **Locations:** `engine/src/events/mid_war.cpp:480`, `engine/src/state_machine.cpp:1164-1172, 1384-1398`, `engine/src/events/early_war.cpp:203-210`, `engine/src/events/late_war.cpp:288-297`, `engine/src/action_mask.cpp:715-717`, `engine/src/card_dispatcher.cpp:687-712`
- **Cause:** `trigger_grain_sales` sets `allow_early_stop = 1`; playing a drawn US/Neutral card as `Resolution::EVENT` clears only `resolving_card = 0`, leaving `allow_early_stop = 1`. Neither `trigger_olympic_games` nor `trigger_wargames` clears it.
- **Impact:** `action_mask.cpp:715` adds illegal `CONFIRM_DONE` (slot 208) alongside `BRANCH 0` and `BRANCH 1`. In **Olympic Games**, choosing `CONFIRM_DONE` (`primary_id = 255`) fails `primary_id == 0` and executes **`else` (Boycott — degrading DEFCON / potential DEFCON suicide + 4 Ops)**; in **Wargames** it duplicates `BRANCH 1` (Pass). For `HOW_I_LEARNED`, `CHERNOBYL`, `CIA_CREATED`, `ABM_TREATY`, `KAL_007`, and `TEAR_DOWN_THIS_WALL`, stale `allow_early_stop = 1` also leaks into `ctx_slots::ALLOW_EARLY_STOP` in the observation.
- **Fix direction:** Clear `state.ctx().allow_early_stop = 0;` when committing any `Resolution` in `SELECT_PLAY_MODE` (`state_machine.cpp:1384`).

### E-9. `fire_owed_event` does not clear `start_influence_nodes` or `op_mode`
- **Locations:** `engine/src/state_machine.cpp:369-384`, `engine/src/ops.cpp:149-170`, `engine/src/action_mask.cpp:378-386`, `engine/src/observation.cpp:122-123, 405-408, 491-502`
- **Cause:** `fire_owed_event` runs the opponent's owed Event on the same frame where `start_influence_nodes` still holds the *phasing player's* start-of-round bitmap (with `START_INFLUENCE_PENDING == 0`) and `op_mode` still holds `COUP` or `REALIGN` from the Ops-first play.
- **Impact:**
  1. **Rules + observation bug after Ops-first `OPS_INFLUENCE`:** `Operations::can_place_influence(state, opp, cid)` (`ops.cpp:161-169`) validates `opp`'s adjacency against the *phasing player's* snapshot. At `SELECT_OP_MODE` for event-granted Ops (**CIA Created**, **Lone Gunman**, **Grain Sales** empty hand / hand-back, **KAL-007**, **Glasnost**), `OpMode::INFLUENCE` can be illegally omitted from the mask, and `board[i, 19]` (`can_my_place`) is corrupted throughout the owed Event.
  2. **Observation bug after Ops-first `OPS_COUP`/`OPS_REALIGN`:** `observation.cpp:494` gates `obs_features::PLACEMENT_LEGAL` on `(phase == SETUP || op_mode == INFLUENCE)`, silently disabling `PLACEMENT_LEGAL` during owed event placement/removal runs (Marshall Plan, Decolonization, Comecon, Socialist Governments, Suez Crisis, Voice of America, De-Stalinization, etc.) while `observation.cpp:405-408` reports stale `OP_MODE`.
- **Fix direction:** Clear `start_influence_nodes = {0, 0}`, `op_mode = OpMode::INFLUENCE` (or neutral), and `START_INFLUENCE_PENDING` in `fire_owed_event`.

---

## Medium-Severity Rules Bugs

### E-10. Flower Power skipped when US headlines `KOREAN_WAR` or `ARAB_ISRAELI_WAR`
- **Locations:** `engine/src/state_machine.cpp:269-286, 1134-1142`, `engine/include/ts/war_events.hpp:170-172`
- **Impact:** `exec_player` (and thus `roller`) is `Player::USSR`, so `war_events.hpp:170` (`roller == Player::US`) is false and headline dispatch never charges Flower Power; US pays 0 VP instead of -2 VP.

### E-11. `UN_INTERVENTION` can trigger during `Phase::HEADLINE` via `MISSILE_ENVY`
- **Locations:** `engine/src/card_dispatcher.cpp:162-178, 1130-1138`, `engine/src/events/mid_war.cpp:169-182`
- **Impact:** `can_trigger_event(state, UN_INTERVENTION, player)` does not check `state.current_phase != Phase::HEADLINE`. A headlined Missile Envy taking UN Intervention (only 1-Op cards in hand, or Red Scare/Purge tying 2-Op cards with 1-Op UN Intervention) executes UN Intervention during Headline, violating the rule enforced for Grain Sales and Star Wars (`engine/AGENTS.md` §23).

### E-12. Trapped or Missile-Envy-forced player cannot decline optional AR8
- **Location:** `engine/src/action_mask.cpp:109-203` vs `226-228`
- **Impact:** Blocks 3 (Missile Envy) and 4 (Quagmire/Bear Trap) return early before line 226 (`if (action_round == 8) mask_out[0] = 1;`), forcing a player with North Sea Oil / Space Station to spend a card on optional AR8 even with no held scoring cards.

### E-13. `SELECT_OP_MODE` decline in Headline skips card relocation
- **Location:** `engine/src/state_machine.cpp:1461-1466`
- **Impact:** When `current_phase == HEADLINE && ctx_stack_depth == 0`, declining `SELECT_OP_MODE` calls `advance_headline_step` directly instead of `advance_after_ops`, skipping `op_card` relocation at lines 500–511 when a headlined Missile Envy took an opponent card into hand for Ops and all three op modes are illegal.

---

## Observation Leaks & P56 Interaction Bugs

### E-14. Hidden-information leak: `CardLocation::PEEKED_TEMP` exposes Our Man in Tehran's 5 secret cards to `Player::USSR`
- **Locations:** `engine/src/observation.cpp:245-247`, `engine/src/events/mid_war.cpp:631-633, 656`
- **Cause:** `extract_into` maps `CardLocation::PEEKED_TEMP` to `card_slots::PEEKED` without checking `my_player == Player::US`.
- **Impact:** Actively hit in **P56 training** whenever USSR plays Our Man in Tehran Event-first, finishes an Ops-first play (including via `roll_outcomes` on a coup/realignment), or commits the 2nd headline resolving Tehran, because `keep_afterstate` / `roll_outcomes` (`bindings/ts_bindings.cpp:289, 1876`) extract observations from `side = Player::USSR`.
- **Minor related note (`observation.cpp:290-296, 483-486`):** Extracting from the *opponent's* perspective during a normal AR `SELECT_PLAY_MODE` exposes the hidden card via `ACTIVE_CARD = 1.0` and `OPS_BUDGET[0]` (only relevant to external probes calling `extract_observation_features(state, opponent, features)` mid-play).

### E-15. `extract_observation_features` fails to resolve `perspective == Player::NONE`
- **Location:** `engine/src/observation.cpp:95-100` vs `472-534`
- **Impact:** `extract_into` resolves `Player::NONE` to `decision_player` (else `phasing_player`, else `US`), but `extract_observation_features` passes raw `perspective == Player::NONE` into all optional feature blocks (`ops_modifier`, `PLACEMENT_LEGAL`, `PLACEMENT_ORDER`, `revealed_headline`), silently zeroing them.

### E-16. First headliner's afterstate observation is blind to the chosen card (P56 interaction)
- **Locations:** `engine/src/state_machine.cpp:1081-1095`, `engine/src/observation.cpp:213-332`
- **Cause:** During `headline_stage == 0`, after the first player selects card `C`, `card_locations[C]` remains `HAND_*` (not `HEADLINE_COMMITTED`), `pending_op_card` and `resolving_card` remain `0`, and `extract_into` ignores `headline_us_card`/`headline_ussr_card`.
- **Impact:** `extract_observation_features(after, first_picker, features)` is bit-for-bit identical regardless of which headline card was chosen (and identical to pre-decision), so P56's afterstate critic receives zero signal about the first headliner's move.

---

## Action Mask, Canonical Order, Macro View & Engine Helper Bugs

### E-17. `forced_event_target` ignores `CANONICAL_*` order and breaks `Engine::auto_advance_step`
- **Locations:** `engine/src/engine.cpp:21-75, 195-203`, `engine/include/ts/canonical.hpp:21-26`, `engine/src/state_machine.cpp:910-920, 934, 946-949`
- **Impact:**
  - `SUEZ_CRISIS` and `EAST_EUROPEAN_UNREST` are in `canonical::EVENT_CARDS`, yet `forced_event_target` iterates fixed country orders (`{UK, France, Israel}` and `0..83`). Under `CANONICAL_FIRST`, step 2 can propose a country ranked below `floor` that `StateMachine::step` refuses, causing `auto_advance_step` to break early with `final_mask_valid = false` on a single-choice state.
  - Under `CANONICAL_LEARNED`, step 1 of any forced multi-country Suez/EEU refuses without `priority_given`, breaking `auto_advance_step`.
  - When a `CANONICAL_LEARNED` run ends on a pushed frame and pops (`state_machine.cpp:934`), it returns before clearing `canon_prio` / `canon_prio_tag` (lines 946–949).

### E-18. `ActionMask::apply_macro_expansion` crosses stage/event boundaries and offers macros on removals/coups
- **Location:** `engine/src/action_mask.cpp:813-838`
- **Cause:** The expansion loop checks only `decision_type == POINT_NODE && decision_player == side && control != side`, never verifying that `resolving_card`, `event_stage`, or `op_mode` stayed unchanged or that the step is a placement.
- **Impact:**
  1. **`DE_STALINIZATION` remove → place:** If USSR has 4 influence in an uncontrolled country `c`, macro expansion removes 4 in `STAGE_REMOVE` and immediately places 2 right back into `c` in `STAGE_PLACE`.
  2. **`OpMode::COUP` → `CUBAN_MISSILE_CRISIS`:** Under `CMC_ACTIVE_USSR` with ≥2 US influence in uncontrolled West Germany/Turkey at DEFCON 5, macro slots are offered at the coup target choice (step 1 targets the coup, step 2 cancels CMC from that country).
  3. **Removal events:** `SUEZ_CRISIS`, `SOCIALIST_GOVERNMENTS`, `THE_VOICE_OF_AMERICA`, `MARINE_BARRACKS_BOMBING`, and `DE_STALINIZATION` Stage 1 all get "until-control" macro slots offered despite being removals.

### E-19. `Engine::is_held_scoring_game_over` / `is_held_scoring_loss` misclassifies AR8 and Mil-Ops wins
- **Location:** `engine/src/engine.cpp:258-269` (consumed by `ai/rewards/reward_calculator.py:224-226`, `bindings/ts_env.py:588-590`)
- **Cause:** Computes `max_ar = (state.turn <= 3) ? 6 : 7` ignoring `NORTH_SEA_OIL_ACTIVE` and `SpaceRace::has_space_station_ar8`, and does not distinguish Phase E (`evaluate_military_ops` reaching ±20 VP) from Phase F's held-scoring check.
- **Impact:** Any Europe Control, 20 VP, Wargames, or CMC suicide win on AR8 (`action_round == 8 > 7`) while either player holds a scoring card — or a Phase E Mil-Ops ±20 VP win where the *winner* holds a scoring card — is misclassified as a held-scoring loss.

---

## Minor / Subtle C++ Edge Cases

- **`ALWAYS_ASK_DISCARD` on `MISSILE_ENVY` diverges in stack/context/observation** (`engine/src/events/mid_war.cpp:172` vs `engine/src/card_dispatcher.cpp:1131-1138`): direct 1-card path pushes a fresh `DecisionContext{}`, while the tie/`ALWAYS_ASK_DISCARD` path reuses the frame leaving `ctx_stack_depth = 0`, `remaining_steps = 1`, `OP_MODE = INFLUENCE`, and both `MISSILE_ENVY` and `chosen_card` marked `ACTIVE_NOW = 1.0`.
- **`SUMMIT` forced-dice convention differs from `REALIGNMENT`/`OLYMPIC_GAMES`** (`engine/src/card_dispatcher.cpp:962-963` vs `state_machine.cpp:1617-1618`): `SUMMIT` reads `primary_id` = US die and `secondary_id` = USSR die regardless of actor (harmless for P56's symmetric 11-difference distribution, relied on by `ts_replayer_convert.py`).
- **Padding-byte `memcmp` on `DecisionContext`** (`engine/src/state_machine.cpp:1208`): `DecisionContext` (`alignas(64)`, 128 bytes) has 51 padding bytes; copy-construction followed by `std::memcmp` is not strictly guaranteed by C++ to preserve padding bytes (works under clang in practice).
