# Twilight Struggle AI: C++ Simulation Core Guide

This directory contains the zero-allocation, high-throughput simulation engine for the Deluxe
Edition of **Twilight Struggle** (110 cards). Section numbers are cited from elsewhere; keep them.

---

## 1. Core Architectural Pillars

1. **Zero heap allocation in the simulation core**: `GameState`, `DecisionContext` and `MicroAction` are fixed-size; `GameState` is trivially copyable and `sizeof(GameState) <= 4096`.
2. **Micro-decision state machine**: every multi-step action (Ops, multi-target events, headlines, space race) is a sequence of atomic steps (`DecisionType`).
3. **Throughput**: over 2,000,000 steps/second on one desktop core (`ts_benchmark` reports the machine at hand), which is what makes RL-scale self-play affordable.
4. **Bit-for-bit determinism**: all randomness comes from the SplitMix64 `Prng` in `state.rng_state`.
5. **Uniform sub-decisions**: `SELECT_OP_MODE`, `POINT_NODE`, `CHOOSE_BRANCH`, ... are handled the same way in `Phase::HEADLINE` and `Phase::ACTION_ROUND`.

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Engine Documentation Synchronized**:
> Whenever adding or modifying card handlers (`events/*.cpp`), updating `GameState`, adding tests in `tests/`, or adjusting fuzzer/sanitizer flags, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
>
> **Mandatory Engine Change Rule**:
> Agents must **NEVER make any changes to the C++ engine (`engine/`) without explicitly asking the user and obtaining prior confirmation**.

---

## 3. Directory Structure

```
engine/
├── CMakeLists.txt              // ts_engine_core plus ts_tests, ts_fuzz, ts_fuzz_events, ts_benchmark
├── include/ts/
│   ├── types.hpp               // Enums: Player, Phase, WarEra, CardLocation, DecisionType, ActionType, Region, SubRegion, Resolution, TimingBranch, OpMode
│   ├── constants.hpp           // Country IDs, Card IDs (1..110), 64-bit effect flags, bit masks
│   ├── micro_action.hpp        // 4-byte packed MicroAction (alignas(4))
│   ├── game_state.hpp          // trivially copyable GameState; obs_flags / obs_features bits
│   ├── prng.hpp                // SplitMix64 PRNG
│   ├── map_data.hpp            // 84 countries, adjacency as 128-bit masks
│   ├── card_data.hpp           // 110 cards (Ops, side, era, asterisk, scoring)
│   ├── scoring.hpp             // region scoring (Presence/Domination/Control), victory checks;
│   │                           // ScoringTap: a thread-local report of each region's scoring (P34's labeller)
│   ├── ops.hpp                 // influence, coups, realignments, dynamic costs
│   ├── defcon.hpp              // DEFCON-1 game end, provoked/unprovoked classification
│   ├── space_race.hpp          // space race tracks, rewards, abilities
│   ├── card_handlers.hpp       // event handlers and sub-decision dispatch
│   ├── war_events.hpp          // shared helpers for the five war cards
│   ├── invariant.hpp           // invariant_failed(): abort naming what broke
│   ├── action_mask.hpp         // per-decision masks and the flat mask (220 slots; the _212 names are historical)
│   ├── state_machine.hpp       // turn / action-round lifecycle, headline resolution
│   ├── observation.hpp         // observation extractor (ObservationBufferV23, the one layout)
│   ├── serialization.hpp       // binary snapshot and JSON serialization
│   ├── engine.hpp              // ts::Engine public interface
│   └── cached_state.hpp        // CachedState: a state that keeps its own legal mask (§6b)
├── src/                        // implementations of the above; events/early_war.cpp (cards 1-35, 103-106),
│                               // events/mid_war.cpp (36-81, 107-108), events/late_war.cpp (82-102, 109-110),
│                               // card_dispatcher.cpp (event dispatch, sub-decision router, mask filters)
└── tests/
    ├── test_framework.hpp, game_test_wrapper.hpp, test_main.cpp
    ├── test_map.cpp, test_scoring.cpp, test_ops.cpp  // topology, scoring formulas, Ops/coups/NATO
    ├── test_cards_{early,mid,late}.cpp, test_card_interactions.cpp, test_card_edge_cases.cpp
    ├── test_bugs_regression.cpp // one regression per fixed card bug
    ├── test_rules_audit.cpp    // one regression per rule fixed by the 2026-09-28 audit (§12)
    ├── test_defcon_suicide.cpp, test_reentrancy.cpp, test_state_lifecycle.cpp, test_states.cpp
    ├── test_full_game.cpp      // full games to final scoring
    ├── test_auto_advance.cpp   // Engine::auto_advance_step: forced decisions taken without asking
    ├── test_cached_state.cpp   // CachedState against the plain engine, byte for byte (§6b)
    ├── test_fuzz_influence_placement.cpp
    ├── test_fuzz.cpp           // ts_fuzz: invariant fuzzer (--games <N>, --steps <N>, --seed <S>)
    ├── test_fuzz_events.cpp    // ts_fuzz_events: same, biased toward events (--event-bias)
    └── test_benchmark.cpp      // ts_benchmark
```

---

## 4. Micro-Decision Pipeline & Action Representation

The engine splits turns into a stream of 4-byte `MicroAction`s:

```cpp
struct alignas(4) MicroAction {
    DecisionType decision_type; // SELECT_CARD, SELECT_PLAY_MODE, SELECT_OP_MODE, POINT_NODE, CHOOSE_BRANCH, ROLL_DIE
    uint8_t      primary_id;    // Card ID (1..110), Country ID (0..83), Branch ID (0..7), Resolution, OpMode, or CONFIRM_DONE (0x80)
    uint8_t      secondary_id;  // sub-choice / quantity / manual die roll
    uint8_t      flags;         // modifiers / opponent manual die roll
};
```

**P23 / E4.1: the merged-influence view.** `ActionMask::generate_flat_mask_merged` and the
`merged_influence` overloads of `Engine::get_flat_action_mask` / `Engine::step_flat` offer, at the two
op-choice nodes (`SELECT_PLAY_MODE`, `SELECT_OP_MODE`), "ops for influence, first point in X" as the
NODE slot X, and "ops for influence, place nothing" as `OPS_INFLUENCE`. Both are **defined** as the
two E4 steps they name (`OPS_INFLUENCE` then X, or then `CONFIRM_DONE`): the mask is read off a copy
of the state after `OPS_INFLUENCE` and a composed step is applied atomically, so no rule, `GameState`
field or observation slot changes, and with the view off everything is bit-identical to E4.
`tests/bindings/test_merged_influence.py` pins the definition. Where the commit itself ends the game
(a We Will Bury You penalty falling due, §9a) the view offers `OPS_INFLUENCE` as the bare commit;
anywhere else a commit that does not reach the mover's placement offers no influence and calls
`report_anomaly` — it composes, it never substitutes (never seen over 96,010 influence-legal nodes).

**P42: the macro (until-control) view.** The flat space is **304** wide (`flat_slots::SIZE`; the base views use
slots 0..219, `flat_slots::BASE_SIZE`, and never offer the rest). In `ActionView::MACRO` slot `220 + c`
(`flat_slots::MACRO_NODE + c`) means "place influence in country c, point by point, until the mover controls it or
the placement ends" — **defined** as repeated one-point `POINT_NODE c` steps through `StateMachine::step`
(`ActionMask::apply_macro_expansion`), stopping at game end, at a change of decision type or decider, or at control.
`generate_flat_mask_macro` offers it at a legal node only where that expansion places **at least 2 points** (where
one point controls, the base slot already says it). `Engine::step_flat(state, idx, auto_advance, ActionView::MACRO)`
expands it, restores the state on any refusal, then auto-advances. Exclusive with E4.1. No rule, `GameState` field or
observation slot changes, the replay records the base steps, and with the view off everything is bit-identical to
E4 (`tests/test_macro_view.cpp`: base views never offer 220..303; every macro step equals its base steps).

**`ROLL_DIE` is the one decision whose `primary_id` is data, not an index**: the acting player's
forced die, `secondary_id` the opponent's, **0 meaning "roll normally"**. So the flat mask cannot
constrain it — the range is checked in `StateMachine::step` beside the `decision_type` guard; it is
decoded first, because the 255 "no selection" sentinel would read as a forced 255; and
`generate_flat_mask_212` has an explicit `ROLL_DIE` case, because its "ensure one legal action"
fallback would otherwise supply a wrong-but-accepted confirm. Prefer a real case to that fallback.
Chance nodes in `Engine::auto_advance_step` and the batch runner are resolved with an explicit
`{ROLL_DIE, 0, 0, 0}`.

**`step` returns `[[nodiscard]] bool`, and ignoring it does not compile** (`StateMachine::step`,
`Engine::step`, `Engine::step_flat`; `-Werror=unused-result`, MSVC `/we4834`), so every unchecked
call site fails at build time rather than only those a test reaches. Throwing was rejected: `step`
is `noexcept`, runs inside the batch runner's parallel loops where an escaping exception is
undefined behaviour, and the batch path needs a per-game verdict. (Adding it found a production
loop that would have spun forever on a refused `ROLL_DIE`.)

---

## 5. How to Build, Test, and Benchmark

The root `CMakeLists.txt` builds `engine/` and `bindings/` into `build/release`. **The engine is
built with clang** and CMake refuses any other compiler (+20% over GCC, identical games; install
with `apt-get install clang llvm lld` or `tools/scripts/install_clang_userspace.sh`). The root
`CMakeLists.txt` is part of the engine fingerprint, so a change of compiler or flags marks every
build stale.

Native builds use `-O3 -march=native` and **thin LTO** (`TS_LTO`, on; `TS_LTO_MODE` thin or full,
measured the same; `-DTS_LTO=OFF` disables), so calls across source files can be inlined (~9% off a
single-threaded search's tree time). Static libraries then hold LLVM bitcode, archived with
`llvm-ar` or GNU `ar` through LLVM's gold plugin. The browser build has neither LTO nor `-march`;
`tests/web/test_wasm_engine.py` holding the two to identical games shows LTO changed only speed.

```bash
cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j

./build/release/engine/ts_tests
TS_TEST_FILTER=RulesAudit ./build/release/engine/ts_tests         # tests whose name contains a substring
./build/release/engine/ts_fuzz --games 10000                       # invariant fuzzer
./build/release/engine/ts_fuzz --steps 5000000 --seed 42
./build/release/engine/ts_fuzz_events --steps 1000000 --event-bias 95
./build/release/engine/ts_benchmark
```

The sanitizer build (ASan + UBSan, `build_san/`) is in `CLAUDE.md`; run `ts_tests` and `ts_fuzz`
from it the same way, with `ASAN_OPTIONS=detect_leaks=0`.

---

## 6. Guidelines for Extending the Engine

1. **Never allocate heap memory in state types**: `static_assert(std::is_trivially_copyable_v<GameState>)` must pass.
2. **Deterministic PRNG**: dice and shuffles use `Prng::roll_d6(state.rng_state)` / `Prng::random_index(state.rng_state, n)`.
3. **Pass / confirm**: multi-step cards (*Suez Crisis*, *Muslim Revolution*, *Independent Reds*, *Special Relationship*, ...) must accept an early pass (`primary_id == 0` or `CONFIRM_DONE`) when no eligible target remains.
4. **Influence Ops are spent in full** (2026-09-26): an influence play opens with
   `allow_early_stop = 0`, and the mask offers `CONFIRM_DONE` (`DecisionContext::is_ops_influence_play`)
   only once no country can take the next point. Realignment keeps its stop; a coup never had one.
   **The observation keeps the old value**: `ctx_slots::ALLOW_EARLY_STOP` reads 1 throughout an
   influence play, as every checkpoint saw it, until the observation is next revised.
5. **Passing an action round**: a player holding cards must play one; `SELECT_CARD` offers the pass
   (mask index 0, flat 211) only to a player with none. Exceptions: a holder whose only playable card
   is the China Card is offered both it and the pass, and the eighth action round (North Sea Oil,
   Space Station) is always declinable. A player with an empty hand is passed by
   `advance_after_action_round` without reaching the mask.
6. **The China Card can be raced**: `SELECT_PLAY_MODE` offers Operations and, where the box's Ops
   requirement is met, the Space Race (a poor play, not an illegal one; ts-replayer game 247, turn 10
   AR4). Either way it passes face down to the opponent, so `SpaceRace::attempt_space` sets
   `china_card_holder` / `china_card_playable`; its Asia bonus pays nothing on the space track.
7. **Headline cards leave hand on commitment**: once both are selected each becomes
   `CardLocation::HEADLINE_COMMITTED`, so a hand-reading card (Cambridge Five, Missile Envy, Grain
   Sales, Lone Gunman, CIA Created, Aldrich Ames, Five Year Plan, Blockade, Latin American Debt
   Crisis) never finds it in a hand. It is a waypoint overwritten by the post-resolution cleanup,
   deliberately not the discard pile, which Star Wars and SALT read.
8. **Ops spent in a headline discard their own card**: `advance_after_ops` relocates
   `pending_op_card` in the HEADLINE phase, since the headline machinery clears only the two
   headline cards — a card played *through* one (Grain Sales) would otherwise stay in hand.
9. **Traps judge a card on its effective Ops**: Quagmire / Bear Trap eligibility is
   `Operations::get_effective_ops(state, card, p)` with no target region (Containment, Brezhnev
   Doctrine, Red Scare/Purge move it), and the mask and the `SELECT_CARD` handler must agree.
10. **A card in play is not in hand**: an Ops card stays in its owner's hand until the play finishes,
   so Grain Sales, Five Year Plan, Terrorism and Missile Envy skip whatever `resolving_card` names.
11. **Cancelling Cuban Missile Crisis is a choice, at two moments**: at the head of the payer's own
   action round (declining allowed and usual) and inside a coup they make (declining loses the game).
   Both are a `POINT_NODE` with `resolving_card == CUBAN_MISSILE_CRISIS` (US: West Germany or Turkey;
   USSR: Cuba); the coup's is asked before its die, and `ctx().pending_roll == RollType::COUP` tells
   the two apart. With one payer or none, `execute_coup` settles it inline.
12. **An opponent's card owes its Event in a headline too**: since P17 retired
   `CHOOSE_TIMING_BRANCH`, `Resolution::EVENT` on an opponent card means event-first and any `OPS_*`
   ops-first; `timing_branch` survives as internal state. `advance_after_ops` fires the owed Event on
   the headline path as well as the action round's (ts-replayer game 137, turn 4: Grain Sales takes
   Willy Brandt, whose Event must follow). A headlined card itself is excluded: it was played as its
   Event.
13. **Shuttle Diplomacy removes a battleground, and a country with it**, from USSR totals in Asia or
   Middle East scoring — only if there is a battleground to take (game 323, turn 10 AR1: Lebanon only,
   worth 5 to the US, not 8). Final scoring is exempt.
14. **Defectors cancels the USSR headline however it reaches the table**: the headlined case is
   settled by the check on `headline_us_card` in `step`; any other route (Five Year Plan, Grain Sales,
   Star Wars) fires it after commitment, so `trigger_defectors` cancels the *second* headline card
   where the USSR owns it and `headline_stage < 2` (game 313, turn 2).
15. **Each headline card resolves in its own frame**: `advance_headline_step` gives the second card a
   fresh `DecisionContext`, so per-event state such as the one-per-country visited bitmap does not
   carry over; lasting effects live in the state (game 92, turn 7: Colonial Rear Guards then
   Decolonization).
16. **A trap never holds a scoring card past the turn**: a scoring card is playable out of a trap when
   nothing in hand is eligible or the player holds as many scoring cards as action rounds left
   (game 63, turn 4 AR7).
17. **A headline ends with the stack empty**: `advance_after_ops` unwinds the stack on the HEADLINE
   path before advancing the headline, stopping at a frame holding an unanswered `SELECT_OP_MODE`
   (Ops still owed inside the headline).
18. **Always update tests when changing card logic**: add cases in `engine/tests/` for every new behaviour, interaction or edge case.

---

## 6a. The observation is not yours to change

Adding a feature, removing one, changing what a slot means or changing the width are representation
decisions that belong to the project owner. (An `obs_features` bit may also be a *rule bit*,
`obs_features::RULES`: no width and no change to the observation, but the rule a model's seat plays,
recorded with the model like any feature -- section 19.) Ask first. A network reads fixed slices, so a changed
observation never raises — the checkpoint loads and misreads, and a content change at unchanged
width slips past the width checks — and every checkpoint and `(seed, actions)` dataset is
invalidated (`CLAUDE.md`, "Never change the observation without asking"). Two standing
preferences: no large vector for a rare mechanism (a per-country or per-card bit for one card costs
84 or 110 floats), and a partial feature is worse than none.

**An approved addition is an appended block, not an edit of the base.** `obs_features`
(`include/ts/game_state.hpp`) names each optional block by a bit and gives its width
(`extra_width`); `extract_observation_features` (`src/observation.cpp`) writes the base — the same
code as `Observation::extract` — then each requested block in bit order. Bit 0 is retired with
`STAGED_CARDS` and never reused. A new block takes the next bit, appends after the existing ones,
and is mirrored in `ai/models/ladder_net.OBS_FEATURE_WIDTHS` (a test compares the two).

| bit | name | width | contents |
|:---|:---|---:|:---|
| 1<<1 | `OPS_BUDGET` | 3 | at SELECT_PLAY_MODE, `Operations::grant_ops_for_card(pending_op_card, decider)/5`, else 0; my per-card Ops modifier; the opponent's (Containment/Brezhnev +1, Red Scare/Purge −1, summed) |
| 1<<2 | `HEADLINE_REVEAL` | 0 | **no block: edits the base's card rows** (owner, 2026-10-20). While the side holding Man in Earth Orbit alone chooses its headline, the opponent's -- chosen first and revealed (`SpaceRace::revealed_headline`) -- shows as `KNOWN_OPPONENT_HAND` with `ACTIVE_CARD` = `ACTIVE_NEXT`, as the second headline does while the first resolves; the base layout leaves it hidden. The one bit that changes base slots, so a model reads it only if trained with it and every other model's view is unchanged. A new network reads it by default since E8 (`tools/train.py --obs-features`) |

**The extractor computes each country's control once** (2026-10-06) and reads it everywhere —
control features, realignment neighbour counts (from a list of the 112 borders), regional scoring
(`Scoring::evaluate_region_with_control`) — takes placement and coup legality from
`Operations::placeable_countries` / `coupable_countries`, and copies static board features from a
per-side table: 7.7 → 4.6 µs per observation, bit-identical on 318,198 observations. Any change here
must stay bit-identical: `tests/engine_logic/test_observation_golden.py` pins it.

---

## 6b. A state that keeps its legal mask: `CachedState` (2026-10-06)

`include/ts/cached_state.hpp` keeps a state's flat mask with it, for the C++ containers that own
states (the search tree, the batch runner), so validating a step, auto-advance and the next
decision share one mask generation (3.15 → 1.19 per search node).

* **The state is private.** `state()` is a const reference; the only mutable access is `modify(f)`,
  which invalidates the mask before and after `f`; `step`, `step_flat` and `auto_advance` go through
  it, so a write that skips invalidation does not compile.
* **One write keeps the mask: `set_rng`.** The mask reads neither the RNG nor the three die-roll
  fields `step` clears before validating (`CachedState.MaskIgnoresRngAndLastRolls`), which lets a
  search child be validated against its parent's mask.
* `StateMachine::step(state, action, legal)` validates against a mask the caller holds for exactly
  this state (the two-argument form generates it); `Engine::auto_advance_step(state, max,
  final_mask, &valid)` hands back the mask of the decision it stops at.
* **Checked as well as argued.** `set_mask_cache_checks(true)` regenerates the mask on every cached
  read and aborts on a difference; the engine's CachedState tests and `tests/conftest.py` (the whole
  Python suite) turn it on, and `CachedState.PlaysExactlyAsThePlainEngine` plays 100 random games
  side by side byte for byte.

Python's `GameState` is the plain struct with no cache, so nothing Python writes can leave one stale.

---

## 6c. What each Event did to VP and DEFCON: `last_event_tracks` (2026-10-19)

A step can resolve several Events -- the second headline commit resolves both headlines, Missile
Envy fires the card it takes -- and a log that compares the state before and after the step sees
only their sum. At turn 8's headline of the workbench game `nuclear_test_ban`, Nuclear Test Ban's
+1 VP for the US and Pershing II's +1 VP for the USSR netted to nothing and neither was logged.

* `GameState::last_event_tracks` (with `last_event_track_count`) holds one `EventTrackRecord`
  (card, VP delta toward the US, DEFCON delta) per Event of the current step that moved either
  track, in the order each finished. `StateMachine::step` clears it with `last_roll`; like the
  die-roll fields, the mask never reads it, so `CachedState` validation is unaffected.
* It is written in one place: the public `CardHandlers::trigger_event` and `handle_event_step`
  run their private bodies (`run_trigger_event`, `run_event_step`) and record the change. An
  Event that set off another is recorded **net** of it -- the inner one recorded itself first --
  so Missile Envy taking Nuclear Test Ban records Nuclear Test Ban's VP once.
* Display only: no rule and no observation reads it, and games are bit-identical. A change outside
  an Event (Flower Power's charge, U-2 Incident's rider, the turn's cleanup) has no record.
* Six slots, the depth `ctx_stack` nests Events to; a seventh is reported through
  `report_anomaly` rather than dropped silently. Tests: `StateLifecycleTest.HeadlineState_EachEventRecordsItsOwnTrackChange`,
  `StateLifecycleTest.EventTracks_NestedEventIsRecordedOnce`.

## 7. Why hand knowledge lives in `CardLocation`

`card_locations` distinguishes `HAND_US_KNOWN` from `HAND_US_UNKNOWN` (and the USSR pair) rather
than carrying a parallel "known" bitset, and the bare `HAND_US` / `HAND_USSR` names deliberately do
not exist:

```
UNAVAILABLE=0  DRAW_DECK=1  HAND_US_UNKNOWN=2  HAND_US_KNOWN=3  HAND_USSR_UNKNOWN=4
HAND_USSR_KNOWN=5  DISCARD_PILE=6  REMOVED_FROM_GAME=7  ONGOING_EVENT=8  PEEKED_TEMP=9
HEADLINE_COMMITTED=10
```

A holder always knows their own hand, so "known" can only mean known to the non-holder; one field
costs no bytes and makes knowledge monotone for free (every write that moves a card out of a hand
overwrites its knownness, where a bitset would need clearing at a hundred-odd sites).

Access goes through `is_in_any_hand`, `in_hand_of`, `known_to_opponent`, `hand_of`, `hand_holder`
and `revealed`, never a bare comparison: removing the bare names turned ~50 sites that would have
silently treated a known card as not in hand into compile errors. **Do not reintroduce them.**

A known hand card behaves identically to an unknown one in **every rule** and differs **only in the
observation**: an opponent-held card is `KNOWN_OPPONENT_HAND` when known and folds into
`DECK_OR_HIDDEN` when not (anything else would leak the hand); to its holder both are `MY_HAND`.

---

## 8. Free-coup events must go through `Operations::can_coup`

An event that grants a coup outside the ordinary Operations path gets no target validation for
free: `can_coup_or_realign` (`src/ops.cpp`) holds the opponent-influence requirement, the DEFCON
regional restrictions, NATO and The Reformer. Ortega Elected in Nicaragua (#91) and Che (#107) once
built their own target lists and offered illegal coups (one took DEFCON to 1 and won); both now call
`Operations::can_coup` at the target mask and where the target is applied
(`tests/engine_logic/test_free_coup_target_legality.py`). Any new free-coup event must do the same.

Unrelated and correct: `resolve_defcon_one_loss` (`include/ts/defcon.hpp`) makes the **phasing**
player lose regardless of who drove DEFCON down. That is the rule.

---

## 9. Fixed: UN Intervention's companion mask was unfiltered (ENG-1)

The companion decision offers only opponent-associated, non-scoring cards (a `UN_INTERVENTION` case
in `get_event_action_mask`'s `SELECT_CARD` switch); since the audit (§12) UN Intervention is not
offered as an Event at all without such a card. History: ENG-1, in `BUGS.md`'s history.

## 9a. Not an issue: "UN Intervention for Ops dead-ends the game" (retracted 2026-09-23)

A P23 census reported an empty mask after `OPS_INFLUENCE` for UN Intervention (seed 231357, turn 4,
AR 5). **The game had ended**: a pending We Will Bury You pays the USSR 3 VP on any non-Event play,
taking -18 to -20, and the census had not checked `is_terminal`. The engine is correct. It did
expose a merged-view defect, fixed: "the commit ends the game" now offers `OPS_INFLUENCE` as the
bare commit (`tests/bindings/test_merged_influence.py`, `p23_wwby_pending_position.json`).

## 10. The era transitions do not reshuffle the discard (fixed in E6, 2026-09-28)

`StateMachine::finish_end_turn` adds the Mid War cards at turn 4 and the Late War cards at turn 8
with `add_era_cards_to_deck` and nothing else (rule 4.4: the discards stay in the discard pile); the
discard is reshuffled only when a draw runs the deck out (§13). Until E6 both transitions
also reshuffled the discard, so every pre-E6 checkpoint trained on that deck; the converter missed it
because it sets hands from the log. Held by `EraTransitionKeepsTheDiscard.*`
(`tests/test_bugs_regression.cpp`) and
`test_the_era_transitions_leave_the_discard_pile` (`tests/engine_logic/test_hand_knowledge_triggers.py`).

## 11. A card's Event handler does not place the card (fixed in E6, 2026-09-29)

Where a played card goes is decided in one place: every site that fires an Event asks
`CardHandlers::event_has_effect`, and `relocate_played_card` removes a starred card whose Event
happened and discards everything else. Prerequisites live in `can_trigger_event`, board conditions
(Kitchen Debates, Our Man in Tehran) in `event_has_effect`; a handler never writes its own
`card_locations` entry. (Kitchen Debates broke this until E6 and the E5 models learned to replay it;
`KitchenDebates_*`, `test_kitchen_debates_*`.) A special case in `relocate_played_card` belongs in
`event_has_effect` instead.

---

## 12. Rules audit, 2026-09-28

Every card handler, the Ops/coup/realignment/war code, scoring, the space race and the turn sequence
were read against `rules/cards.json`, `rules/rules.md`, the FAQ and struggler's rulings table; the
map and card table match struggler's field by field. Each change is pinned by
`tests/test_rules_audit.cpp` (or the test named) and shown to fail with its fix reverted:

| rule | before | source |
|:---|:---|:---|
| A war's defender's **superpower** counts as an adjacent controlled country (-1) | only the 84 countries were counted: a USSR Brush War on Mexico, a US one on Afghanistan, rolled a point too high | Rule 2.1.5 (superpower spaces count for "events and realignments"); FAQ under Brush War (a reversal of an earlier ruling) |
| The region-bonus ladder falls back to the event's **modified grant**, not the card's printed Ops | an Olympic boycott under Vietnam Revolts fell to the card's printed 2 on leaving Southeast Asia, losing two of the event's four Ops | `DecisionContext::ops_plain`, set by `Operations::grant_*_ops_to_ctx`. Rule 7.4.3 for "modified" |
| **UN Intervention** is an Event only with an opponent-associated card in hand | offered and fizzled, which cancelled a pending We Will Bury You for nothing | Card text; FAQ card #32 |
| **We Will Bury You** settles on a trapped US round | a Quagmire discard round skipped the check and the 3 VP waited for the next card play | Card text: UN Intervention was not played in that round |
| **NORAD** arms on DEFCON *moving* to 2 | Cuban Missile Crisis or How I Learned setting 2 at 2 armed it | Card text: "moved to 2" |
| **U-2 Incident** reaches UN Intervention played through Grain Sales | the route paid nothing | Card text |
| **Summit** counts regions without Shuttle Diplomacy or Formosan Resolution | both scoring-only effects moved Summit's die | FAQ card #73; card #35 "for scoring purposes only". `Scoring::dominates_or_controls` |

**Checked and left as it was: event-granted Ops take the Ops modifiers** (Containment, Brezhnev
Doctrine, Red Scare/Purge apply to Ops from CIA Created, Lone Gunman, ABM Treaty, the Olympic
boycott, Grain Sales, KAL-007, Glasnost, and to the coups of Junta, Che, Ortega and Tear Down This
Wall). Deluxe rule 7.4.3 and 7.4.2's Example 3 say so; the FAQ line "Ops from events are not
affected" is contradicted by the rulebook, which outranks it. The `*_OpsModifiers_Suite` tests in
`test_card_edge_cases.cpp` pin this reading.

These fixes define engine **E7** (`research/findings/engine/engine_revisions.md`); each is on a rare
situation. The human corpus (`tests/replayer`) passes with them.

---

## 13. A draw that runs the deck out reshuffles there, inside an Event too (ENG-4, 2026-10-19)

Three places draw from the deck: the deal (`StateMachine::deal_cards_to_hands`), Our Man in
Tehran's five (`trigger_our_man_in_tehran`, `src/events/mid_war.cpp`) and Ask Not's replacements
(its confirm in `src/card_dispatcher.cpp`). All three treat the deck running out the same way: a
draw that finds it empty, or takes its last card, reshuffles the discard pile into it **at that
moment** (`StateMachine::reshuffle_discard_into_draw`; rules.md's Reshuffle Invariant), and the
draws still owed come from the new deck. Until ENG-4 the two Events drew at most what the deck held
and left the reshuffle to the next deal: in the human corpus, at turn 7 AR2 of ts-replayer game 207
(Tehran, two cards in the deck), turn 7 AR6 of game 314 (Tehran, three), turn 6 AR4 of game 156
(Tehran, exactly five) and turn 7 AR7 of game 272 (Ask Not, two from two), the log prints
`*RESHUFFLE*` inside the Event.

* **Tehran's peek stays out of the new deck.** The cards already drawn are at `PEEKED_TEMP`, not
  in the pile; Tehran itself is still in hand or `HEADLINE_COMMITTED` (§11). **Ask Not's own
  discards go in**: they are in the pile when the deck runs out, and the Reshuffle Invariant makes
  no exception for cards discarded this turn.
* **The empty deck reveals what it reveals at the deal.** At the deal both hands become public:
  every card not in a hand is. Ask Not does the same (`reveal_both_hands`). Tehran reveals the USSR
  hand only: the US sees the peek and can name the USSR hand, but the USSR cannot tell the US hand
  from the cards being looked at. A card drawn after the reveal is hidden again, as at the deal.
* **The PRNG stream is untouched unless the deck runs out.** The pool is rebuilt only after a draw
  empties it, so a deck holding more than Tehran's five, or more than Ask Not's redraws, is drawn
  exactly as before. This still moves the engine's behaviour digest (`runner`, `rollout` and `mcts`
  sections; `tools/lib/engine_behaviour.py`), so it defines engine **E8**
  (`research/findings/engine/engine_revisions.md`).

Held by `Card108_OurManInTehran_ShortDeckReshufflesMidDraw`, `_DeckOfFiveReshufflesAfterTheDraw`,
`_LongDeckLeavesThePileAlone`, `Card77_AskNot_RedrawEmptiesTheDeck` and
`_DeckRunsOutBetweenRedraws` (`tests/test_cards_mid.cpp`).

## 14. The deal alternates, the USSR first (ENG-5, 2026-10-22)

`StateMachine::deal_cards_to_hands` deals one card at a time, USSR then US, and a hand already at
the target (8 on turns 1-3, 9 after) sits out. Until ENG-5 it filled the USSR hand completely and
then the US hand. The rules do not name an order ("deal" each player up to the hand size, §4.2 of
`rules/rules.md`), and the order is invisible unless the deck runs out partway through the deal --
which happens on **every turn 3**, when the last few Early War cards are dealt before the
reshuffle. Filling the USSR first gave the USSR nearly every card left in the old deck; the US drew
almost entirely from the reshuffled discard pile. That made a card the USSR missed on turn 2 near
certain to reach it on turn 3, so the bots valued the USSR's turn-2 draws at almost nothing
(an Op +0.05 pp, against about 1.3 pp on every other turn).

* **The human games alternate.** In the ts-replayer corpus all 274 games that reach turn 4
  reshuffle at turn 3. Of the 9.2 Early War cards on average not yet discarded at that deal (the
  hands' kept cards and the deck), the US's turn-3 hand holds 3.8 (at least 3 in 247 games) and the
  USSR's 4.3; a USSR-first fill leaves the US only the one or two cards it kept from turn 2. The USSR's half-card excess is what a USSR-first alternation gives an odd
  number of cards. The reference engine (`external/struggler`, `_deal_to_limit`) alternates USSR
  first too.
* **Every seed deals different hands.** The PRNG draws the same number of times, but the draws
  alternate between the hands, so from turn 1 a seed's hands are not E8's. The distribution of a
  deal is unchanged except when the deck runs out partway through it. The behaviour digest moves in
  every section (`runner`, `state`, `rollout`, `mcts`), so this needs a new engine epoch.
* **A forced deal is still per player** (`GameState::set_forced_deal`): each hand's queue is drawn
  in order whenever that hand's turn comes, so a recorded deal is replayed whatever the interleaving.

Held by `CardEdgeCasesTest.Deal_AlternatesSoAShortDeckIsShared`, `Deal_USSRDrawsFirst` and
`Deal_AFullHandSitsOut` (`tests/test_card_edge_cases.cpp`).

---

## 15. Review fixes (2026-10-28)

* **One VP award** (`award_vp`, `include/ts/victory_points.hpp`): clamps to [-20, 20], ends the
  game at either end, and leaves a finished game's score alone. Every event, scoring and space
  award goes through it -- about 40 sites used to clamp by hand, and some forgot the game end.
  Two rare bookkeeping changes follow: Flower Power's penalty on the war that just won the US the
  game no longer leaves 18 VP under GAME_OVER, and Special Relationship's winning 2 VP no longer
  opens a placement in a finished game.
* **We Will Bury You settles on the US's next action round, however it goes**
  (`settle_we_will_bury_you`, `src/state_machine.cpp`): a card played, a trapped discard, a pass
  holding only the China Card, or a round started with an empty hand (which used to be skipped,
  leaving the 3 VP for whenever the US next played a card). Only UN Intervention's Event answers
  it (owner, 2026-10-28). Its DEFCON-1 loss goes through `resolve_defcon_one_loss`, so a US forced
  into it off this card's Ops is recorded as provoked like every other forced DEFCON loss.
* **Out of range is refused, never a confirm.** `encode_micro_action_212` returns -1 for anything
  no slot encodes (an out-of-range id, `NONE`, the retired `CHOOSE_TIMING_BRANCH`), and
  `decode_flat_action_212` returns a `NONE` action for an index outside the space; both are refused
  by `StateMachine::step`, and `Engine::step_flat` checks the bound itself. A failed placement in
  Canada (id 0) is refused rather than read as "stop the Ops".
* **`RegionScoreSummary` reports what a scoring counted**: `ussr_*_scored` and
  `shuttle_diplomacy_applied` beside the board's counts, so the page shows Shuttle Diplomacy's
  adjusted counts from the engine instead of re-scoring the board to find them.
* **`anomalies_reported()`** (`include/ts/invariant.hpp`) counts this thread's `report_anomaly`
  calls, so a caller that must stop at one (the human-replay converter) can check it.
* Shared instead of copied: the Ops-budget ladder (`withdraw_conditional_ops_bonus`),
  `SpaceRace::advance_to`, Muslim Revolution's targets (`MUSLIM_REVOLUTION_TARGETS`) and the
  forced-removal auto-advance (`forced_event_target`). Dead: the `CHOOSE_TIMING_BRANCH` step handler
  (the mask cases stay, empty), `war_helpers::is_war_card`, `CardInfo::requires_context_push` /
  `switches_priority`, and the unused `forced_roll` of `CardHandlers::trigger_event`.
  `GameState::pop_context` on an empty stack is an invariant failure.

The behaviour digest is unchanged except for the We Will Bury You DEFCON-1 classification (its
`runner` section hashes each game's ending reason); no decision or result moved. Tests:
`tests/test_review_fixes.cpp`.

## 16. A card played on a pushed frame (ENG-6, E8 in place, 2026-11-02)

Events nest on `ctx_stack`, and Ops finishing on a pushed frame were taken to be Ops an Event granted.
Grain Sales fired through another card's Event breaks that: the card it draws is played on the frame
pushed for Grain Sales, a real play. The drawn card used for Ops stayed in the US hand, playable again;
an opponent's card's Event never fired, behind the depth guard; and a card played as its Event or for
Space went on to the next action round with the frame still pushed. In stats game b97d05b2, at turn 10's
headline, the US headlines Star Wars, takes Grain Sales, draws NORAD and places three Influence, and
NORAD was still in hand at the end of the game.

`advance_after_ops` now settles such a card first (`settle_card_played_on_pushed_frame`): still in a
hand is what separates it from granted Ops, whose card is placed as soon as its Event is triggered --
the same test the base frame uses. `finish_play_without_ops` unwinds the Event and Space endings. The
depth guards on the base frame's owed Event stay: a pushed frame starts zeroed, which reads as
OPS_FIRST. Tests: `PushedFrame.*` in `engine/tests/test_rules_audit.cpp`.

## 17. Eagle/Bear Has Landed may not discard a scoring card (ENG-7, E8 in place, 2026-11-04)

Space Race box 6 lets its holder discard a held card at the end of the turn, and the discard runs
before the held-card check (`end_turn`, then `finish_end_turn`). Its mask offered every card in the
hand, scoring cards included, so a player holding one threw it away instead of losing for holding it.
E8's top network had learned to: holding a scoring card at that decision, it discarded it 29 times of 30
(4,000 sampled self-play games, 0.73% of them; none after the fix). Scoring cards may never be
held (rules.md, the held-card invariant), so the mask now leaves them out; holding only a scoring card,
the pass is all there is, and the check then ends the game. The step refuses one too, since the mask
decides legality. Tests: `SpaceRace_Box6_EagleBearHasLanded_*` in `engine/tests/test_card_edge_cases.cpp`.

## 18. An event-first Event opens nothing to the Ops after it (ENG-8, E8 in place, 2026-11-06)

Influence placed by Operations goes in, or next to, a country holding friendly Influence at the start
of the action round (rules.md, Adjacency Requirement). `begin_op_mode` took that snapshot
(`DecisionContext::start_influence_nodes`) when the Ops began, which on an opponent's card played
event-first is after the Event, and `Operations::can_place_influence` tested the target country against
the current board. So the USSR playing The Iron Lady event-first, with nothing in or next to Argentina,
could stack all its Ops there and reach Chile, Uruguay and Paraguay off the Event's own point.

Now the event-first branch of `SELECT_PLAY_MODE` takes the snapshot on the Ops frame before it pushes
the Event, and `begin_op_mode` keeps it, dropping any country the Event emptied: the fix withholds what
the Event gave and gives back nothing it took, so removal Events play as before. The target country,
like a neighbour, must hold friendly Influence now *and* in the snapshot. Bits above country 83 mark
the snapshot's owner (an unmarked one, from an older save, gates neighbours only) and a *pending* one,
taken before the Event: it answers its owner at once but gates the other side's question -- the
observation asks both -- only once the Ops begin, when a snapshot always has. The observation therefore
moves only where an Event gave the phasing player Influence; `test_observation_golden.py` is unchanged.
1 of 2,000 seeded random games plays differently. Tests: `Card83_TheIronLady_*` in
`engine/tests/test_cards_late.cpp`.

## 19. An opponent-held card's Event is offered only where it can fire -- a rule bit (ENG-9, 2026-11-06)

Before ENG-9 the play-mode mask offered EVENT on an opponent's card unconditionally, so one whose
prerequisite is unmet (NATO before Warsaw Pact Formed or the Marshall Plan, Solidarity, the flag-gated
Events, The Cambridge Five in the Late War, Star Wars with the US not ahead) fizzled into an
event-first Ops play. The fix asks `can_trigger_event` for the player the Event fires for.

It is a **rule bit**, `obs_features::OPP_EVENT_GATED` (bit 3, no width, in `obs_features::RULES`):
every earlier network plays the fizzle (about once a game) and chooses its Ops differently without it,
so a network plays the rule it was trained with. Each seat's rule bits live in
`GameState::seat_rules`, which every mask, step validation and auto-advance reads
(`seat_obs_features`, `seat_both`, `seat_plays_rule`). **Zero is the rule from before every rule
bit**: a bare state and every pre-bit network play it, and the unseated engine's digest is E8's.

Who sets the seats: a harness seats each model from its feature set (`bindings/AGENTS.md`); a
**search plays both sides of its tree under its own network's rules** -- `BatchedSearch::reset` and
`RolloutBatch::add` `seat_both` the roots they copy, the Python trees `ts_env.seat_tree` -- since the
network models both players, so an old and a new model can search against each other; the web page
seats a person's side with every rule bit (the current rules) and only the auto-play side with its
model's (`web/ui/AGENTS.md`).

Tests: `*OpponentEvent*`, `*NATOEvent*`, `*Solidarity*`, `UnseatedStatePlaysThePreFixRule` in
`engine/tests/test_bugs_regression.cpp`; `tests/bindings/test_obs_rule_bits.py`,
`tests/training/test_search_mixed_rules.py`.

## 20. A region bonus pays only for Ops spent in its region (ENG-10, E8 in place, 2026-11-07)

The China Card's Asia bonus and Vietnam Revolts' Southeast Asia bonus are granted up front and
withdrawn on the first Op spent outside their region -- the ladder: all in Southeast Asia keeps both,
elsewhere in Asia forfeits Vietnam Revolts, outside Asia forfeits the China Card's too. The withdrawal
ran only *after* the Op was spent, and the masks checked the ladder only in part: the influence mask
asked the China Card's rule *or* Vietnam Revolts', never both, and the realignment mask asked neither.
So the last bonus Op could go anywhere -- with the China Card under Vietnam Revolts, five Ops in
Southeast Asia and the sixth in India; a realignment's final bonus roll in Iran.

The ladder is now one function, `Operations::budget_if_spent_in(state, player, country)`, which the
withdrawal (`withdraw_conditional_ops_bonus`) applies and `Operations::mask_region_budget` asks for
every influence and realignment target: a target is offered only if the Ops already spent plus its
cost (its Influence cost, or 1 for a roll) fit the budget spending there keeps. 19 of 2,000 seeded
random games play differently. Tests: `RegionBonusBudget.*` in `engine/tests/test_bugs_regression.cpp`.

## 21. Canonical placement order -- two rule bits (P43, owner 2026-11-07)

A run of one side's placements reaches the same board in many orders, so a per-step policy has no
pressure on order (`research/plans/P43_canonical_placement_order.md`). `obs_features::CANONICAL_FIRST`
and `CANONICAL_LAST` (bits 4 and 5, no width, in `obs_features::RULES`, at most one per seat) make a
seat place every **placement run** in non-decreasing rank of its order, so every board a run can reach
has one action sequence. The helpers are `ts/canonical.hpp`.

* **The runs** (`canonical::run_order`): the setup's three (USSR 6; US 7; US 2), every Ops influence
  play (`DecisionContext::is_ops_influence_play`), and the event runs of `canonical::EVENT_CARDS` --
  placement and removal runs each verified to reach exactly free order's boards. Coups, realignments
  and every other event stay in free order.
* **The order** (`MapData::placement_rank`, the important-last rank, a permutation of 0..83): regions
  from least to most scoring weight (Africa, South America, Central America, Asia, the Middle East,
  Europe), non-battlegrounds before battlegrounds within each, Europe's battlegrounds last as Italy,
  France, West Germany, East Germany, Poland; ties by id. CANONICAL_FIRST is its reverse (83 - rank).
* **The floor** (`DecisionContext::canon_floor`, `canon_tag`, `canon_left`): 1 + the rank of the last
  country placed, the run it belongs to (`canonical::run_tag`, constant through a run) and the points it
  left. `StateMachine::step` (around `step_impl`) writes them after a placement in a covered run and
  clears them after anything else in the frame; only for a seat playing a canonical bit, so every game
  without one is unchanged (the behaviour digest did not move). A save writes them only when set.
* **The dead-end release** (owner): where a run has points left and free order has a target but none is
  at or above the floor -- an Ops point that would cost 2, a capped event run that jumped too high --
  the mask offers free order's targets for that choice, and the run goes on canonically from the
  country chosen. So a run reaches exactly free order's boards, never more and never fewer; a board
  reached through a release may also have a canonical sequence (uniqueness is lost only there).
  CONFIRM_DONE still follows free order.
* **V3, a learned order** (`obs_features::CANONICAL_LEARNED`, bit 6, owner 2026-11-07): the run's order is
  the deciding network's own priority, its logits over the 84 countries at the run's first decision,
  highest first. The engine cannot compute it, so the harness gives it: `canonical::priority_wanted` says a
  LEARNED run is opening without its order, `canonical::set_run_priority` stores the ranks in
  `GameState::canon_prio` (tagged with the run, cleared when it ends; saved only when set). A placement in a
  LEARNED run without its order is **refused** unless it is forced (one legal target), so a harness that forgets
  fails loudly instead of playing some order nobody gave.
* **Seeing the run's order** (owner): `obs_features::PLACEMENT_LEGAL` (bit 7) puts the decider's legal mask in
  slot 19 during its placement decisions -- correct, but it changes the slot at 71% of them and made V1 worse at
  400M. `PLACEMENT_ORDER` (bit 8, 2026-11-08) keeps slot 19 as access and marks only the countries the run's order
  blocks now as 0.5 (0 no access, 1 access), while a floor is in force and not released. Both edit a base slot at no
  width, so only a model trained with the bit reads them so.
* **A person plays none of them** (`obs_features::PERSON_RULES` for the page's current rules: ENG-9's bit and §22's).

Tests: `CanonicalPlacement.*` in `engine/tests/test_canonical_placement.cpp` -- every setup board reached
by exactly one sequence, exhaustively, for both orders; and on Ops plays and covered event runs from 120
random games, the boards a canonical seat reaches equal free order's, exhaustively per run, for both
orders. `tests/bindings/test_canonical_placement.py`.

## 22. A person's rules: events that would do nothing, and discards that tell (owner 2026-11-11)

Two **rule bits** of no width, in `obs_features::RULES` and `PERSON_RULES` (a person plays them; no
training name, so no network does). Unseated, both are the rule from before them, and the behaviour
digest is unchanged (E8's, as on `main`).

* **`OWN_EVENT_GATED`** (bit 9): a seat's own or neutral card offers its Event only where
  `CardHandlers::event_has_effect` says something would happen. That already covered Our Man in Tehran and
  Kitchen Debates; for a seat playing the bit it also covers **One Small Step** (only behind in space) and
  **Wargames** (only at DEFCON 2), whose Events are nothing but their "If ...". Played anyway (a headline),
  such a card did not occur, so a starred one is discarded rather than removed (`relocate_played_card`).
  Without the bit both cases return true, as before: offered, nothing happens, removed.
* **`ALWAYS_ASK_DISCARD`** (bit 10): Blockade and Latin American Debt Crisis ask the US even when it holds no
  card of 3+ effective Ops; its one legal answer is then CONFIRM_DONE. Resolving at once told the USSR the
  US hand held no such card, which at a table nobody learns. Read on **either** seat
  (`either_seat_plays_rule`): it guards the USSR's ignorance as much as it asks the US, so a person on either
  side of a game against a network gets it. With auto-advance the game is the same as without the bit (the
  forced "done" is taken), one micro-step longer.

The page also has a display-only hook here: `StateMachine::pause_between_headlines` (thread-local, off
except inside the wasm `ts_first_headline_json`) stops a copy of the game between the two headlines, so the
first headline's VP can be shown alone.

Tests: `OwnEventGated*`, `AlwaysAskDiscard*` in `engine/tests/test_bugs_regression.cpp`.
