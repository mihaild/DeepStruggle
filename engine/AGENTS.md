# Twilight Struggle AI: C++ Simulation Core Guide

This directory contains the zero-allocation, high-throughput simulation engine for the Deluxe Edition of **Twilight Struggle** (110 Cards).

---

## 1. Core Architectural Pillars

1. **Zero Heap Allocation in Simulation Core**: The core data structures (`GameState`, `DecisionContext`, `MicroAction`) use fixed-size memory layouts. `GameState` is strictly trivially copyable and under 4 KB (`sizeof(GameState) <= 4096`).
2. **Micro-Decision State Machine**: All multi-step actions (Ops, events with multiple target choices, headline selection, space races) are decomposed into fine-grained atomic steps (`DecisionType`).
3. **High Simulation Throughput**: Over 2,000,000 steps/second on one core of a modern desktop CPU (`ts_benchmark` reports the figure for the machine at hand), which is what makes reinforcement-learning-scale self-play affordable.
4. **Deterministic Bit-for-Bit State**: Built-in 64-bit SplitMix64 PRNG (`Prng`) ensures bit-for-bit replayability from integer seeds.
5. **Unified Sub-Decision Processing**: State machine handles sub-decisions (e.g. `SELECT_OP_MODE`, `POINT_NODE`, `CHOOSE_BRANCH`) uniformly across both `Phase::HEADLINE` and `Phase::ACTION_ROUND`.

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
├── CMakeLists.txt              // Builds ts_engine_core plus ts_tests, ts_fuzz, ts_fuzz_events, ts_benchmark
├── AGENTS.md                   // Developer and agent documentation (this file)
├── include/ts/                 // Public and internal engine headers
│   ├── types.hpp               // Enums: Player, Phase, WarEra, CardLocation, DecisionType, ActionType, Region, SubRegion, Resolution, TimingBranch, OpMode
│   ├── constants.hpp           // Country IDs, Card IDs (1..110), 64-bit Effect Flags, Bit Masks
│   ├── micro_action.hpp        // 4-byte packed MicroAction struct (alignas(4))
│   ├── game_state.hpp          // Contiguous trivially copyable GameState struct
│   ├── prng.hpp                // Deterministic SplitMix64 PRNG
│   ├── map_data.hpp            // 84 countries static metadata, graph adjacency 128-bit bitmasks
│   ├── card_data.hpp           // 110 cards static metadata (Ops, side, era, asterisk, scoring)
│   ├── scoring.hpp             // Region scoring formulas (Presence/Domination/Control), victory evaluation
│   ├── ops.hpp                 // Influence placement, Coup mechanics, Realignment rolls, dynamic costs
│   ├── defcon.hpp              // Shared DEFCON-1 game-end resolution and provoked/unprovoked classification
│   ├── space_race.hpp          // Space race tracks, milestone rewards, and special abilities
│   ├── card_handlers.hpp       // Card event handlers and sub-decision dispatch declarations
│   ├── war_events.hpp          // Shared helpers for the five war cards (Korean, Arab-Israeli, Indo-Pakistani, Brush, Iran-Iraq)
│   ├── invariant.hpp           // invariant_failed(): aborts naming what broke, rather than carrying on
│   ├── action_mask.hpp         // Legal action mask generator (per DecisionType and unified 212-dim flat mask)
│   ├── state_machine.hpp       // Turn and Action Round lifecycle, Headline resolution
│   ├── observation.hpp         // Neural observation extractor (ObservationBufferV23, the one layout)
│   ├── serialization.hpp       // Binary snapshot and JSON serialization
│   └── engine.hpp              // Top-level ts::Engine public interface
├── src/                        // Engine implementations
│   ├── map_data.cpp            // Static country array and adjacency lookup tables
│   ├── card_data.cpp           // Static card metadata array
│   ├── scoring.cpp             // Regional scoring and final scoring math
│   ├── ops.cpp                 // Ops execution and target validation
│   ├── space_race.cpp          // Space race advancement
│   ├── events/
│   │   ├── early_war.cpp       // Cards 1-35, 103-106 event handlers
│   │   ├── mid_war.cpp         // Cards 36-81, 107-108 event handlers
│   │   └── late_war.cpp        // Cards 82-102, 109-110 event handlers
│   ├── card_dispatcher.cpp     // Event trigger dispatch, sub-decision step router, action mask filters
│   ├── action_mask.cpp         // ActionMask::generate_mask, generate_flat_mask_212, decode_flat_action_212, encode_micro_action_212
│   ├── state_machine.cpp       // State machine turn loop, setup, headline, and AR transitions
│   ├── observation.cpp         // Feature extractor implementation
│   ├── serialization.cpp       // Serializer binary & JSON implementations
│   ├── invariant.cpp           // Out-of-line invariant reporting
│   └── engine.cpp              // ts::Engine API implementation
└── tests/                      // Engine test suites
    ├── test_framework.hpp      // Lightweight assertion & test registry framework
    ├── game_test_wrapper.hpp   // High-level full game execution wrapper & policy harness
    ├── test_main.cpp           // Test runner
    ├── test_map.cpp            // Topology, battlegrounds count, adjacency symmetry tests
    ├── test_scoring.cpp        // Regional formulas, Europe control instant win tests
    ├── test_bugs_regression.cpp// Specific regression tests for card bugs
    ├── test_ops.cpp            // Dynamic cost drop, Coup DEFCON degradation, NATO tests
    ├── test_cards_early.cpp    // Early war card unit tests
    ├── test_cards_mid.cpp      // Mid war card unit tests
    ├── test_cards_late.cpp     // Late war card unit tests
    ├── test_defcon_suicide.cpp // DEFCON suicide priority in OPS_FIRST vs EVENT_FIRST
    ├── test_reentrancy.cpp     // Re-entrant DecisionContext stack tests
    ├── test_card_interactions.cpp // Multi-card interaction test suite
    ├── test_state_lifecycle.cpp// Turn, headline, and phase lifecycle tests
    ├── test_states.cpp         // Continuous state effects and modifier tests
    ├── test_card_edge_cases.cpp// Complete edge-case tests across all cards
    ├── test_full_game.cpp      // 10-turn full game integration tests ending in final scoring
    ├── test_auto_advance.cpp   // Engine::auto_advance_step: forced/degenerate decisions taken without asking
    ├── test_fuzz_influence_placement.cpp // Randomized influence placement against the mask
    ├── test_fuzz.cpp           // ts_fuzz: invariant fuzzer (--games <N>, --steps <N>, --seed <S>)
    ├── test_fuzz_events.cpp    // ts_fuzz_events: same, biased toward firing events (--event-bias)
    └── test_benchmark.cpp      // ts_benchmark: throughput benchmark
```

---

## 4. Micro-Decision Pipeline & Action Representation

**P23 / E4.1: the merged-influence view.** `ActionMask::generate_flat_mask_merged` and the
`merged_influence` overloads of `Engine::get_flat_action_mask` / `Engine::step_flat` offer, at the two
op-choice nodes (`SELECT_PLAY_MODE`, `SELECT_OP_MODE`), "ops for influence, first point in X" as the
NODE slot X, and "ops for influence, place nothing" as `OPS_INFLUENCE`. Both are **defined** as the
two E4 steps they name (`OPS_INFLUENCE` then X, or then `CONFIRM_DONE`), the mask is read off a copy
of the state after `OPS_INFLUENCE`, and a composed step is applied atomically. So no rule, no
`GameState` field and no observation slot changes; the view is the deciding agent's, not the game's,
and with it off everything is bit-identical to E4 (checked against a frozen decision stream).
`tests/bindings/test_merged_influence.py` pins the definition: mask equality and byte-identical states.
Where the commit itself ends the game (a We Will Bury You penalty falling due, §9a) the merged view
offers `OPS_INFLUENCE` as that bare commit -- there is nothing to compose. Anywhere else the commit
does not reach the mover's placement, it offers no influence and calls `report_anomaly`; it
composes, it never substitutes. Over 96,010 influence-legal op-choice nodes that has not happened.

The engine splits complex turns into a sequential stream of atomic 4-byte `MicroAction` structures:

```cpp
struct alignas(4) MicroAction {
    DecisionType decision_type; // 1 byte: SELECT_CARD, SELECT_PLAY_MODE, SELECT_OP_MODE, POINT_NODE, CHOOSE_BRANCH, ROLL_DIE
    uint8_t      primary_id;    // 1 byte: Card ID (1..110), Country ID (0..83), Branch ID (0..7), Resolution, OpMode, TimingBranch, or CONFIRM_DONE (0x80)
    uint8_t      secondary_id;  // 1 byte: Sub-choice / quantity / manual die roll
    uint8_t      flags;         // 1 byte: Additional modifiers / opponent manual die roll
};
```

**`ROLL_DIE` is the one decision whose `primary_id` is data, not an index.** It carries the forced
die for the acting player, and `secondary_id` the opponent's, with **0 meaning "roll normally"**
(`Operations` reads `forced_roll > 0` as "a die was forced"). Three consequences, each of which has
already caused a bug:

* **The 212-wide mask cannot constrain it.** Every other decision type validates by the action
  being a legal index; a die value has no index to check. The range is therefore checked in
  `StateMachine::step`, beside the `decision_type` guard, or nowhere.
* **The 255 "no selection" sentinel collides with the value space.** `decode_flat_action_212(211)`
  returns `primary_id = 255` for every other decision type, which is harmless there and was read as
  a forced die of 255 here. `ROLL_DIE` is now decoded first and yields 0.
* **A missing mask case is not a missing action.** `generate_flat_mask_212`'s switch had no
  `ROLL_DIE` case, so the "ensure at least one action is legal" fallback at the bottom supplied 211
  as a generic confirm/done — a wrong-but-accepted action rather than a loud failure. The case is
  now explicit. Prefer a real case to relying on that fallback.

Nothing measured was invalidated: `Engine::auto_advance_step` (`engine/src/engine.cpp:46`) and
`VectorizedBatchRunner` (`bindings/ts_bindings.cpp:1087,1128`) resolve chance nodes with an explicit
`{ROLL_DIE, 0, 0, 0}` and never decode, and no `ROLL_DIE` node was ever handed to an agent —
verified over 879 single-env decisions and 12,800 vectorized env-steps.

**`step` returns `[[nodiscard]] bool`, and the build treats ignoring it as an error.**
`StateMachine::step`, `Engine::step` and `Engine::step_flat` all carry the attribute, and both
`CMakeLists.txt` files pass `-Werror=unused-result` (MSVC `/we4834`), so a discarded verdict does
not compile. This is deliberately stricter than a runtime check: every bad call site fails at build
time rather than only the ones a given test run happens to reach.

Throwing instead was considered and rejected. `step` is `noexcept` and is called from inside
`#pragma omp parallel for` in `VectorizedBatchRunner::step_flat_all`, where an escaping exception is
undefined behaviour; and the batch path needs a *per-game* verdict, which a scalar throw cannot
express. Compile-time enforcement gives more safety at no runtime cost and keeps both properties.

Adding the attribute found 152 discarding call sites in `engine/tests` and one in production:
`VectorizedBatchRunner::refresh_single` drained chance nodes with an unchecked `step`, so a refused
`ROLL_DIE` would leave the loop condition unchanged and spin forever. Its sibling loop in
`step_flat_all` already had the `if (!step(...)) break;`. All 372 C++ tests passed once the sites
asserted, so none of them was driving an illegal action; the Python side was not so clean.

---

## 5. How to Build, Test, and Benchmark

### Standard Build:
The root `CMakeLists.txt` orchestrates `engine/` and `bindings/` together, and the rest of the
repository expects the result in `build/release`.
```bash
cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

### Build with Sanitizers (AddressSanitizer + UndefinedBehaviorSanitizer):
```bash
cmake -B build_san -S . \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build_san -j
```

### Run Unit Tests:
```bash
./build/release/engine/ts_tests
# Or with sanitizers:
./build_san/engine/ts_tests
```

### Run the Fuzzers:
```bash
# Invariant fuzzer: 10,000 full games, or a step budget from a fixed seed
./build/release/engine/ts_fuzz --games 10000
./build/release/engine/ts_fuzz --steps 5000000 --seed 42

# Event-biased fuzzer: same invariants, steering play toward card events
./build/release/engine/ts_fuzz_events --steps 1000000 --event-bias 95

# Under ASan + UBSan:
./build_san/engine/ts_fuzz --games 10000
```

### Run Performance Benchmark:
```bash
./build/release/engine/ts_benchmark
```

---

## 6. Guidelines for Extending the Engine

1. **Never Allocate Heap Memory in State Types**: Always ensure `static_assert(std::is_trivially_copyable_v<GameState>);` passes.
2. **Deterministic PRNG**: Use `Prng::roll_d6(state.rng_state)` or `Prng::random_index(state.rng_state, n)` whenever game state dice or shuffles are executed.
3. **Pass / Confirm Handling**: Multi-step cards (e.g. *Suez Crisis*, *Muslim Revolution*, *Independent Reds*, *Special Relationship*) must support early pass (`action.primary_id == 0` or `CONFIRM_DONE`) when no eligible targets remain.
4. **Passing an Action Round**: a player holding cards must play one -- `SELECT_CARD` offers the
   pass (mask index 0, flat 211) only to a player with none. Two exceptions: the China Card is not
   part of the hand a player is required to spend, so a holder whose only playable card is the
   China Card is offered both it and the pass; and the eighth action round is granted by North Sea
   Oil or a Space Station to the one player who earned it and is theirs to decline, so the pass is
   always on offer there. A player with an empty hand never reaches the mask at all --
   `advance_after_action_round` passes them itself.
5. **The China Card Can Be Raced**: it carries no Event of its own, so `SELECT_PLAY_MODE` offers
   Operations and, where the next box's Ops requirement is met, the Space Race. Racing with the
   best Ops card in the game is a poor play and not an illegal one -- at turn 10 AR4 of
   ts-replayer game 247 the US races to box 5 with it. Whatever it is played for it passes to the
   opponent face down and is never discarded, so `SpaceRace::attempt_space` sets
   `china_card_holder` and `china_card_playable` rather than touching `card_locations`. Its Asia
   bonus pays nothing here: the space track is not in a region.
6. **Headline Cards Leave Hand On Commitment**: both headlines are played at once, face down, and
   only then resolved in Ops order, so as soon as both are selected each card's location becomes
   `CardLocation::HEADLINE_COMMITTED` -- neither is in a hand while the other resolves, and a card that
   reads a hand (The Cambridge Five, Missile Envy, Grain Sales To Soviets, "Lone Gunman", CIA
   Created, Aldrich Ames Remix, Five Year Plan, Blockade, Latin American Debt Crisis) must not
   find it there. `HEADLINE_COMMITTED` is a waypoint: the ordinary post-resolution cleanup overwrites it
   with the card's real destination. It is deliberately not the discard pile, which Star Wars and
   SALT Negotiations read.
7. **Ops Spent In A Headline Discard Their Own Card**: `advance_after_ops` relocates
   `pending_op_card` in the HEADLINE phase, because the headline machinery only clears the two
   headline cards themselves. A card played *through* one of them -- Grain Sales To Soviets draws
   from the opponent's hand and has its player play what it draws -- otherwise stays in the hand
   it was played from, in the running for Missile Envy and playable a second time.
8. **Traps Judge A Card On Its Effective Ops**: Quagmire and Bear Trap take a discard of 2 Ops
   or more, and Containment, Brezhnev Doctrine and Red Scare/Purge all move that value, so
   eligibility is `Operations::get_effective_ops(state, card, p)` and not the printed value --
   with no target region, since a discard has none. The mask and the SELECT_CARD handler must
   agree: a card only one of them accepts either cannot be discarded or falls through to an
   ordinary play.
9. **A Card In Play Is Not In Hand**: the engine leaves an Ops card in its owner's hand until
   the play finishes, and an opponent's card played for Operations fires its own event -- so an
   event that reads that hand can find the very card in front of it. Grain Sales To Soviets,
   Five Year Plan, Terrorism and Missile Envy all skip whatever `resolving_card` names, which is
   that card. Only the first two are reachable this way (the other two are neutral, and a
   neutral card played for Operations fires no event).
10. **Cancelling Cuban Missile Crisis Is A Choice, At Two Moments**: the crisis can be paid off at
   any time; the engine offers it at the head of the payer's own action round, where declining is
   allowed and usual, and inside a coup they make, where it is not -- couping without paying
   loses the game. Both are a `POINT_NODE` with `resolving_card == CUBAN_MISSILE_CRISIS`: the US
   pays from West Germany or Turkey, the USSR from Cuba. The coup's is asked before its die, so
   the coup is already staged and the chance node opens on the far side of the answer;
   `ctx().pending_roll == RollType::COUP` is what tells the two apart. With one payer, or
   none, `execute_coup` settles it inline as before.
11. **An Opponent's Card Owes Its Event In A Headline Too**: playing an opponent's card
   carries the order in the resolution itself -- P17 retired `CHOOSE_TIMING_BRANCH`, so
   `Resolution::EVENT` on an opponent card means event-first and any `OPS_*` means
   ops-first. The `timing_branch` field survives as internal state, set from whichever was
   chosen, and `OPS_FIRST` still leaves the Event owed until the Ops are spent. `advance_after_ops` fires it on both paths -- the headline's as well as the
   action round's, which it used to unwind past. At turn 4's headline of ts-replayer game 137
   the US headlines Grain Sales To Soviets, takes Willy Brandt and realigns Cuba twice with it,
   and Willy Brandt's Event must still follow. A headlined card is excluded: it is played as
   its Event, so Ops belonging to one are Ops its Event gave away and it has already fired.
12. **Shuttle Diplomacy Removes A Battleground, And A Country With It**: in Asia or Middle East
   scoring it takes one USSR-controlled *battleground* off their totals, and the country count
   goes with it because that battleground is a country -- both matter, since Domination and
   Control are decided by who holds more countries. All of it is conditional on there being a
   battleground to take: the USSR can be put out of Presence by losing their one battleground,
   never by losing their one non-battleground country. At turn 10 AR1 of ts-replayer game 323
   they hold Lebanon and no battleground at all, and the region is worth 5 to the US, not 8.
   Final scoring is exempt.
13. **Defectors Cancels The USSR Headline However It Reaches The Table**: the headlined case is
   settled before either card resolves, by the check on `headline_us_card` in `step`. Any other
   route -- Five Year Plan discarding it out of the USSR hand, Grain Sales To Soviets handing
   it to the US, Star Wars taking it out of the discard pile -- fires it once the pair is
   committed, so `trigger_defectors` cancels the *second* headline card where the USSR owns it
   and `headline_stage < 2`. A USSR card that has already resolved is untouched: Defectors
   after it is too late. At turn 2's headline of ts-replayer game 313 the USSR headlines
   Vietnam Revolts against Five Year Plan, the higher Ops, and the Defectors it discards leaves
   Vietnam at [0][0].
14. **Each Headline Card Resolves In Its Own Frame**: `advance_headline_step` gives the second
   card a fresh `DecisionContext` rather than writing over the first's. What lasts belongs to
   the state -- the effect bits, the card Missile Envy forced on its recipient -- and what does
   not includes the visited bitmap that enforces "no more than one per country". At turn 7's
   headline of ts-replayer game 92 the US's Colonial Rear Guards places in Zaire, Angola,
   Zimbabwe and Nigeria, and the USSR's Decolonization was then offered none of them: three of
   its four Influence had nowhere to go.
15. **A Trap Never Holds A Scoring Card Past The Turn**: Quagmire and Bear Trap take a card of
   2 effective Ops or more each action round, and a scoring card is not one -- but it is
   playable out of a trap on either of two counts: nothing in hand is eligible, or the player
   holds as many scoring cards as they have action rounds left to play them in. The second is
   what stops a trap costing a player the game, since a scoring card held at a turn's end is a
   loss outright. At turn 4 AR7 of ts-replayer game 63 the USSR has spent two rounds discarding
   to Bear Trap and plays Central America Scoring on the last one.
16. **A Headline Ends With The Stack Empty**: an event that grants Ops does not finish when it
   is triggered, so the frame it was fired in stays open until those Ops are spent. Missile Envy
   fires the card it takes inside a pushed frame; a card like ABM Treaty leaves it behind.
   `advance_after_ops` unwinds the stack on the HEADLINE path before advancing the headline,
   stopping at a frame that holds an unanswered `SELECT_OP_MODE` -- those are Ops still owed
   inside the headline, which is what the stack is for.
17. **Always Update Tests When Changing Card Logic**: Add unit test cases in `engine/tests/` for any new card behaviors, interactions, or edge cases.

---

## 6a. The observation is not yours to change

Adding a feature, removing one, changing what a slot means or changing the width are all
representation decisions and all belong to the project owner. Ask first.

A network reads fixed slices, so a changed observation never raises — the checkpoint loads and
misreads, and a content change at unchanged width slips past the width assertions too. Every
checkpoint and every `(seed, actions)` dataset is invalidated by a width change (root `AGENTS.md`
invariant 10), which means the cost lands on every number measured before it.

Two standing preferences: a large vector for a rare mechanism is not worth it (a per-country or
per-card bit serving one card costs 84 or 110 floats), and a partial feature is worse than none.

---

## 7. Why hand knowledge lives in `CardLocation`

`card_locations` distinguishes `HAND_US_KNOWN` from `HAND_US_UNKNOWN` rather than carrying a
parallel "known" bitset, and the bare `HAND_US` / `HAND_USSR` names deliberately do not exist.
Eleven values (`include/ts/types.hpp`), four of them hand variants:

```
UNAVAILABLE=0  DRAW_DECK=1  HAND_US_UNKNOWN=2  HAND_US_KNOWN=3  HAND_USSR_UNKNOWN=4
HAND_USSR_KNOWN=5  DISCARD_PILE=6  REMOVED_FROM_GAME=7  ONGOING_EVENT=8  PEEKED_TEMP=9
HEADLINE_COMMITTED=10
```

One field is enough because a card's holder always knows their own hand, so the only fact that
varies is whether the *other* player knows: the holder is in the value, and "known" can only mean
known to the non-holder. `card_locations` is a `uint8_t[111]` with values to spare, so the two
extra variants cost no bytes inside a `GameState` capped at 4 KB. They also make knowledge
monotone for free -- every write that moves a card out of a hand overwrites its knownness, where a
parallel bitset would have to be cleared by hand at each of the hundred-odd write sites, and a
missed one would report the opponent holding a card visibly in the discard.

Access goes through `is_in_any_hand`, `in_hand_of`, `known_to_opponent`, `hand_of`, `hand_holder`
and `revealed`, never a bare comparison -- and that is the load-bearing part. The ~50 sites that
once read `card_locations[c] == HAND_US` would all still compile if the bare names existed, and
would silently treat a known card as not in hand: its holder could not play it, and it would
vanish from hand counts and from discard selection. Removing the names turned every one of them
into a compile error. **Do not reintroduce them**, and add new reads through the helpers.

A known hand card must behave identically to an unknown one in **every rule** and differ **only in
the observation**: the card block (`src/observation.cpp`) gives an opponent-held card the
`KNOWN_OPPONENT_HAND` slot when it is known and folds it into `DECK_OR_HIDDEN` when it is not,
since from the observer's side those two are the same thing and anything else would leak the hand.
From the holder's own perspective both variants are `MY_HAND`.

---

## 8. Free-coup events must go through `Operations::can_coup`

An event that grants a coup outside the ordinary Operations path does not get target validation
for free. `can_coup_or_realign` (`src/ops.cpp`) is what refuses a country the opponent has no
influence in, and it is also where the DEFCON regional restrictions, NATO and The Reformer live;
`get_coup_target_mask` is built on it, so an ordinary Ops coup is filtered correctly.

Ortega Elected in Nicaragua (#91) and Che (#107) each once built their own target list -- adjacency
to Nicaragua for Ortega, region and battleground status for Che -- and never consulted `can_coup`.
Both therefore offered coups the rules forbid, including a coup on a battleground the opponent held
no influence in, which took DEFCON from 2 to 1 and ended the game against the phasing player: an
illegal move that won. Both handlers now call `Operations::can_coup`, at the target mask in
`get_event_action_mask` and again where the chosen target is applied, and
`tests/engine_logic/test_free_coup_target_legality.py` covers them. Any new free-coup event must
do the same.

Unrelated and correct, since it is easy to misread as part of the same bug:
`resolve_defcon_one_loss` (`include/ts/defcon.hpp`) makes the **phasing** player lose regardless of
who drove DEFCON down. That is the rule.

---

## 9. Known issue: UN Intervention's companion mask is unfiltered

UN Intervention (#32) is played simultaneously with a card carrying the opponent's associated
Event, so its companion must be an opponent-associated, non-scoring card. **The action mask offers
every card in hand**, scoring cards and the player's own included.

`trigger_un_intervention` (`src/events/early_war.cpp`) sets `ctx().resolving_card`, so
`action_mask.cpp`'s `if (ctx.resolving_card != 0)` branch fires first and delegates to
`get_event_action_mask`, whose `SELECT_CARD` switch has no `UN_INTERVENTION` case and falls
through to `default:` -- every card in hand. The correct filter lives just below that branch
(`pending_op_card == UN_INTERVENTION && is_opponent_card(i, p)`) and is shadowed, so it never runs.

The illegal choice is then absorbed silently: the resolver (`src/card_dispatcher.cpp`) re-checks
the rule and on failure clears `resolving_card` and returns. Nothing is corrupted -- the companion
stays in hand -- but UN Intervention goes to the discard pile with no event fired and no Ops
granted, so the player loses a whole action round with no error reported.

**Unfixed.** The fix is a `case card_ids::UN_INTERVENTION:` in `get_event_action_mask`'s
`SELECT_CARD` switch mirroring the resolver's own test, deletion of the shadowed branch rather
than a second copy of the rule, and a regression test pinning the companion mask to opponent
non-scoring cards. It changes the decision stream, so it is an engine change under §2.

## 9a. Not an issue: "UN Intervention for Ops dead-ends the game" (retracted 2026-09-23)

The P23 census reported, and an earlier version of this section recorded, that choosing
`OPS_INFLUENCE` for UN Intervention (seed 231357, turn 4, AR 5, US at -18 VP) left the game on a
`SELECT_PLAY_MODE` node with an empty mask. **The game had ended.** The USSR had played We Will Bury
You (`WE_WILL_BURY_YOU_PENDING`), whose text is "unless #32 UN Intervention is played **as an
Event** on the US's next action round, the USSR receives 3 VP". Any Ops play -- influence, coup or
realignment -- pays those 3 VP, which takes -18 to -20, and the USSR wins; EVENT cancels the
penalty. The engine is correct. The census had checked the next decision's type and mask but not
`is_terminal`, and a finished game offers nothing. Nothing about UN Intervention or 1-Op cards is
special here: any card but UN Intervention-as-Event would do the same.

What it did expose was a P23 defect, fixed in the same change: the merged view treated "the commit
ends the game" as a failed composition and dropped influence, removing a legal E4 option. It now
offers `OPS_INFLUENCE` there as the bare commit (`tests/bindings/test_merged_influence.py`, pinned to
this position in `p23_wwby_pending_position.json`).
