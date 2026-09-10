# Twilight Struggle AI: C++ Simulation Core Guide

This directory contains the zero-allocation, high-throughput simulation engine for the Deluxe Edition of **Twilight Struggle** (110 Cards).

---

## 1. Core Architectural Pillars

1. **Zero Heap Allocation in Simulation Core**: The core data structures (`GameState`, `DecisionContext`, `MicroAction`) use fixed-size memory layouts. `GameState` is strictly trivially copyable and under 4 KB (`sizeof(GameState) <= 4096`).
2. **Micro-Decision State Machine**: All multi-step actions (Ops, events with multiple target choices, headline selection, space races) are decomposed into fine-grained atomic steps (`DecisionType`).
3. **High Simulation Throughput**: Single-core simulation speed exceeds **2,000,000 steps/second**, meeting reinforcement learning and MCTS training requirements.
4. **Deterministic Bit-for-Bit State**: Built-in 64-bit SplitMix64 PRNG (`Prng`) ensures bit-for-bit replayability from integer seeds.
5. **Unified Sub-Decision Processing**: State machine handles sub-decisions (e.g. `SELECT_OP_MODE`, `POINT_NODE`, `CHOOSE_TIMING_BRANCH`, `CHOOSE_BRANCH`) uniformly across both `Phase::HEADLINE` and `Phase::ACTION_ROUND`.

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
├── CMakeLists.txt              // Build target for ts_engine library, ts_tests, ts_fuzz, ts_benchmark
├── AGENTS.md                   // Developer and agent documentation (this file)
├── progress.md                 // Progress report and future work items
├── include/ts/                 // Public and internal engine headers
│   ├── types.hpp               // Enums: Player, Phase, WarEra, CardLocation, DecisionType, ActionType, Region, SubRegion, PlayMode, TimingBranch, OpMode
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
│   ├── action_mask.hpp         // Legal action mask generator (per DecisionType and unified 212-dim flat mask)
│   ├── state_machine.hpp       // Turn and Action Round lifecycle, Headline resolution
│   ├── observation.hpp         // Neural observation feature extractor (ObservationBuffer)
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
    ├── test_fuzz.cpp           // Invariant fuzzer (--games <N>, --steps <N>, --seed <S>)
    └── test_benchmark.cpp      // 500k-step throughput benchmark
```

---

## 4. Micro-Decision Pipeline & Action Representation

The engine splits complex turns into a sequential stream of atomic 4-byte `MicroAction` structures:

```cpp
struct alignas(4) MicroAction {
    DecisionType decision_type; // 1 byte: SELECT_CARD, SELECT_PLAY_MODE, CHOOSE_TIMING_BRANCH, SELECT_OP_MODE, POINT_NODE, CHOOSE_BRANCH
    uint8_t      primary_id;    // 1 byte: Card ID (1..110), Country ID (0..83), Branch ID (0..7), PlayMode, OpMode, TimingBranch, or CONFIRM_DONE (0x80)
    uint8_t      secondary_id;  // 1 byte: Sub-choice / quantity / manual die roll
    uint8_t      flags;         // 1 byte: Additional modifiers / opponent manual die roll
};
```

---

## 5. How to Build, Test, and Benchmark

### Standard Build:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
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

### Run Unit Tests (299 tests):
```bash
./build/engine/ts_tests
# Or with sanitizers:
./build_san/engine/ts_tests
```

### Run Invariant Fuzzer:
```bash
# Run 10,000 full games:
./build/engine/ts_fuzz --games 10000

# Run 5,000,000 steps:
./build/engine/ts_fuzz --steps 5000000 --seed 42

# Under ASan + UBSan:
./build_san/engine/ts_fuzz --games 10000
```

### Run Performance Benchmark:
```bash
./build/engine/ts_benchmark
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
11. **An Opponent's Card Owes Its Event In A Headline Too**: playing an opponent's card for
   Operations asks which resolves first, and `OPS_FIRST` leaves the Event owed until the Ops
   are spent. `advance_after_ops` fires it on both paths -- the headline's as well as the
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

## 7. Why hand knowledge lives in `CardLocation`

`card_locations` distinguishes `HAND_US_KNOWN` from `HAND_US_UNKNOWN` rather than carrying a
parallel "known" bitset, and the bare `HAND_US` / `HAND_USSR` constants deliberately no longer
exist. Both decisions are load-bearing, and re-adding either name would reintroduce roughly
thirty silent rules bugs.

**As built** (`engine/include/ts/types.hpp`), eleven values, of which four are hand variants:

```
UNAVAILABLE=0  DRAW_DECK=1  HAND_US_UNKNOWN=2  HAND_US_KNOWN=3  HAND_USSR_UNKNOWN=4
HAND_USSR_KNOWN=5  DISCARD_PILE=6  REMOVED_FROM_GAME=7  ONGOING_EVENT=8  PEEKED_TEMP=9
HEADLINE_COMMITTED=10
```

Access goes through `is_in_any_hand`, `in_hand_of`, `known_to_opponent`, `hand_of`, `hand_holder`
and `revealed`, never through a bare comparison.

The argument that settled the design follows, kept as it was written in the experiment log
(`research/experiments.md` §19.5). It proposes `HAND_US` / `HAND_USSR` for the unknown variants;
those were named `HAND_US_UNKNOWN` / `HAND_USSR_UNKNOWN` when built, precisely so that no old bare
name survives. Read it for the reasoning, not for the spelling.


§19.3 proposed a `known` bitset alongside `card_locations`, and noted that knowledge is
per-observer so it would need *two* bitsets. **Both of those were wrong.**

**One field is enough.** A card's holder always knows their own hand, so the only fact that varies
is whether the *other* player knows. `HAND_US_KNOWN` therefore reads unambiguously as "in the US
hand, and the USSR knows it" — the holder is in the value, and "known" can only mean known to the
non-holder. There is no second observer to track. The full space is the one proposed:

```
UNAVAILABLE, DRAW_DECK, HAND_US, HAND_US_KNOWN, HAND_USSR, HAND_USSR_KNOWN,
DISCARD_PILE, REMOVED_FROM_GAME, ONGOING_EVENT, PEEKED_TEMP, HEADLINE_COMMITTED
```

`card_locations` is already `uint8_t[111]` using 9 of 256 values, so **two more cost zero bytes** —
against 14 bytes for a bitset, inside a `GameState` capped at 4 KB.

**And it puts the risk where the compiler can find it.** This is the real argument, and it is a
counting argument:

| | sites | what goes wrong if one is missed |
|---|---:|---|
| writes to `card_locations` | **105** | with a bitset: the card moves to the discard and the bit is not cleared, so the observation reports the opponent holding a card that is visibly in the discard. Silent, and it corrupts the new feature. |
| reads comparing to a hand | **52** | with separate locations: a known card fails `== HAND_US`, so its holder cannot play it, it vanishes from hand counts and from discard selection. A rules bug — but one that can be made a *compile* error. |

With separate locations the 105 writes are correct by construction: assigning any new location
destroys the knownness, which is exactly the monotonicity rule — knowledge ends when the card
leaves the hand, and it ends automatically. With a bitset every one of those 105 sites has to
remember to clear it.

So the proposal has fewer risky sites (52 against 105) *and* moves the risk from silent to
detectable. It is the better design on both counts.

**The one condition.** The 52 reads are all bare equality — `card_locations[c] == HAND_US`, or the
`loc = (p == US) ? HAND_US : HAND_USSR` idiom that then compares. Adding values silently breaks
every one. So the change must be made compiler-visible: **remove or rename the bare `HAND_US` /
`HAND_USSR` constants** so that every existing site fails to compile, and reintroduce access through
helpers:

```cpp
bool in_hand_of(CardLocation loc, Player p) noexcept;   // either variant
bool known_to_opponent(CardLocation loc) noexcept;
CardLocation hand_of(Player p, bool known) noexcept;
```

Done that way the compiler enumerates all 52 call sites and none can be forgotten. Done by *adding*
values while leaving the old names in place, roughly thirty of them become silent rules bugs, and
the engine has been bitten by exactly this before — the `keeps_own_card_location` comment in
`game_state.hpp` documents Missile Envy being discarded out of a hand it had just been moved into,
stranding `forced_card_id` on a card nobody held, "and the action mask, which only forces a card
that is actually in hand, then drops the forced play without a trace."

**A note on precedent.** `PEEKED_TEMP` and `HEADLINE_COMMITTED` are existing non-obvious location
values, but neither is a *hand variant* — both mean "not in a hand right now", and the code treats
them as out of play. `HAND_US_KNOWN` would be the first location that must behave **identically to
an existing location in every rule** and differ **only in the observation**. That is what makes the
read audit the whole job, and it is why the helper-plus-rename discipline is not optional.

**Observation side.** `canon_loc` (`observation.cpp:160-185`) currently folds opponent-hand cards
into slot 0 with the draw deck. It gains one case: a card in the opponent's hand that is *known*
maps to a new slot rather than to 0, while an unknown one keeps folding into 0. From the holder's
own perspective both variants map to `MY_HAND` unchanged. That is one extra card feature — 110
floats — against the 512 being removed with the history.

Recommendation unchanged from §19.4, with the mechanism settled: do it as separate locations, in the
same breaking change as removing the history, behind helpers that force the compiler to walk the 52
sites. And keep §19.4's caveat — this addresses the §14–§17 card-play cluster, not the game-length
constraint that §18 identifies as binding.

---

## 8. Free-coup events must go through `Operations::can_coup`

An event that grants a coup outside the ordinary Operations path does not get target
validation for free. Two handlers once built their own target lists and offered coups the rules
forbid; the account below is kept because the consequence reached the training signal, not just
a metric. Fixed, with `tests/engine_logic/test_free_coup_target_legality.py` covering both.


My first reading of these, that DEFCON-1 losses were attributed to the wrong player, was **wrong**.
`resolve_defcon_one_loss` (`engine/include/ts/defcon.hpp:28`) makes the *phasing* player lose
regardless of who drove DEFCON down, which is the rule. The engine is right about that.

The actual defect is narrower and worse. **Two events run their own free-coup target lists and
never consult `Operations::can_coup`:**

| card | site | what it validates |
|:---|:---|:---|
| #91 Ortega Elected in Nicaragua | `card_dispatcher.cpp:1431` | adjacency to Nicaragua only |
| #107 Che | `card_dispatcher.cpp:1402` | region, non-battleground, not visited |

`can_coup_or_realign` refuses a country the opponent has no influence in
(`engine/src/ops.cpp:119`), and `get_coup_target_mask` is built on it, so an ordinary Ops coup is
filtered correctly. These two bypass it, and so offer coups the rules forbid — along with,
presumably, the DEFCON regional restrictions, NATO and The Reformer, which live in the same
function.

**Replay 139, turn 9, action round 2.** The US played Ortega — a USSR card — for Ops, so its event
fired and handed the USSR a free coup. The engine offered **Cuba**. The log records Cuba as
`inflUS 0 / inflUSSR 3` at *every* entry of turn 9, and the engine state agrees exactly, so the
reconstruction is correct and the board is not in doubt. With no US influence there, the USSR
cannot coup Cuba. But Cuba is a battleground, so the offered coup took DEFCON 2 → 1 and ended the
game against the phasing player, the US. That is why it scored as a USSR "win".

Same shape at **replay 16 T9 AR3**, **replay 165 T9 AR3**, **replay 245 T8 AR1** — all Ortega, all
Cuba, all `US 0 / USSR 3`.

`tests/engine_logic/test_free_coup_target_legality.py` reproduces both synthetically: Ortega offers
`[67, 68, 71]` including Cuba with zero US influence, and Che offers 26 countries without checking
influence at all. A third test confirms the ordinary Ops path filters correctly, so the defect is
in the two event handlers, not in the coup rule. **Fixed** (approved): both handlers now call `Operations::can_coup(state, Player::USSR, i)`, at
the target mask in `get_event_action_mask` and again where the chosen target is applied. The
forced win at replay 139 T9 AR2 is gone, all 368 C++ tests pass, the fuzzer is clean over 3,000
games, and all 282 corpus games still convert with 0 failures. One existing C++ test,
`OrtegaElected_CanCoupCuba_AndAdjacentCountries`, asserted the old behaviour -- it gave Cuba US
influence but left Costa Rica and Honduras empty and expected them offered anyway -- and now sets
up influence in those two and additionally asserts that an adjacent country with none is refused.

**Why this matters beyond the metric.** `classify_legal_actions` reads the engine's terminal
utility, and so does every reward. A policy trained against this learns that an opponent's Ortega
is a free win whenever a battleground sits next to Nicaragua — a move the rules do not permit.

