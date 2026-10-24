# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood. Missing features and unfinished work belong
in the documentation for the area they concern.

Fixed bugs keep a one-line entry at the end, because code and research documents cite their IDs;
the full write-up is in git history.

---

## EVAL-1 — leaderboard records do not identify overlapping deal sets

**Area:** leaderboard evaluation · **Severity:** P2 · **Status:** open

Duplicate detection compares base seeds, while the deals played depend on the base, game count
and chunk size. With 1,000 games per side and chunk size 1,000, neighboring bases share 998 of
1,000 deal seeds for the same pairing, yet both records are accepted as independent evidence.
Historical match records omit chunk size, so their actual seed sets cannot be reconstructed
reliably from the records alone.

**Fix.** Record the actual deal set and sampling protocol for future matches and reject overlap
per pairing. Recover historical chunk provenance before migrating existing evidence; preserve
its original interpretation rather than inferring a chunk size from current defaults.

---

## TEST-1 — the differential fuzzing suite does not run

**Area:** tests · **Severity:** low · **Status:** open

`tests/differential/` cross-checks this engine against an independent implementation
(`external/struggler`). Its modules fail at *import*, so `tests/conftest.py` keeps them out of
collection and they sit behind the `differential_fuzz` marker and a `--run-fuzz` flag. The
repository's cross-engine check is therefore not currently a check; nothing else depends on it.

**Fix.** Repair the imports and re-enable collection, or remove the suite. Do not leave it
collectable but broken: a suite that aborts the run is worse than one that is honestly gated.

---

## CONV-1 (remainder) — the committed verdict bank predates the final converter

**Area:** human replay conversion / eval · **Severity:** low · **Status:** open, owner's decision

The converter fix itself is done (below). But of the 100 positions in
`ai/eval/banks/disagreement_verdicts.jsonl`, 34 are no longer produced by the current converter:
nearly all differ in what the network observes (the mover's hand holds other cards, or fewer),
and one (replay 147, turn 9) is in a turn now left out as untrusted. 28 of the first 33 carry
marks. The reviewer
judged those exact positions, so the verdicts were not moved; carrying them to the current
positions, or re-reviewing, is the owner's call. `tests/training/test_disagreement_bank.py`
checks that each position loads and each mark is legal, not that the converter still produces it.

They also predate CONV-2: 94 of the 100 positions hold in the discard pile a card the log had not
spent by that turn, 7.1 on average, and the network observes the pile.

---

## BIND-1 — a long disagreement-bank run can stop on a nanobind instance collision

**Area:** bindings · **Severity:** low · **Status:** open, not reproduced

A long `tools/scripts/disagreement_bank.py` scan occasionally stops on a nanobind instance
collision (`tools/README.md` §7b); the scan was made resumable (`--resume`) rather than fixed. No
traceback or reproducer was recorded. A collision means two Python objects claimed one C++
address -- an object's lifetime, e.g. a `GameState` or runner slot freed while Python still names
it -- and the same fault could corrupt a position silently instead of stopping the run.

**Next.** Reproduce under the sanitizer build (`tools/scripts/run_asan.sh`) with a long scan, and
record the traceback here.

---

## CONV-3 — from turn 4 the converted piles still drift from the log's

**Area:** human replay conversion · **Severity:** low · **Status:** open

Since CONV-2 and CONV-4 the discard pile at a converted decision is the log's through turn 3
(`tests/replayer/test_replay_deck_and_discards.py`), and since ENG-4 all 469 of the corpus's
`*RESHUFFLE*` markers fall on the entry where the engine reshuffles. From turn 4 a decision's pile
can still hold cards the log has not spent and lack ones it has: 0.15 and 0.03 per decision,
averaged over every converted decision (13.2% and 1.6% of decisions have any; 0.25 and 0.04 before
ENG-4, which fixed the mid-draw reshuffles of Our Man in Tehran and Ask Not in 156, 207, 272 and
314). Part of the 0.15 is the measurement's own: an ongoing card such as Shuttle Diplomacy, played
before a reshuffle and discarded after it, is counted as never spent. What is known of the rest:

* **A card received through Missile Envy is missing from the receiver's solved hand** until the
  entry that shows the receiver with it. In replay 109 the USSR headlines Missile Envy at turn 5;
  the engine holds it as `HAND_US_KNOWN`, the solved US hand for the USSR's AR1 leaves it out, and
  at those decisions it sits in the deck. Missile Envy is the card most often missing (339
  decisions in 74 games); only replay 109 has been traced.

The engine also reshuffles at five entries with no marker (replays 201, 221, 256, 265, 299), none
of them wrong. Each is the last entry the conversion reaches, at the end of a turn, and the engine
deals the next turn; in 201 and 256 the log's next entry, which the conversion stops before,
carries the marker.

---

## Fixed (kept for the IDs cited elsewhere)

| ID | what it was | fixed in |
|:---|:---|:---|
| ENG-1 | UN Intervention's companion mask offered every card in hand (the correct filter in `action_mask.cpp` was unreachable); an illegal pick silently cost the action round | `3ef2d92` (P17 6a); `engine/AGENTS.md` §9, test in `engine/tests/test_cards_early.cpp` |
| ENG-2 | 2026-09-28 rules audit: seven rare-situation rules wrong (war-roll defender, Vietnam Revolts ladder, UN Intervention as an event, We Will Bury You on a trapped round, NORAD at 2, Grain Sales→UN with U-2, Summit counting scoring-only cards). Defines engine **E7** | `acc68d6`; `engine/AGENTS.md` §12, `engine/tests/test_rules_audit.cpp`, `research/findings/engine/engine_revisions.md` |
| ENG-4 | Our Man in Tehran drew at most what the deck held, and Ask Not's replacement draws left an emptied deck for the next deal to reshuffle: the humans' game reshuffled inside the draw (ts-replayer 156 turn 6 AR4, 207 turn 7 AR2 and 314 turn 7 AR6, Tehran; 272 turn 7 AR7, Ask Not), the engine a turn or two later, with every pile in between wrong. Both now reshuffle the moment the deck runs out, as the deal does, and reveal what an empty deck reveals there: all 469 of the corpus's `*RESHUFFLE*` markers fall where the engine reshuffles (465 before), and the pile holds 0.15 unspent cards per decision (0.25). 10 of 2,000 non-suicidal random games play differently, all from the first Ask Not or Tehran that runs the deck out. Moves the behaviour digest. Defines engine **E8** | `380358d`; `engine/AGENTS.md` §13, `Card108_OurManInTehran_*`, `Card77_AskNot_*` in `engine/tests/test_cards_mid.cpp` |
| ENG-5 | The deal filled the USSR hand to the limit before dealing the US any card. The order shows only when the deck runs out mid-deal, which it does on every turn 3: the USSR got nearly every Early War card left in the old deck and the US drew from the reshuffled pile, so a card the USSR missed on turn 2 nearly always reached it on turn 3 (the bots valued a USSR turn-2 Op at 0.05 pp, against 1.3 on other turns). The rules name no order; the human corpus alternates (the US's turn-3 hand holds 3.8 never-discarded Early War cards on average, ≥3 in 247 of 274 games) and so does struggler. Now one card at a time, USSR first. Every seed deals different hands; moves the behaviour digest in every section, so it needs a new engine epoch | `pr/engine-alternate-deal`; `engine/AGENTS.md` §14, `Deal_*` in `engine/tests/test_card_edge_cases.cpp` |
| CONV-1 | the human-log converter seated cards in hands the player never held (Our Man in Tehran peeks, reveals not used as evidence, late arrivals placed at AR1): over-limit hands in trusted turns 178 games / 2,590 decisions → 39 / 186. Rebuild anything made from converted human positions after it | `8aec4db`; `tests/replayer/test_replay_hand_contents.py` |
| CONV-2 | the human-log converter put every card the solved hands leave out in the discard pile, not the deck: at 99.7% of converted decisions the pile held cards nobody had played (14.2 on average) and the engine reshuffled where the humans had not, 24 games losing a SALT pick to it. Terrorism's random discard was never steered, and Five Year Plan's took the wrong card in 5 of 237 entries. Now 0.34 per decision from turn 4 (CONV-3), none before. Rebuild anything made from converted human positions after it | `0f112d7`; `tests/replayer/test_replay_deck_and_discards.py`, `tests/replayer/test_replay_five_year_plan_discard.py` |
| CONV-4 | the hand solver freed a starred card its event had removed at the next `*RESHUFFLE*`, like any spent card, and the converter seated it out of the engine's removed cards: a removed card was back in a hand or the deck at 19.1% of converted decisions, in 252 of the 300 games. Each one left the engine's deck a card bigger than the humans', so the turn-7 deals of replays 59, 119, 228 and 302 did not reshuffle where the log does. Our Man in Tehran's peek was seeded from every discard in its entry, so in replay 277 Our Man in Tehran, fired by a Five Year Plan discard, went into its own peek and on to the pile. Now none; the pile holds 0.25 unspent cards per decision, from 0.34, and 465 of 469 markers match, from 461. Rebuild anything made from converted human positions after it | `de64176`; `tests/replayer/test_replay_removed_cards.py` |
| TODO (forced deal) | no way to drive a deal from a recording, so replays could not survive a shuffle change | `dfadce5`: `GameState.set_forced_deal` / `get_forced_deal_remaining`; `tests/bindings/test_forced_deal.py` |
