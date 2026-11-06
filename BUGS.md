# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood. Missing features and unfinished work belong
in the documentation for the area they concern.

A fixed bug is removed: its ID and write-up stay in this file's history (search it for the ID), and
the rule it fixed is documented with the code.

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

The converter fix itself is done (`8aec4db`). But of the 100 positions in
`ai/eval/banks/disagreement_verdicts.jsonl`, 34 are no longer produced by the current converter:
nearly all differ in what the network observes (the mover's hand holds other cards, or fewer),
and one (replay 147, turn 9) is in a turn now left out as untrusted. 28 of the first 33 carry
marks. The reviewer
judged those exact positions, so the verdicts were not moved; carrying them to the current
positions, or re-reviewing, is the owner's call. `tests/training/test_disagreement_bank.py`
checks that each position loads and each mark is legal, not that the converter still produces it.

They also predate CONV-2 (`0f112d7`): 94 of the 100 positions hold in the discard pile a card the log had not
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

Since CONV-2 (`0f112d7`) and CONV-4 (`de64176`) the discard pile at a converted decision is the log's through turn 3
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
