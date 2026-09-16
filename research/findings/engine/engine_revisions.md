# Engine revisions — which game each arm was actually trained on

What the simulator did at each point in the project's history, what each change to it fixed, and
what that change invalidated. **Update discipline: rewritten in place** — when a revision boundary
is re-dated or a new one opens, this file changes and the experiments stay in `../../log/`.

The three-letter scheme itself (when the letter bumps, what a name means) is
[`../../method/run_nomenclature.md`](../../method/run_nomenclature.md); the arms sitting on each
revision are [`../../runs.md`](../../runs.md). This file is the other half: not *what a name
means* but *what the engine did*, because an Elo number is a statement about a game and three
different games have been played here.

**The letter is coarse.** It has three values and the project has had several dozen `fix(engine)`
commits. A letter bumps when someone judged that the decision stream moved; the commits inside a
letter are the ones judged not to move it, and in two cases that judgement was measured rather
than assumed ([below](#changes-held-inside-e3-by-measurement)).

## The ladder

| letter | boundary commit(s) | date | what changed in the game |
|:---|:---|:---|:---|
| **E1** | everything before `cff2344` | to 2026-09-09 | a starred card spent for Operations was **deleted from the game**. Arms A–G and everything older. Not runnable today: their observation layouts are gone and `check_checkpoint_layout` refuses their checkpoints by width |
| **E2** | `cff2344`, `25d9b70` (+ the 09-10 batch below) | 2026-09-10 | the starred-card fix, plus observation v2.3. Arms H, H2, I |
| **E3** | `1a3b782`, `a18ceab` | 2026-09-11 | the Aldrich Ames discard and the Star Wars pick made mandatory. Everything from P1 onward |

## E1 was not one engine either

Two waves of rules fixes landed inside E1 and neither bumped a letter, because the scheme did not
exist yet.

* **2026-09-05** — per-card headline decision frames (`bea6240`), Defectors cancelling the USSR
  headline however it reaches the table (`a4c8ce6`), Shuttle Diplomacy (`fbc1b1e`), the
  scoring-card trap rule and the last turn's hand fill (`8db2928`), among others.
  [`../../log/engine_reanchor_and_human_control.md`](../../log/engine_reanchor_and_human_control.md)
  §7 re-anchored the ladder after them and is the project's first evidence on what an engine
  change costs: **the ordering survived** (K=40 > control > HeuristicBot > Random, and K=40's rate
  against the anchor read 90.2% against 88.9% before the change) while **everything measured
  through the self-play distribution did not** — the §4 deficiency tables, ending mixes and
  battleground counts had to be re-measured.
* **2026-09-06** — a free coup from Che or Ortega is still a coup (`d014cd4`). Before it, those
  two events ran their own target lists without consulting `Operations::can_coup` and offered
  coups against countries the opponent had no influence in; couping a battleground took DEFCON
  2 → 1 and ended the game. **29 of the 64 "forced wins" the human-corpus metric counted were
  these illegal coups**, which is most of why that result was withdrawn
  ([`../../log/engine_reanchor_and_human_control.md`](../../log/engine_reanchor_and_human_control.md)
  §8.4, and `engine/AGENTS.md` §8). A policy trained against it learned that an opponent's Ortega
  is a free win whenever a battleground sits next to Nicaragua.

## The E1 → E2 boundary is a batch, not a commit

"The starred events change" names two commits on 2026-09-10, and eight more landed the same day.
Anything that dates the assumption in
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md)
is dating this whole batch.

| commit | what it changed |
|:---|:---|
| `cff2344` | a starred card **spent for Operations** is discarded, not removed. Measured over 600 games: 19 of 19 starred cards played for Ops by their own side had been permanently removed; after, 12 of 12 discarded and 0 removed |
| `25d9b70` | the other half — a starred card whose event **fires and cannot occur** is discarded. Seven cards can reach that state; six of them were being removed |
| `42c4b78` | the event-occurred rule reaches the five sites outside the state machine that had drifted from it |
| `712bce4` | the pending die roll gets named fields, and forced dice leave the state |
| `bde245e`, `49ed564` | `temp_cards` removed; the cards a decision is about are read from where the cards are |
| `18a398e` | `node_counts` packed; event chains nest six deep instead of three |
| `32902a3` | Socialist Governments enforces its own per-country cap |
| `430ba9b`, `9591470` | Europe Control becomes its own ending rather than an ordinary 20 VP win |
| `4fb3cff` | observation **v2.3** — a separate axis; see below |

The starred bug's consequence was structural rather than local: every starred own-card Ops play
deleted a card from the game, so the deck shrank, reshuffles came early, and those events could
never appear again for either side. It stood for 380 of the repository's 389 commits, and the
observation's `removed` and `not_in_game` slots carried wrong values throughout. The human corpus
could not have caught it — the converter checks per-country influence and never looks at card
locations (`cff2344`'s own commit message).

**What this boundary invalidated, as the record states it.**
[`../../log/corrected_engine_arms_H_I.md`](../../log/corrected_engine_arms_H_I.md): *"Nothing before
this is comparable to it. The fix changes the decision stream, so every Elo in §1–§24 is on a
different ladder."* Note that the same section then rates pre-fix arms D and E **in the
corrected-engine field**, at 1739.7 and 1763.2, below the post-fix arms — so the record asserts
incomparability and performs the comparison on the same page. Both readings are defensible (an old
checkpoint is a valid *opponent* even when its rating is not on the same ladder) but the tension is
real and is one of the reasons the assumption document exists.

## The E2 → E3 boundary

`1a3b782` and `a18ceab`, 2026-09-11: the Aldrich Ames discard and the Star Wars pick stop being
optional. Two cards, both rare.

The standing handicap statement is
[`../../method/run_nomenclature.md`](../../method/run_nomenclature.md) rule 2 — an E2 checkpoint
evaluated on an E3 engine *"is playing a game it never trained on; the handicap is small for the
two mandatory-choice cards but it is not zero, and it biases in favour of the E3 arm"* — repeated
in [`../../log/P9_architecture.md`](../../log/P9_architecture.md) as *"the handicap is small — two
rare cards — but it runs against the reference, so it flatters the E3 arms slightly."*
**Neither statement has a measurement behind it.** Nothing in the record puts a number on the two
cards' frequency or on what the handicap is worth in Elo, and `E2-02-21-480M` is nonetheless the
project's standing rating anchor and the reference every P1 arm was rated against.

## Changes held inside E3 by measurement

Two `engine/` changes were deliberately *not* given a new letter, and in both cases the record says
what was measured rather than what was assumed. That is the practice this project has converged on.

| change | what was measured | where |
|:---|:---|:---|
| `b6874af`, `9f78026` — reject a mismatched `decision_type`; a chance node's only legal action rolled 255 instead of a die | no `ROLL_DIE` node is ever handed to an agent (0 in 879 single-env and 12,800 vectorized env-steps) and the runner always forces die 0 | [`../../runs.md`](../../runs.md), *E3-21* |
| **P14** — `StateMachine::step` validates against the mask; the Missile Envy forced-play rule fixed (`5938e52`) | 1,068 games / 385,812 steps under four policies, comparing outcome, length, chosen action **and the legal mask itself** at every step: 0 divergences | [`engine_change_decision_stream.md`](engine_change_decision_stream.md) |

Both kept `E3-20-28` usable as a matched baseline for `E3-21-28`, which is the difference between
running one arm and two.

## The observation is a separate axis, and a harder one

The letter tracks the *game*. The observation tracks what the network is told about it, and it has
its own history — legacy 4,293 floats, v2.1, v2.2, then **v2.3 at 3,824 floats, the only one that
still exists**. The two axes moved together at the E1 → E2 boundary (`4fb3cff` landed with the
starred fix) which is why arms D through G cannot be re-rated even in principle: v2.2 is retired
with its `temp_card_count` slot.

An observation change is stricter than an engine change in one specific way. An engine change
alters what the same checkpoint *plays*; an observation change makes the checkpoint unloadable, or
worse, loadable and silently misreading — the failure mode that produced four separate measurement
bugs ([`../../method/measurement_pitfalls.md`](../../method/measurement_pitfalls.md), *the input the
model was given*). `check_obs_width` and `check_checkpoint_layout` now refuse by width, which
catches a changed *width* and cannot catch changed *content* at the same width. What each layout
was worth is [`../../log/observation_layout.md`](../../log/observation_layout.md).

## Open, and worth knowing

* **No arm has ever been re-run across a letter boundary.** No configuration was trained on E1 and
  again on E2, or on E2 and again on E3, so no measurement in this repository shows a *training*
  result surviving — or failing to survive — a rules change. See
  [`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md).
* **`BUGS.md` ENG-1** (UN Intervention offers companion cards the rules forbid) is open, which
  means the current E3 engine is known to be playing a slightly wrong game right now.
* **ENG-3's own text predicted the opposite of what P14 measured.** It said of the forced-play fix:
  *"they change what is legal, so they change the decision stream and invalidate comparisons across
  the change."* The change landed and moved no decision in 385,812 steps. The prediction was a
  reasonable default; the measurement is what settled it.
