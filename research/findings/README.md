# findings/ — what is true now, split by what can invalidate it

Short, current, cited. **Update discipline: rewritten in place** — a findings file states
conclusions and links to the append-only [`../log/`](../log/); when a number is superseded it is
replaced, and the narrative stays in the log. If a file here starts accumulating narrative it is
drifting into log territory and should be cut back.

## The split, and why it is the filing dimension

Two kinds of knowledge were mixed together in one flat directory, and they answer to different
things:

| | | invalidated by | survives |
|:---|:---|:---|:---|
| **[`engine/`](engine/)** | what the simulator does — rules, action-space and mask semantics, revision boundaries, and the instruments that measure them | a code defect found, a rules fix, a probe found to be lying | — a wrong instrument's numbers are simply gone |
| **[`training/`](training/)** | what a training choice is worth — architecture, reward shaping, opponent pooling, value targets, seed variance | a better-powered replication, more seeds | an engine revision, under the assumption below |

The owner's statement of it: *"while engine changes can make old arms incomparable with new, they
(after the starred events change) don't affect the relative effect of training approaches."* That
is a methodological assumption, not a result. What it licenses, what it does not, and the evidence
for and against it are in
[`../method/what_survives_an_engine_change.md`](../method/what_survives_an_engine_change.md) —
read it before carrying a number across a revision boundary in either direction.

The practical consequence: an **engine** finding is pinned to a commit and can invalidate absolute
numbers wholesale; a **training** finding is a difference measured inside one engine and is
expected to outlive the engine it was measured on. Both indexes,
[`../runs.md`](../runs.md) and [`../questions.md`](../questions.md), mark which kind a row is.

## engine/ — what the simulator does, and what changed when

| file | |
|:---|:---|
| [`engine_revisions.md`](engine/engine_revisions.md) | which game each arm was trained on: E1/E2/E3, the batch behind each boundary, what each fix invalidated, and the observation as a separate axis |
| [`engine_change_decision_stream.md`](engine/engine_change_decision_stream.md) | P14 changed the legality code and moved no decision — 1,068 games, 385,812 steps, zero divergences — and how to establish that for the next change |

Most engine-side knowledge does not live here, because it was already filed where it is read:

* **Open defects** — `BUGS.md` at the repository root (ENG-1, ENG-3, TEST-1).
* **Rules and design** — `engine/AGENTS.md`, including why `CardLocation` encodes hand knowledge
  (§7) and the free-coup validation bug (§8).
* **Instruments that lied** — [`../log/measurement_bugs.md`](../log/measurement_bugs.md) in full,
  and [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) as the checklist to
  read before trusting a number. These are engine/code knowledge by the test above: when an
  instrument is found broken, its numbers are gone rather than restated.
* **Reading the human corpus** — [`../log/P7_replayer_conversion.md`](../log/P7_replayer_conversion.md)
  and [`../method/human_play.md`](../method/human_play.md). The converter is an instrument; the
  corpus statistics it produces are inputs to training work.

## training/ — what a training choice is worth

| file | |
|:---|:---|
| [`architecture.md`](training/architecture.md) | what the network is now, and what each change was worth |
| [`pooling.md`](training/pooling.md) | pooled vs non-pooled opponents: every comparison, the current verdict, and the four unrelated things this project calls "pool" |
| [`seed_variance.md`](training/seed_variance.md) | the error bars a comparison has to clear, and the two results they have withdrawn |
| [`defcon_blunders.md`](training/defcon_blunders.md) | the critic cannot see a provoked DEFCON-1 at any node; what windowing does and what it costs |

## Where the split is not clean

Naming these is the point of having the split at all; a boundary with no listed exceptions is one
nobody checked.

* **[`seed_variance.md`](training/seed_variance.md)** is filed under training because seed spread
  is a property of the training process. Three of its rows are not: *a tournament is not
  reproducible from its configuration* (~1.5 points of win rate), *Elo is not comparable across
  tournaments* (~±15), and the binomial floor are facts about the measuring apparatus, and they
  hold whatever the engine does.
* **[`defcon_blunders.md`](training/defcon_blunders.md)** is a training finding — what the critic
  represents, and what a reward window costs — resting on an engine-side taxonomy. The
  provoked/self-inflicted split is computed from the engine's own `DEFCON_SUICIDE_PROVOKED` flag,
  and one of its sibling metrics was wrong for a rules reason (a turn-10 Wargames counted as final
  scoring; [`../log/measurement_bugs.md`](../log/measurement_bugs.md)). If the DEFCON rules change,
  the *behavioural* percentages move and the −68 Elo verdict on windowing does not.
* **Search** — [`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md) has no
  findings file yet. It is the clearest "both": the questions are training-side (does search buy
  strength, at what coverage) and its published numbers were invalidated twice by code fixes, once
  in the engine. Every row in §1 carries the commit it was measured at, for that reason.
* **The observation** is neither. It is a representation decision, owner-held (`CLAUDE.md`), and it
  invalidates checkpoints outright rather than making them incomparable — including, at equal
  width, silently. [`engine/engine_revisions.md`](engine/engine_revisions.md) treats it as a third
  axis.
* **Human data** — the corpus conversion is an instrument (engine side); *"is BC warmup on human
  games better than on self-play"* is a training question. They are filed apart deliberately.
