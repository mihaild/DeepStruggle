# What survives an engine change — the assumption this record runs on

The project keeps two kinds of knowledge and treats them as one, which is why an engine fix reads
as though it invalidates everything. It does not, and the owner states the distinction as:

> While engine changes can make old arms incomparable with new, they (after the starred events
> change) don't affect the relative effect of training approaches.

This file states that as a **methodological assumption**: what it licenses, what it does not, and
what the record actually contains for and against it. **Update discipline: living reference,
rewritten in place.** It is filed in `method/` rather than `findings/` because it is not a result —
it is a rule about how results may be read, and the split it justifies is the one
[`../findings/README.md`](../findings/README.md) implements.

## The claim, stated precisely

Let *A* and *B* be two training configurations differing in one factor — an architecture change, a
reward term, an opponent pool, a value target — each measured against its own matched control
**inside one engine revision**. The claim is that the *sign and rough size* of (A − B) transfers
across a revision boundary, while the *absolute* rating of A or B does not.

Two things it does not say, both of which have been read into it at some point:

* It says nothing about a checkpoint's rating. An Elo number is a statement about a specific pool
  of opponents playing a specific game; change the game and the number is about something else.
* It is scoped **after the starred events change**, not to the whole history.

## Scope — which date, and what exactly landed on it

"The starred events change" is `cff2344` and `25d9b70`, both **2026-09-10**, the E1 → E2 boundary.
That is the date the assumption's scope begins.

It is a *batch*, not a commit: eight further engine and observation commits landed the same day,
including observation v2.3 (`4fb3cff`), Europe Control becoming its own ending (`430ba9b`,
`9591470`) and the Socialist Governments cap (`32902a3`). Anything claimed "after the starred
change" is claimed after all of them. The full list is
[`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md).

Why that boundary and not another: the starred bug deleted a card from the game on every starred
own-card Ops play, so the deck shrank and reshuffles came early. That is not a rules detail with a
local effect — it changes the resource every strategy is built on, and it stood for 380 of the
repository's 389 commits at the time. A training result measured under it is a result about a
different game, not a noisier measurement of this one.

## What it licenses

* **Comparing a training result across a revision boundary.** "The self-transform was worth +53 and
  +83 Elo against its control" remains a claim about the self-transform after a later engine fix,
  provided both arms of the comparison sat on the same engine when they were run.
* **Not re-running a control after an engine fix**, when the fix leaves the decision stream alone.
  That saves roughly the compute of the arm itself; it is the reason `E3-20-28` is still a matched
  baseline for the value-bootstrap arm rather than dead weight.
* **Carrying a *rejection* forward.** Per-entity heads replacing the dense logit cost 219 Elo; that
  design is not revisited because the engine was fixed underneath it.

## What it does not license

* **Comparing Elo, win rate, or any absolute rating across revisions.** These are the numbers the
  owner's own sentence says go incomparable. `E2-02-21-480M` is the standing anchor and it is an
  E2 checkpoint; every P1 arm rated against it carries a cross-engine asterisk.
* **Comparing anything measured through the self-play distribution.** Ending mixes, mean ply,
  empty-battleground counts and blunder rates are properties of the *game*, and the first
  re-anchor in this record ([`../log/engine_reanchor_and_human_control.md`](../log/engine_reanchor_and_human_control.md)
  §7) found exactly that split: the ordering of four players survived six engine fixes and every
  distributional table had to be re-measured.
* **Skipping the measurement because a change "looks small".** See the evidence below: the change
  the record predicted would move the stream moved nothing, and a batch of harness and engine fixes
  that nobody expected to matter reversed two search conclusions.
* **Anything about an observation change.** The observation is a third axis. A checkpoint on a
  retired layout does not play a different game, it misreads this one — and at equal width it does
  so silently. Nothing here applies to it.
* **Comparing across budgets or without seeds.** The assumption removes one confound; it does not
  touch the two that have actually invalidated results here, which are seed spread (~95 Elo) and
  stopping point (~30 Elo). See [`../findings/training/seed_variance.md`](../findings/training/seed_variance.md).

## The evidence, for and against

### For

**1. The E1 re-anchor — the ordering survived six engine fixes.**
[`../log/engine_reanchor_and_human_control.md`](../log/engine_reanchor_and_human_control.md) §7,
2026-09-05 batch, 4 models, 6,000 games. K=40 > control > HeuristicBot > Random before and after,
and K=40's rate against the anchor read 90.2% after against 88.9% before. This is the closest thing
in the record to a direct test, and it has two limits: it compares *checkpoint rankings*, which is
weaker than comparing an intervention's *effect size*, and it is **inside E1**, before the
boundary the assumption names.

**2. P14 — an engine correctness change that moved no decision at all.**
[`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md).
The mask/step collapse plus the Missile Envy forced-play fix (`5938e52`), measured against
`a09e15a`: 1,068 games, 385,812 steps, four policies — a policy-free deterministic walk plus argmax
play from `E3-20-28@160M`, `E3-20-28@320M` and `E3-17-25@160M` — comparing outcome, game length,
chosen action *and the legal mask itself* at every step. **Zero divergences of any kind.** An
engine change that does not move the decision stream cannot move a training result, so this is a
concrete instance of the claim holding, and it is the cheap way to establish it for any future
change.

Its limit is the reason it is so clean: P14 fixed conditions that do not arise in self-play (a
1.54M-position sweep found zero naturally occurring mask/step disagreements). It demonstrates the
claim for a change that touches nothing. It says nothing about a change that touches something.

**3. The `ROLL_DIE` pair, held at E3 by measurement.** `b6874af` and `9f78026` changed `engine/`
between `E3-20-28` and the registered `E3-21-28`, and the letter was held because no `ROLL_DIE`
node is ever handed to an agent (0 in 879 single-env and 12,800 vectorized env-steps). Same
pattern: measured, not assumed ([`../runs.md`](../runs.md), *E3-21*).

### Against, or thin

**4. The transition the claim names has never been tested.** No configuration has been trained on
one side of a letter boundary and re-trained on the other. The claim that a *relative* effect
survives the starred-card fix is therefore **asserted, not measured** — as is the same claim for
the E2 → E3 mandatory-choice fixes.

**5. The one cross-boundary rating in the record is knowingly biased.** Every P1 arm was rated
against `E2-02-21-480M`, an E2 checkpoint, on an E3 engine. The record says the handicap "is small
for the two mandatory-choice cards but it is not zero" and "flatters the E3 arms slightly"
([`run_nomenclature.md`](run_nomenclature.md),
[`../log/P9_architecture.md`](../log/P9_architecture.md)). **No number exists for it** — not a
frequency for the two cards, not an Elo cost. A bias that is asserted and never sized is a bias
that cannot be subtracted.

**6. A code-fix batch did reverse relative conclusions once — in search.**
[`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md) §9. Re-measured at
`c7e3731` after `8533a68`, `b6874af` and `9f78026`, on the same checkpoint:

| comparison | before the fixes | after |
|:---|---:|---:|
| privileged search vs honest search | 100% vs 58.3% (12 games each) | +2.5 pp ± 5.5, z = 0.46 (120 games each) |
| 384 sims vs 96 sims | 76.7% vs 72.5% (30 and 40 games) | −1.7 pp ± 5.6, z = −0.29 (120 games each) |

Both are *relative* comparisons between two decision-making configurations, and both reversed.
Two caveats keep this from being a clean counterexample, and they matter: the sample sizes rose
three- to tenfold, so the old figures were two noisy point estimates and the re-measurement is the
better instrument regardless; and two of the three fixes are harness code (`8533a68`) or affect the
probe loop rather than training. But the third, `9f78026`, is `engine/` — a chance node whose only
legal action rolled 255 instead of a die, which makes space race attempts and coups succeed
automatically for *both* sides. The record itself says of the old numbers: *"Neither is proven to
have fired in these particular probes, and neither is ruled out."*

**7. `BUGS.md` ENG-3 asserted the opposite, and was wrong.** Of the forced-play fix: *"Both halves
are the owner's call: they change what is legal, so they change the decision stream and invalidate
comparisons across the change."* It landed as P14 and moved nothing in 385,812 steps. This cuts
*for* the assumption on the facts and *against* the practice of predicting either way without
measuring — which is the same lesson from both directions.

**8. Game shape does not replicate even within one engine.** Arm H suggested the corrected engine
lengthened games; its own seed replicate H2 landed on the pre-fix arms instead
([`../log/corrected_engine_arms_H_I.md`](../log/corrected_engine_arms_H_I.md)). Some of what looks
like an engine effect is seed variance, in both directions.

## Standing verdict

**Plausible, partly evidenced, and untested on the boundary it names.** Nothing in the record
contradicts it; nothing in the record demonstrates it for a rules change that actually moves play.
The strongest support (P14) is for a change that moves nothing, and the strongest support for the
general shape (the E1 re-anchor) is about rankings rather than effect sizes and sits before the
scope begins.

Treat it as a working assumption with a cheap escape hatch, not as a licence:

1. **Measure the decision stream instead of arguing about it.** Build both engines, replay fixed
   seeds under a deterministic walk and under the arms' own policies, and diff the decision type,
   the chosen action and the legal mask's hash at every step. P14 cost a few hours and settled a
   question that was otherwise worth a re-run at double compute. The method is in
   [`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md).
2. **Record the verdict in the arm's row**, not in a session transcript. The registry is where a
   later reader asks whether a pairing is matched ([`bookkeeping.md`](bookkeeping.md)).
3. **If the stream does move, the assumption is what is on trial.** The first arm re-run across a
   boundary that genuinely changes play is the experiment that would settle this, and it has not
   been run. Until then, say "asserted" where this file says asserted.
