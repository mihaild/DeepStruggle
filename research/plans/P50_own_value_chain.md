# P50 — each side's own value chain for the main, and the lambda clock (owner, 2026-11-20)

**Proposed: one branch of the base, per-player GAE at matched reach, against the base continued.** No build.
Not proposed: lambda per card play as an experiment, the own-view chain with nodes at the opponent's decisions,
a true-state critic. Measurements: [`../log/P50_lambda_clock_and_own_view.md`](../log/P50_lambda_clock_and_own_view.md).

## The two questions

1. *Should lambda be fixed per action round, not per micro-step?*
2. *Does it make sense to train or tune the main with each side's own value chain?*

## 1. The lambda clock: a unit, not an experiment

* Measured, a lambda counted per card play gives the same estimator: at equal return error the advantage spread
  is 0.2408 (decay at the play's boundary) or 0.2378 (spread over the play's decisions) against the default's
  0.2371, and the advantages correlate 0.994-0.995.
* The unevenness it would remove is modest: a card play is 2 to 6 decisions for the middle 80% of plays, so the
  weight halves after 6 to 11 card plays.
* **So it is not a strength experiment.** Its use is as the unit lambda is stated in: 0.921 a card play is
  today's reach whatever the number of nodes -- the interleaved stream (0.98 a decision), a side's own decisions
  (0.96), a chain with nodes after a decision and around the dice, a merged action space. The per-player runs
  of E3 and of 2026-11-20 were set at the wrong reach for want of it.
* If it is built, the decay goes at the play's boundary: that form needs nothing from the future, where
  `Lambda^(1/n)` needs the play's length, unknown for a play the buffer's end cuts. Not built for this plan; the
  arm below sets its lambda by hand.

## 2. Each side's own chain: what the case is

**For.**

* The default negates the next value when the mover changes. That value is the other side's view, which cannot
  see the mover's hand; 45% of the default's bootstrap weight sits there.
* A side's value depends on its own hand by 0.075 (SD over redeals of what its opponent cannot see), about 4
  points of win probability, and that is most of what the two views disagree about.
* So the value of cards still in the hand reaches a decision only through the side's own later decisions and
  through whatever has become public: a choice about *which card now and which later* is under-credited, by
  about half of the part not yet on the board within lambda's reach. My arithmetic puts that at 5-25% for value
  realised later in the same turn; it is an estimate, not a measurement.
* The decisions this touches are of a kind the site games show as errors: the turn's Space Race attempt spent
  before the card that needs it, the China Card's timing
  (the analysis session's record, branch `worktree-belief-site-probe`).
* Per-player GAE is the same chain at no cost: it bootstraps a side's decision from its next own decision, so
  the value never changes hands. At matched reach its advantage spread and return error are the default's to
  the third digit. It is built and tested (`--per-player-gae`).
* In games against a fixed opponent (P49c's league rows) it also keeps every value on the learner's own rows.

**Against.**

* At matched reach the per-player advantages correlate 0.986 with the default's, and the 0.040 they differ by
  is the same on card choices as on every other decision. Nothing measured places the difference in decisions
  about the hand.
* The opponent's hand adds 0.002 AUC to the critic ([`../log/P30_oracle_critic_auc.md`](../log/P30_oracle_critic_auc.md)):
  hidden cards move the result little.
* The one run of per-player GAE lost 520 Elo at 80M (E3). It ran at the default's reach at lambda 0.99, and the
  default at 0.99 lost 242 and 286 Elo from scratch (E4), so that run no longer counts against it; nothing
  counts for it either.

**Verdict.** It makes sense as one cheap branch and not as a build. The expected gain is small. The nodes at
the opponent's decisions (the own-view chain) add a forward pass a decision and measure the same as per-player
on every count, so they are left out.

## The run

| | start | change | steps |
|:---|:---|:---|---:|
| **arm** | the base, `E8.1-A8-R132-S45@11600M+R139@12400M+R155`, at its final state (13,200M) | `--per-player-gae --gae-lambda 0.96`; every other flag the base's, read with `tools/scripts/launch_flags.py` | 800M |
| **control** | the same state | none: the base continued under its own name | 800M |

* 0.96 is 0.98 squared: a side moves at half the decisions, and the sweep puts the match at 0.9603.
* The value loss stays on every row (in self-play every row is some side's own decision).
* The trainer refuses per-player GAE with held rows and with the blunder windows; the base is pure win/loss, so
  neither should apply. Checked at launch.
* Two slots, about 4.5 hours, after the both-seat diagnostics. The launch is the owner's.

**Reads.** Judged at +800M; at +400M only the standard evaluations.

1. **Arm against control at the same step,** SWAs of the last 80M, greedy, 2,000 deals a side
   (`tools/tournament.py`), per seat from the JSON.
2. The references (the old reference, J1's best), as context: their noise between runs is 4-5 points.
3. The critic against results (AUC and Brier from the training metrics), which compare across the two.
4. On the CPU, the redeal probe on both: the SD of a side's value over its own redealt hands (0.075 now). It
   says whether the critic leans on the hand more, not whether play is better.
5. The site-game positions that are about the hand, the probability of the move the owner marked right, arm
   beside control. A handful of positions: context.

**Stop early** only on a clear problem inside 100M: entropy rising, explained variance falling, one seat running
away (E3's gap was plain at 20M).

**Pass.** Read 1 at 52.5% or over for the arm (three standard errors), neither seat significantly under the
control's own self-play rate there, and read 2 not under the control's by more than its noise. Then P49c's main
takes the form for all its games. Otherwise the default stays; nothing else depends on this.

**Prediction.** Read 1 between 49% and 53%; reads 2 and 3 level. I would not be surprised by a null.

**What would overturn the reasoning.** A redeal of the hand is uniform over the unseen cards and real hands late
in a turn are not, so 0.075 may overstate what a hand is worth; the 5-25% is arithmetic on an idealised game;
and if the critic already reads through the other side's play what that side holds, the other view is less
blind than assumed.
