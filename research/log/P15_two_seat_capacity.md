# Can one network hold both seats? Yes — twice over

**Measured 2026-09-17.** Every collapse in this session is seat-specific: a US seat that dissolves
into no policy at all, a USSR seat that drops 15 points in 5M steps. None of them separates
*training dynamics went wrong* from *one network cannot represent both seats*. This settles it.

## Method

The best available US player and the best available USSR player play each other, and the acting
teacher's full distribution at each of its own decisions becomes a distillation target — so the
state distribution is the one the **combined** policy would visit, not either teacher's self-play.
`tools/generate_policy_targets.py`, 2,000 games, **805,132 targets, 110/110 cards covered, every
card at least 25 times**. Student initialised from the US teacher, 3 epochs of soft CE at 1e-4.

* US teacher — `p28_200M`, the balanced checkpoint and the only one with a sane opening.
* USSR teacher — `E3-31-28` @15M, the peak of the most recent arm, before its 25M collapse.

## Result: the student holds both

Against `frozen_280M`, temperature 0, 300 games a side (`/workspace/data/tournaments/P15_two_seat/`):

| | as USSR | as US | field Elo |
|:---|---:|---:|---:|
| USSR teacher | 83.3% | 79.7% | 1766.9 |
| **student** | **82.0%** | **57.7%** | 1652.8 |
| US teacher | 53.7% | 55.7% | 1522.3 |
| `frozen_280M` | — | — | 1500.0 |

**The student matches each teacher on that teacher's seat**: 82.0 against 83.3 as USSR, 57.7
against 55.7 as US. It gained **+28.3 pp on USSR** (53.7 → 82.0) while its US seat went 55.7 →
57.7 — **no interference at all**.

One caveat, stated because it bounds the claim: the student started *from* the US teacher, so its
US play began correct and survival is the weaker half of the evidence. The *gain* half is the
strong one, and it is large.

## The sharper finding: a single checkpoint already plays both seats

The "USSR teacher" is 83.3% as USSR **and 79.7% as US**. `E3-31-28` @15M is already a strong
two-seat model in one network — no distillation required. The student is only mediocre at US
because it was given a mediocre US teacher, not because a network cannot hold two strong seats.

## What this settles

**Capacity is not the constraint, and the representation is not the bottleneck.** Both are
answered twice: a single network absorbs two distilled specialists without interference, and a
single network already reaches ~80% on both seats unaided. Every seat collapse measured today is
therefore a **training-dynamics** problem.

It also settles the sequencing question raised earlier: the action-representation refactor
([`../findings/engine/flattening_card_play.md`](../findings/engine/flattening_card_play.md)) is a
throughput-and-clarity change worth doing on its merits, and **nothing in this evidence suggests
it would address the collapse**. The 17.5% shorter episodes and the un-aliased decline index are
real gains; neither is a fix.

## Caveats

* One student, one teacher pair, one initialisation.
* The US teacher is weak in absolute terms (55.7% against the 280M anchor), so "the student
  matched it" is a lower bar on that seat than on USSR.
* Distillation imitates a policy; it does not show that RL *from* that policy would be stable —
  which is precisely what X4a step 4 already showed it is not.
