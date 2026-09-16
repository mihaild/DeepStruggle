# Seed and stopping-point variance — how many seeds an arm needs

The error bars every comparison in this project has to clear, and what they have already
invalidated. **Update discipline: rewritten in place.** The experiments are in
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and
[`../log/P9_architecture.md`](../log/P9_architecture.md), *seed variance is ~95 Elo*; how to run
against these bars is [`../method/running_experiments.md`](../method/running_experiments.md).

## The numbers

| source of variation | size | measured where |
|:---|---:|:---|
| **between seeds, same configuration, 80M** | **~95 Elo** | E3-15 seeds 21/22 in one field; 55.1 for E3-17 |
| within a run, across its last four snapshots | ~30.7 Elo mean | five `var_*` arms at 80M |
| between runs, reading the final snapshot only | 56.6–58.5 Elo SD | same five arms |
| between runs, reading the mean of the last four | 20.1–33.0 Elo SD | same five arms |
| a continuation leg's own sampling seed | ~15 Elo | continuation pairs |
| a leg's *gain* between budgets | ±16 Elo | same |
| the same tournament command, re-run | ~1.5 points of win rate | three identical 6,000-game runs |
| Elo between two tournaments over the same files | ~±15 | — |
| binomial SE at 3,200 games | 0.88 pp | — |

The last two lines are the point. **The statistical error is an order of magnitude below the
seed effect**, so an underpowered comparison is not fixed by playing more games. At 3,200 games a
pairing the binomial error is 0.88 pp while the spread across seed pairings is 20.4 pp.

## What this rules out, and what it costs

* **A single-cell comparison cannot resolve below ~50 Elo**, which is most effects worth arguing
  about.
* **Rate four late snapshots and pool the sixteen pairings.** Free, and it cuts the between-run
  SD by up to 2.9×. It controls *within*-run wobble only and says nothing about between-seed
  variation — the two are separate problems and the four-snapshot method solves the smaller one.
* **Two or three seeds per arm** suffices for effects above ~40 Elo. Below that the arm is not
  worth running at this budget.
* Seed spread also appears in *game shape*, not only in rating: it varies between seeds by as
  much as it varies between arms.

## What it has already invalidated

**"0 graph layers beats 2" is withdrawn.** E3-16/E3-17 measured graph depth at seed 21 and
concluded that removing the map graph was positive (2 layers −59 Elo, 1 layer −84, anchored on
E3-17). Rating both seeds of E3-15 and E3-17 in one field reverses the sign:

| | win rate for E3-15 | n |
|:---|---:|---:|
| E3-15-21 vs E3-17-21 | **35.0%** | 800 |
| E3-15-21 vs E3-17-22 | 44.6% | 800 |
| E3-15-22 vs E3-17-21 | 50.7% | 800 |
| E3-15-22 vs E3-17-22 | **55.4%** | 800 |
| pooled | 46.4% | 3,200 |

Matched on seed 21 E3-17 is much better; matched on seed 22 E3-15 is. Arm means differ by
+18.9 Elo against a between-seed SE of 54.9 — 0.34 standard errors. **Graph depth is an open
question**, and E3-17 (`--graph-layers 0`) is nonetheless the recipe underneath every arm from
E3-18 to E3-21, including the whole pooling programme
([`pooling.md`](pooling.md)). The throughput advantage of 0 layers is unaffected; it is not a
play-strength claim.

**The pooling Elo result does not clear this bar either.** Four pooled arms at 160M span 221 Elo
among themselves against a 98.7 Elo difference from the unpooled condition. The balance result
does clear it, by separating completely. See [`pooling.md`](pooling.md) §3.

**A single arm of an oscillating quantity settles nothing.** Two no-pool arms differing only in
seed sat 0.1183 apart on the side-balance endpoint while the pooled-vs-no-pool difference was
0.0169 — seven times smaller. Three earlier pooling claims were retracted on that basis.

## What it does not touch

Within-run comparisons — an arm against its own earlier self on `adv_std_raw`, critic AUC, or
per-side win rate — carry no seed-variance problem, because the seed is held. The
advantage-collapse findings are of that kind and reproduce across two architectures, two winning
strategies and two seed families.

## The cause is probably removable, and has not been tried

The learning rate is constant for the entire run — there is no decay schedule — which is exactly
what leaves a policy wandering at the end rather than settling. Linear or cosine decay, or
evaluating a weight EMA instead of the live policy, would remove the ~30 Elo oscillation rather
than averaging over it. Neither has been tried, and neither addresses the between-seed spread,
which is the larger term.
