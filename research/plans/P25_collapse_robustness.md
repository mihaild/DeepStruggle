# P25 — a training process that gives the same quality on any seed

**Status:** running. Steps 0–2 are implemented (`9c49329`), and the stress bench (step 3) is
queued behind the late-dynamics arms.
**Needs approval:** none of it touches `engine/` or the observation. The later steps (4–6) come
back for approval with the bench's result.

## Goal

The owner's framing: *"It is not sustainable to watch seeds for quality, we need process that
produces the same quality on any seed."*

Collapses are recoverable on average
([`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)): a recovered
collapse costs −1.4 Elo, and one that does not recover costs −394. That average is not a process,
though. Two things break it:
* whether a given seed recovers is decided after the fact;
* every long run so far has needed a person to watch it: E4-08-05 at 209M, E4.1-01-03, and the
  λ 0.99 arms.

This plan counts as done when a fixed recipe, launched on 6 seeds and left unattended, lands
within a narrow Elo band.

## The mechanism these steps target

What a collapse looks like in the metrics (seed 3 of E4-27 below; the findings cited above):
* one seat's self-play win share falls toward 0;
* `adv_std_raw` falls;
* entropy rises;
* the learner starts losing to its own pool.

The hypothesis is a **no-signal loop**:
1. Nearly every game the losing seat plays is lost, so its returns are nearly constant.
2. Its advantages therefore shrink.
3. Advantages are normalised by one divisor shared with the winning seat, so they shrink further.
4. With almost no policy gradient left, the entropy bonus acts almost alone.
5. The rising entropy makes the losing seat lose more, and the loop closes.

**Step 3 of this loop is contradicted by the first collapse logged per seat** (Results, below):
the two seats' spreads stay equal. The rest of the loop stands.

Recovery happens when something restores the losing seat's advantage spread. That is why
`adv_std_raw` returning is the only thing measured at onset that predicts recovery.

## Steps

| # | what | status |
|:---|:---|:---|
| 0 | **Log the per-seat signal**: pre-normalisation `adv_mean/std/n` by acting seat, and learner entropy by seat. This splits the pooled `adv_std_raw`, which averages the seat that has signal with the one that has lost it. | done (`9c49329`) |
| 1 | **`--seat-balance`**: steer toward the losing seat. The pool tracks the self-play US win share and computes pressure = min(1, \|sp_us − 0.5\| / 0.3). Under pressure it puts the learner on the losing seat with probability 0.5 + 0.4·pressure, raises the pool-game fraction toward 0.8, and draws opponents by PFSP x(1−x) on that seat's own record, which favours opponents the losing seat can still beat about half the time. That feeds the loop the non-constant returns it lacks. | done (`9c49329`), untested in training |
| 2 | **`--per-seat-adv-norm`**: normalise each seat by its own statistics, which removes step 3 of the loop directly. | done (`9c49329`), untested in training |
| 3 | **Stress bench**: λ 0.99 from scratch, which collapsed on 2 of 2 seeds (below). Each lever is run alone on seeds 3 and 5 to 60M. | queued |
| 4 | Slow π_ref as a schedule (5M after a switch point). It won late and lost from scratch ([`../log/E4_late_dynamics.md`](../log/E4_late_dynamics.md)). | after 3 |
| 5 | An auto-rewind supervisor: roll back to the last healthy snapshot under a new seed when a collapse does not recover within N steps. Branches from inside a pin escape 2 of 2 times ([`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)). | a safety net, after 3 |
| 6 | **Acceptance**: the chosen recipe on 6 seeds, unattended, to 160M, with the Elo spread across seeds reported. | after 3–5 |

This comes before [P24](P24_league.md) and before any long run. A league trained on a process
that collapses would inherit its seed lottery.

## The stress bench

`E4-27-0{3,5}` (λ 0.99, M2d, from scratch; otherwise the lineage flags) is the bench, because it
collapses quickly and on both seeds. Self-play USSR win share by 5M bucket:

| run | 15M | 20M | 30M | 40M | 60M |
|:---|---:|---:|---:|---:|---:|
| E4-27-03 | 0.56 | **0.89** | 0.92 | 0.96 | 0.97 |
| E4-27-05 | 0.50 | 0.70 | 0.84 | 0.90 | **0.97** |

The arms:

| run | lever | seeds | steps |
|:---|:---|:---|:---|
| **E4-35** | `--per-seat-adv-norm` | 3, 5 | 0 → 60M |
| **E4-36** | `--seat-balance` | 3, 5 | 0 → 60M |

**A lever passes** only if both of these hold on both seeds:
* the self-play win share of neither seat stays above 0.9 for a 5M bucket by 60M;
* it is not weaker than E4-27-0s at matched steps in a head-to-head, per seat.

A lever that stops the collapse only by holding both seats at 50% while playing worse has failed.
Under `--seat-balance` the self-play share is also partly the lever's own doing, so the per-seat
head-to-head decides, not the share.

If both levers pass, the next arm combines them. If neither passes, the no-signal hypothesis is
wrong or incomplete, and steps 4–5 carry the plan.

## Results

### The first collapse seen with the per-seat signal contradicts step 2's premise

`E4.1-01-05` is the P23 A/B arm on seed 5, launched after step 0 went in, so it logs the per-seat
signal. It collapsed with the **US** losing. Figures are 1M buckets:

| step | USSR self-play share | `adv_std_us` / `_ussr` | `adv_mean_us` / `_ussr` | `entropy_us` / `_ussr` | explained var. |
|---:|---:|---:|---:|---:|---:|
| 15M | 0.39 | 0.288 / 0.283 | −0.010 / +0.004 | 1.95 / 2.03 | 0.85 |
| 18M | 0.52 | 0.300 / 0.298 | −0.007 / +0.011 | 2.07 / 1.87 | 0.84 |
| 21M | 0.84 | 0.220 / 0.229 | +0.008 / −0.006 | 2.43 / 1.49 | 0.93 |
| 24M | 0.92 | 0.197 / 0.202 | +0.010 / −0.007 | 2.67 / 1.86 | 0.95 |
| 28M | 0.82 | 0.223 / 0.234 | +0.012 / 0.000 | 2.54 / 1.88 | 0.91 |

* **The two seats' advantage spreads stay equal.** The spread shrinks for both seats together,
  because the critic gets *better* as the outcome becomes predictable: explained variance rises
  from 0.85 to 0.95. This looks structural. Under zero-sum GAE a TD error on one seat's decision is
  mirrored on the other's, so in the same games one seat cannot have a much smaller spread than
  the other.
* **The per-seat means are ~0**, at ±0.01 against a std of ~0.2. The losing seat is not being fed
  a negative bias.
* **What does separate the seats is entropy:** the losing seat's rises from 1.9 to 2.7 while the
  winning seat's falls.

So step 3 of the loop as written above ("the shared divisor scales the loser down further") does
not happen. The shared normalisation already re-inflates both seats' shrinking spread to unit
scale.

The mechanism the data fits better is **signal made of noise**. Once the losing seat loses nearly
every game, its advantages measure the critic's residual error rather than the quality of its
moves. Normalisation scales that noise up to unit size, and the entropy bonus is the only
consistent term left in its update.

**Predictions for the bench:**
* **E4-35** (`--per-seat-adv-norm`) should behave like E4-27, since the spreads it would equalise
  are already equal. It stays in the bench as the test of that prediction.
* **E4-36** (`--seat-balance`) acts on the cause: it gives the losing seat games whose outcome is
  in doubt.
