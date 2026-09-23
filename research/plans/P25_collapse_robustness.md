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

*(pending)*
