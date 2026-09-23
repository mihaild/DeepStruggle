# P25 — a training process that gives the same quality on any seed

**Status:** running. The step-3 bench is done ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)). No lever passes on both seeds with a gain:
* seat balancing: +151 and no collapse on seed 5, level on seed 3;
* WoLF seat weights as built (surrogate only): no collapse on either seed, +131 on seed 3, −105 on seed 5, because it held entropy up.

Next, running: WoLF weights on the whole per-seat policy objective (step 3c, E4-39).
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
| 1 | **`--seat-balance`**: steer toward the losing seat. The pool tracks the self-play US win share and computes pressure = min(1, \|sp_us − 0.5\| / 0.3). Under pressure it puts the learner on the losing seat with probability 0.5 + 0.4·pressure, raises the pool-game fraction toward 0.8, and draws opponents by PFSP x(1−x) on that seat's own record, which favours opponents the losing seat can still beat about half the time. That feeds the loop the non-constant returns it lacks. | done (`9c49329`); bench: partial (E4-36) |
| 2 | **`--per-seat-adv-norm`**: normalise each seat by its own statistics, which removes step 3 of the loop directly. | done (`9c49329`); bench: fails (E4-35) |
| 3 | **Stress bench**: λ 0.99 from scratch, which collapsed on 2 of 2 seeds (below). Each lever is run alone to 60M. Step 4's slow π_ref is pulled forward into it. | done: seat balancing partial, the other two fail ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)) |
| 3b | **`--wolf-seat-weight`** (the owner's proposal; "win or learn fast", Bowling & Veloso 2002): scale each seat's PPO surrogate by w_us = 2x, w_ussr = 2(1−x) at power 1, with x the USSR's smoothed pure-self-play share. It acts on the *winning* seat, which the bench showed seat balancing leaves at full speed. Only the surrogate is weighted, so the losing seat's gradient gains on the entropy bonus and the winning seat is held closer to π_ref. Scaling a seat's gradient changes its speed, not where it stops, so it does not bias a game whose equilibrium is not 50/50. | **fails as built** ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)): no collapse on either seed (peaks 0.68, 0.71), but +62..+131 on seed 3 and −45..−105 on seed 5. Weighting the surrogate alone left the entropy bonus unweighted, which pushed the down-weighted seat toward uniform, and entropy stayed ~1.75 for all 60M. Next: weight the whole per-seat policy objective, which is WoLF proper |
| 3c | **`--wolf-scope policy`**: the same weights applied to each seat's whole policy objective (surrogate, entropy bonus, KL to π_ref), i.e. a per-seat learning rate, WoLF as published. It fixes 3b's side effect, where only the surrogate was scaled and the unweighted entropy bonus pushed the down-weighted seat toward uniform. | implemented; E4-39-03/05 on the bench |
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
| **E4-36** | `--seat-balance` | 3, 5 | 0 → 60M |
| **E4-37** | `--ref-update-freq 5000000` (slow π_ref) | 3, 5 | 0 → 60M |
| **E4-35** | `--per-seat-adv-norm` | 3 only | 0 → 60M |

The bench changed before launch, after the first per-seat collapse was read (Results):
* **E4-37 added.** It tests step 4's lever on the same bench. When it was added, no slow-π_ref arm had collapsed. *Corrected at 15:08:* E4-31-05 then collapsed at 235M, 20–40M later than the two seed-5 controls from 160M, so slow π_ref delays a collapse rather than preventing it ([`../log/E4_collapse_census_per_seat.md`](../log/E4_collapse_census_per_seat.md)). E4-37 stays on the bench to measure the delay on a fast-collapsing recipe. From scratch it cost Elo in E4-26 (−168 and −68 at 80M).
* **E4-35 cut to one seed.** It is now predicted to do nothing, and seed 3 collapses fastest.

**A lever passes** only if both of these hold on both seeds:
* the self-play win share of neither seat stays above 0.9 for a 5M bucket by 60M;
* it is not weaker than E4-27-0s at matched steps in a head-to-head, per seat.

A lever that stops the collapse only by holding both seats at 50% while playing worse has failed.

**Which share to read.** Under `--seat-balance` the logged `ussr_win_rate` is not a self-play
measure. It includes the pool games, and at full pressure the lever puts the learner on the weak
seat in ~90% of those, against older snapshots it usually beats. So the logged share is pulled
toward 50% by the lever itself: E4-36-03 logged 0.31–0.66 USSR at 26–30M, while its pure
self-play share was 0.84–0.90. For the seat-balanced arms, read the pool's own pure self-play
estimate, `opp_seat_sp_us`. The controls' logged share is diluted too, by their 30% pool games,
but symmetrically, so it *understates* their extremity. That makes the pure-self-play test on the
E4-36 arms the stricter of the two. The per-seat head-to-head decides in the end, not the share.

If more than one lever passes, the next arm combines them. If none passes, the mechanism below is
wrong or incomplete, and the auto-rewind of step 5 carries the plan.

## Results

### The first collapse logged per seat contradicts step 2's premise

In `E4.1-01-05`'s collapse the two seats' advantage spreads stayed equal (0.197 / 0.202 at the peak), and the per-seat means stayed ~0. Only the losing seat's entropy separated, rising from 1.9 to 2.7. The critic got *better* as the outcome became predictable. The reading that fits is "signal made of noise" rather than "signal scaled away". The table and the argument are in [`../log/E4_collapse_census_per_seat.md`](../log/E4_collapse_census_per_seat.md).

**Predictions for the bench:**
* E4-35 (`--per-seat-adv-norm`) behaves like E4-27.
* E4-36 (`--seat-balance`) is the live lever.
* E4-37 (slow π_ref) is expected to delay the collapse rather than prevent it.

### The census

Across the late-dynamics and E4.1 runs, all 11 collapse episodes have the **US** as the losing seat. Loosening the update late (λ 0.99, π_ref 100k, η 0.05) brings one on in 15–25M. Slow π_ref delayed one on seed 5 (235M against 195M and 215M), but did not prevent it. Same log.

### The bench, 2026-09-23

[`../log/P25_stress_bench.md`](../log/P25_stress_bench.md).

| lever | collapse half | strength half, against E4-27 at 60M |
|:---|:---|:---|
| `--seat-balance` | passes on both seeds; seed 3 is borderline (pure self-play 0.85–0.89 over 25–60M) | seed 5 **+151**, better on both seats; seed 3 +10, level |
| slow π_ref from scratch | seed 3 passes, seed 5 fails (matches its control) | +9 and +8, level |
| `--per-seat-adv-norm` | fails, collapsing at 15M against the control's 30M | +24, level |

Two lessons carry forward:
* **Balanced self-play is not health.** E4-37-03 was balanced because both of its seats were weak.
* **Changing which games are played does not slow the winning seat,** which kept sharpening in E4-36-03.
