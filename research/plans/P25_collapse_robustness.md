# P25 — a training process that gives the same quality on any seed

**Status:** running. The step-3 bench is done ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)). No lever passes on both seeds with a gain:
* seat balancing: +151 and no collapse on seed 5, level on seed 3;
* WoLF seat weights, both scopes (E4-38 surrogate only, E4-39 whole policy objective): no collapse on either seed, but up to +131 on seed 3 and down to −114 on seed 5, where both seats are weak. The scope made no difference, and the entropy explanation offered for E4-38 is withdrawn.
* **Added 2026-09-23** (merged from `worktree-research-references`): three levers that act on the loop itself — an advantage-normaliser floor (3j), a per-seat entropy target (3k) and a KL target with per-seat early stopping (3l) — a seat-gated rescue distillation (3m), and step 5 rewritten as a codified rewind-and-gating rule. Rationale in *The mechanism, revised*.
* **2026-09-24:** WoLF is closed (collapses on 3 of 4 seeds); the late collapse was the pool resume bug (3h). Owner's criterion: a collapse matters only if training dies or is delayed for a long time. 3j–3l run next on the λ 0.99 bench, 4 new seeds, 80M (*The 3j–3l bench*, below).

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

### The mechanism, revised after the per-seat reading (2026-09-23)

What the per-seat census actually shows is not a *starved* signal but a **signal made of
noise**, and the code says where it is made. `rollout_buffer.normalise_advantages`
(`ai/training/rollout_buffer.py:382`) divides every rollout's advantages by
`(std + 1e-8)` — a per-batch normaliser with no floor. When the losing seat's games all end
the same way, both seats' raw spread falls together (0.30 → 0.20; the critic gets *better*,
explained variance 0.85 → 0.95), and what remains is the critic's residual error. The
normaliser rescales that residual to unit variance and PPO consumes it at full weight, while
the fixed entropy bonus (`ent_coef` 0.01) is the only *coherent* pressure left on the seat —
which is why the losing seat's entropy is the one series that separates (1.9 → 2.7), and why
`--per-seat-adv-norm` collapsed **earlier** than its control: it normalises harder.

That reading changes which levers are mechanism-matched. Anything that gives the losing seat
winnable games (step 1) or slows the winning seat (step 3b) works around the loop; steps 3j–3l
below act on the loop itself, at the two places it closes — the normaliser and the entropy
bonus — plus the update size that the late-dynamics arms showed brings a collapse on. They
are the standard remedies for the same failure in adversarial training: the GAN
vanishing-gradient literature's non-saturating losses and two-time-scale updates, and the
PPO practice of a KL target. None has been run here.

## Steps

| # | what | status |
|:---|:---|:---|
| 0 | **Log the per-seat signal**: pre-normalisation `adv_mean/std/n` by acting seat, and learner entropy by seat. This splits the pooled `adv_std_raw`, which averages the seat that has signal with the one that has lost it. | done (`9c49329`) |
| 1 | **`--seat-balance`**: steer toward the losing seat. The pool tracks the self-play US win share and computes pressure = min(1, \|sp_us − 0.5\| / 0.3). Under pressure it puts the learner on the losing seat with probability 0.5 + 0.4·pressure, raises the pool-game fraction toward 0.8, and draws opponents by PFSP x(1−x) on that seat's own record, which favours opponents the losing seat can still beat about half the time. That feeds the loop the non-constant returns it lacks. | done (`9c49329`); bench: partial (E4-36) |
| 2 | **`--per-seat-adv-norm`**: normalise each seat by its own statistics, which removes step 3 of the loop directly. | done (`9c49329`); bench: fails (E4-35) |
| 3 | **Stress bench**: λ 0.99 from scratch, which collapsed on 2 of 2 seeds (below). Each lever is run alone to 60M. Step 4's slow π_ref is pulled forward into it. | done: seat balancing partial, the other two fail ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)) |
| 3b | **`--wolf-seat-weight`** (the owner's proposal; "win or learn fast", Bowling & Veloso 2002): scale each seat's PPO surrogate by w_us = 2x, w_ussr = 2(1−x) at power 1, with x the USSR's smoothed pure-self-play share. It acts on the *winning* seat, which the bench showed seat balancing leaves at full speed. Only the surrogate is weighted, so the losing seat's gradient gains on the entropy bonus and the winning seat is held closer to π_ref. Scaling a seat's gradient changes its speed, not where it stops, so it does not bias a game whose equilibrium is not 50/50. | **fails as built** ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)): no collapse on either seed (peaks 0.68, 0.71), but +62..+131 on seed 3 and −45..−105 on seed 5. Weighting the surrogate alone left the entropy bonus unweighted, which pushed the down-weighted seat toward uniform, and entropy stayed ~1.75 for all 60M. Next: weight the whole per-seat policy objective, which is WoLF proper |
| 3c | **`--wolf-scope policy`**: the same weights applied to each seat's whole policy objective (surrogate, entropy bonus, KL to π_ref), i.e. a per-seat learning rate, WoLF as published. It fixes 3b's side effect, where only the surrogate was scaled and the unweighted entropy bonus pushed the down-weighted seat toward uniform. | **fails** ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)): no collapse on either seed, but no better than 3b on strength: +27..+91 on seed 3, −75..−114 on seed 5. Entropy stays ~1.75 as in 3b, so 3b's entropy explanation is withdrawn |
| 3d | **Gentler WoLF.** E4-38/39 kept the self-play split near even but traded the lead back and forth: the USSR weight swung 0.28–1.65, entropy never fell, and seed 5 lost on both seats. Two softer forms, each with the whole policy objective weighted: `--wolf-power 0.5` (E4-40; 3:1 at a 90/10 split instead of 9:1), and `--wolf-dead-zone 0.15` (E4-41; no weighting inside 35–65% USSR). | running |
| 3e | **WoLF on the normal recipe (E4-42).** λ 0.98, which does not collapse early on seeds 3 or 5: does WoLF cost strength there? Against E4-08-0{3,5} at 60M and 80M, per seat. | queued behind 3d |
| 3f | **Dead zone with a jump (E4-43)** and **the late-collapse test (E4-44/45).** E4-41's dead zone (0.15, shift) was the first WoLF variant to sharpen and beat its control on both seeds, but seed 5 still drifted to 0.96 against the softer brake; `--wolf-dead-zone-mode jump` applies the full rule outside the zone. Then both modes are run from `E4-08-05@160M`, where every continuation so far collapsed, to 240M. | queued |
| 3g | **Breaking the brake/sharpen trade-off.** The shift dead zone (E4-41) sharpened and beat its control on both seeds, but seed 5 collapsed at 50–60M. The jump (E4-43) held the split (peaks 0.71/0.73) but stopped sharpening (probe entropy 1.38/1.64). Two middles: shift + `--seat-balance` as a game-mix backstop (E4-46), and a narrower shift zone of 0.10 (E4-47). | done, neither passes ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)): E4-46 holds the split but sharpens one seat only and is level with the control; E4-47 holds it (seed 3 at 0.86 and rising) and sits between E4-43 and E4-41 in strength. The dead zone is one dial with no setting that passes both halves on both seeds. |
| 3h | **The pool resume bug and the late collapse (E4-48).** A resume recorded the restored pool at step 0, so spacing eviction drained it to 2 pre-resume members and the next resume dropped them (fixed in `3803d5d`). Every continuation from `E4-08-05@160M` ran on that drained pool, and every one collapsed. E4-48 reruns the plain continuation from a repaired state, seed 5 and a seed-11 branch. The bench (from scratch, never resumed) is not affected. | done: the fixed pool passes on 3 of 3 where the drained pool failed on 5 of 5, and ends 170–390 Elo above its drained twins. On seed 3 (no collapse) it is +55..+124 over its twin and level with the seed-11 twin at 240M ([`../log/P25_pool_resume_bug.md`](../log/P25_pool_resume_bug.md)) |
| 3i | **New seeds (6, 7)** for the bench control and E4-41: is the bench collapse, and E4-41's gain, a property of seeds 3 and 5? | done: E4-41 collapses on 3 of 4 seeds (5, 6, 7) and the control on 2 of 4 (3, 5); E4-41 is behind its control on both new seeds. WoLF is not a robustness fix, and a two-seed bench cannot rank levers ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md)) |
| 3j | **`--adv-norm-floor`**: floor the advantage normaliser. Divide by `max(batch_std, c · EMA_std)` with the EMA taken over the run (c ≈ 0.5 as the first cell; a fixed minimum is the fallback cell), so a vanished signal stays *small* instead of being rescaled to unit noise. The opposite direction from step 2, which is why that one collapsed earlier. Post-normalisation `adv_std` is then allowed to fall below 1, and `adv_std_raw` keeps its meaning. Alternative form, same intent: normalise returns (PopArt) and leave advantages raw — second cell only if the floor is null. | **E4-50 (c 0.8), E4-51 (c 0.5)**, queued |
| 3k | **`--entropy-target`**: replace the fixed bonus by an auto-tuned coefficient toward a per-seat target entropy (SAC-style dual variable; bonus becomes a penalty above target). Target from the run's own healthy window (~1.9–2.1 nats on the bench). Breaks steps 4–5 of the loop directly — the losing seat cannot inflate to 2.7 — and is the same lever the *other* failure mode, entropy inflation ([`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md)), calls for. A cheaper variant if the dual is unstable: scale `ent_coef` by the batch's pre-normalisation advantage magnitude, so the bonus cannot dominate a dead signal. | **E4-52**, as a one-sided ceiling (below), queued |
| 3l | **`--target-kl`** with per-seat early stopping: stop a seat's PPO epochs when its approximate KL to the rollout policy exceeds the target (adaptive β is the alternative). `nash_pg.py` has `clip_eps` and η but no KL target; every late collapse in the census followed a looser update (λ 0.99, π_ref 100k, η 0.05), and the late arms showed damping helps. This makes tightness respond to the seat that is moving too fast instead of being a global schedule. Complements WoLF (3b): WoLF scales the surrogate, this bounds the step. | **E4-53**, queued behind the controls |
| 3m | **Rescue distillation**: search CE targets *only* for the losing seat's decisions and *only* while that seat's frozen-anchor win rate is below a threshold. Search distillation is the one intervention with zero episodes in 2 of 2 arms across the collapse window ([`../log/E4_search_distillation.md`](../log/E4_search_distillation.md)) at ~40× per-step cost; gating it by seat and by alarm bounds the cost to the episode. Run only if 3j–3l leave a seed stalling. | after 3j–3l |
| 4 | Slow π_ref as a schedule (5M after a switch point). It won late and lost from scratch ([`../log/E4_late_dynamics.md`](../log/E4_late_dynamics.md)). | after 3 |
| 5 | **Codified rewind + pool gating** — a *rule*, not a safety net. (a) Trigger on the arbiter, not on the share: the losing seat's win rate against the frozen anchor pair below θ for N consecutive snapshots **and** `adv_std_raw` not recovered to the run's healthy band (the census shows the λ 0.99 collapses sit at 0.12, invisible to a fixed `adv_std_raw` threshold alone). (b) Action: resume from the last snapshot that passed, with `--seed-sampling` advanced by a fixed rule; branches from inside a pin escaped 2 of 2 ([`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)). (c) **Pool gating**: a snapshot is admitted to the opponent pool only if it passes the same per-seat check, so the pool never fills with copies of the collapsed self (AlphaZero-style gating). Because trigger, action and cadence are fixed in advance and driven by the seed, the run stays reproducible and no data is discarded by hand — the objection to "watching seeds" is answered by making the selection part of the algorithm (PBT's exploit step, OpenAI Five's rollback), not by removing it. | implement after 3; on by default for step 6 |
| 6 | **Acceptance**: the chosen recipe on 6 seeds, unattended, to 160M, with the Elo spread across seeds reported. | after 3–5 |

**The 3j–3l bench (decided 2026-09-24).** λ 0.99 from scratch (E4-27's flags), 80M, seeds 10–13,
at most two runs at a time; every λ 0.99 run before it is archived to
`data/checkpoints/_archive/lambda_0.99/` and retrained, control included.

| arm | lever |
|:---|:---|
| E4-49-10..13 | control |
| E4-50-10..13 | `--adv-norm-floor 0.8`: the healthy→collapse ratio of the raw spread on the bench is ~0.75–0.8 (0.32–0.37 → 0.25–0.30), so 0.8 binds at onset |
| E4-51-10..13 | `--adv-norm-floor 0.5`: the plan's first cell; binds only inside a deep pin (0.15–0.17) |
| E4-52-10..13 | `--entropy-ceiling 1.80`, **one-sided**: the per-seat coefficient never exceeds `ent_coef`, so below the ceiling it is the control; above it the bonus shrinks and turns into a penalty (floor −0.02). A two-sided target would also push a sharpening seat back up, and sharpening is where the WoLF runs' strength came from |
| E4-53-10..13 | `--target-kl k`, per seat; k = the 90th percentile of per-seat, per-iteration approximate KL (rollout → current policy) in the four controls over 5–40M, fixed before E4-53 launches |

**Pass rule (owner's criterion).** A shallow excursion that recovers is not a failure. Per seed:
the longest pin (consecutive 5M buckets with one seat's self-play share ≥ 0.95) and total pinned
steps; strength per seat at 40, 60 and 80M in a per-seed field with frozen anchors (E4-08-03@80M,
E4-08-05@80M, HeuristicBot), plus one cross-seed field at 80M. A lever passes if its mean 80M Elo
over the four seeds is not below the control's and no seed stalls (pinned > 20M steps, or 80M Elo
below its 40M) where the control's same seed did not.

**Predictions, written before the bench.** 3j: the losing seat's per-seat entropy stops rising
and its post-normalisation `adv_std` reads well below 1 during the episode; collapse depth
shallower, recovery earlier, no strength cost on the winning seat. 3k: entropy pinned near
target on both seats; if the collapse still enters, the loop is not entropy-driven and 3k is
kept only for the inflation mode. 3l: fewer episodes on the λ 0.99 bench specifically, since
that bench is a loosened update. If 3j and 3k both pass, the P25 hypothesis as revised above is
confirmed; if neither moves anything, the noise-amplification reading is wrong and 3m/step 5
carry the plan.

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
