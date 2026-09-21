# E4 — which source of randomness carries the M2d collapse?

Eight arms, `E4-10-01` … `E4-10-08`, 80M steps each, run 2026-09-20/21.
Companion to [`../findings/training/side_collapse.md`](../findings/training/side_collapse.md),
which is the census this experiment tries to explain.

## The question

The census found 4 terminal collapses in 32 distinct seeds of the M2d rung. Every one of those
seeds was a single `--seed`, and a single `--seed` drives four independent things at once:

| stream | what it decides |
|:---|:---|
| `init` | the starting weights |
| `sampling` | action sampling, and minibatch order |
| `env` | the engine's card deals and dice |
| `pool` | which opponent is drawn per game, and which side the learner takes |

So "seed 1 collapses and seed 3 does not" names no mechanism: the two differ in four ways
simultaneously. `--seed-init`, `--seed-sampling`, `--seed-env` and `--seed-pool` (commit `85d3bd8`)
each override one, so one stream can be moved at a time.

`np.random.seed` was originally counted as a fifth source and is not one. The only consumers of
numpy's *global* stream are `warmup_dataset_loader` (`np.random.choice`) and `behavioral_cloning`
(`np.random.randint`); `nash_pg` and `generic_trainer` use explicit `default_rng` / `RandomState`
instances, which `np.random.seed` does not touch. These are cold-start runs with no warmup dataset,
so it is inert here.

## Design

Seed 1 collapses and seed 3 is clean, both **deterministically** — three independent 80M runs of
seed 1 produced byte-identical final checkpoints. That matters for how the arms are read: each arm
is a *fact about its configuration*, not a sample from it. One arm settles what that configuration
does.

```
NECESSITY    base seed 1 (collapses), one stream moved to 3
  E4-10-01 init      E4-10-02 sampling      E4-10-03 env      E4-10-04 pool
SUFFICIENCY  base seed 3 (clean), one stream moved to 1
  E4-10-05 init      E4-10-06 sampling      E4-10-07 env      E4-10-08 pool
```

## Result

**All eight arms are clean.** No arm entered the pinned state at all except `E4-10-01`, which
touched it for 3 consecutive rows out of 1,221 and recovered.

| arm | direction | stream moved | verdict | terminal | longest pinned run |
|:---|:---|:---|:---|---:|---:|
| E4-10-01 | necessity | init ← 3 | CLEAN | 0.349 | 3 |
| E4-10-02 | necessity | sampling ← 3 | CLEAN | 0.248 | 1 |
| E4-10-03 | necessity | env ← 3 | CLEAN | 0.362 | 0 |
| E4-10-04 | necessity | pool ← 3 | CLEAN | 0.215 | 0 |
| E4-10-05 | sufficiency | init ← 1 | CLEAN | 0.320 | 0 |
| E4-10-06 | sufficiency | sampling ← 1 | CLEAN | 0.127 | 0 |
| E4-10-07 | sufficiency | env ← 1 | CLEAN | 0.311 | 0 |
| E4-10-08 | sufficiency | pool ← 1 | CLEAN | 0.501 | 0 |

## What this establishes, and what it does not

**No single stream is sufficient.** This is conclusive, and it is the real result. If seed 1's
initial weights carried the collapse, `E4-10-05` — those exact weights, with seed 3's sampling,
deals and opponent draw — would have collapsed. It did not, and determinism means that is settled
for that configuration rather than merely unlikely. The same holds for the other three. In
particular **initialisation is refuted in both directions**, which is worth stating because it was
the a priori favourite: "some seeds just start in a bad place" is wrong. The starting weights
carry the collapse neither in nor out.

**Every stream is necessary — but only in the weakest sense.** Removing any one of the four from
seed 1's configuration removes the collapse, which is what "necessary" means. So the collapse is a
conjunction: no proper subset of seed 1's streams reproduces it.

That conjunction reading, however, is **not distinguishable here from the null**, and this is the
limitation to carry forward. Under a null in which the collapsing configurations are simply a
12.5%-measure subset of tuple space with no coordinate-wise structure, *any* perturbation is a
fresh draw, and:

* P(all 4 necessity arms clean) = 0.875⁴ = **0.59**
* P(all 8 arms clean) = 0.875⁸ = **0.34**

All-clean is the modal outcome under the null. The necessity direction in particular cannot
discriminate at all — under the null it predicts exactly what was observed, so it contributes no
evidence. "Everything is necessary" is what a 12.5% base rate looks like from the inside.

The sufficiency direction carries what little discrimination there is. If any single stream value
strongly elevated risk — say `init=1` giving P(collapse) = 0.5 whatever the other three are — then
four clean sufficiency arms would have probability 0.5⁴ = **0.06**; at a more modest 0.30 it is
0.7⁴ = **0.24**. So the data mildly disfavour a strong single-coordinate effect and say nothing
about a moderate one.

## What would actually answer it

Measure a conditional rate directly: hold one stream at its seed-1 value, draw the other three
across many values, and compare P(collapse | stream fixed) against the 12.5% marginal. Rough sizing
from the numbers above — distinguishing 0.125 from 0.50 needs on the order of 15–20 arms per cell;
distinguishing 0.125 from 0.30 needs closer to 50. Four cells at ~30 minutes an arm puts the cheap
version at roughly 30–40 GPU-hours and the sensitive version out of reach.

Given that the collapse costs a rung's worth of Elo and is detectable live from `adv_std_raw`, the
better use of that compute is probably prevention and detection rather than attribution. Recorded
here so the attribution question is not re-opened without the sizing.

## Method notes

* The verdict is the terminal state — the mean of the last 40 logged rows — not the pinned
  fraction. An earlier detector used the fraction and scored seed 21 as a collapse when it had in
  fact entered the pinned state and climbed out, which stopped a sweep early on a false positive.
* The detector checks both tails. Its first version tested only `us_episode_frac < 0.02` and would
  have scored a US-side collapse as perfectly clean.
* Arms are named `<engine>-<attempt>-<seed>` but these eight are not seed-indexed in the last
  field, because each is a *pair* of seeds; the base and the moved stream are in each run's
  `description` and in its `metadata.json`, which now records all four resolved streams.
