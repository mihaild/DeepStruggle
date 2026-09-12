# Measurement tiers — what runs during training, per snapshot, and once per arm

Three probes were built or specified in this session, and they differ in cost by four orders of
magnitude. Putting each at the wrong tier either starves a run of signal or halves its
throughput, so the placement is derived from measured cost rather than from how interesting the
probe is.

> **Tier on the artifact as well as the compute.** A probe that costs 0.1% of a step but emits
> 880 TensorBoard series is a tier-3 probe. The question is not only what it costs to compute but
> what it costs to store and whether anyone can read the result.

## The budget

512 environments at ~15,000 env-steps/s is **29.3 vectorized steps/s, so 34 ms per vectorized
step**. Anything the training loop does per environment per step is multiplied by 512.

| primitive | per call | × 512 envs | % of budget | terminals only (≈5.7 envs/step) |
|:---|---:|---:|---:|---:|
| read a few scalars | 0.1 µs | 0.04 ms | **0.1%** | ~0% |
| `decode_flat_action` | 0.1 µs | 0.03 ms | **0.1%** | ~0% |
| `hand_of` | 6.7 µs | 3.4 ms | 10.0% | 0.11% |
| `defcon_suicide_cards` | 45.4 µs | 23.2 ms | **68.1%** | 0.76% |

`defcon_suicide_cards` walks all 84 countries through `MapData.get_country_info`, which is why it
is 450× a scalar read. It cannot be called per environment per step. It *can* be called on the
environments that terminated (0.76%), or at play-mode nodes while DEFCON is 2, which were 9.1% of
env-steps and cost **6.2%** of the budget.

> A caution, because this was got wrong once: `steps_per_sec` counts **env-steps**, not
> vectorized steps. Dividing by 512 is what turns it into a per-iteration budget. Treating it as
> vectorized steps makes everything look 512× too expensive and would have tiered all three
> probes into the final evaluation.

## Tier 1 — during training, every iteration

Costs under ~1% of the step budget, so it can be logged continuously and read as a curve.

| probe | what it costs | why it is affordable |
|:---|:---|:---|
| **DEFCON-1 ending class** — headline vs action round, own goal, bad bet, forced vs unforced trap | 0.76% | everything needed is available on the environments that *terminated*, and there are only ~5.7 of those per vectorized step. The phase, the victory-point sign and the coup flag are scalar reads; `defcon_suicide_cards` is called once per terminal, not once per decision |

**Play-mode counts per (card, side) do not belong here, despite costing 0.1%.** Compute-cheap is
not log-cheap: 110 cards x 2 sides x 4 modes is 880 series, which bloats the event file and is
unreadable as curves. It is tier 3.

The forced/unforced split needs the safe-alternative test at the *card play*, not at the
terminal. Carrying a per-environment record of the last play, as `defcon_endings.py` already
does, keeps that at terminal cost -- but the record must be written at the play-mode node, and
computing `defcon_suicide_cards` there costs the 6.2%. **Log the coarse split (phase, own goal,
bad bet) at 0.76%, and gate the forced/unforced split behind a flag** for runs that want it.

## Tier 2 — per snapshot

Snapshots are every 900 s, so a probe may cost tens of seconds without mattering. These need
their own self-play games because they ask about states the training rollout does not retain.

| probe | cost | note |
|:---|:---|:---|
| blunder rates (`measure_blunders_batched`) | ~1 s | already here |
| four-way DEFCON classification on a fixed 400-game sample | ~30 s | the tier-1 version is a curve over the live rollout; this one is a like-for-like sample across arms |
| setup probe | ~5 s | 15 batched forwards, no rollout |
| position diagnostics, decisive probe | ~10 s | already here |

**Every one of these must be run at temperature 0.1.** A probe at 1.0 reported the same
checkpoint ending 60.3% of games at DEFCON 1 against the 31.4% its training logged, inflating
self-inflicted endings fourfold. Any new probe reproduces a number the training loop already logs
before its output is used.

## Tier 3 — once per arm, at the end

Minutes, and slow-moving, so per-snapshot would be waste.

| probe | cost | why it belongs here |
|:---|:---|:---|
| **trunk recoverability** (`state_readout.py`) | ~2-3 min | 19,200 positions plus ridge solves against 84 + 110 + 6 targets. It measures a property of the representation that changes over a whole run, not between snapshots |
| **play-mode distribution per (card, side)** (`play_modes.py`) | ~1 min | the computation is trivial but the artifact is an 880-cell table. It is read once, by a person, comparing two arms -- not watched as curves |
| per-battleground detail | included above | same collection, extra solves |
| pooled head-to-head rating | ~5 min batched | needs four late snapshots, so it cannot exist until the arm is over |

For the rating specifically: a **fixed core** of the control plus one reference, with arms added
in batches. A pooled win rate against a fixed reference is a direct measurement and does not
need a common pool; only the fitted Elo column does. The all-pairs form is O(N²) and cost two
hours for a pool whose 1,431 pairings included ~1,400 nobody read.

## Summary

| probe | tier | cost |
|:---|:---|---:|
| DEFCON-1 ending class, coarse | training | 0.76% |
| DEFCON-1 forced/unforced split | training, behind a flag | 6.2% |
| blunder rates, setup, diagnostics | snapshot | seconds |
| four-way classification, fixed sample | snapshot | ~30 s |
| trunk recoverability, battleground detail | final | ~3 min |
| play-mode distribution per (card, side) | final | ~1 min |
| pooled rating | final | ~5 min |
