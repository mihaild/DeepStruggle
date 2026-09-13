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

### How the live classification is computed

On each vectorized step, for every environment that reported an ending this step:

1. `phase_at[i]` -- the phase recorded *before* the step that ended the game. If it is
   `Phase::HEADLINE` the ending is counted as **headline** and classification stops. The two
   headline cards are simultaneous and neither player saw the other's, so a scheme built for
   sequential decisions does not apply.
2. the **loser** is read from the sign of `victory_points`, not from `phasing_player`.
   `phasing_player` is right at *resolution* time -- during a headline it is whoever's card is
   resolving -- but the last thing the loop observes is the card *selection*, before either
   resolves, and there it is still the turn's player.
3. if the engine's `DEFCON_SUICIDE_PROVOKED` flag is clear, the ending is the loser's own:
   **own goal** if the action was a coup in a battleground at DEFCON 2, or its own mandatory
   degrader or Olympic Games played for the Event; **bad bet** for Summit, Missile Envy or
   Five Year Plan.
4. if the flag is set, the loser played an opponent card whose event reached DEFCON 1:
   **unforced trap** if a safe alternative was in hand at that play, **forced trap** if not.

The safe-alternative test is `defcon_suicide_cards` evaluated at the play-mode node and carried
forward per environment, which is the 6.2%. **On by default.** It is the category that accounts
for 19-24% of all games; it should stay on until that number falls far enough that the split
stops being informative, at which point retire it rather than pay for it.

### Live and per-snapshot are different measurements and carry different names

The live classification is over the training rollout: on-policy, whatever states the policy
reached this iteration, a denominator that shifts as the policy does. The snapshot version plays
a **fixed 400-game sample at temperature 0.1** from a frozen checkpoint, so it is comparable
across arms and across time.

They will disagree, and a headline result was nearly lost this session to two versions of one
metric being read against each other. So they never share a name:

| | prefix | denominator | comparable across arms |
|:---|:---|:---|:---|
| live, in-loop | `live/defcon1_*` | episodes completed this iteration | **no** |
| snapshot probe | `probe/defcon1_*` | a fixed 400-game sample at T=0.1 | yes |

A number from one is never quoted against the other.

### Retired

`ending_frac_defcon1_self` and `ending_frac_defcon1_provoked` are **replaced**, not kept
alongside. `self` merged an own goal that loses for nothing with Summit, which is a deliberate
gamble; `provoked` said nothing about whether the loser had an alternative, which is the whole
question. Carrying both doubles the series and invites cross-definition comparison. The date of
the switch is recorded here so a step change in an old chart can be read as the definition
moving.

### The anchor is a sanity check, not a ranking

`eval_win_rate_HeuristicBot` has inverted three arm orderings in one session, always overrating
the weaker arm -- a fixed script can be exploited without being beaten in general. It is kept and
renamed to `sanity/beats_heuristic`: **every model should win, and a value that stops being high
means something broke.** It is not read as a comparison between arms.

For comparison, the anchor is an **E3-01 checkpoint** at the budget being compared -- same engine,
same recipe, no cross-engine asterisk. `E3-01-21-80M`, `-160M` and `-240M` all exist.

**Play-mode counts per (card, side) do not belong here, despite costing 0.1%.** Compute-cheap is
not log-cheap: 110 cards x 2 sides x 4 modes is 880 series, which bloats the event file and is
unreadable as curves. It is tier 3.

### Critic discrimination (`critic/auc`, `critic/brier_skill`) — tier 1

**Why it is tier 1.** The per-step cost is an integer compare and an occasional append; the
aggregate is one sort over at most 20,000 held samples, once per iteration. Well under the 1%
bar, and it has to be a curve rather than a snapshot reading because what it detects is a
*transition* — the point at which the critic stops discriminating.

**What it catches.** Once one side finds a strategy the other has not answered, outcomes become
predictable, the critic has no reason to discriminate, and it degenerates toward the base rate.
Advantages then vanish and both policies freeze. This happened in E3-17-22 and, independently and
via a different winning strategy, in E3-15-22.

**Why not `explained_variance`, which we already log.** The GAE return is built as `G = A + V`, so
`G - V = A` exactly and `EV = 1 - Var(A)/Var(G)`. A critic whose advantages collapse scores near
1.0 *by construction*: EV rose to 0.996 across precisely the window in which the critic became
useless. It is a restatement of the failure, not a detector of it.

**Why AUC rather than accuracy or correlation.** The collapse is accompanied by a base-rate shift
— the US win rate fell from 54% to 13% over the same window. Accuracy is dominated by the base
rate, and point-biserial correlation against a binary outcome is itself bounded by how balanced
that outcome is, so both move for reasons unrelated to the critic. Measured on E3-17-22 the two
families disagree in *sign*:

| checkpoint | base rate | accuracy | **AUC** | Brier skill |
|:---|---:|---:|---:|---:|
| 40M | 0.522 | — | **0.870** | 0.404 |
| 80M | 0.528 | 79.2% | **0.865** | 0.387 |
| 160M | 0.905 | **87.5%** | **0.728** | 0.156 |

Accuracy *rises* from 79.2% to 87.5% while the critic is degrading. AUC falls, correctly, because
it is invariant to the base rate and to any monotone rescaling of the value.

The pair is worth keeping rather than AUC alone: AUC measures only *ranking*, and GAE subtracts
`V` rather than ranking it. At 160M the critic still ranks above chance (0.728) while its
calibrated usefulness has fallen 2.6x — so it is not that it knows nothing, it is that it no
longer converts what it knows into values that separate actions. **`brier_skill <= 0` is the
alarm**: at or below zero the critic is no better than a constant at the base rate.

**Where it is sampled.** One state per game per turn, at the turn's first decision, from the fixed
US perspective (`v_win * acting_player`), held until the game ends and the winner is known.
Averaging over every step would mix turn-1 states with nearly-decided turn-9 states in a
proportion that shifts as game length changes during training. `critic/auc_turn3` reports a single
fixed turn for a number that is comparable across runs whose games differ in length.

Read it smoothed. A single iteration resolves only the games that happened to finish, so the
per-iteration value is noisy; the trend is what matters.

## Tier 2 — per snapshot

Snapshots are every 900 s, so a probe may cost tens of seconds without mattering. These need
their own self-play games because they ask about states the training rollout does not retain.

| probe | cost | note |
|:---|:---|:---|
| blunder rates (`measure_blunders_batched`) | ~1 s | already here |
| **trunk read-out, cut down** (`state_readout.py`) | ~10 s | tracks plus a 20-country subset on ~2,000 positions. The full version is tier 3; this one exists so a representation collapse shows up mid-run rather than in the post-mortem |
| four-way DEFCON classification on a fixed 400-game sample | ~30 s | the tier-1 version is a curve over the live rollout; this one is a like-for-like sample across arms |
| setup probe | ~5 s | 15 batched forwards, no rollout |
| position diagnostics (`position_diagnostics.py`) | ~10 s | empty and uncontrolled battlegrounds at turns 5 and 8, ply distribution. Measures the board self-play produces rather than the result |
| decisive probe (`decisive_probe.py`) | ~10 s | forced-win take rate and avoidable-forced-loss rate. Decisive choices are ~0.6% of decisions, so win rate barely registers them |

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
| DEFCON-1 forced/unforced split | training, on by default | 6.2% |
| blunder rates, setup, position diagnostics, decisive probe | snapshot | seconds |
| trunk read-out, cut down | snapshot | ~10 s |
| four-way classification, fixed sample | snapshot | ~30 s |
| trunk recoverability, battleground detail | final | ~3 min |
| play-mode distribution per (card, side) | final | ~1 min |
| pooled rating | final | ~5 min |
