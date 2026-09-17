# Before you trust a number — the checklist

The curated short list of ways an instrument in this repo has lied, and the cheapest check for
each. **Update discipline: living reference, rewritten in place.** Entries are added when a new
class of fault is found and rewritten when the check improves; an entry is removed only when the
fault becomes structurally impossible, never because it is old. Keep it under ~200 lines — it is
the file people actually read. The full history of each, in the detail it was diagnosed with, is
[`../log/measurement_bugs.md`](../log/measurement_bugs.md); the variance entries are in
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and the probe-methodology ones in
[`../log/P9_architecture.md`](../log/P9_architecture.md).

> **Distrust a measurement before trusting a result.** Seven separate diagnostics in this repo
> reported confident numbers that were wrong. Every one of them looked plausible. When an
> experiment produces a surprising result, the cheapest first hypothesis is that the instrument
> is broken -- check that before building on the finding.

---

## The input the model was given

**A wrong-width observation is misread, not rejected.** A model reads fixed slices, so handing it
the wrong layout returns a number instead of raising. *Shows up as:* a metric that contradicts
the run's own rollouts — a mean final turn of 1.37 against 6.7, a 2.0% anchor rate for a
checkpoint that scores 81.3% on its own layout. *Check:* the layout comes from the model
(`layout_for_model`, `check_obs_width`, `NeuralAgent._assert_width`), never from a default.

**A probe that builds its own environment inherits the default layout.** The same bug's fourth
instance, in `profile_self_play_batched` and `measure_decisive_batched`. *Shows up as:*
`empty_battlegrounds_turn8` of 0.0 — an average over an empty set, because nothing reached turn
8. *Check:* `tests/training/test_probe_observation_layout.py` pins the layout each probe
constructs its env with; a second, independent walk of live games should agree with the probe.

**A chance node handed to a policy.** At `ROLL_DIE`, `ctx().decision_player` is `NONE`, and a
fallback to `phasing_player` has a network choosing its own dice. *Shows up as:* the sequential
and vectorized evaluation paths disagreeing by >25 points on the same deterministic matchup.
*Check:* drain chance nodes explicitly; the two paths must be bit-identical on identical seeds.

**A checkpoint that cannot load into the tournament path.** A categorical value head failed
`NeuralAgent.from_checkpoint`, so every number for it came from the training loop's own
evaluation, which builds the model itself. *Check:* if an arm has never appeared in a tournament
report, ask why before quoting its rate.

## The sample you actually drew

**Survivorship bias in a batched probe.** Running N envs and stopping at `num_episodes` <
`num_envs` samples the *fastest* envs, and auto-reset lets a quick env be counted repeatedly.
*Shows up as:* every late-game metric pinned at zero, a mean final turn half the true one.
*Check:* measure the **first episode of every env** and drain until all finish; assert
`num_games == num_envs`.

**A corpus filtered by how the game ended.** ts-replayer's 119 finished logs are the subset whose
*recording* completed, which is not independent of the ending. *Shows up as:* 1.7% DEFCON-1
endings against ITS's 11.7% at n=44,136. *Check:* use ITS for length and ending mix, ts-replayer
only where moves are needed ([`human_play.md`](human_play.md)).

**`harvest()` in an analysis script calls `retire_stale()`.** *Shows up as:* a 1,000-position
sample silently becoming 91. *Check:* take positions out of the pool each round.

**A held-out split that permutes positions.** Successive samples from one env are the same game a
few steps apart. *Check:* hold out whole environments or whole games. (Worth doing; here it was
worth about one point.)

**A sliding-window monitor is not a trend.** The per-iteration monitor means over the last 40
logged iterations, which overlap heavily. *Shows up as:* a "monotone late-run climb" to 66% that
is a 48.5–58.0% oscillation when binned by 20M steps. *Check:* bin before believing a trend.

## The comparison

**A single snapshot cannot resolve an effect smaller than the run's oscillation.** Within-run SD
across four late snapshots is 5–33 Elo and mean oscillation ~30. *Shows up as:* an effect that
changes sign when re-measured — `staged_cards` read +52.8 Elo at 80M from one snapshot and +29.7
pooled, and +4.7 became −11.4. *Check:* rate the last four snapshots and report the mean; the
error bar on a single cell is the oscillation, not the binomial SE.

**An endpoint read from self-play alone can be satisfied without getting stronger.** Self-play
side balance — *mean |ussr_win_rate − 0.5|* — was pre-registered as the primary endpoint of the
4 × 4 pooling experiment and then abandoned, because it turned out to be uncorrelated with strength
against a fixed opponent. *Why:* an arm can drive its own USSR win rate to 0.5 by having both of
its sides drift together; that is a fact about the pair, not about either side. Balancing against
yourself and beating a third party are different quantities. *Shows up as:* an arm that looks
converged on the pre-registered metric and unremarkable on the ladder. *Check:* measure the thing
against an opponent that does not move. If a metric can be satisfied by both sides changing
together, it is measuring the pair.
See [`../findings/training/pooling.md`](../findings/training/pooling.md) §3.

**A single cell is not a comparison.** Two arms' final snapshots meeting once gave a ~64 Elo seed
effect where sixteen pairings give ~13. *Check:* pool all pairings of four snapshots a side.

**A tournament is not reproducible from its configuration.** Deals are seeded; the agents sample.
*Shows up as:* two identical 6,000-game runs differing by 1.6 points on the headline matchup.
*Check:* treat ~1.5–2 points at 1,000 games a pair as the floor, or replicate before believing a
small gap.

**Elo is not comparable across tournaments, and not additive within one.** The identical
comparison read 60.50% in one pool and 62.25% in another. *Check:* re-anchor and measure the
specific pair directly; quote the anchored column only for the overall picture.

**A leg's gain carries ±16 Elo.** The same 80M leg from the same resume state was worth +86.9 on
one seed and +54.7 on another. *Check:* do not read a sequence of leg gains as compressing
returns.

**The anchor overrates weaker arms, and past ~120M it stops discriminating at all.** Two
distinct failures of the same metric. *Overrating:* `HeuristicBot` is a fixed script, and a
differently-trained policy can exploit its habits without being stronger — an arm above the control
on the anchor and 77–110 Elo below it head to head, three times. *Saturation:* by 120M every one of
the eight 4 × 4 arms beats `HeuristicBot` above 89% and `RandomBot` above 99%, so the anchor
separates pooled from unpooled by 8.0 pp against 8.8 pp — a null — where the 20-model peer
tournament separates them completely, 5.4 pp against 33.2 pp. Overrating gives you a wrong
ordering; saturation gives you no ordering, and reads exactly like a real null. *Check:* rate
against a strong pooled reference; **no live training metric can rank arms.** If the arms are
beating the anchor above ~85%, the anchor is furniture, not an instrument.

**A wall-clock budget hands two arms different amounts of training**, and contention is not
symmetric between parallel arms (7,044 vs 7,833 steps/s). *Check:* budget by `--train-steps`.

**Two models with the same filename collide in the Bradley-Terry fit**, and `snapshot_final.pt`
is weight-identical to the last step snapshot — listing both enters one player twice. *Check:*
distinct filenames; compare tensors before assuming two files are two models.

**Sorting paths numerically does not sort them numerically.** `sort -t_ -k2 -n` over a path full
of underscores compares every line equal and leaves `ls`'s lexical order, which equals numeric
order only while every step count has the same digit count. *Shows up as:* an arm's "late"
snapshots being its earliest ones. *Check:* extract the step count and sort on that; print the
selection. **A check that passes on one arm by arithmetic accident is not a check.**

**A metric's definition can move underneath its series.** The blunder taxonomy was redefined
mid-programme. *Check:* a step change at a boundary is the definition moving; re-measure before
comparing across it.

## The probe itself

**A probe is only a test of understanding if its answer key is absent from the observation.**
`global_features[64..69]` already carry the live per-region VP differential, and board slot 24 is
`my_deficit`. *Shows up as:* a linear probe on the raw input scoring 0.886 AUC before the encoder
contributes anything.

**An estimator that has never been shown a known answer is not trusted.** Rounding a least-squares
fit scored *below* the constant baseline; one-hot class regression masks intermediate classes;
a single global penalty held the `raw` rung to 85% with the answer in slot 0. *Check:* gate the
estimator — feed it the noiseless column and require near-perfect recovery, and feed it pure
noise and require exactly the baseline (`tests/training/test_state_readout_probe.py`); tune any
penalty per stage on a validation split.

**An aggregate over ratios with near-zero denominators is decided by noise.** India has 4%
headroom, so two points of probe noise become −431%. *Check:* aggregate as
`sum(acc - base) / sum(headroom)`.

**A classifier that follows only *forced* continuations scores a win taken by another line as a
decline**, and a simultaneous headline decision cannot be judged as an action round at all.
*Check:* ask whether the action the player chose also wins, not whether it is in the
classifier's set; exclude headlines.

**A turn-boundary convention can invert a classification.** A completed game terminates holding
turn **11** while a human log numbers it turn 10, so `>= 10` caught only turn-10 Wargames.
*Check:* measure game length in plies (`ai/game_length.py`), and take `game_ended` from the
converter's terminal test, not from the log's own fields.

**A benchmark that never played a game can report a 16% win.** Driving the engine with
`get_legal_action_indices` — a different index space — made `step_flat` reject every action and
both modes stall identically. *Check:* drive with `ActionMask.generate_flat_mask` and assert
`step_flat` returned true; sanity-check that games completed.

**A representation probe moving the right way is not evidence a change helped.** The arm that
most improved the trunk ladder is the one that halved play.

**An aggregate can answer nothing.** Hand AUC is ~0.86 for every arm, including ones that
provably cannot identify a card, because most cards are separable by properties. *Check:* build
the restricted test where the shortcut is unavailable — here, positions where exactly one member
of a same-feature group is in hand.

## Sampling temperature is a property of the measurement, not of the model

A rating is uninterpretable without the temperature it was taken at, and `tools/tournament.py`
now records it in the report header and the JSON. Measured on `p28_200M`
([`log/P15_temperature_selfplay.md`](../log/P15_temperature_selfplay.md)): greedy and the old
default of 0.1 are the **same player** (50.0% over 300 games, 6.9 Elo), 0.25 is still inside
noise, and then it collapses — 0.5 costs 134 Elo, 1.0 costs 374.

Two consequences. Older numbers taken at 0.1 are comparable with newer ones taken at 0, which
makes a whole class of cross-tournament comparisons legal again. And the variable still has to be
controlled whenever a treatment changes the policy's entropy, because then the two arms are no
longer equally far from their own argmax — the X4b arm sits at 0.56 nats against its control's
1.17.

## One-sidedness of self-play is not evidence that a side is degrading

`critic_base_rate` is `max(p, 1 − p)` over resolved self-play games (`critic_tracker.py:140`) —
the **majority-class rate**. Two consequences, and both have caught reports in this repo:

* **It has no direction.** It is always ≥ 0.5 and says nothing about *which* side leads, so it can
  never support a claim about a named side.
  [`log/seed_variance_and_pooling.md`](../log/seed_variance_and_pooling.md) already records this
  and notes that earlier reports reading it as "US-leaning" were wrong.
* **It is a fact about the pair, not about either side.** A base rate of 0.96 is equally
  consistent with:

  * one side collapsing, and
  * both sides improving at different rates.

  The statistic cannot distinguish those, so it cannot support the conclusion that a side is
  getting worse. This is the self-play side-advantage pitfall above in another guise, and it is easy to fall
into because `critic_base_rate` is logged every iteration and a rising line looks alarming.

**The instrument for the question is win rate as each side against a frozen opponent** — a fixed
anchor checkpoint, rated per seat, which is what the per-side matrices in `tools/tournament.py`
reports give. A checkpoint that wins 61.9% as USSR and 81.0% as US is not "imbalanced" in any
sense that matters; it is strong on both seats and stronger on one.

`critic_base_rate` remains useful as a cheap in-run hint worth following up with a real
measurement. It is never the measurement.
