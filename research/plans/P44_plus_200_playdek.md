# P44 — toward +200 on the Playdek scale: what to measure, and the next runs in order (2026-11-10)

**Status:** proposed; written down at the owner's request (2026-11-10). The three choices under
*Decide before running* are open, each with the default this plan would take.
**Gate:** none for step 0; step 1 needs the small trainer flag named there.
**Needs approval:** none (no engine, bindings or observation change).

## Goal

+200 Elo on the Playdek scale for the deployed player (the network under the rollout root): from about
1,700 to about 1,900, a very good but not top human. One GPU, where two trainers share the throughput of
one; a rented machine only on the evidence listed under *What would justify renting*.

## Why

A review of the record on 2026-11-10 (five digests: the E8 runs and P37-P43, search and distillation, pools
and leagues, human measurement, outside literature), then the human games read from the stats database.
The numbers are the logs'; the readings below are this plan's and replace several stated conclusions.

**Human games** (clean, rated games as `tools/playdek_estimate.py` counts them, to 2026-11-10; every game is
US +2, the mode the models train in; intervals are a 95% bootstrap over people):

| | E7 R34 soup, keyed 32-world root | E8 R98, 64-world root, race 16 |
|:---|---:|---:|
| games / people | 129 / 22 (one person 53) | 54 / 12 |
| human rating, median (range) | 1633 (1200-2010) | 1800 (1615-1980) |
| model score | 49.2% [25.0, 61.2] | 34.3% [12.0, 52.6] |
| as US | 31.4% [13.0, 54.7] (43 games) | 21.2% [3.6, 50.0] (26 games) |
| as USSR | 58.1% [35.7, 70.0] (86 games) | 46.4% [18.2, 59.0] (28 games) |
| against 1700-1900 | 30.4% (46 games, 12 people) | 36.5% (48 games, 8 people) |
| against 1900+ | 16.7% (12 games, 4 people) | 25.0% (4 games, 2 people) |
| losses by Europe Control | 8 of 64 (12%) | 4 of 35 (11%) |
| critic in lost games, end of turn 1 / 3 / 5 / 7 | -0.06 / -0.15 / -0.17 / -0.31 | -0.01 / -0.09 / -0.30 / -0.43 |
| critic in won games, same turns | +0.28 / +0.46 / +0.57 / +0.58 | +0.35 / +0.63 / +0.62 / +0.46 |

* **The US seat is the weak one against people**, by about 25 points in both players (the intervals
  overlap; the direction agrees). Self-play balance cannot show a weakness of this kind.
* **Europe Control ends 11-12% of the models' losses against 3.3% of their self-play games.** The owner's
  expectation of a strong USSR against J1's US setup is the same line: West Germany early, a high Europe
  score, control with some luck. It is also what J1's USSR exploiter does (West Germany on turn 1 in 30%
  of games against 3%, Europe controlled in 8.7% against 3.2%; [P41](P41_jaxcalibur_ingredients.md)).
* **Losses are not late surprises**: the critic starts level and falls from turn 3; won games are already
  ahead after turn 1.
* **Self-play Elo's transfer to people is unmeasured.** The estimate fixes the slope at 1. Fitted freely it
  is 1.09 [0.22, 2.00] for the E7 soup and 0.93 [-0.06, 1.67] pooled over 26 people; for R98 the ratings
  span too little to fit one, and people at its estimated position score about 74% against it. R98's
  higher estimate than the E7 soup's is not supported by its own games.

**Training:**

* **The plateau is the policy heads losing plasticity, and J1 has it too.** At 13,600M J1's own heads fit
  random labels at 36.7% against 99.1% for a fresh head on the same trunk, with 11.6% of head units
  dormant and head norms 14-34 times fresh ([P41](P41_jaxcalibur_ingredients.md),
  [P37](P37_known_playbook_remainder.md)). J1 has read about 62% against the reference since ~13,000M, and
  its 17,200M state scores 50.2% against its own 16,800M. The zero-avoiding term removes saturation of the
  *output* (top-two logit gap 1.23 against 3.50) and delays the plateau; it does not keep the heads
  trainable. P37's remedies for that were never run.
* **J1 is exploitable in about an hour of GPU.** Seat-locked exploiters, 400M each from J1's 15,210M
  state, gain +5.5 (US) and +2.1 (USSR) over its self-play bars (R108, R109), from Europe left uncontested
  after the setup and not from the setup itself.
* **No league run tested whether a main absorbs an exploit.** Every attempt was one alternating-seat
  exploiter behind a 400-game publish bar; the E7 null was a league that held one member for 490M
  ([P24](P24_league.md)). Pool fraction, size and spacing were never varied.
* **"Search distillation is closed" holds for the cells that were run.** The students that failed either
  could not fit (saturated heads: 21.9-40.9% of training departures) or never learned the game (from
  scratch, 43-45% overall agreement). A student both plastic and competent was not measured
  ([P36](P36_issue_bank_and_distillation_test.md)). What may bind instead is margin: a root departure
  leads by about 0.03 of value, and a fresh action-value head ranks the root's leads at r = 0.80 yet gains
  nothing when it acts on them.
* **"Capacity does not help" is not established.** No larger network reached a plateau; each negative was
  read by 2,000-4,800M under an optimisation symptom. Distil-then-RL loses its start on A8 itself (43.1%
  against its own start at 410M), so that loss belongs to the procedure and not to depth.
* **The token transformer is not competitive at this throughput**: 16 times slower in wall-clock at about
  A8's parameter count, behind J1 head to head from 2,400M (44-46%); the read is confounded by the reward
  switch at 930M and an untuned rate. No GPU for it under this plan.
* **J1 is one seed** (J1' differs in the entropy bonus). Its lead over R98 under search (1693 against
  1650) is transitive through two opponents; they never played each other.

**Outside this project** (checked 2026-11-10 unless marked):

* Juliani and Ash 2024 (arXiv:2405.19153), the one systematic study of plasticity loss in PPO: final-layer
  resets, plasticity injection and dormant-unit recycling failed; L2 toward the initial weights, soft
  shrink-and-perturb and LayerNorm worked, best combined.
* Lyle et al. 2024 (arXiv:2407.01800): under normalisation, growth of the weight norm is a decay of the
  effective learning rate; projecting weights back to their initial norm holds it constant.
* Igl et al. 2021 (arXiv:2006.05826), ITER: a fresh network distilled from the current one in parallel on
  the live PPO batches, with the distillation weight annealed to zero, then swapped in. No use of it in
  self-play was found.
* Timbers et al. 2022 (arXiv:2004.09677): the win rate of a best response trained against a frozen agent
  as approximate exploitability, a measure independent of the agent's own lineage.
* Jaxcalibur is a Pokemon Showdown random-battles agent, not a Twilight Struggle one. Its play-time
  search has the rollout root's shape and margin (32 worlds, depth 4, +100-150) and is not distilled
  either.

## What to measure (standing, from now)

1. **Human games by seat, rating band and ending**, as in the table above, each time the estimate is
   refreshed. Two models are compared on the human scale only when their fields overlap in rating; twelve
   people in one band cannot order them.
2. **The exploiter margin, per seat**: a fresh 400M seat-locked exploiter (R108/R109's recipe) against the
   frozen candidate, with the Europe Control rate and West Germany on turn 1 in its games. The USSR
   exploiter against the candidate's US seat is the human-relevant half.
3. **The head-plasticity probe every 2,000M on every long run**: own-head random-label fit, dormant
   fraction, head norm against fresh (P37a's probe). J1 has two points; A8 fell from 47% at 2,000M to 35%
   at 6,400M.
4. **A new fixed reference.** The A13+R48 SWA no longer separates players at J1's level; J1's
   13,520-13,600M SWA takes over for new runs, the old one kept beside it for one overlap.
5. **Setups at every read** (`tools/scripts/checkpoint_report.py`'s census): West Germany's share in the
   US setup, by step.

## Procedure

One step, one question; the order is by information per GPU-hour. A8 alone runs about 8.6 billion steps a
day. Run codes are allocated at launch, after checking `main` and `experimental-main`.

### Step 0 — short runs (about half a day)

**0a. Does a plateau network absorb an exploit?** Branch J1 at 15,210M with R108 and R109 added to its
pool as fixed opponents in about 15% of games, 800M, everything else unchanged. Then two fresh seat-locked
exploiters against the result, 400M each. About 1,600M, five hours.

* Measure: the fresh margins against +5.5 / +2.1 (±1.6), the Europe Control rate in those games, the read
  against the fixed reference.
* Decision: both fresh margins lower by more than 2 SE of the difference with the reference read not
  lower -> the exploit is absorbed at a plateau, and step 3 is a league from J1's state. Otherwise repeat
  0a from the new control's 4,000M state (step 1), where the heads still train; a league from a plateau is
  then closed.

**0b. Why does distil-then-RL bleed?** Resume J1 from 15,210M with a fresh optimiser state, 50M (ten
minutes), read against its own start. A bleed like D7's (43.1% at 410M, most of it early) puts the cause
in the optimiser restart, and ITER starts with a rate warm-up; no bleed puts it in the distilled function
(its sampled play away from the teacher's top move), and ITER's on-policy terms are the test.

**0c. J1 against R98, both under the deployed root**, one match, 1,000 deals a side. It decides which
network people play next.

### Step 1 — a second seed as the new control, paired with one plasticity arm (about 3.2 days)

**The control:** J1's recipe from scratch on seed 45 with the pure win/loss target (R119, the default
since 2026-11-06), to 14,000M. It differs from J1 in seed, reward and the engine fixes since, so it is a
new baseline and not a replicate: a large difference from J1 could not be attributed. Pure win/loss was
level as a branch (59.3% against 60.4%) and from scratch at 400M (17.3% against 16.0%). Beyond being a
control it answers whether the empty West Germany is the recipe's (J1, J1' and J5 all lost it; the R48
line kept it) and what the seed spread is at this recipe's plateau.

**The arm, beside it from step 0 and on the same engine:** the control plus `--ladder-head-norm` and L2
toward the initial weights on the policy and value heads (a new trainer flag; coefficient from the
1e-4 to 1e-2 range of Juliani and Ash, fixed by a 60M check that the head norm stays within 2 times
fresh). Two changes in one arm, against rule 6 of this directory, deliberately: head LayerNorm alone
removed dormant units on R48 without a matched-step gain, and under normalisation what remains is norm
growth; the pair is the combination the literature reports working. If the arm wins, the two are
separated afterwards.

* Both runs: measures 3 and 5, saved states at 2,000M and 4,000M as branch points, reads every 400M
  against the new reference.
* Early screen: the arm's own-head fit not clearly above the control's by 4,000-6,000M -> stop the arm,
  and the slot goes to the next remedy (weight projection to the initial norm, then soft
  shrink-and-perturb).
* Decision at 14,000M: the arm passes when its probe stays near a fresh head's and it is above the
  control per seat by the usual rule, or still rising over the last 2,000M where the control is flat.
  Close arms are not ordered before ~1,600M.

### Step 2 — in the gaps between long runs

**2a. Search distillation, the cell that was not run** (about a day): fresh policy heads on J1's frozen
trunk fitted to the root's targets (P36c's set rebuilt on J1, 200k entry decisions at 64 worlds). Measure
held-out agreement on departures against the teacher's own 75-79%, and the realised gain of the student's
departures by paired playouts.

* No realised gain -> the limit is the margin, not the student; search distillation stays closed until a
  teacher with a larger lead exists (a belief-weighted root is the owner's call and not part of this plan).
* A gain -> search targets go into step 3's ITER student.

**2b.** 0a repeated from the control's 4,000M state, if 0a says so.

### Step 3 — by the outcomes above

* **ITER** on the better of step 1's two runs: a fresh A8 distilled in parallel on the live batches, the
  distillation weight annealed to zero, swapped in; then the same with a wider student, which is the
  capacity test this record lacks.
* **A league on J1's recipe**, if 0a or 2b shows absorption: seat-locked exploiters, 400M generations,
  every generation published (the margin is read against the self-play bar, so no publish bar), from the
  state where absorption was shown; judged by measure 2 on the final main, not by self-play.
* **Auxiliary heads on A8** (opponent's next action, hidden hand: P41's J2/J3, gated there on the
  transformer for no technical reason), from scratch against step 1's control.

## What would justify renting

* Step 1's arm still rising where the control is flat: longer runs pay, and a faster machine shortens
  every read.
* ITER into a wider student holds its start and gains: capacity pays.

Neither is shown by a run that merely finishes sooner.

## Decide before running

1. Step 0 as written? (Default: yes.)
2. Step 1 paired from the start, or the control alone first (1.6 days) and arms after? (Default: paired;
   the total is the same and the matched reads come earlier.)
3. J1's searched player in front of people as it is? (Default: yes; it costs no GPU, and Europe Control
   losses as US are the direct read on the empty West Germany.)

## Runs (filled in while running)
