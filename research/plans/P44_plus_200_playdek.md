# P44 — a stronger model: what to measure, and the next runs in order (2026-11-10, revised 2026-11-12)

**Status:** running (step 0 done, step 1 running); **revised 2026-11-12** after step 0, step 1's first reads and
the owner's corrections -- read *Revision of 2026-11-12* first; where it and the older text disagree, it holds.
**Gate:** none.
**Needs approval:** none (no engine, bindings or observation change).

## Goal

**A stronger model** (owner, 2026-11-12: exact human ratings are not important). The yardstick is the old
reference, `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@12730..12800M`, greedy: J1 reads 61-64% against it, and
**the target is 75% or higher**. The plan was first framed as +200 Elo on the Playdek scale for the deployed
player (the network under the rollout root), about 1,700 to about 1,900; that stays the motive, not the
measure. One GPU, where two trainers share the throughput of one; a rented machine only on the evidence
listed under *What would justify renting*.

## Revision of 2026-11-12

Numbers and instruments: [`../log/P44_checks_2026-11-12.md`](../log/P44_checks_2026-11-12.md).

* **Withdrawn: "the plateau is the policy heads losing plasticity".** The probe decays mostly by 6,000M while
  J1 rises to ~13,000M (*Runs*, below); the head-norm arm is stronger than the control with a lower probe;
  J1's policy moves as much per update at 17,000M as at 4,000M while its head norms double; a head reset at
  the plateau (R87) had already returned only to the band, which this plan under-weighted. With it go the
  probe as a decision rule, L2 toward the initial weights (the 60M check this plan wrote -- norm within 2x
  fresh -- asked for heads that cannot grow; the control's are 3-8x fresh by 400M), weight projection and
  shrink-and-perturb. The working reading, a hypothesis: the plateau is signal against noise at the current
  step size, and weight growth is a slow decay of that step (log §4).
* **Withdrawn: "the old reference no longer separates players at J1's level"** (measure 4). It was this plan's
  claim; J1 is at 61-64% on it and the target is 75%. The old reference is the primary read again.
* **Withdrawn: "the US seat is the weak one against people"**, and "won games are already ahead after turn 1".
  The reported games miss about two thirds of the models' late wins in both seats -- a losing player leaves,
  and nothing is sent -- and US wins are the late ones (log §5). What stands is how people win: Europe Control
  against the models' USSR, 9 of 50 reported wins as US against 1.3-1.4% in self-play.
* **Found in step 0's data: the league branch is stronger than J1's continuation** -- 64.2% against 60.9% on
  the old reference, 54.7% head to head -- although its exploiters sat in their untrained seat half the time
  and no member played a whole game (log §1-§2). The league is now the first line of work:
  **[P45](P45_league.md)**.
* **Step 1:** the arm is the head norm alone (A31). It beats the control 55.3% directly at 3,200M; run-to-run
  spread at that step is ~7 points on the reference, so it is judged where the control caught J1, at
  4,800-5,600M, and again at 8,000M (against the control if it has run that far, else against J1's own read
  at that step): ahead by 2 points or more on the reference at both -> A31 is the base architecture for the
  next from-scratch run. The control may pause at a tagged state for P45's L1 (it has been level with J1
  head to head since 4,800M; J1's own curve stands in past that).
* **Dropped:** step 2a (the search-distillation cell: every student so far switches moves wrongly as often as
  rightly, and "the network's own second choice" at 51% of departures beats all of them; a fresh head on the
  same trunk has the same inputs), L2 at 1e-5, measures 1 and 3 as things decisions ride on.
* **After P45's first reads, in this order:** a soup-teacher cycle on the plateau (three seed branches, their
  soup as a frozen teacher for 400M: +60-75 Elo over the plain leg on E7, never run on E8); a rate anneal at
  the end of the final recipe; a distilled student at a matched step (about 5e-5), A8-sized and wider, as the
  capacity test -- the earlier students lost their start at 3-6 times J1's step (log §4), so that failure
  said nothing about capacity; auxiliary heads on A8.

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

**The table is what was reported, and what was reported is selected (revision of 2026-11-12):** a player who is
losing a long game leaves, and nothing is sent. About two thirds of the models' late wins are missing in both
seats, so the scores, the seat split, the rating bands and the critic rows above are not estimates of
anything; the first reading of them here ("the US seat is the weak one", "won games are already ahead after
turn 1", a free-slope fit) is withdrawn.

* **Europe Control is how people beat the models' USSR:** 9 of the 50 reported human wins as US, against
  1.3-1.4% of US wins in the two players' self-play; as USSR people win by it at the self-play rate (3 of
  49). The table's "11-12% of losses" row is that, pooled over seats. (The first version put it on the US
  seat and on J1's empty West Germany; the games do not say that.)
* **Self-play Elo's transfer to people is unmeasured**, and these games cannot measure it.

**Training:**

* **J1 has a plateau, and its heads read as unplastic on the probe** -- the first version called the second
  the cause of the first; **withdrawn** (revision of 2026-11-12). At 13,600M J1's own heads fit random
  labels at 36.7% against 99.1% for a fresh head on the same trunk, with 11.6% of head units dormant and
  head norms 14-34 times fresh ([P41](P41_jaxcalibur_ingredients.md),
  [P37](P37_known_playbook_remainder.md)). J1 has read about 62% against the reference since ~13,000M, and
  its 17,200M state scores 50.2% against its own 16,800M. The zero-avoiding term removes saturation of the
  *output* (top-two logit gap 1.23 against 3.50) and delays the plateau.
* **J1 is exploitable in about an hour of GPU.** Seat-locked exploiters, 400M each from J1's 15,210M
  state, gain +5.5 (US) and +2.1 (USSR) over its self-play bars (R108, R109), from Europe left uncontested
  after the setup and not from the setup itself.
* **No league run tested whether a main absorbs an exploit.** Every attempt was one alternating-seat
  exploiter behind a 400-game publish bar; the E7 null was a league that held one member for 490M
  ([P24](P24_league.md)). Pool fraction, size and spacing were never varied. (Step 0a did test it, and its
  branch came out stronger: [P45](P45_league.md).)
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

1. **Human games: how people win, nothing else** (revised 2026-11-12). The reported games are selected by who
   finishes them, so scores, seats and bands are not read; the endings of human wins are.
2. **The exploiter margin, per seat**: a fresh 400M seat-locked exploiter (R108/R109's recipe) against the
   frozen candidate, with the Europe Control rate and West Germany on turn 1 in its games.
3. **The head-plasticity probe every 2,000M on every long run**: own-head random-label fit, dormant
   fraction, head norm against fresh (P37a's probe). Logged as a curve; no decision rides on it (revised
   2026-11-12: its timing does not follow the plateau).
4. **The old reference is the primary read, and the target on it is 75%** (owner, 2026-11-12; the first
   version retired it, wrongly). Every arm and its matched control are read on it at the same step, on the
   same seed sets; J1's 13,520-13,600M SWA and a head to head against the matched control are read beside
   it, because a gain can be larger inside a lineage than outside it (R94: +34 Elo on its own past, +10 on
   the reference; the league branch: +33 head to head, +25 on the reference).
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

**As run and as judged (revision of 2026-11-12):** the arm is the head norm alone (A31; the L2 term cost half
the strength at 400M at both coefficients and is dropped). The probe screen and the 14,000M probe rule that
stood here are withdrawn with the premise.

* Both runs: measures 3 and 5, saved states at 2,000M and 4,000M as branch points, reads every 400M
  against the old reference and J1's 13,520-13,600M SWA.
* The arm is judged on strength at matched steps: against the control on the old reference and head to head,
  at 4,800-5,600M (where the control caught J1) and at 8,000M (against J1's own read there if the control
  has paused). Ahead by 2 points or more on the reference at both -> A31 is the base architecture of the
  next from-scratch run. Close arms are not ordered before ~1,600M, and run-to-run spread at 3,200M is ~7
  points on the reference.
* The control may pause at a tagged state once it has shown what it can (level with J1 head to head since
  4,800M), its slot going to P45's L1; it resumes in place if the plateau's seed spread is wanted.

### Step 2 — in the gaps between long runs

**2a is dropped (revision of 2026-11-12); the text is kept for what it proposed.**

**2a. Search distillation, the cell that was not run** (about a day): fresh policy heads on J1's frozen
trunk fitted to the root's targets (P36c's set rebuilt on J1, 200k entry decisions at 64 worlds). Measure
held-out agreement on departures against the teacher's own 75-79%, and the realised gain of the student's
departures by paired playouts.

* No realised gain -> the limit is the margin, not the student; search distillation stays closed until a
  teacher with a larger lead exists (a belief-weighted root is the owner's call and not part of this plan).
* A gain -> search targets go into step 3's ITER student.

**2b.** 0a repeated from the control's 4,000M state, if 0a says so.

### Step 3 — by the outcomes above

Revised 2026-11-12; the order is the revision's.

* **A league on J1's recipe: [P45](P45_league.md).** 0a's branch came out stronger than J1's continuation
  with the league seated wrongly; P45 seats it properly (a member only in the seat it trained in, one member
  for the whole game) and lists the runs. Judged on the old reference.
* **A soup-teacher cycle on the plateau:** three seed branches of the best main (400M each), their soup as a
  frozen teacher for 400M of the main (R45's form, `--search-teacher-policy`: the soup's raw policy as the
  target). On E7 it was the largest plateau gain on record (R32 59.0%, R39 60.7%, R45 level with R39 against
  the plain leg: +60-75 Elo, the student above its teacher's raw policy);
  never run on E8. Read against the plain leg on the old reference; a second cycle only if the first gains
  2 points or more.
* **A rate anneal at the end of the final recipe** (R46 on E7: +15-25 Elo on SWAs, +48 raw): part of the
  recipe, not an arm.
* **The capacity test, as a distilled student at a matched step:** J1's SWA distilled into a fresh A8 and
  into a wider network (R114's pipeline, about an hour each), then RL at a rate that gives J1's plateau step
  (about 5e-5; the earlier students ran at 3-6 times that step and lost their start, so they tested nothing),
  the teacher's pool kept, 2,000M each, read against J1's SWA and the old reference. The wider one rising
  where the A8 one does not is capacity; both rising is fresh weights; neither is signal. ITER proper (the
  student trained in parallel on the live batches) only if the A8 student rises.
* **Auxiliary heads on A8** (opponent's next action, hidden hand: P41's J2/J3, gated there on the
  transformer for no technical reason), from scratch against the base recipe of the time.

## What would justify renting

* A recipe still rising on the old reference where J1's is flat -- the league's main over P45's L3, or the
  head-norm run past 13,000M: longer runs pay, and a faster machine shortens every read.
* The wider distilled student rises where the A8-sized one does not: capacity pays.

Neither is shown by a run that merely finishes sooner.

## Decide before running

1. Step 0 as written? (Default: yes.)
2. Step 1 paired from the start, or the control alone first (1.6 days) and arms after? (Default: paired;
   the total is the same and the matched reads come earlier.)
3. J1's searched player in front of people as it is? (Default: yes; it costs no GPU, and Europe Control
   losses as US are the direct read on the empty West Germany.)

Open since the revision of 2026-11-12:

4. Pause the control at its next tagged state, its slot to P45's L1? (Default: yes; it resumes in place.)
5. P45's L3 takes both slots: at which read does the head-norm arm pause? (Default: after its 8,000M read.)
6. P45's L6 and L7 change what every run's own pool does (one opponent for the whole game; draws weighted to
   recent snapshots): run them? (No default; the owner's call.)

## Runs (filled in while running)

**2026-11-11 (owner: "step 0 and step 1 in parallel").** At most two trainers at once, so step 1's control has
one slot and step 0 runs in the other, one trainer at a time; the arm follows step 0 in that slot (about ten
hours behind the control, compared at matched steps). Defaults taken for the three choices above.

* **Step 1's control: `E8.1-A8-R119-S45`** (launched 00:10 UTC): J1's recipe on R119 from scratch, seed 45, the
  default observation (headline_reveal + opp_event_gated, so E8.1 -- one more difference from J1, the default
  for a new network since 2026-11-06). Reads every 400M against J1's 13,520-13,600M SWA (measure 4) and
  A13+R48's.
* **0a: `E8-A8-R102-S44@15210M+R120`** (00:13 UTC): J1 at 15,210M with R108 / R109 (15,610M) as league members,
  `--league-frac 0.5` of the pool's 0.3 (~15% of games; seats alternate, so each exploiter is in its own seat in
  half its games), J1's blunder-aware reward given explicitly (the default changed on 2026-11-06; a first launch
  that took the new default was stopped within a minute and voided). Then, in the same slot: its read and
  greedy self-play bar at 16,000M; **0b** (`E8-A8-R102-S44@15210M+R121`, `--resume-fresh-optimizer`, 50M, read
  against J1's 15,210M snapshot); the two fresh exploiters (`...+R120@16010M+R108` / `+R109`) against 0a's
  15,920-16,000M SWA, read at +390M.
* **0c** (J1 against R98, both under the 64-world race-16 root, 1,000 deals a side) is playing beside the
  trainers through `tools/leaderboard_play.py`.
* **Step 1's arm: `E8.1-A31-R122-S45`**, queued after step 0 in the second slot: the control plus
  `--ladder-head-norm` (A31) and `--head-l2-init-coef 1e-3` (R122; the trainer flag added for it, b973653), the
  60M check first.

**Step 0's results (2026-11-11).**

* **0c:** J1's 64-world race-16 player beats R98's 61.6% +- 1.1 (J1 as US 64.1, as USSR 59.2; `leaderboard/matches/E8.jsonl`):
  J1 is the network to put in front of people.
* **0b:** J1 resumed at 15,210M with a fresh optimiser scores 49.8% +- 1.1 against its own start after 50M: an
  optimiser restart alone does not bleed. This does not, as the rule above has it, put the loss in the distilled
  function: D7's fresh start also reset the pool (seeded with the student) and pi_ref (the student) and ran its own
  rate, entropy and pi_ref schedule (R116), and 0b kept all of those.
* **0a:** 800M with R108 / R109 in ~15% of games cost nothing (63.5% against A13+R48's SWA at 16,000M, J1's own
  continuation 61.9; 53.4% against J1's 13,600M SWA). Fresh exploiters against it: the US exploiter (it attacks the
  target's USSR seat) -0.8 against R108's +5.5 over J1 (a drop of ~2.3 SE); the USSR exploiter (the target's US
  seat, the human-relevant half) -0.5 against R109's +2.1, which was within noise. The rule is met in one seat, and
  that seat's drop is not attributable: R108 attacked J1 at 13,600M and no exploiter attacked J1's own continuation at
  16,000M without the league, so a less exploitable J1 at 16,000M explains it as well. The missing cell: one 400M US
  exploiter (R108's recipe) against J1's own 15,920-16,000M SWA.

**Step 1's arm, revised (owner, 2026-11-11).** L2 toward the initial weights at 1e-3 (R122) and at 1e-4 (R123), each with
the head norm, lagged the control by half at 400M (8.5 / 11.5% against A13+R48's SWA, the control 18.6%); the head
norm alone (`E8.1-A31-R119-S45`, launched 14:38 UTC) read level (19.2%; 52.8% against J1 at 400M). The L2 term at
these coefficients was the cost -- our term sums over every head parameter, so the literature's range does not carry
over. **The arm continues as the head norm alone**; L2 at ~1e-5 waits for a slot. Its plasticity is read at 2,000M
against the control's 54.5% own-head fit (J1 53.4%).

**The plasticity record re-read (2026-11-11, current-engine positions).** J1's own-head random-label fit falls
53.4 -> 43.0 -> 39.8 -> 37.2% over 2,000-8,000M and 36.0 -> 33.3% over 10,000-15,200M, a fresh head on its trunk
84-87% throughout; dormant head units reach ~16% by 8,000M and stay; head norms keep growing (6-14x -> 15-37x fresh).
J1' runs 1-2 points above J1 at every step. Most of the loss is over by 6,000M while J1's read kept rising to
~13,000M, so the plateau's timing does not follow the probe: "the plateau is the heads losing plasticity" is not
supported by timing, and the arm is judged on strength at matched steps, not on the probe.
