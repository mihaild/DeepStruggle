# P15 — Breaking the oscillate-then-stall cycle

**Status:** proposed, awaiting approval
**Gate:** none for X0–X3 (flags and eval config; X1 needs a side-lock in the trainer);
X4 needs the P3 search-in-training build (trainer only — no engine, no observation change).
**Needs approval:** the X1 side-lock and the X4 build; X2/X3 are existing flags or small
trainer changes.

## The failure, as the record states it

Once one side finds a strategy the other has not answered, outcomes become predictable, the
critic degenerates to the base rate, `adv_std_raw` collapses ~6×, and both policies freeze
([`../log/europe_control_and_held_scoring.md`](../log/europe_control_and_held_scoring.md));
the direction is an accident of the run (E3-14-21 ran away toward the *US*), the strategy
differs by run, and the imbalance **oscillates rather than converges** — at 320M both extended
pooled arms moved *away* from balance, in the same direction
([`../findings/training/pooling.md`](../findings/training/pooling.md) §4). The opponent pool
settles balance decisively and is suggestive on strength (pooled 4/4 improve over the second
80M where unpooled decay, p = 0.057 paired; 61.5% cross-condition), but no arm of any
configuration progresses after ~160M. Meanwhile the honest searcher over the *same network*
beats the raw policy ~75% (76.7% at 384 sims, ~72.5% at 96, saturating between —
[`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md)), so the weights
contain ~150–190 Elo the policy does not express. And there is **no in-run instrument that can
rate an arm after ~120M** — HeuristicBot and RandomBot are saturated
([`../log/measurement_bugs.md`](../log/measurement_bugs.md)).

This is the textbook behaviour of simultaneous gradient dynamics in a zero-sum game: the
last iterate *orbits* the equilibrium (best-response cycling) instead of converging, and when
one orbit segment makes outcomes predictable the gradient dies until the other side drifts
into an answer. The standard remedies form four families, and the current recipe is a weak
version of two of them:

| family | canonical form | what this project runs today |
|:---|:---|:---|
| **average / remember** — play the history, not the latest | fictitious play; NFSP; δ-uniform sampling (OpenAI Five); PFSP + league (AlphaStar); PSRO meta-Nash mixtures | pool = uniform over the **last 12 snapshots ≈ a 60M-step sliding window**, 30% of envs — memory shorter than the oscillation it should damp, and a pooled game faces ~5 snapshots in sequence (pooling.md §3b) |
| **anchor on a slower timescale** — regularize toward something that does not follow the cycle | R-NaD's two-timescale reward transform; MMD magnet (EMA); NashPG's outer loop; Ataraxos's annealed damping | KL to `π_ref` at η = 0.1 (worth +169 Elo, settled) — but `ref_update_freq` = **200k steps ≈ 16 seconds of wall clock** at 45M steps/h. The anchor is refreshed ~400× per 80M leg: it *tracks* the cycle, so it damps step size but cannot damp the orbit. NashPG's own convergence story assumes the inner loop approaches the regularized fixed point *before* the reference moves |
| **optimism / extragradient** — anticipate the opponent's update | OGDA, extragradient, optimistic mirror descent | absent; deeper optimizer surgery, kept in reserve |
| **inject gradient the outcome cannot supply** — a training signal that is non-zero when win/loss is decided | expert iteration (ExIt, AlphaZero); Gumbel policy improvement at 2–16 sims; KataGo's auxiliary targets | absent — search-in-training does not exist (search_cost_and_coverage.md §5); the auxiliary control head is ranked in [restoring_advantage_signal.md](restoring_advantage_signal.md) |

The search result changes which family is most promising. Distilling the searcher's decision
back into the policy (family 4) has gradient *even where the outcome-advantage is exactly
zero*, because its target is "what the improved policy prefers", not "who won" — it is the one
lever that does not route through the dead terminal signal. And the feasibility objection is
answered by the coverage table: the infeasible number (108h/40M) is *every decision at 64
sims*; at 32 sims, card/play-mode nodes, 1-in-8 subsample, an 80M leg costs **~7.7h**
(§4 of the search log), with Gumbel-style improvement guarantees available at exactly this
sim count.

## The experiments

Ordered so each is read through the one before it. Budgets use 45M steps/h.

### X0 — frozen-anchor gate (instrument; prerequisite for everything else)

Add frozen peer anchors to the in-run snapshot evals: E3-20-28@160M (strong), E3-20-28@80M
(mid), HeuristicBot kept for continuity with old traces. Log **per-side** win rate against
each anchor at every snapshot. This restores an instrument above 120M, makes the oscillation
*quantifiable* (amplitude and period of per-side WR vs a fixed opponent, sliding window), and
supplies the endpoint pooling.md says nothing has computed — external side balance averaged
over the final 40M. Eval-time only; `--eval-opponents` already accepts checkpoints.
*Decide before running:* games per anchor per snapshot (200/side keeps snapshot cost minutes).

**Detail.**
- Anchor files: mind the directory trap in [`../checkpoints.md`](../checkpoints.md) — the
  bare-name directory is sometimes the continuation; pin the anchors by full path + step count
  in the run's `metadata.json`, and record them as `eval_anchors` so the trace is attributable
  later.
- Two derived series, logged per snapshot: `anchor_wr_us`, `anchor_wr_ussr` (per anchor), and
  `anchor_side_gap = |wr_us − wr_ussr|`. The oscillation metric is the sliding-window
  (±20M-step) amplitude of `anchor_side_gap`; the strength metric is mean anchor WR.
- Backfill: run the same evals over the *existing* snapshot ladders of E3-17-22 (unpooled),
  E3-20-28 and E3-20-29 (pooled) — a few CPU/GPU-hours — so the oscillation's amplitude and
  period are measured on the arms already on disk **before** any new arm runs. That
  retro-trace is what X2/X3 screens are compared against.

### X1 — the frozen exploiter (P10 experiment 2, unrun; the decisive diagnostic)

Freeze the stalled 160M policy as USSR; train a US-locked learner initialised from the 80M
state against it (~2h). If the learner's win rate climbs well above the ~10% start, **a
response exists and self-play failed to supply the gradient to find it** — the
cycling/starvation diagnosis is confirmed and X2–X4 are attacking the right thing. If it
plateaus, the runaway strategy is near-unbeatable at this capacity and the queue changes
(architecture/capacity, not dynamics). Needs the side-lock; the pool machinery already plays
frozen opponents. This also doubles as the first *main exploiter* in the AlphaStar sense, and
its product is a pool member X3 can use.

**Detail.**
- Register as the next free arm number, baseline pair E3-17-22 (it is that lineage's own
  stall being probed). Opponent: E3-17-22 final (160M), frozen, always USSR. Learner: init
  from `resume_80019456steps.pt`, always US.
- Mechanism: the opponent-pool path with pool = that one snapshot, `--opponent-frac 1.0`,
  plus a new `--opponent-side-lock {US,USSR}` (learner takes the other side; the pool's
  per-episode side alternation is bypassed). Policy loss masked to learner transitions as the
  pool already does; value trains on all states.
- Budget 80M (~2.2h at pooled throughput); only ~half the transitions carry policy gradient,
  which the extra length buys back (P10's arithmetic).
- Read: learner WR vs the frozen opponent over training (the primary curve), `adv_std_raw`,
  `critic/auc` on the learner. P10's caveat stands: a best response to one snapshot is not a
  Nash improvement — before concluding anything general, tournament the exploiter against the
  wider field.
- *Decide before running:* whether to run the mirrored cell too (freeze a US-runaway arm,
  e.g. E3-14-21, train USSR) — cheap and makes the diagnosis side-symmetric.

### X2 — slow the anchor (two-timescale regularization, the theory-aligned flag)

`--ref-update-freq` 200k → {5M, 20M}, η = 0.1 held; one optional cell η = 0.3 at 5M. Resume
from the 80M healthy state, +80M. Screen the three cells at one seed each (~1.8h/cell), read
the X0 oscillation trace (a within-run trace, far less seed-noisy than an endpoint Elo), then
confirm the winner at 2 seeds. Risk: a slow anchor over-regularizes fresh learning — which is
why the arms resume from 80M rather than start cold; if the screen looks good, a cold-start
confirmation decides the adopted schedule (fast-early / slow-late is the expected shape,
matching Ataraxos's annealed damping).

**Detail.**
- Cells: `--ref-update-freq {5000000, 20000000}` at `--eta 0.1`, one optional
  `--ref-update-freq 5000000 --eta 0.3`. Everything else the E3-20-28 recipe; resume from its
  80M state with a fresh `--seed` (branch points and reseed-on-resume exist and are tested).
  This also subsumes P10's experiment 3 (seed-resume: does the collapse recur?) — the control
  cell for these screens *is* that experiment, so register it as such and write both answers.
- Watch `internal/kl_div` for scale: with a 100× slower anchor the KL term's magnitude grows;
  if it dominates the loss the right response is the η=0.3-at-5M middle cell, not silently
  rescaling anything mid-run.
- Screens are judged on the X0 trace (oscillation amplitude vs the retro-trace of the
  matched baseline), which is within-run and cheaper to read than endpoint Elo; the 2-seed
  confirmation of the winning cell is judged on both.

### X3 — longer memory, adversarial weighting (upgrade the pool from window-FSP toward league)

Three sub-arms, each one factor on top of the pooled baseline (frac 0.30 unchanged):

- **a. Span-the-run pool.** Keep every ~10M-step snapshot (reservoir-capped), δ-mix draws:
  ~80% from the recent window, ~20% uniform over the whole history (the OpenAI Five recipe).
  Directly fixes "memory shorter than the cycle".
- **b. PFSP** — already built (`--opponent-pfsp`, variance weighting, exposure-weighted
  attribution). Run the registered arm; pooling.md §3b already narrows the claim it can make
  (it shifts the mixture, not a per-game identity).
- **c. One-opponent-per-episode** — the coherence follow-up pooling.md §3b leaves open. Only
  if a or b moves anything; it trades the diversity that the balance result credits.

*Kept out for now:* PSRO-lite (sample the pool by meta-Nash of the head-to-head matrix) —
queued in reserve; it needs a periodic in-run tournament and only pays if uniform-over-history
is insufficient.

**Detail.**
- (a) is a retention-policy change in the pool: today it keeps the last 12 snapshots at 5M
  cadence — a ~60M sliding window; change to a capped span-the-run set (keep every ~10M, or
  reservoir), draw ~80% from the recent window / ~20% uniform over the whole history.
  One factor: frac stays 0.30, capacity stays 12, cadence stays 5M so pool growth matches the
  baseline (`compare_runs.py` checks this — the E3-22 void is the cautionary example).
- (b) is the registered PFSP arm (`--opponent-pfsp`, variance weighting, uniform-mix 0.25),
  exposure-weighted attribution already built. Run it as registered; pooling.md §3b bounds
  its claim — it shifts the mixture's composition, not a per-game opponent identity.
- Each sub-arm resumes from the E3-20-28 80M state, +80M, one seed screen → 2-seed confirm
  of anything that moves the X0 trace.

### X4 — expert-iteration distillation (family 4; the strongest single bet)

Build P3's trainer hook: at card/play-mode decisions, 1-in-8 subsample, run the honest
searcher (`determinize=True`, 32 sims — Gumbel root if available, completed-Q values
otherwise) and add a CE term pulling the policy toward the search policy **on searched
decisions only**; everything else unchanged. Resume from the 160M stall, +80M ≈ 8h; 2 seeds.
The claim being tested is precise: the ~75% search edge is expressible without search at play
time, and its gradient does not die with the outcome signal. The owner's no-search preference
is honoured — search runs in training only.

**Detail.**
- **Distill, don't act.** Rollouts keep sampling from the raw policy; search runs only to
  produce CE targets on the subsampled decisions. This keeps the state distribution identical
  to the baseline's — one factor — and is what Ataraxos does (direct policy sampling for
  data, improvement operators elsewhere). Acting on the search policy is a *second* arm if
  the first works, never the same arm.
- Searcher config: `BatchedMCTS`, `determinize=True` (the honest form — privilege is measured
  to be worth nothing at strength, and an honest teacher cannot leak hidden-hand information
  into the policy), 32 sims, `node_filter=card_playmode`, `subsample=8`. Target: visit
  distribution with completed-Q / Gumbel weighting if available, else visit counts.
- Loss: `+ ce_coef · CE(π_θ(·|s), π_search(·|s))` on searched decisions only. *Decide before
  running:* `ce_coef` (first guess 0.5, annealed only if it dominates) and whether searched
  states are exempt from advantage filtering (they should be — the CE term is the point).
- Cost model from the coverage table: the 2,886 st/s figure was measured *acting* with
  search at this setting; target-only search costs about the same forward passes, so plan
  ~8h per 80M leg and measure the real figure in the first hour.
- The E3-22 lesson applies here in reverse and is worth stating: that arm improved every
  offline estimate and lost 520 Elo; this arm's CE target is *measured* to be +150–190 Elo
  stronger online, which is the right kind of evidence — but adoption still waits for the
  training outcome, not the target's pedigree.

## Registration and hygiene

Every arm: a row in [`../runs.md`](../runs.md) before launch (next free E3 number), matched
baseline named, `--snapshot-every-steps 5000000` so pool growth matches, `metadata.json`
carrying the anchors and any new flags, `tools/compare_runs.py` diff against the baseline
**before** any metric is read, and `tools/scripts/watch_run.py` armed on launch. Screens are
one seed; nothing is adopted from fewer than two.

## Measure

X0's traces first (oscillation amplitude, per-side WR vs frozen anchors, `adv_std_raw`,
`critic/auc`), then endpoint Elo vs the anchors, pooled over four late snapshots per
[`../method/running_experiments.md`](../method/running_experiments.md). The success criterion
is **not** "the US recovers" (the runaway is bidirectional): it is that neither side's
advantage signal dies, oscillation amplitude shrinks, and strength against *frozen* anchors
resumes rising after 160M. Effects under ~40 Elo are unmeasurable at affordable seeds
([`../findings/training/seed_variance.md`](../findings/training/seed_variance.md)) — every
adopt decision clears that bar or is dropped.

## Decision rules

- X1 climbs → diagnosis confirmed; X1 flat → stop X2–X4, reopen architecture/capacity.
- X2 adopted iff oscillation amplitude falls *and* anchor-Elo is not worse at matched steps;
  the cells are flags, so kill freely.
- X3a/b adopted on the same rule; if both are neutral the pool stays as it is and PSRO-lite
  stays in reserve.
- X4 adopted iff the no-search policy gains ≥ 40 Elo vs frozen anchors over the matched
  continuation; if it gains nothing while the searcher still beats it 75%, the distillation
  target or coverage is wrong before the idea is (check CE on searched states falls).

## Budget

X0 eval-only; X1 ~2h; X2 ~5.5h screen + ~3.6h confirm; X3 two arms ~3.6h each at one seed,
confirm winner ~3.6h; X4 build + ~16h (2 seeds). Roughly **40–50 GPU-hours** for the whole
programme — under a week of 4090 nights, with X1 gating the rest after the first evening.

## Follow-ups

- X2 or X3 adopted → re-run the 320M extension question on the new recipe (does it still
  oscillate away from balance?).
- X4 adopted → sweep coverage upward (1-in-4, card/play-mode-all) along the measured
  coverage curve; consider the searcher as the *evaluation* exploiter for approximate
  exploitability of future arms.
- All neutral → the reserve entries in order: optimism/extragradient in the optimizer,
  PSRO-lite meta-Nash sampling, per-side capacity.

## Runs

(none yet)
