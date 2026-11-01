# P35 — seven checks toward a much stronger base model: capacity, belief, training dynamics (2026-10-18)

**Goal.** After P34 closed (the critic is at the greedy-value ceiling; the leaks the searched player
was thought to keep were the judge's own follow-up), the owner's priorities for a significantly
stronger base model, in order: a transformer encoder at scale, capacity at saturation, a belief over
the hidden state, the KL regularisation checks, search-computed advantages, prioritised opponent
sampling, strategy conditioning for diversity. This plan gives each a concrete arm, says what runs in
parallel on one GPU with two training slots, and names the interactions that decide the order.

**What the record fixes before any of it.** The policy's argmax loses ~180 Elo to its own critic over
sixteen playouts of its own top four, and that gap is thousands of two-to-four-point corrections with
no rule in them. A judge built from the network can no longer find systematic errors. Every capacity
read so far was taken before any line's plateau (A3 at 560M, tokens at 400M, lines plateau at
4,000–12,000M), and width was the only branch still rising at 8,000M. The KL term is a 2%-of-loss
trust region on a three-iteration-old reference (the reference refreshes every 200k steps, an
iteration is 65,536), never ablated on E7. None of these seven has been measured where it matters.

**Labels.** `P35a`… for measurements that train nothing; every arm is an A or R code assigned at
launch by `run_codes.py`; cells are named here. Observation changes (items 3 and 7) are the owner's
decision and are marked as such.

## The seven items

### 1. The transformer encoder at scale (A code; from scratch)

* **What.** Countries and cards as tokens with attention between them (A6's `--ladder-token-layers`,
  d = 128, 2 layers to start), on top of A8's card-conditioned heads, the current recipe (pool, 16k
  minibatch, the gap penalty switched on late as the ladder does), trained **to its own plateau**
  by the plain line's rule (the mean of the last two 400M windows within +1.5 of the two before).
* **Why.** AlphaStar, Ataraxos and DeepNash all attend every entity to every other; here the token
  variant was judged at 400M on a line that plateaus at 4,000M, so it was never tested.
* **Cost.** ~9× the compute per step of A8 at 2 layers; ~15–20k steps/s with `--compile-update`;
  a 12,000M plateau is 8–10 days of one slot. Checkpoints every 400M, so partial reads exist and
  the run can be stopped by the rule.
* **Cheap early read (grown in place, 400M).** Token layers added at identity onto A13+R48's
  plateau (the A13 / Cell A pattern: zero-initialised projection, day-one play unchanged). Known
  false-negative risk: Cell A's region attention was active but only sharpened a saturated policy.
  A positive here is a strong signal; a level result does not cancel the from-scratch run.
* **Read.** Matched-step SWAs against the heads line (`E7-A8-R1-S44`) at 400M windows raw, then the
  plateau's SWA against A13+R48's under the keyed root. Adopt if its plateau beats A13+R48's by the
  per-seat rule raw and searched.

### 2. Capacity at saturation (A code; from scratch)

* **What.** The *current* architecture scaled, no tokens: the 512-wide projections (A13), fusion
  width 480 → 768 and 4 residual blocks (A3's shape halfway), the same heads and recipe, to its own
  plateau. The control for item 1's encoder: it separates "more parameters" from "attention".
* **Cost.** ~3× A8's compute per step, ~35k steps/s; a 12,000M plateau ≈ 4–5 days of one slot.
* **Cheap early read.** Residual blocks grown at identity onto A13+R48's plateau, 400M (the A14
  value-block pattern applied to the trunk). Same caveat as item 1's.
* **Read.** As item 1, and head to head with item 1's plateau.

### 3. A belief over the hidden state (`P35a`, then a search change, then the owner's call)

* **`P35a`, the probe (CPU, a day).** A belief network on logged self-play histories of A13+R48
  (the public record: cards each side played and when, hand sizes held over, discards, removed cards,
  the board) predicting the opponent's hidden hand (sigmoids over the hidden pool). Scored on held-out
  games against the uniform prior (log loss 0.3914) and the public-information probe (0.3780, the
  trunk's level; `ai/eval/belief_probe.py`). **Decision:** a gain at least three times the public
  probe's (≤ ~0.35) opens 3b and 3c; less closes the item for both the search and the policy.
* **3b, belief-weighted worlds for the search (no observation change).** The rollout root deals its
  worlds from the belief net's posterior instead of uniformly (importance-weighted or sampled;
  `ts.determinize` takes a per-card weight). Measured as a player: the keyed root with belief
  worlds against the keyed root, strong tier (1810). Adopt by the per-seat rule.
* **3c, a belief input for the policy (the owner's decision).** The compact form is the belief net's
  posterior as an appended `obs_features` block (110 floats, the per-card probability the opponent
  holds it; the base 3,824 unchanged, every checkpoint keeps running). Cost: 110 floats and one
  small forward per step. Precedent: the v2.1 turn-history slice was unread at 80–240M on the old
  architecture, so a *raw* history slice is not the form; a learned posterior is. If approved it goes
  into items 1 and 2's from-scratch runs rather than into a branch, since a saturated policy will
  not learn to read a new input (the Cell A lesson).

### 4. The KL regularisation checks (R codes; 400M branches from A13+R48's plateau)

* **4a, η = 0.** Does the term still earn its place at the plateau? On E3 at 80M removing it cost
  169 Elo as a stabiliser (no cycles found either way); on the current line it is 0.0006 of a
  0.035 policy loss while the gap penalty takes 46% of the gradient.
* **4b, the slow reference, `--ref-update-freq 5000000`** (75 iterations instead of 3) at η 0.1.
  Harmful from scratch on E4 (−168 / −68 at 80M) and the best late arm in the E4 record when
  switched on after 160M (+78 to +108 at 240M); never tried on E7 or from a plateau. E3-31-28
  collapsed under it by KL domination, so `kl_term` and entropy are watched.
* **Read.** Raw and under the keyed root at 80M windows, per seat, against the unchanged control;
  the stability panel. **Decision:** 4a level → the term leaves the recipe and the from-scratch
  runs use Ataraxos's annealed reverse-KL damping instead; 4b positive → adopted as a *late switch*
  in every from-scratch run (on after the first plateau), never from step 0.

### 5. Search-computed advantages for the policy gradient (R code; a build, then one branch)

**Closed as a branch (2026-10-20, P36c):** the root's corrections at entry decisions are reproducible
but not learnable from the observation by any student at feasible scale, and a plateau network cannot
fit even the training departures. If this form runs at all it is from scratch, active, as P36's
verdict describes -- not as a branch from a plateau.

* **What.** At one in eight of the learner's *entry* decisions (card choice, headline; ~3.4% of
  decisions) the keyed root (k = 4, 16 worlds, `earlier`+own) on a **slow frozen snapshot** of the
  learner (refreshed every 50M steps, not with π_ref, to keep the R29 self-referential loop out)
  gives Q(s, a) for the four candidates. The policy gradient at those decisions uses the root's
  advantage, Q(s, a_taken) − mean_a Q, when the taken action is a candidate, and adds an
  expected-advantage term Σ_a π(a|s)(Q(a) − V) over the candidates. **Not** a cross-entropy toward
  the root's argmax: that form failed seven times on near-ties; this one tells PPO which of its own
  moves is better and by how much.
* **Cost.** ~30× the trainer's inference, ~5–10× slower; 400M ≈ a day of one slot.
* **Read.** Raw and searched as usual; the own-search gain (the root over the raw network) is the
  number of interest, expected to shrink if the policy absorbs the root's ranking.

### 6. Prioritised fictitious self-play (R code; one 400M branch)

* **What.** `--opponent-pfsp` (built, `var` weighting, uniform floor), never measured. The pool's
  own note predicts it should hurt in pure self-play by starving the trailing seat of winnable
  games; `opp_pfsp_entropy` is the guard.
* **Read.** Per seat raw and searched; the seat balance. Adopt only if both seats hold.

### 7. Strategy conditioning for diversity (`P35b` first; the build is the owner's decision)

* **`P35b`, the measurement.** How diverse is the policy, and does the lack of it cost strength? The
  setup is one placement in 100% of games; the USSR seat is exploitable by +11.6. Measure the
  strategy entropy of self-play (setups, the regions invested by turn 3, coup and space usage by
  side) against the searched player's and the human corpus's distributions, and the exploit
  margin of a fresh seat-locked exploiter against the current line.
* **The build, if `P35b` says diversity is the missing piece.** AlphaStar's z: a strategy summary
  sampled per game (from self-play's own distribution plus designed openings), given to the policy
  as an appended observation block (~20–30 floats) with a pseudo-reward for following it. An
  observation change; the owner's call; into a from-scratch run only.

## What runs in parallel

One GPU, two training slots, evaluation matches and CPU probes beside them.

| days | slot 1 | slot 2 | CPU / CI | GPU evaluation |
|:---|:---|:---|:---|:---|
| 1–2 | 4a η = 0 (400M) | 4b slow reference (400M) | `P35a` belief probe; `P35b` diversity read; builds: item 5's trainer, 3b's weighted determinize, the token and scaled configs | 80M-window reads of 4a / 4b |
| 2–3 | 6 PFSP (400M) | 1's grown token layers (400M) | builds continue | reads |
| 3–4 | 5 search advantages (400M, ~a day) | 2's grown blocks (400M) | 3b's belief-world root, if `P35a` passed | reads; 3b's strong-tier match |
| 5 → ~19 | **1, the encoder from scratch** | **2, capacity from scratch** | | 400M-window reads of both |

The two from-scratch runs take the GPU for about two weeks, so everything that can change their
recipe or their input runs first: items 4 and 6 set the recipe, `P35a` with the owner sets the input.
Two arms at once give +20% total throughput over one, so the pair costs little over running them in
sequence. Nothing else trains during the flagship fortnight; branches of the flagships' own plateaus
come after.

## Interactions

* **1 ↔ 2.** Both are capacity; they share the control, the recipe and the launch day so their
  plateaus are comparable, and item 2 is item 1's control for "attention" against "parameters".
  Their value depends on the recipe (4, 6) and on the input (3c): a flagship launched before those
  read runs with a recipe known to be improvable, which is why the branches go first.
* **3 → 1, 2.** A belief input is an architecture decision that must be made before the flagships
  launch; it cannot be added to them later without a second fortnight. `P35a` is therefore on the
  critical path even though it trains nothing.
* **3b ↔ 5.** Search-computed advantages use whatever root is deployed; if belief worlds pass, item
  5's root should carry them, so 3b's read precedes 5's launch where the schedule allows.
* **4 ↔ 6.** Both change the update's dynamics (the anchor, the opponents). Never combined in one
  arm; if both pass, one combined 400M arm before either enters the recipe.
* **4 ↔ the gap penalty.** The penalty is 46% of the gradient; a slow anchor pulls against its
  spreading of the policy, and η = 0 removes the only other pull. Entropy and `logit_gap_frac` are
  read beside the usual panel in 4a and 4b.
* **4 ↔ 5.** Item 5's root reads a snapshot of the learner; with the reference refreshed every three
  iterations the R29 loop returns, which is why its snapshot refresh is its own, slow, cadence.
* **5 ↔ the critic.** The root's Q is the critic over playouts; item 5 inherits the critic's
  resolution (P34), so its gain is bounded by what sixteen worlds can rank. It cannot teach what the
  root cannot see.
* **6 ↔ 7.** PFSP concentrates the opponent draw; strategy conditioning widens the population.
  Opposite pressures on diversity; `P35b` is read before either is adopted.
* **Branch gains against from-scratch gains.** The gap penalty helped from a plateau and cost
  150–240 Elo from step 0; the slow anchor the reverse. Every recipe item that passes as a branch
  enters a from-scratch run as a *late switch* unless a from-scratch cell shows it safe early.
* **Grown-in-place reads against from-scratch runs.** Cell A and A14 were active and level: a
  saturated policy sharpens rather than corrects with new capacity. The grown reads of 1 and 2 are
  screens with a known false-negative risk and do not cancel the flagships; a positive grown read
  raises the flagships' prior, a level one changes nothing.

## Decision rules, collected

* Item 1 / 2: adopt a plateau that beats A13+R48's by the per-seat rule raw **and** under the keyed
  root; between them, the better plateau head to head.
* `P35a`: belief gain ≥ 3× the public probe's → 3b runs and 3c goes to the owner; else closed.
* 3b: adopt by the per-seat rule against the keyed root on the strong tier.
* 4a level → the term leaves the recipe; 4b positive by the rule → late switch in the flagships.
* 5: adopt by the rule with the own-search gain not larger than the control's.
* 6: adopt only with both seats holding and `opp_pfsp_entropy` not collapsed.
* `P35b`: a measured strength cost of low diversity (an exploiter margin above the current +11.6,
  or a searched-player distribution the policy cannot cover) → the z build goes to the owner.

## What is not in this plan

Evaluator or critic training on playout labels (P34, closed); policy distillation of search in any
form; the league; human cloning from the uncurated corpus; more search compute as a result.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

**Pair 1, item 4 (2026-10-18): both level.** 400M from A13+R48's 12,400M state against the control's unchanged leg (12,400 → 12,800M), 80M-window SWAs, 1,000 deals a side:

| window | 4a η = 0 (R55) | 4b slow reference (R56) |
|:---|---:|---:|
| 12,400–12,480M | 51.7% | 48.4% |
| 12,480–12,560M | 49.0% | 50.4% |
| 12,560–12,640M | 48.3% | 50.4% |
| 12,640–12,720M | 48.8% | 50.1% |
| 12,720–12,800M | 49.9% (US 46.6 / USSR 53.1) | 49.5% (US 46.1 / USSR 53.0) |
| same, keyed root (16 worlds) | 48.4% (US 46.3 / USSR 50.6) | 49.1% (US 47.0 / USSR 51.2) |

Per seat, see *The per-seat bar* below: level in both seats. Neither arm was unstable (entropy 0.74 / 0.78; 4b's KL 0.016, no domination). By the decision rule, 4a level means the KL term is not earning its place at the plateau, so the from-scratch runs would use annealed reverse-KL damping instead (owner's call before it changes a recipe); 4b level means the slow reference is not adopted as a late switch.

**Pair 2, item 6 (PFSP, R57): level, slightly above.** Raw 51.3 / 49.9 / 49.4 / 49.8 / 51.1% over the five windows; the last window US 49.0 / USSR 53.3 (+7.5 / −4.7 against the control's self-play), under the keyed root 51.1% (US 48.1 / USSR 54.2, about +1 SE). Both seats hold, so by the rule it may be adopted, but nothing here shows it gains: the predicted harm is absent, the gain unmeasured at this size.

**Item 3's ceiling, the oracle root (2026-10-18, owner's request).** A13+R48's SWA (12,730–12,800M)
under the keyed root (k 4, 16 worlds, `earlier`+own) with the oracle's worlds -- every world keeps the
opponent's true hand, only the deck is reshuffled; deck order, dice and an unrevealed headline stay
unknown, and the leaves are read honestly from the mover's side (`rollout:...:keyed:oracle`, bindings
`cf09dc5`) -- against the same network under the honest keyed root, 1,000 deals a side: **57.6%** (US 52.6
/ USSR 62.6; ±1.1, about 7 SE; `data/reports/p35_oracle_root_vs_keyed.json`). So the opponent's hand is
worth **+7.6 points (~+53 Elo) to the root**, against +1.5 / +3.1 for perfect-information MCTS on the E7
shallow models (`../log/P30_search_headroom_e7.md`) and +0.002 AUC to a critic
(`../log/P30_oracle_critic_auc.md`): little for predicting the result, a good deal for choosing the move,
because the honest worlds deal the hand **uniformly** from the unseen cards. The ceiling is loose upward:
exact knowledge is more than any belief can have, and the playout opponent (the same network, true hand)
plays almost exactly as the real one does. What a belief can recover is the share of +7.6 that its
posterior's sharpness allows -- P35a's probe measures that sharpness, 3b measures the recovery.

**P35a, the belief probe (2026-10-18): passes, by the policy-likelihood posterior.**
`ai/eval/belief_history_probe.py` on A13+R48's SWA (12,730–12,800M), 1,024 self-play games at T = 1, read on
held-out games. Besides P33f's predictors, two from the public record: a *history probe* (P33f's input plus
per card the turn it entered the hidden pool and the opponent's draws since) and the *policy-likelihood
posterior* -- uniform hands of the opponent's unseen size from the decider's hidden pool, each weighted by
the network's probability of the opponent's actual decisions this turn, every past position rebuilt with
the hand (exact for the true hand, tested; 99.86% of hypothesised worlds keep the actual move legal, the
rest are excluded by the rules, e.g. an unplayable held scoring card).

| predictor (log loss; prior 0.3839 on these 1,000 positions) | all positions | with an opponent decision this turn |
|:---|---:|---:|
| public probe (P33f) | 0.3707 (+3.5%) | 0.3633 |
| history probe | 0.3720 (+3.1%) | 0.3644 |
| posterior, card and play-mode choices, 256 hands (4,000 positions, own prior 0.3945) | 0.3645 (+7.6%) | 0.3502 |
| posterior, every opponent decision, 256 hands | 0.3530 (+8.1%) | 0.3446 |
| **posterior, every opponent decision, 2,048 hands** | **0.3363 (+12.4%)** | **0.3272** |

* **The history is in the opponent's choices, not in the timing arithmetic** (the history probe adds
  nothing). Conditioning on every decision -- where it coups, realigns and places influence -- is the
  owner's point (the board at each choice); with 256 hands it barely showed (+7.8% against +7.6%) because
  the estimate was sampling-limited: effective hands median 47 of 256 (13% of positions below 10), 369 of
  2,048. With 2,048 the posterior gains 3.6× the public probe's -- **past the rule (≤ ~0.35): 3b and 3c open**
  (3c is the owner's decision).
* Not converged: more hands or a particle filter (resample and perturb after each opponent decision)
  should sharpen it further, and the posterior uses this turn only -- before the opponent's first move
  (4% of positions) it is the prior. Carrying the particles across the deal (held cards) extends it.
* The learned combination (history probe + posterior logit, fitted on ~2,000 positions) overfit; not read.

**3b, the belief root (2026-10-19): +3.4, about 45% of the oracle's ceiling.** `ai/search/belief.py`
(commits `1199961`, `ed4b852`): the tracker sees every decision of its games through the tournament's new
hooks and, at a root decision, weights 2,048 uniform hands by the network's likelihood of the opponent's
decisions this turn (all of them -- P35a's best form), then resamples the root's 16 worlds from them; the
likelihood rows are one observation per opposing decision with each hidden card's own-hand / deck slot set
on the device (exactly the engine's rows, tested). A13+R48's SWA (12,730–12,800M) under the keyed root with
belief worlds against the same network under the honest keyed root, 1,000 deals a side: **53.4%** (US 47.4
/ USSR 59.5; ±1.1, about 3 SE; `data/reports/p35_belief_root_vs_keyed.json`) -- **about +24 Elo**, against
the oracle root's 57.6% (US 52.6 / USSR 62.6). The keyed root's own per-seat self-play bar is not measured;
against the control's last-window self-play (41.5 / 58.0, an outlier -- *The per-seat bar*) both seats are above it. Cost: ~6 s a game for the belief on the
current code (the 2,000 games took 4 h 15 min beside A18 against ~50 min for the keyed root's), dominated
by per-decision host-device transfers that do not batch across games. Not yet used: earlier turns (before
the opponent's first move of a turn it is the uniform prior) and more hands (P35a: not converged at
2,048).

**3b against humans: the belief is tuned to opponents that play like our network (2026-10-19).**
`ai/eval/belief_human_probe.py`: the same posterior on the human corpus (256 games converted, 36,048
positions, true hands from the converter), card and play-mode choices only (the logs order a play's points
arbitrarily), 256 hands, at several likelihood temperatures. Prior log loss 0.4595:

| likelihood temperature | 1 | 1.5 | 2 | 3 | 5 | 8 |
|:---|---:|---:|---:|---:|---:|---:|
| log loss | 0.5131 | 0.4755 | 0.4617 | 0.4533 | **0.4518** | 0.4534 |
| vs the prior | −11.7% | −3.5% | −0.5% | +1.3% | **+1.7%** | +1.3% |

On self-play the same settings gain +7.6% at T = 1 (P35a). Against humans the network's peaked policy is a
wrong model: a human who plays a card our network would not is read as not holding the cards it would have
played, confidently and wrongly. Flattened (T ≈ 5) the belief barely beats the prior. So the measured 3b
gain (+3.4 against the keyed root) is against an opponent the likelihood models exactly; one belief mode
for the leaderboard and the browser (owner) has to choose its likelihood for both.

**One belief for both opponents: the temperature from the opponent's play.** Each temperature of a grid
(1, 1.5, 2, 3, 5, 8) weighed by how well it explains this opponent's choices so far in the game (each
finished turn's evidence log mean_k prod pi_T(a | hand k), summed, plus the current turn's) and the
per-temperature posteriors mixed by those weights -- Bayesian model averaging over T, uniform prior, at
almost no cost (one network pass; the softmax per T). Same probe, same positions as above (`--source
selfplay`: 256 of the network's own games at T = 1):

| | prior | T = 1 | best fixed T | **model average** | its weight on T = 1 / 1.5 / 2 / 3 / 5 / 8 |
|:---|---:|---:|---:|---:|:---|
| self-play (log loss) | 0.3912 | 0.3607 (+7.8%) | 0.3582 (+8.4%, T 1.5) | **0.3593 (+8.1%)** | 0.63 / 0.24 / 0.08 / 0.02 / 0.01 / 0.01 |
| humans (log loss) | 0.4595 | 0.5131 (−11.7%) | 0.4518 (+1.7%, T 5) | **0.4558 (+0.8%)** | 0.03 / 0.09 / 0.31 / 0.44 / 0.10 / 0.04 |

It finds the opponent unasked: on the network's games it keeps the whole gain, on humans it avoids the harm
(a small gain instead of −11.7%). This is the belief mode for the leaderboard and the browser alike.

**(2026-11-01: every belief-root number in this file and in P36 predates a fix: as the USSR, the root's
worlds always held the US's unrevealed headline, because the tracker conditioned on that choice. Read them
as upper bounds. `bindings/belief.cpp` no longer does.)**

**The shared belief root, re-measured (2026-10-19): 55.0%.** The belief as built for both the leaderboard
and the browser (`bindings/belief.hpp`, commits `38da571`, `abfbf43`: the temperature model-averaged from
the opponent's choices, draws keyed by the position, 2,048 hands, every opposing decision of the turn),
A13+R48's SWA under the keyed root with belief worlds against the same network under the honest keyed
root, 1,000 deals a side: **55.0%** (US 49.2 / USSR 60.9; ±1.1, about 4.5 SE;
`data/reports/p35_belief_avg_root_vs_keyed.json`) -- **about +35 Elo**, against 53.4% for the first build
(T = 1, numpy draws; the difference is about 1 SE) and the oracle root's 57.6%: two thirds of the ceiling.
The player id is `…~rollout(…,chance=keyed,belief=2048)`, the page's `search=rollout:…:keyed:belief2048`.

**Against a network the belief does not model: the same edge (2026-10-19).** The belief root and the
honest keyed root (both A13+R48's SWA) each against the R34 soup
(`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`) under the keyed root, 1,000 deals a side:

| | overall | US | USSR |
|:---|---:|---:|---:|
| belief root vs the R34 soup's root | 52.8% | 56.3% | 49.3% |
| honest keyed root vs the R34 soup's root | 47.5% | 49.6% | 45.4% |
| **the belief's edge** | **+5.3** | +6.7 | +3.9 |

±1.1 each, the difference about 3.3 SE (`data/reports/p35_{belief,honest}_root_vs_r34soup.json`): the same
+5 as against its own network, in both seats. With the human probe (the temperature average avoids the harm
there), the belief is not tuned to its own network's play.

**The per-seat bar (2026-10-19, corrected).** The window reads were first read per seat against one bar,
the control's self-play in its last window (US 41.5 / USSR 58.0), which made every arm look "+5 as US, −5
as USSR". The control's self-play per seat moves with the window, and the last window is the outlier. Each
window's exact control SWA against itself on the reads' deals (raw, 1,000 a side;
`data/reports/p35_ctrl_selfplay_*.json`), and each arm's seat rates minus it (± ~1.6 per seat):

| window | control self-play (US / USSR) | R55 η 0 | R56 slow ref | R57 PFSP | A18 tokens |
|:---|---:|---:|---:|---:|---:|
| 12,410–12,480M | 45.7 / 53.1 | +2.7 / +1.9 | −0.4 / −1.5 | +0.3 / +3.5 | +0.6 / +0.5 |
| 12,490–12,560M | 45.8 / 52.4 | +1.4 / −1.5 | +0.8 / +1.8 | −0.6 / +2.1 | +0.6 / +0.3 |
| 12,570–12,640M | 47.4 / 51.6 | −3.7 / +1.3 | +2.4 / −0.6 | 0.0 / −0.3 | −2.1 / +0.3 |
| 12,650–12,720M | 46.4 / 52.6 | −0.7 / −0.8 | −0.3 / +1.5 | +0.4 / +0.1 | |
| 12,730–12,800M | **41.5 / 58.0** | +5.1 / −4.9 | +4.6 / −5.0 | +7.5 / −4.7 | |

In windows 1–4 no arm shows a seat pattern (differences within noise, signs mixed). The +5 / −5 appears only
in the last window, in all three arms alike: it is the control's 12,730–12,800M SWA that plays a different
seat balance (unusually strong as USSR, weak as US against itself), not the arms.

**Pair 2, item 1 (A18): level.** Raw 50.0 / 49.5 / 48.6 / 49.0 / 49.0% over the five windows, level per seat against each window's own control self-play, and 50.1% under the keyed root. A level grown read does not cancel the from-scratch run (above).

**Pair 2, item 1, the schedule.** Item 1's grown token layers (A18) need the whole GPU (~9× A8's compute; 4.7k steps/s beside R57, which fell from 69k to 10–20k), so A18 runs alone after R57 instead of beside it.
