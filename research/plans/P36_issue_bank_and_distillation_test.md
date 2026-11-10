# P36 — the issues bank, and whether search distillation is limited by signal or by capacity (2026-10-19)

**Origin.** The running session's P36 proposal (its reply to the owner, 2026-10-19; not a file) grouped
the owner's issue list into causes and proposed an issues bank, expert iteration as the compounding
engine, rule-derived search candidates, planning inputs and a turn-level root, delayed-consequence
value targets, a decided-position tiebreak, and capacity from scratch. This file is the plan as
reviewed and decided by the owner: what stays, what changes, and the test that comes before any
generation of distillation is spent.

**Decisions (owner, 2026-10-19).**

* **Pure search before belief.** The honest keyed root is +180 over the raw network; the belief root is
  +35 over the honest root. The policy's input today determines the honest root's expected choice
  (uniform draws over the unseen cards given the observation), so that teacher is representable by a
  memoryless policy; the belief root's edge is not, until a belief input exists. The belief head
  (P35 3c) goes to the reserve; 3b stays a product gain.
* **Expert iteration is not launched as a generation.** Seven policy-side forms and one value-side form
  (R51) were null or harmful; the proposal did not say what would differ. Before any generation, a
  one-day offline test (below) says whether the limit is the targets or the student.
* **Capacity from scratch starts now** (P35 items 1 and 2), not after richer targets: a bigger student
  is a precondition for distillation, not a follow-up, and the runs are independent of this plan.
* **Dropped:** delayed-consequence value targets (exact endgames, region-scoring auxiliaries) -- P34
  closed the value track; the region swings were in D1's labels and bought nothing. To the reserve.
* **No code by the planning session; the running session builds and runs.**

## The issue groups (from the proposal, kept as the bank's structure)

1. **Options never tried**: coups at DEFCON 2 under Nuclear Subs; Wargames declined; events at the
   wrong moment; the choice inside Aldrich Ames. Self-reinforcing avoidance; the root's top-4 never
   holds a move the policy gives ~0.
2. **Delayed, conditional value**: Europe control; the opponent's possible scoring cards; holding a
   3-Ops card against an unknown Blockade.
3. **No plan for the turn's cards**: high-Ops early and scoring late; Five Year Plan before the last
   round; Five Year Plan with Formosan Resolution and NATO in hand.
4. **Drift in decided positions** (critic below 5% or above 95%): advantages near zero, won
   positions not converted (the trainer's decisive-win take rate ~76%).
5. **Opening quirks**: Yugoslavia 3 then Independent Reds; the setup locked at probability 1.

## `P36a` -- the issues bank (CPU; for the owner's review before it is used)

* **What.** For each group, positions with a verified right answer, built programmatically where the
  owner's examples allow (a Subs coup, a Europe trade, a Blockade hold, the Five Year Plan / NATO /
  Formosan hand, the Yugoslavia + Independent Reds opening) and scored for every checkpoint and
  searched player, the way `ai/eval/banks/blunders_E7.jsonl` is.
* **The verification rule, from P33c's collapse.** A bank's "right answer" is never the network's own
  greedy continuation on a multi-step plan -- that is how the leak bank certified its own judge's
  follow-up. Labels come from the exact solver where the game ends in the turn, the oracle root where
  the hand decides, long keyed paired playouts with the follow-up's bias reported against the exact
  slice, and **the owner's judgment** for the positions that are the point of the bank. Each row records
  which.
* **Deliverable for review.** The candidate position set with its proposed label source per row, before
  any labelling runs. The owner chooses which situations go in.
* **Use.** Every arm in this plan and in P35 is scored on it; self-play Elo cannot see a leak both
  seats share.

## `P36b` -- measure the teacher first (GPU, ~1 h): the pick's reproducibility

* **What.** The honest keyed root (k 4, `earlier`+own, `chance=keyed`) on A13+R48's SWA, 3,000 entry
  decisions from its self-play, searched **three times each** at 16, 64 and 128 keyed worlds; per
  world count: the share of decisions where it departs from the policy, the share of one run's
  departures that recur in another, the share with the same move, and the three-run agreement
  (`ai/eval/rollout_consistency.py` extended to world counts; at 16 shared-stream worlds the record
  has 62–64% / 57–58% / 188 of 432).
* **Why first.** This is the target's signal-to-noise. P34's reliability (0.41 at 16 keyed worlds) was of
  outcome means, the wrong quantity for a policy target; the pick's agreement is the right one and is
  unmeasured above 16 worlds. **The line:** the world count whose three-run agreement on departures
  reaches ~90% is the one the targets are built at; if 128 does not reach it, the test below still runs
  at 128 and reports the agreement as the ceiling.

## `P36c` -- targets and students: signal or capacity (offline, ~a day)

The search's own seed-to-seed agreement on held-out positions is the ceiling for any student's
held-out agreement with it. Three readings per cell separate the causes.

* **Targets.** The keyed root's improved policy (`softmax(logit + σ(completed Q))` over the candidates,
  R32's form) on 200k entry decisions of A13+R48's self-play, at 16, 64 and 128 keyed worlds (the
  128-world root is ~7k network rows a decision; the set is ~30 min on the GPU). Held out by game.
* **Students, trained to convergence** -- not the two epochs at 1e-4 of the earlier offline rounds, which
  underfit by design and never separated these cases: the current network fine-tuned on its policy
  path; the same shape from scratch; a 5–10× network (the deep or token shape of P35). Supervised on the
  policy heads, the value path frozen; under an hour each. Each student an A+R pair.
* **Three numbers per cell:** (1) the training-set fit -- the share of the teacher's departures that are
  the student's argmax on positions it trained on; (2) held-out agreement with the teacher; (3) the
  teacher's own seed-to-seed agreement on the same held-out positions (`P36b`).
* **Reading the cells.**
  - fit rises with world count at a fixed student → **the targets were the limit**;
  - the big student fits the training departures and the current one does not, at convergence →
    **capacity**;
  - no student fits the training set at convergence → the targets are not a consistent function of
    the input (noise, or hidden information) → **no student can help**;
  - every student fits the training set and none generalises → position-specific corrections with no
    rule (the depth log's diagnosis) → only the number of targets helps, the expert-iteration cost
    regime;
  - a student's held-out agreement at the teacher's self-agreement → that student is at the signal
    ceiling; capacity is pointless until the targets improve.
* **Games** only for cells that pass: the student greedy against A13+R48's SWA, and under the keyed
  root (the own-search gain should shrink if the ranking was absorbed), per seat.
* **Then, and only then,** the expert-iteration design: targets at the measured world count, the
  advantage form over the candidates (P35 item 5) rather than a cross-entropy toward the argmax, the
  student size the test chose, as one from-scratch programme.

## The remaining arms (after `P36a`–`P36c`; each scored on the bank)

* **Rule-derived candidates in the root** (a search change): coups where DEFCON allows them and always
  under Nuclear Subs; events that score at once or reach 20 VP; for each card, play now or hold. The
  k = 6 null does not apply (that was the policy's own ranks 5–6, not moves outside its support).
  Cautions: the critic has never valued a post-Subs coup, and the z2 rule may never clear on a
  candidate the critic misprices; the exploration half in training is a separate arm, and the fixes
  reach the network only through a distillation that `P36c` has shown can work.
* **A root over the turn's card order**: search the sequence of the hand's plays across the action
  rounds, not one decision at a time -- the most promising search-approach change in the proposal,
  with no precedent here. Design first; the bank's group-3 rows are its read.
* **Planning inputs.** The observation already carries the turn (`globals[6]`), the action round
  (`globals[7]`) and both hand sizes, so rounds left and surplus cards are a subtraction away; the
  expert review's two-to-three-float convenience block is the owner's call, low prior as a cause.
* **The decided-position tiebreak**: a small VP-margin or safety term where the critic is past 95% or
  below 5% (the plain VP shaping was level in P30); one 400M arm, judged on the bank's group-4 rows
  and the decisive-win take rate.
* **Capacity from scratch**: P35 items 1 and 2, launched now, the GPU being free after A18.

## Sequencing and parallelism

| now | GPU | CPU | owner |
|:---|:---|:---|:---|
| day 1 | `P36b` (1 h), then `P36c`'s targets (30 min), then P35's two from-scratch flagships launched in both slots | `P36a` candidate positions and label sources | reviews `P36a`'s set |
| day 1–2 | `P36c`'s student fits interleaved with the flagships' first windows (each < 1 h) | the turn-level root's design; the rule-candidate build | approves `P36a`; decides on the convenience block |
| after | the cells that pass `P36c` go to games; the tiebreak arm and rule candidates follow, scored on the bank | | |

`P36b` and `P36c` are on the critical path of any distillation; `P36a` is on the critical path of every
read. The flagships are independent of both and take the slots as soon as `P36b`/`P36c`'s GPU hour
is done.

## Decision rules

* `P36b`: targets are built at the smallest world count with ~90% three-run departure agreement;
  below that at 128, the agreement is reported as the ceiling and `P36c` runs anyway.
* `P36c`: the cell readings above decide the cause; a student goes to games only if its training fit
  is above ~90% of the teacher's departures and its held-out agreement is within 5 points of the
  teacher's self-agreement.
* Expert iteration is designed only after `P36c` names a student and a world count; never from the
  proposal's form.
* Every other arm: the per-seat rule raw and under the keyed root, plus the bank's rows for its group.

## Reserve (from this plan)

The belief input (P35 3c) pending pure search; delayed-consequence value targets; the convenience
planning block.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

**`P36b` (2026-10-19): no world count reaches 90%; the targets are built at 128, ceiling 72.7%.**
`ai/eval/rollout_consistency.py --keyed --stop earlier --leaf own --entry --worlds 16 64 128 --positions 3000`
on A13+R48's SWA (12,730–12,800M): 3,000 entry decisions (card choices) from 256 greedy games, each searched
three times per world count (`data/logs/temp/p36b.out`):

| keyed worlds | departs | of a run's departures, another departs too / same move | stable in all three | improved policy: TV between runs (median) / top move agrees |
|:---|---:|:---|:---|:---|
| 16 | 17–18% | 64–67% / 57–60% | 237 of 518 (**45.8%**) | 0.214 (0.055) / 77% |
| 64 | 27% | 79–81% / 73–75% | 519 of 808 (**64.2%**) | 0.164 (0.018) / 83% |
| 128 | 32% | 85–86% / 80–81% | 701 of 964 (**72.7%**) | 0.135 (0.003) / 86% |

* 16 keyed worlds reproduce the record's 16 shared-stream numbers (62–64% / 57–58%, 188 of 432): keyed
  chance did not make the pick steadier at equal worlds.
* The departure share itself grows with worlds (17 → 27 → 32%): the z2 rule clears more leads as their
  standard error shrinks, so each larger set adds borderline departures; the stable ones carry larger
  leads (+0.045 against +0.031 at 128).
* The improved policy (the P36c target) is steadier than the pick: its median TV between seeds falls to
  0.003 at 128, top move 86%.
* By the rule, P36c builds at 128 and reads every student against 72.7% (the departures) and 86% (the
  target's top move) as the teacher's own ceiling. Cost: 49 decisions/s at 128 worlds eager (CUDA graphs
  slower and out of memory), so the 200k-decision set is ~2 h at the three world counts, not ~30 min.

**Caveat added the same night: the teacher has a horizon bug, so this reading is provisional.** The
`stop=earlier, leaf=own` root counts its 4 action-round boundaries from different points for different
candidates of one decision: a move that ends the mover's action round inside its own settling does not count
that boundary, a move whose round ends later in the playout (further choices: Ops placements, an event's
choices, the opponent placing its event's influence) does. Those candidates are read one action round
earlier. Found on the owner's replay (`data/replays/us_search_decol.json`, turn 3 AR3: the root plays the
USSR's Decolonization for its event at +0.10 over space; read at the turn's end for every candidate, space
leads by +0.12). At card choices it shifts every card that resolves without a choice (all scoring cards)
by an action round against the rest, so P36c's targets carry artifacts there. The fix landed as the
`aligned` option (below, with the search-mode sweep); P36b/P36c are still to be re-read on it.
The 2M-target data-scaling run (test 1) was stopped for the same reason.

**`P36c` (2026-10-19): the rollout root's corrections do not transfer to unseen positions -- not by
capacity, not by world count, and barely by data.** Targets: `tools/distill_targets.py`, 200,000 entry
decisions from 2,000 greedy games of A13+R48's SWA (12,730–12,800M), 20,675 held out by game, the keyed root's
improved policy at 16 / 64 / 128 worlds, the held-out rows searched twice (2 h 40 min; the raw states
re-collected on the rebased build, below). Students by `tools/train.py --mode policy-fit` (policy heads, value
frozen, to convergence: 60 epochs). "Departures": positions where the target's top move is not the network's
argmax (41.4%). Held-out readings on the 8,540 held-out departures:

| student | training fit (departures) | held-out (departures) | held-out overall | teacher's self-agreement |
|:---|---:|---:|---:|---:|
| A13+R48 fine-tuned, lr 3e-4 (`…+A13@12800M+A19-R58`) | 21.9% | 15.2% | 58.8% | 79.1% / 86.0% |
| fine-tuned, lr 1e-3 (`…+A19-R62`) | 40.9% | 20.2% | 57.4%* | 79.1% / 86.0% |
| same shape from scratch (`E7-A19-R59-S44`) | 96.4% | 24.7% | 42.8% | 79.1% / 86.0% |
| deep shape, 8.7x (`E7-A20-R59-S44`) | 98.9% | 23.6% | 44.9% | 79.1% / 86.0% |
| same shape, 16 / 64 worlds (`R60`, `R61`) | 92.4 / 94.8% | 25.1 / 24.5% | | 57.4 / 72.4% |
| same shape, 25 / 50% of the games (`R63`, `R64`) | 94.6 / 95.9% | 23.2 / 23.4% | | 79.1% |
| fine-tuned lr 1e-3, 25 / 50% (`R65`, `R66`) | 39.9 / 40.5% | 16.6 / 18.6% | | 79.1% |

\* from the move breakdown below. **Baselines on the held-out departures:** uniform over the other legal moves
26.6%; **the network's own second choice is the teacher's move 51.3%** of the time; the current network 0% by
construction (overall 58.6%).

* **Capacity is not the limit**: the 8.7x student fits the training departures as the same shape does (98.9%
  against 96.4%) and generalises no better (23.6% against 24.7%).
* **The targets' noise is not the limit**: held-out agreement is flat across 16 / 64 / 128 worlds (25.1 / 24.5
  / 24.7%) while the teacher's own agreement rises 57 → 72 → 79%.
* **The from-scratch students memorise** (96–99% training fit) and are at chance on held-out departures (24.7%
  against uniform's 26.6%) -- but they never learned the game from 180k positions (42.8% overall against the
  network's 58.6%), so their numbers say nothing about transfer. The fine-tuned student is the reading that
  counts.
* **The fine-tuned student does not transfer the corrections**: on held-out departures it keeps its own move
  64.1%, takes the teacher's 20.3%, another move 15.6%; on held-out positions where the teacher agrees with
  the network it *leaves* that move 16.8% -- net, slightly worse than the network (57.4% overall agreement
  against 58.6%). The teacher's margin does not matter (85.7% of departures have a margin over 0.6; agreement
  ~20% in every band).
* **Data barely helps**: 25 / 50 / 100% of the games give 16.6 / 18.6 / 20.2% -- about +1.8 points a
  doubling. Reaching even the network's own second choice (51%) would take ~16 more doublings.
* **By the decision rules no student goes to games** (training fit < 90% for the fine-tune; held-out far
  from the teacher's self-agreement for all). By the cell table this is the "fits but does not generalise"
  cell, but its remedy -- more targets, expert iteration's cost regime -- is ruled out by the slope: **an
  expert-iteration programme on this teacher's entry-decision corrections is not viable** at any affordable
  number of targets. The corrections are reproducible given the position (79%) but are not a function the
  network learns from the observation at this scale: each is a position-specific consequence of lookahead.
* **The raw-state bug found on the way**: the dataset's states were written by the pre-rebase build; main's
  `c9e5689` added `GameState` fields with the behaviour digest unchanged, so they decoded as other positions
  (the first fit's step-0 agreement was 1.1%). Re-collected on this build (games and argmax identical); raw
  state datasets are now read only by the build that wrote them (`712381e`; P34's D1/D2 included).

**`P36a` (2026-10-19): candidate set delivered for review.** 954 rows over 18 patterns in the five groups,
mined from 2,000 greedy games of A13+R48 (`ai/eval/issue_bank.py`), two patterns constructed from real
positions (Nuclear Subs set in effect; NATO and Formosan Resolution swapped into a Five Year Plan hand),
each row with a workbench link and its proposed label source -- `ai/eval/banks/issues_P36a_candidates.jsonl`,
the review page [`../log/P36a_issue_bank_candidates.md`](../log/P36a_issue_bank_candidates.md). Nothing is
labelled until the owner chooses the situations.


**The rollout root corrected, and the search-mode sweep (2026-10-19/20; the owner's question: the optimal
search mode).**

*The bugs.* (1) **The horizon was counted from different points for different candidates of one
decision** (found on the owner's Decolonization replay, `data/replays/us_search_decol.json`, turn 3 AR3):
a move that ends the mover's action round inside its own settling (space, a decision-free event) did not
count that boundary, one that leaves choices in the round (Ops points, an event's choices) did, later in
the playout -- so those were read an action round earlier. The deployed root played Decolonization for its
event over space at +0.10; read at one point it trails by 0.12. (2) **The headline counted 1 to 3
boundaries** by how its events resolved (an event that stops for a choice sets the phasing side to its
owner at a decision), decided at a headline choice by the candidate itself. (3) **Space Race box 4**: the
side holding Man in Earth Orbit alone sees the opponent's headline before choosing, but every world forgot it and
asked it again. Fixed as two opt-in options so every measured player keeps its meaning and the E7
behaviour digest is unchanged (measured): `aligned` (every candidate's horizon counted from the end of the
mover's own round, the headline one boundary; `bef6b4c`, `eedf70e`) and `box4` (the revealed headline kept
in every world and every belief hand). Player id `…,aligned=true,box4=true)`, spec `:aligned:box4`, in both
builds. Under the deployed root (150 positions each, A13+R48's SWA) the candidates' leaves were at different
points of the turn in 27% of worlds at the first headline choice (a full action round) and 7% at the
second; aligned: 0.0% at the second; at the first the 19% left (and ~11% at action-round card choices) is
leaf=own's "the mover's next decision" falling in the opponent's half of a round on some paths -- the
playout's course, not the count. The box-4 case is 8.5% of headline choices in self-play, in half the
games; no observation slot shows the revealed card (a zero-float option is with the owner).

*The sweep.* Each mode on A13+R48's SWA (12,730–12,800M), keyed, k = 4, 16 worlds, z2, against the deployed
root (`4:16:4:z2:earlier:own:keyed`, unaligned), 1,000 deals a side (`data/reports/p36_mode_*`,
`p36_aligned_box4_root_vs_keyed`; minutes are wall time with three matches sharing the GPU). With `aligned`
and leaf=own, horizon 1 reads at the mover's next round, 2 two rounds on, 4 three, 6 four.

| mode (all `aligned`, `box4`) | overall | US | USSR | min |
|:---|---:|---:|---:|---:|
| horizon 4, stop=earlier | **52.3% ± 1.1** | 50.3 | 53.3 | 58 |
| horizon 2, stop=earlier | 51.6% ± 1.1 | 48.9 | 53.7 | 59 |
| horizon 6, stop=earlier | 51.1% ± 1.1 | 47.1 | 54.2 | 79 |
| horizon 4, no turn cap (stop=boundaries) | 51.1% ± 1.1 | 48.9 | 52.2 | 75 |
| stop=turn (the next turn, after the deal) | 52.6% ± 1.1 | 48.0 | 56.3 | 97 |
| horizon 1, stop=earlier | 47.1% ± 1.1 | 44.2 | 49.2 | 47 |
| horizon 4, rule z1 | 52.4% ± 1.1 | 49.0 | 55.0 | 103 |
| horizon 4, k = 6 | 53.1% ± 1.1 | 49.6 | 55.7 | 117 |
| **horizon 4, belief worlds (2,048 hands)** | **56.6% ± 1.1** | 51.5 | 60.3 | 155 |

Reads. **The correction is worth ~+16 Elo** (52.3%, ~2 SE). **The rule and k no longer matter**: z1 is
level with z2 (52.4%) -- the conservative rule was guarding against biased leads too -- and k = 6 is +0.8
points for ~1.5x the search. **Belief adds on top**: 56.6% (~+46 Elo), against 55.0% for the belief root on
the unaligned count; the two gains roughly add. **Depth flattens from two rounds on**:
horizons 2, 4, 6, the uncapped 4 and the turn-end stop are all level (51–53%); horizon 1 is −20 Elo against
the deployed root and −36 against the corrected 4. The old sweep's horizon 1 (−170, R32) was mostly the
bug, which is largest there (Ops candidates read before the opponent moved, the rest after). Reading past
the deal adds nothing, at up to 1.7x the time. **The mode: horizon 4, stop=earlier, leaf=own, keyed,
aligned, box4** (52.3%); horizon 2 is level and cheaper. **Belief is not adopted** (owner, 2026-10-20): how it
plays against humans is hard to check, and it is a one-time gain where continuous improvement is wanted; its
32-world leaderboard entry was stopped before any pairing finished and unregistered.

*On the leaderboard (E7, strong tier, 1,000 deals a side).* The corrected keyed root at 32 worlds,
`…+A13@11920..12000M~rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed,aligned=true,box4=true)`:
**1844 ± 4, second on E7** -- +12 over the same network's 32-world root on the old count (1832) and above its
64-world one (1839); the R34 soup's 64-world root (1852) stays first. By opponent, against the old 32-world
root on the same network: the R34 soup's rollout root **60.3%** (57.6%), the R34 soup 74.1% (73.2%), A13+R48
75.2% (73.7%), the heads soup 74.1% (73.7%); against greedy players both are near the ceiling, and the
searched opponent separates them.

## Verdict (2026-10-20)

* **Search distillation as a correction is closed, by a test that separated the causes.** `P36c`: the
  keyed root's entry-decision corrections are reproducible given the position (teacher self-agreement
  79% on held-out departures at 128 worlds) but no student learns them from the observation. Held-out
  agreement is flat across 16 / 64 / 128 worlds while the teacher's rises (57 → 79%), so **target noise
  is not the limit**; an 8.7× network memorises the training departures (98.9%) and generalises at
  chance (23.6%), so **capacity is not the limit**; data scaling is ~1.8 points a doubling against a
  59-point gap; the teacher's move is the network's own second choice 51% of the time and a student
  picks it 20%. The "fits but does not generalise" cell with its remedy ruled out. **Item 2 is closed in
  the proposal's form; P35 item 5 (search-computed advantages) is closed as a branch by the same
  result.** No generation of expert iteration is spent.
* **What the test did not cover, and the one form left.** Both from-scratch students failed to learn
  the game from 180k positions (overall agreement 43–45% against the network's 58%), so their chance
  generalisation is confounded; and the fine-tuned students' inability to fit even the *training*
  departures (22–41% against 96–99% from scratch) is the signature of a network that has lost
  plasticity, which the literature on long-trained networks predicts (see the precedents below). What
  was never tested is search targets **inside a from-scratch RL run** -- active, the learner acting with
  the root at a sparse fraction of its own decisions (the R54 form of P33). **Approved by the owner
  (2026-10-20), in the gated form:** not from step 0 (a random network's root is noise at four times the
  cost); search joins the rollouts the first time a 200-game check, run every 40M steps, has the root
  beating its own raw snapshot by ~55% -- "search is now worth something to this network", a rule that
  transfers to any architecture. Then 1 in 64 entry decisions: the root's move played as environment
  (no policy gradient on it) and the improved policy as a soft target there; the root searches a
  lagged snapshot of the learner against the R29 loop, refreshed every 10M until 400M and every 50M
  after. Cost ~4× a plain run per step once joined. **The bounded test:** the heads run's *exact* flags
  (A8, the 4,096 minibatch it trained at, the gap penalty off as it was) from scratch to 2,000M, so the
  gated search is the only difference from its control -- the 16k minibatch would add a second variable
  worth +15–25 Elo on its own (R47) -- read at matched steps against the heads run's own trajectory at 400M, 800M and
  2,000M (per seat, raw and under the keyed root), continued only if ahead; if the gate never opens,
  that is the answer at no search cost.
* **The from-scratch active test failed (2026-10-22, `E7-A8-R69-S44`).** Against the heads run at
  matched steps, raw / under the corrected keyed root: 400M 39.8 / 38.5%, 800M 42.7 / 43.0%, 2,000M
  **42.2 / 40.1%** (per seat −4.7 / −10.3 at 2,000M) -- behind at every read, closing to 800M and not
  after. The gate opened at the first check (the root beat its raw snapshot 87% at 40M), so the test was
  active search from 40M; the run carried the R29 signature throughout (entropy 0.60 against 0.50, a
  lower pool win rate), with a self-referential soft target from a lagged snapshot at ~270× a normal
  row's gradient. **Search in the loop is closed in this form too**; search stays off the flagships.
  The one variant not run is the root's move played with **no** cross-entropy target (the pure active
  form); it goes to the reserve with its cost (~6 h from scratch to 400M).
* **Launch order (owner, 2026-10-20), after the running session's current report is generated.**
  Slot 1: the gated search test on the A8 recipe from scratch (~a day). Slot 2, at the same time:
  P35 item 2, the scaled conventional architecture from scratch (~3× A8 per step), with the same
  40M gate check running from its start so that search joins it the moment the test passes and its
  own gate opens -- no relaunch. Slot 1, when the test finishes: P35 item 1, the encoder from scratch
  (~9× A8 per step), **without** search unless the test's gain is large enough to pay 4× more. All
  three read per seat, raw and under the corrected keyed root, at 400M windows; the flagships to
  their own plateaus by the plain line's rule. The search test is cheap and its answer changes the
  flagships' recipe; the flagships' answers change nothing about it, which fixes the order.
* **Precedents for "fails as a correction, works from scratch".** In this project: the card-conditioned
  heads -- level on strength as 400M branches (E7-30/31/32-44), then from scratch (E7-33-44 → A8) level
  to 2,000M, 53.9% ahead at 4,000M and a plateau 3,000M later than the plain line; and the branch-vs-
  scratch reversals in both directions (the gap penalty +7.6/+2.1 late, −150–240 from step 0; the slow
  reference the opposite). In the literature: warm-started networks generalise worse than retrained ones
  on the same data (Ash & Adams 2020); loss of plasticity in long-trained networks, with last-layer
  resets or continual backprop as the remedy (Nikishin et al. 2022, Lyle et al. 2023, Dohare et al.
  2024); and the tandem effect -- a network trained passively on another agent's decisions, even the same
  data, underperforms the active learner (Ostrovski et al. 2021), which is the offline-distillation
  failure here in its general form. None of these says the from-scratch form *would* work; they say the
  correction form was expected to fail and does not settle the question.
* **Belief not adopted** (owner, 2026-10-20): hard to check against humans, a one-time gain; to the
  reserve with the belief input.
* **Open from this plan:** `P36a` awaits the owner's review; the rule-derived candidates and the
  turn-order root are search-side product gains, built when wanted; the corrected keyed root (aligned,
  box4) is the deployed mode, 1844 at 32 worlds. **The base-model bet left is P35's two from-scratch
  flagships**, independent of everything above; the GPU is free.

**Launched (2026-10-20, 19:42 UTC), after the corrected root's behaviour report** (8,192 games, owner;
its leak census is paused at 1,032 of 1,500 self-play games, resumable). Slot 1, `E7-A8-R69-S44` (code
R69): the heads run's exact flags -- the flag diff against `E7-A8-R1-S44` is the active-search spec and the
budget only, no box-4 bit (owner) -- with the gate (`ai/training/active_search.py`, `c21f47f`). Slot 2,
`E7-A21-R70-S44` (codes A21, R70): A8 with 512 projections, fusion 768, 4 residual blocks, the box-4
observation bit (`HEADLINE_REVEAL`, `726b9d5`: no width, the opponent's revealed headline shown to the
box-4 holder; a match against a model without the bit gives each side its own view), R47's 16,384 minibatch,
the gate checks only. Built for this: the root's moves are batched by step (a search step with probability
1/64 plays all of that step's learner entry decisions) -- one at a time an open rollout cost 40x a plain one
(200 s against 5 s; the root's cost barely grows with the positions searched together, 2.1 s for one and
3.5 s for forty), batched ~2-3x; the behaviour digest's feature paths are pinned to OPS_BUDGET so a new bit
is not a new epoch.

**The gate opened at the first check, 40M, in both runs (2026-10-20).** The root against its own raw snapshot,
200 games: `E7-A8-R69-S44` **87.0% ± 2.4** (US 90 / USSR 83), `E7-A21-R70-S44` **92.0% ± 1.9** (US 87 / USSR
97), against the 55% threshold -- at 40M search is already worth 37–42 points to the network, so the gate
filters nothing and the test is active search from 40M. Both runs' own checks had failed to start (the child
process had no path to the engine; fixed `133e78f`); the two were played by hand from the saved 40M gate
snapshots with the trainer's command, written into the resume states, and both runs resumed in place from
40.04M (~3.5M steps repeated; the metric lines past 40M kept beside as `training_metrics.jsonl.before_resume_40M`).
Slot 1 then plays ~340 root moves a rollout at ~17k steps/s (45k closed, 2.6x); slot 2 runs at ~54k with its
gate recorded open, so search can join it at a resume the moment slot 1 reads ahead.

**Slot 2 moved to E8 (2026-10-21, owner).** After the rebase onto main's E8 (ENG-4: a draw that empties the
deck reshuffles inside the event), the flagship was relaunched from scratch as `E8-A22-R70-S44` (same flags;
A22 is A21 at E8's default view, headline_reveal) and the E7 run stopped at 564M. Slot 1 stays on E7 to
finish its matched-step test against the E7 heads run; its reads are played on a copy of the E7 build kept in
`build/e7` (git-ignored), so its checkpoints and the heads run's meet on the engine they were trained on.

**Slot 1 at 400M (2026-10-21): behind, raw and searched.** SWAs of 320–400M (8 snapshots each) of
`E7-A8-R69-S44` and the heads run `E7-A8-R1-S44`, on the E7 build, 1,000 deals a side
(`data/reports/p36_slot1_400M_*`):

| | overall | US | USSR |
|:---|---:|---:|---:|
| raw, slot 1 against the heads run | **39.8% ± 1.1** | 41.2 | 37.3 |
| the heads run against itself (the per-seat bar) | 50.0% | 45.8 | 53.0 |
| **per seat over the bar** | | **−4.6** | **−15.7** |
| both under the corrected keyed root (k4, 16 worlds) | **38.5% ± 1.1** | 41.1 | 35.2 |

About −70 Elo raw and −80 searched at matched steps. The training metrics say it learns more slowly: entropy
0.60 against 0.50 at 240–400M (0.64 / 0.57 at 120–240M), KL 0.024 / 0.019, and it beats its own pool 73% against
the heads run's 79%. The search CE itself is not in `training_metrics.jsonl` (not on the metrics list), so its
pull is not yet measured. Candidates: the CE is averaged over the few searched rows of a minibatch (~15 of
4,096), so each such row carries ~270x a normal row's policy gradient, as R39 did at 1/256 (level there, as a
branch); the searched snapshot lags the learner by up to 10M while it changes fastest; the root's moves
played as environment take those rows out of the policy gradient. The plan reads it again at 800M and 2,000M.

**Slot 1 at 800M (2026-10-21): still behind, the gap closing.** SWAs of 720–800M, the same protocol
(`data/reports/p36_slot1_800M_*`):

| | 400M | 800M |
|:---|---:|---:|
| raw, slot 1 against the heads run | 39.8% | **42.7% ± 1.1** (US 41.9 / USSR 42.5) |
| the heads run's own bar (US / USSR) | 45.8 / 53.0 | 47.9 / 51.6 |
| raw, per seat over the bar | −4.6 / −15.7 | **−6.0 / −9.1** |
| both under the corrected keyed root | 38.5% | **43.0% ± 1.1** (US 43.7 / USSR 41.4) |

About −50 Elo at 800M against −70/−80 at 400M: behind in both seats, closing by ~3–4.5 points a 400M. The
plan's decision point is 2,000M (on past it only if ahead).

**Slot 1 at 2,000M (2026-10-22): the gated from-scratch search test fails.** SWAs of 1,920–2,000M, the same
protocol (`data/reports/p36_slot1_2000M_*`):

| | 400M | 800M | 2,000M |
|:---|---:|---:|---:|
| raw, slot 1 against the heads run | 39.8% | 42.7% | **42.2% ± 1.1** (US 43.8 / USSR 39.9) |
| the heads run's own bar (US / USSR) | 45.8 / 53.0 | 47.9 / 51.6 | 48.5 / 50.2 |
| raw, per seat over the bar | −4.6 / −15.7 | −6.0 / −9.1 | **−4.7 / −10.3** |
| both under the corrected keyed root | 38.5% | 43.0% | **40.1% ± 1.1** (US 43.3 / USSR 36.1) |

Behind in both seats at every read, about −55 Elo raw and −70 searched at 2,000M, and no longer closing after
800M. By the plan's rule it stops at 2,000M (on past it only if ahead). **Active search targets inside a
from-scratch run -- the one form P36c left untested -- make the base model worse here, raw and under the
root**, at ~2.6x the cost per step once joined. So search stays off the flagships: the encoder (P35 item 1)
launches in slot 1 without it, and slot 2's recorded gate is not used. What it does not settle: the form
tested (1/64 of entry decisions played as environment, the CE averaged over the searched rows so each carries
~270x a normal row's gradient, a 10M-lagged snapshot) -- a lighter CE weight or targets without the
environment moves are untested.

**Does the search run train slower to the same plateau? (owner's question, 2026-10-22) -- no sign of it.** Both
trajectories against one fixed reference, the heads run's SWA 1,920–2,000M, raw, 1,000 deals a side, 80M-window
SWAs (`data/reports/p36_curve_*`):

| window ending at | slot 1 (search) | heads run | gap |
|---:|---:|---:|---:|
| 400M | 22.2% | 24.0% | −1.8 |
| 800M | 29.6% | 38.9% | −9.3 |
| 1,200M | 37.5% | 43.2% | −5.7 |
| 1,600M | 40.1% | 45.5% | −5.4 |
| 2,000M | 42.2% | (the reference) | |

The gap opens between 400M and 800M, as search takes effect, and then holds: from 1,200M both gain ~2–2.5 points a
400M, so slot 1 is not closing. At 2,000M it plays where the heads run did at ~1,150M (43.2%); at matched strength
it climbs at half the rate per step (37.5 → 42.2% in 800M, the heads run 38.9 → 43.2% in 400M), at 2.6x the cost
a step -- ~5x the compute a point. Nothing in the curve points to a higher plateau, so it is not trained on. A
different form (a much lighter CE weight, or the improved-policy targets without the root's moves played as
environment) would be a new test, not more steps of this one.

**Why slot 1's search CE was missing (2026-10-26).** Not a missing name on the list: `generic_trainer`
registered the search-CE metrics only when the `--search-ce-coef` *argument* was above 0, and an
active-search run sets the trainer's coefficient from `--active-search-ce-coef` with that argument left at 0.
Fixed in this change (the trainer's coefficient decides; `tests/training/test_search_ce.py` refuses the
argument form). `E7-A8-R69-S44` logs the CE only from a resume on fixed code; its snapshots can be read
offline meanwhile with `tools/distill_probe.py track`.

**The correction form re-read, closed in every form tried (2026-10-26;
[`../log/E7_distill_ladder.md`](../log/E7_distill_ladder.md)).** P36c's targets predate the aligned root, and
a pick throws away the root's per-candidate values and their noise, so three ways out were left. On 200k
positions of A13+R48's SWA, offline:

* a teacher ladder (h0 the critic alone, horizons 1, 2 and 4, 16 to 128 worlds, the soup as a positive
  control): every root's departures are local on the trunk (+2 to +4 points at the nearest neighbours,
  the soup +9);
* P36c's best student refitted on the aligned 128-world targets: 18.3% of held-out departures against
  the teacher's 75.2% (P36c 20.2%);
* an action-value head fitted to each candidate's lead, weighted by its standard error: it ranks the
  root's candidates at r = 0.80 disattenuated (the critic alone 0.37), within a position too, and its
  argmax still plays worse than the network, −0.10 per decision ×100 on the judge's held-out seeds
  against the root's +1.35. A 4× noisier fit, a trained trunk and the policy logits as the head change
  nothing; a doubling of the data gains +0.06, toward the network's 0.

What stays open is the active form (slot 1), the root's own bias (a blind human-judge bank, built on the
fork, labels not in) and a better critic (P34 revision 2). **A pre-registered check for any further
distillation proposal:** a realised gain above 0 on the judge's held-out seeds, at a departure rate of at
least 5%, before any games.
