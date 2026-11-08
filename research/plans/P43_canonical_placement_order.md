# P43 — canonical placement order: one action sequence per board (2026-11-07)

**Status: built (owner's go, 2026-11-07)** -- two rule bits, `canonical_first` and `canonical_last` (the order a
parameter, owner), covering every placement run: the setup, Ops influence plays and the verified event runs; a run
that would strand points releases its floor for that choice instead of a lookahead (owner), so a run reaches exactly
free order's boards (`engine/AGENTS.md` §21). View codes **V1** (important first) and **V2** (important last); a
branch is `E8-A35-R102-S44@930M+R119@<step>M+V1` with `--resume-add-rules canonical_first`. Phase B (Ops runs) needs the exact completion check below before it can be built; the branch is not launched.
Asked for by the owner on 2026-11-07, after the setup probes.

## Why

A run of influence placements by one side, with no chance or opponent decision between them, reaches the same board
in many orders. The policy is trained per step, so the order is a flat direction of the objective: the policy
gradient on "Austria at placement k" is about π_k(Austria) · (Q_k(Austria) − V_k), so probability grows wherever it
already is and no order is ever selected, while the per-step entropy bonus actively rewards spreading over orders
(randomness that costs nothing in outcome). Measured on J5 (`E8-A35-R102-S44@930M+R119`) at 1,200M, its USSR setup
sampled at T=1, 4,000 deals (`austria_by_step.py`):

| USSR placement | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| P(Austria), Austria still empty | 1.0% | 2.2% | 4.4% | 5.6% | 4.8% | 4.6% |
| P(Austria), Austria already placed | – | 0.8% | 1.6% | 5.1% | 3.5% | 4.0% |

Austria is a near-independent ~5% at each late placement (22% of sampled setups hold it, some with two points), not a
plan "one point in Austria". The consistency signal -- is a second point worth less than the first -- is learned
only from the rare sampled paths that already placed one. J5's sampled US setups also carry a stray West Germany
point in 25.5% of deals and never four (`data/reports/setup_wg_yug_aus.txt`).

**Canonical order removes the redundancy at the mask:** within a placement run, a country is legal only if its
rank is not below the last country placed in that run. Each final board then has exactly one action sequence; the
model in effect chooses a count per country in a fixed order, and per-step entropy measures outcome entropy. This is
how Diplomacy agents generate a turn's orders (DipNet, Paquette et al. 2019; the DORA / Cicero line, Bakhtin et al.
2021-22: one order per unit, autoregressively, in a fixed province order).

**Literature** ([canonical_order_literature.md](../log/canonical_order_literature.md), Fable, 2026-11-07):
* A fixed order chosen by hand and enforced is standard: the Diplomacy line, and DeepNash's setup sorted by piece type.
* No source compares free order against canonical order under RL. Phase A would be, as far as was found, the first
  such measurement under PPO self-play with a per-step entropy bonus.
* Order matters for learning (Vinyals et al.: ≥ 5% absolute; DipNet: "an order that does not jump across the map").
  This supports battlegrounds first, then region by region, with plain country id as the control.
* A canonical order learned without being enforced leaks (GraphGEN: 9.1% canonical), which is why it goes in the mask.
* Pokémon random singles, Jaxcalibur's game, has one categorical choice per side per turn. It is immune to this
  problem because of the game's structure, not because of anything in the recipe.

## Where it applies

A run qualifies only if every order of the same multiset is legal and gives the same state. Checked in the engine:

1. **The setup** (`state_machine.cpp`, `Phase::SETUP`): USSR 6 in Eastern Europe; US 7 in Western Europe; US 2 bonus
   points where the US already has influence. Three runs; nothing happens between their placements. **Phase A.**
2. **Ops spent as influence** (`POINT_NODE` within one play): eligibility uses the start-of-action snapshot
   (`ops.cpp` `has_start_influence`), a country's cost depends only on its own control, so order across countries
   does not change cost or legality. The region-bonus ladder (ENG-10, `Operations::budget_if_spent_in`) is the one
   order-sensitive rule to verify: the reachable multisets must be identical under canonical order (test below).
   **Phase B.**
3. **Event placements and removals** (Marshall Plan, Decolonization, De-Stalinization, Socialist Governments, Comecon,
   Warsaw Pact, ...): sets or capped multisets, same argument per handler. **Phase C**, only if B pays.

Not realignments (dice between them, and each changes the next one's modifiers) and not coups (single choices).

## Design

* **A rule bit**, as ENG-9: `obs_features::CANONICAL_PLACEMENT` (the next free bit, 1 << 4), no width, one of
  `obs_features::RULES`; a seat plays it iff its model was trained with it (`GameState::seat_rules`). Bit 0 -- every
  existing network -- is today's mask, so the behaviour digest and every current run are unchanged. Phase A and B
  could be two bits if the owner wants them separable.
* **Mask.** The run's lower bound is the rank of the last country placed in the current run (kept in the decision
  context, reset when the run starts). `generate_flat_mask_212` additionally zeroes `NODE + c` for rank(c) < bound
  while the bit is seated. The P42 macro slots follow the same bound (an until-control in c sets the bound to c).
* **The order.** Fixed, static. Proposal: **battlegrounds first, then by country id** within region -- the
  factorization decides the scoring-relevant counts first, which is where Vinyals et al. (2016, "Order Matters")
  found the order of a set's factorization changes learnability. Country id alone is the cheaper control if the owner
  prefers no judgement in it.
* **Does the network see the bound?** In the setup yes, implicitly: the starting influence is fixed, so the board
  is the partial multiset and the bound is its highest rank. In an Ops run it is not visible from the board (the
  pre-play board is not observed). Options, the owner's call: (a) nothing -- the mask renormalises the network's
  logits over what remains, a constraint on the policy class but no wrong reads; (b) let LadderNet read the mask it
  already receives in `forward` as an input (an architecture flag, not an observation change); (c) an observation
  block (owner; costs floats). Phase A needs none of them.

## The regularisers under a canonical order (owner's question, 2026-11-07)

Does the order make the entropy bonus or the zero-avoiding term prefer low ranks, because those keep the next step's
choice wide? **Not as implemented:** both are per-decision loss terms (`nash_pg.py`: the bonus on each decision's
entropy, `zero_avoid_loss` on each decision's legal set), so they shape π(·|s) at each state and never value the
entropy of the states an action leads to. That would be maximum-entropy RL with entropy in the return, which we do not
run.

**They do acquire a rank bias, in the other direction:**

* Each decision is pulled toward uniform over its own legal set, and a run's legal set shrinks to the suffix of ranks at
  or above the last country placed.
* The highest-ranked countries are legal at every step of a run, the lowest only at its first. So the floor
  (≈ c/(nΔ) per country, n the legal count, which shrinks along the run) and the entropy pull pile stray mass onto
  high ranks late in the run.
* Free order has no rank asymmetry (its per-step uniform target is independent uniform points), though it does favour
  spread-out boards.

The size is the floor's (0.13-0.22% per move measured for the zero-avoiding term), well below J5's ~5% Austria per
placement, but it is systematic and depends on the hand-chosen order. Remedies, in order of preference:

1. **Regularise the board, not the step.** With one sequence per board, log P(board) is the sum of the run's per-step
   log-probs: exact and free, the quantity remedy 4 needs a lattice for. Board entropy (−E log P), the KL to π_ref and
   a floor can then be taken per run. The correct gradient of board entropy does contain the "keep options open" term
   (a choice is credited with the entropy that follows it), which here is what makes the target uniform over boards
   instead of skewed to high ranks.
2. **Keep the per-step terms only on each run's first decision**, where the legal set is the run's full set.
3. Accept the bias and choose the order so the high ranks are the countries where a stray point does least harm.
   Fragile, and it puts judgement in the order.

Phase A measures it directly: stray single points by rank, against the free-order control.

## The free-order alternative: correct the DAG instead of fixing the order (owner's question, 2026-11-07)

The owner asked whether the variational identity for DAGs replaces the canonical order: Malkin et al. (ICLR 2023,
*GFlowNets and variational inference*: trajectory balance is hierarchical VI with the backward policy as the
posterior over paths) and Deleu et al. (UAI 2024, *Discrete probabilistic inference as control in multi-path
environments*). The identity, in our terms: a free-order run is a DAG -- one board, many sequences -- and any
objective with an entropy or a per-step divergence lives on sequences, so its optimum gives a board mass
proportional to its number of sequences (k!/∏mᵢ! for a run of k points, mᵢ per country: spread boards over
concentrated ones). Deleu's theorem: add α·log P_B(s_t | s_{t+1}) to each transition's reward for any backward
policy P_B (uniform P_B: −log of the number of countries placed so far in the run), and the MaxEnt optimum's board
marginal is ∝ exp(R) with no path-count term; the GFlowNet objectives (TB, SubTB, DB) are PCL and soft Q-learning
with that correction. The correction is loss-side only: a count from the board diff since the run's start, no
engine, no mask, no dead ends, no run cursor the network cannot see, no re-sorting of human runs for BC, and the
policy keeps its own order (it may decide West Germany first and look at the board after, the owner's intuition).

**Verdict: not instead of the order. It is the free-order fallback and the justification for remedy 1 above.**

1. *It corrects a bias that is floor-sized here.* We do not run MaxEnt RL; the per-step terms are the zero-avoiding
   floor (0.13–0.22% per move) and the KL to π_ref, and the only sequence-level bias they carry is the one measured
   above. The pathology that motivated P43 -- Austria as an independent ~5% at each of four placements, a second
   point barely devalued -- is not a path-count bias of a regulariser. It is the policy realising one board decision
   as k separate chances, because order is a flat direction and nothing collapses the k opportunities into one. The
   canonical order collapses them (Austria is asked once, when its rank comes). The correction leaves them: it changes
   what the regularisers prefer among the sequences of a board and leaves *when to decide* flat, at zero gradient
   instead of a small pull toward spreading.
2. *Where it does apply, the tree makes it exact and free.* Under one sequence per board, log P(board) is the run's
   sum of per-step log-probs, so remedy 1 (regularise the board) needs no P_B, no lattice and no bound. Under free
   order the same quantity needs the lattice (remedy 4, ×1.66) or the variational bound, which with a uniform P_B is
   tight only for a policy uniform over orders -- the opposite of one that has settled on an order -- and otherwise
   needs a learned P_B (trajectory balance's), a second network.
3. *The bias's direction bears on P42.* Under free order the regularisers favour spread boards over concentrated ones
   by the number of orderings (a four-point takeover has one sequence; four single points up to 24), the direction of
   the West Germany hole; the canonical order removes it, the macro adds the opposite prior. Under the canonical order
   the macro reintroduces one duplicate per run (until-control against 1+1+1+1 in the same country): harmless to PPO
   credit, at most ×2 for the regularisers; note it, do not mask it.

When it would be the right tool: if V1/V2 fail on what the canonical order costs -- the run cursor the network cannot
see in Ops runs, dead-end resets at a rate that matters, or an order that proves wrong both ways -- the free-order
branch with the P_B-corrected per-step terms is the cheaper fallback than the lattice. Not before.

## A policy-ordered run (owner's proposal, 2026-11-07): the order is the step-0 ranking

The owner's objection to a fixed order: which country is "most important" depends on the board, and wherever
the fixed order is wrong the model must plan around it ("I need a point here even though the important country is
elsewhere"). The proposal: at a run's first step take the policy's own distribution over countries, rank them by
it, and use that ranking as the run's canonical order -- the top choice is placed now or not at all in this run.

**What it is.** The same rule as V1/V2 (a country is legal only if its rank is not below the last placed), with the
ranking a function of the state through the policy instead of a hand-chosen table. Per state and per snapshot,
one sequence per board still holds, so the redundancy is gone and the regularisers act on boards as under V1/V2.
It is "important first" with importance read off the policy in the current state, and it is not what Vinyals' and
the graph-generation work call a learned order (a free order plus posterior credit, which locks early): it is a
constraint derived from the policy's first preference, no new output. Every board stays reachable, as under any
canonical order; "wrong order" costs learning difficulty, never expressiveness (dead ends aside, and the release
handles those).

**Its one clear advantage: it minimises the exclusion tax.** Under any canonical order, sampling a country ranked
below a wanted one at step 0 excludes the wanted one for the run (a fixed [West Germany, Italy, …] loses West
Germany whenever Italy is sampled first). The tax is the same shape under every order; the policy-ranked order
puts the most likely first choice at the top, so the most likely sample excludes nothing wanted, which no fixed
order can promise in every state.

**Its costs.**

1. *The mask depends on the network.* V1/V2 are a function of the state, computed in the engine; this needs the
   policy's ranking supplied to the batch runner per env at each run's start and carried through the run
   (`set_run_order(env, perm)`-style), and the same from every search harness (rollout root, Gumbel, MCTS: each
   leaf evaluation at a run start sets the order) and from the page. The replayer stays in free order (human logs
   are multisets, and legality must not depend on a model); BC would sort each human run by the model's own
   ranking at training time, per model.
2. *The mask moves.* Near-ties in the step-0 distribution flip the ranking between updates, so the same state
   presents different legal sets at later steps from one snapshot to the next, and the network cannot see the
   ranking except by recomputing it; under a fixed order the exclusions are a function of what has been placed.
   A learning-stability cost of unknown size.
3. *It is V1/V2's plumbing plus a policy round-trip*, so it is the more expensive build, with nothing yet showing
   that the order matters.

**Where it sits: V3, after V1 and V2 have been read.** V1 and V2 are built and are the two ends of the order
question; their gap against each other and against the free-order control is the measurement of whether the order
matters at all. If V1 ≈ V2 ≈ free, the question is moot. If the order matters, V3 is the next arm, and it should
then be compared with the other learned-order route, free order with the order-marginal loss (remedy 4, ×1.66,
no mask plumbing, the policy keeps any order). **P43a, no training, first:** on free-order J5 self-play runs, rank
by the policy's step-0 distribution, sample at T=1 and count how often the run's greedy top country is excluded
by an earlier sample, under the policy ranking and under V1's and V2's tables -- the exclusion tax each order
would charge today, which says how much V3 could gain over V1/V2 before anything is built.

## First runs (2026-11-07): V1 and V2 from scratch on A8, 400M

J1's network and recipe (A8, R119), J1's view plus one canonical bit, against the free-order control
`E8-A8-R119-S44` (data/reports/p43_*_400M.*):

| | read 200M | read 400M | head to head at 400M | releases / game |
|---|---:|---:|---|---:|
| control (free order) | 11.3% | 17.3% | 51.0% vs V1, 60.8% vs V2 | -- |
| V1, important first | 9.3% | 16.2% | 58.5% vs V2 | 2.34 |
| V2, important last | 7.8% | 12.8% | | 2.58 |

* **V1 is level with free order at 400M** (51.0% / 49.0% head to head, 1.1 behind on the read, inside its error);
  by the owner's rule a big change is not judged this early. **V2 is clearly behind.**
* **Why V2 lags: forced tails.** Once a run reaches the top country of its order, every remaining point is forced
  there and auto-advance plays it: V2's USSR always opens Poland 6 and its US West Germany 4, France 3, Italy 2 (the
  human West Germany 4, reached as a tail). Those placements are never decided, so never trained. Important first
  has the same property only at the bottom of the order, which a run rarely reaches.
* **Releases are rare where it matters and common in capped event runs.** Ops 0.01% (V1) / 0.22% (V2) of floored
  decisions, the setup 0%, but Marshall Plan 26%, Comecon 26-27%, East European Unrest 30-39%, Decolonization 9-18%,
  Colonial Rear Guards 13%, Suez Crisis 12% (V1): in one-per-country runs the policy often picks above the countries
  it still needs, so uniqueness is lost in a few decisions per game.
* **Setups moved:** the control opens USSR Poland 5 + Austria 1 (Austria in 93.5% of sampled setups) and US France 5,
  Italy 2, Iran, Philippines; V1 opens USSR Finland 3, Poland 3 and US France 3, Canada 2, West Germany 2; neither
  canonical arm keeps a stray Austria point (1.7% / 0.2% at T = 1).

## The run cursor must be observed (owner, 2026-11-07)

Under free order the board plus `REMAINING_STEPS` is a sufficient state for a placement decision: where the points
already went in this run changes nothing about what is best next. **Under a canonical order it is not.** The legal
set at a later step depends on the rank of the last country placed in the run -- the cursor -- and the observation
does not carry it: there is no "placed this run" and no history input. In the setup the network can recover it
(the pre-setup board is fixed by the rules and influence is per side, so what it has placed is the difference),
but in an Ops or event run it cannot, so **the state is hidden from the network in exactly the runs P43 was
extended to.** Both heads suffer: the critic's value at a mid-run state averages over cursors it cannot see,
which is the bootstrap and baseline of every step in the run (GAE runs through placement steps), and the policy
cannot plan the run ("Austria now or it is gone") -- the masked softmax only renormalises its free-order
preferences over whatever is legal. The free-order network has no such gap, nor does the P42 view. This was raised
at the build (15:16) and left open; it should be closed before V1/V2 launch.

**Proposal (an observation change: the owner's call, this is the ask).** An appended `obs_features` block,
`PLACEMENT_CURSOR`, **one float**: (rank of the last country placed in the current run + 1) / 85 in the active view's
own rank table, 0 outside a run or before its first point. It specifies the hidden state completely under V1 and
V2 (the table is fixed, so the rank is the mask), so it is not a partial feature; it costs one float; it is appended,
so J1/J5 checkpoints keep running; a model's set is recorded in its weights and `metadata.json` as today
(`--obs-features headline_reveal,placement_cursor`), and V1/V2 models are trained with it from their first step.
Engine (`observation.cpp`) and bindings work, so approval, the C++ suite, `tests/bindings tests/engine_logic`, the
wasm parity test and the stub regeneration. If V3 (the policy-ordered run) is ever built, a scalar no longer
suffices -- the excluded set is then the policy's ranking -- and the block becomes the 84 country-legality bits,
appended as a further block; not now.

**The better way to close it (running session, 2026-11-07): feed the mask, not a new observation.** LadderNet's
`forward()` already receives the legal mask; routing its 84 NODE bits into the trunk as an architecture flag (an A
code) gives both heads the exact legal set under V1, V2 and V3 alike, touches no engine code and no observation
layout, and leaves every old checkpoint as it is. It supersedes the one-float block above: the block is complete
only under a fixed table, the mask input is complete under any order rule, and it is Python only. The
preference stands as: **the mask input first; the observation block only if the owner wants the engine to own it.**
Either way it has to be in the network before a canonical-order read is clean.

**V1/V2 ran to 400M on A8 before this was raised (owner's order), without the cursor:** V1 (important first)
16.2% against the free-order control's 17.3% (R119 from scratch), head to head 51.0% for the control -- level;
V2 (important last) 12.8%. So the order matters (3.4 points between the two tables), which is the evidence V3 waits
on, and important first is the right end. But both reads are of a canonical order **with the run state hidden**
from the network in every Ops and event run, so they are a lower bound on what the order can do, not its
measurement; V1 continues 410 → 810M on the owner's order, and the clean test is V1 with the mask input against
the same control. V3 is built in a scratch tree, not merged or launched.

## placement_legal and V3 at 400M (2026-11-08)

Both from scratch on A36 (A8 reading `placement_legal`) and R119, against the free-order control `E8-A8-R119-S44` and V1
without the slot (`data/reports/p43_pl_*_400M.*`):

| | read 200M | read 400M | control's score head to head | V1's score head to head |
|---|---:|---:|---:|---:|
| V1 (no slot) | 9.3% | 16.2% | 51.0% | -- |
| V1 + placement_legal | 6.9% | 11.4% | 62.4% | 75.0% |
| V3 + placement_legal | 9.0% | 16.1% | 59.4% | 56.3% |

* **The slot made V1 much worse.** It is correct -- checked in V1+slot's own self-play: at every placement decision it is
  the mask, nothing else differs, and no other decision changes -- but it differs from bare eligibility at 71% of
  placement decisions (cost-2 countries a run cannot afford, the budget, the canonical floor), so one input of every
  country carries two meanings depending on the decision. The likely cost (a hypothesis, untested) is that overloading
  a slot the network reads everywhere hurts more than the information helps; a separate channel (the mask into the
  trunk, or an appended 84-bit block) would not overload anything. V1+slot also never found Europe control in training
  (0.0-0.4% of endings against 6-10% in every other arm), which may be the seed as much as the slot.
* **V3 with the slot is level with V1 on the read but behind head to head** (43.7% against V1, 40.6% against the
  control). Its releases are the fewest of any arm (0.86 per game).
* No canonical arm beats free order at 400M; by the owner's instruction J5 resumed (2026-11-08, from 1,390M).

## Next (owner, 2026-11-08): planned, not scheduled

Two experiments, run when the GPU is free of the main line (J5). Both on A8 and R119 from scratch, with the free-order
control `E8-A8-R119-S44` continued to the same budget beside them, so every comparison is at matched steps; reads every
400M against A13+R48's SWA and one head-to-head at the end, judged late per the owner's rule (a big change is not
judged on early reads unless a clear problem shows).

1. **V1 for longer.** V1 (`E8-A8-R119-V1-S44`) was level with free order at 400M (16.2 vs 17.3% read, 51.0% head to
   head for the control) and read 21.5 / 26.4% at 600 / 800M with no control beside it. Continue V1 in place from
   810M and the control from 410M, both to 2,000M (the scale at which A8 recipes have been separated before). Watch
   besides the reads: the release rate per source (Marshall Plan, Comecon and East European Unrest released in 18-30%
   of their floored decisions at 400M), setup outcomes, and whether the stray single points (Austria, West Germany)
   stay gone.
2. **V1 with order-based legality on its own channel.** `placement_legal` reused slot 19 and made V1 worse (75.0% for
   V1 without it, 400M); the slot was correct but changed meaning at 71% of placement decisions. Give the network the
   same information without overloading anything. Two candidates, the owner to choose:
   * **the mask into the trunk** -- LadderNet already receives the legal mask in `forward`; feed its 84 NODE bits
     (or the whole mask) through a small projection into the trunk. An architecture flag (a new A code), no engine or
     observation change, old checkpoints untouched;
   * **an appended observation block** -- an `obs_features` bit whose 84 floats, appended after the base, hold the
     decider's currently legal countries during its own placement decisions (0 elsewhere). The engine owns it;
     costs 84 floats; needs the owner's approval as an observation change.

   Run V1 + the chosen channel from scratch to 400M first (against the control and V1 without it), then to 2,000M if
   it is not behind. V3 (the network's own order) is the natural second arm on the same channel, since it needs the
   legal set even more: its order moves with its own logits.

## V3: the network's own order (owner, 2026-11-07)

At each placement run's first decision the deciding network's logits over the 84 countries, highest first, fix
that run's order; the run then places in non-decreasing rank of it (higher priority first), one sequence per board,
with the same dead-end release. Fixed at the run's start rather than recomputed each step, so the order cannot move
under a run and uniqueness holds. The engine cannot compute logits, so the harness gives the order at the run's
first decision (`set_run_priority`); a placement without it is refused unless forced. Built: the trainer's
rollouts (learner and pool members alike, from the logits each env's decider produced), the match runner and
`NeuralAgent`; searches, ONNX agents and the page refuse a V3 network for now. View code **V3**
(`canonical_learned`).

## The order: important last (owner's question, 2026-11-07)

Important first lets the factorisation decide the counts that matter before the extras; important last means a
run that strays past a country loses the least. Under a canonical order a step to rank r forfeits every country
below r for the rest of the run, so the cost of a stray step (T = 1 sampling, the zero-avoiding floor) is the value
of what lies below it. With the important countries last, a stray early step into a minor country costs one point
and the run can still reach every battleground; with them first, a stray step past West Germany loses West Germany
for the whole setup. Important last also puts the regularisers' high-rank floor bias (above) on the countries where
an extra point does least harm. The network sees the whole board at every step, so it does not need the important
countries first to take them into account. Built: regions by scoring weight (Africa, South America, Central
America, Asia, the Middle East, Europe), non-battlegrounds before battlegrounds, Europe's battlegrounds last (Italy,
France, West Germany, East Germany, Poland). Learning the order instead is discussed in
[placement_order_redundancy.md](../log/placement_order_redundancy.md).

**Phase B's open problem: dead ends.** An Ops point into an opponent-controlled country costs 2. A run that jumps to a
high-ranked country it still cannot afford to keep adding to may strand its last point, where free order would have
spent it lower down -- an outcome free order cannot reach (CONFIRM_DONE is offered when nothing is legal). The mask
must therefore offer a country only if the run can still be completed from there exactly as free order could
(per-country costs are independent, so a small dynamic programme over (country, points left) decides it). Event runs
with a cap per country (Marshall Plan) need the same check against their caps.

## Tests (before any run)

* Equivalence, exhaustively on sampled positions: at every placement run of ≤ 4 points (and the setup's 6 / 7 / 2),
  the set of reachable final boards under the canonical mask equals the set under the free mask, and each board has
  exactly one canonical sequence. Includes China Card and Vietnam Revolts runs (ENG-10's ladder), Missile Envy /
  Containment / Brezhnev / Red Scare modifiers, and event-first snapshots.
* Bit 0: 2,000+ seeded games bit-identical to today's engine (digest unchanged); ts_fuzz with the bit seated on
  both sides; the wasm build plays the same games.
* The page: a person plays every rule bit (`WasmEngine.currentRules`), so a human would play canonical order too --
  the owner decides whether the bit is in `currentRules` (probably not: it constrains how a person clicks, not the
  game).

## Runs

At most two trainers; J5 keeps its slot.

* **Phase A, setup only -- a branch.** J5 (or J1) at a resume state, seated with the bit; the branch's own setups
  change at once (J5's greedy "Poland 5, East Germany 1" becomes Poland 6 if it picks Poland first and East Germany
  ranks lower), so the first reads show the cost of the switch, not the method. Measured at +100 / +200 / +400M
  against the unbranched run: the setup outcome distribution (sampled), its entropy against the per-step entropy,
  stray single points (West Germany, Austria), second-point suppression, the critic's value of the setups it plays,
  and the 400M read. The setup is two runs per game, so a strength change is not expected to be visible; the setup
  metrics are the result.
* **Phase B, Ops runs -- from scratch** (the policy over ~a dozen placements per turn changes, so a branch would
  carry the old factorization): a canonical-order arm against the same recipe without it, both on the current
  default target, reads every 400M to 2,000M and head to head at 1,200 / 2,000M; judged late per the owner's rule
  (wait longer on big changes unless there is a clear early problem).

## Costs and risks

* Engine and bindings: the bound in the decision context, the mask rule, a rule bit (as ENG-9's plumbing: runner
  seats, `legal_mask`, searches seat their trees, datasets record the bits).
* Human logs: a human's placement order is free; behaviour cloning into a canonical model needs the human's run
  re-sorted into canonical order (deterministic, lossless), only if a demonstration warm start is used.
* The factorization is fixed by hand. If the chosen order is a bad one to learn in, Phase A shows it cheaply.
* Alternatives considered: count actions ("k points in c", which generalise P42's until-control) remove redundancy
  only together with one visit per country plus an order; an order-marginal loss (sum over orders in the learner)
  needs no engine change but costs learner compute -- both in `research/log/placement_order_redundancy.md`.
