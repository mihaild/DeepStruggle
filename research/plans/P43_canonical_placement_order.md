# P43 — canonical placement order: one action sequence per board (2026-11-07)

**Status: plan for the owner's approval. Nothing is built.** It changes the legal mask (engine code), so it needs the
owner's explicit go before the first line (invariant 11). Asked for by the owner on 2026-11-07, after the setup probes.

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
