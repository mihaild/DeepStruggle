# Placement-order redundancy: the problem and three remedies (2026-11-07)

Owner's question, after the setup probes: setup order does not matter, so sampling has no pressure to place
Austria first rather than second -- how to deal with that, what is the gradient push, and what is the standard approach
for micro-actions with many orders to the same board? The plan for the remedy chosen to try first is
[P43](../plans/P43_canonical_placement_order.md).

## The measurement

J5 (`E8-A35-R102-S44@930M+R119`) at 1,200M, its USSR setup sampled at T=1 over 4,000 deals (`austria_by_step.py`):
P(Austria) is 1.0 / 2.2 / 4.4 / 5.6 / 4.8 / 4.6% at placements 1-6 while Austria is empty, and 0.8 / 1.6 / 5.1 / 3.5 /
4.0% at placements 2-6 once it holds a point -- a near-independent ~5% per late placement (22% of setups hold it), not
a plan. Setup sensitivity to the hand is in `data/reports/setup_by_cards_J1_J5.txt`, West Germany / Yugoslavia /
Austria rates in `data/reports/setup_wg_yug_aus.txt`.

## The gradient

Per step, PPO's expected push on the logit of country c at placement k is π_k(c) · (Q_k(c) − V_k) (plus the
regularisers). If Austria is good, it is good at every placement, so its probability grows at each placement in
proportion to what that placement already has: the split of the mass over orders is never selected, it drifts. The
per-step entropy bonus pushes it toward uniform over orders, because an order's randomness costs nothing in outcome --
so the logged per-step entropy overstates outcome diversity. What the gradient can teach is consistency (a second
Austria point worth less than the first), but only from the sampled paths that already placed one.

## Remedy 1 -- canonical order (P43)

The mask allows a run's placements only in non-decreasing country rank, so each board has one sequence. Zero learner
cost, removes the redundancy entirely, fixes the factorization by hand. Engine change (a rule bit).

## Remedy 2 -- count actions, and how they relate to P42

"Place k points in c" is the generalisation of the pair discussed for P42: "place 1" is k = 1 and "until control" is
the state-dependent k = points needed to control c (capped by the run's remaining points). A full count head lets the
policy pick any k ∈ {1, ..., remaining}. Count actions **alone do not remove the redundancy**: Poland 3 then Poland 2
equals Poland 5, and the order across countries is still free. They remove it only with two constraints together:
each country at most once per run, and the countries in a fixed order -- which is remedy 1 with counts instead of
single points. Note that P42's macro, as built, *adds* paths to the same board (until-control, or 1 + 1 + 1 + 1)
while it improves the exploration geometry; the two problems are different, and P42 does not address this one.

## Remedy 4 -- the order-marginal loss

Keep sampling exactly as now; change only the learner. Treat a run as one compound action, its final multiset M. Under
the autoregressive policy,

    P(M) = Σ over orders σ of M of Π_k π(σ_k | S_{k-1}(σ)),

where S_{k-1}(σ) is the partial multiset after σ's first k−1 placements. The partial multisets form a lattice with
Π_c (m_c + 1) nodes for counts m_c (USSR "Poland 5, East Germany 1": 6 × 2 = 12 states; US "France 4, Italy 3": 20;
an Ops run of 4 points: at most 16), so P(M) is a forward pass over the lattice:

    α(∅) = 1,   α(S) = Σ_{c ∈ S} α(S − c) · π(c | S − c),   P(M) = α(M),

differentiable end to end. The policy gradient for the compound action is A · ∇ log P(M) = A · Σ_σ w_σ ∇ log P(σ), with
w_σ = P(σ) / P(M) the posterior over orders given the outcome: every order of the sampled board is credited in
proportion to its probability, instead of the one order sampled.

What it changes:

* **Variance.** It is the Rao-Blackwellised estimator over the order: the noise from which order happened to be
  sampled is gone. The order is still a flat direction -- the posterior weights reinforce the orders the policy
  already prefers -- but nothing drives it any more.
* **Consistency is trained on every sample.** At a lattice node S that already holds Austria, the gradient raises
  π(c | S) only for c in M − S. When M has one Austria point, every node holding it pushes the next placement away
  from a second one. Today that signal exists only on sampled paths that happened to place Austria early.
* **One PPO step per run.** The ratio becomes P_new(M) / P_old(M), clipped once; the run is one time step for GAE (the
  critic at its start and end only). The per-step regularisers should move to the outcome too:
  * the KL to π_ref as KL(π_ref(M) ‖ π(M)), estimated on the sample;
  * entropy as outcome entropy, estimated as −E log P(M) with the score-function estimator, because the exact sum
    runs over all multisets (5,005 for the USSR setup);
  * the zero-avoiding term KL(uniform ‖ π) over outcomes. It has no cheap exact form, so it needs an approximation,
    e.g. keep it per step.

What it costs:

* **Learner compute.** One forward per lattice node instead of one per placement: about 2× for the setup's runs, up
  to ~4× for a 4-point Ops run with every point in a different country. Applied to every placement run, on the order
  of +30-60% learner forwards.
* **Buffer plumbing.** Each run's start state (4 KB) goes into the buffer. The lattice states are rebuilt by the
  engine and need their own observations and masks, which are CPU work per sample.
* **The certificate.** Every order of M must be legal and must give the same state. Orders the mask forbids (if any,
  e.g. ENG-10's ladder) get π = 0 at their lattice node, so the engine must produce masks at every node.

What it leaves alone: the engine, the mask, the action space and every checkpoint (no rule bit, no new epoch);
search; rollouts.

## Comparison

| | redundancy removed | engine change | learner cost | order chosen by |
|---|---|---|---|---|
| 1 canonical order | fully | yes (rule bit) | none | hand (static rank) |
| 2 count actions | only with 1's constraints | yes (action view) | none | hand |
| 4 order-marginal loss | in credit only; the policy keeps all orders | no | +30-60% learner forwards (all runs) | the policy, unpushed |

Recommendation: try 1 on the setup first (cheap, and the setup is the cleanest case). Remedy 4 is the fallback if the
owner prefers no engine change, or if a hand-fixed order turns out hard to learn in.
