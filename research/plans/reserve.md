# Reserve — ideas with a trigger, not a slot

Each entry names the measurement that would promote it to a step file. Nothing here is queued.
Add new ideas here first (`README.md`, rule 4).

## Annealed potential shaping
**Trigger:** empty battlegrounds at turn 8 and the setup probe are still flat after P1, P2 and
P4 — i.e. the target is fixed and the agent still does not expand.
Form: `R_t = (1−α_t)·[Φ(s′)−Φ(s)] + α_t·R_game`, α_t → 1 (`ideas_and_plans.md` §2), with the
battleground term of `Scoring::compute_useful_actions_potential` made monotone in the gap to
control (`experiments.md` §12.4) — that is an engine change and needs approval. Policy-invariant
only under a perfect critic; while it is on, the agent learns Φ's mistakes, which is why it is
annealed and why it is behind the target fixes. The owner regards shaping as a stretch; it runs
only with a specific flat-probe result to justify it.

## League / PFSP
**Trigger:** head-to-head results between snapshots of one lineage are non-transitive
(A > B > C > A) or a specific exploit (DEFCON trap, space rush) appears in self-play and
disappears when the opponent changes. `ideas_and_plans.md` §4. Not before, because self-play
against the current policy is still gaining ~57 Elo per doubling and nothing has shown cycling.

## AIVAT-style evaluation
**Trigger:** two arms land within ~20 Elo of each other and the decision matters. Value-based
control variates at die nodes (P2's clone average is exactly the die-node baseline) and at deal
nodes (the pre-deal critic). Would take a 1,000-game tournament from ~20 Elo to ~5. `metrics.md`
§7 and `references.md` §3.

## Q-boosting proper (action-value critic, Expected SARSA(λ))
**Trigger:** P2 and P5 are adopted and the P0 decomposition attributes most remaining return
variance to policy sampling. `references.md` §3. An action-value head over the 212-dim space is
a larger change than P2 and is only worth it once chance is handled.

## Shallow expectimax at play time
**Trigger:** the owner wants a stronger product than the no-search network and accepts search
at play time. One or two plies over the enumerated dice with the critic at the leaves — the
TD-Gammon form (`references.md` §3); P2's helper is the building block. Not a training change.

## Belief head with a real target weight
**Trigger:** P5's belief AUC is near chance while the VOA-exposure probe stays at baseline.
Raise `belief_loss_coef`, or give the belief its own tokens in P6's backbone.

## Categorical head with ending-type atoms
**Trigger:** DEFCON-1 endings stay mispriced after P1 (calibration curve split by ending type).
The 51-atom "VP & end types" head of `ideas_and_plans.md` §3.

## Entropy / KL-to-π_ref schedule (Ataraxos damping)
**Trigger:** P3 finds search helps on *beyond-horizon* decisions (setup, battlegrounds), which
would say the policy head, not the critic, is what is weak.

## The cloud consolidation run
**Trigger:** the recipe is fixed — no queued step would change it. Then one ~400M-step run of
the winning configuration on rented GPUs (arm E was still gaining ~57 Elo per budget doubling
at 240M, so this is worth ~+100 Elo on top of the recipe changes). Spend the budget once, on a
recipe that has been screened and confirmed on the 4090; never on a screen.

## Play-time search as a product
**Trigger:** the no-search player has reached the mediocre-human bar and the owner wants more.
P3's determinized searcher with an MMD root step (Ataraxos). A few seconds per move is
acceptable.
