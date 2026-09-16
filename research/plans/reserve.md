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
**Promoted 2026-09-16.** The trigger fired: the stall after 160M with oscillating side balance
is the cycling this entry was waiting for. The pool, PFSP and the first exploiter are now
[P15](P15_breaking_the_cycle.md) X1/X3; a full multi-exploiter league remains here, triggered
by X1 succeeding *and* X3 proving insufficient.

## PSRO-lite: meta-Nash opponent sampling
**Trigger:** P15-X3's span-the-run pool and PFSP are both neutral. Sample pool opponents from
the meta-game Nash of the pool's head-to-head matrix (Lanctot et al. 2017) instead of
uniform/variance weighting; needs a periodic in-run mini-tournament, which X0's anchor evals
already half-build.

## Optimism / extragradient in the optimizer
**Trigger:** P15-X2 shows the anchor timescale matters but no refresh cadence both damps the
cycle and keeps learning speed. OGDA / extragradient / optimistic mirror descent are the
last-iterate-convergent update rules; deeper surgery than a flag, so it waits for evidence
that the anchor family is the right one but insufficient.

## Per-side capacity
**Trigger:** P15-X1 shows a response exists but side-conditioned policy quality keeps
collapsing on one side while the capability is measurably in the weights (the P10
observation: 0.15 vs 0.04 mass on the same contested-battleground play depending on
perspective). Options: side-specific adapters or heads, or a side-conditioned trunk film
layer — an architecture change, so it queues behind the dynamics fixes.

## Voice of America exposure probe
**Trigger:** the agent contests battlegrounds — empty battlegrounds at turn 8 down from 6.2 of 29
toward the human rate, and Saudi Arabia / India / Algeria no longer conceded in almost every game
(`experiments.md` §25). Until then a probe measuring how it *defends* a foothold measures a
foothold it never takes.

Was probe 2 of P0 and is a correct description of a real mistake, so it is kept in full rather
than rewritten later. VOA (`cid 74`) is a US event removing **4 USSR Influence from non-European
countries, at most 2 per country** (`mid_war.cpp:475`), so a USSR position of exactly 1–2 there
can be erased outright and one of 3+ cannot. Measure over self-play games at each `TURN_CLEANUP`
node — the same node P0's pre-deal calibration samples, so the two share a pass: is VOA still
unplayed (`get_card_location(74)` in the deck or either US hand slot, read from the true state,
not from the observation), and how many non-European countries does the USSR hold at 1–2
influence. Report `min(count, 2)` alongside the raw count, because 4 points at 2 per country can
clear at most two of them, and split controlled from uncontrolled rather than excluding
controlled: 48 of the 63 non-European countries have stability ≤ 2, so 1–2 influence there is
frequently *control*, and losing it is the worse case, not the excluded one. Yardstick from the
human corpus, with the caveat that VOA reaches play in a fraction of 266 games — under 30 and the
punished-rate half is not reportable.

## Layout attribution: which v2.2 change did it, and are the card slots worth anything
**Status: dead, and not merely deprioritised.** It asked which of v2.2's changes carried the
~+92 Elo (§24) and whether v2.1's card-tracking slots do anything (§23.2). Both questions are
now unanswerable by the means proposed: v2.2 and v2.1 are deleted along with every argument that
could select one, and the checkpoints trained on them are refused by `check_checkpoint_layout`
rather than misread. Re-running the arms would mean reintroducing the layouts, which is the bug
class the single-layout refactor removed — five silent wrong-layout failures in this repository,
the last costing a published diagnostic.

What survives is the *question*, if a future observation decision ever depends on it: it would
have to be asked forward, as an ablation against the current single layout, not backward against
retired ones. Nothing currently depends on the answer.

## AIVAT-style evaluation
**Trigger:** two arms land within ~20 Elo of each other and the decision matters. Value-based
control variates at die nodes (P2's clone average is exactly the die-node baseline) and at deal
nodes (the pre-deal critic). Would take a 1,000-game tournament from ~20 Elo to ~5. `../log/variance_and_noise.md`
and `../method/references.md` §3.

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
**Partly promoted:** the ref-refresh half is now [P15](P15_breaking_the_cycle.md) X2. What
stays here is the *entropy* schedule (annealing `--entropy-coef` late in training), triggered
by X2 adopting a slow anchor and entropy then reading as the residual noise floor.

## Human policy anchor, card-level only (piKL form)
**Trigger:** P7d — the §22 replication with the new corpus — *still* costs Elo, i.e. the
mechanism, not the data, was at fault. A fixed human-trained policy π_h as a second KL anchor
in `nash_pg.py` (the machinery already exists for π_ref; this one is never refreshed), small
coefficient annealed to zero, applied only at card-level decision types (`SELECT_CARD`,
headline, play mode, space) where strong players agree and self-play is blind; placement
micro-actions untouched. `paper_cicero_diplomacy.md`, `paper_kl_regularized_search.md`.

## Advantage-filtered imitation
**Trigger:** P7d now *helps*, i.e. the old corpus was at fault. Imitate a human decision only
where the current critic does not rate it clearly bad (the P1 filtering machinery, applied to
the injected batch), so mistakes in the corpus are not copied. Still policy-side; still behind
P7b/P7c.

## Expert iteration on human positions
**Trigger:** P7b adopted and P3 finds search helps. Run P3's determinized searcher on human
positions — boards self-play never reaches — and distil into the network. Combines the
state-distribution benefit with the model's own value; no human *decision* is imitated.

## Belief-head targets from reconstructed human hands
**Trigger:** P5 adopted and the opponent hands in the P7 corpus are determined by the solver
for a useful share of positions. Supervised targets for β_opp on human play.

## Setup anchor from human setups
**Trigger:** P4 (macro-action credit) and its MC-target fallback both fail to move the setup
probe. ~3,000 strong setups are a good prior for a decision with ~1,000 candidates; yardstick
first (P0), prior only if learning it from the game fails.

## The cloud consolidation run
**Trigger:** the recipe is fixed — no queued step would change it. Then one long run (~400M+
steps) of the winning configuration on rented GPUs: v2.2 was still gaining at 320M where earlier
layouts had stopped (`experiments.md` §24), so budget keeps paying on the current recipe — though
per-leg gains carry ±16 Elo (`../log/variance_and_noise.md`), so the payoff is not extrapolatable to a
number in advance. Spend the budget once, on a recipe that has been screened and confirmed on
the 4090; never on a screen.

## Play-time search as a product
**Trigger:** the no-search player has reached the mediocre-human bar and the owner wants more.
P3's determinized searcher with an MMD root step (Ataraxos). A few seconds per move is
acceptable.
