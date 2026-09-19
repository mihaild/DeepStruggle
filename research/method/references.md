# Literature: what transfers to a game shaped like Twilight Struggle

Working bibliography for the research programme. The existing `paper_*.md` reviews cover the
imperfect-information / equilibrium lineage (DeepNash, NashPG, ReBeL, Student of Games, CICERO,
Suphx, OBL, AlphaStar, DouZero); this file collects the papers found while asking a narrower
question, and records the mapping to this game so it does not have to be re-derived.

**The question.** Twilight Struggle has hidden information, but it is *weak*: the board and VP
track are public, the deck composition is public, cards become public when played, and hands are
largely reconstructible (the replayer's z3 solver determines most hands from the log alone). It
is *chance-heavy*: coups, realignments, Space Race and several events are die rolls, and the
turn-by-turn deal decides which scoring cards and which ops are available. Its horizon is long
(10 turns, ~350 micro-decisions per game) and its interactions are global (one card play or one
scoring card reprices a whole region and the VP track). Stratego — the reference success story
for imperfect-information self-play — has the opposite profile: hidden information dominates,
there is no chance, interactions are local. So the point of the review is to separate the parts
of the modern recipe that were built for Stratego's problem from the parts built for ours.

Verification status: papers marked **[full]** were read in full (HTML or PDF) during the review;
**[abs]** means only the abstract or a secondary description was read and specific numbers
should be re-checked before being quoted in an experiment entry.

---

## 1. The modern self-play recipe

### Ataraxos — superhuman Stratego for a few thousand dollars **[full]**
arXiv:2511.07312 (Nov 2025). Superhuman Stratego with 16 H100 for one week, versus DeepNash's
far larger budget. Ingredients, in the order of how much they seem to matter for us:

- **Categorical value head** (win/loss/draw) trained by cross-entropy on λ-returns, not MSE on
  a scalar. → E8. Also the natural head for a chance-heavy game, where outcomes are genuinely
  multimodal (coup hits or misses; the scoring card is or is not in the hand) and a scalar
  regression averages the modes into a value that is never observed.
- **Advantage filtering**: drop low-|advantage| samples from the policy update. Reported 2.5×
  wall-clock and a better asymptote. → E9. Cheap, rides along with any arm.
- **Setup network trained on Monte Carlo returns** with an entropy bonus, separately from the
  bootstrapped value path, because the value net is least reliable at the start of the game.
  → the setup treatment in P4 (macro-action credit, below).
- **Direct policy sampling for data generation** — no search in the training loop. Search is a
  test-time addition. → matches the owner's preference for a no-search player.
- **Dynamically damped self-play**: annealed entropy and reverse-KL regularisation instead of
  R-NaD's fixed regularised Nash dynamics. → our NashPG already has the KL-to-π_ref term.
- **Test-time search**: belief-net sampling of the opponent's hidden state, ~1000 rollouts of
  depth ~40 with the policy net, average the values, one tabular magnetic-mirror-descent step at
  the root (Sokota et al.), sample. → E7's shape, with determinization replacing the belief net.

What does *not* transfer: the learned belief network and the equilibrium machinery exist because
Stratego is a bluffing game. See §2.

### Policy improvement by planning with Gumbel **[abs]**
Danihelka, Guez, Schrittwieser, Silver, ICLR 2022. Gumbel AlphaZero/MuZero: guaranteed policy
improvement with as few as 2–16 simulations per move by using Gumbel-top-k sampling and a
sequential-halving root. Relevance: retires the "MCTS costs 32× per decision" objection in
`next_step_brief.md` §4C. If search-in-the-loop is ever wanted, this is the cheap form.

### AlphaZero-inspired game learning: MCTS only at test time **[abs]**
arXiv:2204.13307 (Scheiermann & Konen). Trains without search and adds MCTS at test time; reports
that the test-time search still improves play substantially. Supports the E7 ordering: measure
the search-vs-no-search gap first, then decide whether to distil it.

### Self-play surveys and successors **[abs]**
arXiv:2408.01072 (self-play in RL, survey) and arXiv:2609.01549 (NashDreamer, model-based
Nash self-play). Background only; nothing changes the plan.

---

## 2. Hidden information that is weak — when determinization is legitimate

### Understanding the success of Perfect Information Monte Carlo sampling **[abs]**
Long, Sturtevant, Buro, Furtak, AAAI 2010. Characterises when PIMC (sample a consistent
full-information world, search it, average over samples) works and when it is fooled, using
synthetic game trees with three tunable properties:

- **leaf correlation** — how often sibling terminal positions that differ only in hidden
  information share a value;
- **bias** — how strongly the game favours one side regardless of hidden information;
- **disambiguation factor** — how quickly hidden information becomes public as play proceeds.

PIMC performs well with high disambiguation and high leaf correlation (trick-taking card games:
Skat, bridge) and poorly in the opposite corner (poker, Stratego). Twilight Struggle sits in the
favourable corner: cards are public when played, hand sizes are known, the deck is public, the z3
reconstruction shows hands are heavily determined, and most of the value is on the public board.
**Consequence:** determinized rollouts (uniform over hands consistent with the observation's
known-card tracking) are the right search for this game, not a compromise pending a belief net.

### Maven — world-championship-calibre Scrabble **[abs]**
Sheppard, Artificial Intelligence 134 (2002). Samples opponent racks, does shallow simulation,
evaluates with a learned static evaluation; superhuman without any belief modelling. The
precedent for treating a hidden hand by sampling in a game whose value is mostly public.

### Off-Belief Learning (already reviewed: `paper_off_belief_learning.md`)
OBL's concern — self-play conventions that break against humans — is real in Hanabi, where
communication is the whole game. In a weak-hidden-info game the exposure is much smaller; kept in
mind for the VOA-type decisions (inference about what the opponent may hold), not a priority.

---

## 3. Chance that is heavy — reduce variance in the target, not depth in the search

### TD-Gammon **[abs]**
Tesauro, *Communications of the ACM* 38(3), 1995 (and Neural Computation 1994). The original
chance-heavy success. Two lessons that hold up: under heavy dice noise, TD(λ) bootstrapping from
the value function is far better than Monte Carlo outcome targets; and once the value function is
good, 1–2-ply expectimax over the dice is enough for superhuman play — deep search bought little.
The value function does the work.

### Stochastic MuZero **[abs]**
Antonoglou, Schrittwieser, Ozair, Hubert, Silver, ICLR 2022. Chance nodes as first-class objects:
the model predicts an *afterstate* (after the decision, before the chance outcome) and a
distribution over discrete chance codes; backgammon, 2048, and stochastic Go-like domains. The
point for us: value belongs on afterstates and chance should be averaged out explicitly, not
sampled once. The engine already gives us both objects: every roll is a `ROLL_DIE` node that
accepts a forced die value (`engine/src/state_machine.cpp`, `RollType` switch), and the deal is
itself a `ROLL_DIE` node of type `TURN_CLEANUP`, i.e. an explicit pre-deal afterstate.

### GAE falls short in imperfect-information self-play RL (Q-boosting / VRPO) **[full]**
arXiv:2605.19235. Observation: even with a perfect state-value critic, GAE's advantage estimate
retains variance from *sampling* the future along the trajectory (future actions, and by
extension chance) instead of taking the expectation. Fix: a centralised action-value critic with a
multi-step Expected-SARSA(λ) trace ("Q-boosting"), packaged as VRPO; reported to beat PerfectDou
with 40% of the training steps and to gain +33 mBB/hand against Slumbot. For us the dominant
sampled-future noise is the die, and dice are enumerable: the 6 (or 36) outcomes of a pending roll
can be stepped on clones of a 4 KB state and averaged exactly. → P2 "chance-aware targets".
(A first fetch of this paper's PDF returned a fabricated summary; the HTML version was used.)

### Suphx — global reward prediction (already reviewed: `paper_suphx_mahjong.md`)
The round-boundary value baseline that stops per-round credit being swamped by tile luck across
rounds. The analogue is a critic trained to be accurate at the pre-deal `TURN_CLEANUP` state, with
the λ-return truncated and bootstrapped there rather than sampled through one random hand.

### AIVAT — variance reduction for evaluation **[abs]**
Burch, Schmid, Moravčík, Morrill, Bowling, AAAI 2018. Subtracts value-based control variates at
chance nodes and at the opponent's decision nodes; unbiased, and cuts evaluation variance by an
order of magnitude in poker. Our paired-die-stream tournaments are a crude version
(`../log/variance_and_noise.md`);
the full form would let 1,000-game tournaments resolve ~5 Elo instead of ~20. Worth adopting once
arms land within noise of each other.

---

## 4. Long horizon and global coupling

### Potential-based reward shaping **[abs]**
Ng, Harada, Russell, ICML 1999. Shaping by F = γΦ(s′) − Φ(s) leaves the optimal policy unchanged.
With γ = 1 the shaped return is outcome − Φ(s_t): advantages are unchanged under a perfect critic
and the benefit is purely the credit path through the immediate term in the trace; the risk is
learning Φ's mistakes while shaping is on (hence annealing). Our existing Φ
(`Scoring::compute_useful_actions_potential`) prices battleground control as the same step
function the critic learned (`experiments.md` §12), so it is not an improvement over the critic
until its battleground term is made monotone in the gap to control (§12.4). Kept in reserve.

### Settlers of Catan / Dominion self-play RL **[abs]**
Several small-scale works (PPO self-play, single-GPU scale) reach "intermediate club" level and
plateau with terminal-only reward — the same failure profile as arm E here. Nothing to borrow
beyond the confirmation that vanilla PPO with sparse terminal reward is the plateau, not the
ceiling. Not reviewed in detail; no specific paper is relied on.

### Receptive field **[derived, not a paper]**
Ataraxos and AlphaStar encode the board with a transformer so every unit attends to every other.
Our backbone is a 3-layer graph convolution over the 84-country map, so a country's embedding sees
three hops before global pooling; Europe and South America only meet in the pooled vector, and
Poland's value depends on the Europe scoring card, DEFCON and the VP track. `experiments.md` §5
finds capacity is not the current bottleneck on v2, so this stays behind the signal fixes — but it
is the one architectural change with a specific mechanism behind it.

---

## 5. Mapping: which ingredient answers which failure

| measured failure (`experiments.md`) | property of the game | ingredient | plan stage |
|:---|:---|:---|:---|
| critic prices control as a step function (§12); setup ignores Poland / W. Germany | long horizon, credit path through micro-actions | categorical value head; macro-action credit (λ=1 inside a placement block, bootstrap at its boundary); MC-return setup target (Ataraxos) | P1, P4 |
| critic over-optimistic at forced wins (§4), pessimistic on human boards (§17) | chance-heavy | exact dice expectation in targets (Q-boosting/Stochastic MuZero); pre-deal bootstrapping (Suphx GRP) | P2 |
| empty battlegrounds plateau at ~7.5 from turn 8 (§18.2) | global coupling, terminal-only reward | pre-deal critic that prices the board; attention over country+card tokens if the perturbation probe stays flat | P2, P6 |
| games end turn ~6 with DEFCON-1 (§18.2) | chance-heavy (coups) | dice expectation in targets; categorical head | P1, P2 |
| unknown: is strength in the value function but not the policy? | weak hidden info | determinized search (PIMC, Maven) as a diagnostic; expert iteration if the gap is large | P3 |
| deal luck in value targets | chance-heavy, weak hidden info | oracle critic (Suphx) as a deal-side variance reducer — implemented, never measured | P5 |
| tournament noise ~20 Elo (`../log/variance_and_noise.md`) | chance-heavy | AIVAT-style baselines at deal and die nodes | reserve |

**One-line summary.** The game is backgammon-shaped, not Stratego-shaped: the parts of the modern
recipe that transfer are the value-centric ones (categorical value, advantage filtering, MC-return
setup, no search in the loop), the belief and equilibrium machinery is optional, and the missing
ingredient is taking expectations over chance in the training target, which the engine's explicit
`ROLL_DIE` and `TURN_CLEANUP` nodes make exact and cheap.

---

## 6. Setup: the hand is not the obstacle

Setup is placed after the first deal (`rules/rules.md` §3, `engine/src/state_machine.cpp`
`init_new_game`), so it is conditioned on the mover's own 8 cards; the opponent's hand and every
later deal are unknown; the US additionally sees the USSR placement. Three things are easily
conflated here:

1. **Own hand is known.** The policy sees it, and the expectation over own hands is what SGD over
   ~230k games per 80M steps computes; a hand-conditional setup policy is something the network
   fits, not a sum to evaluate. Poland ≥ 3 / West Germany ≥ 4 is good in expectation over almost
   every hand, which is why humans do it near-unconditionally; the hand moves the last placement
   or two.
2. **Opponent hand and future deals are sampled, not enumerated** (§2 above). A few dozen
   consistent opponent hands, each rolled out to the end of turn 1–2 and evaluated with the
   critic, give an estimate whose error is dominated by the critic, not the sample. Setup is one
   decision per game with a small sensible candidate set, so even a search costing seconds is
   free amortised over a game.
3. **The credit path is what breaks.** USSR setup is six consecutive `PLACE_INFLUENCE`
   micro-actions; under a step-function critic the first two into Poland have zero measured
   advantage and the third gets everything, and the policy never reaches the third. Fixes: a
   monotone (categorical) critic, and macro-action credit — no bootstrapping between the
   micro-actions of one placement block, one bootstrap at the first real position after it (the
   turn-1 headline state). The same boundary rule applies to the placement sequence inside any
   ops play. This is a `rollout_buffer.py` change, not an engine change.

Acceptance is a probe, not Elo: the rate of Poland ≥ 3 (USSR) and West Germany ≥ 4 (US) over
~2,000 openings, with the 266-game human corpus as the yardstick.

---

## 7. Budgeting note

### What an 80M screen does and does not measure

A screen at 80M measures **how fast an approach learns the basics**, not what it converges to.
Those are different quantities and the literature is clear that they can disagree.

**Scaling Scaling Laws with Board Games** — Jones, arXiv:2104.03113, and the closest published
setting to ours. Training AlphaZero on Hex, **Elo is linear in the log of training compute**,
with a slope that stays constant across board sizes, so the compute needed for a target strength
can be read off in advance. Train-time and test-time compute trade off against each other on a
simple relationship. The practical consequence for this repo: under a log-linear law an
intervention can move the **intercept** (reaches a given strength sooner, same slope) or the
**slope** (changes the rate, so the gap grows or inverts with budget). Every number we have is an
intercept measurement at one budget. We have never measured a slope, and the two have completely
different implications for whether a change is worth adopting. See also *AlphaZero Neural Scaling
and Zipf's Law* (arXiv:2412.11979) for power laws in the same family of agents.

**How often does the small-scale winner win at scale?** *DataDecide* (arXiv:2504.11393) measures
this directly for pretraining-data choices: experiments at **100-200x less compute pick the
large-scale winner about 92% of the time**. That is reassuring for screening as a method and is
also a real 8% failure rate, concentrated where behaviour is non-smooth or emergent rather than
where it is gradual.

**Curve crossing is a named, documented phenomenon**, not a hypothetical: *The Shape of Learning
Curves: a Review* (arXiv:2103.10948) catalogues curves where one method has the lower error early
and another the lower error asymptotically, and notes crossings are likelier to be found over
small windows than large ones — which is what a screen is.

**In RL the tradeoff is a standard shape.** Model-based methods are more sample-efficient and
reach a lower asymptote than model-free ones, and algorithm families exist specifically to get
both at once (e.g. Aggressive Q-Learning with Ensembles, arXiv:2111.09159). "Better early, no
better at the ceiling" is the expected pattern for a class of changes, not an exotic risk.

**So: read a screen as a rate, and say so.** An arm that wins at 80M has demonstrated that it
learns the basics faster. Claiming it is *better* requires either a second budget or an argument
that the mechanism does not saturate. Entropy is the cheapest early warning — a cell running at
materially lower entropy has committed sooner, which is what fast early learning looks like and
also what an earlier plateau looks like.



An 80M-step arm is ~1.5 h on the 4090 and gains on the current recipe are still measurable at
320M (`experiments.md` §24). Screen every factor at 2 seeds × 80M (3 h); confirm only the winner
of a screen at 2 seeds × 240M (9 h); rate the last four snapshots and pool all sixteen
pairings; never compare across budgets or tournaments. A leg's gain carries ±16 Elo
(`../log/variance_and_noise.md`), so within-lineage trends need that bar. The current ordering and budget
rule live in [`../plans/README.md`](../plans/README.md).
