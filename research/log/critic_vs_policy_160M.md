# Policy or critic? The counterfactual rounds, and what the finished 160M run was worth

Records §15–§18 of the original `experiments.md`, the second half of the diagnosis run against
`run_v2_20260907_long160M` (snapshots to 160M; provenance in
[`critic_positional_value.md`](critic_positional_value.md) §12.0), post-E1 engine, legacy
observation. §15 asks on real held positions whether the policy or the critic is at fault and finds
the answer differs by card; §16 puts the human's own executed board against the model's round and
concludes the critic is wrong about the card; §17 plays both boards out, finds §16 over-stated, and
establishes that the critic is over-pessimistic about human boards by 7–10 win-rate points; §18 is
the finished run — what `v_win` is really predicting, the tournament where it ties rather than beats
the previous best, and 3.7 points of extra human agreement bought for no Elo. Note that §22, in
[`P7_human_injection_and_its_cost.md`](P7_human_injection_and_its_cost.md), later invalidates the
*magnitude* of §18.4's reading, because this whole run carried the injection handicap. **This file
is append-only history**: §16's over-statement and §17.3's correction of it both stay as written,
and nothing is edited to match current belief.

---

## 15. Policy or critic? Asked on real held positions — and the answer differs by card

Two corrections to §14 first, both of which change numbers.

**Real positions, not constructed ones.** §14's US arm swapped the card into an arbitrary US node.
That answers a weaker question: the hand was never dealt to anyone, and the swap has to discard
something to make room, so the rest of the hand is wrong too. The corpus has real US-held positions
in quantity — **1029 for Decolonization, 636 for De-Stalinization** — and on those the model is
markedly better than the swap suggested:

| | swapped-in (§14.2) | really held |
|---|---|---|
| US spaces Decolonization, 70M | 49.2% | **68.5%** |
| US spaces Decolonization, 100M | 51.6% | **65.0%** |
| US spaces De-Stalinization, 100M | 59.9% | **67.0%** |

So §14 understated US competence by roughly 15 points. The USSR figures are unaffected — that arm
always used real hands — and 100M Decolonization cross-checks exactly: 102 of 270 is the 37.8%
already reported.

**The method.** From each real held position, resolve the card the human's way (event for the USSR,
space for the US) and the model's greedy way, with the model making every follow-on choice in
*both* branches so the mode is the only difference, then read `v_win` from the mover's own side.
`gap` is v(human) − v(model): positive means the critic prefers the human's line. The rightmost
column counts, among positions where the *policy* chose against the human, how often the *critic*
still preferred the human's. Snapshots to 140M.

### 15.1 Decolonization: the critic knows and the policy ignores it

USSR really holding it, 276 positions:

| snapshot | policy agrees | gap | critic sides with human |
|---|---|---|---|
| warm start | 21% | **−0.276** ± 0.051 | 50% |
| 35M | 9% | +0.140 ± 0.006 | 92% |
| 70M | 23% | +0.029 ± 0.005 | 69% |
| 100M | 38% | +0.079 ± 0.011 | 73% |
| 140M | 38% | **+0.225** ± 0.013 | **93%** |

At 140M the critic prefers firing the event in **93% of the positions where the policy chose not
to**, by a wide and well-determined margin — and the policy plays it for Ops anyway in 62% of
positions. For this card the two heads have come apart: **the value function has learned the right
answer and the policy is not following it.** That is the encouraging case, because it is what
policy imitation, or simply more of the same RL, can close.

Note also the warm start: gap −0.276, critic siding with the human only 50% of the time. The
behaviour-cloned init has *neither* head right, which corroborates §14.2's finding that BC never
learned this decision at all.

### 15.2 De-Stalinization: the critic does not know

USSR really holding it, 451 positions:

| snapshot | policy agrees | gap | critic sides with human |
|---|---|---|---|
| warm start | 18% | −0.176 ± 0.043 | 51% |
| 35M | 6% | +0.028 ± 0.004 | 65% |
| 70M | 12% | −0.018 ± 0.003 | 42% |
| 100M | 32% | −0.040 ± 0.007 | **32%** |
| 140M | 32% | +0.039 ± 0.008 | 59% |

The gap changes sign three times and at 100M the critic actively *prefers* the Ops line, siding
with the human in under a third of disagreements. **Both heads are wrong here**, and no amount of
policy imitation will hold a behaviour the critic scores as a mistake.

**The likely reason is not that the critic misprices the mode — it is that the model cannot execute
the card.** De-Stalinization asks for eight choices, four removals and four placements, against
Decolonization's four placements into one restricted region. §14.3 shows the execution: at 100M the
model fires De-Stalinization by stripping **East Germany ×259**, a battleground it controls and
needs for Eastern Europe, and scattering into Cameroon, Lebanon and Guatemala. A critic that scores
*that* below taking three Ops is not obviously wrong. Both branches here are resolved by the model,
so what the comparison shows is the value of the event **as this model would play it**, and for
De-Stalinization that is genuinely bad.

This makes the two cards a clean pair rather than a contradiction: Decolonization is hard to botch,
so the critic can price the mode and the policy is the only thing lagging; De-Stalinization is easy
to botch, the model botches it, and the critic prices the botched version accurately.

### 15.3 The US side: both heads improving, together

| | policy agrees (space) | gap | critic sides with human |
|---|---|---|---|
| Decolonization, warm start | 8% | −0.074 ± 0.020 | 44% |
| Decolonization, 140M | 55% | +0.063 ± 0.005 | 75% |
| De-Stalinization, warm start | 11% | −0.001 ± 0.026 | 51% |
| De-Stalinization, 140M | 59% | +0.055 ± 0.007 | 75% |

Both start with the critic mildly preferring the Ops line and end with it preferring space three
times out of four, while the policy moves from ~10% correct to ~55–59%. This is the one place in
§12–§15 where RL improves both heads steadily and in the same direction, and it is also the only
decision whose cost lands inside the same game — handing the opponent a free event is punished
immediately, which is exactly the credit-assignment argument §12.4 makes.

### 15.4 What to do with this

* **Decolonization is a policy-side fix.** The critic's own ranking already carries the answer at
  140M. Anything that makes the policy follow its own value estimate more closely — imitation,
  a sharper advantage, more training — should move it.
* **De-Stalinization is an execution fix first.** Teaching the policy to fire an event it plays
  badly makes things worse, and the critic is right to say so. The prerequisite is the §12.4
  shaping question: a value that is continuous in distance-to-control would price the removals out
  of East Germany correctly, and only then is firing the event worth imitating.
* The confound is stated above and is not removable by this method: both branches are resolved by
  the model, so a negative gap cannot distinguish "the critic misprices the mode" from "the critic
  correctly dislikes the model's execution". Separating them needs the human's own targets replayed
  from the log, which the converter has and this measurement does not yet use.

## 16. The human's own board against the model's round — the critic is wrong about the card

§15.2 found the critic scoring De-Stalinization below Ops and offered an excuse for it: both
branches there were resolved by the model, and the model executes that card badly (§14.3, East
Germany stripped 259 times), so a critic that dislikes the result might be right about the
execution rather than wrong about the card. **That excuse does not survive the test.**

**Method.** A second seam on `convert_game`, `on_entry`, hands over the board *after* a human entry
has been replayed and checked against the log's own next position — the humans' actual removals and
placements, not a model's replay of their mode choice. From the identical pre-round board the model
then plays that Action Round however it likes: its own card, its own mode, its own targets, with no
obligation to touch De-Stalinization. Both boards are scored by the same value head from the USSR's
side. `ai/eval/round_counterfactual.py`, 274 deduplicated corpus games, early war.

*Comparability check:* in all 105 rounds both boards end on the same turn, the same `action_round`,
and in `Phase.ACTION_ROUND` — so the gap measures the play, not where the two branches stopped.

### 16.1 De-Stalinization: 105 real human rounds

| snapshot | v(human) | v(model) | gap | critic prefers human | model played it too |
|---|---|---|---|---|---|
| warm start | −0.263 | −0.241 | −0.022 ± 0.051 | 51% | 87/105 |
| 35M | +0.055 | +0.110 | −0.055 ± 0.010 | 32% | 36/105 |
| 70M | −0.157 | −0.100 | −0.057 ± 0.009 | 28% | 25/105 |
| 100M | −0.316 | −0.158 | **−0.158** ± 0.022 | **21%** | 7/105 |
| 140M | −0.010 | +0.087 | −0.098 ± 0.016 | 28% | 23/105 |

**Every snapshot prefers its own round to a real human De-Stalinization**, and the margin is widest
at 100M, where the critic sides with the human in 21% of rounds. With the humans' own competent
execution on the board, the critic still says its own line is better. So this is not the model
mispricing its own bad targeting. **The critic is wrong about the card.**

### 16.2 What it prefers instead, and this is the alarming part

The card the model chooses instead is not the whole story — spacing an opponent's card is correct
play, and only firing its event is an error — so the mode has to be read too. In those same 105
rounds:

| | 100M | 140M |
|---|---|---|
| **US card played for Ops** (fires the US event) | **38.1%** | **34.3%** |
| US card spaced (correct) | 21.9% | 8.6% |
| USSR card played as event | 7.6% | 12.4% |
| USSR card played for Ops | 6.7% | 22.9% |

Most common single choices at 100M: Truman Doctrine [US] for Ops ×16, Five Year Plan [US] spaced
×11, Five Year Plan [US] for Ops ×8, Special Relationship [US] for Ops ×6, Defectors [US] for Ops
×5. At 140M the top pick is its own De-Stalinization played for **Ops** ×16, then Truman Doctrine
[US] for Ops ×13 and Five Year Plan [US] for Ops ×10.

So in the modal case the USSR declines its own strongest early event in order to play a *US* card
for Operations — firing Truman Doctrine or Five Year Plan against itself — and the value head rates
the resulting board above the human's. §14 found the US doing this with USSR cards; the error is
symmetric, it runs in both directions, and the critic endorses it.

### 16.3 This refines §15, it does not contradict it

§15 asked a within-card question: *given* that you play this card, is event better than Ops? On
that question the critic is right about Decolonization (93% of disagreements at 140M, gap +0.225).
§16 asks a between-card question: is playing the card at all better than what else the model would
do? There the same critic is near-neutral on Decolonization —

| snapshot | gap | critic prefers human |
|---|---|---|
| warm start | −0.015 ± 0.050 | 41% |
| 35M | +0.012 ± 0.012 | 57% |
| 70M | −0.008 ± 0.013 | 47% |
| 100M | −0.035 ± 0.017 | 49% |
| 140M | +0.041 ± 0.019 | 63% |

— drifting positive only by 140M, and much weaker than the within-card result. Both readings are
true at once: the critic has learned how to play a card it has decided to play, and has not learned
which card to play. That is a narrower and more useful statement than either section alone.

### 16.4 Consequences

* **§15.4's split was wrong about De-Stalinization.** I proposed execution as the prerequisite. It
  is not sufficient: even with human execution on the board the critic prefers its own line, so
  fixing targeting alone would leave the value function still steering away from the card.
* **Opponent-card discipline is the bigger error and it is bidirectional.** A third of rounds at
  both 100M and 140M are a US card played for Ops by the USSR. §11 measured what the space-race
  dominance error costs (~3 points); this one has never been costed and looks larger, since it
  hands over a full event rather than a dominated space.
* **The critic, not the policy, is the thing to fix here.** Imitation, injection and longer RL all
  push the policy toward the human line while the value function pushes back. §12.4's shaping
  proposal is aimed at the same organ, and this is a second, independent reason to take it up.
* Open, and now the cheapest thing to measure: the cost of playing an opponent's card for Ops,
  using the fork-and-play-out method in `ai/eval/dominance_cost.py`. If it is worth several points,
  it outranks everything in §9–§11.

## 17. Playing both boards out — the on-policy defence is half right, and §16 was over-stated

§16 concluded "the critic is wrong about the card" from value gaps alone. That inference has a hole:
`v_win` is an **on-policy** estimate. If this policy will not defend or build on De-Stalinization's
spread, the spread really is worth less to it than to a human, and a critic reporting that is
accurate rather than miscalibrated. §12 gives every reason to expect exactly that.

The only way to separate the two is to finish both boards under the model's own policy and see which
actually wins. 12 rollouts per board, temperature 0.1, `ai/eval/round_counterfactual.py`.

**Paired per position.** Pooling 12 rollouts of 105 positions and quoting a binomial error over
~1260 games overstates precision badly: twelve rollouts of one position are not twelve independent
games, and the unit that repeats is the position. Differencing the two arms position by position
also removes the position's own difficulty, which dominates the variance. All figures below are
per-position means with a standard error over positions.

### 17.1 De-Stalinization, 105 positions

| snapshot | realized gap | critic predicted | critic error |
|---|---|---|---|
| 100M | **−0.91** ± 2.56 | −7.89 ± 1.08 | **−6.98** ± 2.47 |
| 140M | **+5.46** ± 2.75 | −4.88 ± 0.80 | **−10.34** ± 2.88 |
| 160M final | **+5.11** ± 2.83 | −3.42 ± 0.73 | **−8.53** ± 2.87 |

Two findings, and they point in different directions.

**The on-policy defence holds at 100M.** The human's De-Stalinization board is worth −0.91 ± 2.56 to
that policy — indistinguishable from nothing. A human's competent spread genuinely bought the 100M
model no wins at all. The critic's *sign* was right, and the intuition behind it is right: this
policy could not use the position.

**It stops holding after that, and the critic never notices.** By 140M and 160M the same human
boards are worth **+5.46** and **+5.11** win-rate points — the policy learned to capitalize — while
the critic still predicts −4.88 and −3.42. Its error is −6.98, −10.34 and −8.53 points, every one of
them 2.8σ or more. So the critic is not merely reporting a weak policy; it is **systematically
over-pessimistic about human boards by 7–10 win-rate points**, and it did not update when the policy
improved underneath it.

For scale: §11 measured the space-race dominance error at ~3.12 points and treated that as the
ceiling on a whole line of work. A human's De-Stalinization round is worth **+5.11 points** to the
finished model, and the model plays that card as the human would in a minority of positions.

### 17.2 Decolonization, 79 positions — the critic is roughly right

| snapshot | realized gap | critic predicted | critic error |
|---|---|---|---|
| 100M | +2.33 ± 2.25 | −1.76 ± 0.84 | −4.09 ± 2.39 |
| 140M | +1.34 ± 2.61 | +2.04 ± 0.96 | **+0.70** ± 2.74 |
| 160M final | +3.01 ± 2.56 | +0.44 ± 0.86 | −2.57 ± 2.65 |

By 140M the critic's error is +0.70 ± 2.74 — calibrated. **The miscalibration is card-specific**,
concentrated on De-Stalinization, which is the card with eight choices and the one §14.3 shows the
model executing worst.

### 17.3 Correcting §16.2 — the "opponent card for Ops" bucket was too crude

§16.2 reported that 38.1% of rounds at 100M were "a US card played for Operations by the USSR,
firing the US event against itself", and presented the whole bucket as error. That is wrong, and the
largest component of it is not an error at all. Measuring what each play actually costs the USSR
(before minus after, so positive means Influence lost):

| card | mode | n (100M) | Europe Influence lost | total | cards lost |
|---|---|---:|---:|---:|---:|
| Truman Doctrine [US] | ops | 16 | **+1.06** | +0.50 | 1.00 |
| Five Year Plan [US] | space | 11 | 0.00 | 0.00 | 1.00 |
| Five Year Plan [US] | ops | 8 | 0.00 | −2.75 | **2.00** |
| Special Relationship [US] | ops | 6 | −0.67 | −0.67 | 1.00 |
| Defectors [US] | ops | 5 | −0.20 | −1.80 | 1.00 |
| De-Stalinization [USSR] | ops | 4 | 0.00 | −3.00 | 1.00 |

**Truman Doctrine for Ops costs the USSR 1.06 Influence in Europe** (1.08 at 140M) and buys a
placement back — the card removes all USSR Influence from one non-US-controlled European country,
and the model is picking a country where it holds one. That is a reasonable price for the Ops, not a
blunder, and it is 16 of the 40 plays §16.2 counted at 100M and 13 of 36 at 140M. Those should never
have been in the error column.

Two entries do belong there, for reasons an Influence count does not show:

* **Five Year Plan for Ops** gains 2.75 Influence but loses **two** cards — the card played plus the
  random discard its event forces. The cost is the discard, not the board.
* **Five Year Plan spaced** costs no Influence at all, which is exactly why the Influence metric
  missed it: what it spends is the turn's space attempt, on a card whose Ops the USSR could have
  had. §16.2 scored these 11 plays as *correct* ("US card spaced"), and that was wrong in the other
  direction.

So the honest statement is narrower than §16.2's: the model does decline its own strongest early
event, and some of what it does instead is a real error, but the bucket cannot be scored by side and
mode alone. Each card needs its own adjudication, and two of the three largest components were
mis-scored — one as error when it is sound, one as sound when it is error.

### 17.4 Where this leaves it

* §16's headline stands for De-Stalinization but for a narrower reason than it gave: the critic is
  over-pessimistic about human boards by 7–10 points, significantly, and it is stale — the policy
  improved from 140M and the value function did not follow.
* The on-policy objection is real and was worth raising: at 100M it is the correct account, and any
  conclusion drawn from value gaps alone, in §15 or §16, is unsafe without a rollout behind it.
* Decolonization is calibrated, so this is not a general property of the critic. It is the card the
  model cannot execute that it also cannot price.
* The 160M run reached its budget (`snapshot_final.pt`, 160,038,912 steps) and its own final
  diagnostics say the §12 problem is untouched: `mean_final_turn` 6.4, `frac_reaching_turn9` 0.22,
  `empty_battlegrounds_turn8` 8.1. Longer RL did not rediscover the strategy. It did, however, move
  the realized value of a human De-Stalinization board from ~0 to +5 points, which is the policy
  learning to use a position it still will not create.

## 18. The finished 160M run: stronger agreement, no more strength

### 18.1 What the critic is actually predicting

The question is settled in the engine, not by measurement. Every abrupt ending writes ±20 into
`victory_points`:

* DEFCON-1 and Cuban Missile Crisis suicide — `ops.cpp:200,216`, `victory_points = ±20`, `GAME_OVER`;
* a held scoring card at turn end — `state_machine.cpp:574-576`, the same;
* a 20 VP win — the cap by definition;
* final scoring — the only ending that leaves VP interior.

`Engine::get_terminal_utility` (`engine.cpp:174`) is then `sign(victory_points)`. So the game does
reduce to *final VP, with DEFCON suicide and held scoring normalised to a ±20 result*, and `v_win`
regresses onto the sign of that.

An attempt to measure whether the learned head behaves more like win-probability or like VP margin
**failed to discriminate, and is reported as such**: at roughly 92% of terminals VP sits at ±20, so
the two comparators coincide. corr(v_win, win) = 0.421 against corr(v_win, VP/20) = 0.419 at 160M is
not evidence for either reading. Separating them needs the final-scoring subset alone, which is
~7% of games.

What the same run did establish:

| | warm start | 70M | 160M final |
|---|---|---|---|
| Brier against the actual win | **0.378** | 0.218 | **0.207** |
| corr(v_win, outcome) | 0.119 | 0.357 | 0.421 |
| share of games ending by DEFCON-1 | **70%** | 57% | **40%** |

The BC warm start's value head is *worse than always predicting even* (0.25) — behaviour cloning
fits the policy and leaves the critic actively misleading. RL repairs it. And 40–70% of self-play
games end in mutual destruction, against `mean_final_turn` 6.4.

### 18.2 Why more RL does not rediscover De-Stalinization

§17 showed the policy learning to *use* a human's De-Stalinization board (+5.11 points by 160M)
while still refusing to *create* one. The natural expectation is that the second follows the first
eventually. 18.1 says why it does not, at least not here.

In this model's own game distribution games end around turn 6 by DEFCON-1. Regional scoring that
would pay for spread Influence arrives at turns 8–10 and mostly never arrives at all. The critic is
not being irrational about the card; it is fitted to a world where positional investment is rarely
collected. That closes a loop: short games → positional value seldom realised → critic prices it low
→ policy never invests → games stay short and decided by coups and DEFCON.

This predicts that more of the same RL will not fix it, and the run agrees: 160M steps left
`empty_battlegrounds_turn8` at 8.1 and `frac_reaching_turn9` at 0.22. It also predicts where to
intervene — anything that makes games last (DEFCON discipline, the §12.4 shaping term) should move
the card play as a side effect, and is worth more than teaching the card directly.

### 18.3 Tournament: tied with the previous best, not ahead of it

500 games per pair (250 per side), `--auto-advance`, Bradley-Terry MLE Elo anchored on
HeuristicBot = 1500. Report at
`data/checkpoints/run_v2_20260907_long160M/vs_prior_runs.md`.

| rank | model | Elo | overall win rate |
|---:|---|---:|---:|
| 1 | `dec_turns40` | **1956.6** | 84.4% |
| 2 | **`run_v2_20260907_long160M`** | **1908.4** | 80.0% |
| 3 | `run_v2_blunder_aware_9h` | 1725.1 | 60.0% |
| 4 | `g_w1e1` | 1695.0 | 56.4% |
| 5 | `inj_every1` | 1633.5 | 49.1% |
| 6 | HeuristicBot | 1500.0 | 34.3% |
| 7 | `hum_inj1` | 1497.2 | 34.0% |
| 8 | RandomBot | 937.9 | 1.7% |

**Head to head the two leaders are tied**: `dec_turns40` takes 50.8% of 500 games against the 160M
run. The standard error on a 500-game win rate is 2.2 points, so 50.8% is indistinguishable from
even. The 48-point Elo gap comes from the rest of the matrix — `dec_turns40` beats the weaker field
harder (84.6% against `blunder_aware_9h` where the 160M run manages 73.2%).

Two things worth noting about that comparison. `dec_turns40` is the checkpoint that generated the
synthetic warmup set the 160M run was initialised from, so they are not independent lineages. And
`hum_inj1`, the most human-weighted injection arm, finishes *below HeuristicBot*.

### 18.4 Agreement rose and strength did not follow

Measured on 47 distinct genuinely-unseen games, 23,191 decisions. **The first attempt at this was
wrong and is worth recording**: scoring against a fresh seed-7 split of the deduplicated dataset gave
the warm start 61.45%, because these models were fitted under the split of the *old* 280-id dataset
and 58% of the new held-out set had been trained on. The valid evaluation set is the old split's own
held-out games, minus duplicate-leaked and duplicate-copy ids.

| checkpoint | agreement |
|---|---|
| warm start (BC) | **48.61%** |
| 5M | 38.80% |
| 35M | 38.27% |
| 100M | **32.29%** |
| 140M | 35.47% |
| **160M final** | **37.43%** |
| `dec_turns40` | 33.73% |
| `run_v2_blunder_aware_9h` | 29.86% |
| `inj_every1` | 34.66% |

48.61% reconciles with the 49.33% recorded in §9.13, which validates the measurement. The shape is
the familiar washout — 48.6% down to 32.3% at 100M — followed by a recovery to 37.4%.

**The finished run agrees with humans more than any prior checkpoint (37.43% against
`dec_turns40`'s 33.73%) and is not stronger than it.** That is the cleanest statement yet of the
problem §9–§11 kept circling: human agreement is not a proxy for strength on this axis. 3.7 points
of extra agreement bought nothing measurable in Elo, which is consistent with §11's finding that the
behaviours being imitated are worth a few points at most and with §17's finding that the largest
mispricing is in the critic rather than the policy.

*(Minor correction to §14.4: of the four held-out games named as leaking into training, two — 216
and 217 — are empty downloads carrying no decisions. The real leak is two games.)*
