# The perspective negation is load-bearing for GAE

`compute_gae` bootstraps across a change of mover by negating the next step's value:

```python
sign     = curr_p * next_p          # -1 when the mover changes
next_val = sign * values_win[t + 1]
```

That assumes `V(s, me) = -V(s, opponent)`. **The assumption is false here, and replacing it makes
things much worse.** Both halves of that are measured below.

## The assumption is false

`v_win` is computed from `extract_observation(state, perspective)`, which hides the opponent's
hand, so `V(s, US)` and `V(s, USSR)` evaluate two different information sets rather than one state.
Over 227 positions sampled from self-play replays with the 240M checkpoint, `v_US + v_USSR` — zero
if the identity held:

| | |
|:---|---:|
| mean | +0.051 |
| mean absolute | **0.144** |
| p90 absolute | 0.374 |
| max absolute | 0.849 |
| positions disagreeing by more than 0.2 | 28.6% |

The error lands hardest where a player takes the last decision of its turn — a card played for its
event, resolving with no further decisions — because that advantage is then priced entirely by the
*opponent's* opinion of the result, formed without seeing the deciding player's hand.

## Replacing it is worse, and the reason is the estimator

`--same-perspective-bootstrap` stores `V(s_{t+1}, p_t)` — the next state as the player who just
moved sees it — and uses it directly, with no sign. `E3-21-28` ran that against `E3-20-28` at
matched seed and matched everything else.

The critic got steadily worse, and the gap widened rather than closing:

| window | AUC base | AUC arm | Brier base | Brier arm |
|:---|---:|---:|---:|---:|
| first third | 0.6185 | 0.5296 | +0.030 | −0.345 |
| middle | 0.6892 | 0.5725 | +0.116 | −0.247 |
| last third | 0.7089 | **0.5547** | +0.137 | **−0.373** |

Both are scored against actual game outcomes, so they compare across arms; `explained_variance`
does not, because the bootstrap changes the target it measures fit against.

### It is not the wiring, and not the values

Two hypotheses were tested and rejected before the real one was found.

**The stored bootstrap is correct.** Against the negated version it correlates **0.956** overall,
**1.000** where the mover does not change (the two definitions are identical there by
construction), and **0.785** where it does, differing by 0.175 — matching the asymmetry measured
independently above.

**The non-mover perspective is not an unsupervised wasteland.** Only mover-perspective observations
enter a training batch, so `V(·, non-mover)` never receives a gradient — but a trained critic
generalises across the flip almost perfectly. On policy-visited positions with the 160M checkpoint:

| evaluated from | AUC | Brier skill |
|:---|---:|---:|
| the mover (trained) | 0.9301 | 0.5592 |
| the non-mover (never trained) | 0.9214 | 0.5293 |

### It is the telescoping

GAE is a sum of discounted TD errors and closes only if every δ is written against one value
function:

```
A_t = δ_t + γλ · sign · A_{t+1}
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

Under the default, δ_t's forward term is `sign·V(s_{t+1}, p_{t+1})` — the same quantity δ_{t+1}
subtracts as its `V(s_t)`. They cancel and the sum telescopes to the return.

Under the same-perspective bootstrap, δ_t carries `V(s_{t+1}, p_t)` while δ_{t+1} subtracts
`V(s_{t+1}, p_{t+1})`. Those differ by 0.175, so **the telescope does not close**, and the residual
is injected at every change of mover and carried with weight (γλ)^k — λ = 0.98 over ~400-step
games.

Measured on 11,847 steps that reached a realised outcome, comparing each method's `returns_win`
against what actually happened:

| | bias | RMSE | corr with outcome |
|:---|---:|---:|---:|
| default (negated) | −0.0073 | **0.551** | **0.861** |
| same-perspective | −0.0031 | 0.789 | 0.621 |

**Not a bias problem — a variance problem.** The same-perspective bias is marginally smaller; its
RMSE is 43% worse and its correlation with the outcome falls from 0.86 to 0.62. The two methods'
return targets differ by mean 0.275 and **max 3.86**, on a scale where the outcome is ±1. A bounded
per-step difference (0.175) producing an unbounded return difference is accumulation, which is what
a broken telescope looks like.

## Status

`--same-perspective-bootstrap` stays in the code, defaulted off, as the reproduction for this
result. `E3-21-28` was stopped at **40,632,320 steps** rather than run to 160M once the mechanism
was established; the run directory `E3-21-28_20260916_092205` is kept as the artifact, with
`resume_26673152steps.pt` and the metric series.

At the stop the critic had reached `critic_auc` **0.5133** — indistinguishable from chance —
against a baseline at ~0.71 and still climbing at the same step count.

## What to do instead

The negation is not an approximation the project got away with; it is what makes GAE valid in an
alternating-move game. Trading it for a pointwise-better value estimate buys a small bias
correction and pays for it with a much noisier regression target.

### What the fix is not

An auxiliary loss pulling `V(s, p)` toward `−V(s, p̄)` was proposed and **withdrawn before being
built**. It would force the perfect-information identity onto quantities that are not supposed to
satisfy it: the two players hold different information, so the position genuinely is worth
different amounts to them, and the 0.144 asymmetry is signal rather than error. Training it away
would destroy information to protect an estimator.

The real choice is between two coherent positions, where the code currently sits between them —
it computes information-set values and combines them with a world-state identity:

| commit to | means | cost |
|:---|:---|:---|
| **world-state value** | a centralised critic that sees the true state, both hands included. Antisymmetry then holds by construction and the telescope is valid. Suphx's oracle guiding; MADDPG/COMA | queued as [`../../plans/P5_oracle_critic.md`](../../plans/P5_oracle_critic.md); old implementation recoverable from `574e04a` |
| **information-set value** | per-player trajectories: each player's own GAE over its own decision points, bootstrapping from its next *own* decision, opponent rewards folded in. No cross-perspective bootstrap occurs at all | contained to `compute_gae`, no extra forward passes |

A belief-conditioned value (ReBeL, Student of Games —
[`../../papers/README.md`](../../papers/README.md)) is the principled third option and much heavier
than this problem warrants.

**Cheap diagnostic first:** `gae_lambda = 1.0` removes the cross-perspective term from the
estimator entirely. If the critic gap disappears there, the bootstrap is the whole story and the
run puts a number on what it costs. That has not been run.

---

## The per-player option was built, run, and is decisively worse (E3-22-28, 2026-09-16)

The second row of that table — per-player trajectories — was implemented behind
`--per-player-gae` and run against `E3-20-28` as a matched baseline. **It loses by 520 Elo at
80M steps.** The idea is refuted, not the implementation.

### Elo, on one anchored scale

`tools/tournament.py`, 500 games per side per pair, 45,000 games, anchored on `HeuristicBot` at
1500. Report: `data/tournaments/E3-22-28_vs_E3-20-28/`.

| steps | baseline Elo | per-player Elo | gap | baseline's head-to-head win rate |
|---:|---:|---:|---:|---:|
| 20M | 1880.9 | 1659.2 | **−221.7** | 82.0% |
| 40M | 2076.3 | 1720.7 | **−355.6** | 89.0% |
| 60M | 2180.8 | 1839.3 | **−341.5** | 88.3% |
| 80M | 2274.8 | 1754.4 | **−520.4** | 96.1% |

Two things beyond the size of the gap. The arm **regressed** between 60M and 80M (1839 → 1754)
while the baseline was still climbing steeply. And `arm_80M` is weaker than `base_20M` — four
times the compute for a worse player.

### The mechanism, measured on identical data

`adv_std_raw` ran 40–55% above the baseline for the whole run, but each arm measures that on its
own rollouts from its own policy, so it cannot separate "this estimator is noisier" from "this
policy reached noisier states". Computing **both estimators over one rollout** from the same
`base_80M` checkpoint separates them:

| | interleaved | per-player |
|:---|---:|---:|
| raw advantage SD | 0.2314 | **0.2963** (1.28x) |
| return-target SD | 0.5412 | 0.5376 |
| returns within [−1, 1] | 100% | 100% |
| correlation between the two estimators' targets | 0.979 | 0.979 |

The **value targets are fine** — same spread, same bounds, 0.979 correlated with the estimator
that works. It is the **advantages** that are noisier, by 28% on identical data, and advantages
are what the policy gradient consumes. In training the gap widened to 40–55% because a noisier
gradient produces a worse policy which reaches noisier states.

So this is not the failure E3-21-28 had. That one broke telescoping and wrecked the regression
target. This one telescopes exactly (pinned at λ=1 in `tests/training/test_per_player_gae.py`)
and produces sane targets — and still destroys the policy, through variance alone.

### The owner called this in advance

From the conversation that proposed the change:

> It is entirely possible that we made a good decision, board looked good after it, but opponent
> did something strange / got a lot of luck, and made our position significantly worse. This
> change will be then attributed to the last micro action, even though it has nothing to do with
> the change.

That is exactly what the 1.28x measures. Per-player delta spans the opponent's move and any dice,
so the opponent's variance lands on the mover's advantage at full weight; the interleaved form
puts it in the following delta where λ damps it.

**I answered that objection with the wrong measurement.** Pre-launch I measured *one-step*
advantage spread at a frozen policy — 0.0882 per-player against 0.0940 interleaved — and
concluded variance would improve. One-step spread is not what the policy gradient consumes, and
measured on the actual quantity the sign reverses.

### The transferable lesson

**A better offline return estimate is not a better training signal.** Per-player GAE estimates
returns *more accurately* than the default — RMSE 0.551 → 0.481, correlation with the realised
outcome 0.861 → 0.901 at λ=0.98, and tighter exact telescoping at λ=1 (0.162 vs 0.214). Every one
of those numbers favours it, and it loses by 520 Elo. Accuracy of the target and variance of the
advantage are different properties, and only the second reaches the policy.

Anything proposed on offline estimator quality alone should be checked for advantage variance on
a shared rollout before a run is spent on it. That check is nine lines and would have cost
minutes instead of two runs.

### What is still open

`gae_lambda = 1.0` remains unrun, and is now the cheapest remaining probe of the bootstrap's
cost. The oracle critic ([`../../plans/P5_oracle_critic.md`](../../plans/P5_oracle_critic.md)) is
untouched by this result — it removes the asymmetry at its source rather than routing around it,
and is the row of the table above that has not been tested.
