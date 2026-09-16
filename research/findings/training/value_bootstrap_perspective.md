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
