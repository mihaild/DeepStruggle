# The lambda clock and each side's own view, on one rollout of the base (2026-11-20)

**Conclusion.**

* **A lambda counted per card play is the same estimator as the one counted per decision.** At equal return
  error its advantage spread is 0.2408 (decay at the play's boundary) or 0.2378 (spread evenly over the play's
  decisions) against 0.2371, and its advantages correlate 0.994-0.995 with the default's. Today's 0.98 a decision
  is about 0.921 a card play.
* **The per-decision clock is uneven, but not by much:** a card play is 4.07 decisions on average (2 to 6 for
  the middle 80%, 13 at the 99th percentile), so the weight halves after 6 to 11 card plays (median 9).
* **Per-player GAE matches the default's reach at lambda 0.96** (0.98 squared: a side moves at half the
  decisions), not 0.955; there the two give advantages correlated 0.986, differing by 0.040 a decision (the
  spread is 0.237), **equally on every type of decision**.
* **The two sides' values differ mostly because of what each holds, not because the critic is noisy:** the
  mover's value moves by 0.075 (SD) when its own hidden hand is redealt, and the true hand's value is 0.068 from
  the redeals' mean; `|v_US + v_USSR|` on the same rows is 0.106.
* **45% of the default's bootstrap weight sits on the other side's view,** which cannot see those cards.

**Question** (owner, 2026-11-20). Should lambda be fixed per action round instead of per micro-step; and does
it make sense to train the main with each side's own value chain. Proposal:
[`../plans/P50_own_value_chain.md`](../plans/P50_own_value_chain.md).

**Instrument.** Self-play of `E8.1-A8-R132-S45@11600M+R139@12400M+R155` (SWA 13,120-13,200M), sampled at
temperature 1, 400 games, 178,815 decisions (447 a game, 110 card plays a game), CPU, the running worktree's
engine build. A session probe (not in the repository), its estimators those of the running session's stage-0
probe; `data/reports/critic_lambda_clock_r155.json` (untracked). A card play is one side's action round (its
interrupts included); a turn's headline phase, its setup and its end are one unit each. gamma 1. Return error
is the RMSE of the estimator's value target against the mover's realised result: it falls as the target leans on
the result and is 0 at lambda 1, so estimators are compared at equal return error, never at equal lambda.

## 1. Decisions per card play

| | mean | p10 | p25 | p50 | p75 | p90 | p99 | max |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| decisions in a card play | 4.07 | 2 | 3 | 4 | 5 | 6 | 13 | 17 |
| the play's decay at 0.98 a decision | 0.921 | 0.960 | | 0.922 | | 0.886 | 0.769 | |

Card plays covered before the weight halves (34 decisions at 0.98), over decisions: 6 / 8 / 9 / 10 / 11 at the
5th / 25th / 50th / 75th / 95th percentile.

## 2. The frontier: advantage SD / return RMSE

| clock | | | | | | | |
|:---|:---|:---|:---|:---|:---|:---|:---|
| interleaved, per decision | 0.95: 0.156/0.823 | 0.97: 0.196/0.798 | 0.975: 0.214/0.785 | **0.98: 0.237/0.766** | 0.985: 0.271/0.736 | 0.99: 0.328/0.678 | |
| per-player, per own decision | 0.90: 0.151/0.825 | 0.94: 0.193/0.799 | 0.95: 0.211/0.786 | **0.96: 0.236/0.767** | 0.97: 0.271/0.737 | 0.98: 0.327/0.679 | |
| interleaved, per card play (decay at the boundary) | 0.85: 0.179/0.813 | 0.90: 0.215/0.787 | **0.92: 0.239/0.769** | 0.94: 0.272/0.738 | 0.95: 0.296/0.715 | 0.96: 0.328/0.681 | 0.97: 0.373/0.629 |
| per-player, per card play | 0.85: 0.185/0.811 | 0.90: 0.221/0.786 | **0.92: 0.244/0.767** | 0.94: 0.277/0.736 | 0.95: 0.301/0.713 | 0.96: 0.332/0.679 | 0.97: 0.377/0.627 |
| interleaved, per card play (Lambda^(1/n) a decision) | 0.85: 0.174/0.814 | 0.90: 0.211/0.788 | **0.92: 0.235/0.769** | 0.94: 0.269/0.739 | 0.95: 0.293/0.715 | 0.96: 0.325/0.682 | 0.97: 0.371/0.630 |
| each side's own-view chain, per decision | 0.95: 0.147/0.827 | 0.97: 0.189/0.801 | 0.975: 0.207/0.789 | **0.98: 0.231/0.770** | 0.985: 0.266/0.739 | 0.99: 0.322/0.682 | |

At the default's return error (0.766), by linear interpolation:

| clock | advantage SD | at lambda |
|:---|---:|---:|
| interleaved, per decision | 0.2371 | 0.980 |
| per-player, per own decision | 0.2368 | 0.9603 |
| interleaved, per card play, at the boundary | 0.2408 | 0.9214 |
| per-player, per card play | 0.2442 | 0.9203 |
| interleaved, per card play, spread evenly | 0.2378 | 0.9219 |
| own-view chain, per decision | 0.2347 | 0.9806 |

The own-view chain reads `V(s, non-mover)` from a network never trained on that view.

## 3. How far the estimators differ on the same decision, at matched reach

Against the default at 0.98: the correlation of the advantages, and the SD of their difference (the default's
advantage SD is 0.237).

| against | all | card choice | play mode | op mode | point |
|:---|:---|:---|:---|:---|:---|
| per-player, 0.96 | 0.986 / 0.040 | 0.986 / 0.039 | 0.987 / 0.039 | 0.988 / 0.038 | 0.985 / 0.040 |
| own-view chain, 0.98 | 0.982 / 0.045 | 0.982 / 0.045 | 0.982 / 0.045 | 0.983 / 0.044 | 0.981 / 0.046 |
| interleaved, per card play at the boundary, 0.92 | 0.994 / 0.026 | 0.994 / 0.026 | 0.994 / 0.026 | 0.994 / 0.026 | 0.994 / 0.027 |
| interleaved, per card play spread evenly, 0.92 | 0.995 / 0.024 | 0.995 / 0.024 | 0.995 / 0.023 | 0.995 / 0.024 | 0.995 / 0.024 |

## 4. How much a side's value depends on its own hand

On 1,646 action-round card choices (4% sampled), the mover's value read from its own view in 8 worlds where the
cards its opponent cannot see -- its hand and the draw deck -- are redealt (`ts.determinize` from the
opponent's side), the board and everything public unchanged.

| | |
|:---|---:|
| SD of the mover's value over the redeals, mean (median) | 0.075 (0.065) |
| the true hand's value against the redeals' mean, mean absolute | 0.068 |
| `\|v_US + v_USSR\|` on the same rows | 0.106 |
| `\|v_US + v_USSR\|` over all decisions | 0.149 |
| by action round, 1 to 7 | 0.085, 0.083, 0.079, 0.073, 0.069, 0.067, 0.071 |
| the default's bootstrap weight on the other side's view (0.98), mean | 0.448 |

Two independent hands each worth 0.075 (SD) would by themselves put the mean of `|v_US + v_USSR|` near 0.085;
0.106 is measured, so most of the two views' disagreement is what the sides hold.

**What these do not show.** Whether an estimator's advantages are closer to the true ones: return error against
the result cannot tell an own-view bootstrap from the other side's (each view knows one hand), and the 0.040
difference in section 3 is the same on every type of decision, so nothing here locates it in decisions about the
hand. A redealt hand is drawn from the unseen cards uniformly, which is not how hands are distributed late in a
turn (the cards a side kept are not a random draw). One rollout of one network.

**Earlier runs this bears on.** E3's per-player arm (-520 Elo at 80M) ran at lambda 0.98, the default's reach at
0.99 ([`../archive/E3_ladder/findings/value_bootstrap_perspective.md`](../archive/E3_ladder/findings/value_bootstrap_perspective.md));
the default itself at 0.99 from scratch lost 242 and 286 Elo at 80M
([`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md)). Per-player GAE at matched reach has not been run.

## The run: per-player GAE at matched reach against the base continued (2026-11-20)

**Verdict: a fail on the plan's bar, with no cost found.** The arm scores 51.5% ± 0.8 against the control at
+800M (the bar was 52.5%), and leads by 0.7 and 2.1 points on the references, inside the 4-5 points two runs
differ by. The default stays.

Arm `E8.1-A8-R132-S45@11600M+R139@12400M+R155@13200M+R164` (`--per-player-gae --gae-lambda 0.96`, the two lines
`launch_flags.py --diff` shows) and control `E8.1-A8-R132-S45@11600M+R139@12400M+R155` continued in place, both
from 13,200M to 14,000M in the two slots. SWAs of the last 80M, greedy, `tools/tournament.py`; a draw counts
half; per seat from the JSON (`data/reports/p50_*`, untracked). Runs and matches by the running session.

| | +400M (13,520-13,600M) | +800M (13,920-14,000M) |
|:---|:---|:---|
| arm against control, 2,000 deals a side | 50.7% ± 0.8 (as US 49.4, as USSR 52.0) | **51.5% ± 0.8** (as US 50.3, as USSR 52.7) |
| J1's best, 1,000 deals a side: arm / control | 61.7 / 58.8 | 61.0 / 60.3 |
| the old reference, 1,000 deals a side: arm / control | 72.5 / 69.0 | 71.5 / 69.4 |
| SD of a side's value over its own redealt hands: arm / control (base 0.075) | 0.080 / 0.073 | 0.078 / 0.073 |
| the same in action rounds 1 / 2 / 3: arm | 0.096 / 0.096 / 0.084 | 0.094 / 0.085 / 0.085 |
| the same in action rounds 1 / 2 / 3: control | 0.085 / 0.078 / 0.075 | 0.085 / 0.080 / 0.072 |

* Health level throughout (arm / control): entropy 1.05 / 1.04, explained variance 0.869 / 0.866, value loss
  0.0441 / 0.0448, self-play US win rate 0.485 / 0.476 at +700M. E3's failure of per-player GAE does not occur
  at matched reach.
* The arm ran 14% slower while sharing the GPU (40.5k against 46.9k steps a second).
* The arm's critic leans on the side's own hand a little more (0.078-0.080 against 0.073), most where the hand
  is full. That is the direction the plan expected and is not a measure of play.

**What would overturn it.** One seed, one branch point, a network already on its plateau: a difference of 1.5
points is 1.9 standard errors and could be either real and small or nothing. The site-game positions about the
hand (the plan's read 5) were not run.
