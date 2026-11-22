# P56 — the afterstate critic inside the advantage (owner, 2026-11-21)

**Approved by the owner, 2026-11-21: step 1 built; checks, then pairs of 800M branches from R169 at 2,000M** (the
section "The approved plan"). The first assessment follows.

**Assessment: sound, and worth a short screen of branches rather than one arm,** because it is two changes at
once -- a much shorter reach, and a split of the step at the afterstate -- and one arm cannot tell which of them
did what. Measurements: [`../log/P56_afterstate_advantage_probe.md`](../log/P56_afterstate_advantage_probe.md).

> **Correction (2026-11-21, on the owner's objection that a clearer signal from the critic is no risk).** The
> first version of this plan mixed two things and overstated the second.
>
> * **Reading the critic right after the side's own action, from its own view, is no risk.** The advantage is
>   already made of the critic's values at later nodes; this only replaces a node that cannot see the mover's
>   hand by one that can. It is P50's reasoning with a nearer node.
> * **"The policy is pulled toward the critic's favourite move" was wrong for this estimator.** The gradient
>   reads the afterstate only of the move that was played, and every such afterstate is a training row of the
>   critic. The match of the critic's-best-move player against the policy tests the critic on moves nobody
>   played; it is withdrawn as a gate. The locked setups of P44-P46 came from pool games, which condition 3
>   already keeps out.
> * **What remains a trade is only `lambda_opp` under 1:** how far a systematic error of the afterstate's value
>   is left uncorrected by what happens next. The probe found that error small where no die follows.
> * **Where a die follows, the owner's expectation fails and the fix is exact:** the critic before a roll is not
>   the average of the critic after it (it differs by 0.052 a position against 0.064 of luck, and carries about
>   a sixth of the move's effect; the log's section 4). Taking the rolls' average of the after-roll value as the
>   afterstate's value removes the luck with no bias, whatever the critic's quality.
>
> **The order is therefore changed:** (1) the own-view afterstate as a node with `lambda_opp` 1, and the rolls'
> average for moves followed by a die, in the advantage and as the before-roll row's target -- both justified
> by what they are, judged as P50 was (does it cost anything); (2) `lambda_opp` under 1 as the one arm that has
> to show a gain, on top of (1). The screen below stands for (2); "The risk" and the CPU match are superseded by
> this note.

## The proposal

In each side's own chain ([P50](P50_own_value_chain.md)), the step from a decision to the side's next decision is
split at the afterstate, the position right after the side's action:

```
d1 = V_after(s+, me) - V(s, me)                  my move's own effect
d2 = r + V(s_next_mine, me) - V_after(s+, me)    the dice and the opponent's turn
A  = d1 + lambda_opp * (d2 + lambda * A_next)
```

`lambda_opp` 1 is P50's chain; 0 is `V_after - V`, the critic's own opinion of the move and nothing else.

## Is it sound

Yes. For any `lambda_opp` and `lambda` in [0, 1] the advantage is a mixture of n-step returns that all bootstrap
from one side's own values, each value added once and subtracted once; at 1 and 1 it is the result minus the
value. It is the textbook TD(lambda) with a decay that depends on the node. Four conditions keep it so:

1. **The value targets keep the long chain.** Only the policy's advantage takes `lambda_opp`; `V` and `V_after`
   stay fitted to the lambda-return they have today. Fitted to a target that is half its own afterstate value,
   the critic would bootstrap from itself at a reach of one card play.
2. **The afterstate's value is the rollout-time network's,** as every value in the buffer is, and present for
   every learner decision that has an afterstate (`--afterstate-frac 1.0`, the after-decision moment aligned to
   its buffer row). A decision with no afterstate -- the action leads straight to the side's own next decision,
   70% of decisions -- takes the plain step with `lambda`.
3. **Rows against a fixed opponent keep `lambda_opp` 1** until P46's question is closed for them: the pool taught
   the critic to rate unfamiliar play as losing, and `lambda_opp` is the weight with which the critic's opinion
   of a move enters the policy's gradient directly.
4. **Tests:** at `lambda_opp` 1 the advantages equal per-player GAE's to rounding; at both decays 1 each side's
   target is its realised return; a row without an afterstate equals the plain step.

## What it is, measured

* **The noise it removes is real and large:** where there is an afterstate, the dice and the opponent's turn
  are 90% of the step's variance (0.104 against the move's own 0.042).
* **It is mostly a shorter reach.** `lambda_opp` 0.8 / 0.5 has the advantage spread of the plain chain at
  lambda 0.88 / 0.69: today the weight halves after about nine card plays, at 0.5 after one opponent's reply.
  Nothing shorter than 0.97 has ever been trained here (0.97 was +39 to +66 Elo on one seed in E4, 0.99 lost).
* **The split itself is worth 6-15% of spread** at equal return error, more the lower `lambda_opp` is. That is
  the part a flag cannot give.
* **The afterstate critic is fit to bootstrap from:** as calibrated as the critic at decisions, its own error a
  seventh of what a move changes.

## The risk

Today the critic's direct opinion of a move carries 2-4% of the advantage's weight; at `lambda_opp` 0.5 it
carries half. The policy is then pulled toward the move the critic likes best.

* The critic separates the moves the policy considers by only 0.013, and agrees with the policy's top move in
  52% of decisions. Whether the moves it prefers are better is not known.
* A critic that undervalues what it has seldom seen makes the policy play it less, which leaves the critic with
  less to learn from: the mechanism behind the locked setups of P44-P46, with far more weight on the critic than
  it had then. The zero-avoiding term keeps rare moves alive, which is the brake.
* In positions the critic reads as decided its values are compressed, so the policy would get almost no signal
  there (it gets little now).

## Before the build: one match on the CPU

The player that picks the critic's best move (`critic:<checkpoint>`) against the policy of the same network,
R169's latest SWA, greedy, 1,000 deals a side, `tools/tournament.py`. It says which way a small `lambda_opp`
pulls:

* 50% or over: a small `lambda_opp` is a step of policy improvement, and `lambda_opp` 0 is a serious arm;
* under 40%: `lambda_opp` 0 is dropped and 0.5 is expected to cost;
* between: as planned below.

## The build

`rollout_buffer` computes the policy's advantage on the per-player chain with a decay at the afterstate
(`--afterstate-adv-lambda`, default 1 = off); the value targets are untouched. It needs the after-decision
value of every learner decision beside its buffer row; the batch runner already records that moment. No engine
change, no observation change.

### The rolls' average: what it needs (the running session's reading of the code, 2026-11-21)

* **No engine change.** A `ROLL_DIE` step already takes forced dice (a replayer and test affordance): one die for
  a coup, the Space Race, a trap's escape and the wars; both dice for a realignment (36 pairs) and the Summit; the
  Olympic Games re-roll a tie from the random state, so their 30 non-tie pairs are enumerated and renormalised.
* **A bindings change** in the batch runner's afterstate capture: at each before-roll moment of a learner
  decision, copy the state, step each face, take the observation from the decider's side. About 3 microseconds a
  face; small against the step.
* **A trainer change:** the faces' values are read at rollout time with the rollout-time network (about 500
  value-only rows a step at 512 envs; their observations are too many to keep for the update), and one float a
  before-roll row is kept: the average. It is the afterstate node's value in the advantage and the before-roll
  row's training target; the roll's own step is left out of the advantage.
* **Cost:** a few percent of throughput expected, to be measured before any run.
* **Tests:** a forced face lands where a reseeded roll of that face lands; the average equals the reseeded mean
  within sampling error; `lambda_opp` 1 without a die equals per-player GAE.

The engine is untouched; a bindings change is the owner's to approve.

## The approved plan (owner, 2026-11-21): checks, then pairs of branches

Step 1 is built (`--afterstate-advantage`: each side's own chain with the afterstate node and the die's exact
average, `--afterstate-lambda-opp` default 1). The baseline is R169 at 2,000M. This section replaces the screen
below it.

### Checks, before any long run

1. **Throughput,** alone on an idle GPU, the baseline's flags with and without `--afterstate-advantage`.
2. **The new advantage on one rollout of the baseline** (CPU, the trainer's own code path on a recorded
   rollout):
   * the advantage's spread with the roll's luck dropped and with it kept, overall and by kind of move (coup,
     realignment, the Space Race, a war, moves with no die);
   * the advantage's mean by kind of move, dropped against kept: they must agree within their errors. This is
     the check on real data that leaving the luck out does not bias a kind of move;
   * the outcomes read per roll after merging (2.9 on random play).
3. **The noise between two runs of one recipe:** C' below. It is a run, listed here because every read is
   interpreted against it.

> **Checks 1 and 2 done (2026-11-21; the log's section 5): both pass.** T1's flags cost 4.5% of throughput
> (98.6k to 94.1k steps a second alone, on the later build `fc65976`; 11% as first measured); leaving the luck out biases no kind of move (every paired difference of
> the mean within 1.2 standard errors) and takes 4% off the advantage's spread, 15% on wars; 4.16 outcomes read a
> roll on trained play. S's policy lambda is **0.86** (A8's spread on the baseline), which needs a knob for the
> policy's chain apart from the value targets', not built. The baseline for read 3 is 0.029 mean absolute.

> **Pair 1 (2026-11-22; the log's section 6): T1 meets its rule.** 53.7% ± 0.8 against C at +800M (as US 50.3, as
> USSR 56.0), above C on both references (+1.7 and +2.6), health level. Whether that is more than one recipe's own
> noise waits for C'. Read 3's first line is withdrawn: the critic before a roll did not move (0.028 against
> C's 0.028) because the observation there does not show which roll is pending (identical for every roll move at
> 75% of decisions); the advantage never reads it, taking the outcomes' average of the values after the roll.

### The arms

All from R169's state at 2,000M, every other flag the baseline's (`launch_flags.py --diff` showing only the
lines named), 800M each, two at a time.

| pair | arm | what it is | compared with | rule |
|:---|:---|:---|:---|:---|
| 1 | **C** | the baseline continued under its own name | | |
| 1 | **T1** | `--afterstate-advantage`, `--afterstate-lambda-opp 1` | C | it must cost nothing |
| 2 | **C'** | the baseline continued with another sampling seed | C | the noise of one recipe, head to head and on the references |
| 2 | **A8** | T1 with `--afterstate-lambda-opp 0.8` | T1 | it must show a gain |
| 3, only if A8 gains | **A5** | T1 with `--afterstate-lambda-opp 0.5` | T1 | it must show a gain |
| 3, only if A8 gains | **S** | T1 with its lambda lowered to A8's reach (0.86 by check 2's sweep), no decay at the afterstate | A8 | is A8's gain the split or the shorter reach |

* **0.8 before 0.5** (owner): what a move is worth depends on whether the opponent can answer it, and the
  positions after the reply are where the critic's first view of the move is corrected. 0.5 runs only if 0.8
  helps.
* T1 runs the per-player chain at the lambda that matches the baseline's reach (0.96 for a baseline at 0.98).
* Rows against a fixed opponent, if the baseline has any, keep `lambda_opp` 1 (the build's default).

### Reads at +800M

1. Head to head on the same deals, SWAs of the last 80M, greedy, 2,000 deals a side, per seat from the JSON:
   T1 against C, C' against C, A8 against T1 (and against C).
2. The references (the old reference, J1's best), every arm, as context.
3. For T1 and A8, on the CPU: the critic before a roll against the outcomes' exact average (off by 0.029 mean
   absolute on the baseline at 2,010M; it is now that row's training target, so it should fall), and the
   critic's AUC and Brier against results beside C's.
4. Health throughout: entropy, clip fraction, the size of a policy update, explained variance, the self-play
   seat balance, the commonest opening setup's share in each seat.

### How each is judged

* **T1, by what it is:** not under C. The two-standard-error floor of its head to head at 49% or over, the
  references not under C's by more than C' differs from C, health level. A gain is welcome and not required.
* **A8 (and A5), a trade:** 52% or over against T1 on 4,000 games, and by more than C' differs from C; the
  references not under T1's; health level.
* **S** is read only against A8: level with it, the gain was the reach, and a flag gives it; under it, the split
  at the afterstate is what gained.

**Stop early** only on a clear problem inside 100M: entropy rising or falling steadily, explained variance
falling, one seat running away.

**Prediction.** T1 level to +2 against C; C' within 1.5 points of C; A8 level to +2 against T1. Held loosely.

**Then.** T1 not under C: the baseline takes it. A8 over the bar: the baseline takes 0.8 and pair 3 says whether
to go lower and why.

## The first version's screen (superseded by the approved plan above)

All from the baseline's state when the screen starts (R169 at 2,000M if the owner adopts it), every other flag
the baseline's (`launch_flags.py --diff`), 300M each, in pairs in the two slots.

| pair | arm | what it is | what it answers |
|:---|:---|:---|:---|
| 1 | **C** | the baseline continued | the control of every arm |
| 1 | **A5** | per-player chain, lambda 0.96, `lambda_opp` 0.5 | is there anything here |
| 2 | **A8** | the same, `lambda_opp` 0.8 | the cautious setting |
| 2 | **S** | per-player chain, lambda 0.69, no split (flags only) | A5's reach without the afterstate: is it the reach or the split |
| 3 | **A0** | `lambda_opp` 0 | the critic's opinion alone (dropped if the CPU match says under 40%) |
| 3 | **P** | per-player chain, lambda 0.96 | P50's form on this base: the chain without the decay |

Pair 1 decides whether pairs 2 and 3 are run. About 85 minutes a pair. The best arm and C then continue to
+1,000M, where it is judged.

**Reads.**

1. Each arm against C at +300M, SWAs, greedy, 2,000 deals a side, per seat from the JSON; the same at +1,000M
   for the arm that continues.
2. The references, as context.
3. Health from the training metrics: entropy, clip fraction, the size of a policy update, explained variance,
   the afterstate loss.
4. The lock-in watch: the share of opening setups on the commonest one in each seat, and (the log's probe, CPU)
   the share of decisions where the policy's top move is the critic's best -- 52% now; a climb toward 100% is the
   policy collapsing onto the critic.

**How it is judged, and why.** P50's change was right by what it is and needed only to cost nothing. This one
is a trade of noise against the critic's errors, so it has to show a gain: at +1,000M the arm at 52% or over
against C on 4,000 games (2.5 standard errors), the references not under C's, and read 4 quiet. The +300M reads
only order the arms.

**Stop early** on entropy falling steadily, read 4's share passing 75%, or one seat running away.

**Prediction.** A8 level to +2; A5 either way; A0 a loss; S close to A5. I hold none of these firmly: no network
here has been trained at a reach this short.

**Not in this plan.** The dice's step as an exact average over the roll instead of a sample; the advantage
measured against the policy's average over every legal move's afterstate (18 more value reads a decision), which
would remove the 0.029 by which the afterstates' mean differs from the decision's value. Both only if the screen
shows a gain.

**What would overturn the reasoning.** The probe is one rollout of a network at 1,200M, where the critic is
still improving fast; shares of variance and the critic's agreement with the policy may differ on a plateau
network. Return error is not quality, so "6-15% better" is a statement about spread at equal reach, not about
the gradient.
