# P56 — the afterstate critic inside the advantage (owner, 2026-11-21)

**Assessment: sound, and worth a short screen of branches rather than one arm,** because it is two changes at
once -- a much shorter reach, and a split of the step at the afterstate -- and one arm cannot tell which of them
did what. Measurements: [`../log/P56_afterstate_advantage_probe.md`](../log/P56_afterstate_advantage_probe.md).

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

## The runs: a screen, then one long pair

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
