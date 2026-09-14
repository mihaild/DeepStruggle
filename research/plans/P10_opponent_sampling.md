# P10 — Breaking the advantage collapse

**Status:** proposed, **awaiting approval**. Nothing here has been run.
**Needs approval:** experiments 2 and 4 change the rollout loop (trainer only — no engine, no
observation, no checkpoint invalidation). Experiments 1 and 3 need no code at all.
**Instruments:** `critic/auc`, `critic/brier_skill`, `adv_std_raw` — all now logged live.

**Throughput: ~45M steps/hour**, so 80M steps is under two hours. Measured from E3-17-22's
own snapshot timestamps: 5M at 16:55, 80M at 18:34, 160M at 20:20 — 3.5h for the whole run.
`metadata.json` records `duration_seconds: 86400`, which is the *configured budget*, not
elapsed time; reading it as elapsed overstated every cost here by about 7x.

## What is being tested

Once one side finds a strategy the other has not answered, outcomes become predictable, the critic
degenerates toward the base rate, advantages vanish and both policies freeze. Measured on
E3-17-22 and reproduced independently on E3-15-22 with a different architecture and a different
winning strategy. Full evidence in
[`../log/europe_control_and_held_scoring.md`](../log/europe_control_and_held_scoring.md).

**The success criterion is not "the US recovers."** The runaway is bidirectional — E3-14-21 ran
away 29.1% -> 63.4% in the *US's* favour. The target is that neither side's advantage signal dies.

One observation shapes the priors below: the network already knows how to contest European
battlegrounds — as the USSR it puts 0.15 probability mass on a contested West Germany, against the
US's 0.04 in the identical cost regime. The *capability* is in the weights; what differs is the
perspective-conditioned policy. A response may therefore be close at hand.

---

## Experiment 1 — continue 160M -> 240M. Does it fix itself?

**Why first.** It is the control, it needs no code, and if the answer is yes then everything else
here is unnecessary. The US still wins ~10% of games, so the signal is not identically zero.

```
--resume <E3-17-22>/resume_160038912steps.pt   (+80M steps, ~1.8h)
```

**Read:** `critic/auc` (0.73 at 160M), `critic/brier_skill` (0.16), `adv_std_raw` (0.03-0.05),
win rate by side, Europe-control ending rate.

**Decision.** AUC back above ~0.85 and `adv_std_raw` back to ~0.2 without intervention → the trap
is transient and nothing needs building. Flat → it is absorbing, and 1 is the null result that
justifies 2 and 4.

---

## Experiment 2 — freeze the 160M USSR, train the US against it

**The decisive experiment, and the cheapest way to ask the real question: does a response exist?**
This is AlphaStar's *main exploiter* in miniature, and a special case of experiment 4 — pool of
one snapshot, fraction 1.0, learner side-locked to US — so building 4's mechanism gives this for
free, and this is the thing worth running first with it.

**Setup.** Opponent: `E3-17-22/snapshot_final.pt` (160M), frozen, always USSR. Learner:
initialised from **`resume_80019456steps.pt` (80M)**, always US, only its own transitions
receiving policy gradient.

Starting the learner at 80M rather than at the opponent's own 160M matters. By 160M the US side of
the policy has already degenerated -- probability mass on a contested France fell from 0.177 to
0.096, and the critic from 0.865 AUC to 0.728 -- so initialising there would hand the experiment a
learner that has already unlearned the thing being asked for. 80M is the last checkpoint where
both the US policy and the critic are still healthy.

**Budget.** 80M steps (~1.8h). Only about half the transitions carry policy gradient, so this is
roughly 40M policy-updated steps; the extra length buys back that halving rather than paying for
a longer experiment.

**Read:** US win rate against the frozen opponent over training; `adv_std_raw` and `critic/auc` on
the learner.

**Decision.** Win rate climbs clearly above its ~10% start → a response exists and self-play simply
failed to supply the gradient to find it. That validates the whole diagnosis and makes 4 worth
building properly. Plateaus near 10-20% → the strategy is close to unbeatable at this skill level,
and the question moves to the game itself or to much longer training. Either answer is worth the
6 hours.

**Caveat.** A best response to *one* frozen opponent is not a Nash improvement; it may be a
narrow counter that loses to everything else. That is expected and is exactly why 4 uses a pool
rather than a single opponent. Check the counter against the wider field in the tournament before
concluding anything general.

---

## Experiment 3 — resume from 80M with a different seed. Does it get stuck again?

**Setup.** `--resume <E3-17-22>/resume_80019456steps.pt --seed <new>`, +80M steps (~1.8h). No code.

**Read:** whether the collapse recurs, and if so whether it takes the same form (USSR via Europe
control) or a different one.

**Decision.** Collapses the same way → the 80M state is already inside the basin and any resume
experiment should start earlier. Collapses differently, or toward the US → the direction is a
symmetry-breaking accident, which matches E3-15-22 (no Europe control) and E3-14-21 (US runs
away), and means the mechanism is general while the specific exploit is not.

---

## Experiment 4 — historical opponent sampling

### The forward-pass cost, corrected

Not a doubling. Today one network serves both sides and the whole 512-env batch goes through a
single call. With a second network the batch is partitioned by *which network is acting*, giving
two calls on complementary subsets — the same total FLOPs, plus one extra kernel launch and a
second set of weights resident in memory.

### Design decisions

**Fraction.** 25-30% of environments play the current policy against a pooled snapshot; the rest
stay pure self-play. (Tablut used 25% past-iteration play.)

**Which transitions train — the one non-obvious choice.** Store **all** transitions in the buffer
and mask the *policy* loss to the learner's. Dropping the opponent's transitions instead would
leave the learner's non-consecutive, and GAE is a backward recursion over consecutive steps: the
existing `sign = curr_p * next_p` machinery already handles alternating perspectives correctly,
and from the learner's point of view the opponent's move is part of the environment. The value
head can keep training on every state, since a state's value does not depend on who chose the
action that reached it.

**KL-to-`π_ref`: unmasked**, decided. It is computed on every state, including those the
opponent's move led to, which gives the regulariser broader state coverage than the policy loss
gets. Revisit only if the KL term behaves oddly in the mixed environments -- worth watching
`internal/kl_div` for a step change when mixing switches on, since a KL measured over a wider
state distribution is not on the same scale as one measured over the learner's own.

**Pool composition.** Uniform over a capped set (8-12) of snapshots spanning the run, refreshed as
new ones appear. Uniform maximises outcome variance, which is the quantity being restored.

**Side assignment.** Alternate per episode so the learner plays *both* roles against the pool —
necessary because the runaway is bidirectional.

**What `π_ref` is not.** NashPG's frozen reference is a snapshot of the active policy used for KL
regularisation. The opponent pool is a different object with a different purpose; they should not
be conflated or shared.

**Cost.** In mixed environments roughly half the transitions carry policy gradient, so at 25-30%
mixing the effective data loss is ~12-15%.

**Risk.** The learner may get good at beating old snapshots rather than the current one. Mitigated
by keeping the majority of games pure self-play, and detectable in the head-to-head tournament.

### New instrument

Split `adv_std_raw` by environment type — self-play against mixed. If the diagnosis is right, the
mixed environments should carry visibly more advantage variance, and that split says directly
whether the mechanism is doing what it is supposed to.

---

## Suggested order and cost

| | needs code | steps | cost | why here |
|:---|:---|---:|---:|:---|
| 1. continue 160M -> 240M | no | 80M | ~1.8h | control; if it self-corrects, stop |
| 3. resume 80M, new seed | no | 80M | ~1.8h | is the basin reached deterministically |
| build the opponent mechanism | yes | — | — | gives 2 and 4 |
| 2. frozen 160M USSR, learner from 80M | (above) | 80M | ~1.8h | decisive: does a response exist |
| 4. full pooled sampling | (above) | 160M | ~3.5h | the actual fix, if 2 says yes |

Everything is cheap at ~45M steps/hour: 1 and 3 together are under four hours and need no code at
all. The GPU is at 98% utilisation with ~14 GB of 24 GB free, so two runs fit in memory but
contend for compute; sequential is still the honest schedule while E3-15-22 finishes.

Build the mechanism while 1 and 3 run, then 2, then 4 only if 2 says a response exists. Because
the runs are this short, 2 is worth extending rather than cutting if its trend is ambiguous.
