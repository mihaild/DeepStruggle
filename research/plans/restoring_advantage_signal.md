# Restoring the advantage signal

The measured problem (see [`../log/europe_control_and_held_scoring.md`](../log/europe_control_and_held_scoring.md)):
once the USSR finds the Europe-control strategy at ~110M, outcomes become predictable, the critic
degenerates to the base rate (beating "always predict USSR" by 0.2pp at 160M), and `adv_std_raw`
falls 6x. Both sides stop learning; only one of them needed to.

## Why komi does not port

Komi works in Go because the outcome **is** the score: shifting komi shifts the win threshold
continuously and keeps P(win) near 0.5. Twilight Struggle has several terminal conditions and only
some are VP-mediated. Europe control writes `state.victory_points = 20` and ends the game
(`engine/src/scoring.cpp:170-181`), so it bypasses any VP handicap entirely -- a US given +6 VP at
the start still loses the same games the same way. And a handicap large enough to matter changes
what the US is playing for: it starts racing for an early 20 instead of learning to contest Europe.

Two consequences for the design:

* **The handicap has to act on the board, not the scoreboard.**
* **A raw VP-delta shaping term is actively dangerous here.** `ShapedZeroSumReward` (vp_scale 0.02)
  would hand a one-step `+20 * 0.02 = +0.4` spike to the Europe-control move -- reinforcing the
  exact strategy in question. Any VP term needs centering and saturation, not linearity.

## Two families

**A. Stop the objective saturating.** KataGo's answer to a decided game is the *score* utility, not
komi: keep optimising something that still varies when the win rate does not, with the utility
re-centred each search on the currently expected score and saturating far from it, so the incentive
sits on realistic marginal gains.

**B. Stop the state distribution being dominated by decided positions.** KataGo caps visits once
the losing side is under 5% for five turns, and plays handicap games. The analogue is to spend
rollout budget on positions that are still undecided.

## Options, ranked by cost

### 1. Turn on the start-state pool (family B) -- already built, currently off

`ai/training/start_pool.py` exists, is tested, and `--start-pool-frac` defaults to **0.0**.
E3-17-22 did not use it. Its own docstring states the mechanism this whole investigation arrived
at independently: *"A decided position contributes no gradient at all: with terminal-only reward
every action there returns the same value."* `is_salvageable` filters on VP balance **and** region
balance, so it selects genuinely undecided positions, and positions are captured pre-deal with
`rng_state` reseeded so one position does not become N copies of one future.

Cost: one flag. Instrument: `adv_std_raw` should stop collapsing.

**Known weakness:** the pool is tagged by the snapshot that produced it, so once the policy is
lopsided the captured positions come from lopsided games. `is_salvageable` removes the worst, but
seeding the pool from *older* checkpoints or from human-corpus positions would be strictly better
and is a small change.

### 2. Put VP into the advantage, centred and saturating (family A)

`values_vp` and `returns_vp` already exist but feed **only** the value loss
(`nash_pg._value_loss`); the VP head never reaches the policy. The advantage is computed from
`rewards` and `values_win` alone (`rollout_buffer.compute_gae`).

The change is a second GAE on the VP stream blended into the advantage, with the VP term centred
on the critic's own predicted VP -- the direct translation of KataGo's `x_0 = mean predicted score
at the root` -- and passed through a saturating transform. Centering is what answers the "it will
just chase VP" objection: the reward is for improving on what is already expected, and saturation
caps what a single large swing is worth.

### 3. Auxiliary per-country control head (family A, representation side)

KataGo's ownership target: predict who controls each country at game end, used only for its
auxiliary loss. This does not restore the advantage by itself, but it keeps the trunk learning
board structure while `v_win` is flat -- and board sensitivity is measured to be the weak half
(hand/board perturbation ratio 1.41 at 160M). An auxiliary loss is not an observation change, so
it costs no checkpoint compatibility.

### 4. Downweight decided positions in the batch (family B)

The analogue of KataGo's visit capping: truncate or downweight a trajectory once the critic has
been confident for N consecutive steps. Complements 1 rather than replacing it. Note that
`priority_indices(alpha)` already samples by `|advantage|`, which cannot help here because the
advantages are what collapsed.

### 5. Opening randomisation (family B, weak alone)

The `--opening` machinery from `tools/lib/openings.py` can diversify the start distribution. On its
own it is the weakest option -- the forced human opening was measured not to change the win rate at
all (14.4% both arms) -- but it composes with the others and is nearly free.

## Suggested order

Start from the **80M resume checkpoint**, the last point at which the critic is still worth
something (79.2% outcome accuracy against a 53.9% base rate, correlation +0.654).

1. Option 1 alone, ~20M steps, watching `adv_std_raw` and the Europe-control ending rate. It is one
   flag, so it costs almost nothing to rule in or out.
2. If the signal still collapses, add option 2.
3. Option 3 is worth doing regardless for the board-sensitivity problem, but it is a bigger change
   and should not be confounded with the first experiment.


---

## Revision after checking the history and the prior start-pool result

### Is the runaway new? No -- but it is worse now, and it goes both ways

Side balance read straight from every run's TensorBoard (`endgame_win_rate_us/ussr`, first 40%
of steps against last 20%), which covers runs whose checkpoints are on retired observation
layouts and can no longer be loaded. 29 runs had a readable split; **6 swung by 15pp or more**:

| run | steps (M) | US early | US late | swing |
|:---|---:|---:|---:|---:|
| p1_categorical_nofilter_VOID_unscaled_target | 80 | 44.7% | 3.6% | **-41.0** |
| E3-17-22 | 160 | 46.1% | 13.2% | **-32.9** |
| E3-15-22 | 129 | 48.2% | 26.5% | **-21.7** |
| E3-13-22 | 80 | 54.0% | 37.0% | -17.1 |
| **E3-14-22** | 25 | 41.7% | 58.9% | **+17.2** |
| **E3-14-21** | 80 | 29.1% | 63.4% | **+34.3** |

Most other runs drift 3-14pp in the US's disfavour (`p1_scalar_nofilter` at 240M: -13.7). So mild
drift is old and common; the severe form is not new either, but the two worst non-void cases are
the two most recent long E3 arms.

**The two positive rows are the important ones.** In E3-14-21 and E3-14-22 it is the *US* that
runs away, from 29.1% to 63.4%. That rules out any account in which the game, the engine or the
observation favours the USSR: the dynamic is "whichever side finds something the other has not
answered", and which side that is varies by run.

### The start-state pool: demoted

`research/log/early_training_signal.md` §3.2-3.3 already settles the first-order question. The
clean A/B makes the pool arm **worse**: Elo 1645.1 against the control's 1760.2. §3.3 then shows
the mechanism itself works -- replayed on 1,000 turn-8 positions the pool arm wins 53.4% +/- 2.2%
-- and that the *allocation* is what fails: +3.4 points in a regime that occurs in 5.6% of
episodes, paid for with -17.3 points from the opening.

There is a second problem specific to using it for *this*: `is_salvageable` filters on **board
balance** (|VP| <= 10, |region net| <= 20), not on **outcome uncertainty**. Once one side has a
strategy that wins from most boards, a position can be perfectly balanced on those two measures
and still have a 90/10 outcome. The filter is on the wrong quantity for restoring advantage
variance, and a filter on the right quantity means playing candidate positions out repeatedly and
keeping the high-entropy ones, which is not cheap.

And the honest answer to "how does it help the asymmetry": **it does not, directly.** It changes
which positions are sampled, not the quality of the signal within a position. The US's problem is
that in the positions it actually reaches, no US action is distinguishable from any other.

### The filtering family cannot fix this either

`--adv-filter-quantile` already exists (P1) and drops low-|advantage| samples from the policy
update. It is the same family as option 4 above, and neither can work here, for one reason:
**filtering selects relative magnitude within a batch; it cannot create absolute signal.** With
`adv_std_raw` at 0.049 the surviving top quantile is mostly noise, so filtering concentrates the
update on the largest noise. Option 4's own variant is undermined the same way -- the natural
signal for "this position is decided" is |v_win|, and at 160M v_win sits near -0.8 almost
everywhere, so it would flag everything.

Option 4 is therefore withdrawn.

### Revised order

1. **VP margin in the advantage, centred and saturating.** The only option that creates gradient
   where the terminal signal gives none: in a game already lost, losing by 8 rather than 20 is
   distinguishable, so US actions become comparable to each other again. Centring on the critic's
   own predicted VP is what stops it becoming "chase VP", and saturation caps what one large swing
   is worth -- which also defuses the Europe-control `+20` spike that makes plain `vp_scale`
   shaping dangerous.
2. **Opponent diversity against older snapshots.** Distinct from opponent *prioritisation*, which
   is pointless in pure self-play -- the US already faces the strongest USSR there is. Diversity is
   a different claim: against an 80M opponent the trailing side has games it can win, so terminal
   signal and advantage variance return. The standing risk is that it learns to beat weak opponents
   rather than the current one.
3. **Auxiliary per-country control head.** Keeps the trunk learning board structure while `v_win`
   is flat. Does not itself restore policy gradient, so it is a complement, not a fix.
4. ~~Downweighting decided positions~~ -- withdrawn, see above.
5. ~~Start-state pool~~ -- demoted; measured harmful as configured, and filters on the wrong
   quantity for this purpose.


---

## Second revision: the VP-margin proposal is mostly dead

The objection was that making the win/loss signal depend on VP will not help against Europe
control, because that ending is an instant win regardless of the score. Measured, it is worse than
that. Terminal VP over 512 games from E3-17-22 @160M:

| ending | share | terminal \|VP\| = 20 | VP sd |
|:---|---:|---:|---:|
| 20vp | 40.4% | 100.0% | 6.14 |
| defcon1_self | 23.2% | 100.0% | 18.23 |
| defcon1_provoked | 16.4% | 100.0% | 17.04 |
| europe_control | 11.5% | 100.0% | 0.00 |
| held_scoring | 3.9% | 100.0% | 17.32 |
| final_scoring | 3.9% | **0.0%** | 8.35 |
| wargames | 0.6% | 0.0% | 2.87 |

**95.5% of games end at exactly |VP| = 20.** The engine normalises every abrupt ending to the cap
— 20 VP by definition, DEFCON-1, held scoring and Europe control by fiat — so the terminal margin
is a *constant* across all but the 4.5% that reach final scoring. It carries no information beyond
its sign, which is what win/loss already is.

So the version of the proposal that blends terminal VP margin into the advantage is dead, and not
only for Europe control: it dies on DEFCON-1 and held-scoring endings too.

**What survives, and how much.** Only the *per-step* form: shaping on VP deltas along the way
(`ShapedZeroSumReward`, or potential-based shaping with `phi = VP`), which reads the VP trajectory
before termination rather than the terminal margin. That does vary and does produce gradient in
every game. Two limits keep it from being the answer:

* It says nothing about the thing that actually decides these games. Contesting West Germany earns
  no VP until a scoring card fires, so the shaping does not differentiate the action that matters.
* Europe control writes `victory_points = 20` in a single step, so a naive potential term hands a
  large spike to precisely the strategy in question. The usual `phi(terminal) = 0` convention
  avoids that, at the cost of removing terminal shaping altogether.

It is worth having as a cheap secondary, not as the fix.

## The actual next proposal: historical opponent sampling

The root condition is that **the outcome is forced regardless of what the trailing side does**.
Nothing that re-weights, filters or rescales an existing signal escapes that, because there is no
signal to rescale. The outcome has to genuinely depend on the trailing side's actions again, and
the cheapest way to arrange that is to let it sometimes play someone it can beat.

**Change.** A fraction of rollout environments (~25-30%, the figure Tablut used) play the current
policy against a snapshot sampled from the run's own history, with sides alternated so both roles
get the varied opponent. Only the current policy's transitions enter the buffer; the opponent's
are not trained on.

**Why this and not opponent *prioritisation*.** Prioritisation — sampling opponents you lose to —
is pointless in pure self-play, because the current policy already faces the strongest opponent
there is. Diversity is a different claim: against an 80M snapshot the trailing side has games it
can actually win, so the terminal signal separates its actions again and `adv_std_raw` has
something to be non-zero about.

**Cost.** Moderate. The rollout loop currently runs one network for both sides; this needs a
second forward pass and a per-environment assignment of which network acts. No engine change, no
observation change, no checkpoint invalidation.

**Risk.** The policy may learn to beat old snapshots rather than the current one. Mitigated by
keeping the majority of games pure self-play, and visible in the head-to-head tournament.

**How it is judged.** `critic_auc` and `critic_brier_skill` staying up, and `adv_std_raw` not
collapsing — all three now logged live. The success criterion is *not* "the US recovers": the
runaway is bidirectional (E3-14-21 ran away in the US's favour), so the target is that neither
side's advantage signal dies.
