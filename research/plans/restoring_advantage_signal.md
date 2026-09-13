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
