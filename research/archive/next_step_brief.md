# Briefing: where the Twilight Struggle agent is stuck, and what to do next

Self-contained brief for an outside reviewer. No prior context assumed. Everything below is
measured in this repository unless marked as an assumption.

---

## 1. The project

Build a competent AI for **Twilight Struggle** (Deluxe, 110 cards) — a two-player, zero-sum,
imperfect-information card/board game. Target strength: *a human who has played a few games,
not tens*. Not superhuman.

**Hardware budget: a single RTX 4090.** This is the binding constraint on every design
choice below. A typical experiment is ~3 hours per arm.

**Stack**
- **Engine**: C++20, zero-allocation. `GameState` is trivially copyable and ≤ 4 KB, so
  cloning a position is cheap. All randomness goes through a seeded SplitMix64 PRNG, so
  replays are bit-reproducible.
- **Action space**: a single flat 212-dim space. Turns decompose into a stream of 4-byte
  micro-actions driven by a state machine.
- **Chance nodes**: die rolls appear as explicit `ROLL_DIE` nodes where the "decision player"
  is NONE. These are resolved by the engine, not chosen by a policy. (Two separate bugs in
  this codebase came from code that failed to notice this and let a network pick its own
  dice.)
- **Training**: NashPG — PPO-style clipped objective plus a KL penalty against a periodically
  refreshed frozen reference policy `π_ref`, intended to converge toward Nash without cycling.
  GAE, **γ = 1** (the game is zero-sum and decided at the end). Rewards are terminal and
  zero-sum, perspective-aligned, with the players alternating.
- **Scale**: 512 parallel C++-driven envs, ~7,300 env-steps/s during training (~15,400 solo).
  A 3-hour run is ~78M env steps.
- **Model**: experiments below use "v2" (GNN + card + global ResNet with masked action
  heads, ~12M params). Larger v3/v4 variants exist but are not well trained.

**Standing constraints from the project owner**
1. The agent should **learn** these decisions. Encoding them as explicit rules in the bot
   (e.g. "if a winning action exists, take it") defeats the purpose. Even reward shaping is
   considered a stretch.
2. The engine is treated as trustworthy; changes to it require justification up front.
3. Training and replay generation must stay reproducible from a seed.

---

## 2. Where the agent is

Current best (v2, 78M steps, ~3h): beats a hand-written heuristic bot **80.2%**, Elo 1760 vs
heuristic 1500 vs random 859. So it is clearly learning. But:

**It misses forced wins.** Measured over 512 games / ~198k decisions, greedy play:

| | |
|:---|---:|
| instant-win opportunities | 348 (0.175% of decisions, 0.68 per game, in 56% of games) |
| implied over a 78M-step run | **~137,000** |
| **taken** | **80.5%** |
| avoidable losses (11x more frequent, 8.4/game) | avoided 94.4% |

An "instant win" here means a legal action that immediately ends the game in the mover's
favour. It is in the mask, one ply from terminal.

**It is not rarity, and not sampling noise.** 137k opportunities per run, spread evenly over
turns 2–10, is not a data-starved regime. Greedy take rate (80.5%) barely differs from
sampling at temperature 0.1 (79.3%), so the failures survive argmax.

**It is low advantage, driven by critic optimism.** The policy is not blind in general — it
puts a **median 0.980** probability mass on the winning action (mean 0.759; above 0.5 in 77%
of cases; below 0.01 in only 4.6%), against a uniform baseline of 0.285 over ~6.7 legal
actions. The misses sort by what the critic already believes:

| critic `v_win` at the opportunity (+1 = certain win) | mean |
|:---|---:|
| win **was taken** (n=280) | +0.419 |
| win was **missed** (n=68) | **+0.626** |

It declines to end the game precisely when it already thinks it is winning comfortably. At
v_win = +0.63 the certain win is worth only ~0.37 more than playing on — and the critic is
overconfident there, because "probably winning" is not "won".

**More training does not fix it.** Take rate across the run's own 13 snapshots: 0.569 → ~0.75
within the first 30M steps, then no trend over the remaining 45M (0.717 at the end).
Loss-avoidance is flat at ~0.94 from initialization onward.

**It never contests most of the map.** Battlegrounds empty in late positions (turn ≥ 6),
512 games: Algeria 97.7%, Saudi Arabia 95.5%, Libya 90.7%, Nigeria 58.0%, Zaire 53.8%,
**West Germany 48.6%**, Mexico 47.2%, **France 42.5%**, India 39.9%. The count of empty
battlegrounds *plateaus* at ~7.5 from turn 8 through 10 — it stops expanding once its early
cards are spent. Two top-value Europe battlegrounds sit empty in nearly half of late
positions.

**Games are short and lopsided.** Mean final turn 6.50; only 37.9% reach turn 8, 11.7% reach
turn 10. Of five sampled self-play games, four were 20-VP runaways ending turns 4–6 and one
was a turn-3 self-inflicted DEFCON-1 loss.

---

## 3. What has been tried

**Mid-game start sampling — tested, negative.** Hypothesis: self-play rarely reaches late
turns, so resume a share of environments from saved turn-boundary positions (mix 50/15/15/10/10
over turns 1/4/6/8/10), keeping only positions that are not already decided. Result at matched
78M steps: the **control beat it 67.3%** head-to-head over 2,000 games, and beat the heuristic
80.2% vs 70.5%. A targeted follow-up confirmed the mechanism works but the allocation is bad —
resuming from 1,000 turn-8 positions, the pool arm wins **53.4% ± 2.2**, i.e. +3.4 points in a
regime that occurs in 5.6% of episodes, bought with −17.3 points from the opening.

**Blunder-window credit confinement — retained.** Confining an unprovoked blunder's penalty to
the turn it occurred in cut self-inflicted DEFCON-1 losses from 46 to 6 across four snapshots.
Win rate was a wash. (Measured before the fixes below; direction is trusted, absolute counts
are not.)

**A cautionary note on measurement.** Four diagnostic/evaluation bugs were found and fixed in
this codebase, three of them the same defect in three files: a batched diagnostic that stopped
after N completed episodes with N far below the env count, making the sample "the fastest N
games". It reported mean final turn 3.30 against 6.67 true, and 0% of games reaching turn 9
against 27%. A fourth let a policy choose its own dice during evaluation. An earlier version of
the start-pool A/B was also confounded: evaluation cost consumed 37% and 61% of the two arms'
wall clock, giving them 67M vs 31M steps, and produced the opposite (wrong) conclusion.
**Any number quoted from before those fixes is suspect.** The numbers in §2 are post-fix.

---

## 4. The candidate directions

**A. One-ply terminal-exact value correction (currently favoured).** During rollout, cheaply
determine whether any legal action immediately ends the game and with what utility, and use it
to set an *exact* value target (+1 / −1) at those states — correcting the training signal only,
never overriding action selection at play time, so the network still learns the decision.

- Measured cost: a one-ply scan over every legal action at every decision is **1.4x the
  rollout loop**, and rollout is ~8% of training wall time, so **~3–4% throughput**. Fits in a
  3-hour A/B.
- Depth 1 is information-safe: whether the game ends after *our own* move is determined by
  public state (VP ≥ 20, DEFCON 1). Depth ≥ 2 needs the opponent's hidden hand, hence
  determinization or belief sampling.
- Open subtlety: an action may land on an unresolved `ROLL_DIE` node, so "wins immediately"
  can be conditional on a die.

**B. Opponent league (AlphaStar-style PFSP).** Never tested. Self-play against only the
current policy lets both sides tacitly agree to ignore Algeria forever — nothing punishes it.
A league of past snapshots and exploiters would.

**C. Full MCTS.** Ruled out on cost: at 32 simulations per decision, NN evaluations rise ~32x,
7,300 steps/s becomes ~230, and a 78M-step run takes ~94 hours instead of 3.

---

## 5. Questions

1. **Is the diagnosis right?** Does "critic optimism ⇒ low advantage on the terminal action"
   adequately explain an 80.5% forced-win rate that plateaus after 30M steps, or is there a
   more likely cause we have not tested? What measurement would discriminate?

2. **Is γ = 1 with purely terminal, zero-sum rewards the underlying problem?** With no
   discounting, ending the game now and ending it three turns later are worth the same to the
   critic, and only risk distinguishes them. Is a small discount, or an explicit
   terminal-distance signal, a cleaner fix than one-ply lookahead — and what would it cost in
   endgame bias?

3. **Is direction A the right first move**, or does B (league) dominate it given that the
   positional failure (unclaimed battlegrounds) is the larger strength gap and A explicitly
   does not address it?

4. **Cheaper alternatives to A for critic calibration**: would n-step returns, a different
   value-loss form, or removing the value-clipping term address the overconfidence at less
   complexity than a lookahead pass?

5. **Chance nodes in a one-ply scan.** What is the correct treatment — restrict to actions that
   terminate without passing a chance node, enumerate die outcomes, or something else — such
   that the value target stays unbiased?

6. **Does A conflict with NashPG's KL-to-`π_ref` regularization?** The reference policy is
   refreshed from an earlier snapshot; if corrected value targets move the policy sharply on a
   narrow class of states, does the KL term fight it, and should `π_ref` refresh cadence or η
   change alongside?

7. **The empty-battleground failure.** Given the constraint against reward shaping and against
   hand-coded heuristics, what mechanism would most plausibly teach an agent to contest
   territory it has learned to ignore? Is a league sufficient, or is directed exploration
   needed?

8. **Do v2 conclusions transfer?** All results above are on a ~12M-parameter v2. Is it
   defensible to select a training-signal change on v2 and then scale, or is there a specific
   risk that the plateau is capacity-bound and the ranking of options inverts at larger scale?

9. **Experiment sequencing.** Given ~3 hours per arm on one 4090 and a step-budgeted A/B
   harness that produces matched arms, what is the highest-information order in which to test
   these? What would you *not* spend a run on?

10. **Anything missing.** Is there a standard technique for this failure mode — an agent that
    knows the winning move (median 0.98 probability) but declines it when it feels safe — that
    is not on the list above?
