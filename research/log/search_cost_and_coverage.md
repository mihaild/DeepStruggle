# Search: what it costs, and how much of it is the coverage

What a "train with search" arm would actually cost, and the measurement that has to happen before
one is launched. Written while planning an arm from the 160M checkpoint of `E3-20-28`.

Everything here was measured on the current engine
(`2fb70052dee0164871632209060cbaa23fd4f0fb99cc2e956241e231b0e8e691`) with
`tools/scripts/check_engine_fresh.sh` clean.

---

## 1. What search was actually measured at, recovered from the session transcript

These numbers existed but were never written to this log, so they had already decayed into a
half-remembered "+27pp". Recorded here in full, with the configuration each belongs to, because
the attribution was wrong in the first version of this entry and the error changed the conclusion.

**Against the raw policy of the same net, `E3-20-28` @320M.** Every row carries the commit the
code was at when it was measured, because three of them predate fixes that could have changed them.

| searcher | budget | games | win rate | deployable? | measured at |
|---|---:|---:|---:|---|---|
| **PIMCTS** (reads the true `GameState`) | 48 sims | 12 | **100%** (12/12) | no — sees the opponent's hand | `3ef3d9b`, 09-15 15:11 |
| DMCTS (honest) | 48 sims | 12 | 58.3% (7/12) | yes — not distinguishable from chance at n=12 | `3ef3d9b`, 09-15 15:11 |
| DMCTS, 1 world | 96 sims | 40 | **72.5%** | yes | `5985175`, 09-15 ~17:42 |
| DMCTS, 4 worlds × 24 | 96 total | 40 | 65.0% | yes | `5985175`, 09-15 ~17:42 |
| DMCTS, 16 worlds × 6 | 96 total | 40 | 55.0% | yes | `5985175`, 09-15 ~17:42 |
| DMCTS, 1 world | 384 sims | 30 | **76.7%** (23/30) | yes | pre-`5985175` tree, 09-15 16:56 |
| DMCTS, 4 worlds × 96 | 384 total | 30 | 73.3% (22/30) | yes | `5985175`, 09-15 17:42 |

The 48-sim pair is recorded in `3ef3d9b`'s own commit message, so that one is certain. The
384-simulation row was taken from a working tree while batched MCTS was being written, before
`5985175` committed it.

> **Every row above predates three fixes and should be re-taken before being relied on**
> (invariant 13). In commit order: `8533a68` "search the state the caller holds, and refuse to
> return an illegal action"; `b6874af` "reject a micro-action whose decision_type is not the one
> being asked"; `9f78026` "a chance node's only legal action rolled 255 instead of a die".
>
> The searcher's *own* chance handling was never affected — `settle()` uses `auto_advance_step` or
> an explicit `MicroAction(ROLL_DIE, 0, 0, 0)`, both of which roll properly. The exposure is the
> harness that played the probe games: any loop that read the legal mask and called `step_flat` at
> a chance node forced a die of 255 before `9f78026`, which makes space race attempts and coups
> succeed automatically for *both* sides. And before `8533a68` the searcher could return an action
> illegal in the caller's state, which a loop ignoring the return value re-offers forever. Neither
> is proven to have fired in these particular probes, and neither is ruled out.

So the privileged searcher was **100%**, and the honest one reached **76.7%** given enough
simulations. The "+27pp" that was carried forward is 76.7% − 50%, i.e. the **honest, deployable**
DMCTS at 384 simulations — *not* a privileged number, and not one obtained at 48 or 64 sims.

**Depth beats breadth, and this was already measured.** At a fixed total simulation budget, splitting
it across more sampled worlds makes the searcher worse:

| total sims | worlds | sims/world | win rate |
|---:|---:|---:|---:|
| 384 | 1 | 384 | **76.7%** |
| 384 | 4 | 96 | 73.3% |
| 96 | 1 | 96 | **72.5%** |
| 96 | 4 | 24 | 65.0% |
| 96 | 16 | 6 | 55.0% |

Sims-per-world dominates; 16 worlds at 6 simulations each is "indistinguishable from no search at
all". An earlier sweep that split 48 sims over 8 worlds gave each tree 6 simulations and was
therefore confounded — it could not distinguish determinization from having no search.

**What this means for the coverage sweep below.** The sweep was run with `determinize=False`, i.e.
the privileged searcher, at 64 sims and from the **160M** checkpoint — so it is not directly
comparable to any row above, which are @320M. It measures how strength varies with *coverage*, on
one consistent footing; it does not re-measure the privileged/honest gap, which the table above
already puts at 100% vs 58.3% at 48 sims.

**The consequence for an arm is worse than the first version of this entry suggested.** The honest
searcher needs ~384 simulations to reach 76.7%, and §2 measures 64 simulations at 103 decisions/s.
384 simulations is roughly six times that cost again, before any coverage restriction.

## 2. What search costs

Batched searcher (`ai/search/batched_mcts.py`), batch of 256 roots, warmed, median of 15, on the
4090. `reuse_subtree` off (it is off by default: measured 1.20x faster but 11pp weaker).

| sims | ms per batch of 256 | decisions/s |
|---:|---:|---:|
| 16 | 448 | 572 |
| 32 | 1,131 | 226 |
| 64 | 2,493 | 103 |

A first version of this benchmark timed one `run()` per configuration with no warmup while giving
the plain baseline ten warmup iterations. Re-measuring properly moved the 64-sim figure only from
119 to 103 decisions/s, so warmup was **not** the explanation for anything — the cost is real.

Reference points for the same GPU:

* plain policy forward + engine step, batched: ~76,000 decisions/s;
* **end-to-end training: ~8,300 env-steps/s**, from the `E3-20-28` continuation (snapshots every
  600 s, 4,980,736 steps apart). This is the number an arm's cost should be compared against,
  because it already includes the learning updates.

So a 64-simulation search is **~70x slower than the rate training actually runs at**.

## 3. Decision mix

Measured over the 8 regenerated self-play replays (3,010 recorded decisions):

| decision | share |
|---|---:|
| `POINT_NODE` (influence placement) | 39.5% |
| `SELECT_CARD` | 24.2% |
| `SELECT_PLAY_MODE` | 17.8% |
| `SELECT_OP_MODE` | 13.6% |
| `CHOOSE_TIMING_BRANCH` | 4.6% |

**`SELECT_CARD` + `SELECT_PLAY_MODE` = 42.0%** of all decisions. P3's restriction therefore removes
well under half the work by itself; the subsample is where the real saving is.

## 4. What an arm costs

Combining the two, for a 40M / 80M step arm continuing from 160M:

| sims | searched | % of decisions | steps/s | 40M | 80M |
|---:|---|---:|---:|---:|---:|
| 64 | **every decision (the +27pp setting)** | 100% | 103 | **108h** | 216h |
| 64 | card/play-mode only | 42% | 241 | 46h | 92h |
| 64 | card/play-mode, 1 in 4 | 10.5% | 887 | 12.5h | 25h |
| 64 | card/play-mode, 1 in 8 | 5.2% | 1,603 | 6.9h | 13.9h |
| 32 | every decision | 100% | 226 | 49h | 98h |
| 32 | card/play-mode, 1 in 8 | 5.2% | 2,886 | 3.8h | 7.7h |
| 16 | every decision | 100% | 572 | 19.4h | 39h |
| 16 | card/play-mode, 1 in 8 | 5.2% | 4,856 | 2.3h | 4.6h |

Every configuration that fits an 8-hour budget searches 5–10% of decisions. The only configuration
with a strength number attached costs 108h for 40M steps.

## 5. Search-in-training does not exist

There are **zero** references to search or MCTS in `tools/train.py`, `ai/training/train.py`,
`ai/training/generic_trainer.py` or `ai/training/nash_pg.py`. The searcher is wired only into
`tools/play_match.py`, for evaluation. An expert-iteration arm has to be built before it can be
run; P3 is the plan for it.

## 6. Two things that had to be fixed to measure coverage at all

**The searcher had no coverage knob.** `BatchedMCTSConfig` now carries `node_filter`
(`"all"` / `"card_playmode"`) and `subsample`, with `BatchedMCTS.should_search`. Where search is
skipped the agent plays *its own greedy policy*, so a coverage sweep varies exactly one thing.
Exposed on the agent spec as `search:<ckpt>:<sims>:<determinize>:<node_filter>:<subsample>`.

**Evaluating a searcher was accidentally sequential.** `BatchMatchRunner` called
`agent.select_action(state)` in a Python loop, one game at a time, which is the batched searcher's
worst case — its own docstring says so. `BatchedMCTSAgent.select_actions_batch` searches every
covered position in the batch in one call and runs the policy for the rest; the runner prefers it
when present. A batch of 256 roots runs at 103 decisions/s against roughly 0.4/s one at a time.

**`--auto-advance` is required for a search tournament, not optional.** The searcher settles its
root (`advance_root=True`). A caller that does not settle identically receives an action that is
illegal in the state it holds — measured at 7 of 128 positions, which the engine refuses and the
step guard now raises on. With the caller settling the same way, 0 of 128 were illegal.

## 7. Coverage sweep

Measured at `f08cde8` (the working tree it was committed from), engine
`2fb70052dee0164871632209060cbaa23fd4f0fb99cc2e956241e231b0e8e691` — so this is the first search
measurement taken *after* `8533a68`, `b6874af` and `9f78026`, and the only one on this page not
subject to the caveat in §1.

Configuration, spelled out because the comparison to §1 turns on it: the **privileged** searcher
(`determinize=False`, the default), **64 simulations**, from the **160M** checkpoint, against the
plain policy of that same checkpoint, 150 games per side, `--auto-advance`. One matchup per
configuration rather than a round robin: the question is how much strength survives restriction,
and search-vs-search pairs cost the most while answering nothing.

**64 simulations, not 384.** The honest 76.7% in §1 needed 384; nothing here was run at that
budget, which is one of the three reasons these numbers do not compare to it.

| searched | % of decisions | search win % | vs plain | Elo | 40M arm |
|---|---:|---:|---:|---:|---:|
| card/play-mode, 1 in 8 | 5.2% | 55.3% ± 2.9 | +5.3pp | +37 | 6.9h |
| card/play-mode, 1 in 4 | 10.5% | 56.7% ± 2.9 | +6.7pp | +49 | 12.5h |
| card/play-mode, all | 42.0% | 62.7% ± 2.8 | +12.7pp | +90 | 46.1h |
| every decision | 100.0% | 71.0% ± 2.6 | **+21.0pp** | +157 | 107.9h |

**Coverage buys strength and does not saturate.** The gain is sublinear in cost -- 19x the compute
for 4x the edge -- but there is no cheap plateau: the 1-in-8 setting keeps a quarter of the gain
for a sixteenth of the cost, and every step up the ladder is worth real points. An intermediate
reading of the curve, taken after only the two cheapest points were in, looked flat and was wrong;
two points were not enough to see the shape.

**This does not reproduce any earlier figure, and should not be read as doing so.** 71.0% here is
the *privileged* searcher at 64 sims from the 160M checkpoint. The comparable privileged number in
§1 is 100% at 48 sims from 320M, and the 76.7% "+27pp" is the *honest* searcher at 384 sims from
320M. Three variables differ at once -- privilege, budget, checkpoint -- so the only thing this
sweep establishes is the shape of the coverage curve.

**These numbers are an upper bound, because the searcher cheats.** `determinize` defaults to
False, which `BatchedMCTSConfig` itself calls "a privileged teacher": the tree steps the real
`GameState`, so at opponent decision nodes the opponent's legal moves come from their *actual*
hand. A deployable searcher (`determinize=True`) would be weaker by an unmeasured amount, and that
gap is the next thing worth measuring, ahead of any arm.

### What `determinize=True` actually does, and its limits

`BatchedMCTS.run` samples the hidden state **once per search**, at the root:
`determinize(s, acting_player(s), rng)` reshuffles only the cards the mover cannot see -- the
opponent's `HAND_*_UNKNOWN` cards plus the draw deck -- preserving the opponent's hand count, the
deck count, and every card already revealed (`HAND_*_KNOWN`). The whole tree then descends through
that one sampled world, so at an opponent node the legal moves come from the sampled hand and the
opponent is modelled by the same network.

Two limits follow, and the second is not in `dmcts.py`'s docstring:

* **Strategy fusion**, as documented there: the tree may act differently in each sampled world when
  one policy must in truth cover them all.
* **`BatchedMCTS` samples exactly one world per search.** `dmcts.py` describes running an
  independent tree per sample and summing root visit counts; the batched implementation does not.
  So the honest batched searcher carries the full sampling noise of a single determinization with
  no averaging to reduce it -- the weakest form of the technique. `reuse_subtree` is correctly
  disabled under `determinize`, since a tree grown in one sampled world says nothing about the
  next.

## 8. The value bootstrap crosses an information-set boundary

`compute_gae` bootstraps across a change of mover by negating the next state's value:

```python
sign = (curr_p * next_p).float()      # -1 when the mover changes
next_val = sign * self.values_win[t + 1]
delta = self.rewards[t] + gamma * next_val * non_terminal - v_t
```

This is exact in a perfect-information game, where one value function of the state serves both
sides. **It is not exact here.** `v_win` is computed from `extract_observation(state, perspective)`,
which hides the opponent's hand, so `V(s, US)` and `V(s, USSR)` evaluate two different information
sets rather than one state, and are not negatives of each other.

Measured on 227 positions sampled from the s240 self-play replays, with the 240M checkpoint:

| `v_win(s, US) + v_win(s, USSR)` | |
|---|---:|
| mean | +0.051 |
| mean absolute | 0.144 |
| median absolute | 0.067 |
| p90 absolute | 0.374 |
| max absolute | 0.849 |
| positions disagreeing by > 0.2 | 28.6% |

Zero for every position if the assumption held. The observation differs by perspective in 227 of
227 positions, as it must.

**What this costs.** Take the USSR playing a card as an event that resolves with no further
decisions. Its advantage is

```
A = r + γ·( −V(s', US) ) − V(s, USSR)
```

so the continuation is priced by **the US's opinion of the resulting position, formed without
seeing the USSR's hand**. An event whose value depends on what the USSR still holds -- a setup for
a combo, a card it can now afford to discard -- is invisible to that term. The error is not just
noise: the mean is +0.05, so it is biased, by 7.2% of the value range on average and up to 42% in
the tail.

It is self-consistent as a fixed point, because each row's target is built from the next row's
value with the same flip, so the value head converges to something coherent. It is simply not the
information-set value it is being read as.

**The cheap fix, if it is wanted:** bootstrap from the *same player's* evaluation of the next
state -- store `V(s_{t+1}, p_t)` alongside `V(s_{t+1}, p_{t+1})` -- which is one extra forward per
step and removes the boundary crossing entirely. That changes the algorithm, so it is recorded here
rather than done.

