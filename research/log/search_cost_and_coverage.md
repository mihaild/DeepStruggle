# Search: what it costs, and how much of it is the coverage

What a "train with search" arm would actually cost, and the measurement that has to happen before
one is launched. Written while planning an arm from the 160M checkpoint of `E3-20-28`.

Everything here was measured on the current engine
(`2fb70052dee0164871632209060cbaa23fd4f0fb99cc2e956241e231b0e8e691`) with
`tools/scripts/check_engine_fresh.sh` clean.

---

## 1. The strength number we had was measured at full coverage

Earlier in this line of work the batched searcher was measured at roughly **+27pp** win rate over
the plain policy with a trained critic, and **−17pp** with an untrained one. That result was never
written to this log; it is recorded here so it stops being folklore.

**It was measured with search at *every* decision.** `_SearchBot.select_from_state` in
`tools/play_match.py` searches whatever node it is handed, and `BatchedMCTS` has no node-type
filter at all — its only mention of `decision_type` is in an error message. There was no
restriction to "the decisions that matter".

This matters because `research/plans/P3_determinized_search_expert_iteration.md` proposes search
"at a `SELECT_CARD` / `SELECT_PLAY_MODE` node (the decisions that matter; skip placement
micro-actions)" on a subsample, "1 in 8 searched decisions is the first guess". Those restrictions
are **8–45x cheaper and were never measured**. Reading the +27pp as though it applied to them is
the error this entry exists to prevent.

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

Each configuration plays the **plain policy of the same 160M checkpoint**, 150 games per side, 64
simulations, `--auto-advance`. One matchup per configuration rather than a round robin: the
question is how much strength survives restriction, and search-vs-search pairs cost the most while
answering nothing.

(results appended below when the sweep completes)
