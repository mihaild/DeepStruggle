# P17: the same checkpoints score the same on both engines

Plan §7's sanity check, finally run properly:
**three checkpoints, 300 games per pair, T=0.1, old engine natively vs new engine through the
adapter.**

| matchup | old engine | new + adapter | delta | z |
|:---|---:|---:|---:|---:|
| `snapshot_0s` vs `28M` | 59.7% (179/300) | 59.0% (177/300) | −0.7pp | −0.17 |
| `snapshot_0s` vs `36M` | 98.3% (295/300) | 98.0% (294/300) | −0.3pp | −0.30 |
| `28M` vs `36M` | 94.0% (282/300) | 95.7% (287/300) | +1.7pp | 0.92 |

Largest |z| = 0.92 against a two-sided 5% threshold of 1.96. **The merge did not move playing
strength.** The checkpoints are from `E3-35-28_20260918_001501`; the old engine is `ebfd55d`
rebuilt from a `git archive` export.

`snapshot_0s` is **not** untrained, which is how it was first described here and it was wrong.
`E3-35-28` is a resumed run (`resumed_from: E3-34-28_20260917_222015`, `warmup_checkpoint: None`),
so its `0s` file is the inherited model — bit-identical across all 101 tensors to `E3-34-28`'s
final 28.0M snapshot, with the resume chain continuing back to `E3-29-28`. It therefore tops the
ladder because it is the *pre-collapse* model and the run degraded away from it, not because an
untrained network beats trained ones. See
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md) §"A checkpoint's filename
is not its provenance". The wide 98% / 2% matchups this produces are useful here regardless: they
make the comparison far more sensitive than 50/50 ones would.

## What made it possible

Plan §7 designed around a constraint that no longer holds:

> `GameState` cannot be serialised through the bindings — no `to_bytes`, no pickle support.

`to_save_dict` / `state_from_save_dict` carry scalars, RNG, card locations, influence, headline
owners, the die-roll record **and the ctx_stack**, and the two trees share a byte-identical
`game_state.hpp`. Transplanting a live new-engine position into the old engine yields an identical
legal mask and an identical observation, **max abs difference 0**. No C++ engine change was needed.

Two engines in one process needs **both**:

* a distinct module name — CPython locates an extension by its `PyInit_<name>` symbol, so two
  builds both called `ts_engine` collide no matter what key they take in `sys.modules`;
* a distinct nanobind domain — the type registry is process-global, and the second build otherwise
  aborts with *"refusing to add duplicate key NONE to enumeration ts_engine.RollType"*.

The old build is therefore compiled as `ts_engine_old` with `NB_DOMAIN ts_old`.

## The wrong turn, and what it cost

The first adapter reconstructed the old masks and asked the old policy on the **new** engine's
observation. It differed from the old engine (largest |z| 3.04) because the observation carries an
8-wide one-hot over `DecisionType` at `ctx_slots::DECISION_TYPE` — so the old policy's 2nd and 3rd
questions arrived under a `SELECT_PLAY_MODE` identity it never saw at those nodes.

Restating just that one-hot made it **worse: |z| 3.04 → 8.87.** With `pending_ops_value`, `op_mode`
and the timing slots still holding their `SELECT_PLAY_MODE` values, the result is an internally
inconsistent vector — off the training manifold rather than merely at the wrong point on it.

> **A partial correction to a structured input is worse than no correction.** Fix every coupled
> field or none.

The working answer is the owner's: each merged resolution is a *sequence* of old decisions, so
walk that sequence on a real old engine and harvest the decisions. The old state is a throwaway —
stepping it resolves events and burns its own RNG, and none of that is kept.

## The bug the whole-game test could never have found

With the chain walked on a real engine, the merged node verified **1109/1109 exact** against the
native old chain — and the tournament *still* differed, now compressed uniformly toward 50%
(59.7→52.0, 98.3→84.0, 94.0→76.7). Uniform compression toward chance is the signature of **added
policy noise**, not of a mapping error.

The cause was in the adapter's own comment, which asserted that the deferred `SELECT_OP_MODE`
"occupies the same index in both spaces". It does not:

| | old | new (merged) |
|:---|:---|:---|
| Ops mode | **116 / 117 / 118** | **112 / 113 / 114** |

The node exists in both engines, so it looked like a pass-through. But the merge put the deferred
choice on the `OPS_*` resolution slots, while the old policy's head learned it at 116–118 — so on
every event-first play the adapter read logits that meant SPACE / PASS / OPS_FIRST to that network.
Remapping the mask fixed it, and the three matchups collapsed to |z| ≤ 0.92.

After the fix the deferred node agrees **934/942 (99.15%)**. All 8 remainders are P17's *intended*
tightening: `state_machine.cpp:1197` refuses INFLUENCE when `free_action::region_locked`, where the
old engine offered it as a skip. The adapter records them as divergences rather than quietly
choosing, which is how they were found.

## Method note: the instrument was wrong for three rounds

The whole-game tournament reported "differs" three times, with rising confidence (|z| 3.04, then
8.87, then 6.19), and never once indicated *where*. A decision-level fidelity check over ~900
positions located the defect exactly, in one run, and would have caught the one-hot mistake before
any of the tournaments were spent.

> **Validate a translation layer at the translation, not at the outcome.** A win rate integrates
> every decision in a game, so it can say "wrong" but never "wrong here" — and its confidence grows
> with sample size whether or not the diagnosis is any closer.

## Reproducing

```bash
# old engine, native
cd /workspace/data/p17/old_engine_ebfd55d
PYTHONPATH=.:build/release python tools/tournament.py --models <ckpts> \
    --games-per-side 150 --temperature 0.1 --output-json tourney_OLD.json

# new engine, old checkpoints through the adapter
export TS_OLD_ENGINE_SO=/workspace/data/p17/old_engine_ebfd55d/build/release/ts_engine_old.*.so
PYTHONPATH=.:build/release python tools/tournament.py \
    --models "legacy:<ckpt>" ... --games-per-side 150 --temperature 0.1
```

`legacy:<checkpoint>` is a `load_agent` prefix alongside `temp:` and `search:`, so the comparison
runs through the same CLI as every other match (invariant 9). Artifacts under `/workspace/data/p17/`.
Feeds §7 of [`../plans/P17_action_representation.md`](../archive/E3_ladder/plans/P17_action_representation.md); see
also [`P17_corpus_restoration.md`](P17_corpus_restoration.md).

---

# The wide check: seven agents, 50,400 games per engine

§7's sanity check at scale, on deliberately diverse opponents. Seven agents, 21 pairs,
**1,200 games per side (2,400 per pair) = 50,400 games per engine**, T=0.1.

| agent | why it is in the field |
|:---|:---|
| `E3-17-26` @ 200M | 200M, arm A |
| `E3-20-28_20260915` @ 200M | 200M, arm B — cross-arm at equal steps |
| `E3-20-29_20260915` @ 200M | 200M, arm C |
| `E3-17-26` @ 165M | same arm as A, different step count |
| `E3-35-28` @ 36.5M | a different arm entirely, much weaker |
| `E3-35-28` @ 0 steps | the inherited pre-collapse model — strongest here |
| `HeuristicBot` | rule-based, so its only cross-tree difference is index mapping |

The Elo ladder spans 1500–2320, with three arms tied at 200M separated by ~135 Elo. That narrow
band is the sensitive part: a representation change that moved strength at all would show there
first.

## Result

| check | 400/side | 1,200/side |
|:---|---:|---:|
| pairs with \|z\| > 1.96 | 0 of 21 | **0 of 21** |
| largest \|z\| | 1.60 | **1.66** |
| global chi2 (21 df) | 10.6 | **13.1** |
| largest Elo shift | 11.7 | **18.9** |
| engine anomalies, old / new | 0 / 0 | **0 / 0** |
| step refusals | 0 | **0** |

Three guards, all independent: `report_anomaly` on stderr (the `may_fizzle` / POINT_NODE guard),
the adapter's own divergence counters, and `IllegalActionError` at `batch_tournament.py:325` —
which raises if the engine ever refuses a stepped action, so a run that *completes* proves no
refusal occurred.

## Two things not to over-read

**The chi2 is below its own degrees of freedom** (13.1 on 21; 10.6 on 21 at the smaller size).
Under a true null with independent binomial sampling the expectation is 21. Coming in low means
the two tournaments are positively correlated — they share seeding — so the per-pair z is
*conservative*, not precise. The honest claim is "no detectable shift at this resolution", not a
tight two-sided bound.

**Every neural model's Elo moved down** (−4.7 to −18.9) with HeuristicBot pinned at 1500, which
looks like a systematic cohort shift. It is not six independent observations: Bradley-Terry fits
them jointly against a fixed anchor, so a common move is one degree of freedom. Measured directly,
the pooled neural-vs-heuristic win rate went 93.91% → 93.63%, a **−0.28pp shift at z = −0.98 —
not significant**, and worth ~8 Elo at that base. Near 94% a fraction of a percentage point maps
to a lot of Elo, which is the whole of the apparent effect.

## Divergences: named, not zero

Over ~12.7M adapter decisions:

| divergence | count | status |
|:---|---:|:---|
| `deferred op mode: no Ops mode offered` | 2,614 (0.02%) | **intended.** `state_machine.cpp:1197` refuses INFLUENCE when `free_action::region_locked`, where the old engine offered it as a skip |
| `old engine reached SELECT_PLAY_MODE, not SELECT_OP_MODE` | 43 (0.0003%) | **open** |

The second is always Arab-Israeli War, and it is *not* understood. After the adapter steps the old
engine through OPS and OPS_FIRST it expects `SELECT_OP_MODE` and sometimes finds a play-mode
question again. A probe forcing OPS_FIRST over 400 games reproduced it **zero** times, so it
depends on the sampled path in a way that has not been isolated. The adapter falls back to the
first legal action, which bounds the damage at 4 decisions per 100,000 — invisible in the outcome
comparison above, but recorded here rather than rounded away.

## What is NOT covered

**Search agents do not go through the adapter.** `search:` builds a `BatchedMCTS` that reads the
model's *distribution* over the merged action space — the same class of defect as the deferred
op-mode bug, relocated inside the tree. Adapting it means mapping old priors per leaf:

```
P(EVENT)  = p(EVENT)                        own / neutral
          = p(OPS) * p(EVENT_FIRST)         opponent
P(SPACE)  = p(SPACE)
P(OPS_x)  = p(OPS) * [p(OPS_FIRST)] * p(mode = x)
```

which is well defined but needs an old-engine transplant per leaf, at thousands of leaves per
decision. `HeuristicBot` stands in for play-style diversity instead, and anchors the ladder at
6.5%.

Artifacts: `/workspace/data/p17/big2_{OLD,NEW}.{json,md,stderr}`.
