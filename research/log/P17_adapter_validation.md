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
Feeds §7 of [`../plans/P17_action_representation.md`](../plans/P17_action_representation.md); see
also [`P17_corpus_restoration.md`](P17_corpus_restoration.md).
