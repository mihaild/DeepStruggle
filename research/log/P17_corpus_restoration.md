# P17: restoring the human corpus after the resolution merge

Companion to [`../plans/P17_action_representation.md`](../plans/P17_action_representation.md).
What the merge broke in the log converter, how each was found, and what the measurement bar is.

## The bar, and how it was nearly lost

`AGENTS.md` §6 and `tools/README.md` §6 both state **every downloaded game converts**. The
corpus is 300 games. After the merge it converted 294, and for a while I treated 294/300 as the
baseline and the six as a known floor. It was not a baseline — it was a regression I had
introduced, and calling it a floor almost retired a check that had been passing.

The rule this is an instance of: **a number that used to be 100% is not a baseline just because
it is the number you are looking at.** Read what the docs claim before deciding what "expected"
means. Cross-reference: [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md).

| stage | converted | guessed | board mismatches |
|:---|---:|---:|---:|
| after the merge | 294 / 300 | 0 | 0 |
| after the Flower Power fix | 298 / 300 | 0 | 0 |
| after the `cur_mode` fix | **300 / 300** | **0** | **0** |

Final: 29,820 / 30,620 entries, **122,056 decisions**. The 800 unconverted entries are the
documented ones — turns whose recording stops part way — not failures.

## Bug 1: Flower Power stranded in the ops branch (games 56, 138, 258, 272)

Flower Power charges the US 2 VP for playing a "war" card. The pre-merge flow charged it at the
`PlayMode::OPS` step, which every path to Ops went through. The merged handler left the charge
inside the `is_ops` branch — and **event-first does not pass through it**. Korean War* and
Arab-Israeli War are USSR cards, so a US player plays them as opponent cards; played event-first,
the charge never happened and the turn's military-ops VP came out 2 short.

Fix: `commit_card_to_ops()` in `state_machine.cpp`, holding `clear_force_for_card`, the Flower
Power charge and the Formosan Resolution clear, called from **both** branches.

The failures presented as "score mismatch at the turn's last action round", which is where
military ops are paid — not where the card was played. The distance between symptom and cause is
the reason this took a full log dump to see.

## Bug 2: the resolution node never recorded `cur_mode` (games 123, 124)

`cur_mode` tracks which Ops mode is being spent, and the coup/realign reconciliation reads it to
decide whether a die needs explaining. The deferred `SELECT_OP_MODE` set it. **The merged
resolution node did not** — so an ops-first play inherited the *previous* card's mode.

Turn 9's headline of replay 123 is the position that needs all of it to go wrong at once:

```
USSR headlines ABM Treaty   -> event improves DEFCON, then spends the card's 4 Ops
                               to coup Pakistan            -> Pakistan [0][0]
US   headlines Grain Sales  -> US plays the revealed Portuguese Empire Crumbles*
                               ops-first, placing +2 in Pakistan -> Pakistan [2][0]
```

The placement was still labelled `"coup"`, so the converter went looking for a die that would
leave Pakistan at its *final* `[2][0]` — a board the coup never produced, and no roll reaches it.
It needs a coup and a second Ops resolution in the same entry **on the same country**, which is
why 2 games in 300 hit it.

The trace that settled it, with `_peek_op_mode` and `_find` instrumented:

```
PEEK -> 'coup'      outcomes=[(32,0,0)]
  FIND SELECT_OP_MODE -> 113        deferred op mode = OPS_COUP, sets cur_mode='coup'
  FIND POINT_NODE     -> 151        coup Pakistan; coup_roll_queue non-empty -> `pass`, correct
PEEK -> 'influence' outcomes=[(32,2,0)]   Portuguese card, ops-first, via the RESOLUTION node
  FIND POINT_NODE     -> 151        the +2 placement
  FORCE_OUTCOME {32:(2,0)} -> False treated as a COUP
```

### A wrong diagnosis worth recording

My first reading was that `step_outcomes` — one list shared by reference across both
`_peek_op_mode` call sites — was being clobbered by the later peek. That is *observably true* in
the trace and completely beside the point: the coup had already consumed its own entry before the
overwrite. The shared buffer looked guilty because it was the thing that visibly changed. The
actual cause was one assignment that never happened.

**A state mutation you can see is not thereby the cause of a failure downstream of it.**

## A method note: the stale `.so`

Four corpus sweeps run after the Flower Power fix reported the identical pre-fix six failures. I
read that as the fix having failed. It had not: the sweeps were *launched* before the rebuild
finished and had the old extension mapped for their whole life. `check_engine_fresh.sh` passing
afterwards says nothing about a process that started earlier.

Running the six games alone took seconds and showed four already clean. **When a batch result
disagrees with a fix you believe in, re-run one case in a fresh process before re-opening the
diagnosis.** Invariant 13 covers the run you launch; it does not cover the one already in flight.

## What else the merge had not reached

Found while migrating, all in the same change:

- **`bindings/action_encoder.py` still described the pre-merge layout.** `encode`/`decode`
  delegate to C++, so the codec was right and the *constants* were wrong — which is worse, because
  nothing failed. `ai/training/behavioral_cloning.py` had already been migrated to `ts.Resolution`
  and was indexing `OP_MODE_OFFSET + OpMode` = 116/117/118, the three slots the merge left
  unassigned and the engine refuses. The deferred Ops choice shares the `OPS_*` resolution slots,
  so `OP_MODE_OFFSET` is **112**.
- **All six bots** chose play modes by the old indices and answered a node that no longer occurs.
  `strategic_bot`'s op-mode logic is now a helper both nodes call; `exploratory_bot`'s 60% Ops is
  split 27/21/12 by its own 45/35/20 op-mode weights, which leaves the joint distribution over
  (play mode, op mode) exactly what it was.
- **`tests/training/test_batched_mcts.py`** — not a regression, a latent config mismatch the merge
  exposed. See below; this is the answer to the standing question about search returning illegal
  actions.

## Search returns illegal actions because the root is advanced, not because of determinization

`BatchedMCTSConfig.advance_root` defaults to **True**, and its own docstring says what that costs:
the tree roots at a later decision than the caller holds and returns an action illegal there. With
`auto_advance=True` the searcher settles past any decision with no discretion — **including a
single-legal-action node**. The failing case is a `POINT_NODE` whose only legal action is 211
(confirm/done); the searcher skips it and returns 112, legal one decision later.

This has nothing to do with determinization, which was my earlier explanation and was wrong —
measurement had already shown that filter to be a no-op (402/402 masks identical). The merge
removed ~20% of decisions and moved which positions the scripted openings reach, so they now land
on such a node and the mismatch surfaces.

The two tests assert the pick is legal *in the state handed over*, which is what
`advance_root=False` provides, so that is where the flag belongs. **Left open for the owner:**
whether `best_actions` should carry the "search proposes, true mask disposes" guard the agent path
already has at `batched_mcts.py:457`, or whether the config contract is the right place to keep
it. Adding the guard would hide the mismatch rather than surface it, so it is a call about what
the API promises, not a bug fix.

## Artifacts

- Corpus sweep script: `$CLAUDE_JOB_DIR/tmp/corpus_sweep.py` (not committed — it is a harness,
  and `tests/replayer -m corpus_full` is the committed form of the same check)
- Replays regenerated across six bot matchups under `data/replays/`; the pre-merge logs are in
  `data/replays/archive_pre_p17/`, following the existing `archive_pre_p14` convention
- Verification: pyrefly 0 errors; **1593 passed / 5 skipped** across `tests/bindings`,
  `tests/engine_logic`, `tests/replayer`, `tests/training`; 372 C++ tests
