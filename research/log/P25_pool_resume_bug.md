# The opponent pool drained on every resume (fixed in 3803d5d), and the late collapse

2026-09-24. Plan: [P25](../plans/P25_collapse_robustness.md), step 3h. Runs: E4-48 in
[`../runs.md`](../runs.md).

## The bug

`OpponentPool.load_state_dict` restored each member's id and win/game record, but not the step it
was captured at. The constructor records every seed net at step 0, so after a resume the whole
restored pool sat at step 0. Two things follow:

1. **The pool drained to its recent end.** Eviction keeps the pool spread by dropping the interior
   member whose neighbours are closest together. Members at one step have zero spacing, so every new
   snapshot after a resume evicted a restored member, until only two remained: the first and last
   in the (arbitrary) order of the zero-step block. Within about 30M steps of a resume the pool was
   those two plus the leg's own recent snapshots.
2. **The next resume dropped them.** A resume finds members on disk by step
   (`snapshot_<steps>steps.pt` in the resumed run's directory). A member recorded at step 0 matches
   nothing, so it is dropped with the message `N of 12 recorded members have no snapshot on disk`.
   The same lookup also lost any member written by an earlier leg of the lineage, since those live
   in that leg's directory.

The E4-08-05 lineage, from each leg's recorded pool (ids → steps):

| leg | pool at the leg's end | what the next leg restored |
|:---|:---|:---|
| 0 → 80M (`E4-08-05_20260919_233851`, fresh) | 0, 5, 15, 20, 30, 35, 45, 50, 55, 65, 70, 80 | 11 members, all recorded at step 0 |
| 80 → 160M (`E4-08-05_20260921_033333`) | **0, 0** (really 5 and 80), 85, 90, 100, 105, 115, 120, 130, 140, 150, 160 | 10 (the two step-0 members dropped) |
| 160 → 240M (`E4-08-05_20260923_095214`) | **0, 0** (really 85 and 160), 165, 175, 185, 190, 200, 205, 215, 225, 235, 240 | — |

The E4-08-03 lineage has the same shape. E4-44-05's continuation at 190M came back with 6 of 12,
which is how this was found.

**Scope.** 57 E4 run directories are resumes of a self-growing pool. Every one of them trained after
its resume point against a pool drained to its recent end. Runs from scratch never resume, so they
are unaffected, and that includes the whole P25 stress bench. A comparison between two
continuations is still fair, since both were drained in the same way. A continuation compared with
the pool the design describes is not.

**The fix** (`3803d5d`):
* restores the steps;
* records each member's snapshot path, so a continuation keeps the members earlier legs wrote;
* keeps the steps in the rebuild fallback too.

Tests: `tests/training/test_opponent_pool_resume.py`, including a check that a resume leaves the
same pool as running straight through. States written before the fix are repaired by
`data/logs/p25/repair_pool_160M.py`, which maps the step-0 ids back to steps through the previous
leg's own record. It writes a copy and never touches the original.

## E4-48: the late-collapse test on a fixed pool

Every continuation from `E4-08-05@160M` collapsed, and all of them ran on the drained pool:
* the plain continuation, pinned at 195M;
* the seed-11 branch, at 215M;
* slow π_ref (E4-31-05), at 235M;
* the two WoLF late tests (E4-44/45), at 180–190M.

E4-48 is the plain continuation again, with no change except the fix. It resumes from a repaired
160M state holding 12 members at 5, 80, 85 … 160M. `launch_flags.py --diff` against each baseline:
identical.

### Collapse half

Self-play USSR share by 5M bucket, 160–240M:

| run | 160–240M, by 5M | peak | buckets ≥ 0.9 |
|:---|:---|---:|:---|
| **E4-48-05** (fixed pool) | .77 .72 .81 .79 .74 .79 .90 .90 .78 .73 .80 .71 .65 .69 .79 .69 | 0.898 | none |
| E4-08-05 (drained) | .75 .66 .65 .55 .64 .80 .82 .98 .99 .97 .99 .99 .97 .98 .99 .99 | 0.99 | 195–240M |
| **E4-48-05-160M.11** (fixed pool) | SEED11 | | |
| E4-08-05-160M.11 (drained) | .77 .87 .88 .83 .80 .77 .75 .34 .30 .64 .75 .99 .98 .99 .99 .99 | 0.99 | 215–240M |

E4-48-05 went to the edge at 190–200M (0.897, then 0.898). Its advantage spread fell to 0.186 and
its US entropy rose to 1.83, the same signature every baseline showed on the way into its pin. It
then came back: 0.78 by 200–205M and 0.65–0.80 for the rest of the leg. The drained baseline
entered the same episode 5M earlier and stayed pinned at 0.97–0.99 for the remaining 45M.

RESULTS

## Open

* E4-48-05-160M.11's first leg died at ~215M on a CUDA `unspecified launch failure`, raised at
  the host-to-device copy that follows the two graph replays in `_graphed_forward`. The error is
  asynchronous, so the faulting kernel is earlier, most likely inside a replay. This is the first
  such fault in about 20 runs since graphs were introduced. The kernel log is not readable here, so
  a hardware Xid cannot be ruled out. It was continued from its 215M resume state.
