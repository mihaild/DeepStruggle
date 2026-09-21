# P21 — M2a / M2b / M2c: what the country head actually needs

One tournament, 17 entrants, 100 games per side at `temperature 0.0`, run 2026-09-21.
`/workspace/data/reports/P21_M2abc.{json,md}`.
Decomposes the mechanism measured in [`P21_ladder_status.md`](P21_ladder_status.md) as worth
**+282 Elo**.

`pe_country`'s input is three things concatenated, and nothing had separated them:

```
pe_country( [ 15 dynamic slots ‖ 11 static constants ‖ 64-wide ctx ] ) → 64 → 1
```

| rung | head input | width | `pe_trunk` |
|:---|:---|---:|:---|
| M2d | dynamic + constants + ctx | 90 | yes |
| **M2a** | dynamic + constants, no ctx | 26 | **removed** |
| **M2b** | dynamic + ctx, no constants | 79 | yes |
| **M2c** | dynamic only | 15 | **removed** |

The head widths were verified by construction before launch, because a flag that silently did
nothing would train happily and quietly duplicate M2d. Parameter deltas from M2d are at most
34,880 of 3,184,960 — **1.1%** — so nothing here can be a capacity effect.

**Seeds 3 and 5 for every rung**, rather than the ladder's usual seed 1. Both are clean for M2d at
both budgets and already rated there, so every comparison below is **within-seed** and free of the
~100 Elo seed spread. Seed 1 is also the seed on which M2d collapses — it produced the false
"the country head buys nothing" once already.

## Result: every part is pulling weight

| rung | 80M s3 | 80M s5 | 160M s3 | Δ80M s3 | Δ80M s5 | Δ160M s3 |
|:---|---:|---:|---:|---:|---:|---:|
| **M2d** | 2076.2 | 2007.2 | 2205.7 | — | — | — |
| M2a — no ctx | 1946.9 | 1923.2 | 2059.7 | **−129.3** | **−84.0** | **−146.0** |
| M2b — no constants | 2003.1 | 1958.1 | 2165.2 | **−73.1** | **−49.1** | **−40.4** |
| M2c — dynamic only | 1989.3 | 1882.0 | 2016.4 | **−86.9** | **−125.2** | **−189.3** |

**All three lose on both seeds at 80M, so none is adopted.** M2d's full head input stands.

## What each part is worth

**Context is the expensive one to remove: −84 to −146 Elo.** This refutes the prediction recorded
when the rung was launched. The reasoning was that dropping `ctx` still leaves the 15 dynamic
slots, so the correction continues to track influence, control and can-place and only loses
*global* state — DEFCON, turn, score, space race. That turned out to be exactly the state the
correction needs. `pe_trunk` (480×64) stays.

**The hand-designed constants are worth less but are not free: −40 to −73 Elo.** The positional
trunk does not already carry them, as M2b was testing. This is the smallest effect of the three
and the only one that shrinks with budget (−73/−49 at 80M against −40 at 160M), which is the
signature of something the network can partly learn around given time.

**The minimal head still carries most of the mechanism.** M2c — 15 inputs, no trunk connection —
rates +251.7 over M1 on seed 3 and +144.4 on seed 5. Against M2d's +338.6 and +269.6, the dynamic
slots alone are worth roughly **54–74%** of the whole head. So the mechanism is mostly "let each
country adjust its own logit from its own current state", and context and constants are
refinements on top rather than the substance.

**The removals are not additive.** M2a −129.3 and M2b −73.1 on seed 3 would predict about −202 for
M2c; it measures −86.9. On seed 5 the ordering of M2a and M2c reverses entirely. Whatever the
interaction is, it is inside seed noise at two seeds and should not be interpreted.

## Throughput does not rescue them

Dropping the trunk connection is the fastest configuration — M2a 54,895 steps/s and M2c 53,644
against M2d's ~46,600–50,500, about **+10%**. That buys nothing: −129 Elo for +10% throughput is a
far worse trade than the ladder's existing options, and M2d at 160M already beats every ablation
at 160M while running only marginally slower.

## Side balance, noted but not interpreted

M2a at 160M is the only ladder arm with a **negative** side gap (−4.7), closer to the anchor's
profile (−9.8) than anything else measured, while M2c at 160M is the most skewed (+18.5). Single
arms, and side gap has been unstable across budgets throughout this study, so this is recorded
rather than claimed.

## Stability

**All six rung arms are CLEAN.** No collapse, no pinned rows beyond a single isolated row in
`E4-14-03`. Worth recording under the protocol's per-rung collapse column: three architecture
variants, six arms, zero collapses, against M2d's 1-in-8 on its own seeds. Too few arms to compare
rates, but it is the first evidence bearing on whether the collapse is M2d-specific.

## Method notes

* Ratings are field-relative. M1 rates 1737.6 here and 1761.8 in the previous tournament,
  unchanged. Only deltas inside this JSON mean anything.
* The 80M point of each 160M arm is that arm's own ~80M snapshot, so it shares a byte-identical
  prefix with its 160M sibling rather than being a separate run.
* `E4-08-03`'s 42,268 steps/s is contention from the day it trained, not architecture; its
  siblings run ~50,000.
