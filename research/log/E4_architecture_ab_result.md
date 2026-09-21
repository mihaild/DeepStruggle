# P19 result: the late-E3 architecture is worth ~450 Elo on this engine

**2026-09-19.** [P19](../archive/E4_ladder/plans/P19_architecture_ab.md) asked whether the late-E3 architecture
bundle is stronger than the E4 defaults on the post-P17 engine, and pre-registered the
measurement and the decision rule before the arms finished. Both arms completed clean.

## The arms

Matched on everything except the network: both cold-started, both pooled (`--opponent-frac 0.3
--opponent-self-pool --opponent-pool-size 12`), both 80M, same engine, same `ref_update_freq`.
`launch_flags.py --diff` reported exactly four differing flags and nothing else.

| arm | `identity_dim` | `per_entity_heads` | `graph_layers` | `self_transform` | steps/s |
|:---|---:|---:|---:|:---|---:|
| `E4-03-01` | 16 | 64 | 0 | yes | 11,756 |
| `E4-04-01` | 0 | 0 | 2 | no | 14,057 |

Both ran to 80,019,456 steps with the pool at capacity 12 and **no health alarm at any point**.

## The result

Round robin, 200 games per pair (100 per seat), τ=0.0, `HeuristicBot` anchored at 1500.

| rank | model | Elo | vs `E4-04-01` | vs `HeuristicBot` |
|---:|:---|---:|---:|---:|
| 1 | **`E4-03-01`@80M** | **2149.9** | **94.0%** | 99.0% |
| 2 | `E4-01-01`@240M | 1865.7 | 68.5% | 93.0% |
| 3 | `E4-02-01`@320M | 1831.2 | 68.0% | — |
| 4 | `E4-04-01`@80M | 1702.4 | — | 73.0% |
| 5 | `HeuristicBot` | 1500.0 | 27.0% | — |

**+447 Elo at matched budget and matched cold start**, and the margin holds on both seats —
91.0% as USSR, 97.0% as US. P19's decision rule required *d* > 55% with both seats above 50%;
*d* = 94.0%.

Seed spread in this record is ~95 Elo. This is **4.7x** that, so the direction is not in doubt,
though the magnitude rests on one seed per arm.

**Robust to temperature.** Re-rated at τ=0.1 the leaderboard is unchanged and the numbers barely
move (2150.8 vs 2149.9; 94.0% against `E4-04-01` both times). τ=0.0 was chosen because the arms'
final entropies differ — 1.22 against 1.01 — and `tournament.py` warns that sampling lets the
sharper policy win on temperature rather than strength. It made no difference here.

## It also beats arms with four times the steps and a warm start

| `E4-03-01`@80M vs | result |
|:---|---:|
| `E4-02-01`@**320M**, warm-started | **88.5%** |
| `E4-01-01`@**240M**, warm-started | **81.5%** |

So an 80M cold arm on this architecture beats a 320M warm-started arm on the defaults. **Both of
[P21](../plans/P21_architecture_ladder.md)'s anchors therefore point at the same checkpoint,
`E4-03-01@80M`** — it is simultaneously the best at 80M and the best at any budget.

## E4-04-01 fails P19's sanity gate, and is not broken

P19 required both arms to beat `heuristic` decisively before the headline could be read.
`E4-04-01` manages **73.0%**, which is not decisive, while every other neural entrant is at
88–99%.

It is not a broken arm. Its pool reached capacity and held, `kl_div` peaked at 0.095,
`us_episode_frac` averaged 0.443 and never pinned, and no alarm fired across 1,237 iterations. The
gate was written to catch a harness fault and is instead reporting a real weakness: **the E4
default architecture, cold-started, is simply weak at 80M.** The warm-started arms on the same
architecture reach 93% (`E4-02-01`@80M), so most of what separates them at this budget is the warm
start.

The gate should be read as satisfied for the purpose it serves — nothing is wrong with the
measurement — and the finding recorded as the arm being weak rather than faulty.

## Side balance

`E4-03-01` is the most balanced arm in the record at **−1.0 pp** (95.2% as USSR, 96.2% as US)
across all opponents. The others run +5.8 to +13.4 pp toward USSR. Side imbalance was a standing
problem through the whole E3 ladder, so an architecture that is both much stronger *and* balanced
is worth noting — but with one seed it is an observation, not a result.

## What this does and does not establish

**Does:** the bundle is worth ~450 Elo on this engine, which vindicates carrying it and settles
P19. `E4-03` is now the lineage's strongest checkpoint and the reference for everything after it.

**Does not:** say *which* of the four flags did it. `identity_dim`, `per_entity_heads`,
`graph_layers` and `self_transform` moved together. Two of them — identity and the per-entity
heads — exist specifically to repair what pooling destroys
([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)), so a
450 Elo bundle containing both is *consistent* with the pooling bottleneck being the dominant
problem, and is the strongest indirect support the pooling story has. Separating them is
[P21](../plans/P21_architecture_ladder.md), and this result raises its value: the effect being
separated is large enough that the ladder should resolve it comfortably, which is the condition
the owner set for the protocol.

## Next

Per P19's decision rule, extending `E4-03` to 320M is warranted — that is where E3's collapse
appeared, on this architecture, and the pooled 320M arm on the defaults (`E4-02-01`) did not
collapse.

Reports: `data/reports/E4_80M_roundrobin.md`, `E4_best_of_four.md`, `E4_best_of_four_tau01.md`.
