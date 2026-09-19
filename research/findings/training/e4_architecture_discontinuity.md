# E4 was launched with a different network from every late E3 arm

**Found 2026-09-19**, by checking rather than assuming. Both E4 arms ran the bare architecture
defaults; all eleven late E3 arms did not.

| flag | E3-30 … E3-37 (11 runs) | **E4-01-01, E4-02-01** |
|:---|:---|:---|
| `identity_dim` | 16 | **0** |
| `per_entity_heads` | 64 | **0** |
| `graph_layers` | 0 | **2** |
| `self_transform` | True | **False** |

The state dicts confirm it independently of the metadata, which is the check that matters —
metadata records what was asked for, weights record what was built:

```
only in E3:  card_identity (110,16)   country_identity (84,16)
             pe_card.0 (64,158)       pe_country.0 (64,170)      board_fc.0 (64,42)
only in E4:  gconv1.linear (64,26)    gconv2.linear (64,64)
differing :  card_fc.0  E3 (64,30) -> E4 (64,14)     [the 16-wide identity concat is gone]
             policy_head.3  E3 (212,...) -> E4 (220,...)   [expected: the P17 repack]
```

Only the `policy_head` difference is the intended one.

## Cause

The training command in `CLAUDE.md` does not carry the architecture flags, and E4 was launched
from it without diffing against a recent run's `metadata.json`. This is the same failure as the
opponent-pool omission a few hours earlier, and **invariant 14 already says to do that diff** — it
was written after the pool incident and then not followed for architecture.

The continuations inherit from the originals, so all four E4 directories share it.

## It costs ~18% per forward, and does NOT explain the GPU anomaly

Both nets built fresh at equal action width, batch 512, same device:

```
E4 (as launched)  3.09M params   340.0 fwd/s   2.94 ms
E3 (late arms)    3.15M params   288.9 fwd/s   3.46 ms
```

Near-identical parameter counts; E3's is 18% more expensive. That is far too small to explain the
100% → 55% utilisation drop between engines, so
[`../../plans/P18_training_throughput.md`](../../plans/P18_training_throughput.md)'s headline
anomaly stands.

## What it confounds

Every E3↔E4 comparison made on 2026-09-19 now carries an architecture term:

* **Throughput.** E3-22-28's 12,157 st/s against E4's 14,335 is +18%, and the architecture alone is
  worth ~18% per forward. The +40% "game progress per second" figure in P18 mixes two causes and
  cannot be split without rebuilding one arm. The **16% fewer decisions per turn** measurement is
  unaffected — it counts decisions, not time.
* **Blunder rates.** E3-20-28 against E4-02-01 at matched steps compares two networks as well as
  two engines.
* **Strength.** E3 checkpoints cannot load on the E4 action space anyway, so no direct comparison
  was made; but any future one must account for this too.

Nothing *internal* to E4 is affected. Both arms share the architecture, so E4-01 against E4-02 —
the pooled/unpooled comparison, the round robin, the collapse analysis — is sound.

## The open decision

E4 is internally consistent and works as a fresh ladder. But it is not comparable to E3, and
`per_entity_heads` was carried by every late E3 arm rather than being an experiment, which
suggests it had been found worth having. Before spending seeds and a sweep on this architecture,
the choice is:

1. **Accept E4 as the new baseline** and record the discontinuity here; or
2. **Relaunch E4-02 with matching flags** so the ladder continues the architecture E3 settled on.

Worth deciding before the compute goes in, not after.

## The general lesson

Invariant 14 tells you to diff flags against a recent healthy run before launching. It was written
for the opponent pool and applied only to the opponent pool. **Diff the whole `metadata.json`, not
the flags you happen to be thinking about** — the ones you are not thinking about are exactly the
ones that drift.
