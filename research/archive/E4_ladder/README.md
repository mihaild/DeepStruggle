# E4 — closed plans

Unlike [`../E3_ladder/`](../E3_ladder/README.md), **the numbers here are current**. E4 is the
running programme; this bucket holds only the plans inside it that are *finished*, so the queue in
[`../../plans/`](../../plans/README.md) stays a queue.

The record for E4 is not here. It lives in [`../../log/`](../../log/README.md) — chiefly
[`P21_ladder_status.md`](../../log/P21_ladder_status.md) — and in
[`../../findings/training/`](../../findings/training/README.md).

| plan | question | answer |
|:---|:---|:---|
| [`P19_architecture_ab.md`](plans/P19_architecture_ab.md) | is the late-E3 architecture bundle actually stronger than the defaults on this engine? | **Yes, +447 Elo** — `E4-03-01` against its matched control `E4-04-01`, both cold, both pooled, 80M. Measurement and decision rule were pre-registered ([`../../log/E4_architecture_ab_result.md`](../../log/E4_architecture_ab_result.md)) |
| [`P20_positional_board_encoder.md`](plans/P20_positional_board_encoder.md) | the board reaches the trunk only through a symmetric pool, which discards which country is which. Does a positional path do better? | **Yes, and P21 answered it without citing it.** Every ladder rung from M1 up runs `aggregation="flatten"` — the positional path this plan proposed. M1 alone is **+109 Elo** over the flat MLP. Archived un-run |

## Why P20 is worth opening

It was closed by a programme that never referenced it, which is the failure mode this bucket
exists to make visible. Its *diagnosis* also outlived its proposal: the probe numbers showing the
pre-pooling token holds ~89–94% of a country's influence against the trunk's ~14%, and the
reading of why E3-13's attention read-out could not have worked (a single-query read-out is still
pooling), remain the clearest statement in the record of why pooling was the bottleneck.

Both are reasons a closed plan is archived rather than deleted — see maintenance rule 3 in
[`../../plans/README.md`](../../plans/README.md).
