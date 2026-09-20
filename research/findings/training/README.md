# findings/training — what a training choice is worth

Each finding is a **difference measured inside one engine**: one configuration against another, on
matched seeds and budgets. Under the assumption in
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md)
these outlive the engine revision they were measured on, where an absolute rating does not.

| file | what it settles |
|:---|:---|
| [`../../archive/E3_ladder/findings/architecture.md`](../../archive/E3_ladder/findings/architecture.md) | the backbone progression — identity, self-transform, per-entity heads — and what each was worth |
| [`../../archive/E3_ladder/findings/pooling.md`](../../archive/E3_ladder/findings/pooling.md) | the opponent pool: decisive on side balance, unresolved on strength; and the four unrelated things the word "pool" names here |
| [`../../archive/E3_ladder/findings/seed_variance.md`](../../archive/E3_ladder/findings/seed_variance.md) | how much of a rating is just where the run stopped — the error bars, and the two conclusions they withdraw |
| [`../../archive/E3_ladder/findings/value_bootstrap_perspective.md`](../../archive/E3_ladder/findings/value_bootstrap_perspective.md) | why the perspective negation in GAE is load-bearing: the assumption behind it is false (0.144 mean absolute), and both replacements lose — patching the term breaks telescoping (E3-21-28), and per-player trajectories telescope correctly but add 28% advantage variance and lose **520 Elo** at 80M (E3-22-28). Also: a better offline return estimate is not a better training signal |
| [`../../archive/E3_ladder/findings/throughput.md`](../../archive/E3_ladder/findings/throughput.md) | what a run actually costs per step — and the two confounds (GPU model, concurrent runs) that made every earlier rate misleading |
| [`../../archive/E3_ladder/findings/defcon_blunders.md`](../../archive/E3_ladder/findings/defcon_blunders.md) | that the critic never sees a provoked DEFCON-1, and what windowing the credit costs |
| [`forward_pass_trace.md`](forward_pass_trace.md) | the actual forward pass: every country goes through one shared transform, and its token reaches the logits by four routes — pooled trunk, cross-attention keys, a disabled read-out, and the unpooled per-entity head. Pooling is the bottleneck (token ~89%, trunk ~14%) |
| [`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md) | only 30 of 84 countries are distinguishable by their static observation slots — Egypt, Iran and Libya are identical — so identity is a per-country bias disambiguating the other 54; ablating it moves per-country logits by more than the whole across-country spread, yet it encodes no map structure. Graph depth remains open, not settled at 0 |
| [`side_collapse.md`](side_collapse.md) | the first collapse this repository can measure rather than describe: 4 of 30 seeds, **bitwise reproducible**, a learning-arrest rather than a degradation, and **escapable** — one arm entered the pinned state for 45 rows and climbed out. `adv_std_raw` and `explained_variance` separate entry perfectly but predict nothing about escape |

Read [`../../archive/E3_ladder/findings/seed_variance.md`](../../archive/E3_ladder/findings/seed_variance.md) before believing any single-arm result on this page.
The spread between same-condition arms is the yardstick an effect has to clear, and it has already
withdrawn two conclusions that looked solid — including the map-graph depth result that every arm
from E3-17 onward is nonetheless built on.

The arms behind these are registered in [`../../runs.md`](../../runs.md); the question each answers,
with its verdict, is in [`../../questions.md`](../../questions.md).
