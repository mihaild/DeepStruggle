# findings/training — what a training choice is worth

Each finding is a **difference measured inside one engine**: one configuration against another, on
matched seeds and budgets. Under the assumption in
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md)
these outlive the engine revision they were measured on, where an absolute rating does not.

| file | what it settles |
|:---|:---|
| [`architecture.md`](architecture.md) | the backbone progression — identity, self-transform, per-entity heads — and what each was worth |
| [`pooling.md`](pooling.md) | the opponent pool: decisive on side balance, unresolved on strength; and the four unrelated things the word "pool" names here |
| [`seed_variance.md`](seed_variance.md) | how much of a rating is just where the run stopped — the error bars, and the two conclusions they withdraw |
| [`value_bootstrap_perspective.md`](value_bootstrap_perspective.md) | why the perspective negation in GAE is load-bearing: the assumption behind it is false (0.144 mean absolute), and replacing it breaks the telescoping and costs more than it fixes |
| [`throughput.md`](throughput.md) | what a run actually costs per step — and the two confounds (GPU model, concurrent runs) that made every earlier rate misleading |
| [`defcon_blunders.md`](defcon_blunders.md) | that the critic never sees a provoked DEFCON-1, and what windowing the credit costs |

Read [`seed_variance.md`](seed_variance.md) before believing any single-arm result on this page.
The spread between same-condition arms is the yardstick an effect has to clear, and it has already
withdrawn two conclusions that looked solid — including the map-graph depth result that every arm
from E3-17 onward is nonetheless built on.

The arms behind these are registered in [`../../runs.md`](../../runs.md); the question each answers,
with its verdict, is in [`../../questions.md`](../../questions.md).
