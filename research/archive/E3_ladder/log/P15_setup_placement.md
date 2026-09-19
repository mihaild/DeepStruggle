# Opening influence placement: frozen for 155M steps, then unfrozen by a resume

**Measured 2026-09-17**, after the owner noticed that in five `E3-30-28` self-play games the US put
everything into the United Kingdom and the USSR everything into Yugoslavia.

Setup is the cheapest probe of a policy there is — 6 USSR placements and 9 US (7 in Western
Europe plus 2 anywhere) before a card is drawn, structurally identical every game. Tool:
`tools/scripts/setup_placement.py`, temperature 0, so what is reported is the argmax the model
actually plays.

## It is not a quirk of one checkpoint. It is the early-training default.

| checkpoint | USSR distinct | USSR placements | US distinct | US placements |
|:---|---:|:---|---:|:---|
| `E3-30-28` @10M | **1** | Yugoslavia ×180 | **1** | United Kingdom ×270 |
| `E3-30-28` @40M | 1 | identical | 1 | identical |
| `E3-30-28` @120M | 1 | identical | 1 | identical |
| `E3-30-28` @240M | **1** | identical | **1** | identical |
| `E3-20-28` @10M | 1 | Yugoslavia ×120 | 3 | Italy ×140, Iran, Panama |
| `E3-20-28` @100M | 1 | Yugoslavia | 3 | Italy ×140 |
| `E3-20-28` @150M | 1 | Yugoslavia | 3 | Italy ×140 |
| continuation @165M | **1** | Yugoslavia | 3 | Italy ×140 |
| continuation @180M | **2** | Yugoslavia 109, East Germany 11 | **5** | Italy 74, France 66 |
| continuation @200M = `p28_200M` | **6** | Poland 73, Finland 54, Romania 39 | **11** | W.Germany 98, Italy 68, Canada 25, France 22 |

Both from-scratch lineages dump every point of a side's setup into **one country**, and hold that
for a very long time. `E3-30-28`'s opening is **byte-identical from 10M to 240M** — 230M steps of
training moved it not at all.

## Pooling is the factor: every unpooled arm is more diverse than every pooled one

The 4 × 4 pooling experiment gives four seeds a side differing in one flag, all probed at 160M:

| arm | pool | USSR distinct | USSR entropy | US distinct | US entropy |
|:---|:---|---:|---:|---:|---:|
| P22 | yes | 1 | 0.000 | 4 | 1.273 |
| P27 | yes | 2 | 0.693 | 8 | 1.530 |
| P28 | yes | 1 | 0.000 | 3 | 0.684 |
| P29 | yes | 1 | 0.000 | 5 | 1.451 |
| N22 | no | **7** | 1.376 | 6 | 1.163 |
| N24 | no | **4** | 0.362 | **16** | 2.417 |
| N25 | no | **4** | 1.122 | **16** | 2.233 |
| N26 | no | **6** | 1.134 | 9 | 1.626 |

**USSR distinct countries separate completely**: pooled 1, 2, 1, 1 against unpooled 7, 4, 4, 6. No
overlap, so the exact Mann-Whitney gives U = 16 of 16 and a one-tailed **p = 1/70 ≈ 0.014**. Means
are 1.25 against 5.25 countries for USSR and 5.0 against 11.75 for US.

Note also that P29 opens in **Finland**, not Yugoslavia — so the frozen choice is not a fixed
attractor of the game, it is whatever that seed settled on early.

### And the diverse arms are the weaker ones

The pooled arms are the stronger half of that experiment — mean Elo 2061.1 against 1962.5, a
difference of +98.7 ([`../checkpoints.md`](../checkpoints.md)). So **opening diversity
anti-correlates with strength here**, and a frozen opening evidently costs little.

The likely reason it costs little is that **nothing in the measurement punishes it**: every arm in
the field opens narrowly, self-play faces the policy with its own habits, and the pool faces it
with its own past snapshots, which share them. A bad opening is only exposed by an opponent that
opens differently — a human, or a lineage trained another way. That makes this a blind spot of the
whole evaluation setup rather than a property of one arm.

It also cuts against reading diversity as a health metric. The right reading is narrower: a policy
that plays one opening has stopped searching that part of the space, which matters for what it
*could* learn, not necessarily for what it currently scores.

## The unfreezing coincides with a resume — but not with a pool reset

**Correction, same day.** This section first attributed the change to the resume emptying the
opponent pool. **It did not.** `E3-20-28`'s continuation records
`base_commit = c9062336`, which *is* the commit "fix(training): rebuild the self-growing opponent
pool when resuming" — so that resume rebuilt the pool from the run's own snapshots. The pool was
not reset, and the mechanism proposed here was wrong.

What a resume did still change at that time: the rebuild samples snapshots **evenly over the
run's history**, which lands on a different pool than the live one (eviction is by spacing and
depends on the order members arrived), and it **zeroed every opponent's win/game record**. So the
opponent distribution was perturbed, just not emptied. Whether that is enough to unfreeze a
decision is unmeasured.

Both of those are now gone as accidental effects: the resume state carries the pool's exact
membership and statistics, and discarding it requires `--reset-opponent-pool`. A future resume is
therefore a cleaner control, and re-running this comparison against one would say whether the
perturbation mattered at all.

## The timing, which stands regardless

`E3-20-28`'s USSR opening is unchanged from 10M through **165M** and has moved by **180M**. The
continuation leg resumed from the 160M state, so the change lands 5–20M steps after a resume,
after 155M steps of nothing.

What a resume changes is narrow and documented: `load_resume_state` restores the model, the
optimiser moments, π_ref, the step counter and the RNG — and the comment beside it says
**"not the pool"**. So the opponent pool starts empty and refills from scratch.

`E3-30-28` ran its whole 240M in a single leg with **no resume**, and never unfroze. That is one
pair of lineages and a coincidence of timing, not a demonstrated cause — but it is a sharp enough
coincidence to name, and it is cheap to test: resume `E3-30-28` from its own 240M state and probe
the opening again 20M later.

## Why setup is the decision most likely to freeze

* **It is rare.** 15 of ~440 decisions a game, about 3.4%, and exactly once per game rather than
  spread through it.
* **Its reward is maximally distant.** Every other decision is closer to the outcome.
* **Nothing corrects it.** A policy that opens badly still plays 400 more decisions, and the
  blunder probes do not look at setup.
* **The KL anchor holds it in place.** η·KL pulls toward π_ref at every node including the ones the
  policy has stopped exploring, so a decision with no gradient pressure to move has active
  pressure to stay. That is the mechanism a pool reset would disturb — new opponents change the
  state distribution the anchor is measured over.

## Caveats

* **Temperature 0**, so this is the argmax. The underlying distribution may be broader; what is
  established is what the model *plays*, which is what the ratings measured.
* Two lineages, one probe per checkpoint at 20–30 games. The placements are deterministic enough
  that more games add nothing, but more *lineages* would.
* The resume coincidence is a hypothesis with one instance on each side.

## See also

* [`../../../findings/engine/flattening_card_play.md`](../../../findings/engine/flattening_card_play.md) — the
  action-representation refactor this feeds into
* [`P15_X2_slow_anchor.md`](P15_X2_slow_anchor.md) — `E3-30-28`'s broken US seat, of which the
  frozen opening is one visible piece
