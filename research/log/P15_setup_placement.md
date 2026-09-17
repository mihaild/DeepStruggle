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

## The unfreezing coincides with a resume, not with a step count

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

* [`findings/engine/flattening_card_play.md`](../findings/engine/flattening_card_play.md) — the
  action-representation refactor this feeds into
* [`P15_X2_slow_anchor.md`](P15_X2_slow_anchor.md) — `E3-30-28`'s broken US seat, of which the
  frozen opening is one visible piece
