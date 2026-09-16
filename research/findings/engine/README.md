# findings/engine — what the simulator does

Pinned to a commit. A finding here can invalidate **absolute** numbers wholesale: if a rules defect
or a lying instrument is found, the ratings measured through it are gone rather than re-sized.

| file | what it settles |
|:---|:---|
| [`engine_revisions.md`](engine_revisions.md) | the E1 → E2 → E3 ladder, what each boundary changed, and which comparisons each one breaks |
| [`engine_change_decision_stream.md`](engine_change_decision_stream.md) | that P14 — the mask/step collapse and the Missile Envy rule — moved **no** decision: 1,068 games, 385,812 steps, four policies, zero divergences including in the legal mask |
| [`hidden_information_legality.md`](hidden_information_legality.md) | that the legal action **set** can depend on the opponent's hidden hand — The Cambridge Five names regions from cards only the opponent can see — which breaks the assumption every determinized search makes |
| [`ply_headline_order.md`](ply_headline_order.md) | that `ai.game_length.ply` numbered headlines by side while the engine resolves them in Ops order — fixed by ordering on `headline_stage`; the correction is 0.00026 SD of terminal ply, so earlier decisiveness numbers stand |

The second is the concrete evidence for the assumption in
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md),
and also its limit: it shows an engine change that moves nothing, which is the easy direction. It
says nothing about a rules fix that does move play.

Open defects live in [`../../../BUGS.md`](../../../BUGS.md), not here. A bug leaves that file when
it is fixed and has a regression test; if the fix changed what the engine does, the *consequence*
is what gets written up here.

The instrument failures — every probe that reported confident nonsense — are in
[`../../log/measurement_bugs.md`](../../log/measurement_bugs.md), with the short checklist drawn
from them in [`../../method/measurement_pitfalls.md`](../../method/measurement_pitfalls.md).
