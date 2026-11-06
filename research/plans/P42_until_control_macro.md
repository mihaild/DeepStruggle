# P42 — the until-control macro: one choice to take a country (2026-11-06)

**Status: spec for the owner's approval. Nothing is built.** It changes engine code (`Engine::step_flat`, the flat
mask), so it needs the owner's explicit go before the first line (invariant 11); the owner asked for this spec on
2026-11-06 and for the growth route from J5. It starts after J5's 800M decision, never instead of it.

## Why

A takeover needs several consistent influence placements, so exploring one from the policy's floor costs a
**product** of small probabilities. Measured on J5 (`E8-A35-R102-S44`) at 520M, greedy, the policy's own masked
softmax (`data/reports/j5_wg_probability.txt`, `continue_probe.txt`):

| | P(start West Germany) | P(continue West Germany, started this play) | a 4-point takeover per opportunity |
|---|---:|---:|---:|
| J5 520M | 1.8% | 24% | ~0.03% (0.018 × 0.24³) |
| J1 plateau | 6.1% | 65.5% | ~1.7% |

The network does learn persistence in general (P(same country again) 43.8% for J5, 55.3% for J1, against 3.2% for
starting a given country), but West Germany fell to the floor (0.3% per US setup placement, 1.5% of setups sampled
at T=1; J5 at 300M 100%, J1 at 520M 98.5%), and the USSR found Europe control through it at ~500M (6.7% of its
self-play wins at 510–520M). With a macro the plan is one draw from the floor: ~1.8% for J5, ~60× more often,
and PPO credits the whole takeover to the decision that chose it.

The planning assessment is in [P41](P41_jaxcalibur_ingredients.md) ("The until-control macro"): sound, because
within a play (and in the setup) one side's points go down with no opponent decision or chance between them, so
the macro is a deterministic expansion into the existing one-point steps -- the game is unchanged, only the
policy class and the exploration geometry.

## The view

A third action view beside E4 and E4.1 (`tools/lib/action_view.py`), **exclusive with E4.1** (the merged
influence view cost 235 Elo on A2).

* **Slots.** The flat space grows from 220 to **304**: slots 0..219 unchanged; **slot 220 + c = "place in country
  c until the acting side controls it, or until the play's points run out"**. One categorical, one mask, one
  log-prob -- the planning session's point: every consumer (rollout buffer, NashPG loss, MCTS priors, Gumbel
  logits, the rollout root, ONNX `logits`, the page) already assumes one action, so only the width changes.
* **Mask.** Slot 220 + c is legal iff the base slot `NODE_OFFSET + c` is legal at a `POINT_NODE` and the
  expansion would place **at least two** points (one point does not already give control, and the play has a
  second point to spend there). Elsewhere it is 0, so the legal set only grows where the macro means something.
  The check simulates the expansion on a copy of the state for the legal countries only (≤ ~30); its cost in the
  batch runner is measured before adoption, with an analytic count (points needed from influence, stability and
  the 2-point cost into opponent-controlled countries, re-evaluated per point) as the fallback.
* **Expansion** (`Engine::step_flat`, the E4.1 pattern): validate against the macro mask; then apply the base
  `POINT_NODE c` step repeatedly through `StateMachine::step` while the decision is still the same player's
  `POINT_NODE`, c is still legal, and the acting side does not control c; restore the state if any step fails.
  Costs, legality and control are the engine's own at every point -- nothing is computed outside it.
* **Setup.** The setup's placements are `POINT_NODE`s too: West Germany "until control" in the US setup is WG 4
  as one choice.
* **"Until control" only**, no "all remaining points" slot; no separate amount floor (the existing zero-avoiding
  term over the legal joint slots gives the macro its floor, mask and entropy); c stays 0.01.

## What it touches

* Engine/bindings (approval): the macro mask and expansion; the flat width becomes **per view** (220 is a compiled
  constant in `rollout_batch.cpp`, `batched_search.hpp`, the encoder) -- actions are not padded per view the way
  observations are.
* Python: `ActionEncoder` and the masks per view, the rollout buffer's action width, the NashPG loss (width only),
  `action_view.py` (`macro_influence` and `macro_influence_from_step` in `metadata.json`, as `merged_influence`),
  every harness building masks in the deciding agent's view (invariant 2).
* Search: `BatchedSearch`, the Gumbel root and the rollout root expand macro slots through the same engine step.
* Export and page: ONNX `logits` 304 wide in this view; the workbench names a macro slot ("until control: West
  Germany").
* Replays: a macro model's replay records the **base steps** it expanded into, plus its view, so the schema and the
  ts-replayer are untouched; human-log agreement scoring is unchanged (one base action per point).
* Behaviour cloning from human logs needs a canonical rule (a run of ≥2 same-country points reaching control =
  the macro); only a demonstration warm start would need it.

## The run: growth from J5

The owner (2026-11-06): in-place growth has worked, so grow from J5 rather than start from scratch.

* Branch at a J5 resume state chosen at the 800M decision: the policy head's dense output grows 220 → 304 with the
  84 new rows at weight 0, bias **−10**, and the per-entity country correction emits a second (zero-initialised)
  output for the macro slot; every base logit is unchanged, so the network starts as J5 (macro probability
  ~e^-10 against the base options). The zero-avoiding term then lifts the macro slots to the floor -- the
  exploration this is for.
* Identity test before launch: on positions from J5 self-play, base-slot log-probs equal J5's to float tolerance and
  the macro slots' total probability is < 1e-4.
* Name: a view was an engine sub-version (E4.1), so `E8.1-A35-R102-S44@<step>M+…` by precedent; the owner
  confirms the name and the R code at launch (`run_codes --update` / `--run`).
* Reads at +200 / +400 / +800M against J5 proper's continuation at matched steps; the 60M-style checks against J5's
  own numbers (per-update KL, entropy, head rank).
* What the macro should move, read at each step: West Germany's setup probability and share of sampled setups
  (`wg_prob.py`), the continuation probe, uncontrolled battlegrounds at turn 8, Europe-control endings by side, and
  the share of placements spent through the macro.

## Rejected: merging card and play mode

Asked by the owner and declined (2026-11-06). A play is already card, then mode, with the play-mode head reading
the chosen card's own token -- conditioned, no chance or opponent move between. Joint, (card, mode) would be one
choice among ~36 legal pairs (110 × 5 = 550 slots), but:

* the hard part is a **single** choice (the card is played anyway), so P(Our Man in Tehran, event) is the same
  product either way -- no exploration gain, unlike a multi-point takeover;
* the zero-avoiding floor scales like c/(n·Δ): today the event competes with 4 other modes, jointly with ~36 pairs,
  so the event's floor would fall;
* E4.1 merged a decision to save steps and lost 235 Elo (A2).

The event-mode holes -- Our Man in Tehran's event 0% for J5 at 520M even with a Middle East country controlled
(P 1.9%) against humans' 45% and J1's 3%; Chernobyl's headline, which J1 found only at 8–10B -- need exploration
aimed at the mode decision (a floor on the event mode where it is legal, play-mode credit, or search targets),
sized by the searched-leak census; a separate proposal.
