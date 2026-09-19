# The observation layout programme — dead history out, card tracking in, decision context in

Records §19, §23 and §24 of the original `experiments.md`: the one line of work that produced a
measurable observation change. §19 establishes that the 512-float history slice was never written by
anything and that opponent-card knowledge is genuinely absent, and proposes both changes (§19.5
reverses §19.3 on how to encode the knowledge). §23 measures the resulting layout **v2.1** against
legacy at three budgets (80M, 240M, 320M), finds it neutral but cheaper, and adopts it. §24 measures
**v2.2** — the decision context in, unread features out — at +91.7 Elo over v2.1, confirmed on a
second seed, the first observation change in the project to clear noise; §24.1 records the
`staged_cards` flag as not demonstrated and §24.2 why width is not a version number. Arms D
(legacy), E (v2.1) and F/F2/G (v2.2) are cold starts with injection off, on the engine as it was
before the starred-card fix. **Every layout named here is now retired** — there is one layout, v2.3,
and arms D–G cannot be re-rated even in principle (see
[`corrected_engine_arms_H_I.md`](corrected_engine_arms_H_I.md) §25) — but this file is append-only
history and is not edited to match current belief.

---

## 19. The history input is not weak — it was never wired up

### 19.1 Value targets in the human warmup: yes, on 58% of it

`HumanCorpusWriter` stores `win` (the engine's ±1 verdict), `vp` (final margin / 20) and
`has_outcome`, and masks the loss where the recording stops rather than the game
(`ai/training/human_corpus_dataset.py:100-102`). On the deduplicated rebuild:

* 254 games written, **119 with an outcome** (47%);
* 132,126 samples, **76,709 with a value target** (58%).

So the value head does get human supervision, on rather less than the policy head does. That is
worth holding next to §18.1: the BC warm start's critic scores **Brier 0.378** against the actual
win, *worse than always predicting even* (0.25). It is fitted on 58% of a corpus of human games and
then asked about self-play positions, and it does not transfer.

### 19.2 The history slice is a constant zero

`ObservationBuffer` is 4293 floats: board 2352, cards 1320, global 76, **history 512**,
turn aggregates 32, active player 1. Zeroing each slice at inference and measuring how far the
policy moves, over 6,000 self-play positions:

| slice | width | constant columns | argmax changed | mean KL | mean abs Δv_win |
|---|---:|---:|---:|---:|---:|
| board | 2352 | 1833 | 44.5% | 0.307 | 0.129 |
| cards | 1320 | 1113 | 63.6% | 0.350 | 0.709 |
| global | 76 | 43 | 54.0% | 0.354 | 0.304 |
| **history** | **512** | **512** | **0.00%** | **0.0000** | **0.0000** |
| turn aggregates | 32 | 27 | 0.00% | 0.0000 | 0.0000 |

(160M final; the warm start and `dec_turns40` give the same two zero rows.)

Every one of the 512 history floats is constant across every position sampled, at every checkpoint.
The reason is not that the network ignores it. **`ActionHistoryBuffer::record()` is never called
anywhere in the repository** — `grep -rn "\.record(" engine/ bindings/` returns nothing. The ring
buffer is declared, carried inside the 4 KB `GameState`, read by `observation.cpp:258-276`, and
never written. The observation is `memset` to zero first, so the slice is a permanent zero vector.

Two further pieces of the same waste:

* Even if it were populated, the encoder writes only slots 0–8 of each 32-wide step
  (`observation.cpp:264-275`), so **368 of the 512 floats are structurally dead** regardless.
* `turn_aggregates` is written only for `coups_by_region` and `realignments_by_region`
  (`ops.cpp:322,412`), and the observation reads coups but not realignments while also reading
  `ops_spent_by_region`, `headlines_played` and `space_attempts`, none of which anything writes.
  **20 of its 32 floats are structurally dead**, and zeroing the whole slice changes no decision.

Cost, on the 160M model: the `hist_conv` branch is 69,024 parameters and it feeds 128 of the 768
fusion inputs, which is another 65,536 weights in `fusion_in`. **134,560 parameters, 4.17% of the
network, are devoted to encoding a constant zero** — and 128 of the trunk's 768 inputs are a
learned bias rather than a signal.

**So dropping the history input is not a simplification with a trade-off to weigh. It removes dead
code.** Nothing measured in §12–§18 involved it, no result is invalidated, and no ablation study is
needed to justify it because the ablation is already the identity. The one caution is that removing
it changes the observation width, so every existing checkpoint stops loading and every dataset in
the `(seed, actions)` format has to be regenerated — the §CLAUDE.md invariant-10 problem. That is a
one-off cost, not an argument against.

### 19.3 Opponent card knowledge is genuinely absent, and the proposal is sound

The observation currently folds opponent-hand cards into slot 0 — the same slot as the draw deck:

> `canon_loc = 0; // OPPONENT_HAND is hidden: fold into slot 0 (UNKNOWN/UNAVAILABLE)`
> — `observation.cpp:164-167`

So a card the opponent demonstrably holds is indistinguishable from a card in the deck. The engine
*knows* the truth in `card_locations`; the observation deliberately discards it, and there is no
"known to me" channel to put it back. The proposal therefore adds information the network has never
had, rather than re-weighting information it already has.

Every mechanism named exists in the engine already:

* `reshuffle_discard_into_draw` (`state_machine.cpp:43`), called at five sites;
* `SALT_NEGOTIATIONS` (43), `CIA_CREATED` (26), `LONE_GUNMAN` (62), `ALDRICH_AMES` (98) in
  `constants.hpp`;
* `ALDRICH_AMES_ACTIVE` is *already* a persistent-effect bit, and persistent effects are already in
  the observation (`global_features[12+b]`). So the network can currently see *that* the US hand is
  revealed and not *what* the reveal showed — which is close to the worst of both.

The deduction rule is right, and worth stating precisely because the reason matters. Knowledge is
**monotone within a card's stay in a hand**: a card cannot leave a hand secretly, so once its
location is public it stays public until it is played or discarded, at which point its location
becomes public anyway. That is what makes a single "known" bit per card sufficient and cheap — 110
floats against the 512 being removed — and it is why the state must be *tracked* rather than
recomputed: the reveal happened in the past, and nothing in the present position records it.

Three points to settle before implementing, none of them objections:

1. **The reshuffle rule needs to be stated as deck exhaustion, not as the shuffle.** The reason all
   opponent cards become known is that the draw deck has emptied, so every card is accounted for in
   a hand, the discard, removed, or in play; the opponent's hand is then the complement of what you
   can see. The shuffle is the consequence, not the cause. Implemented as "on the transition that
   empties the draw deck, mark every card in the opponent's hand known", which is the same moment
   but a rule that is true for a reason.
2. **It is a `GameState` change, and `GameState` is capped at 4 KB and must stay trivially
   copyable** (invariant 1). 110 bits is 14 bytes as a bitset — but removing the history buffer
   frees 16 `ActionToken`s from the same struct, so the net change is comfortably negative.
3. **Knowledge is per-observer.** "The USSR knows the US holds card X" and the reverse are different
   facts and need two bitsets, not one, since the observation is extracted per perspective.

### 19.4 Whether it will help is a separate question from whether it is sound

Both changes are sound. Neither is likely to be what is holding the agent back, and §18 is the
reason to say so plainly: the binding constraint found so far is that 40% of games end by DEFCON-1
at turn 6, so positional value is never collected and the critic prices it at nothing. Knowing the
opponent's hand does not lengthen games.

Where opponent-hand knowledge should help is precisely the §14–§17 cluster: holding an opponent's
card, choosing what to space, and judging whether firing an event now is safe are all decisions
whose right answer depends on what the opponent can answer with. Those are worth 3–5 points each by
§11 and §17's measurements. That makes this a reasonable thing to try *and* a poor candidate for the
next single experiment, unless run as one arm alongside something that addresses game length.

The recommendation is to take both, in this order and for these reasons: remove the history because
it is dead weight and costs nothing to remove; add the knowledge bits in the same breaking change,
since both alter the observation width and a second regeneration of every checkpoint and dataset is
the expensive part. Then measure game length, not agreement, as the primary read.

### 19.5 Separate locations beat a parallel "known" bit — and this reverses §19.3

**Moved to [`../../engine/AGENTS.md`](../../engine/AGENTS.md) §7**, which is where someone about to
touch `card_locations` will read it.

The conclusion, since §19.3 and §19.4 above propose the feature and this settles how it was
built: knowledge is encoded as extra `CardLocation` values, not as a parallel bitset. One
field suffices because the holder always knows their own hand, so "known" can only mean known
to the non-holder. It costs zero bytes in a `GameState` capped at 4 KB, and it moves the risk
from 105 write sites that must each remember to clear a bit to 52 read sites the compiler can
be made to enumerate — which is why the bare `HAND_US` / `HAND_USSR` constants were removed
rather than supplemented.

## 23. Observation layout v2.1: card tracking in, dead history out — NEUTRAL at 80M and at 240M

**Question.** §19.2 showed the history slice is a constant zero and §19.5 settled how opponent-card
knowledge should be encoded. Both changes were made together, as one breaking change to the
observation. Does the resulting layout play better, worse, or the same?

**Setup.** Two arms matched on everything except the observation layout:

| | arm D | arm E |
|:---|:---|:---|
| layout | legacy, 4,293 floats, 12 card features | **v2.1**, 3,891 floats, 13 card features |
| history branch | present (encoding a constant zero) | removed |
| opponent's known cards | folded into slot 0 with the draw deck | own slot |
| parameters | 3,230,279 | **3,095,783** |
| seed | 20260916 | 20260917 |

Both cold starts, no injection, `blunder_aware` with K=40, 512 envs, `--snapshot-every-steps
5000000`. Continued by true resume (`--resume`: optimiser moments, reference policy and step
counter restored), so each is one run carrying on rather than a warm start from its own snapshot.
Checkpoints in `data/checkpoints/arm_{D,E}_*`. Rated on **four late snapshots per arm** per §20.3,
400 games a side, `HeuristicBot` anchored at 1500.

**Result.**

| budget | E (v2.1) mean | D (legacy) mean | difference | pooled head-to-head |
|:---|---:|---:|---:|---:|
| 80M | 1769.1 (SD 19.6) | 1764.5 (SD 14.6) | **+4.5** | **50.00%** ±0.87 |
| 240M | 1867.5 (SD 6.1) | 1879.8 (SD 16.0) | **−12.2** | **49.15%** ±0.87 |

The head-to-head figure pools all 16 snapshot pairings, 12,800 games. At 80M a third arm, D′
(legacy, a different seed), sat 19 Elo from D and is the reference for what a seed draw alone is
worth; §20.3 puts between-run SD on snapshot-averaged means at about 20.

**Verdict — the layout is neutral, measured twice at three times the budget.** Both differences are
smaller than the seed reference and point in opposite directions, and both pooled head-to-heads are
within a point of even. v2.1 is adopted as the baseline on the grounds it was proposed on: it does
the same work with **134,496 fewer parameters** and 402 fewer observation floats, and it removes a
branch that §19.2 proved was encoding nothing.

**What has *not* been shown is that card tracking helps.** The layout carries information legacy
cannot express — which cards the opponent is known to hold — and strength did not move at either
budget. The plausible reason is that reveal events are rare, so the channel is mostly zero and the
signal is sparse; that is a hypothesis, not a measurement. Anything claiming a benefit from hand
knowledge needs a test that makes reveals matter, not another matched arm.

**Budget, which mattered more than the layout.** Within arm D, in the 240M pool: 160M rates 1822.2
and 240M rates 1879.8, **+57.6 Elo**, head-to-head 56.0%. The previous doubling (80M → 160M) was
worth roughly +110. Returns are compressing but have not stopped, and §22.4's warning about false
plateaus still applies. Both arms now beat `dec_turns40` — for a long stretch the strongest model
here — by 62.9% (D) and 65.8% (E).

**Caveats.** The arms differ in seed as well as layout, so each single comparison confounds the two;
what licenses the verdict is that both budgets agree and both sit inside the seed reference, not
either one alone. Four snapshots from one run are correlated, so their SD understates run-level
variance and the honest reference is §20.3's between-run figure. And 240M is one lineage per layout:
this says the layout costs nothing, not that no layout could help.

### 23.1 Three measurement faults found while running this — moved to [`measurement_bugs.md`](measurement_bugs.md)

A model fed the wrong observation width misread it silently and reported 2.0% against
`HeuristicBot` where it should have reported 81.3%; `snapshot_final.pt` is weight-identical to
the last step snapshot and would have entered one player into a tournament twice; and sorting
snapshot paths on the wrong field selected the wrong four snapshots for one arm while
selecting the right four for the other. **Every arm E evaluation logged before the first fix
is void.**

### 23.2 At 320M, still tied — and v2.1 is adopted as the baseline

Both arms continued 240M -> 320M by true resume, unchanged in every other respect.

| budget | E (v2.1) mean | D (legacy) mean | difference | pooled head-to-head |
|:---|---:|---:|---:|---:|
| 80M | 1769.1 (SD 19.6) | 1764.5 (SD 14.6) | +4.5 | 50.00% ±0.87 |
| 240M | 1867.5 (SD 6.1) | 1879.8 (SD 16.0) | −12.2 | 49.15% ±0.87 |
| **320M** | **1868.5** (SD 33.0) | **1883.2** (SD 15.4) | **−14.8** | **47.57% ±0.87** |

Head-to-head pools all 16 snapshot pairings, 12,800 games. The drift is monotone toward legacy and
at 320M is about 2.8σ below even, so within *this pair of runs* legacy is genuinely a little ahead.
It is still inside the seed envelope (§20.6: two runs of one configuration differ by ~15 Elo), and
the arms differ in seed as well as layout, so it does not attribute to the layout.

**Decision: v2.1 is the baseline.** Not because it plays better — three budgets say it does not
play differently at all — but because it is the same strength for 134,496 fewer parameters and 402
fewer observation floats, with the dead history branch gone, and because the card-tracking channel
is information the agent should have and costs **64 parameters** to carry (one card feature, 12 ->
13, across the 110-card block). `--obs-layout` now defaults to `v2.1`; the baseline checkpoint is
`data/checkpoints/arm_E_cont_240to320/snapshot_final.pt` at 320,012,288 steps.

**What is still unmeasured is card tracking itself.** v2.1 changed two things, but only one carries
information: §19.2 established the history slice was a constant zero, so removing it removed no
signal — the ablation was already the identity. What it did remove is 134,560 parameters (4.17% of
the net) that encoded that constant, which reached the trunk as an effective learned bias on top of
the biases the trunk already had: redundant capacity, not information. So the arms differ by the
card slots (+64 parameters) and by that redundancy, and the card slots have now had three budgets to
show a benefit without showing one. Isolating them needs a third arm — history removed, 12 card
features — at 2–3 seeds, which §20.6 puts at several GPU-days rather than one.

**Budget, restated with an error bar.** The within-lineage gains for an 80M leg measured +74.1,
+52.7 and +10.9 Elo in the 240M and 320M pools, which looked like compression. §20.6 shows the same
experiment repeated on a different seed spans +86.9 to +54.7, so the first two are indistinguishable
and the trend is not established. E's +10.9 sits below both replicates, which makes its flatline the
one figure here that may be real; a seed replicate of it is running.

**Caveats.** One lineage per layout, and the arms differ in seed. Elo is not comparable across
tournaments: the identical comparison (D's late four against its own 160M start) read 60.50% in one
pool and 62.25% in another. Within a tournament it is fine; across them, ±15 Elo.

---

## 24. Observation layout v2.2 — the first change to the observation that was worth anything

**Question.** §23 found v2.1 neutral against legacy at three budgets. v2.2 removes what nothing read
and adds what the network was never told. Does it move anything?

**What changed, from v2.1's 3,891 floats to v2.2's 3,825.** Removed: `turn_aggregates` (32) and
`active_player` (1) — twenty of the aggregate floats had no writer anywhere in `engine/src`, and
the decisive fact is that `ColdWarNetV2.forward` never sliced past the global block, so all 33 were
invisible to the network including the 12 that were written; and the per-country realignment
legality (168), since `can_realign` is exactly `can_coup_or_realign` and `can_coup` is that plus
one condition, so the two differ only under The Reformer. Added: the 20-float decision context —
`DecisionContext` carries twelve fields describing what is being asked and the observation used
*one* of them, so mid-play the network was told to place a point without being told which card it
was spending or how many points remained; a card feature marking the card being played; the China
Card visible as a hand card; committed headlines visible to their owner; Europe Control reported as
a win; and Chernobyl's region as a one-hot.

**Result — three cold starts, matched on everything but the layout.**

| budget | v2.2 (F) | v2.1 (E) | legacy (D) | F vs E | F vs D |
|:---|---:|---:|---:|---:|---:|
| 80M | **1817.5** | 1736.9 | 1729.3 | 61.52% ±0.84 (+80.5) | 63.55% ±0.83 (+88.1) |
| 160M | **1909.8** | 1818.2 | 1800.8 | 62.73% ±0.84 (+91.6) | 66.42% ±0.82 (+109.0) |

Head-to-heads pool all sixteen snapshot pairings, 12,800 games each. At 160M all three arms had
snapshots at identical step counts, so the four rated points are the same in every arm.

**Verdict — settled, and the first observation change in this project to clear noise.** The lead
*grew* from +80.5 to +91.6, which is not what a lucky draw does; within-arm SD fell to 8–9 Elo, the
tightest measured here; and the E-vs-D control stayed near even at both budgets (+7.6, +17.4),
consistent with §23's four independent measurements of that pair. At roughly 4.5σ against §20.6's
~20 Elo between-run SD this is not a seed effect.

**Confirmed on a second seed.** Arm F2 (20260921 against F's 20260920) reached 1830.1 against F's
1807.1 — 23.0 Elo apart, head-to-head 46.82%, inside the seed floor — while v2.1 sat at 1726.8. The
combined v2.2 mean is +91.7 Elo over v2.1, the individual seeds +80.2 and +103.2. The falsification
criterion set before the run (F2 landing near v2.1's 1736.9) did not occur. This is also the first
seed spread measured *within* v2.2, so the noise floor the verdict rests on is its own rather than
borrowed.

**What is not established is which of the changes did it.** The China Card fix, which had the
clearest mechanism, is ruled out: DEFCON-1 losses with a playable China Card in hand were 31.8% for
v2.2 against v2.1's 29.4% at matched budget — no improvement. The decision context is the natural
suspect, being the only change that alters what the network knows at *every* microstep rather than
in rare positions, but that is inference. Isolating it is three or four arms at two seeds.

**Budget.** v2.2 keeps paying at budgets where the others stopped. Arm G reached **1957.2 at 320M**,
the strongest checkpoint measured in this project, beating `dec_turns40` 84.8%, and gained +80.9 Elo
pooled from 80M to 160M — against v2.1's +4.1/+19.3 over its last leg and legacy's +52.7.

### 24.1 The `staged_cards` engine flag — NOT DEMONSTRATED

**Question.** The engine stages a card in `ctx.temp_cards` when a decision is about that card and
does not touch `card_locations`, so at the Grain Sales branch the US is asked to keep or return a
card that reads as the draw deck. Showing it to the player deciding is obviously *correct*. Is it
worth anything?

> **Superseded.** The flag is retired and does nothing. `ctx.temp_cards` no longer exists: Grain
> Sales moves the drawn card to `PEEKED_TEMP`, so it reaches the observation through the ordinary
> card-location chain and the visibility is structural rather than conditional. The question this
> arm asked can no longer be posed as an A/B, because there is no B. The result below stands as a
> record of what was measured at the time.

**Setup.** Arm G, v2.2 with the flag, seeded 20260920 to pair with arm F — the flag the only
difference between them.

| | 4 late snapshots | mean | SD |
|:---|:---|---:|---:|
| G @80M | 1727, 1726, 1770, 1776 | 1749.9 | 27.0 |
| F @80M | 1717, 1725, 1711, 1728 | 1720.2 | 7.5 |
| G @160M | 1808, 1811, 1844, 1861 | 1830.8 | 25.7 |
| F @160M | 1835, 1836, 1859, 1839 | 1842.2 | 11.2 |

**G vs F: 52.73% ±0.86 at 80M (+29.7 Elo), 48.61% ±0.87 at 160M (−11.4).** Opposite directions,
both inside the 23–25 Elo seed floor. **Not demonstrated.** The flag stays off by default; the
argument for it was correctness, not strength, and the mechanism is rare enough — three cards, and
the Grain Sales branch in a minority of games — that a measurable Elo effect was never likely.

**A measurement caution, and it is the point of this entry.** The first pass used a *single*
snapshot per budget: 800-game cells at ±24 Elo, against §20.3's 30.7 Elo mean within-run
oscillation. It reported +52.8 Elo at 80M and +4.7 at 160M, and the 80M figure was the headline
until pooling replaced it with +29.7 and flipped 160M to −11.4. G's within-run SD is 27.0 against
F's 7.5 — three times as much oscillation from one flag at the same seed, unexplained — so its
final snapshot flattered it. **A single snapshot cannot measure an effect smaller than a run's own
oscillation**, which is most effects worth arguing about.

### 24.2 Engine configuration, and why width is not a version number

Two layouts of the same width can differ in *content*, and nothing in the stack can tell them
apart: not the width assertions guarding the extraction sites, not `layout_of`, not a forward pass.
The staged-card reveal was briefly added to v2.2 after arms F and F2 had trained on it, which would
have had them evaluated against an observation they had never seen, silently.

A run now records what it trained under in its `metadata.json` as `engine_config`, set by
`--engine-flag NAME`, and **a missing entry is false** — so a checkpoint from before a flag existed
keeps the behaviour it learned without anyone having to remember. The flags travel with the
checkpoint wherever the layout does: `NeuralAgent` reads them from the metadata beside the
snapshot, the batched tournament treats `(layout, flags)` as the pair that must match and falls
back to per-state extraction when they differ, and `self_play` does the same.

An earlier attempt at this used a wider layout so the difference would be inferable from the
weights. That worked and was the wrong trade: it spent 110 floats — a per-card feature — on a
mechanism that fires for three cards.

---
