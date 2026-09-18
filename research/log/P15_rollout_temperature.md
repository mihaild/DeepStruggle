# Rollout temperature: sampling AT the policy beats sampling sharper than it

**Measured 2026-09-18**, arm `E3-33-30_20260917_201441` against its matched control
`E3-30-28_20260917_104226`. Distinct from
[`P15_temperature_selfplay.md`](P15_temperature_selfplay.md), which measured *evaluation*
temperature on frozen weights. This one varies the temperature the trainer **collects rollouts
at**, which changes what the policy learns from rather than how a finished policy is scored.

## What the arm varies, and nothing else

The default rollout bands are `0.15 0.50 0.10 0.35` across the four environment groups. Every one
is **below 1.0**, so the schedule *sharpens* every environment relative to the policy's own
distribution, despite the source describing it as exploration. The arm replaces them with
`0.8 1.2 0.7 1.1` — straddling 1.0, i.e. sampling at roughly the policy as trained, which is the
textbook PPO choice and had never been measured here.

The pair is genuinely matched. A field-by-field diff of the two `metadata.json` files leaves
exactly one behavioural difference, `rollout_temps`; same seed 20260930, same architecture
(`identity_dim 16`, `graph_layers 0`, `per_entity_heads 64`, `self_transform`), same
`ref_update_freq` 5M, same pool (frac 0.3, capacity 12), same reward scheme, no search in either.
`train_steps` differs (240M vs 80M) but **nothing anneals on it**: there is no LR schedule in
`generic_trainer.py` or `nash_pg.py` at all, and the one budget-scaled quantity,
`curriculum_switch_at`, sits behind `is_curriculum`, which is false for `blunder_aware`. So the
budget sets only where each run stops.

## Result: per seat against the frozen anchor `p28_280M`

100 games a side, 200 a point, identical protocol on both arms — the per-seat split rather than
the pooled number, since a pooled rate hid a collapsing seat before
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)).

| steps | control US | control USSR | control overall | **temp US** | **temp USSR** | **temp overall** |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0M | 0.0% | 1.5% | 0.8% | 1.0% | 0.5% | 0.8% |
| 20.1M | 0.5% | 1.0% | 0.8% | **12.0%** | 3.5% | **7.8%** |
| 30.0M | 4.0% | 3.5% | 3.8% | **14.5%** | 7.5% | **11.0%** |
| 40.0M | 2.0% | 7.0% | 4.5% | **21.0%** | 10.0% | **15.5%** |

Two-proportion z on the overall rate: 20.1M **z = 3.45** (p ≈ 0.0006), 30.0M **z = 2.75**
(p ≈ 0.006), 40.0M **z = 3.66** (p ≈ 0.0003). The two arms are indistinguishable at 10M and the
treatment is ahead at every point after.

## The pre-registration was wrong, and in the interesting direction

The arm's registered prediction, recorded in its own `metadata.json` before launch, was:

> higher bands trade rollout quality for coverage, so the arm is expected to look **WORSE**
> early. Judge at 40M and 80M against the matched control [...] an arm that is behind at 40M and
> level or ahead at 80M is the shape that would justify changing the default.

It is not behind at 40M. It is **ahead by a factor of three** on the pooled rate and ahead on both
seats. Whatever the sharpened bands were buying, it was not early progress, and the "trades
quality for coverage" story has no support at any measured point.

Worth stating plainly because the registration also defined what would justify changing the
default, and the observed shape is strictly stronger than the one nominated.

## The seat asymmetry runs the other way

The control is the arm whose US seat never recovers: across the full 240M its US rate sits in a
16–23% band from 120M onward and **ends at 13.5%**, while its USSR seat climbs to 56.0%. It is a
USSR specialist that never learned the other seat.

The temperature arm at 40M is **US-leaning** — 21.0% US against 10.0% USSR. It reaches the
control's *plateau* US strength (first crossed around 120–130M) at **40M**, roughly a threefold
step saving on the seat the control never fixed.

That is the result that would matter if it holds, and it is also the one most likely to be noise
of a different kind: one seed, one anchor, and an asymmetry measured at a single point on a
trajectory that is still climbing steeply. The control's own US seat wandered by 9 pp between
adjacent 10M evals (13.5% at 100M, 16.0% at 110M, 19.5% at 120M), so a single 21.0% is not a
stable property yet.

## The mechanism: sharpened rollouts train on nuclear war

The obvious hypothesis — higher bands buy a better-calibrated critic — is **wrong**, and the
metrics say so directly. At matched steps the treatment's `critic_auc` is *lower*
(0.709 vs 0.731 at 30M, 0.708 vs 0.742 at 40M) and its `critic_brier_skill` lower with it.

What actually separates the arms is how the games **end**. Self-play ending fractions at matched
steps:

| | control 30M | temp 30M | control 40M | temp 40M |
|:---|---:|---:|---:|---:|
| ends in **DEFCON 1** | **59.7%** | 28.9% | **48.9%** | 32.6% |
|  — of which own decision | 23.5% | 10.8% | 20.0% | 12.6% |
| reaches **final scoring** | 6.7% | **25.3%** | 7.8% | **23.2%** |
| mean turn reached | 6.38 | **7.71** | 6.93 | **8.08** |
| mean ply | 89.3 | **108.2** | 97.9 | **115.2** |

**Half to sixty percent of the control's training games end in nuclear war**, most before turn 7,
and only one game in fifteen reaches final scoring. The treatment halves the DEFCON-1 rate,
halves the pure-blunder share of it (`defcon1_self`), and more than triples the fraction of games
that play to the end.

The direction is counterintuitive until you write it out: sharpening concentrates mass on the
argmax, so wherever the policy's argmax is a DEFCON-lowering move, sharpening takes it *every
time*, while sampling near T=1 takes an alternative some of the time. Sharpening does not remove
blunders — it makes a single systematic one universal. That is the opposite of what a schedule
described as exploration is supposed to do.

It also supplies a candidate explanation for the control's permanently weak US seat, the arm's
most conspicuous pathology: if most games end by turn 6 in mutual destruction, the late game is
barely in the training distribution at all, and the late game is where the US seat's points come
from. The treatment reaches turn 8 on average and its US self-play win rate at 40M is 40.0%
against the control's 31.1%.

**The lower `critic_auc` is not evidence of a worse critic.** AUC is a property of the prediction
problem as much as the predictor, and the two arms are not predicting the same problem: longer,
more balanced games that run to scoring are harder to call than games decided by a turn-5
nuclear exchange. This is the `critic_base_rate` trap in another costume — a discrimination
metric compared across two different outcome distributions
([`method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). `explained_variance`,
which does not depend on class balance the same way, runs the other way: 0.904 vs 0.761 at 30M
and 0.867 vs 0.788 at 40M, favouring the treatment.

## What this does not settle

* **One seed per arm.** Seed variance on this lineage is documented and not small
  ([`seed_variance_and_pooling.md`](seed_variance_and_pooling.md)); a three-fold gap at 40M is
  well outside it, a seat asymmetry at one point is not obviously so.
* **Early lead is not a final result.** Both arms climb from cold toward an anchor with 280M
  behind it. The registered second judgment point is 80M and the arm is resuming toward it.
* **The mechanism above is a correlation, not a demonstrated cause.** Fewer DEFCON-1 endings and
  a stronger policy move together here; nothing in this arm shows the ending mix is what *drives*
  the strength rather than another symptom of it. The clean test is an arm that keeps the default
  sharpened bands but penalises or masks self-inflicted DEFCON-1 endings, so the ending mix moves
  without the temperature moving.
* **It says nothing about the collapse.** Neither arm runs search, so this is not evidence about
  the X4b failure ([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)).

## Operational note

The arm was paused at 40.04M to give the GPU to the dense collapse replay, and resumed toward
80M. The pause landed on the first registered judgment point by coincidence, which is why this
entry exists now rather than at 80M.

---

## Update, 50M (appended 2026-09-18)

The arm resumed and passed its next matched point. Same anchor, same protocol:

| steps | control US | control USSR | control overall | **temp US** | **temp USSR** | **temp overall** |
|---:|---:|---:|---:|---:|---:|---:|
| 50.0M | 5.0% | 9.5% | 7.2% | **35.0%** | 11.0% | **23.0%** |

The lead holds and widens in absolute terms (23.0% against 7.2%), and it is still carried by the
US seat: **35.0% against 5.0%, a sevenfold gap**, while the USSR seats are nearly level (11.0%
against 9.5%).

That sharpens the reading of the 40M entry above. The treatment is not uniformly better — on the
USSR seat the two arms are indistinguishable at both 40M and 50M. What the higher rollout bands
buy is **the seat the control never learns**, which is consistent with the DEFCON-1 mechanism: it
is the US seat whose points come from the late game, and the late game is what the control's
turn-6 nuclear exchanges remove from its training distribution.

Still one seed, and 80M remains the registered judgment point.

---

## The opening, 70M (appended 2026-09-18) — and a tension worth naming

`tools/scripts/setup_placement.py`, 40 games a side, against the owner's stated target: **US 4
West Germany / 3 Italy** (absent Marshall Plan), **USSR 4 East Germany / 4 Poland / 1 Yugoslavia
or Austria**.

| checkpoint | US, per game vs target | USSR, per game vs target | ref overlap US / USSR |
|:---|:---|:---|:---|
| temperature @70M | **West Germany 0.0/4**, Italy 5.6/3 | **Yugoslavia 5.1/1**, Poland 0.8/4, East Germany 0.1/4 | **33.3% / 31.7%** |
| `x4b_fixed` @28M | **West Germany 4.0/4**, Italy 2.0/3 | Poland 3.1/4, East Germany 0.0/4 | **66.4% / 51.2%** |
| `p28_280M` anchor | West Germany 3.7/4, Italy 2.0/3 | Poland 2.4/4, Yugoslavia 1.0/1 | 63.1% / 61.3% |

**The temperature arm has a bad opening and wins anyway.** It never places in West Germany, puts
5.6 of a 3-point target into Italy, and dumps 5.1 points into Yugoslavia against a target of 1 —
the exact degenerate shape flagged in the replays earlier (US ignores the contested battleground,
USSR goes all-in on Yugoslavia). Its reference overlap is roughly **half** the anchor's on both
seats. Yet at 70M it beats that anchor 44.5% to the control's 14.2%.

Two readings, and this probe does not separate them:

1. **The opening matters less than assumed** for this matchup — the arm recovers in the midgame,
   which its far longer games (mean turn 8.1 against 6.9) would support.
2. **The win rate overstates general strength**, because the arm has found something specific to
   `p28_280M` rather than become a better player. A policy that is 30 pp off the reference opening
   and still winning is exactly what an anchor-specific exploit looks like.

Reading 2 is the one that would matter, and it is testable cheaply: rate the 70M snapshot against
a *different* frozen opponent. If the margin survives an anchor it never trained against, the win
rate is measuring strength; if it collapses, the 40M–70M readings in this entry are measuring an
exploit.

**Until that is done, the recommendation to change the default rollout bands should not be acted
on.** The effect on the ending mix is real and mechanically explained; the strength claim resting
on one anchor and one seed is not yet safe.

Noted in the other direction: the pool-fixed search-CE arm `E3-34-28` has the **best US opening
measured here**, 4.0/4 into West Germany and the highest reference overlap of the three, at 28M
against the anchor's 280M — a second, independent sign that the X4b treatment was doing something
right once its pool was not starved
([`P15_X4b_collapse_is_pool_starvation.md`](P15_X4b_collapse_is_pool_starvation.md)).

---

## Cross-anchor rating, 70M (appended 2026-09-18): not an exploit

The previous section put the recommendation on hold pending a rating against opponents the arm
never trained or evaluated against. `/workspace/data/tournaments/P15_temp_cross_anchor.{md,json}`,
720 games, 60 a side, T = 0.1 — the same temperature the in-training evals use.

| rank | model | Elo | vs temp@70M |
|---:|:---|---:|---:|
| 1 | `p28_280M` | 1500.0 | 65.8% |
| 2 | **temperature @70M** | **1438.5** | — |
| 3 | `n26_280M` | 1255.6 | 19.2% |
| 4 | `p29_280M` | 1127.2 | 12.5% |

**The strength is general, not anchor-specific.** The arm beats `n26_280M` **80.0%** and
`p29_280M` **87.5%** — two 280M checkpoints from independent lineages it has never met — putting
it 183 and 311 Elo above them at **70M steps**. It is 61.5 Elo behind `p28_280M`, which this field
shows is simply the strongest of the three (72.5% and 90.8% against the other two).

So of the two readings offered above, the first survives: **a policy can be 30 pp off the
reference opening and still be genuinely strong.** The degenerate opening is real and worth fixing,
but it is not evidence that the win rate was measuring an exploit.

Its side balance across the whole field is **USSR 69.4% / US 64.4%, +5.0 pp** — near even, and a
sharp contrast with its own control lineage, which was a USSR specialist whose US seat never
recovered.

### A correction, and a note on error bars

The same checkpoint against the same opponent at the same temperature reads **33.3% here (120
games)** and **44.5% in the in-training eval (200 games)**. That is about 2 SE apart — inside the
range two honest samples can differ by, but a useful reminder that every per-seat percentage in
this document carries roughly **±4–5 pp**.

One claim made earlier in this session does not survive that: that the treatment at 70M "exceeds
the control's best over its entire 240M run" (44.5% against 38.8% at 170M). With ±4 pp on each,
and the same pair reading 33.3% on a second sample, **that exceedance is not established**. The
treatment being well ahead of its matched control at every point after 10M *is* — the gaps there
are 3x and larger, not 6 pp.

### Status of the recommendation

Off hold, with a stated scope. The bands `0.8 1.2 0.7 1.1` produce a genuinely stronger policy
than `0.15 0.50 0.10 0.35` on this lineage, at one seed, with a mechanism (the ending mix) that is
measured rather than assumed. Before the default changes, the second seed (`E3-36-31`) should
land, because 83–221 Elo of within-condition seed spread is the documented prior here and one seed
cannot rule it out.

---

## The 80M registered judgment (appended 2026-09-18)

The arm's second pre-registered point, against its matched control at the identical step count and
iteration (1221), same anchor, same protocol, 100 games a side:

| steps | control US | control USSR | control overall | **temp US** | **temp USSR** | **temp overall** |
|---:|---:|---:|---:|---:|---:|---:|
| 70.1M | 11.0% | 17.5% | 14.2% | **49.0%** | **40.0%** | **44.5%** |
| 80.0M | 13.0% | 22.5% | 17.8% | **52.0%** | **40.0%** | **46.0%** |

**46.0% against 17.8%**, two-proportion z = **6.05**, p < 1e-9. Ahead on **both** seats — US 52.0%
against 13.0%, USSR 40.0% against 22.5%.

### Against what was registered

> higher bands trade rollout quality for coverage, so the arm is expected to look WORSE early.
> [...] an arm that is behind at 40M and level or ahead at 80M is the shape that would justify
> changing the default.

The arm was never behind: ahead at 20M, 30M, 40M, 50M, 60M, 70M and 80M. The registered criterion
named a weaker shape than the one observed, so the criterion is met with room to spare — and the
stated rationale for the sharpened bands, that they buy rollout quality, has no support at any
measured point on this lineage.

### Full trajectory, both arms, per seat

| steps | control | treatment | ratio |
|---:|---:|---:|---:|
| 10.0M | 0.8% | 0.8% | 1.0x |
| 20.1M | 0.8% | 7.8% | 9.8x |
| 30.0M | 3.8% | 11.0% | 2.9x |
| 40.0M | 4.5% | 15.5% | 3.4x |
| 50.0M | 7.2% | 23.0% | 3.2x |
| 60.0M | 13.8% | 28.0% | 2.0x |
| 70.1M | 14.2% | 44.5% | 3.1x |
| 80.0M | 17.8% | 46.0% | 2.6x |

The control needed **240M** to reach a 27.8–38.8% band; the treatment is at 46.0% by **80M**.
Stated as a ratio rather than a step-saving on purpose: with ±4–5 pp on each percentage the ratios
are robust where a "3x faster" claim would not be.

### What remains before the default changes

One thing: the second seed. `E3-36-31` runs the same treatment at seed 20260931 with a
pre-registered reading, and the documented within-condition seed spread on this lineage is
83–221 Elo. Everything else that could have undermined the result has now been checked — the pair
is matched on every replayed setting, the anchor is never trained against, and the strength is
general across three lineages.

---

## Second seed, interim at 20M (appended 2026-09-18) — weaker than seed A so far

`E3-36-31`, the treatment at seed 20260931, against the same anchor and protocol:

| steps | control | **seed A** (20260930) | **seed B** (20260931) |
|---:|---:|---:|---:|
| 10.0M | 0.8% | 0.8% | 1.0% |
| 20.1M | 0.8% | **7.8%** | **2.0%** |

At 20M, seed B (US 4.0% / USSR 0.0%) sits much nearer the control than seed A did. Seed A's 20M
gap was z = 3.45; seed B's 2.0% against 0.8% is not significant at 200 games.

**This is below the registered judgment window and should not be read as a verdict.** The
pre-registration for this arm names **40M–60M**: landing at 15–28% there confirms the effect,
landing at 4.5–13.8% (the control's own trajectory) kills the recommendation. Seed A's advantage
also grew rather than appearing at once — 7.8 → 11.0 → 15.5 → 23.0 → 28.0 → 44.5 → 46.0 — so an
early lag is compatible with either outcome.

It is recorded now because it is the first evidence that bears against the headline, and the
headline is the one result of this session that would change a default. **The recommendation stays
unactioned until 40M–60M.**

---

## Second seed at the registered 40M point: a partial replication (2026-09-18)

| steps | control | **seed A** (20260930) | **seed B** (20260931) |
|---:|---:|---:|---:|
| 10.0M | 0.8% | 0.8% | 1.0% |
| 20.1M | 0.8% | **7.8%** | 2.0% |
| 30.0M | 3.8% | **11.0%** | 4.0% |
| 40.0M | 4.5% | **15.5%** | **9.0%** |

Per seat at 40M: control US 2.0 / USSR 7.0; seed A US **21.0** / USSR 10.0; seed B US 5.0 /
USSR **13.0**.

**The effect replicates in direction and not in magnitude.** Seed B is above the control at 40M
(9.0% against 4.5%) but the margin is **not significant** — two-proportion z = 1.79, p = 0.073 —
and it is significantly *below* seed A, z = 1.98, p = 0.048. Seed A ran 3.4x the control at this
point; seed B runs 2.0x.

Pooling the two treatment seeds against the single control gives 12.3% against 4.5%, z = 3.04,
p = 0.002. That is the most favourable honest summary available, and it is weaker than it looks:
there is one control seed, so the pooled test compares two treatment seeds against one control
seed and cannot separate the treatment from between-seed variation in the control.

**The seat asymmetry flipped.** Seed A at 40M was US-leaning (21.0 / 10.0); seed B is USSR-leaning
(5.0 / 13.0). So the asymmetry reported in the earlier entries is a property of a *seed*, not of
the treatment — which retroactively weakens the reading there that the bands "buy the seat the
control never learns". They did on seed A. On seed B they bought the other one.

### Against the registration

> Landing near 15-28 percent at 40M-60M confirms the effect is not a lucky seed. Landing near the
> control trajectory (4.5-13.8 percent) kills the recommendation. Anything between is a real but
> smaller effect needing the full two-arm replication before any default changes.

9.0% at 40M falls in the third case, and by the letter of the band (4.5–13.8) it also falls inside
the range nominated for killing it. The registered conclusion for this outcome is therefore the
one that was written in advance: **a real but smaller effect, needing the full two-arm replication
before any default changes.**

**The recommendation stays unactioned**, and the reason has changed from "one seed" to "two seeds
that disagree by a factor of 1.7 at the matched point". What the session can claim is the
*mechanism* — the ending mix, 59.7% DEFCON 1 against 28.9%, measured on seed A and not yet checked
on seed B, which is now the cheapest thing that would discriminate.

### The mechanism replicates; the strength gain does not

The discriminating check named above, run immediately. Self-play ending mix, seed B against the
same control:

| | control 30M | seed A 30M | seed B 30M | control 40M | seed A 40M | seed B 40M |
|:---|---:|---:|---:|---:|---:|---:|
| ends in **DEFCON 1** | 59.7% | 28.9% | **46.4%** | 48.9% | 32.6% | **29.2%** |
| — own decision | 23.5% | 10.8% | 18.8% | 20.0% | 12.6% | **6.25%** |
| reaches final scoring | 6.7% | 25.3% | 16.1% | 7.8% | 23.2% | — |
| mean turn | 6.38 | 7.71 | 7.55 | 6.93 | 8.08 | 7.61 |
| **win rate vs anchor** | 3.8% | **11.0%** | 4.0% | 4.5% | **15.5%** | **9.0%** |

**Seed B gets the mechanism in full and half the gain.** At 40M it reduces DEFCON-1 endings *more*
than seed A does (29.2% against 32.6%) and cuts the self-inflicted share to **6.25%**, less than
half seed A's 12.6% — and it converts that into 9.0% against seed A's 15.5%.

So the causal chain proposed earlier — bands → fewer nuclear endings → stronger policy — **breaks
at the second link**. The first link is solid and replicated across both seeds: the rollout bands
reliably change the ending mix, halve self-inflicted DEFCON-1, and add a full turn to the average
game. The second link is not: the same ending-mix improvement bought 3.4x the control on one seed
and 2.0x on the other, and the seed with the *better* ending mix is the weaker player.

That is a real correction to this document's own mechanism section. The ending mix is **what the
bands do**, reliably. It is not, on this evidence, **why the strong seed is strong** — something
else separates the two seeds, and nothing measured here identifies it.

---

## Seed B through the registered window: confirmed (2026-09-18)

The 40M entry above read seed B as a partial replication and said the recommendation should stay
unactioned. **The rest of the registered window reverses that.**

| steps | control | **seed A** | **seed B** | B vs control |
|---:|---:|---:|---:|---:|
| 40.0M | 4.5% | 15.5% | 9.0% | 2.0x, z = 1.79 (n.s.) |
| 50.0M | 7.2% | 23.0% | **16.5%** | 2.3x, **z = 2.88**, p = 0.004 |
| 60.0M | 13.8% | 28.0% | **22.5%** | 1.6x, **z = 2.26**, p = 0.024 |

The registration's confirm condition was:

> Landing near 15-28 percent at 40M-60M confirms the effect is not a lucky seed.

Seed B lands at **16.5%** and **22.5%** — inside that band at two of the three points, having
entered it from below rather than starting there. It is significantly above the control at both.

**So the effect replicates.** Both seeds beat the control at every point from 40M to 60M, seed A
by 2.0–3.4x and seed B by 1.6–2.3x. The earlier reading — "not significant against the control" —
was true of the 40M point alone and was reported before the window it belonged to had finished.
That was over-cautious in the same way an early stop is over-confident: a judgment taken at one
end of a registered range.

### What that does and does not change

**Changes:** the effect is no longer one seed. Two independent seeds show it, in the same
direction, within the pre-registered window, both significant by 50M.

**Does not change:** the magnitude disagrees by roughly 1.4–1.7x between seeds, so the *size* of
the gain is still not pinned; the seat asymmetry still flipped between seeds and so remains a seed
property; and the ending-mix mechanism still fails to explain the difference between the two seeds
— seed B has the better ending mix and the smaller gain. There is still **one control seed**, so
none of this separates the treatment from between-seed variation in the control.

**Recommendation:** the bands `0.8 1.2 0.7 1.1` are now supported well enough to be worth a
matched control at a second seed, which is the one arm that would settle it. Changing the shipped
default should wait for that arm rather than for more points on these two.
