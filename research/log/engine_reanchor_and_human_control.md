# E1 and E2 — re-anchoring after the engine changes, and the human corpus as a control

Records §7–§8 of the original `experiments.md`, the pair of experiments that re-established the
ladder after six `fix(engine)` commits (E1, §7) and then used the 300-game human ts-replayer corpus
as the strong-player control the agents could not supply (E2, §8). E2 answers the hypothesis §4.7
left open — that a rules bug favouring the USSR might only show in long games — and clears the
engine of it; its subsections then take the forced-win metric apart on human positions (§8.1–§8.5),
with §8.1 withdrawn twice over. Checkpoints are the pre-layout-change lineage (`dec_turns40`,
`sp2_pool_off`) on the post-E1 engine and the legacy 4,293-float observation. Entries that moved to
the measurement files or to [`P7_replayer_conversion.md`](P7_replayer_conversion.md) keep their
stubs here.
**This file is append-only history**: the withdrawn §8.1 and the reasons it was wrong are the point
of keeping it, and nothing is edited to match current belief.

---

## 7. Re-anchor after the engine changes (E1) — SETTLED

**Question.** Six `fix(engine)` commits (per-card headline decision frames, Defectors, Shuttle
Diplomacy, the scoring-card trap rule, deck refill) landed after every number in §3 and §4 was
taken. Do the checkpoint rankings survive, and are the old Elo anchors still usable?

**Setup.** `tools/tournament.py`, 4 models, 500 games per side per pair (6,000 games), RTX 4090,
`--auto-advance`. Engine verified current via `tools/scripts/check_engine_fresh.sh` (368 C++ tests
pass, fuzzer clean over 2,000 games). Both checkpoints were staged under distinct filenames first:
they are both named `snapshot_final`, which is the exact collision §6 warns about. Report:
`research/e1_reanchor_report.md`.

**Result** (Bradley-Terry, HeuristicBot anchored at 1500):

| Rank | Model | Elo | vs HeuristicBot | overall |
|:---|:---|---:|---:|---:|
| 1 | `dec_turns40` (K=40) | **1880.4** | 90.2% | 82.0% |
| 2 | `sp2_pool_off` (control) | 1836.5 | 87.2% | 76.9% |
| 3 | HeuristicBot | 1500.0 | — | 39.9% |
| 4 | RandomBot | 898.9 | 2.9% | 1.2% |

K=40 beats the control head-to-head 56.0% (1,000 games).

**Verdict.** The §4.3 ordering survives the engine changes: K=40 > control > heuristic > random,
and K=40's win rate over HeuristicBot is 90.2% against the 88.9% recorded pre-change. Old
checkpoints load and run forward passes on the current 4,293-dim observation unchanged, so they
remain valid opponents and Elo anchors. **What does not carry over is anything measured through
self-play distribution** — the §4 deficiency tables, ending mixes and battleground counts were
taken on the old engine and must be re-measured before being quoted again.

### 7.1-7.2 What a tournament number is reproducible to — moved to [`../method/running_experiments.md`](../method/running_experiments.md) (7.1) and [`variance_and_noise.md`](variance_and_noise.md) (7.2)

`--auto-advance` is outcome-neutral, verified bit-exact. A tournament is **not** reproducible
from its configuration: deals are seeded but the agents sample, so two identical 6,000-game
runs differ by about **1.5 points** on a matchup. That is the noise floor at 1,000 games a
pair, not the binomial SE — which assumes away exactly this source of variation.

## 8. The human corpus reproduces the US late-war recovery (E2) — SETTLED

**Question.** §4.7 left one hypothesis open that it could not test: a rules bug favouring the USSR
that only surfaces in long games. Random play ends at turn 2.73, before the asymmetry can express
itself, and the heuristic is too weak to separate "engine bias" from "cannot play the late war".
The 300-game human corpus is the missing control — a strong player on *this* engine.

**Setup.** `ai/eval/human_corpus.py`. All 300 corpus files: 9 are empty cached downloads (verified
by reading them, not converter failures), 9 are skipped for non-standard handicaps, **282 convert
with 0 failures**; 131 reach a terminal state and 151 are fragments whose recording stops. 144,844
decisions classified, none excluded. The VP sign convention was verified empirically rather than
assumed (`victory_points = +20` -> `get_terminal_utility = +1.0`).

**Result — mean VP by turn (US-positive), against the §4.6 arms:**

| turn | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |
| K=40 | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |
| **human** | +0.21 | -1.09 | -1.58 | -1.80 | -1.34 | -0.19 | -0.62 | -0.01 | **+2.36** |
| human N | 267 | 253 | 235 | 210 | 193 | 167 | 145 | 107 | 94 |

Human US win rate over the 131 finished games: **48.9%** (64/65/2, 95% CI ±8.6pp), and **50.0%**
in games ending turns 7-10 (N=124), against 35-40% for both agents.

**Verdict.** The corpus reproduces the standard arc in full. The USSR early lead appears at about
the agents' magnitude (trough -1.80 at turn 5), then the human curve climbs monotonically, crosses
zero by turn 9 and ends **+2.36**, while the control ends at -4.47 and K=40 at -2.18 without ever
reaching parity. **The US late-war edge is fully expressible on this engine**, so the §4.7 rules-bug
hypothesis is not supported: §4.5's side imbalance and §4.6's missing recovery are properties of our
agents, not of the simulation. Training work on the US-side weakness is unblocked.

**Caveats.** This clears the engine only of a bug big enough to erase the late-war edge, not of
smaller ones. It is not a matched comparison — skill, game length and ending mixes all differ. The
per-turn population shifts as in §4.6 but for a different reason: human attrition (267 -> 94) is
mostly the *recording* stopping rather than games ending, and whether that truncation is
outcome-neutral was not tested. The turn 1-4 bucket is empty for humans, so §4.6's short-game row
has no counterpart. And none of this says *why* the agents fail to convert the late war.

### 8.1 Humans decline forced wins twice as often as the agents — WITHDRAWN, see 8.3

| | value |
|:---|---:|
| instant-win opportunities | 149 (0.103% of decisions; agents 0.175-0.196% in §4.2) |
| games with at least one | 95 of 282 (33.7%) |
| **take rate** | **47.7%** (71 of 149, ±8.0pp) |
| declines with a settled outcome | 73 (5 more fell in fragments, excluded) |
| **cost of declining** | **20.5%** (the decliner still won 58 of 73, ±9.3pp) |

This strengthens §4.4 rather than complicating it. A corpus that is 48.9% balanced overall takes
forced wins at **47.7%**, roughly half the control's 81.2% and K=40's 72.4% — so a low take rate is
plainly compatible with strong play, and the metric cannot be an optimisation target. The
opportunity split is also 93 US / 56 USSR, the opposite tilt to the control's 92/158.

### 8.2 §4.2's critic optimism does not reproduce on human positions

`dec_turns40/snapshot_final.pt`'s value head over 84,073 decisions in finished human games:

| v_win bin | N | mean predicted | mean actual | gap |
|:---|---:|---:|---:|---:|
| [-0.75,-0.50) | 11,951 | -0.61 | -0.43 | -0.18 |
| [-0.50,-0.25) | 18,428 | -0.37 | -0.15 | -0.22 |
| [-0.25,0.00) | 15,458 | -0.13 | -0.03 | -0.10 |
| [0.00,+0.25) | 15,987 | +0.13 | +0.17 | -0.04 |
| [+0.25,+0.50) | 14,945 | +0.36 | +0.23 | **+0.13** |
| [+0.50,+0.75) | 6,431 | +0.59 | +0.54 | +0.05 |
| **all** | 84,073 | -0.06 | +0.01 | -0.07 |

Globally the critic is mildly **pessimistic** here, and its largest errors are in the negative bins
— it overstates how lost a losing-looking human position is. Optimism appears only in
[+0.25,+0.50), which is exactly the band §4.2's missed forced wins sit in, and it is smaller than
the pessimism elsewhere. So §4.2's finding is narrower than stated: not a global optimism, but a
miscalibration in one band, measured on a state distribution the agent generates itself. Off its own
distribution the sign flips.


### 8.3 The human take rate was pooling two different things — 8.1 withdrawn

**8.1's 47.7% is not a fact about human play.** Splitting the 149 opportunities by *how* the win
arrives takes the surprise out of it entirely:

| | opportunities | taken | take rate |
|:---|---:|---:|---:|
| all, as 8.1 reported | 149 | 71 | 47.7% |
| — win by VP threshold or scoring | 117 | 68 | 58.1% |
| — win by the game reaching DEFCON 1 | 32 | 3 | 9.4% |
| **excluding the last action round of turn 10** | **64** | **32** | **50.0%** |
| — win by VP threshold or scoring | 35 | 31 | **88.6%** |
| — win by the game reaching DEFCON 1 | 29 | 1 | **3.4%** |

**On ordinary wins humans are better than either agent** — 88.6% against the control's 81.2% and
K=40's 72.4%. The headline was dragged down by a second category humans essentially never take.

Two separate corrections are folded in above:

**The last action round of turn 10 is an artifact.** Final scoring fires after it no matter what,
so every action whose forced continuation ends the game in your favour is labelled a win. 46 of
78 declines sit there and **all 46 decliners won the game anyway**: they were choosing among
winning moves, not missing one. This inflates the human sample specifically, because humans reach
turn 10 far more often than our agents do (§8 records 94 games at turn 10; §4 has agents reaching
it in 11.7%). The metric is therefore not comparable across populations with different game
lengths without this exclusion.

**The DEFCON-1 category is under query — see 8.4.** Humans take 1 of 29. That is not caution; it
is what you would expect if the move is a blunder that the classifier has labelled a win.

### 8.4 The DEFCON-1 "wins" are illegal moves the engine offers — FIXED

**Moved to [`engine/AGENTS.md`](../../engine/AGENTS.md) §8**, an engine rules bug rather than a
finding about play. Ortega Elected and Che ran their own free-coup target lists without
consulting `Operations::can_coup`, so they offered coups against countries the opponent had no
influence in — and couping a battleground took DEFCON 2 → 1 and ended the game against the
phasing player.

**Why it mattered here:** 29 of the 64 non-endgame "forced wins" §8.1 measured were these
illegal coups, which is most of why §8.1 was withdrawn. And a policy trained against it learns
that an opponent's Ortega is a free win whenever a battleground sits next to Nicaragua.

### 8.5 The forced-win metric under-detects as well as over-detects — moved to [`measurement_bugs.md`](measurement_bugs.md)

The instrument follows only *forced* continuations, so a win taken by a line it cannot see is
recorded as a decline; it judges headline decisions, where neither player knows the other's
card, as if they were action rounds; and at least one labelled win does not reach 20 VP.

**Consequence for this file: §8.1 is withdrawn and not replaced.** Of its 64 non-endgame
opportunities, 29 are §8.4's illegal coups and an unknown further number are headlines or
wins taken by another line. Nothing here should be read as a statement about how humans treat
forced wins.

### 8.6-8.7.5 The engine's score against the log's — SETTLED, and the first measurement was wrong

**Moved to [`experiments_replayer_conversion.md`](P7_replayer_conversion.md).**

What bears on this file: §8.6 reported that 81.2% of converted games carried VP drift and
warned that §8's human arc — the evidence that the engine is not USSR-biased — might rest on
a corrupted score. **It does not.** That measurement was comparing the engine against a
lagging bookkeeping field rather than the log's narration. Against the narration the engine
disagrees in **six midgame entries across five games**, all explained by when the military
operations penalty is booked. **§8 stands as written.**

The remaining detail — the Shuttle Diplomacy/Japan log fault and the rule that now detects it,
why the logged score is adopted rather than the rules-correct one, and how far the
turn-ending score can be checked — is conversion mechanics.

---
