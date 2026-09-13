# What the critic prices, and the two early-war event cards

Records §12–§14 of the original `experiments.md`: the diagnosis of *why* battlegrounds sit empty,
run against the 160M injected run `data/checkpoints/run_v2_20260907_long160M` (warm start / 35M /
70M / 100M snapshots; exact provenance in §12.0) on the post-E1 engine and the legacy observation.
§12 perturbs the board and finds the critic prices control and access but not partial progress
toward control; §13 perturbs the hand at six real human decisions and finds the model will not fire
its own De-Stalinization or Decolonization; §14 asks the same question of the whole corpus with a
measured human baseline, and corrects §13's reading of the trajectory as a slide when it is a dip.
§15–§18, in [`critic_vs_policy_160M.md`](critic_vs_policy_160M.md), continue directly from here and
partly overturn §13's and §14's conclusions. **This file is append-only history** — the superseded
readings stay as written and nothing is edited to match current belief.

---

## 12. Battlegrounds: the critic prices *control*, not the road to it — SETTLED

§4 records that battlegrounds sit empty in late positions and that the count plateaus from turn 8.
The tidy explanation was a self-reinforcing loop: the agent never holds battlegrounds → never sees a
scoring card pay out on them → the critic never learns they are worth anything → the policy has no
reason to take them. If that were right, the fix would be exploration.

It is not right. Measured on `ai/eval/battleground_value.py` against the 160M run
(`data/checkpoints/run_v2_20260907_long160M`, snapshots at 0 / 35M / 70M steps), turn-stratified
self-play positions, 30 per turn bucket, perturbing the board one way at a time and reading Δ`v_win`
from the mover's side.

### 12.0 Which snapshots these are

Everything below is measured on three checkpoints from one run,
`data/checkpoints/run_v2_20260907_long160M` — `--arch v2`, `--reward-scheme blunder_aware`,
`--decisiveness-turns 40`, `--num-envs 512`, `--train-steps 160000000`,
`--snapshot-every-steps 5000000`, `--inject-dataset data/datasets/human_corpus --inject-every 1
--inject-weight 1.0`, launched from a dirty tree on `5159bfe` (the dirt being the
`--snapshot-every-steps` support itself).

| name used here | file | what it is |
|---|---|---|
| **warm start** | `snapshot_0s.pt` | zero RL steps — verified tensor-for-tensor identical to `data/checkpoints/warmup_synth_then_human_train.pt`, the §9.13 synthetic-then-human BC init fitted on the 224-game train split (49.33% held-out agreement). Pure behaviour cloning; no self-play has touched it. |
| **35M steps** | `snapshot_35061760steps.pt` | 35.06M env steps of NashPG with continuous human injection |
| **70M steps** | `snapshot_70057984steps.pt` | 70.06M env steps, same |

So "warm start → 70M" is a trajectory *within one run*, not a comparison across configurations,
and the human corpus is being injected throughout — the washout in 12.3 happens *despite*
injection, not in its absence. The run was still training when these were taken; later snapshots
exist and 12.3's open question is whether the collapse is monotone across all of them.

### 12.1 Control is priced; the road to it is not

At 70M steps, Δ`v_win` for the mover, each row adding the same 2 Influence except *control*, which
adds exactly enough to flip the country:

| perturbation | T2 | T3 | T5 | T8 |
|---|---|---|---|---|
| control a battleground | +0.061 | +0.074 | +0.060 | +0.046 |
| presence in a battleground (no control) | +0.024 | +0.024 | +0.028 | +0.011 |
| access: adjacent to a battleground | +0.025 | +0.034 | +0.026 | +0.011 |
| plain influence, no battleground near | +0.020 | +0.023 | +0.019 | +0.007 |
| opponent controls a battleground | −0.073 | −0.078 | −0.096 | −0.043 |

Standard errors run 0.003–0.016, so:

* **Control is real.** Three times a plain Influence point, and symmetric — losing a battleground to
  the opponent costs about what taking one gains. The critic is not blind to battlegrounds.
* **Presence is worth nothing extra.** Two Influence into a battleground that does *not* reach
  control reads the same as two Influence into a backwater with no battleground anywhere near it:
  +0.024 against +0.020 at T2, inside one standard error at every turn.

That is the whole finding. The critic has learned the *step function* — a country is worth
something once it flips and nothing before — and a battleground almost always takes two plays to
take. Every intermediate instalment of the investment is priced at zero, so the policy sees a
two-play sequence whose first play is free money spent for nothing. This is a credit-assignment
gap, not a knowledge gap, and exploration bonuses do not touch it.

The same step function is written into the one shaping potential the repo already has:
`Scoring::compute_useful_actions_potential` (`engine/src/scoring.cpp:326-338`) builds its
battleground term from `get_country_control`, counting controlled battlegrounds and nothing else.
Switching the 160M run from `blunder_aware` to `useful_actions` would therefore reward exactly the
same last-point-only shape. (That run used `blunder_aware`, so the potential was not in play; the
step function above is what the critic learned from terminal outcomes on its own.)

### 12.2 Access *is* priced, once the comparison is honest

Contrasting "I control Thailand and the opponent is shut out of every neighbour" against "…and the
opponent has 1 next door" showed a large effect — but that contrast **deletes** the opponent's
Influence, and the critic prices raw Influence loss regardless of where it was. The matched form
*relocates* one Influence point instead: a Thailand neighbour versus a non-adjacent,
non-battleground country in the same region. Both boards carry identical totals for both players,
and adjacency is the only difference.

At 70M steps, Δ`v_win`:

| contrast | T2 | T3 | T5 | T8 |
|---|---|---|---|---|
| the opponent's access to Thailand costs me | +0.010 | +0.016 | +0.030 | +0.043 |
| my access to a Thailand the opponent holds is worth | +0.038 | +0.028 | +0.024 | +0.018 |

Positive in 8 of 8 cells here, and in 8 of 8 at 35M — and one relocated Influence point moves the
value by as much as two points placed anywhere. **The critic does model access.** The Thailand
intuition is in the network already; it does not need to be taught.

### 12.3 What RL does to the early Americas

Per-Influence-point value of a USSR foothold in each region's battlegrounds, against the
plain-backwater baseline:

| region | warm start T2 | warm start T3 | 70M T2 | 70M T3 |
|---|---|---|---|---|
| Europe | +0.055 | +0.055 | +0.013 | +0.024 |
| Asia | +0.015 | +0.057 | +0.017 | +0.018 |
| Middle East | +0.045 | +0.058 | +0.019 | +0.022 |
| Africa | −0.010 | +0.015 | +0.008 | +0.011 |
| Central America | +0.012 | +0.104 | **−0.005** | +0.003 |
| South America | +0.053 | +0.105 | +0.007 | +0.015 |
| *(plain backwater)* | +0.015 | +0.047 | +0.005 | +0.006 |

Absolute magnitudes shrink everywhere as the value head sharpens, so the ratio to the backwater
baseline is the comparison that means anything. On that basis the human-BC warm start puts South
America among its **best** early destinations at T2 (3.6× backwater) and Central America among its
best at T3 (2.2×). By 70M steps both have collapsed to the **bottom** of the table — South America
1.4× at T2, Central America *negative* — while Europe, Asia and the Middle East hold at 3–4×.

**Part of that ordering is correct.** Europe, Asia and Middle East Scoring are `WarEra::EARLY`
(`engine/src/card_data.cpp:12-14`) and are in the deck from turn 1; Central America, South America
and Africa Scoring are `WarEra::MID` (lines 37, 79, 81) and cannot be drawn before the mid-war deck
is shuffled in. A turn-2 influence point in Europe can be cashed this turn and a turn-2 point in
Brazil cannot, so the early-war regions *should* rank above the Americas at T2. The table's top
half is not the defect.

The defect is the floor. Central America at −0.005 is below a backwater with no battleground
anywhere near it — the critic prefers spending the point in a country that can never be scored for
control or presence over a Central American battleground. South America at 1.4× backwater is barely
distinguishable from the same. Those regions are still worth something at turn 2: the mid-war deck
arrives on turn 4, positions built early are cheap because they are uncontested, and influence
placed there is what makes the region contestable when the scoring card does appear. A defensible
critic ranks them below Europe and above nothing.

The warm start clears that floor comfortably — South America at 3.6× backwater at T2, Central
America at 2.2× at T3. If anything it goes further than the deck argument supports: at T2 it ranks
South America (3.6×) alongside Europe (3.7×) and above the Middle East (3.1×), which is a stronger
claim for the early Americas than "worth something" and may be the human prior overshooting. Either
way the direction of travel is the point. RL does not fail to learn that early Americas presence
matters; it learns it from the human prior and then unlearns it, past a defensible ordering and
down through the floor. This is the §9 washout showing up in the critic rather than the policy.

### 12.4 What this changes

The deadlock in §4 was framed as exploration. It is not:

1. The critic prices control (3× a plain point) and access (a relocated point moves 1–4 points of
   win rate). Both halves of the board understanding are present.
2. What is missing is any value for **partial progress toward control** — and that is precisely the
   quantity a policy needs a gradient on, because taking a battleground costs two plays.
3. The early-Americas valuation exists at the warm start and is destroyed by RL, so injection is
   holding the wrong thing in place: it slows policy washout (§10) while the critic drifts anyway.

The lever that follows is a potential-based shaping term that is **continuous in distance to
control** rather than a step at control — φ rising with each Influence point that shortens the gap.
Potential-based shaping is policy-invariant (Ng, Harada & Russell), so it cannot change the optimal
policy, only the credit path to it. That is a change to `compute_useful_actions_potential` in the
C++ engine and therefore needs approval before anything is written.

Open: whether the step function is also present in `v_vp` (only `v_win` was measured), and whether
the collapse in 12.3 is monotone across all 32 snapshots or happens at a particular point in
training.

## 13. De-Stalinization and Decolonization on turn 2 — the model will not fire its own event

§12 perturbs the board. This perturbs the *hand*, at real human decisions rather than self-play
positions, using a new `on_decision` seam on `convert_game` that hands over the `GameState` as the
human faced it. `ai/eval/card_probe.py` then edits the hand and re-asks the node of either side.

Six corpus games where the USSR played one of the two cards on turn 2 — replays 142, 101, 143
(Decolonization) and 191, 158, 16 (De-Stalinization). In all six the human fired the **event**.
Positions are taken at the USSR's first turn-2 Action Round with the card in hand; headline nodes
are excluded, because neither answer under test exists at a headline. Snapshots as in §12.0.

### 13.1 As the USSR, holding its own card

Given that it plays the card, how does the policy want to play it? Share of the `SELECT_PLAY_MODE`
distribution on **event**, the human's choice in every one of these games:

| replay | card | warm start | 35M | 70M |
|---|---|---|---|---|
| 142 | Decolonization | 0.055 | 0.176 | 0.082 |
| 101 | Decolonization | 0.058 | 0.462 | 0.268 |
| 143 | Decolonization | 0.024 | 0.482 | 0.183 |
| 191 | De-Stalinization | 0.223 | 0.062 | 0.197 |
| 158 | De-Stalinization | 0.201 | 0.014 | 0.068 |
| 16 | De-Stalinization | 0.037 | 0.126 | 0.323 |

The rest is almost entirely **ops**. Not once, at any snapshot, does the event carry the
distribution. Played greedily to the end of turn 2 the USSR spends the card for Ops in 4 of 6 games
at the warm start and 3 of 6 at 70M.

A USSR card played by the USSR for Ops does not fire its event, so this is not a trade — it is
Decolonization bought as a 2-Ops filler and De-Stalinization, a one-time card, bought as 3 Ops and
gone. Decolonization places four Influence across Africa and South-East Asia and De-Stalinization
relocates four; two Ops of Influence placement is not a substitute for either. This is the §12
finding wearing different clothes: the agent will not spend a play on board presence whose payoff is
a scoring card several turns away, and here it declines even when the card hands that presence over
for free.

### 13.2 As the US, holding the same card

Same games, same turn-2 Action Round, the card swapped into the **US** hand in place of their
highest-Ops non-scoring card, so the swap cannot be read as having handed them a weaker hand.

The right answer is to hold it past the turn; failing that, to space it. Ops is the worst available
choice, not a neutral one — an opponent's card played for Operations still owes its Event
(`engine/src/state_machine.cpp:271`), so the US pays a play, hands the USSR the full event, and
keeps only the Ops. What actually became of the card by the end of turn 2:

| replay | warm start | 35M | 70M |
|---|---|---|---|
| 142 | ops | **held** | **held** |
| 101 | ops | *space* | *space* |
| 143 | ops | **held** | *space* |
| 191 | (game ended inside turn 2) | ops | *space* |
| 158 | ops | ops | ops |
| 16 | ops | ops | ops |

Nothing acceptable at the warm start, 3 of 6 at 35M, 4 of 6 at 70M. The mode distribution moves the
same way: space is worth 0.001–0.052 at the warm start and reaches 0.83–0.89 by 70M on the replays
it gets right.

**So RL is teaching this one, and the human prior is not.** That is the opposite of §12.3, where the
prior held the early-Americas valuation and RL destroyed it. The two are consistent under one
reading: self-play punishes handing the opponent a free event within the same game, quickly and
legibly, and it does not punish a thin position in Brazil until a scoring card that may never come.
RL learns what its reward can see.

The two failures at 70M are the sharp ones. In replay 16 the US wants the card *first* — p(select)
0.938, rank 1 of 6 — and then plays it for Ops, which is the most expensive way to hold a card it
should simply not have touched. Replay 158 is the same shape. Both are turn-2 positions where the
US has better Ops available and spends the opponent's card anyway.

### 13.3 Caveats

Six games, one greedy rollout each, one die stream: this locates a behaviour, it does not measure a
rate. The §9.3 dominance suite is the pattern to follow if a rate is wanted — the same question
asked across the whole corpus with a denominator of positions where the choice was actually
available. The USSR result in 13.1 is the one worth that treatment, since it is uniform across every
snapshot rather than trending.

Also unmeasured: whether firing the event would in fact have been better here, as opposed to merely
being what the human did. §11 is the warning — a behavioural gap is not a cost until the cost is
measured, and the fork-and-play-out method in `ai/eval/dominance_cost.py` transfers to this question
directly.

## 14. The same question asked of the whole corpus, with a measured human baseline

§13 asked six hand-picked games and reported a behaviour. This asks every early-war position in the
corpus (turns 1–3, Action Rounds), and — the part §13 was missing — measures what the humans
actually did, rather than asserting it. `ai/eval/early_war_cards.py`; snapshots as in §12.0, plus
15M and 100M.

Denominators: Decolonization, 276 USSR positions holding it (270 with a mode node) and 400 US
positions with the card swapped in; De-Stalinization, 451 (445) and 400.

### 14.1 What humans do — and it is nearly absolute

Read from the raw logs, so it covers games that do not fully convert:

| | n | event | ops | space |
|---|---:|---:|---:|---:|
| USSR, Decolonization | 91 | **100.0%** | 0 | **0** |
| USSR, De-Stalinization | 121 | **97.5%** | 2.5% (3) | **0** |
| US, Decolonization | 159 | 0.6% (1) | 0.6% (1) | **98.1%** |
| US, De-Stalinization | 47 | 6.4% (3) | 4.3% (2) | **89.4%** |

**The USSR never spaces either card in the early war: 0 of 212 plays.** It plays the event in 209
of 212. The three exceptions are all De-Stalinization for Ops — replays 292 T1 AR5, 58 T1 AR3,
319 T3 AR3.

For the US both *event* and *ops* mean the USSR got the event, since an opponent's card played for
Operations still owes it. That is 7 of 206 early-war plays, 3.4%. Checked one at a time:

* **replays 273 (T3, both cards), 253 (T1), 80 (T1)** — the US held more USSR cards than it had
  space-and-hold slots to absorb. In 273 the US hand carries Socialist Governments, Decolonization
  *and* De-Stalinization; one space and one hold cannot cover three. Structurally forced.
* **replays 315 and 316 (T1 AR6)** — the same game recorded twice (see 14.4). The US never spaced
  at all that turn, so spacing was available and unused. One genuine deviation, double-counted.
* **replay 176 (T1 AR4)** — the space race was already spent on Suez Crisis at AR2, but a hold slot
  was open and went to a US card instead. A genuine deviation.

So the standard the user stated holds, with two real exceptions in 206 plays and the rest explained
by having more opponent cards than slots.

### 14.2 The model, against that baseline

Greedy mode share conditional on playing the card at that node. Human column from 14.1.

**USSR — event is the right answer (human 100% / 97.5%)**

| snapshot | Decol event / ops / space | De-Stal event / ops / space |
|---|---|---|
| warm start | 20.7% / 79.3% / 0.0% | 18.0% / 82.0% / 0.0% |
| 15M | 13.7% / 73.3% / 13.0% | 12.8% / 71.5% / 15.7% |
| 35M | **9.3%** / 87.8% / 3.0% | **5.6%** / 91.2% / 3.1% |
| 70M | 23.3% / 43.0% / 33.7% | 12.4% / 51.9% / 35.7% |
| 100M | 37.8% / 45.6% / 16.7% | 31.7% / 49.4% / 18.9% |

**US — ops is the only wrong answer (human 1.3% / 10.6% let the event fire)**

| snapshot | Decol ops / space | De-Stal ops / space |
|---|---|---|
| warm start | 93.0% / 7.0% | 89.1% / 10.9% |
| 35M | 85.2% / 14.8% | 82.6% / 17.4% |
| 100M | 48.4% / 51.6% | 40.1% / 59.9% |

Three things this settles that §13 could not:

1. **The BC warm start never learned it either.** It is behaviour cloning on a corpus where the
   USSR fires the event 100% of the time, and it reproduces that choice 20.7% of the time. The gap
   is not created by RL; RL inherits it. §13 read the trajectory as RL degrading USSR handling,
   which was reading a dip as a trend — see the trough below.
2. **The trajectory is a dip, not a slide.** USSR event share falls to 9.3% / 5.6% at 35M and then
   recovers to 37.8% / 31.7% at 100M, *above* the warm start. On the US side the movement is
   monotone and large: 93% ops down to 48%.
3. **The model uses a mode the humans never use.** The USSR spaces these cards in up to 33.7% of
   positions at 70M, against 0 of 212 human plays. Spacing is legal in 85–98% of positions, so
   this is a preference, not an artefact of what was available.

### 14.3 What it does with the event when it fires it

Both cards allow an early stop, so firing and using are separate questions. Firing is the only
problem: every snapshot spends the whole event — 4.00 of 4 placements for Decolonization (3.94 at
100M), and the full 4 removals plus 4 placements for De-Stalinization.

Where it sends them is the second failure, and it tracks §12.3 exactly:

* **Decolonization**, warm start: Angola\*, Algeria\*, Nigeria\*, Zaire\*, Thailand\* — battlegrounds
  first. At 100M the top destination is **Cameroon ×92**, then Indonesia ×62, Zaire\* ×53,
  Tunisia ×47.
* **De-Stalinization**, warm start: **into** Venezuela\* ×95, Brazil\* ×62, Chile\* ×54,
  Argentina\* ×20, **out of** Romania ×74 and Finland ×45. That is the textbook plan — shed cheap
  Eastern European filler, buy the South American battlegrounds. At 100M it moves **out of East
  Germany ×259**, stripping a battleground it controls and needs for Eastern Europe, and scatters
  into Cameroon, Lebanon, Guatemala, Colombia.

So §12.3's early-Americas collapse is visible as behaviour and not only as a value number: the warm
start uses De-Stalinization to buy South America; by 100M the same card is used to gut its own
Eastern European battleground.

### 14.4 The corpus is 11% duplicates, and 4 held-out games leak

300 files, **266 distinct games**, 34 redundant copies (11.3%); one game appears nine times.
The train/held-out split is by replay id and duplicates carry different ids, so under the
seed-7 split **4 of the 56 held-out games (7.1%) were also trained on**.

**Consequence for this file:** §9.13's 49.33% and every number derived from that split are
inflated by whatever 7.1% memorisation is worth. The split should be fingerprint-based.
Fingerprinting method and the duplicate groups are in
[`experiments_replayer_conversion.md`](P7_replayer_conversion.md).

### 14.5 Bearing on injection

This run carries a human BC warm start **and** a human batch every single RL iteration at weight
1.0 (`--inject-dataset data/datasets/human_corpus --inject-every 1 --inject-weight 1.0`). The
injector trains on the 224-game train split, so roughly four fifths of the positions surveyed above
are in the injection stream every iteration, with the human's own action as the label.

Under that pressure the USSR still plays its own event in at most 37.8% of positions. Injection at
this strength does not hold the behaviour — but 14.2's first point says injection is not the whole
story either, because the pure BC init did not have the behaviour to hold. Before more injection is
tried, the thing to establish is why supervised learning on a 100%-consistent label reproduces it
20% of the time: a decision this uniform should be the easiest thing in the corpus to fit, and if
BC cannot fit it, the loss, the sampling, or the action encoding is where to look — not the
injection schedule.
