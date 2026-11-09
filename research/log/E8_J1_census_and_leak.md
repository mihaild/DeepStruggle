# J1's searched leak and blunder census (2026-11-09): level with R48, with few engine-proven blunders

**Question.** J1, `E8-A8-R102-S44@13520..13600M` (sha256 `243ec4b375c5…`), is P41's best SWA. Where
does it still lose under the search it would play with, and does the raw network throw games away?
These are the two censuses already run on earlier networks, repeated here.

**Answer.**

* **Under the 32-world keyed root, J1 still loses at least +58.7 ± 7.5 points of win probability a
  game.** R48's SWA loses +57.7 ± 8.1 under the same root, a difference of +1.0 ± 11.0, so the census
  does not separate them. Both are lower bounds.
* **The kinds did not move measurably either.** Influence placement is now the largest leak by a
  small margin (+23.1 ± 6.2, against +14.4 ± 6.8 on R48), ahead of action-round card choice (+20.9 ±
  3.1, against +24.9 ± 3.5). Neither difference reaches one standard error.
* **The raw network's engine-proven blunders are rare and cheap.** There are 0.57 per 1,000 decisions
  (100 of 175,004). 75 of the 100 cost nothing, none cost 0.3 or more, and together they cost 1.16
  game-points over 400 games. The E7 networks gave up 4.7 (heads soup) and 2.5 (R34) over 200 games,
  which is 9.4 and 5.0 per 400, though 64 pairs against their 32 explains part of that fall.
* **The four largest "estimated" blunders are not US mistakes in the ordinary sense.** They are the
  US not playing Lone Gunman under its own Cuban Missile Crisis, after which J1's greedy USSR couped
  and lost on the spot. So the hole [the E7 probe found](E7_cuban_missile_crisis_probe.md) is still
  there on J1. These rows are 3.62 of the class's 252.6 points.

The generated tables are [`leak.md`](E8_J1_census_and_leak/leak.md) and
[`blunder_report.md`](E8_J1_census_and_leak/blunder_report.md). The kind table and the costliest
positions are also in the
[searched-leak report](per_checkpoint/E8-A8-R102-S44_13520..13600M_rollout_k_4_worlds_32_horizon_4_rule_z2_stop_earlier_leaf_own_chance_keyed_aligned_true_box4_true.md),
which [J1's behaviour report](per_checkpoint/E8-A8-R102-S44_13520..13600M.md) now links.

**The two censuses measure different players.** The leak census judges J1 under the rollout root. The
blunder census judges the raw greedy network. Their numbers should not be added or compared.

## The searched leak

The player is `rollout:4:32:4:z2:earlier:own:keyed:aligned:box4`, the root R48's census ran under
(its
[report](per_checkpoint/E8-A8-R1-S44_10010M+R48_12720..12800M_rollout_k_4_worlds_32_horizon_4_rule_z2_stop_earlier_leaf_own_chance_keyed_aligned_true_box4_true.md)).
Each census uses 1,500 self-play games and an 8,000-position bank. R48's bank has one row fewer
(7,999). Candidates are played out at 512 pairs, and a lead of at least 1 point and 1.5 SE is
confirmed on 2,048 fresh pairs. Rows that do not escalate count as zero.

| decision kind | J1 | R48 | difference | J1's share |
|:---|---:|---:|---:|---:|
| influence placement | +23.1 ± 6.2 | +14.4 ± 6.8 | +8.7 ± 9.2 | 39% |
| action-round card choice | +20.9 ± 3.1 | +24.9 ± 3.5 | −4.0 ± 4.7 | 36% |
| play mode | +7.6 ± 2.0 | +7.9 ± 2.2 | −0.3 ± 3.0 | 13% |
| headline | +5.3 ± 0.6 | +6.8 ± 1.1 | −1.5 ± 1.25 | 9% |
| coup target | +1.7 ± 1.4 | +2.4 ± 0.9 | −0.7 ± 1.7 | 3% |
| Ops mode | +0.7 ± 0.3 | +1.4 ± 0.5 | −0.7 ± 0.6 | 1% |
| branch choice | +0.1 ± 0.1 | +0.3 ± 0.1 | −0.2 ± 0.1 | 0% |
| realignment target | −0.8 ± 0.7 | −0.3 ± 0.6 | −0.5 ± 0.9 | −1% |
| **all** | **+58.7 ± 7.5** | **+57.7 ± 8.1** | **+1.0 ± 11.0** | |

| | J1 | R48 |
|:---|---:|---:|
| positions escalated | 932 of 8,000 | 946 of 7,999 |
| share of decisions they stand for | 11.88% | 10.15% |
| decisions with a choice, a game | 435.3 | 451.5 |
| leaks the most, of 2,000 resamples | influence 60%, card 40% | card 92%, influence 8% |
| Gumbel k=8 @256 over the player, all rows | −94.1 ± 9.2 | −81.7 ± 7.5 |
| ... on the escalated rows | +43.8 ± 2.8 of +58.7 | +38.7 ± 1.9 of +57.7 |

Three things in the sub-tables:

* **By side.** US influence went from +4.5 to +12.2, and USSR card choice from +15.7 to +8.8. The ±
  in the side table (about 2 and 1) is playout noise alone. The whole kinds' bootstrap ± runs two to
  three times larger than playout noise, because it also resamples positions. Scaled that way, the
  USSR card fall is about 2 SE and the US influence rise about 1.3, and they are two picks from many
  cells. The USSR card fall is the one worth rechecking if another census is run.
* **By era.** The largest cells are T4–7 card choice +9.4, T4–7 influence +8.4, T8–10 influence +8.3,
  T8–10 card choice +6.9 and T1–3 influence +6.4. No era stands out.
* **By DEFCON.** DEFCON 2 carries the largest cells (other kinds +18.9, card choice +16.1), which is
  mostly exposure: most decisions happen there.

**What the comparison rests on.** R48's census ran on 2026-10-29, before ENG-10 changed the region
bonus (`engine/AGENTS.md` §20), so the two ran on slightly different games. The effect should be
small, but it was not measured.

## The blunder census

Raw J1 played 400 greedy self-play games with 175,004 decisions that had a choice. The cheap screen
(Gumbel 32:4 and the engine's own forced-loss and forced-win labels) flagged 51,911 rows, and 6,147
more were drawn as a control (rate 0.05). All 58,058 were validated at k=8 @256 with 64 greedy pairs.
Unlike the E7 censuses, there was no 1,024-simulation escalation, so "share unresolved" is measured
at 256 only.

| figure | J1 | R34 | heads soup |
|:---|---:|---:|---:|
| games | 400 | 200 | 200 |
| exact mistakes per 1,000 decisions | **0.57** (100 of 175,004) | 0.75 (71 of 94,158) | 0.60 (59 of 98,002) |
| game-points they cost, per 200 games | **0.58** | 2.5 | 4.7 |
| exact mistakes costing ≥ 0.3 | **0** | 1 | 7 |
| estimated class per 1,000 (upper bound) | 21.55 | 20.90 | 22.34 |
| control rows clearing z ≥ 2 (the noise floor) | 2.3% | 2.2% | 2.7% |
| candidate rows clearing z ≥ 2 | 4.2% | 4.4% | 3.9% |
| share of confirmed mistakes the cheap screen found | 43.1% | 39.5% | 20.0% |
| share rescued by 256-simulation search | 58.6% | 56.5% | 41.8% |

R34's figures are from `research/log/E7_blunder_census_R34.md` on branch `exp/blunder-census-r34`,
which is not on main.

**The fall in cost is smaller than it looks.** Each exact row is priced by its best alternative on the
same pairs that chose it. That biases the price upwards, and more at 32 pairs than at 64. Part of the
fall is therefore measurement. The fall in costly rows, from 7 to 0, is too large to be only that, but
the census cannot say how much of it is the network.

### The exact class

| | nothing (≤ 0.01) | small (0.01–0.3) | costly (≥ 0.3) |
|:---|---:|---:|---:|
| forced loss taken (44) | 32 | 12 | 0 |
| forced win missed (56) | 45 | 11 | 0 |

* The 100 rows fall in 87 of the 400 games. The largest is +0.219.
* **24 of the 100, costing 0.52 of the 1.16, are headlines,** and they carry a known label flaw. The
  engine's label at a headline reads the opponent's committed headline while it is still hidden. So
  these 24 are not established as blunders, and the 76 other rows (0.65 points) are the firmer count.
* Turn 10 has 24 rows and 0.22 points. The US's last-placement tail that dominated the heads soup's
  cost (3.66 of 4.7) has not reappeared.

### The estimated class

There are 1,569 confirmed candidate rows, summing to 252.6 regret, in 373 games. This is an upper
bound and mostly noise floor: 2.3% of control rows clear the same bar (110 of 4,809 with an
alternative), against 4.2% of candidates (1,586 of 37,769).

| regret at least | candidates | control |
|:---|---:|---:|
| 0.1 | 2.579% | 1.480% |
| 0.2 | 0.501% | 0.293% |
| 0.33 | 0.060% (31 rows, 18 games) | 0.049% (3 rows) |
| 0.5 | 0.027% (14 rows, 9 games) | 0 of 6,147 |

The control's 0 at 0.5 is weak evidence, since the candidates' rate would predict only 1.7 rows.
By kind the class is placement 98.0, card choice 95.9, play mode 31.5, headline 23.4, Ops mode 3.8.
By side it is US 120.4 and USSR 132.2. DEFCON 2 holds 177.4 of the 252.6, but it is 73.1% of the
candidates and 71.7% of the control, so that is exposure.

### The 14 largest rows, read by how the games end

The four largest rows have the same cause:

| row | position | played | best | regret |
|:---|:---|:---|:---|---:|
| `25b378b1e11b51f0` | game 303, US T5 AR3 | Quagmire | Lone Gunman | +0.961 ± 0.023 |
| `72b47d7fa4a9fa1a` | game 151, US T8 AR3 | East European Unrest | Lone Gunman | +0.906 ± 0.037 |
| `de057618cbe5e7bb` | game 151, US T8 AR5 | De Gaulle Leads France | Lone Gunman | +0.891 ± 0.039 |
| `df853aaac6146f7d` | game 303, US T5 AR7 | Asia Scoring | Lone Gunman | +0.859 ± 0.044 |

All four sit under `CMC_ACTIVE_US`. A trace from each position (one line, no redeal) shows the US
playing Lone Gunman, the USSR getting its 1-Op play, and J1's greedy USSR choosing a coup: P(coup) at
the Ops-mode choice is 0.77, 0.94, 0.96 and 0.98. Any coup target then ends the game with
`CMC_SUICIDE_LOSS`, at +20 VP with DEFCON still 2 (`Operations::execute_coup`). In each of the four
the continuation has the US win every pair.

So the price is real against this network and would not hold against a USSR that declined the coup.
That also makes these rows a USSR hole as much as a US miss. `play_safe` guards only the mover, so
the opponent's suicide stays in the score. The bare count is small: 4 rows, 2 games and 3.62 of 252.6
points. Under an active crisis, only those four rows and two others (0.18 points) are confirmed at
all.

The E7 probe measured P(coup) of 0.915 on the heads soup in this situation, and explained why plain
RL does not fix it: the coup's logit leads by 20 to 30 nats, so the gradient vanishes. These are four
positions rather than a rate, so they show the hole is still open on J1, not that it is the same size.
This census did not measure whether the rollout root declines the coup.

The other ten rows at 0.5 or more were traced for 12 pairs (seed 777), with the census's continuation
and the mover's guard. This is a sanity check, not a new measurement:

| row | position | played → best | regret | wins of 12, played → best | how the played line ends |
|:---|:---|:---|---:|---:|:---|
| `b1e6cd18` | game 251, US T7 AR2, VP +9 | Bulgaria → Yugoslavia | 0.953 | 0 → 11 | 20 VP on T7, 12 of 12 |
| `701e98b4` | game 251, US T7 AR2, VP +9 | Colombia → Yugoslavia | 0.922 | 0 → 11 | 20 VP on T7, 12 of 12 |
| `c009506c` | game 74, USSR T10 AR7 | Ops: coup → influence | 0.844 | 5 → 12 | final scoring |
| `f4b6fb23` | game 104, US T10 AR6, VP +15 | Indonesia → South Africa | 0.797 | 3 → 12 | 20 VP on T10, 9 of 12 |
| `c7cd0804` | game 44, USSR T8 AR1 | Star Wars → We Will Bury You | 0.773 | 0 → 11 | DEFCON 1 on T8, 12 of 12 |
| `9b34916b` | game 329, USSR T10 AR7 | Ops: influence → coup | 0.703 | 0 → 7 | final scoring |
| `e2ea9b20` | game 44, USSR T8 AR1 | event → space | 0.625 | 0 → 6 | DEFCON 1 on T8, 12 of 12 |
| `cf45bf30` | game 159, USSR T10 AR5 | Libya → Israel | 0.578 | 0 → 7 | final scoring |
| `e4ef475e` | game 274, USSR T10 AR6 | Angola → Israel | 0.547 | 2 → 3 | final scoring |
| `917c478b` | game 274, USSR T10 AR1 | Poland → Israel | 0.516 | 5 → 12 | final scoring |

Nine of the ten keep their direction clearly. `e4ef475e` does not show at 12 pairs, which is what a
selected row near the noise floor looks like. None involves the crisis. Six of the ten are on turn
10, and search agrees with the best move on eight of them. What these ten do not settle: the DEFCON 1
endings in game 44 were not traced further, so they could be a forced line or J1's own later mistake.

## Reading

* **J1's leak is the same size as R48's.** Whatever J1 gained over R48, the census does not show it
  as a smaller leak under the same root, and the kinds moved within noise.
* **A blunder filter still has little to fix.** This agrees with the E7 census: what is left is not a
  tail of engine-provable catastrophes.
* **Cuban Missile Crisis is the one specific hole the top of the list shows.** J1's own US passed up
  the play at all four rows, as the heads soup's did in the E7 probe. An opponent that knew about it
  could use it: the E7 script was worth about +3 points against the soup, which was not re-measured
  on J1.
* **What this does not say.** The leak total is a lower bound and sums per-decision gains, which is not
  the strength a perfect fix would add. Continuations are the network's own greedy play on both sides,
  so a move whose payoff needs a follow-up J1 does not find reads as worse than it is, and a move that
  needs J1 to err reads as better, which the crisis rows show. The blunder census is one network in
  400 greedy games.

## Replicate

**Code.** Branch `exp/j1-census` (9f5b5043, then f1f798e8), which is c71d6bc4 with the 60-minute job
limits and the action-round-9 loader fix (PR #35, now on main; no game, digest or observation change).
The screening stage ran on 9f5b5043. Validation started there, and its first four runs stopped when
one failed on that loader bug at a position in action round 9 and the other three were cancelled;
the parts they finished were kept. The rest ran on f1f798e8, which changes only the loader.

**CI** (`JamesYouL2/DeepStruggle`).

| stage | workflow | runs |
|:---|:---|:---|
| leak: annotate | `search-bank` | 42517269334, 42517697222, 42519819604, 42521155861, 42524572056, 42528100479, 42531698432 |
| leak: screen | `search-bank` | 17 runs, 42552494099 to 42600937833; prerelease `leak-j1-screen-20261109` |
| leak: confirm | `search-bank` | 42607481109, 42607651838, 42611359728, 42612052881, 42614915578; prerelease `leak-j1-confirm-20261109` |
| blunder: screen | `blunder-screen` | 42517241239 (400 games, seed 0) |
| blunder: validate | `bank-playouts` | 42527349218 (failed on the AR9 loader bug), 42528125920, 42533056877, 42538801293 (cancelled), then 42554222680, 42557034624, 42559281280, 42563571922, 42566486464, 42573536536, 42574521609, 42576555366; prereleases `blunder-input-j1-20261108`, `blunder-input-j1-rest-20261109` |

**Data.** Not published beyond those prereleases, and CI artifacts expire. It is under
`data/reports/j1_census/` in the shared data tree, with one ledger per stage (`*_runs.jsonl`).

```bash
export PYTHONPATH=.:build/release
.venv/bin/python tools/scripts/blunder_census.py report --screen screen_blunder/candidates.jsonl.gz \
  --control screen_blunder/control.jsonl.gz --summary screen_blunder/summary-part-*.json \
  --validation validate/run-*/playouts.jsonl.gz --out validate/report/report.md
.venv/bin/python tools/scripts/checkpoint_report.py \
  --players 'E8-A8-R102-S44@13520..13600M~rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed,aligned=true,box4=true)' \
  --games 0 --leak leak.md.json
```

**Edited by hand in the copied tables.** `blunder_report.md`'s header printed `?` for the pairs and
the validation seed, because it reads them from a `.meta.json` that these merged CI outputs do not
carry. The real values, 64 and 0, are filled in. Its Runtime section's local paths are shortened to be relative to
`data/reports/j1_census/`.
