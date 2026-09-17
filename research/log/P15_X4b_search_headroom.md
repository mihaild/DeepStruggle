# How much does search still add to a policy trained with search targets?

**Measured 2026-09-17.** The expert-iteration question, asked directly of the X4b arm: if
continuous distillation works by installing the searcher's answers into the policy, then search
should have progressively *less* left to add as training proceeds, and KL(search ‖ policy) should
shrink.

For each 5M snapshot of the arm, two measurements against the **same snapshot with no search**:

* **win rate** — `search:<S>:64:determinize:all:1.0` against `<S>`, 100 games a side, **temperature
  0** so the raw policy plays its argmax and is not handicapped by sampling
* **KL and agreement** — the searcher's visit distribution against the policy, over a 60-game
  target set generated from that same snapshot at 64 simulations

Both go through the sanctioned CLIs. Raw output in
`/workspace/data/tournaments/P15_X4b_search_headroom/`.

## The arm: flat on every axis

| snapshot | search win rate | Δ Elo | top-1 agreement | KL(search ‖ policy) |
|---:|---:|---:|---:|---:|
| 5M | 57.5% | +54.1 | 97.1% | 0.0361 |
| 10M | 56.5% | +50.6 | 97.3% | 0.0343 |
| 15M | 61.0% | +77.4 | 97.3% | 0.0357 |
| 20M | 58.5% | +59.4 | 96.9% | 0.0400 |

Each win-rate row is 200 games, so **SE ≈ 3.5 pp** and the 56.5–61.0 spread is entirely consistent
with a constant ~58%. KL does not fall; the 20M value is the highest of the four. Agreement sits
at 97% throughout.

**So the policy is not progressively closing on its teacher** — but it starts much closer than
an undistilled policy does, and stays there. The control comparison below is what makes that
readable; within the arm alone the flat line is ambiguous.

## That is not, by itself, a failure

The teacher here is *the student plus search*. As the student improves the teacher improves with
it, so a constant gap is what expert iteration looks like when it is working — the student is
climbing and the ladder is moving up with it. The absolute gain is real and is measured elsewhere:
the arm beats its step-matched no-search control by +165.6 Elo at 20M
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)).

A shrinking gap would mean the policy had caught its teacher and further distillation had nothing
left to give. A flat gap means the opposite: there is still headroom at 20M, and the process has
not saturated.

## The verdict: search has about half as much to add to the arm

Both baselines landed, all at temperature 0 with the identical searcher (64 sims, determinized,
`all`), 100 games a side:

| matched steps | arm Δ Elo | **control Δ Elo** | arm KL | **control KL** | arm agree | control agree |
|---:|---:|---:|---:|---:|---:|---:|
| 5M | +54.1 | **+108.9** | 0.0361 | **0.0698** | 97.1% | 93.0% |
| 10M | +50.6 | **+116.6** | 0.0343 | **0.0565** | 97.3% | 92.6% |
| 15M | +77.4 | **+114.6** | 0.0357 | **0.0621** | 97.3% | 92.7% |
| 20M | +59.4 | **+154.4** | 0.0400 | **0.0708** | 96.9% | 91.5% |

Undistilled baseline, `p28_200M` re-measured at T=0: **+101.4 Elo**, KL **0.0649**, agreement
94.3%.

**At every matched step search has roughly half the room on the arm that it has on its control**,
with about half the KL and 97% agreement against 92%. The two are step-matched, same seed, same
recipe apart from the CE term, and rated at the same temperature — so this is the clean version of
the comparison, and it says the arm genuinely internalised its teacher.

Note the two curves move in opposite directions. The control's headroom **grows**, +108.9 to
+154.4, because the control is getting worse — it is the run that degenerates into one-sidedness
at ~65M. The arm's stays flat near +50 to +77.

### A cross-check that went the other way, and why it was worth running

The transitive estimate said this comparison was safe to make from X0's old number, because
[`P15_temperature_selfplay.md`](P15_temperature_selfplay.md) measured greedy and T=0.1 as the same
player on `p28_200M` — 50.0%, 6.9 Elo. Measured directly, X0's **+129.2 at T=0.1 reads +101.4 at
T=0**: a gap of **28 Elo**, four times the estimate.

The conclusion is unchanged and the direct baselines are what the table above uses, but the
estimate was not as good as it looked. Self-play against a copy of yourself is not the same
question as playing a *different* opponent at a different temperature, and the transitive step
quietly assumed it was.

## Caveats

* 200 games a row; ±3.5 pp is too coarse to resolve a small trend, though it is ample to show the
  absence of a large one.
* 64 simulations. A deeper searcher would have more to add at every snapshot, so these are
  statements about this teacher, not about search in general.
* The KL figures come from 60-game target sets at 64 sims — comparable to each other, and
  deliberately **not** comparable to the 96-sim census in
  [`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md), whose KL of
  0.0677 was measured with a sharper teacher.
