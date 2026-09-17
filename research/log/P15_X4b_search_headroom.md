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

**So the policy is not progressively closing on its teacher.** Over 15M steps of continuous
distillation, search keeps winning by the same margin.

## That is not, by itself, a failure

The teacher here is *the student plus search*. As the student improves the teacher improves with
it, so a constant gap is what expert iteration looks like when it is working — the student is
climbing and the ladder is moving up with it. The absolute gain is real and is measured elsewhere:
the arm beats its step-matched no-search control by +165.6 Elo at 20M
([`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md)).

A shrinking gap would mean the policy had caught its teacher and further distillation had nothing
left to give. A flat gap means the opposite: there is still headroom at 20M, and the process has
not saturated.

## The comparison that would make this informative, and why it is not quoted yet

The interesting claim is that search adds **less** to the distilled arm than to an undistilled
policy — that is what "internalised" would mean. Two baselines are needed and both are running:

1. **`p28_200M` re-measured at temperature 0.** [`P15_X0_search_on_200M.md`](P15_X0_search_on_200M.md)
   rated search on these weights at **+129.2 Elo / 64.8%**, which is far above the arm's ~58%, and
   the searcher configuration is identical (64 sims, determinized, `all`). **But that measurement
   predates `tools/tournament.py --temperature`**, so its raw policy sampled at 0.1 while every
   number in the table above plays greedy. Greedy is the stronger setting, so the old figure
   flatters search by an unknown amount. Quoting the two side by side would be exactly the
   confound this project keeps having to catch.
2. **The control's snapshots.** `E3-26-28` is the same recipe with search off, so its 5M/10M/15M/20M
   checkpoints are step-matched and strictly weaker. Search should have *more* room on a weaker
   policy. If it has the same room, the arm has internalised nothing and the flat gap above means
   something less flattering.

**Running.** Nothing about internalisation is claimed until both land.

## Caveats

* 200 games a row; ±3.5 pp is too coarse to resolve a small trend, though it is ample to show the
  absence of a large one.
* 64 simulations. A deeper searcher would have more to add at every snapshot, so these are
  statements about this teacher, not about search in general.
* The KL figures come from 60-game target sets at 64 sims — comparable to each other, and
  deliberately **not** comparable to the 96-sim census in
  [`P15_X4a_where_the_search_signal_is.md`](P15_X4a_where_the_search_signal_is.md), whose KL of
  0.0677 was measured with a sharper teacher.
