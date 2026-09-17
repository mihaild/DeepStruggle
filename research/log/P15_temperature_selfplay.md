# What is a sampling temperature worth? One checkpoint against itself

**Measured 2026-09-17.** Every comparison tonight had to pick a temperature, and numbers measured
at different ones are not obviously comparable — [`P15_X0_search_on_200M.md`](P15_X0_search_on_200M.md)
rated search with its raw opponent sampling at 0.1, while
[`P15_X4b_search_headroom.md`](P15_X4b_search_headroom.md) plays every raw policy greedy. I twice
declined to put those side by side on the grounds that the gap was unknown. This measures it.

`p28_200M` against itself at five temperatures, weights identical so temperature is the only
variable. 150 games a side, 300 a pair, anchored on T=0.1 — the old tournament default — so the
ratings read as gain over it. `/workspace/data/tournaments/P15_temperature_selfplay/`.

## Result

| agent | Elo | overall | head to head vs T=0.1 |
|:---|---:|---:|---:|
| `p28_200M@T0` | **1506.9** | 65.8% | **50.0%** |
| `p28_200M@T0.1` | 1500.0 | 64.8% | — |
| `p28_200M@T0.25` | 1483.7 | 62.2% | 49.3% |
| `p28_200M@T0.5` | 1366.0 | 43.9% | 27.7% |
| `p28_200M@T1` | 1126.0 | 12.9% | 12.3% |

**Greedy and T=0.1 are the same player.** 50.0% over 300 games, 6.9 Elo apart, against an SE of
about 2.9 pp. T=0.25 is still inside noise of T=0.1 (49.3%). Then it collapses: T=0.5 costs
**134 Elo**, T=1.0 costs **374**.

The reason is mechanical. Temperature divides the logits, so T=0.1 multiplies them by ten and
flattens the distinction between a sampled draw and the argmax; T=1.0 plays the policy as trained,
entropy and all, which for a policy with ~1.1 nats of entropy at every decision is a great many
deliberate mistakes per game.

## What this settles

**The X0 / X4b comparison is legitimate after all.** Search at 64 simulations beats undistilled
`p28_200M` **64.8%** with the raw side at T=0.1; it beats the X4b arm's snapshots **56.5–61.0%**
with the raw side greedy. The temperature difference between those two setups is worth about
7 Elo, not the unknown amount that made the comparison unsafe. So the ~7 pp reduction is real:
**search has measurably less to add to the distilled arm than to the checkpoint it started from.**

It also transfers to every other rating taken tonight, in the safe direction. The X4b arm sits at
policy entropy 0.56 against its control's 1.17, and a *sharper* policy is pushed closer still to
its argmax by T=0.1 — so the temperature-0 ratings in
[`P15_X4b_search_during_rl.md`](P15_X4b_search_during_rl.md) would not move materially if re-run
at the default.

## What it does not settle

* **One checkpoint.** A policy much flatter than this one would lose more to sampling at any given
  temperature, and one much sharper would lose less. The cliff between 0.25 and 0.5 is a property
  of this policy's logit scale, not a constant.
* **It does not retire `--temperature 0.0` as the right default for these comparisons.** Being
  worth only 7 Elo is a reason the old numbers are usable, not a reason to stop controlling the
  variable — especially when an arm's treatment changes its entropy, which is exactly the X4b
  case.

## Tooling

Expressing this needed per-agent temperature: `tools/tournament.py --temperature` sets one value
for the whole field and cannot play a checkpoint against itself at two settings. Added:

* `tools/lib/player_agent.py` — spec form `temp:<T>:<rest-of-spec>`, composable with the others
  (`temp:0.0:search:ckpt.pt:64:determinize:all` is valid), with the value in the agent's name so
  rows stay distinguishable.
* `tools/lib/batch_tournament.py` — an agent may pin its own temperature, overriding the
  matchup-wide one. Read with `getattr(agent, "temperature", None)`, so agents that pin nothing
  behave exactly as before.

The pin is set with `setattr` rather than declared on the `PlayerAgent` Protocol: declaring it
there makes it required of every implementer, which would oblige each bot in `bot/` to carry a
tournament-only concern.
