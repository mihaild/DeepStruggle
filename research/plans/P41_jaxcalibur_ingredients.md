# P41 — Jaxcalibur's ingredients on the E8 base, one at a time, with training against search repeated on top (2026-10-30)

**Why.** [Jaxcalibur](https://jaxcalibur.github.io/) matched top human players at Gen 9 random-battle
Pokémon with our training algorithm -- pure PPO self-play with GAE, undiscounted win/loss reward, no
search in training, search only at play time (+100–150 Elo, "most of the strength comes from the
network") -- at about four times our compute per run (one H100 for a week, ~100M games) and four
times our parameters (8.5M). It has four ingredients we do not: a zero-avoiding regulariser beside
the entropy bonus, two auxiliary heads about the opponent (its next action, which "made the bot
substantially stronger", and its hidden attributes carried across turns), a token transformer with
position-type-specific MLPs, and the last 16 game events as tokens. Each maps onto a measured gap
here: policy saturation (19–33 nats, the gap penalty's crude target), a trunk that holds only public
information about the opponent's hand (P33f), a residual trunk that collapses where a transformer
over entities does not, and a memoryless policy. This plan tests them one at a time on the best E8
recipe, then repeats the one result of P40 that moved a plateau -- training against a searched
opponent -- on top of whatever they establish.

**Where P40 stands (2026-10-30), which sets the base.** Arm C (the deep network warm-started by
distillation) held its start and did not gain past it; stopped. Arm S (the rollout root as the
opponent in one game in eight, k4 w4 h1 at entry decisions, graphed, hold-and-batch) reads 51.6 /
53.4 / 54.3% at 16,400 / 16,800 / 17,200M from R94's plateau -- the one trend past the band -- and runs
alone. The recipe found on the way is the new from-scratch base: **A8 on R95** (16k minibatch, PPO
clip 0.1, µP's zeroed readouts and per-layer rates at A8's widths) reads **35.8% at 2,000M** against
A13+R48's SWA where the plain control read 27.7%; its control run `E8-A8-R95-S44` continues to its
plateau (10,000M budget) in the next free slot.

## The arms (from scratch on A8-R95, one change each, R codes at launch)

| arm | change | what it maps to here |
|:---|:---|:---|
| **J1** | **the zero-avoiding term**: KL(uniform ‖ π) over the legal moves, i.e. −mean_a log π(a), coefficient c, from step 0, beside the entropy bonus | saturation; the gap penalty (+7 late, −150–240 from scratch) is its crude form. Cells: c small (the entropy coefficient's scale), then 3×. Reads: median top-two logit gap and P(top) (the P37a saturation numbers), the rare-spot card rates (Nuclear Subs, One Small Step, Wargames declined), the Cuba trap suicide rate; strength at matched steps |
| **J1′** | **the zero-avoiding term replacing the entropy bonus** (owner, 2026-10-30): `--entropy-coef 0 --zero-avoid-coef 0.01`. Queued behind J1, which keeps running. KL(uniform ‖ π) is entropy's KL reversed: the two agree to second order near uniform, so J1 beside the bonus pulls about twice as hard toward uniform early, while at saturation entropy's gradient fades (∝ π_a) and this term's does not (1/n on a starved move) | the cleaner one-factor form of J1: the control's flattening pull near uniform, plus the action at saturation. Same reads as J1 |
| **J2** | **the opponent's next-action head**: at the learner's decision, predict the opponent's next entry decision (its next card play, the flat action) from the observation; cross-entropy at weight w; the target is the rollout's own record | the auxiliary that "made the bot substantially stronger"; here an opponent model the trunk lacks (P33f: public information only) |
| **J3** | **the hidden-hand head**: predict the opponent's hand (sigmoids over the hidden pool) from the observation; the true hand from the engine state as the target (a recipe: the policy never reads it); weight w | their item/ability/move predictions as targets; ours is the belief the trunk does not hold. Read on its own target against the probes' numbers (prior 0.3914, public probe 0.3780, the policy-likelihood posterior 0.3363): does the trunk learn a belief when asked to |
| **J4** | J2 + J3 together | their setting; only if each is at least level alone |
| **J5** | **superseded by the design below (owner, 2026-10-31)**; as first written, the token encoder on the R95 recipe: two attention layers over country and card tokens (the `--ladder-token-layers` path, pre-norm) on A8's heads, with the per-entity-type MLPs the ladder already has (their position-type mixture of experts), µP per-layer rates, the stability checks at 60M | P38 rung 4 with its prior raised: a transformer over entities trained under PPO self-play is their whole architecture, and it did not need hyperspherical machinery |
| **J6** | **history tokens** (the owner's decision): the last 16 decisions of the game -- side, decision type, card or country -- as an appended `obs_features` block of integer codes (16 × 3 floats), embedded network-side; into J5's network, not A8's | their 16 event tokens; the cheapest memory, 48 floats |
| **J7** | **matchup features** (the owner's decision, lower priority): per held card, the engine's event and Ops summary against the board (P30's T2/T4: VP, DEFCON and regional deltas if the event fires, countries the Ops could control), ≤ 9 cards × ~10 floats as an appended block | their matchup tokens: pairwise structure handed over rather than learned |
| **J8** | **training against search, repeated on top**: Arm S's searched opponent (`--searched-opponent-spec 4:4:1:z2:earlier:own:keyed:aligned:box4:graphs`, one game in eight) (a) as a branch from the new base's plateau -- Arm S's form, replicated; (b) at the same share from the start of the best combination, gated as R69 was (search joins when the root beats its raw snapshot ≥ 55% in a 200-game check every 40M) | the one P40 result that moved a plateau; it must be shown to hold on the new base, and whether it helps from the start is untested |
| **J9** | **length**: the best combination run to twice the steps its control plateaued at if still rising | their ~100M games are ~45,000M of our steps; our plateaus are at 12,000M |

## J5's design: the token transformer as the main path (owner, 2026-10-31: option B; built, not launched)

The row above (A33, attention beside the grouped trunk) is not the run: it costs a transformer's compute and tests only
"attention beside the trunk", and its relatives were level (E7-09-44 by 400M, A18 grown onto a plateau). What Jaxcalibur
publishes: a non-causal transformer, pre-RMSNorm, different MLP weights per position type, 60 tokens, 8.5M parameters;
**no depth, width, heads or optimiser settings**, so those are ours to choose.

**The network (a new A code; no observation, action-space or engine change).**

* **Tokens:** 195, as the token path has them: 1 global, 84 countries, 110 cards, each type's own projection of its raw
  row plus a learned identity per country and card (`tok_*_in`, `tok_*_id`).
* **Blocks:** L pre-RMSNorm blocks: full self-attention shared by all tokens, then an **MLP per token type** (global,
  country, card), only the token's own running: parameters triple, compute does not.
* **Main path:** the grouped projections go. The trunk `h` is `fusion_in` over the global token's output and the mean of
  the country and of the card tokens (owner, 2026-10-31), so everything reaches the trunk through attention.
* **Heads:** base policy and value from `h`; per-country and per-card heads each from their own token, raw row and the
  trunk context; branch and play-mode heads also read **the active card's token** (owner, 2026-10-31). The 220-wide flat
  action space is untouched.
* **Readouts zeroed** (R95). `--mup` scales nothing at these widths unless a base is defined for the token path: left off the build.
* **Flags (built, `ai/models/token_transformer.py`):** `--ladder-token-main` with `--ladder-entity-proj-dim 0`,
  `--ladder-token-layers`, `--ladder-token-dim`, `--ladder-token-heads`, `--ladder-token-mlp-mult`; recovered from a state
  dict; the ONNX export is checked against torch; a 3 x 192 smoke run through the CLI ran at ~1.6k steps/s beside two trainers.
* **The setup stays a separate problem:** the network sees the phase, the placements left and the board, so "Czechoslovakia
  while the counter is high" remains the easy function; tokens do not add exploration or credit.

**Size: cost per row, measured** (a prototype, bf16, batch 1,024 beside two trainers, so ±~30%: its 2x128
read slower than its 3x128). One forward plus four updates per row; A8's 9.9 µs matches its ~107k steps/s alone, so
steps/s below is 10^6 / cost.

| cell | parameters | µs per row | × A33 | ~steps/s alone | 800M | 2,000M |
|:---|:---|:---|:---|:---|:---|:---|
| A8 | 1.48M | 9.9 | 0.07 | ~100k | -- | -- |
| A33 (2 × 128, shared MLP) | 1.86M | 137 | 1.0 | ~7k (measured 4-10k) | 1.3 d | 3 d |
| 3 × 128, per-type MLP ×2 | 1.06M | 295 | 2.1 | ~3.4k | 2.7 d | 7 d |
| **3 × 192, per-type MLP ×2** | **2.11M** | **456** | **3.3** | **~2.2k** | **4 d** | **10.5 d** |
| 3 × 256, per-type MLP ×2 | 3.55M | 595 | 4.3 | ~1.7k | 5.5 d | 14 d |
| 4 × 256, per-type MLP ×4 (Jaxcalibur's size) | 7.75M | 829 | 6.0 | ~1.2k | 8 d | 19 d |

**Size is decided on a free GPU (owner, 2026-10-31):** every cell is measured with nothing beside it, then chosen; L is
the attention depth, 2-4 in the benchmark, 3 the leaning. The GPU is the limit: 9.9
GB per 1,024 rows at 4 × 256 means passes of ~1,024 rows at the 16k minibatch (`--ladder-token-chunk`), and a neighbour
slows it badly (A18: 4.7k beside R57, which fell from 69k to 10-20k). It runs with the GPU to itself or one light neighbour.

**Recipe:** R95, plus J1's term if J1 holds at 4,000M. Its step size is unknown: lr 3e-4 first, 1e-4 if the 60M check fails.

**Reads and rules (earlier than the plan's 2,000M, at this cost).** 60M: per-update KL, head-input rank >= 2x fresh, heads not
dormant; lr once, then stop. Reads at 200, 400 and 800M against A13+R48's SWA beside the control at matched steps. **At 800M
(~4 days): continue only if at least level with the control then in the base recipe** (A8-R95 18.9%, J1 30.4%), the gap
closing; then to 2,000M for P41's pass rule. E7-09-44's lead was gone by 400M, so an early lead alone decides nothing.

**After it:** J6's history tokens become 16 more tokens in this encoder (the owner's block); J1 + `--setup-mc-credit` for
discovering openings on whichever network this leaves as the base.

## Measure

* **Matched steps against the control** `E8-A8-R95-S44` at 400M windows (80M-window SWAs, raw, 1,000
  deals a side), per seat against the control's own self-play bar; the fixed reference A13+R48's SWA
  beside it; **the own-past gain tracked** (the drift read of P40) so a lineage-specific gain is seen
  for what it is.
* Each arm's own target where it has one: J1's saturation and rare-card rates; J2's next-action
  accuracy on held-out games; J3's hand log loss against the probes.
* The stability note's checks at 60M and the plasticity probe at 400M for every arm; the searched
  reads (under the corrected keyed root) at the plateau only.
* `P40a`, the Czechoslovakia opening priced under the network's own continuation, the searched root
  and the oracle, is read on the J2/J3 and J8 networks: the owner's example is the test of whether an
  opponent model or a searched opponent makes a setup fault cost.

## Decision rules

* An arm **passes** by the per-seat rule against the control at matched steps at 2,000M and holds it
  at 4,000M; a pass enters the base recipe for the arms after it (J1 → J5 → J6 → J2/J3 → J4).
* J1 passes only with the saturation numbers moved (median gap down, rare-card rates up) **and**
  strength not lower; a regulariser that moves the numbers and costs strength is tuned (c ÷ 3) once.
* J3 is read first on its own target: a head that does not beat the public probe (0.378) has not
  taught the trunk a belief, and J4 is skipped.
* J5 passes P38's rule (the gap to its control decreasing at 2,000M with rank ≥ 2× fresh at 60M);
  J6 and J7 go in only on J5's passing network and only with the owner's approval of the block.
* J8a passes if it reproduces Arm S's trend (above the base's plateau band on two consecutive
  400M reads); J8b if it is ahead of J8a's base at matched steps with the own-past gain not
  outrunning the reference gain.
* The from-scratch adoption run is the best passing combination, to its own plateau, J9's length.

## Sequencing (two slots; Arm S keeps one until it reads)

**Order (owner, 2026-10-30): J1 → J5 → J6 → J2 / J3 → J4 → J7 → J8 → J9.** Architecture comes
right after the regulariser because much is likely to depend on it; and **the opponent heads depend
on history**: without a history input the hand head can learn only the public-information prior
(the belief probes: public 0.3780 against 0.3363 for the posterior from this turn's opponent
decisions), and the next-action head only a style prior, so J2/J3 are not run before J6. Jaxcalibur
has both the 16 event tokens and predictions carried across turns; the heads are read on their own
targets only once the network can see the history they need.

| slot 1 | slot 2 |
|:---|:---|
| Arm S to its decision (P40), then the `E8-A8-R95-S44` continuation to its plateau | J1 (2,000M, ~a day beside another arm), then **J5** |
| J8a from the base's plateau once the control has one | **J6** on J5's passing network (the owner's block), then J2, J3, J4 |
| J8b on the best combination | J7 on the owner's word |

Builds, in order of need: the zero-avoiding term (a loss term; the gap penalty's code is the
pattern); µP for the token path and the health checks on it; the history block (16 × 3 integer
codes, embedded network-side, tokens in J5's encoder) once approved; then the two auxiliary heads
with their targets in the rollout buffer (the opponent's next action from the rollout's record; the
opponent's hand from the engine state, never into the observation).

## Not in this plan

Policy distillation of search (closed), value targets from playouts (closed), the belief root as a
product (not adopted), regret matching and HP binning (simultaneous moves; not our game).

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

* **Arm S stopped at 17,737M** (owner, 2026-10-30): 51.6 / 53.4 / 54.3 / 54.6% at 16,400-17,600M
  ([P40](P40_capacity_or_signal.md)). So J8a's bar is that trend: above the base's band on two consecutive reads.
* **`E8-A8-R95-S44`**, the base, continued in place from 2,000M toward its plateau (10,000M budget) since 2026-10-30 20:06 UTC.
* **Setups (2026-10-31; greedy, every 800M on J1 and R94's lineage, and the policy along them).** They move with the
  recipe and stay near ties on the US side (R94 drifted from WG 4 Italy 3 to WG 4 Canada 2 Italy 2 with no gain); the USSR
  sits at Czechoslovakia 3 Poland 3 on most of the lineage (Czechoslovakia 3 Yugoslavia 3 at 8,000-10,000M, until R48's
  switch), Arm S unmoved. **A forced Yugoslavia is not followed up**: next P(Yugoslavia) 0.5-0.7% on J1, ~0-0.1% on R94
  and the base; the setup behaves as a script over the placement count. Opening probabilities at T=1: Czechoslovakia 3
  Poland 3 is 67% on J1, 97.6% on R94, 100% on the base (J1 off-book 1 USSR game in 3); Yugoslavia 1 Poland 4 East Germany
  1 is 2e-8 on J1. **Owner: the network must discover openings itself** (no seeding, P40a not run). The candidate, J1 +
  `--setup-mc-credit`, comes after J5: "major architecture overhaul can change a lot."
* **J1, `E8-A8-R102-S44`**: built (`--zero-avoid-coef`, `zero_avoid_kl` in `ai/training/nash_pg.py`), launched 2026-10-30
  20:08 UTC at c = 0.01, to 2,000M. The term is on the learner's rows only, as the entropy bonus.
