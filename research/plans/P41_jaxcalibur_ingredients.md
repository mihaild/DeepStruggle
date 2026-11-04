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

## Before J5: the J1 batch (owner, 2026-11-03)

**Why a batch on J1 first.** J1 changed the regime rather than a number: saturation gone (median gap
1.23 against 3.50), Nuclear Subs learned unaided, no plateau at 12,000M (a 2,400M flat stretch then +4),
the E8 board's top (greedy 1539, +83 over the previous best raw network; under the 64-world root 1693).
Every verdict taken under saturation that depended on its symptoms is open again: depth (the sphere
trunk's heads died of saturation), grown-in-place capacity (new capacity under saturated heads), the
league (a saturated main could not take the exploiter's lesson). J5 at 3×128 is ~3.2k steps/s, 3×192
~2.1k, so a 2,000M read is a week to ten days of the GPU alone; the J1 line runs at ~110k. A week of
J1-line cells, two slots, sets J5's form and size and tests the reopened verdicts.

**The exploiters (R106 US / R107 USSR, from J1's 15,210M state, the zero-avoiding term on setup placements
only, one frozen greedy opponent = J1's 13,520–13,600M SWA, 20% of games with the human opening forced
on the learner's side as environment).** First read at +390M: **US 51.4% against J1's own bar 47.3
(+4.1); USSR 56.7% against 52.7 (+4.0)**, ±1.6 each. Openings: the US exploiter plays West Germany 4,
France 2, Italy 2, Iran 1 in 94% (J1: France 3, Italy 3, Canada 2, Iran 1, West Germany empty); the USSR
exploiter Poland 4, East Germany +1, Finland +1 in 72% (J1: Finland 3, Poland 3) -- the human opening
with Finland for Yugoslavia. **Confound:** both moved to the forced openings within ~400M and no run
without the 20% share exists, so whether an exploiter *discovers* the hole or copies the shown opening
is unmeasured; and whether the +4 is the opening or better seat play in general is unmeasured (the E7
exploiter's +11.6 was a third general US play). J1′ lost the recipe match (42.5% against J1's best SWA),
so every cell below is on J1's recipe R102.

| cell | what | why first / what it decides |
|:---|:---|:---|
| **B0** | **the exploiters without the human share**: R106/R107 with `--forced-openings-frac 0`, 400M each, same reads; plus a behaviour report of each exploiter at its best read, and the exploiter against J1 *forced to the human opening* | the confound: does a learner find the opening unshown, and is the margin the opening. Decides whether forced openings as environment are a tool and what the league is punishing. ~2 h |
| **B1** | **the deep trunk under the term**: A30 on R95 + `--zero-avoid-coef 0.01`, from scratch, 2,000M against J1's own curve at matched steps. **Redefined after the recipe search (owner, 2026-11-04):** the term's cost on the deep trunk scales with the trunk's step size (A30 takes ~2× J1's per-update KL at the same rate, so the same coefficient pulls ~2× as hard: entropy 1.43 against 1.25, and the cost grows with training), so B1 is the **transfer rule**: choose the pair (coefficient, rate) for which the deep network's per-update KL and entropy at 200M match J1's (0.0018–0.0022; ~1.25), one more cell after D3 (term 0.003 at 1e-4, or the ramp) if D3 does not already match; then **the best deep recipe to 2,000M against J1 at matched steps -- the gate**. A pass (level with or ahead of J1) earns the run to saturation, the capacity read; a constant deficit at 2,000M predicts the same plateau later (P39 arm 3a: a constant 3-point deficit from 2,400M to 4,800M) and ends the deep-MLP work. No-term cells are diagnostics only (D4 separates the rate from the term), never candidates. | the capacity verdict taken under saturation, re-asked; **decides J5's form and size**: level with J1 at 2,000M → J5 from scratch at 3×192; still lagging → J5 at 3×128, distilled from J1 (Arm C's pipeline), and the capacity story stays closed. Either way the transfer rule goes to J5's 60M checks: its coefficient and rate are set so its per-update KL and entropy match J1's, not copied. **Decided (owner, 2026-11-04):** the 150–200M medians -- J1 0.0018 / 1.25; D2 (0.003, 3e-4) 0.0056 / 1.12; D3 (0.003, 1.5e-4) 0.0036 / 1.06; B1 (0.01, 3e-4) 0.0055 / 1.43; A30 no term 0.0056 / 0.75 -- match J1 on neither count; the next cell is **D5, the ramp**: A30 on R95 at lr 1.5e-4 with the term at 0.01 ramped linearly from 0 over 1,600M (`--zero-avoid-ramp-steps`: 0.0025 at 400M, 0.005 at 800M, 0.01 from 1,600M), `E8-A30-R113-S44`, to 2,000M as the gate. D3 stops after its 800M read, D4 after its 400M; B3 (J1 from 15,210M **to 20,000M in any case, no stop rule** -- owner, 2026-11-04 ~08:55 UTC) and D5 launch in the freed slots; sharing the GPU, D5 reaches its 2,000M gate ~11:00 UTC and B3 its 20,000M ~12:00 UTC on 2026-11-05. D5's transfer-rule read (per-update KL and entropy against J1's) is taken at 1,600M, where its coefficient reaches 0.01, not at 200M; its early reads will resemble the no-term A30's.

**D3 / D4 done (2026-11-04):** D3 (0.003, 1.5e-4) 8.6 / 13.5 / 21.6 at 200 / 400 / 800M, level with the no-term A30 at 3e-4 (21.2) and 8.8 behind J1 (30.4); D4 (no term, 1.5e-4) 5.1 / 14.8. **At 800M depth without the term is +2.3 over A8 without it (21.2 against 18.9) and the term on A8 is +11.5 (J1 30.4), but the term on the deep trunk is +0.4: the two gains do not yet combine** -- the deep trunk's problem under this recipe, rather than speed.

**D6 (owner, 2026-11-04): distil J1 into the deep trunk, then continue with the term -- D5 paused for it.**
Arm C's null (held its start, no gain) came from a saturated teacher (R94) and a term-free continuation; J1 is
an unsaturated teacher (entropy 1.25, still climbing), so the student inherits a policy with gradient in it and
the late gain the from-scratch deep trunk has not produced. **Fit:** `--mode online-distill`, A30 in D5's shape
under µP, teacher J1's 13,520–13,600M SWA, J1's self-play at T = 1, the full policy and the value; **bar before any
RL:** argmax agreement ≥ 92%, level with J1 greedy, the plasticity probe at a fresh network's numbers; a miss
resumes D5 at once. **RL:** R102's term at 0.01, lr 1.5e-4, no entropy bonus beside the term (J1′'s form; Arm C's
start was lost to the bonus on plastic heads), 16k, clip 0.1, the pool as J1; the 60M check against J1's KL and
entropy; reads every 400M of RL against the reference and against **J1's own continuation (B3) at matched RL
steps**. **Pass:** above J1's trajectory at matched steps after 800–1,600M of RL. D5 resumes in the first slot that
frees (B3's end) and reaches its gate a day later. D5 and D6 are complementary: D5 asks whether the deep trunk can
learn the late gain itself, D6 whether a deep trunk that has it can use its capacity beyond it; D6 passing makes
"distil, then train" the route for deeper networks and J5's start.
**D6's fit passes (2026-11-04, `E8-A30-R114-S44`, R114):** held-out argmax agreement 92.6% (KL 0.0150, value MSE
0.0006) at 400M positions; against J1's 13,600M SWA greedy 50.1% ± 1.1 (US 45.8 / USSR 52.9); plasticity 99.0%
on random labels under its own head, trunk frozen (a fresh network's number; Arm C's fit 97.0%, a trained
teacher ~32%). **RL launched 15:53 UTC as `E8-A30-R115-S44` (R115):** the warm start, lr 1.5e-4, entropy 0, the
term at 0.01, µP with the distilled readouts kept; 60M check against J1's 0.0018 / ~1.25, then reads every 400M of
RL against the reference and against B3 at matched RL steps (B3: 61.9 / 61.9 / 61.0% at 15,600–16,400M).
**D6's RL lost its start inside 10M (2026-11-04), stopped at ~75M:** each snapshot against J1's 13,600M SWA greedy
50.8 (the fit) → 23.5 (10M) → 18.7 (20M) → 21.4 (40M) → 18.7% (60M) -- a collapse, not a drift back to a line.
**Every check we run stayed normal while it fell:** per-update KL 0.0042 flat, entropy 1.17–1.27 flat, rank 4.1× fresh,
no dormant units, logits bounded. So the 60M step-and-entropy check cannot see this failure; only a match against
the starting point does, and **a warm start's checks must include one from 10M on.** Arm C's failure repeated
without the entropy bonus and at lr 1.5e-4 (R99 held its start only at lr 5e-5 with entropy 0.002, and then did
not gain). With R44 (RL from the soup's weights, lost in 10M) that is four times the same shape: **the first PPO
updates at a normal rate destroy weights produced by another objective**, and the record has never tried the
standard remedy for it -- a frozen anchor. NashPG's π_ref refreshes every 200k steps (three iterations), so it
moves with the policy; the cell that has not been run is **D6b: the RL from the distilled start with π_ref frozen
at that start (`--ref-update-freq` ≥ 50M) and η raised (0.5–1.0) for the first 400M, then the normal cadence**,
at the normal rate, with the match against the start at 10M and 60M as the check. Hold-and-gain → capacity pays
on J1's signal and "distil, then anchor, then train" is the route; hold-and-nothing → Arm C's answer stands in
the new regime. **D6b runs now in D5's slot (owner, 2026-11-04); D5 is paused again, resumable, and goes back in
when a slot frees.** Settings: R115's start and recipe with `--ref-update-freq 50000000` and `--eta 0.5` for
the first 400M, then the normal cadence and η as a continuation with a change; the warm-start check is the
match against J1's 13,600M SWA at 10 / 20 / 40 / 60M (held if 60M is within noise of the fit's 50.8%; stopped
at the first read clearly below ~45%), with the KL to π_ref logged; then the 400M reads against the reference
and B3. If η 0.5 holds the start but nothing moves, η 0.25 next; if it still loses the start, η 1.0.
**D6b holds (2026-11-04, `E8-A30-R116-S44`, R116):** against J1's 13,600M SWA 48.1 / 51.3 / 49.8 / 49.8% at
10 / 20 / 40 / 60M (the fit 50.8; D6 23.5 at 10M), KL to π_ref ~0.02 and not climbing, per-update KL 0.003 --
the anchor restrains the direction without freezing the weights. **400M: 60.7% ± 1.1 against the reference**
(US 57.4 / USSR 62.4); B3 at the matched RL step (15,610M) 61.9% -- **level with J1, within noise**: the deep
network holds J1's level through 400M of RL, which no warm start did before. The anchor phase ended at 410M;
it continues as `E8-A30-R116-S44@410M+R115` (the plain release: R115's cadence and η 0.1), reads every 400M
of RL from 800M against the reference and B3 (61.9 / 61.9 / 61.0 / 60.7% at 15,600–16,800M). The release is
the point at which D6's collapse mechanism returns; a match against J1 at +10M and +60M after it is the
check, and a clear fall restores the anchor (the 50M cadence at η 0.1 first). The question from here is gain
past J1's trajectory, not the hold. ~12 h alone |
| **B2** | **the league from J1's plateau**: the unshown exploiters R108 (US) / R109 (USSR) as one fixed league member (a seat pair: R108 deciding as US, R109 as USSR, since each one's untrained seat collapsed to 3–8%) in J1's pool at the P24 share, the main continuing. **(b), Arm S's searched opponent as the comparator, is dropped (owner, 2026-11-04): the recipe stays simple, and an opponent from another lineage is not simple.** The league itself waits for the owner's word under the same rule | the signal question on a plastic base. Read on the fixed reference, the own-past gain, **the margin a fresh exploiter finds afterwards** (the league's own metric), and J1's West Germany and Europe-control rates |
| **B3** | **J1's ceiling and coefficient**: one more leg to 20,000M on the flat-then-jump precedent (the stop rule fired once on a low read whose next was J1's best); c = 0.003 and 0.03 from scratch to 2,000M; a late anneal of c from J1's plateau (the floor costs 9.9% of temperature-1 games to the player's own DEFCON 1, R94 3.4%) | whether the term is tuned and whether J1 was stopped early |
| **B4** | **setup credit** (`--setup-mc-credit`) on J1 with the term on setup placements (`--zero-avoid-scope setup` keeps openings reachable) | the owner's candidate for openings; the critic's −11 at turn 1 on an unplayed opening is its mechanism; read on the openings and the exploiter margin |
| **B5** | **the head norm** on J1 (`--ladder-head-norm`, A31's +3 at the plateau) | one flag |
| **B6** | **in-place growth** of J1 at its plateau (sphere blocks and width at identity, P39 arm 6) | its blocker, saturated heads, is gone |

**B0 result (2026-11-03, R108 US / R109 USSR: R106/R107 with no human opening shown, 400M each, own
seat against J1's 13,600M SWA greedy, the bars 47.3 / 52.7).** US **52.8% (+5.5)**, USSR **54.8% (+2.1)**
(shown: +4.1 / +4.0; the differences within ±2.2). **Unshown, the US exploiter finds West Germany on its
own** (France 3, West Germany 3, Italy 2, Iran 1 in 75%; shown, it went to West Germany 4); **the USSR
exploiter does not find the human opening** (Finland 3, Poland 3 in 100%). **The margin is not J1's
opening:** J1 forced to the human opening scores US 46.6 / USSR 53.2 against itself, within its bars, and
the exploiters keep their margins against the forced J1 (R106 +8.0, R107 +3.0). It is play after the
setup: the USSR exploiter goes for West Germany and Europe (first USSR influence in West Germany on turn
1 in 30% against 3% in J1's self-play; first to control it 48% against 12%; ever controls Europe 8.7%
against 3.2%; Europe-control wins 4.4% against 2.1%, about +2.3 of its +4); the US exploiter holds West
Germany (80% at the end of turn 3 against 53%) and wins through 20 VP and final scoring. So the human
share bought the opening's *form*, not strength, and the hole is **Europe left uncontested**, the owner's
"Europe control underestimated" measured: a learner finds it in 400M and self-play never punishes it. B2's
league uses the unshown exploiters (R108/R109) and reads J1's West Germany and Europe-control rates
beside the reference; B4's setup credit is no longer motivated by these openings.

**Order (owner, 2026-11-04 ~00:10 UTC):**

1. **B1 -- understand the deep network first** (running since 2026-11-03 22:03 UTC, alone): does it train
   at all, and how does it compare with the shallow one -- B1 against J1 at matched steps, to 2,000M.
2. **B3 -- make sure J1's ceiling is reached**: the extra leg, the coefficient cells, the late anneal.
3. **B6 -- in-place growth, only if B1 fails.**
4. **B2 -- the league** (still planned, as the seat-pair member): after what can be reached without it
   is understood.
5. **B4 -- setup credit**: after B2, and only if the exploiters cannot shift the openings themselves.

**B5 (the head norm) -- the evidence, read against it (running session, 2026-11-04):** A31's revival of
R48's dead heads (dormant 12.2% → 0.7%, rank 89.8 → 101.7) came with reads of 51.0 / 50.4 / 50.2% against
R48's own 52.4% at the same step -- no matched-step gain; the "+3" was against R48's earlier band. A32,
the same norm on the deep A30 trunk from scratch, failed the 400M rank check (102.7 → 74.9, 7.1% against
A30's 8.3%). And neither J1's heads (0.3% dormant) nor B1's (0.0%) are dying -- **corrected (running
session, 2026-11-04): the 0.3% was J1 at 60M.** The full probe on J1's 13,600M SWA: **11.6% dormant head
units at τ 0.1** (5.6% at 0.025), head norms 14–34× a fresh head's, random labels under its own head with the
trunk frozen 36.7% against 99.1% for a fresh head on the same trunk, P36c's targets 73.6%. J1's heads have
lost much of their plasticity by its plateau, the pattern A31's head norm addressed on R48 -- so B5 has a
mechanism on J1, though A31 still showed no matched-step gain. **B5 stays dropped on the strength evidence,
with the mechanism now present on J1; a remedy cell if J1's continuation (B3) stalls.** D6's distilled deep
network on the same probe: own head 99.0%, fresh head 99.2%, P36c targets 97.6%, dormant 0.0% -- a fresh
network's plasticity carrying J1's play.

**J5 is postponed (owner, 2026-11-03 ~23:45 UTC):** nothing about J5 launches until the owner says so.
As first written: **J5 launches on B1's answer**, on R102 plus whatever passed, at the size B1 picks; the
peer's recommendation of 3×128 stands if B1 says distilled. **Rules:** the per-seat rule against J1 at
matched steps for the from-scratch cells, against J1's plateau band for the branches; B2 passes if a
fresh exploiter's margin shrinks with the reference gain not behind the own-past gain. **A distilled J5
is not run first** (Arm C held its start and did not gain; the transformer's fit alone is ~2 days at
456 µs a row).

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
* **Decisions (owner, 2026-10-31).** **J1 continues past 4,000M while it improves** (the base's stop rule), and **J1′ runs when the base
  finishes**: whether training can be simplified is settled before J5 starts. If J1 fails at 4,000M, nothing new is launched and the
  owner reads the numbers to decide J5's configuration. **The stop rule (owner, as amended 2026-11-01):** a run stops when, at two reads in a row, none of the last three reads beats the best
  read from before them by more than 0.5 points (0.5 is a real gain if it accumulates; one firing alone stopped J1 on a low read at 6,800M, and its next read was its best). **J5 before the base's gap-penalty branch.** J5's sizes are measured when the current runs
  have finished, on a free GPU; nothing is paused for it.
* **Lose-now moves (2026-10-31).** Equilibrium of PPO + c KL(uniform ‖ π): π = c / (n Δ), Δ = (1 + v) / raw advantage std (0.21);
  measured at DEFCON <= 2 on J1 at 2,800M: median 1.4e-3 summed over a position's losing moves, predicted 1.1e-3 (base 1e-16,
  R94 3e-4). Entropy alone: exp(-Δ/c), effectively 0. **Top-choice suicides are not blunders**: positions already lost (v -0.95
  to -0.98), the best safe alternative scoring 0-3% in greedy playouts on all three networks, and no safe alternative at
  that step in 38-61%. **What J1 does cost:** own DEFCON 1 ends 9.9% of its temperature-1 games (R94 3.4%, base under 1.3%),
  and held scoring ends 3.4% of its greedy games (base 0.9%), both from ~600 games, the first training noise, the second
  in play.
* **Nuclear Subs (2026-10-31).** J1 at 3,600M events it as the US 21% of the time, P(event) 21.1% (2,000M: 14%, 16.9%); the base and
  R94 0%, P(event) 0.3%. **Paired test at DEFCON 2** (US decisions from sampled games, read with the Subs flag off and on in a copy of
  the state; the flag is in every observation): choosing a coup target, P(battleground) **J1 10.3 -> 78.8%**, base 6.4 -> 6.2%, R94 7.6
  -> 7.7%; coup on offer, P(coup) **J1 6.5 -> 26.0%**, base 3.8 -> 3.8%, R94 2.1 -> 2.3%. J1 has learned what the card does; the
  plateaued networks show no response to it. The clearest sign yet of what the saturation term buys.
  **As the USSR (the US's card; its event fires for the opponent), greedy, ~1,300 games:** share of plays at AR7 **J1 44%**, base 5%, R94 6%;
  held at AR1 J1 chooses it 1.7% (base 28%, R94 18%) and it stays in hand to AR7 373 times (base 49, R94 60). Modes in J1: influence 58%,
  event 26%, coup 12% (on an opponent's card "event" means the event first, then the Ops anyway, and for Nuclear Subs the order changes
  nothing that round: owner). **Setups at 4,000M:** USSR Poland 3 Yugoslavia 3, US WG 4 Italy 2 Spain/Portugal 2 Iran 1 (59%) or WG 4 Italy 3 Iran 1
  Spain/Portugal 1 (41%); the base at 6,000M: USSR Poland 6, US France 3 Italy 2 South Korea 2 Spain/Portugal 2.
* **J1 passes at 4,000M (2026-10-31).** 47.8% against A13+R48's SWA (the base 44.1%). **Head to head 53.4% ± 1.1** (US 50.4 / USSR 54.5; the
  control's self-play bar at 4,000M US 46.1 / USSR 52.3: above it in both seats), held from 52.8% at 2,000M. Saturation: median
  top-two gap **1.23 against 3.50**, mean P(top) 0.64 against 0.85. Rare cards, greedy: Nuclear Subs US event 18% against 0%; One Small
  Step ~0% for both, Wargames 34% against 29%: the first rare card moved, not all. (Event shares on an opponent's card are an order, event before or after the Ops, not whether it fires.) Endings, greedy, ~740 games: held scoring 1.7% against
  3.2% (2,800M's reversal gone), own DEFCON 1.1% against 1.6%; at T=1 own DEFCON still 8.1% against 1.2%. It keeps running past 4,000M.
* **The base stopped at 7,610M (48.5%, its rule); J1′ launched 2026-11-01 03:27 UTC.** J1 is level with R94's read at 5,600M (52.9%), behind it head to head at 5,200M (47.5%).
* **J1 stopped at 6,810M by the rule (2026-11-01):** 51.8 / 52.0 / 52.9 / 51.2 / 52.8 / 49.5% over 4,800-6,800M; **level with R94 head to head, 51.6% ± 1.1 at 6,400M against its 16,000M** (47.5% at 5,200M). A one-recipe route from scratch to the best E8 network's level, without the gap penalty or a branch: the base recipe for what follows if J1′ does not beat it.
* **J5's go (owner, 2026-11-01 ~22:50 UTC, standing until 09:00 UTC 2026-11-02):** if J1′ saturates by then, launch J5 on the stronger of J1 and J1′, J1′ on a tie. Saturates = the two-firings stop rule. Stronger = one match, each run's best-read SWA against the other's, raw, 1,000 deals a side; a tie is within ±2.2 points of 50%. Size: speed is measured only once both J1 and J1′ have finished (owner), so J5 waits for that even if J1′ saturates first; then 3 × 192 unless the measurement contradicts it, open to change when the owner returns.
* **J1 past its flat stretch (2026-11-01):** 51.1 / 51.8 / 55.8 / 56.6% at 7,600 / 8,000 / 8,400 / 8,800M, after averaging 51.8 over 5,600-8,000M: flat for 2,400M was not its ceiling, which also leaves open whether the base, stopped at 7,610M, had reached its own. **Level with Arm S head to head, 50.7% ± 1.1** (J1 8,800M against 17,600M, raw). J1 keeps running; the bar is 57.1, earliest stop 10,400M. **J1′** 46.6 / 46.8 / 46.9% at 2,800 / 3,200 / 3,600M, level with J1 at 3,600M (47.3); its US opening has no West Germany from 1,600M.
* **The night of 2026-11-01/02:** J1 keeps climbing, 57.6 / 58.4 / 56.3 / 58.9 / 59.8% at 9,200-10,800M, so the rule cannot stop it before its 12,000M ceiling; J1′ 49.0 / 51.6 / 51.8 / 52.2 / 53.2% at 4,000-5,600M, level with J1 at the matched steps (J1 47.8 / 48.7 / 51.8 / 52.0 / 52.9%). Neither has saturated, so J5 has not been measured or launched. J1's US opening moved to France 3 Italy 3 Canada 2 Iran 1 at 8,400-8,800M, the same read as its 4-point jump: no West Germany, as J1′ from 1,600M. Whether that is strength or an exploit of the reference is not measured.
* **J1 past Arm S (2026-11-02):** 55.8% ± 1.1 at 12,000M against Arm S's 17,600M SWA (50.7% at 8,800M); J1′ level with it at 6,400M, 50.9%. J1 and J1′ both continue under the rule.
* **J1 stopped at 15,210M by the rule (2026-11-03):** 61.0 / 62.5 / 60.0 / 59.7 / 63.5 / 61.6 / 60.4 / 60.0 / 60.0% at 12,000-15,200M, best 63.5% at 13,600M; a level of ~61 from 12,000M. J1′ continues (best 59.8% at 9,200M, earliest stop 10,800M); J5 follows both, by the owner's sequence.
* **West Germany and Europe control (J1 11,200M, greedy, 1,000 games a match):** the reference as USSR controls West Germany in ~91% of games against J1's empty opening (first control on turn 1 in 51%), and J1 still scores 58-61% as US; it wins by Europe control only 1.7% of those games (its self-play 3.4%). So the empty opening is not tested by an opponent going for Europe control. USSR wins by Europe control: J1 self-play 2.9%, J1 against the reference's US 5.8%; US under 1% everywhere. **The win is taken when it is there:** 219 of 220 USSR action-round chances played it on 3,000 games a match; headline chances usually wait, and all 28 still won by Europe control. Whether the networks play towards control is not measured.
* **The human US opening, forced on J1 12,000M (West Germany 4 Italy 3 Iran 2, self-play, greedy, 3,000 games each):** the US scores 45.9% ± 0.9 against 49.6% ± 0.9 on J1's own opening; the critic at turn 1 says 34.4% against 45.7%, so it triples the 3.7-point gap, and the excess is almost all in that first read (error -11.5 points at turn 1, -3.2 at turn 2; J1's own opening -3.9 and -4.3). The owner's pessimism about unplayed openings, measured; the case for `--setup-mc-credit` after J5. J1 has not practised this opening on either side, so the 3.7 is not a verdict on it. USSR Europe control wins rise to 6.5% with it.
* **The deep-trunk recipe search (owner, night of 2026-11-04: find a good recipe for the deep trunk from scratch).** All A30 (SimbaV2 trunk, 2 blocks, A26's width) under muP, reads against A13+R48's SWA at 200 / 400 / 800M: **J1 (A8, term 0.01) 11.8 / 16.0 / 30.4; A30 on R95 (no term, lr 3e-4) 9.2 / 12.3 / 21.2; B1 (term 0.01, R102) 2.9 / 5.6 / paused at 636M; D2 (term 0.003, R110) 7.4 / 10.5 / 14.0; D3 (term 0.003, lr 1.5e-4, R111) 8.6 / 13.5 / -.** The term holds the deep trunk back at lr 3e-4, the more the stronger, and its cost grows with training (D2 1.8 / 1.8 / 7.2 behind the no-term run) -- the opposite of A8, where it paid off from 800M. Halving the rate helps (D3 3.0 ahead of D2 at 400M, 1.2 ahead of the no-term run at the full rate); the deep trunk still steps ~2x J1's per-update KL at half the rate (0.0036 against 0.0018 at 150-200M). The term also spreads the deep trunk's policy more than A8's (entropy 1.43 against J1's 1.25 at the same coefficient). D4 (R112: no term at lr 1.5e-4) launched 2026-11-04 08:28 UTC to separate the rate from the term. B1 paused at 636M (resumable) and the queued B3 leg cancelled with it, for the owner to restart.
* **The deep-trunk search, continued (2026-11-04).** D3 (term 0.003, lr 1.5e-4) **21.6% at 800M**, level with A30 on R95 at 3e-4 (21.2), 8.8 behind J1 (30.4). D4 (no term, lr 1.5e-4) **5.1 / 14.8%** at 200 / 400M: at the half rate the term is worth +3.5 at 200M but the no-term run leads by 1.3 at 400M. **Option B (the owner's choice): D5**, A30 at lr 1.5e-4 with J1's term ramped in over 1,600M, running (2,000M gate about a day later than first planned). **D6 (owner): J1 distilled into A30 passes every bar** (92.6% agreement, 50.1% against J1, own-head plasticity 99.0%), **and its RL loses the start inside 10M** (50.8 -> 23.5 -> 18.7%) with the term at 0.01, lr 1.5e-4 and no entropy bonus, while KL, entropy and geometry all read normal; stopped at ~75M, D5 resumed. **J1 on the plasticity probe at 13,600M:** own-head random-label fit 36.7%, dormant head units 11.6%, head norms 14-34x fresh -- its heads have lost their plasticity by the plateau (the 0.3% dormant quoted for B5 was at 60M).
* **US Defectors under the USSR's peek, real games (owner: it should pretty much never be headlined):** with the stage matched (turns 7-10, T=1 self-play), J1's mean P is 58 → 30% with the peek and 52 → 36% without over 800-7,200M on one fixed set of real positions. The fall to 4,000M is general; the peek-specific gap opens only after it (+6 at 800M, -6 to -8 from 5,600M), then flat at 28-30% to 7,200M. Late-arriving, so partly underfit, and nowhere near never. Other networks over all turns (T=1): R94 35%, base 56%, E7 best 72% (its USSR cannot see the reveal). Whether 30% is a mistake is not measured.
* **Nuclear Subs at DEFCON 2 (US, flag flipped on the same positions):** P(coup when offered) off → on J1′ 2,400M 2.7 → 35.3%, J1 7,200M 5.0 → 31.7%; P(battleground target) about 10 → 80% for both, already there in J1 at 2,400M.
* **J1, `E8-A8-R102-S44`**: built (`--zero-avoid-coef`, `zero_avoid_kl` in `ai/training/nash_pg.py`), launched 2026-10-30
  20:08 UTC at c = 0.01, to 2,000M. The term is on the learner's rows only, as the entropy bonus.
