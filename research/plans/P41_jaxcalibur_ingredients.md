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
of RL from 800M against the reference and B3 (61.9 / 61.9 / 61.0 / 60.7 / 64.3% at 15,600–17,200M; the 17,200M high is not a gain: 50.2% against its own 16,800M SWA, and J1′ scores 43.3 against it against 42.5 against 13,600M). The release is
the point at which D6's collapse mechanism returns; a match against J1 at +10M and +60M after it is the
check, and a clear fall restores the anchor (the 50M cadence at η 0.1 first). The question from here is gain
past J1's trajectory, not the hold. **The full release fails (2026-11-04, `E8-A30-R116-S44@410M+R115`):** against J1 32.9% at 420M and 25.8% at 470M -- D6's collapse returns as soon as π_ref follows the policy again, after 400M of RL, so the anchor is not a warm-up that the network outgrows. Stopped by PID at ~470M; **the anchor is restored (owner): `E8-A30-R116-S44@410M+R117`, π_ref every 50M at η 0.1 (R117)** from the 410M state, with the same +10M / +60M checks against J1 and reads every 400M from 800M. **Correction (2026-11-04): D6b does not hold its start over 400M, it bleeds slowly.** Against its own distilled start (R114 best.pt) the 410M state scores **45.9% ± 1.1** (US 41.6 / USSR 48.7) and the staged release's 470M **43.8% ± 1.1** (US 41.8 / USSR 44.2) (`data/reports/d6b_410M_vs_start.json`, `d6d_470M_vs_start.json`); against J1 the same slide, 50.1 (the fit) → ~50 (60M) → 44.9 (410M) → 42.2 (470M). The 60M check saw the hold, not the drift after it, and "level with J1" at 400M was level with one J1 read on a reference that no longer separates players at this level. The staged release's checks: the 410M start scores 44.9% ± 1.6 against J1 (the anchored phase slid from ~50 at 60M while reading level on the reference), the branch 44.9 at 420M and 42.2 ± 1.1 at 470M -- within noise of the start, where the full release lost 12 and 19 points. **D7, the shallow control (owner, 2026-11-04):** the staged release is stopped (resumable) and its slot runs D6's two steps on J1's own architecture -- J1 distilled into a fresh A8 (`E8-A8-R114-S44`, R114 unchanged), then RL on R116 (`E8-A8-R116-S44`) with D6b's checks, read at 410M against its own start and J1. The same bleed on A8 puts the loss in the distil-then-RL procedure (the distilled critic, the reset optimiser and pool, or the halved rate), not in the deep network; a hold or a gain on A8 puts it in the deep network. **D7's answer (2026-11-05): the procedure.** The shallow start (89.7% agreement, level with J1, plastic) bleeds as the deep one did -- 47.2 / 47.9 / 48.1 / 46.1% against J1 at 10-60M, then **43.1% against its own start and 43.6% against J1 at 410M** (D6b 45.9 / 44.9). Distilling J1 and continuing RL on R116 loses ~4-7 points over 400M on J1's own architecture, so D6b's bleed says nothing against the deep trunk, and "distil, then RL" is not a route to a stronger network as it stands: the distilled policy is not a fixed point of this RL, whatever the architecture. Not pursued further (the owner moved the GPU to the transformer). If η 0.1 also loses the start, the next step is the 50M cadence at η 0.25 to 0.5; if it holds, the open question is whether a 50M-cadence π_ref still lets the deep network gain past J1 or only holds it there. ~12 h alone |
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

## J5 now, from scratch (owner, 2026-11-04 evening)

**Why now.** The warm-start route is closed: D6b bleeds under the anchor (45.9% against its own start at
410M, 43.8% at 470M; level with J1 on the E7 reference was the blunter instrument) and **D7, the same
procedure on a fresh A8 (J1 → A8 by R114, then RL on R116), bleeds the same way: 43.1% ± 1.1 against its
own start and 43.6% against J1 at 410M** (checks 47.2 / 47.9 / 48.1 / 46.1 at 10–60M from a level, plastic
start) -- **the loss is in distil-then-RL as a procedure, not in depth; D6b says nothing against the deep
trunk.** B3 stopped at 17,600M (63.3%); D7 at 410M. J1's own heads
lose plasticity by its plateau (11.6% dormant, 36.7% random-label fit) and B3 is flat at it. So the
transformer gets no shortcut from J1 and is run from scratch, with the GPU to itself: B3 stopped, D5 left
paused.

**The run.** 3×128 per-type-MLP token transformer as built (`ai/models/token_transformer.py`, the main
path), on J1's recipe R102 (16k, clip 0.1, the zero-avoiding term, the pool at frac 0.3), readouts zeroed,
coefficient and rate set by the transfer rule at the 60M checks (per-update KL and entropy matched to J1's
0.0018–0.0022 / ~1.25; start at 0.01 and 3e-4, relaunch lower rather than continue if the step is clearly
above J1's). Reads at 200M and every 400M against the reference beside J1's curve at matched steps (11.8 /
16.0 / 30.4 / 33.5 / 35.5 / 38.6% at 200–2,000M): **400M the preliminary verdict, 800M the meaningful one
(E7-09-44's lead was gone by 400M), 2,000M the gate.** The compiled token path is benchmarked first.

**Timings alone at 3×128 as the code stood (3.2k steps/s):** 60M ~5 h, 200M ~17 h, 400M ~1.5 d, 800M ~3 d,
2,000M ~7 d; 3×192 (2.06k) ~1.5× longer. The compiled path and the asymmetric layout each roughly halve
them; with both, 400M in ~13 h and the gate in under 3 days. An H100 multiplies by 3–4.

**J5's second shape, built in parallel (the asymmetric layout):** 84 country tokens, up to 10 hand-slot
tokens (an empty status for unused slots), the global token, and the 110 cards as a **read-only memory**
-- card identity plus status embedding projected to keys and values once per forward, no FFN, no
self-update -- attended by the active tokens at every layer; the per-card head reads the hand slots and the
memory. In the bandwidth-bound regime the token transformer runs in (295 µs a row at 3×128 against a
FLOP-bound ~65), cost scales with tokens processed, so ~2× on cost and on activation memory; the
country-to-country and card-to-country attention it keeps is the cheap part. It replaces the run if the
60M checks fail, else it is the next run.
**Measured (2026-11-05, `--ladder-token-card-memory 16`, compiled, alone):** the compiled full layout runs
**6,470 steps/s at 3×128** (3,200 before: fused RMSNorm 4,760, then compile; 2.02×) and 4,028 at 3×192
(1.96×), peak memory unchanged; **the asymmetric layout 7,844 and 5,012 -- 1.2×, not 2×.** Attention
keys still number 211 (101 active + 110 memory), every layer still projects all 110 cards to keys and
values, and the input projections, heads, value head and environment do not shrink, so the token work
that goes away is a smaller share of a step than the FLOP count suggested. It stays the next run, not a
replacement; a once-per-forward memory projection shared across layers would recover a little more.
**J5 (`E8-A35-R102-S44`, 3×128 compiled) launched 2026-11-05: 60M ≈ 2.6 h, 400M ≈ 17 h, 800M ≈ 1.4 d,
2,000M ≈ 3.6 d.** It crashed at 60.03M on the opponent pool's first eviction (capacity 12, a member
every 5M): the token path's bf16 GEMMs take a cuBLASLt workspace, which landed in the first CUDA graph's
private pool, and dropping that graph freed memory the later graphs still wrote to -- the class of E4's
cuBLAS bug, reproduced by a capture/evict/replay stress test at the 14th member every time, compiled or
eager, and absent without autocast or with the workspace off (A8, fp32, survives the same test). Fixed
in `824b0c9` (the workspace off whenever a token network runs under graphs; 6,626 steps/s after, so no
cost) and resumed in place from 60.03M at 03:42 UTC; nothing lost, the 60M checks run at 65M.

**J5's 60M checks (2026-11-05, at 65M, J1 at matched steps in brackets): the step passes, no relaunch,
lr 3e-4 and c 0.01 unchanged.** Per-update KL 0.0025 / 0.0025 at 40–60M (0.0023 / 0.0022, +10%); clip
fraction 0.216 (0.219); window-KL max 0.014–0.016 (0.012–0.015); entropy 1.25 (1.38) -- the transfer
rule's ~1.25; head-input rank 39.8 of 480, 1.8× fresh (21.9), just under the 2× bar but rising and above
J1's own 1.2× (57.0 vs 46.2); dormant 2.3% (0.5%); head norms 1.3–2.5× fresh (1.4–2.9×); |lse| max 13.5
(11.8); explained variance 0.81 (0.82). **Flagged: setup entropy 0.53 against J1's 1.09**, 0.84 at 20–40M
and still falling -- the transformer settles its openings about twice as fast as J1 did. **Planning
position: hold.** The whole-policy entropy is on the rule; the setup component is a diagnostic, not a
read, and what matters is *which* openings it settles into, not how fast (J1's own odd openings were not
what the exploiters punished -- the margin was post-setup play). Raising the coefficient or adding the
setup-scoped term mid-run would also break the matched-step comparison with J1, which is the point of
the run. The 200M read (~9 h) carries the decision: if J5 is at or above J1's 11.8% the sharpening
costs nothing visible and stays as is; if it is below and the behaviour report's openings are
degenerate (one setup repeated, a seat without its West Germany / Poland core), the remedy is a
**branch** (`@200M+` with `--zero-avoid-scope setup` at a higher coefficient, the B4-style term), never a
change in place, so J5 proper keeps its curve. The owner decides either way.

**J5's 200M read (2026-11-05): 21.1% ± 0.9 against A13+R48's SWA (US 18.9 / USSR 22.4), against J1's
11.8% at 200M -- about 9 points ahead at matched steps**, the first architecture to lead J1 on its own
recipe at any step. The caveat stands in the plan: E7-09-44's early lead was gone by 400M, so 400M is
the preliminary verdict and 800M the meaningful one (J1 16.0 / 30.4). *Openings* (80M-window SWA,
greedy, 100 deals): US **West Germany 4, Italy 3, Iran 2 -- the human opening -- in 82%**, Italy 4 /
West Germany 4 / Iran 1 in 18%, West Germany 4 in every US setup from ~120M; USSR **East Germany 3,
Poland 3 in 100%** (the owner reads it as weaker than Poland 3 / Yugoslavia 3, which keeps the route to
Italy; tracked at each read for whether it settles or flips). J1 at 200M: France 5 / Italy 3 / Iran 1
and Poland 3 / Yugoslavia 3. So the low setup entropy at 60M was J5 converging on a sound US opening,
and the hold was right; the setup-scoped term stays unneeded. *Style against J1 to 180M:* Wargames
endings 7.5% vs 0.9%; Europe control 1.3% vs 7.0%; final scoring 19% vs 26%; about 2–3 more
uncontrolled battlegrounds at turn 8; forced wins taken 53–67% vs 34–52%; heuristic ~100% for both.
**Paused at its next resume state (~210M) for the owner's sync onto main's engine fix (ENG-7, Space
Walk), then resumed in place** -- a minor fix inside the epoch, so the run keeps its name and curve;
the 400M read slips by the pause.

**J5's 400M read (2026-11-06): 23.4% ± 0.9 (US 18.0 / USSR 28.1) against J1's 16.0% at 400M -- the
preliminary verdict is still ahead at matched steps, by 7.4 points.** Three flags, together the
pattern of a rate that has become too high. *Slope:* 21.1 → 23.4 over 200–400M (+2.3) against J1's
+4.2; the US seat is flat (18.9 → 18.0) and the gain is all USSR; J1 reached 30.4% at 800M, and at
J5's slope it would be ~26–28 there, so J1 would overtake by the meaningful read -- E7-09-44's pattern.
*Step size:* per-update KL 0.0033 at 340–400M against J1's 0.0014 (2.3×; 1.1× at 60M and rising
steadily), clip fraction 0.225 vs 0.174; entropy 1.32 vs 1.24; explained variance 0.85 vs 0.80. The
transfer rule matched at 60M and no longer holds: the same rate takes a growing step as the policy
sharpens. *Openings flipped* (400M SWA, 100 deals): the US dropped West Germany for five Italy /
Canada / France setups (Italy 3, Canada 2, France 2, Iran 1, South Korea 1 in 46%); the USSR moved to
Poland 3 / Yugoslavia 3 in 100%, the owner's preferred. The USSR flip is fine and the US one is not a
read in itself, but a flat US seat that abandons a sound opening it had settled on is what a step too
large looks like, not exploration. **Planning position: branch now, keep J5 proper.** The step-size
question is the transfer rule's own, so it is tested the way the rule says: `E8-A35-R102-S44@400M+`
at lr 1.5e-4, same coefficient, read at 800M beside J5 proper's 800M (J1 30.4%). Sharing the GPU,
both reach 800M in ~34 h instead of J5 alone in ~17 h; the cost is the 800M read's delay, the gain
is that the 800M read answers the rate question at the same time as the architecture one. Changing
J5 in place would lose the curve. The owner decides; nothing launched.

**Decided (owner, 2026-11-06): J5 continues at 3e-4 to 800M, no lower-rate branch now; decide again
at the 800M read** (J1 30.4%; projected 26–28%). The supporting check: this is not a one-seat
collapse. Self-play US win share 51–55%; critic AUC rising 0.67 → 0.78 and explained variance 0.76 →
0.85, both above J1's at every window from 120M; per-seat KL, entropy and advantage symmetric. The
step drift is J5's own: J1's per-update KL fell 2.3 → 1.4 ×10⁻³ over 0–400M (clip fraction 0.23 →
0.16) while J5's rose 2.5 → 3.3 (clip stuck at 0.22) -- a rate too high for the maturing transformer,
not a seat failing. The flat US seat (18.9 → 18.0 while the USSR gained 5.7) coincides with the US
opening churn (West Germany in every setup at 200M, gone by 400M). What the 800M read decides: at or
above J1's 30.4% the rate stands and the run goes to the 2,000M gate; below it with the drift still
rising, the branch at 1.5e-4 from the 800M state (or from 400M, if the gap is wide enough to want the
earlier start) is the next run.

*520M, interim:* self-play has begun to punish the US hole. The US settled on France 3 / Canada 2 /
Italy 2 (+Iran 1, Philippines 1; 95%) with West Germany empty since 360M, the USSR on Poland 3 /
Yugoslavia 3 since 240M; USSR Europe-control endings 0.1–0.2% of its self-play wins through 490M, then
2.7% at 500–510M and 6.7% at 510–520M. In 200 greedy games at 520M the US controls West Germany at no
turn end in any game; the 12 USSR Europe-control wins all run through the USSR holding West Germany
(40% from turn 3) and then France (70% at turn 3, 14% in the other games). The 800M census says
whether the US comes back to West Germany -- the thing that would tell the opening churn from a seat
the rate has stopped. Events at 520M: Our Man in Tehran 0% by the US (5% at 260M); Chernobyl 0% (J1
the same until 8–10B); Nuclear Subs 24% (headline 22%).

*520M, the West Germany census* (the policy's own masked softmax; `data/reports/j5_wg_probability.txt`):
West Germany is nearly extinct in J5's US policy, and it reads as an exploration hole, not a settled
judgement. At setup P(WG) is 0.2–0.5% at each US placement (mean 0.30%, max 1.4%) and 1.5% of
setups sampled at T=1 put anything there (J5 at 300M: 88% at the first placement, 100% of sampled
setups; J1 at 520M: 98.5%). At 2,447 later US placements with WG legal and empty the mean is 0.38%
against 2.8% uniform over the legal targets -- actively avoided, never taken greedily (1.2% with 1–2
USSR influence there, 3.7% with 3+). The zero-avoiding floor at c = 0.01 holds WG at ~0.3%, so
self-play rarely puts the US there, the critic has little data on it, and the USSR's new
Europe-control punishment can reach the US only through that 1.5% of sampled setups. **This splits
the remedy.** The rate explains the drift and why the policy overshot out of West Germany; it does
not refill a hole already below the floor, so a lower-rate branch alone would carry the hole on. The
setup-scoped term at a higher coefficient (`--zero-avoid-scope setup`, B4's floor, no seeding: it
keeps openings reachable, the network still chooses) raises the floor where the hole is so the
punishment can reach it; alone it leaves the drift to open the next hole. They are complementary,
prevention and cure. **If the 800M census shows the US still out of West Germany, the recommended
next run is one branch from 800M with both** (lr 1.5e-4 and the setup-scoped term raised), read at
1,200M against J5 proper and J1's 33.5%: the question is whether J5 can be rescued, one match per
question, and attribution is a second branch only if the first works. If the US has come back on its
own, the rate alone is the remedy question, as decided.

**J5's 800M read (2026-11-06, from [`../runs.md`](../runs.md)): 33.1% ± 1.1 against A13+R48's SWA (US 28.1 /
USSR 37.1) against J1's 30.4% -- +9.7 over 400–800M (J1 +14.4), ~3 ahead at matched steps on the SWA read, and the
US seat recovered (18.0 → 28.1) without West Germany** (US France 3, Canada 2, Italy 2, Iran 1, Philippines 1 in
66%; USSR Poland 6 in 100%; P(WG) at the US setup still 0.4% per placement, 1.5% of T=1 setups). Raw, the lead
is gone: 24.6% against J1's raw 25.6% at 800M, and **head to head J1 400M beats J5 400M 58.4% (J5's US without
West Germany loses 71% to J1's USSR) while 800M against 800M is level, 49.6%.** So J5's lead over J1 is against
the reference only, and the slowdown read at 400M did not need the branch: the US seat came back on its own,
and neither the rate branch nor the setup-scoped term was launched. **Paused at 930M (owner, 2026-11-06) and
branched as `E8-A35-R102-S44@930M+R119`** -- R119 is pure win/loss (`--reward-scheme terminal
--no-blunder-window`, no held-scoring or DEFCON-1 blunder treatment), the owner's default target from
`37da536`, after R118 (`--no-held-scoring-blunder`) cost nothing on J1's line (60.6% at 14,400M against 60.4%)
and halved T=1 held-scoring losses, and R119's branch read level (59.3%); the from-scratch A8 reads are R118 16.9%
and R119 17.3% at 400M against J1's 16.0%. The 2,000M gate is read on this branch (~22:00 UTC, 2026-11-08).
Meanwhile the owner's canonical placement order is built as [P43](P43_canonical_placement_order.md) (rule bits
V1 `canonical_first` / V2 `canonical_last`, the V code for views; the free-order DAG correction assessed there),
and P42's build is merged as a view beside it.

**1,200M (2026-11-07, on the R119 continuation; stopped at 1,130M for main's ENG-9/ENG-10 rebase and resumed in
place): 36.4% ± 1.1 against A13+R48's SWA (US 33.8 / USSR 38.1)** against J1's 33.5 (29.3 / 36.8) and J1′
(`E8-A8-R103-S44`) 35.8 (30.4 / 40.2) at 1,200M. Raw head to head, 2,000 greedy games: **J5 beats J1 53.1%** (as US
56.7, as USSR 48.1), J5 against J1′ 50.4%, J1 against J1′ 44.2%; the matched-step series (400 / 800 / 1,200M): J1
against J5 58.6 / 50.3 / 46.9, J5 against J1′ 40.0 / 46.2 / 50.4 -- **J1′ led early, J5 caught up by 1,200M, J1 is the
slowest** (`data/reports/j1_j5_j1p_{400,800,1200}M.json`). Setups at 1,200M (T=1): J5's US never controls West
Germany (any WG 25.5%, WG ≥ 4 0%); J1 at 1,200M 100% WG (89.9% ≥ 4) but J1 at 13,600M also drops it (11.0% any,
0% ≥ 4); J1′ keeps it (99.9%) and puts Yugoslavia in 99.2% of USSR setups. Next reads 1,600M (~04:30 UTC) and the
2,000M gate (~22:00 UTC, 2026-11-08). P43 is built for every placement run; its launch waits on the owner's GPU and
order decision.

**J5 through the 2,000M gate and past it (2026-11-08/09, [`../runs.md`](../runs.md)), against A13+R48's SWA, J1 and
J1′ at matched steps:**

| step | J5 (A35 3×128) | J1 (A8, R102) | J1′ (A8, R103) | J5 vs J1 head to head |
|---:|---:|---:|---:|---:|
| 200M | 21.1 | 11.8 | | |
| 400M | 23.4 | 16.0 | | 41.4 |
| 800M | 33.1 | 30.4 | | 49.7 |
| 1,200M | 36.4 | 33.5 | 35.8 | 53.1 |
| 1,600M | 30.8 | 35.5 | 39.2 | 46.7 |
| 2,000M | 38.7 | 38.6 | 40.1 | 54.2 |
| 2,400M | 36.4 | 41.0 | 41.4 | 45.7 |

(J5 on R119, pure win/loss, from 930M; J1 continues 44.8 → 48.5% over 4,400–7,600M.) A 10M sweep shows J5 flat
against the reference from ~1,200M and slipping from ~1,450M, no discontinuity at any resume; its DEFCON-1 endings
in training rose 12.7 → 16.6% over 1,200–1,600M; its openings keep churning (US back to West Germany 2 at 2,000M;
USSR Poland/Romania → Bulgaria/Poland). **Reading (2026-11-10): the gate is passed by the letter (level at 2,000M)
and failed in substance** -- J5 learns the early game about twice as fast as A8 and has made no net progress since
1,200M, swinging 5–8 points between reads where J1's reads are monotone; by 2,400M it is 5 behind both A8 lines
and loses head to head. Two readings fit, and they are told apart cheaply, so **not saturation at the current
settings** (10,000M ≈ 18 days at 6.5k steps/s): (i) the step size -- per-update KL was 2.3× J1's at 400M and
rising, never lowered; non-monotone reads and opening churn are its signature; the fix is the branch deferred at
400M, `E8-A35-R102-S44@930M+R119@2000M+lr1.5e-4` from the 38.7 state, read at +400M / +800M against J5's own
continuation (36.4 at 2,400M) and J1's 41.0 / ~44; (ii) capacity at 3×128 -- a plateau at ~36–39 where A8 keeps
climbing, in which case the lower rate stabilises but does not climb and the question becomes 3×192 or more
layers (J5 was always meant to be sized by the gate). Also open: the recipe confound -- J5 switched to R119 at
930M and stalled ~300M later with DEFCON-1 endings rising; the A8 R119 control stopped at 410M, so there is no
matched A8 read on R119 past that, and J5's own curve cannot separate the recipe from the architecture. Before the
branch, three numbers the running session can pull without training: J5's per-update KL and clip fraction over
1,200–2,400M beside J1's; DEFCON-1 endings per 100M beside J1's and the A8-R119 run's; parameter counts for A35
3×128 against A8, so "capacity" has a number. What we already understand: the token transformer is the faster
learner at matched steps to ~1,200M and the human US opening is reachable for it unprompted; whether it has a
higher ceiling is unanswered, and the current run cannot answer it.

**The three numbers (running session, 2026-11-10; `training_metrics.jsonl`, the 2,400M weights):**

1. *Step size: still ~2× J1, no longer rising.* Per-update KL ×10⁻³, J5 / J1: 2.52 / 1.03 at 1,200–1,300M, 2.13 /
   0.97 at 1,600M, 1.84 / 0.92 at 2,000M, 1.81 / 0.89 at 2,300–2,400M; clip fraction 19.2 / 15.5% → 17.5 / 14.4%;
   KL to π_ref 7.1 / 3.3 → 5.2 / 2.9. The ratio peaked at ~400M (3.13 / 1.42) and J5's own KL has fallen steadily
   since 1,200M -- the stall began as the step was shrinking, so the rate is not the whole story, though J5 still
   takes twice J1's step with three more points clipped.
2. *Self-inflicted DEFCON-1 endings: J5's excess is architectural and predates R119.* Per 100M, J5 / J1: 12.7 / 11.1
   at 1,200M, peaking 16.7 / 10.4 at 1,500M, 13.5 / 8.8 at 2,300M -- J1 falls steadily from 11 to 9, J5 sits at
   13.5–14.5. At 300–400M, all three runs on their first recipe: J5 (R102) 16.2, J1 (R102) 11.1, A8 on R119 12.5;
   R119 costs an A8 about +1.4 at 400M. Provoked DEFCON-1 is also ~2 points above J1 (12.1 vs 10.5). **The recipe
   confound is cleared: the excess was there on R102 from the start.**
3. *Parameters are equal:* A35 1,619k (token transformer 792k + fusion 186k = 978k trunk) against A8 1,485k (914k
   trunk), ~9% more overall; the heads are the same size. The difference is compute (~16× per step) and inductive
   bias, not parameter count -- so "capacity" cannot mean parameters; if it means anything it means **width per
   token** (128, spread over per-type MLPs) against A8's wide fused representation.

**Revised reading.** Three candidates, in the order to test them. (a) *The DEFCON-1 excess, 5 points above J1 from
the first recipe on, is the one concrete architectural defect, and its mechanism is not yet known.* My first guess --
the global token not reaching the per-country decision -- is wrong on the weights (running session, 2,400M): both
models run `head_context`, so every per-country logit reads a 64-wide projection of the trunk hidden, which in J5
is built from the global token's output and the per-type means, and the choice to coup at all goes through the
branch and play-mode heads, which read the trunk directly in both. What can still differ is the width of the path
(DEFCON squeezed through a 64-d context beside a 128-d token, where J1's whole trunk is the global mix) or the
trunk summary itself (global output + type means) losing what a DEFCON-2 coup needs. **A no-training probe first:**
on identical positions at matched steps, P(coup in a battleground | DEFCON 2) and P(any DEFCON-suicide action | it
is legal), J5 against J1, and whether J5's excess sits in those positions; only then an A code -- the candidates are
a wider head context, or the global DEFCON float broadcast into every country token's input MLP (a network-side
routing of a float the observation already holds, not a layout change) -- read from scratch at 3×128 against J5's
own 23.4% at 400M (17 h). **The probe is requested (owner, 2026-11-10), as P41b:** one shared position set drawn
half from each lineage's games, both networks scored on it at 2,400M (400M and 1,200M if cheap); the suicide mass
split by kind (battleground coup, DEFCON-lowering event, opponent card for Ops) and the coup split by stage -- the
play-mode head's mass on a coup at all (reads the trunk directly) against the per-country head's mass on
battlegrounds given a coup (reads the 64-d context) -- with the critic's value at the same positions. An excess in
the first stage points at the trunk summary, in the second at the context path, in events at the card tokens, and
none on shared positions means J5 reaches these states more often rather than misplaying them. (b) *Width:* 3×192 as planned, ~2× slower, from scratch, the same
400M read. (c) *The rate:* the lr 1.5e-4 branch from 2,000M, read at +400M against J5's 36.4 -- cheapest, but the
KL trajectory now says it stabilises at best. Not saturation under any of them until one of the three moves the
400M or the +400M read.

## The until-control macro (proposed by the owner from J5's hole; assessed 2026-11-06, nothing built)

**Now specified as [P42](P42_until_control_macro.md)** (owner's go, 2026-11-06; spec only, the engine work still
needs the owner's approval, after the 800M decision). P42 takes form (c) at flat 304, the expansion in
`Engine::step_flat` through the engine's own steps, the macro mask only where the expansion places ≥ 2
points, setup included, "until control" only, no separate floor, exclusive with E4.1. Two owner decisions
override the note below: **the first run is a growth branch from J5** (new rows at weight 0, bias −10, a
zero second output on the country correction, an identity test before launch, `E8.1-A35-R102-S44@<step>M+…`
by precedent, read at +200 / +400 / +800M against J5 proper's continuation) rather than a from-scratch run;
and **merging card and play mode is rejected** (the event holes are single choices with no product to
collapse, the joint floor would fall from 5 modes to ~36 pairs, and E4.1's −235). The event-mode holes get a
separate proposal sized by the leak census.

**The proposal.** An influence-placement action that places in one country *until control* (or until
the play's points run out) as a single policy decision, beside the one-point action. The case, measured
on J5 at 520M (`data/reports/`): a takeover is several consistent placements, so exploring it from the
floor is a product of probabilities -- J5 starts West Germany at 1.8% of placements and continues at
24% (J1 6.1% / 65.5%; persistence in general 43.8%, J1 55.3%), so a four-point takeover is
~0.018 × 0.24³ ≈ 0.03% per opportunity (J1 ≈ 1.7%). A macro makes the plan one choice at the floor
and gives PPO one decision to credit. Four forms were considered: (a) a second "1 or until control?"
decision (doubles placement steps), (b) independent country and amount heads (cannot pair them),
(c) one softmax over 84 × 2 (country, amount) slots, flat 220 → 304, (d) the running session's
proposal -- the country softmax as now plus a per-country Bernoulli amount, P(c, ∞) = softmax(sel)_c ·
σ(amt_c), with the amount carried in the micro-action's flags and expanded by the engine.

**Assessment.**

1. *The idea is sound and the game is unchanged.* Within a play the points of one side go down with
   no opponent decision and no chance between them, and the same holds in setup, so "until control"
   is a deterministic expansion into the base one-point steps the engine already takes; a macro
   trajectory is a base trajectory. What changes is the policy class (a mixture over decompositions)
   and the exploration geometry: one floor draw instead of a product. The credit half alone was
   tried as P4's setup credit (λ = 1 inside a placement block): it fixed Poland, never West Germany,
   and froze the setup at flat temperature; the macro adds the exploration half, which is the part
   that was missing.

2. *(c), the joint softmax, is the cheapest form, not (d).* Everything downstream assumes one flat
   action per step with one categorical log-prob over one mask: the rollout buffer (`actions`
   `(T, envs)`, `log_probs` `(T, envs)`, `masks` `(T, envs, 220)`), the NashPG loss (one
   `Categorical.log_prob`), MCTS priors, the Gumbel root's per-action logits, the rollout root's KL
   rule and its top-k over the logits, the ONNX export's `("logits", "v_win", "v_vp")` and the web
   page. (d) breaks every one of them: each consumer must form the joint anyway to sample, to mask
   or to take a log-prob, so (d) saves nothing but 84 head parameters and costs a two-part action
   through the whole stack. With (c) the only change is the width, and the amount's exploration
   floor, its masking and its entropy come free from the existing term over the legal joint slots --
   no separate floor, no separate machinery. The owner's concern about the action space is about
   the *legal* choices per placement decision, which roughly double in either form; the 304 slots
   themselves cost nothing a 220-wide softmax does not already pay.

3. *Expand in the view layer, not the engine.* P23's merged view is the precedent: `Engine::step_flat`
   composes two base steps and the mask is read off the committed state; the state machine, the
   4-byte micro-action and the engine's decision stream are untouched. The macro should be the same:
   a view in which slot 220 + cid means "POINT_NODE cid, repeated until control or
   `remaining_steps` = 0", with the mask offering it where it differs from one point (not already
   controlled by us, more than one point left, one point would not already control). Then
   invariant 11 is not touched, the replayer is untouched (it emits one base action per point from an
   unordered per-play multiset, and the macro never has to decide an order), human agreement
   scoring is untouched (it already treats a play as a multiset), and a macro model's replay is its
   base steps plus the view, as E4.1's was. The only expansion the engine side needs is the loop in
   `step_flat` and the per-view mask.

4. *What it breaks, and the cost.* The width: 220 is a compiled constant in `rollout_batch.cpp`,
   `batched_search.hpp`, the action encoder, the buffer, the export and the page, and the batch
   runner pads observations per env and side but not actions, so the width becomes per view, as
   the observation set already is. The merged view (E4.1) is not in current use (A2 only, −235 Elo in
   P29) and must not be combined with this one. Datasets: a macro model's `(seed, actions)` are in
   its view and expand deterministically to base steps; the reverse is not unique, so BC from the
   human set (D1) needs a canonical rule (a run of ≥ 2 same-country points that reaches control is
   the macro) -- it matters only for D-code warm starts, which the ladder does not use. P29's lesson
   does not recur: the merged node could not see the Ops it was spending, but the macro decides at
   a POINT_NODE, where the base observation's `REMAINING_STEPS` global shows the points left. Estimate:
   several days of the running session's work across bindings, search, trainer, export, page and
   tests, all before the first run; a representation change, so the owner's spec and approval.

5. *The open questions.* "Until control or out of points", one macro, not "all remaining" (a second,
   worse macro; add later only if the first earns it). The cheapest control path is moot: one
   country, the cost per point fixed by the state. No amount floor: the joint term covers it. The
   term's interaction: legal slots at placement decisions roughly double, so the floor per slot
   halves; keep c = 0.01 and read the 60M checks (per-update KL, entropy, setup entropy) against
   J5's own, not J1's. In setup the macro applies as it does in play (West Germany 4 is one choice).

6. *Where it sits.* **After the 800M decision, never instead of it.** J5's hole is a regression under
   a too-large step (100% of setups had West Germany at 300M; J1 re-finds takeovers at 1.7% with the
   same floor), and the cheap remedies -- the rate and the setup-scoped term, a branch with no code --
   are untested. The macro is the general bet for the product-of-probabilities problem, which J1's
   plateau may or may not be limited by; the evidence is indirect (Our Man in Tehran and Chernobyl
   are single-choice holes the macro cannot touch). Spec and build it while the 800M branch runs;
   its first run is **from scratch on the then-best J5 recipe with the view**, read at 200M / 400M /
   800M against J5's own curve (21.1 / 23.4 / …) -- same architecture and recipe, the view the only
   change -- not grown from J5: identity-preserving growth is milder than distillation but is still
   a warm start, and D6/D7 is the record on those; a from-scratch 400M is 17 h. Naming by precedent:
   a view was an engine sub-version (E4.1), so **E8.1**, and the run `E8.1-A35-R<n>-S44` (owner to
   confirm the code).

## The regularisers in decided states (owner's question, 2026-11-08)

Should the entropy bonus be scaled by outcome uncertainty, so that where P(win) is near 0 or 1 -- advantages
near zero -- the loss is not the regularisers alone? The diagnosis is right: with γ = 1 and a terminal reward,
every action in a decided state has the same return, the policy-gradient term vanishes and only the
regularisers move the policy there. But the two regularisers behave differently. **The entropy bonus flattens**
(gradient on every action, toward uniform, nothing opposing it in a decided state; the visible symptom is a won
position played loosely -- forced wins taken 34–52% by J1 -- because "take the win now" has only the small
advantage the critic's risk estimate gives it, and the bonus swamps it). **The zero-avoiding term is a floor**
(gradient ≈ c/(n·π_a), on near-zero actions only): it holds every legal move at ~0.1–0.3% and does nothing to the
top of the distribution, so it cannot flatten a decided state. **The fix is already running: J1′ (R103) is R102
with the bonus off and the floor kept, and it is ahead of J1 at matched steps** (35.8 against 33.5 on the
reference at 1,200M; J1 44.2% head to head). That is the standard in the game-AI line too (AlphaZero, DeepNash
and Jaxcalibur run no entropy bonus; Jaxcalibur uses the zero-avoiding term in exactly this role; SAC's automatic
temperature is a global target, not per state). **So: drop the bonus rather than scale it** -- J5's recipe
(R102 → R119) still carries it at 0.01, and its next branch should take R103's `--entropy-coef 0` (owner's call,
nothing launched). Scaling the *floor* by uncertainty is not recommended: it costs little where it stands, and
weighting it by V(1−V) would remove exploration exactly where the critic is confident, including where it is
confidently wrong -- a sampled off-policy action that changes the result is the only signal that a "decided"
state was not. **A no-training read first**, on J1, J1′ and J5 at matched steps, decisions binned by |V|: policy
entropy per bin (J1 against J1′ at |V| > 0.9 shows the bonus's flattening directly), the advantage scale per bin,
forced wins taken per recipe, and the swindle rate (eventual winner with V < −0.8 at some point). If J1′'s
decided-state play is sharp and swindles are rare, nothing beyond R103 is warranted; the uncertainty-scaled bonus
is an arm only if J1′ still plays flat there.

## Targeted exploiters: by win condition (owner's question, 2026-11-09)

**The idea.** The exploiters so far (R106–R109: a seat-locked learner, warm-started from the main's own weights,
against the frozen greedy main, the floor on setup only) found +4 to +5.5 per seat and told us *where* only by
inspection (post-setup play, Europe uncontested). A **condition-targeted exploiter** is the same recipe with a
restricted payoff: +1 for a win by condition X, 0 for any other win, −1 for a loss. Its margin against the main's
own self-play rate of X-wins in that seat is the main's exploitability *along X*, and the set over conditions --
Europe control, the VP track (±20), Wargames, final scoring, an induced DEFCON loss, an induced held-scoring loss --
is a **strategic exploitability profile**, the ending-type analogue of the searched-leak census. The dual, a
**defender**, is the zero-sum payoff plus an extra −1 for losing by X: an opponent that still plays to win but
denies X; added to the main's pool it suppresses the main's X-wins and forces the other routes.

**Antecedents.** Nothing uses win conditions as the target, but each half has a name. A restricted payoff
best response is Johanson et al.'s restricted Nash response (NIPS 2007: a counter-strategy to a fixed model,
exploitation traded against exploitability by a parameter p; data-biased responses after it). Diversifying
best responses by a behavioural descriptor is the behavioural-diversity PSRO line (Perez-Nieves et al. ICML 2021;
Liu et al. 2021 unifying behavioural and response diversity; PSD-PSRO 2023) -- here the descriptor would be the
ending type, chosen by hand. AlphaStar's league is the training use: main exploiters find the main agent's
weaknesses, league exploiters the league's, and the main agents are conditioned on a strategy statistic z with a
pseudo-reward for following it, which is a hand-defined target keeping the league diverse. Rectified Nash
(Balduzzi et al. ICML 2019) trains responses only against opponents one beats, the "target weaknesses" variant.
The caution is the adversarial-Go line: a general exploiter found the cyclic attack (Wang et al. 2023, >97%
against KataGo), and iterated adversarial training made the victim robust to each adversary it had seen and to
no new one (Tseng et al. 2024) -- a defender in the pool fixes exactly X, and the profile has to be re-taken after.

**Assessment.** Worth doing as a *diagnostic* now, as *training* only on what the profile shows.

* The profile is cheap and interpretable: A8 warm starts of ~300M (R104 stopped at +310M) are under an hour each
  on an idle GPU, so six conditions in both seats are a day; each margin is read against the control's own
  self-play rate of X in that seat (the per-seat rule), beside the general exploiter's +5.5 so the restricted
  payoff's overstatement is visible (a targeted exploiter sacrifices every other route, so its X-margin is an
  upper bound on what a real opponent takes through X).
* Rare conditions are reachable from the warm start (Europe control is 1–7% of the main's own wins; −1 for a loss
  keeps the exploiter from throwing games to chase X), so no shaped progress reward is needed -- the exploiter is a
  probe, not the product, and its payoff is the only engineering.
* The induced-loss conditions (DEFCON, held scoring) are the robustness probes in the Go sense: an opponent that
  learns to make the main blunder, found by training, which the leak census cannot see.
* Training on it is B2's league with condition-targeted members, deferred by the owner's order; the profile is
  what would justify it: a condition with a margin well above the general exploiter's share tells the main what
  it cannot defend, and the defender for that condition is the pool member to add.

**First profile:** the frozen target is J5's 2,000M gate checkpoint (or J1′ at matched steps, whichever the gate
chooses); Europe control and Wargames first (the endings where J5 and J1 differ most: 7.5 vs 0.9% Wargames, 1.3 vs
7.0% Europe at 180M), both seats, then the induced DEFCON loss; read beside R108's general margin. Nothing launched;
the owner's call after the gate.

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
