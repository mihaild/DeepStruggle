# P40 — capacity or signal: two arms that separate them (2026-10-29)

**Question (owner).** The plateau is not the readout's plasticity (a head reset recovers only to the
band), and the self-play signal at the plateau is mostly about the lineage's own habits (the R48 line
beats its own past by ~+34 Elo while gaining ~+1 against the fixed reference; the A8 control's two
gains agree, +18 and +15). Where should the next investment go: training mode for the shallow net, a
bigger net, or the training signal?

**What the record already decides.** Training mode on the shallow net is closed as a route to a step
change: the step-down, the minibatch, the clip, the decay, the head norm, the head reset and the gap
penalty are each worth +20 to +45 once, then the same plateau shape. Capacity and signal remain
confounded in every result: a bigger net on the same signal (the deep sphere trunk) tracks the
control's slope at a constant 3-point deficit, which is what either a binding signal or depth's slow
start would show; a stronger signal on the same net has been run only as the exploiter league, which
was level at the plateau.

**The signal problem in one example (owner).** The network opens with three influence in
Czechoslovakia where three in Yugoslavia is clearly better, and is not punished. In self-play a
mistake is punished only if the opponent exploits it, and the opponent is the network itself; a
fault no opponent makes costly has zero gradient at any network size. The leak census found the
same from the other side: once the judge was the network's own continuation, the measured leaks
vanished. "More diverse training data" means opponents that make such faults cost.

## The two arms (the two slots, once the current runs finish)

### Arm C — capacity with the signal held fixed: the distillation warm start (P39 arm 2, built)

* A deep network (the sphere test-bed's shape, `A30`) trained by `--mode online-distill` to the A8
  control's policy and value on the control's own self-play (`ai/training/online_distill.py`), to the
  fit's bar: held-out argmax agreement ≥ 95% with the control, level with it greedy, the plasticity
  probe at a fresh network's numbers. Then RL from those weights on the test-bed's recipe (16k, clip
  0.1, the control's pool), read against the control's trajectory from the control's step that the
  distilled network matches in strength, then at 400M windows against the fixed reference
  (A13+R48's SWA) with the own-past gain tracked beside it.
* **Reads it answers.** Above A8's plateau within 800M of RL → capacity was binding: the self-play
  signal still had something a bigger function could use. Plateaus where A8 does → the signal is
  exhausted for this policy class regardless of size.

### Arm S — signal with the architecture held fixed: the rollout root as the opponent

* A8 on the control's recipe with the clip and the head norm, where in a share of training games the
  **opponent's** side is played by the rollout root on a lagged snapshot of the learner -- the root as
  the punisher, not the teacher (no CE target, no root-played learner moves; the learner's own gradient
  is PPO on outcomes against a stronger opponent, the active form the tandem result asks for). This
  is the one stronger opponent that exists on E8: every E8 network is one lineage and the E7
  references cannot enter a pool (feature bits).
* **Settings.** One game in eight with the searched opponent (the pool's remaining games unchanged);
  a cheap root, k 4, 4 keyed worlds, horizon 2, `earlier`+own (4 worlds were +90 at play time), so the
  cost stays near 3× a plain step; the snapshot the root searches refreshed every 50M; the searched
  games' rows enter the value loss and the learner's PPO as any pool game's do. Build: the opponent
  side of `ai/training/active_search.py`'s mechanism (the root already plays batched entry decisions;
  here on the non-learner seat of a marked share of envs), one flag.
* **Reads it answers.** Read against the fixed reference at matched steps beside the control, with
  the own-past gain and the per-seat rates. The reference gain tracking the own-past gain, and above
  the control's → the plateau was self-exploitation of a single lineage and the searched opponent
  supplies the missing punishment. Level → the root's horizon does not reach the faults that matter,
  or the shallow net cannot take the lesson (the league's result); then Arm C's read says which.
* **The two-lineage cross-pool** (two seeds from scratch, each pool half the other's snapshots) is the
  weaker form of this arm -- a second seed is not much stronger than the first -- and runs only if Arm
  S is level, to test lineage-specificity on its own.

## The outcome matrix (the decision rule)

| Arm C (deep warm start) | Arm S (searched opponent) | where the next investment goes |
|:---|:---|:---|
| passes | null | **capacity**: the from-scratch flagship on the deep shape, P39's knobs |
| null | passes | **signal**: opponents -- the searched opponent at a larger share, then external play (ITSL) |
| passes | passes | both: the deep shape trained against the searched opponent |
| null | null | the self-play signal is at its information limit for this loop; the remaining signal is external (human games), and the architecture question waits for it |

"Passes" is the per-seat rule against the control at matched steps, read on the fixed reference, with
the own-past gain not outrunning it by more than its noise.

## Checks

Both arms carry the stability note's checks (step size at 60M, rank, head norms and dormant units) and
the plasticity probe at 400M; Arm C's distilled network is probed before any RL. Arm S logs the share
of searched games, the root's departures on the opponent side and the learner's win rate against the
searched opponent beside the pool's.

## The Czechoslovakia read (`P40a`, CPU)

Before Arm S's result: the three-in-Czechoslovakia against three-in-Yugoslavia opening priced by the
paired-playout judge under (i) the network's own continuation, (ii) the rollout root on both sides,
(iii) the oracle root. If (i) prices them level and (ii)/(iii) do not, the fault is invisible to the
self-play signal and visible to the searched opponent, which is Arm S's premise measured directly on
the owner's example.

## Sequencing

Once the current runs finish: Arm C's fit first (hours), then its RL in one slot; Arm S's build
meanwhile, then its run in the other slot. `P40a` on CPU at any time. The long control continues; no
other training arm until the matrix reads.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

**Arm S's opponent, priced (2026-10-29).** The rollout root at the plan's settings (k 4, 4 keyed worlds, horizon 2,
earlier+own) on the A8 control's 9,920-10,000M SWA against the same network raw, 1,000 deals a side
(`data/reports/arm_s_root_k4w4h2_p*_vs_raw.json`): at every decision with two or more legal moves -- what `rollout:`
searches, not only the entry decisions (corrected below) -- **66.8% ± 1.1** (US 64.6 / USSR 68.7, ~+122
Elo); at a random 20% of them **55.2% ± 1.1** (US 50.9 / USSR 58.7, ~+36) -- 30% of the full gain, as on E7 (25%
searched kept 35%). Cost in training: the root is called at almost every rollout step (one game in eight is ~64 envs,
about six of them at an opponent entry decision on most steps) at ~0.04-0.06 s a call against a plain step's ~1.8 ms,
so the full root costs ~10-15x a plain iteration and 20% ~3.5x. At +36 the 20% opponent is about the learner's own
level (the learner beats its pool ~63%), too weak a punisher to test the arm's premise; the full root (+122) is, at
its price, unless the batch runner lets a searched game wait so the root batches across steps.

**A greedy opponent instead (owner's question, 2026-10-29).** Training games sample at temperature 1
(`DEFAULT_ROLLOUT_TEMPS`); the same SWA at temperature 0 against itself at 1, 1,000 deals a side
(`data/reports/control_t0_vs_t1.json`): **54.5% ± 1.1** (US 49.8 / USSR 58.0), **~+31 Elo** -- about the 20% root's
edge and a quarter of the full root's. Free for pool games (`--opponent-temperature 0`, built since P24), x2 in learner
rows for a greedy seat in self-play games; a cheap but mild punisher, not the strong opponent the arm's premise needs.

**The fastest strong combination (owner, 2026-10-29: "try to find faster combination for Arm S").** All on the same
SWA against itself raw (greedy), 1,000 deals a side (`data/reports/arm_s_root_k*_vs_raw.json`,
`arm_s_entry_k441_vs_raw.json`); cost per searched position from `arm_s_grid_cost.json` (marginal, in a large batch):

| root | where it searches | score | Elo | ms / position |
|:---|:---|:---|:---|:---|
| k4 w4 h2 (the plan's) | every decision | 66.8% ± 1.1 | +122 | 0.73 |
| **k4 w4 h1** | every decision | 64.7% ± 1.1 | +105 | 0.58 |
| k2 w2 h1 | every decision | 58.0% ± 1.1 | +56 | -- |
| k4 w2 h2 | every decision | 54.9% ± 1.1 | +34 | 0.39 |
| **k4 w4 h1** | **entry decisions only** (`entry:rollout:`) | **60.9% ± 1.1** (US 56.7 / USSR 64.3) | **+77** | 0.58 |
| greedy, no root | -- | 54.5% ± 1.1 | +31 | 0 |

Two worlds lose almost everything (w2: no better than greedy); the second horizon is worth ~17 Elo at ~25% more per
position. The pricing above assumed entry decisions while the strength it quoted searched every decision, which is
~3x as many positions: searched at entry decisions only, k4 w4 h1 keeps 73% of its every-decision gain, ~+108 over
the learner's own temperature-1 play. A call's fixed cost, not the positions, dominates: k4 w4 h1 takes 27 / 29 / 50 /
121 ms for 8 / 16 / 32 / 64 entry positions eager and **19 / 17 / 27 / 38 ms with the root's CUDA graphs** (`:graphs`;
measured beside two trainers). So the opponent is **k4 w4 h1 at entry decisions, graphed, with hold-and-batch**: a
searched game waits at its decision (the env holds it in place) until 32 are queued or the oldest has waited 8 steps,
then one call decides them all; at one game in eight (~8 opponent entry decisions a step) that is ~7 ms a step on
top of a plain step's ~1.8 ms, ~2.5x a plain iteration, against ~10x for a call every step. Built
(`ai/training/searched_opponent.py`, `--searched-opponent-spec 4:4:1:z2:earlier:own:keyed:aligned:box4:graphs`;
tools/README.md); a 128-env smoke run from scratch held ~half of the searched envs' opponent rows at 16 searched
envs, a share that falls with more of them (64 at 512 envs).

**Arm C's fit, done (2026-10-30).** `--mode online-distill` at 50M positions and a constant 3e-4 (E8-A30-R96-S44) underfit
at 86.0% held-out agreement; to 400M positions with a cosine to 5% (E8-A30-R97-S44, 61 min) it reached **92.3%** and
flattened -- below the 95% bar, but the bar's purpose holds: **level with the teacher greedy, 50.4% ± 1.1**, and a fresh
network's plasticity (random labels under its own head, trunk frozen, 90.8% against the teacher's 32.0%; heads 2.5-6.5x a
fresh head's norm against 12-33x). The missing agreement is near-tie decisions that cost no games. RL from it on R90
runs as **E8-A30-R90-S44-2** (a warm start is bookkeeping in the run grammar, so it is R90's second run), read every 400M
against A13+R48's SWA beside the control's 40.6% at 12,000M.

**Arm C's first RL lost its start (2026-10-30).** On R90 unchanged (lr 2e-4, entropy coefficient 0.01) the distilled A30
fell from ~40% to **22.4%** against A13+R48's SWA by 400M: entropy 0.26 -> 0.93 within 34M steps, per-update KL 2-8x the
control's at its plateau. Stopped at 412M -- at 22% it could not reach the control's 40.6% by 800M, the matrix's rule, so
it answers nothing about capacity. The plasticity that made it a good start is what undid it: the teacher's saturated
heads (12-33x a fresh head's norm) hold entropy 0.29 against the same bonus; the distilled heads (2.5-6.5x) do not.
60M probes from the same start, against the teacher greedy: entropy coefficient 0.002 held entropy (~0.37) but larger
steps cost strength (35.3%); lr 5e-5 kept steps small (KL 0.003) while entropy still crept (0.59; 42.8%). Both together
next, then Arm C's RL on whichever holds the start.

**Every arm from the bar (owner, 2026-10-30).** The working recipes all stop in one band against A13+R48's SWA -- R48
48-52%, R94 52.8%, Arm C's second attempt 48.1% at 400M (R99: lr 5e-5, entropy 0.002, which held the distilled start),
Arm S 51.6% at 16,400M -- each ahead of its own control, none past the best E8 network. So arms start from R94 at
16,000M (52.8%) and are judged against its band (49.4-53.3%): Arm S already branches it; Arm C is redistilled from its
SWA (E8-A30-R100-S44) and the A8-control version stopped at 733M; from-scratch arms use A8-on-R95's recipe (16k, clip
0.1, zeroed readouts: 35.8% at 2,000M against 27.7%).

