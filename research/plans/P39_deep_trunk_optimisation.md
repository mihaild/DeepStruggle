# P39 — training the deep trunk: the optimisation knobs not yet tried, in the owner's order (2026-10-26)

**Goal (owner).** More capacity eventually; deep trunks are not abandoned on the ladder's reads so far.
Every deep-trunk failure in P38 was an optimisation symptom -- the first-60M step size, rank collapse,
dead head units -- never a capacity result, and each fix exposed the next. The standard toolbox for
training deeper networks is mostly untouched here. This plan applies it, one knob at a time, to the
deep test-bed that already keeps its step and its rank, and judges by the amended P38 rule.

**Priorities (owner):** 1 learning-rate warmup, 2 warm start, 3 trust region, 4 µP, 5 other
parameters, 6 in-place growth.

## The test-bed, the control, the rule

* **Test-bed:** the sphere trunk on the 16k minibatch with the heads' input at RMS 0.35 (`A30` on `R86`):
  the first deep trunk that keeps both its per-update step (KL 0.023 against the control's 0.016) and
  its rank (2.0–2.3× fresh at 400M), with its heads alive; behind the control at every matched read
  (gap 7.0 at 1,600M, 8.1 at 2,000M).
* **Control:** `E8-A8-R1-S44`, running long; A8's own trajectory is the yardstick at matched steps.
* **The rule** (P38, amended by the owner 2026-10-25): a deep rung passes when its gap to the control
  **decreases** between 1,600M and 2,000M with the checks holding; a pass is extended in place and read
  at 400M windows until its plateau is above or below the control's. Checks:
  [`../method/network_stability.md`](../method/network_stability.md) (step size at 60M, rank ≥ 2× fresh,
  head norms and dormant units, plasticity at 400M).
* **Into the recipe now, independent of the arms:** a LayerNorm in the per-entity, branch and play-mode
  heads (`--ladder-head-norm`, A31's change): on R48's plateau it revived the dead head units (12.2% →
  0.7%), raised the head-input rank (90 → 102) and reads 51.0% against the plateau's 47.2–49.6%. On the
  sphere trunk the same norm cost the trunk its rank (A32), so it enters the deep arms only after a rung
  passes without it; on A8-width runs it is the recipe from here.

Codes at launch; each arm is one change from the test-bed; two arms at a time beside the control if
memory allows, else one.

## 1. Learning-rate warmup (build: small; highest prior)

* **Why.** No warmup exists in the trainer. Every deep-trunk failure appeared inside the first 60M,
  where Adam's early steps are largest and the per-update KL ran 2.6–3.6× the control's; a linear
  warmup is the standard remedy for exactly that window.
* **Arms.** Linear warmup from 0 to the run's rate over 50M steps, then constant (one arm); 100M (a
  second if the first changes the 60M step size but not the 400M read).
* **Read.** The 60M per-update KL and clip fraction (should sit at the control's without the 16k
  batch's help -- also run one cell on the 4,096 minibatch to see whether warmup alone replaces 16k);
  then the matched reads to 2,000M.

## 2. Warm start by offline distillation from the control (build: moderate)

* **Why.** The deep trunk's problem is a slow start, not a ceiling it has reached. Distil the A8
  control's policy *and* value into a deep sphere network on its own self-play, then continue with RL:
  the deep network starts at the control's level with its plasticity intact and the extra capacity
  unused. The earlier wash-outs do not apply: human BC was a weak prior, and P36c's from-scratch
  students failed on 180k positions, not on tens of millions.
* **Data.** 50M positions from the control's self-play at temperature 0.25 (free; `tools/value_labels.py
  positions` at scale), targets the control's full policy distribution and its value, held out by game.
* **Fit.** `--mode distill` / `policy-fit` extended to train the whole network (trunk and all heads,
  value included) from fresh weights of the deep shape; stop when held-out KL to the teacher stops
  falling. Read before any RL: agreement with the control's argmax on held-out positions (target ≥ 95%),
  head-to-head against the control greedy (level), the plasticity probe (a fresh network's numbers).
* **The arm.** RL from the distilled weights on the test-bed's recipe, read against the control's
  trajectory from the control's step that the distilled network matches in strength (its "virtual
  step"), then at 400M windows. **Pass:** above the control at matched strength-step within 800M of RL.
* **Risk.** A distilled policy is sharper than an RL one; watch entropy and the per-update KL at the
  start, and sample the distillation targets at temperature 1 rather than greedy.

## 3. The trust region (build: none; flags exist)

* **Why.** PPO's clip range, the epochs per iteration and `--target-kl` set the per-update movement
  directly; the 16k minibatch helped by quartering the updates per iteration, the same lever by another
  name.
* **Arms,** on the test-bed, one each: clip 0.1 (from 0.2); `--target-kl` at the control's measured
  per-update KL (0.016) with early stopping of the epochs; epochs halved.
* **Read.** The 60M step size, then the matched reads. A cell that matches the control's step on the
  4,096 minibatch is read there too (it would free the recipe from 16k for deep trunks).

## 4. µP proper (build: moderate)

* **Why.** The ladder scaled the global learning rate; µP scales the initialisation variance, the
  output multipliers and the per-layer rate with width, and depth-µP the residual branch by 1/√L. The
  sphere's α at 1/(L+1) is the depth half; the width half was never applied, and "halving the rate cut
  the step 13%" is what a globally scaled rate does when the step is set per layer.
* **Arm.** The test-bed in µP: per-layer Adam parameter groups with rates scaled by fan-in relative to
  A8's widths, output layers zero-initialised with a 1/width multiplier, the base rates tuned on A8
  (the control's) and transferred. Then the same at twice the width as the transfer test: if the 60M
  step size and the 400M read match without retuning, transfer holds, which is the property a cloud run
  would need.
* **Read.** Step size at 60M across widths; matched reads.

## 5. Other parameters (build: small)

* Adam β₂ 0.95 (from 0.999) and ε 1e-6 (from the default; the sphere's unit-norm rows give tiny
  gradient entries).
* `--lr-schedule cosine` to 3e-5 over the run (exists, never used on the ladder; SimbaV2's ablation put
  the loss of its decay at the size of its categorical critic's).
* `--vf-coef` 0.25 and 1.0 (from 0.5): the shared trunk's value-against-policy balance.
* A lower rate on the policy heads than the trunk (a parameter group): the heads saturate first and
  drag the trunk.
* Each one arm on the test-bed, 60M checks then 400M; only a cell that moves the 400M read goes on.

## 6. In-place growth (build: small; the stronger form of the earlier grown reads)

* **Why.** Every grown read so far (A13 gained; A14, A18, Cell A level) grew capacity under saturated
  heads, which P37a showed cannot pass a new gradient on. The untried forms grow where the heads are
  still plastic, or reset them.
* **Arms.** (a) The control at ~2,000M (its head fits random labels at 47%, its trunk at 98%) grown in
  place with A26's width and two sphere blocks at identity (α = 0), continued on 16k, read against the
  control's own continuing trajectory at matched steps. (b) The control at its plateau grown the same
  way with the head LayerNorm and a head reset, read against the plateau's reads. **Pass:** above the
  control at matched steps (a) or above the plateau band (b) by the per-seat rule.
* If (a) passes, the upper rungs (attention, memory) are grown the same way onto the same checkpoint.

## Sequencing

| order | arm | why here |
|:---|:---|:---|
| now | 1 (warmup 50M) and 3a (clip 0.1) | the two cheapest knobs against the one symptom every deep failure shared |
| next | 2 (distillation warm start) built meanwhile; its RL arm when the fit reads level with the control | the only way to test depth's ceiling without paying its slow start |
| then | 4 (µP) and the pass of 1/3 combined | transfer is what a bigger run or a cloud run needs |
| then | 5, one at a time, cosine first | tuning, not structure |
| in parallel when a slot is free | 6a (grow at 2,000M) | independent of the deep arms; uses the control's existing checkpoint |

The head LayerNorm goes into every new A8-width launch from now. Nothing here changes the running
control.

## Decision rules

* An arm **passes** by the amended P38 rule (gap to the control decreasing at 2,000M, checks holding)
  and is extended in place; a pass becomes the new test-bed for the next knob.
* Arms 1 and 3 are read first at the 60M step size: a cell that puts the deep trunk at the control's
  step on the 4,096 minibatch replaces 16k in the deep recipe.
* Arm 2's fit is read before any RL; a fit below 95% argmax agreement or below level greedy is a fit
  problem, not an RL result.
* Arm 4 passes as *transfer* only if the twice-width network's 60M and 400M numbers match without
  retuning.
* **Depth is abandoned** only if, after arms 1–4, no deep trunk's gap to the control decreases by 2,000M
  and the distilled deep network does not improve on the control within 800M of RL. That would be a
  finding about this loss on this game, recorded as such; until then "learns slower" is a start, not
  a verdict.

## Code (built 2026-10-27/28; nothing launched beyond the runs below)

Every arm is one flag set on the test-bed's command (A30 on R86: `--ladder-entity-proj-dim 512 --ladder-hidden-dim 768
--ladder-res-blocks 0 --ladder-sphere-blocks 2 --ladder-sphere-expansion 4 --ladder-sphere-out-scale 0.35 --lr 0.0002
--batch-size 16384`, the A8 flags otherwise). Flags are in [`../../tools/README.md`](../../tools/README.md) §1.

| arm | flags | code |
|:---|:---|:---|
| 1 warmup | `--lr-warmup-steps 50000000` | `ai/training/schedule.py` `warmup_factor` |
| 3 trust region | `--clip-eps 0.1`; `--target-kl 0.016`; `--ppo-epochs 2` | `ai/training/nash_pg.py` |
| 4 muP | `--mup` (base widths `--mup-base-proj-dim 256 --mup-base-hidden-dim 480`, A8's) | `ai/models/mup.py`, per-parameter groups in `make_optimizer` |
| 5 other parameters | `--adam-beta2 0.95`, `--adam-eps 1e-6`, `--lr-schedule cosine --lr-min 3e-5 --lr-schedule-span <steps>`, `--vf-coef 0.25` / `1.0`, `--policy-head-lr-mult 0.5` | `make_optimizer`; the schedule scales every group by its `lr_mult` |
| 2 distillation warm start | `--mode online-distill --distill-teacher <control .pt>` with the deep shape's flags, then the RL arm with `--warmup-checkpoint <run>/best.pt` | `ai/training/online_distill.py`: the teacher's self-play consumed as played (50M stored positions would be ~750 GB), held-out KL / argmax agreement / value MSE, early stop |
| 6 in-place growth | `--resume <control>:<steps> --resume-widen --resume-grow-sphere` with `--ladder-entity-proj-dim 512 --ladder-sphere-blocks 2 --ladder-sphere-carry-norm` (6b adds `--ladder-head-norm --resume-reset-policy-head`) | `ai/models/hyperspherical.py` `carry_norm` (the output keeps the stem's norm: the identity at alpha 0), the generic trainer's resume path |

**Arm 6's width.** The fusion layer is followed by a LayerNorm, so it can be widened without changing the function only
by an integer factor (each unit copied k times keeps the norm's statistics): A8's fusion of 480 grows to 960, not to
A26's 768. What is built grows the projections 256 -> 512 (2x, `--resume-widen`) and inserts the sphere blocks at the
identity, leaving the fusion at 480; a 2x fusion (960) would be a further, function-preserving build. The sphere
trunk inserted is the carry-norm variant -- a new architecture code, not A30 (whose output RMS is fixed at 0.35).

Each launch: derive from the test-bed run with `tools/scripts/launch_flags.py E8-A30-R86-S44_20261026_022112 --diff
<new run>`, register the code (`run_codes.py --update`; codes after main's), checks at 60M (step size, rank >= 2x
fresh) and reads beside the control and the A8-width control at the clip.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

| arm | run | result |
|:---|:---|:---|
| 1 warmup 50M | `E8-A30-R89-S44` | the step returns with the rate (per-update KL 0.021 at 50-60M, A30 0.023); 5.1% at 400M (A30 8.3%); stopped at 408M |
| 3a clip 0.1 | `E8-A30-R90-S44` | the step bounded (KL 0.008) with the rank growing; 23.3% at 2,000M, 32.1% at 4,000M -- ~3.4 behind the A8-width control at the same clip (`E8-A8-R93-S44`, 36.3% at 4,000M), about level with the plain control; undecided, a 4,800M read owed |
| 3b target KL 0.016 | `E8-A30-R91-S44` | rank 1.2x fresh at 60M; stopped at the check |
| 3c epochs 2 | `E8-A30-R92-S44` | rank below fresh at 60M; stopped at the check |
