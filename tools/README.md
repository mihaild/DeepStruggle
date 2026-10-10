# Twilight Struggle AI CLI Tools (`tools/`)

Standalone developer CLIs for training, tournament benchmarking, match simulation, replay
generation, demonstration dataset creation, and checkpoint inspection. Never invoke the training,
tournament or match machinery from an ad-hoc script -- go through these.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning (NashPG) or supervised demonstration warmup with
live snapshot tournament evaluation.

```bash
# RL training run with blunder-aware rewards, snapshotting every 10M env steps
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --warmup-checkpoint <warmup.pt> \
  --train-steps 160000000 \
  --snapshot-every-steps 10000000 \
  --reward-scheme blunder_aware \
  --num-envs 512 \
  --eval-games-per-side 50 \
  --eval-opponents random heuristic \
  --output-dir <run-dir>
```

### Budgeting a run

**Every budget is in env steps. There is no wall-clock flag.** A time budget cannot make two
arms comparable, because steps/sec depends on the policy: the arm whose games run longer has
costlier evaluations and so gets less training, which biases the comparison in a fixed
direction rather than a random one.

`--train-steps` is the budget. `--snapshot-every-steps` (default 10M) sets how often a snapshot is
saved and evaluated, and `--pool-every-steps` (default 5M) sets how often the self-play opponent
pool grows. Give an A/B's two arms the same `--pool-every-steps`: an arm whose pool grows more
slowly trains against a smaller, staler pool. The two used to be one flag, derived from two time
flags, and E3-22-28's first attempt thereby snapshotted every 26.7M steps against its baseline's
~5M: at 45M steps it had 2 pool opponents where the baseline had 9, and with
`--opponent-frac 0.3` that made it a two-factor experiment. It was thrown away.

**The snapshot interval is a reporting setting only.** Snapshot evaluation restores the global
random streams it draws from (`preserves_training_rng`), and the pool has its own schedule, so the
same run evaluated every 1M or every 2M trains bit for bit the same: every logged training metric
and the final weights were checked equal over 5M steps (`research/log/P26_quick_screen.md`). A
pool member that falls between snapshots is written as `pool_<N>steps.pt`. A resume finds these
files the way it finds snapshots. Tournaments, which read `snapshot_*`, ignore them. Snapshots
moved from 5M to 10M on 2026-09-24, because at 5M evaluation cost ~17% of a run's wall time.

TF32 matmuls are on by default since 2026-09-24 (P26): +15% steps/s on M2d, solo or paired, and no strength cost in a 3-seed A/B (`research/log/P26_quick_screen.md`). `--no-tf32` gives fp32, which is what every run before E4-57 used. The setting is recorded in `metadata.json` as `tf32`.

`--block-lambda {off,setup,setup-side,same-side}` sets where GAE uses λ = 1 inside a decision
block (P4, `research/log/P4_setup_arms.md`). **The default is `off`** (again since 2026-09-27).
`setup-side` -- λ = 1 between consecutive setup placements by the same player -- was the default
for one day: at 80M under the old sharpened rollout bands it fixed the USSR's Poland and was +39,
but at flat rollout temperature 1.0 it added no strength (−73 / +30 against E5-11) and froze
whatever opening it had found, including a degenerate one
(`research/log/E5_12_setup_credit_at_flat_temperature.md`). `same-side` chains every same-mover
step with no chance node between them, and it was −289. Runs from E5-12 used `setup-side`;
pass it to reproduce them.

`--setup-entropy-floor <nats>` (off at 0) puts an adaptive entropy floor on the learner's setup
placements. A setup placement at p ≈ 1 is never resampled, so its opening locks whether or not it
is any good (E5-11-43 opens Greece 2 in every game, 5 points worse for its own US than a sane
opening: `research/log/E5_11_setup_lock_and_critic_views.md`). The extra bonus applies to setup
decisions only. Its coefficient moves by `--setup-entropy-lr` × (floor − the rollout's setup
entropy) each iteration within [0, `--setup-entropy-max-coef`], so it is zero until the opening
starts to lock. It is logged as `entropy_setup` (always) and `setup_ent_coef`, and restored on a
resume. `tools/scripts/setup_oracle.py` measures whether an opening is actually worse.

`--forced-opening <name>` (off by default) starts every training game after a scripted setup from
`tools/lib/openings.py`. For `human` that is USSR East Germany 1, Poland 4, Yugoslavia 1 and US
West Germany 4, Italy 3, Iran 2. The learner never makes a setup decision, so its setup is
untrained. Rate such a checkpoint with the same opening: `tools/tournament.py --opening human`
applies it to every agent, and an `opening:<name>:<spec>` model spec applies it to one agent only.
The run's own snapshot match evaluations use the opening automatically; its probes do not.

`--setup-mc-credit` (off by default) credits the learner's setup placements with the **game
result** (advantage = result − V(s), the critic only as a baseline) instead of the λ-return. With
γ 1 and λ 0.98, the λ-return reaches the setup only through the critic's values of the positions
just after it, and the critic over-rates unfamiliar openings three to four times
(`research/log/E5_11_setup_lock_and_critic_views.md`). This is how Ataraxos trains its setup. A
game outlasts a rollout, so each placement waits until its game ends. It is then trained in
batches of at least `--setup-mc-min-batch` (512) with the PPO clip against the log-prob it was
sampled with. Setup rows leave the ordinary surrogate, but keep the entropy bonus, the floor and
the KL. Pair it with `--setup-entropy-floor`, because a placement at p ≈ 1 is never compared with
anything. From scratch the result is a slow teacher: setup entropy stays near uniform for millions
of steps.

`--lr-schedule {constant,step,cosine}` and `--ema-weights <tau>` (P28 step 2, both off by default)
are the optimiser levers at the plateau (`research/plans/P28_strength_on_E6.md`).
* **The schedule** counts env steps from `--lr-schedule-start`, which by default is where the run
  starts, so a schedule switched on at a resume point leaves the steps already trained alone.
  * `step` lowers `--lr` to each of `--lr-schedule-values` (1e-4 3e-5) in turn, every
    `--lr-schedule-every` (60M) steps.
  * `cosine` decays it to `--lr-min` (3e-5) over `--lr-schedule-span` (200M).
  * The rate each iteration ran at is logged as `lr`.
* **`--ema-weights`** keeps an exponential moving average of the weights with a time constant of
  `tau` env steps.
  * Snapshots, `snapshot_final.pt`, pool members and the run's own evaluations all use the
    average, so the rated and pooled model is the average.
  * The live weights keep training, and the resume state holds both.

`--aux-ownership <w>` and `--aux-vp-margin <w>` (P29 bet 2; 0, the default, is off) add two
small heads on the trunk. They are trained from each game's end:
* **ownership:** who controls each of the 84 countries (mine / opponent's / neither, in the mover's
  frame; cross-entropy);
* **VP margin:** the final VP margin (VP/20; squared error).

How they are trained:
* A game outlasts a rollout, so `--aux-sample-frac` (0.1) of decisions wait per env, at half
  precision, until their game ends. `TsVectorizedEnv.record_final_control` reads control just
  before the auto-reset.
* Once `--aux-min-batch` (4096) labelled positions are ready, they train in their own optimizer
  step, as the setup credit does.
* The heads are used only by that loss. `forward()` and the policy are untouched, and checkpoints
  record the heads by their weights (`aux_own_head.*`), so they load anywhere.
* Metrics: `aux_own_loss`, `aux_own_acc`, `aux_vp_loss`, `aux_n`, `aux_pending`.
`--play-mode-temp T` (owner, 2026-10-02; default 1.0 = off) samples the learner's play-mode decisions -- event, space
or Ops for a chosen card -- at temperature `T`, so events the policy rarely plays get tried (the expert review's first
candidate, `research/log/expert_review_E7.md`). Exploration only: the stored log-prob stays the policy's own. The
metric `play_mode_event_frac` is the share of the learner's play-mode decisions that chose the event.

`--vp-potential C` (owner, 2026-10-03; default 0 = off) adds potential-based VP shaping on top of `--reward-scheme`:
`Phi = C x VP` from the US side, `Phi(terminal) = 0`, and each step pays the mover `Phi(s') - Phi(s)`. Over a game it
sums to a constant, so the optimal policy is unchanged; VP gains are paid when they happen and the lead is taken back
at the end. The critic then learns `V(s) - Phi(s)` in the mover's frame. `ai/rewards/reward_calculator.py`,
`VPPotentialShaping`. A win-only reward is `--reward-scheme terminal --no-blunder-window`: the scheme alone still
leaves the blunder window, which pins a blunderer's return to -1 within the turn.

**Branching with an added head.** `--resume` accepts a state written without a training-only head the new run builds
(`--aux-card-events`, `--aux-ownership`, `--aux-vp-margin`): the trunk, policy and value are restored exactly, so the
branch starts as the saved network, and the head starts fresh with fresh Adam moments. Any other mismatch still refuses.

`--setup-script-frac F` (owner, 2026-10-02; default 0 = off; requires `--setup-mc-credit`) sets up a fraction `F`
of games, drawn per game, by one of `--setup-script-openings` (default: the four human variants in
`tools/lib/openings.HUMAN_OPENING_MIX` -- USSR East Germany 4, Poland 4, Yugoslavia or Austria 1; US West Germany 4
with Italy 3 + Iran 3 or Italy 4 + Iran 2) for both sides, and trains those placements as the policy's own: the
stored log-prob is the network's, so the PPO ratio starts at 1, and the credit is the game result against the
critic's baseline. A scripted opening gains probability only where its games beat the baseline, and the network
learns the middlegames that follow openings it never samples (`research/log/E7_shallow_setup_lock.md`). With
`--setup-mc-credit` the setup rows are also left out of the per-seat KL statistics, as they are out of the surrogate.

`--play-mode-floor EPS` (P31 1a, owner 2026-10-06; default 0 = off) puts a uniform floor in the learner's
**behaviour** policy at play-mode decisions and at the non-country choices inside events (rows whose whole legal set is
in the branch block: event branches, DEFCON values, regions): `mu = (1 - EPS) pi + EPS uniform(legal)`. Unlike
`--play-mode-temp`, the exploration is corrected: the stored log-prob stays `log pi`, so PPO's ratio and clip are the
usual `pi_theta / pi_old`, and each floor sample's surrogate is weighted by `pi_old / mu` (detached, at most
`1 / (1 - EPS)`). The ratio must not be taken against `mu`: a rare action the floor drew would sit at `pi / mu ~ 0.25`,
below the clip, so a negative advantage would be clipped to a constant and the floor could only ever raise rare
actions (E7-26-44, voided: entropy 0.32 -> 0.45 in 95M steps). Learner rows only. `--floor-scope seeded` limits it to
`--seed-scenarios` games; `--floor-from S` keeps it at 0 before `S` steps, and `--floor-anneal-from A
--floor-anneal-steps N` take it linearly to 0 over `N` steps from `A`. Off, nothing draws from the RNG, so a run is
bit-identical to one without the flag. Metrics: `floor_eps`, `floor_row_frac` (share of the learner's decisions
covered), `floor_draw_frac` (share of those that took the uniform draw). `ai/training/show_and_decide.py`.

`--seed-scenarios {subs,chernobyl}... --seed-frac F` (P31 1b, owner 2026-10-06; default off) forces, in a fraction
`F` of games drawn at each game start, the listed precursor event for the US as **environment**: the first time the
US holds the card at an action round's own card play with the event able to trigger, the card is played for its
event (Chernobyl's region drawn uniformly). Whether a card choice is the round's card play is asked of the engine --
a clone is stepped with the selection and must land on a US play-mode decision offering the event -- because an
action-round card choice with no resolving card is not always a play (Quagmire's discard). The forced rows are
stored with `learner = 0`, like a frozen opponent's: they keep the GAE recursion and train the critic, but get no
policy gradient and stay out of the learner's statistics (E7-11-44: a scripted action trained as the policy's own is
adopted whatever it is worth). Everything after the forced play is the policy's own. `--seed-scenarios-from S`: no
game is seeded before `S` steps. Metrics: `seed_games_frac`, `seed_forced_<name>` (forced plays so far).

`--aux-opp-legality W` (P31 1d, owner 2026-10-06; default 0 = off) builds a training-only head (`opp_legal_head.*`)
predicting, per country, whether the **opponent** may place influence there and may coup there **at its next
decision**, and trains it with BCE at weight `W`, one optimiser step per iteration. The label is read from the
opponent's own observation when it next moves in that game (its `can_my_place` / `can_my_coup`, board slots 19 / 21),
so it reflects what this side then played. The current observation already carries the opponent's legality
(`can_opp_place` / `can_opp_coup`, slots 20 / 22, Chernobyl-aware), so a label taken at the stored state would teach
only a copy. A fraction `--aux-opp-legality-frac` (0.1) of decisions is sampled; positions whose game ends before the
opponent moves again get no label; an update needs `--aux-opp-legality-min-batch` (4096) answered positions. A
`--resume` from a state without the head adds it fresh. Metrics: `opp_legal_loss`, `opp_legal_acc`,
`opp_legal_copy_acc` (how often the position's own `can_opp` flags were already the answer) and
`opp_legal_changed_frac` (how often they were not: the part the head must learn), `opp_legal_n`.

`--mode-cf-coef C` (P31 1c, owner 2026-10-06; default 0 = off) adds counterfactual mode credit from paired
playouts. At 1 in `--mode-cf-subsample` (16) of the learner's floor decisions (as `--play-mode-floor`'s rows), the
state is cloned before the step; at the rollout's end every legal option is played out to the end of the game by the
current network on both sides (temperature 1), `--mode-cf-playouts` (1) games each, all options of a decision starting
from the same state and RNG, so their dice start in common. An option's value is the mean result from the deciding
side, centred on the option taken, and the policy loss gets `-C * sum_a pi(a) * (Q(a) - Q(taken))` on those rows (the
all-actions policy gradient; centring changes nothing in expectation). Nothing is acted on: rollouts still sample the
policy. `--mode-cf-from S` starts it at `S` steps. **Cost:** a playout plies like a rollout step (~3 ms for the batch)
until its longest game ends; at k = 16 on E7 that is ~7 s per iteration (~8k steps/s against ~100k). Metrics:
`mode_cf_n`, `mode_cf_options`, `mode_cf_spread` (best minus worst option value), `mode_cf_taken_beaten_frac`,
`mode_cf_seconds`, `mode_cf_loss`. `ai/training/mode_cf.py`.

`--force-applicable-events {wargames,arms_race,one_small_step}... --force-event-frac F` (owner, 2026-10-06;
default off) plays the event, with probability `F` (0.1), at the learner's play-mode decision for a listed card
whose event is legal and **applicable** for the side playing it: Wargames at DEFCON 2 with a lead of 7+ VP, Arms Race
ahead in military Ops, One Small Step behind in the space race (`show_and_decide.event_applicable`, read off the
state). The forced play is trained as the **policy's own** (owner's choice): `learner = 1`, stored log-prob
`log pi(EVENT)`, no importance weight, so the event rises where its advantage is positive and falls where it is not,
each step bounded by PPO's clip. An importance-weighted version would leave the expected gradient where it was
without the forcing (the gradient on a rare action's logit is `pi(a) * advantage`). `--force-events-from S` starts it
at `S` steps. Metrics: `force_applicable_<card>` (the learner's applicable plays so far), `force_forced_<card>`.

`--ladder-branch-head` (owner, 2026-10-06; default off) adds a head on the branch block (flat 200..219: event
branches, CONFIRM_DONE, DEFCON values, regions) reading the trunk output plus a one-hot of the card the decision is
about (the observation's ACTIVE_NOW card flag, card slot 13), and adds its output to those logits. The branch slots
are shared by every card with a branch, so a head without the card learns a card-independent average: at Wargames'
branch the trunk predicts whether ending wins at 0.98 AUC while the plain policy's P(end) is ~0.2 at any lead
(`research/log/P31_branch_arms_B1_B2.md`). Zero-initialised, so a network with it starts as the one without; a
`--resume` from a state without it adds it (`branch_head_net.*`). The one-hot spans all 110 cards; only those with a
branch ever reach those rows.

`--ladder-play-mode-head` (owner, 2026-10-07; default off) is the same for the play-mode block (flat 110..114:
EVENT, SPACE and the three Ops modes, the last three also the deferred Ops-mode choice): a correction from the trunk
plus a one-hot of the card being played (ACTIVE_NOW at those decisions). The play-mode slots are shared by all 110
cards, so a push on one card's event otherwise moves every card's (E7-29-44: forcing three events halved the event
share of every play-mode decision). Zero-initialised, addable on `--resume` (`play_mode_head_net.*`).

`--force-event-credit {own,environment}` chooses how `--force-applicable-events` credits a forced play: `own` (the
default, as above) or `environment` (`learner = 0`, no policy gradient, as `--seed-scenarios`). The condition
`wargames_branch` is Wargames at DEFCON 2 **at any lead**: with `environment` credit it supplies visits to Wargames'
branch, where the policy then decides end-or-pass on its own, with both outcomes on offer. `--floor-rows
event_choices` limits `--play-mode-floor` to the non-country choices inside events.

`--ladder-token-layers L` / `--ladder-token-dim D` (P30 C1, default 0 = off; grouped trunk only) add a
token path beside the grouped projections: each of the 84 country rows and 110 card rows becomes a token
(a projection of its row plus a learned identity), the globals one more, and `L` pre-norm transformer
layers of width `D` run over all 195. The global token's output joins the fusion input; each per-entity
head reads its own token. It runs in bf16 on the GPU and costs far more than the trunk it sits beside
(full attention at D=128, L=2 is ~12x the shallow trunk per minibatch), so measure throughput first.
Recovered from the weights (`tok_*`) like every other axis.

`--obs-features NAME...` (P30, default none) appends optional observation blocks to the base layout
(`ops_budget`: the Ops the card at a play-mode decision grants after every modifier, and each side's
per-card Ops modifier -- 3 floats). The set is recorded in the weights and in `metadata.json`, so
tournaments and the match harness build each agent's observation in its own set, and a model with
features plays one without in the same batch. See `engine/AGENTS.md` §6a.

`--aux-card-events W` (P30, default 0 = off) adds a per-card head on the trunk. For every card in the
mover's hand it predicts 17 numbers (`ai/training/card_event_targets.py`):

* **Ops reach (5):** the card's effective Ops with every modifier the engine applies (Red Scare/Purge,
  Containment/Brezhnev, the China Card in Asia, Vietnam Revolts -- a replica of
  `Operations::combine_ops`, checked against the engine's own grants in `tests/training/test_card_event_aux.py`);
  the countries and battlegrounds it could bring under control (two Ops a point while the opponent
  controls); the best coup chance with SALT and Death Squads, excluding coups that lose the game.
* **Event outcome (12):** what its event would do on this board -- VP, DEFCON, the six regional margins,
  battlegrounds and influence of each side. Events with choices are played out on a clone, each choice
  taken greedily by whoever makes it, for their own best fixed score.

The engine labels a sampled fraction of decisions (`--aux-card-sample-frac`, default 0.0005 -- a label costs
~2 ms of Python, so this is ~10% of the time at 96k steps/s) into a FIFO buffer (`--aux-card-buffer`, 65,536).
Once `--aux-card-min-batch` positions are in it, every iteration takes `--aux-card-steps` (4) optimiser steps on
minibatches of `--aux-card-batch` (512) drawn from it. The loss is masked MSE on fixed-scale standardised
targets. The head predicts all 110 cards; the loss reads the held cards' slots, so card identity sits in the
output weights and the board must come through the trunk.
Metrics: `card_aux_loss`, `card_aux_r2` (explained share of the standardised variance), `card_aux_labelled`,
`card_aux_label_s` (seconds spent labelling per iteration -- the throughput cost). Why:
`research/log/P30_card_board_targets.md`; probe the result with `tools/scripts/card_board_probe.py --frozen`.

* `tools/scripts/aux_ownership_probe.py --checkpoint <snapshot>` rates the ownership head per country
  against the country's usual final controller and against its current controller, overall and
  on the games where the usual controller did not win it.

`tools/scripts/average_weights.py` averages finished checkpoints uniformly, for rating, with no
training: one run's late snapshots give its **SWA** (`<run>/swa_480-560M.pt`), and branches of one
trained state give a **model soup**.

`--ladder-head-center` (**on by default since 2026-09-25**: auto, i.e. on for per-entity heads in the E4 view, following the checkpoint on a resume or warm start, off with `--merged-influence`; `--no-ladder-head-center` for the old heads) centres the per-entity heads' hidden features across entities before
their final projection. In E4 no decision compares country actions with other actions, so a
shift common to every country logit is invisible to the policy and gets no gradient. Left free,
it drifts without limit, and every bit of the long runs' logit level was in `pe_country`. The
flag removes that direction: the policy on every country-only decision is unchanged, the final
bias gets zero gradient, and the raw values stay at the size of the differences between
countries. It is recorded as a `pe_center` buffer, so loaders recover it from the weights. It is
refused with `--merged-influence`, where countries do compete with play modes.

`--z-loss-coef c` (default 0) adds `c · mean(logsumexp(policy logits)²)` to the update (PaLM's
z-loss, usually `c = 1e-4`). The softmax ignores a common shift of the logits, so nothing else
bounds their level, and it drifts upward without limit. Both 800M runs (E4-57-43/44) diverged
through that drift (`research/log/E4_long_runs.md`). The level is logged as `logit_lse_mean` /
`logit_lse_absmax` whether the penalty is on or not.

`--compile-update {off,default,max-autotune}` (default off) runs the PPO update's forwards under
`torch.compile`. The rollout keeps its CUDA graphs of the eager network. It measured +10% solo
and +9% paired with `max-autotune`, and its outputs and gradients differ from eager by less than
TF32 does. It is opt-in until its A/B runs (`research/log/P26_quick_screen.md`). Do not use
inductor's CUDA-graph modes: `reduce-overhead` crashed beside the trainer's own graphs.

### Throughput and CPU

`tools/train.py` and `tools/tournament.py` set `OMP_WAIT_POLICY=PASSIVE` before PyTorch or the
engine load OpenMP. With the default policy, idle workers spin: a training run used ~8.5 cores for
the throughput passive waiting gives on ~1.6. An explicit `OMP_WAIT_POLICY` in the environment
still wins.

The rollout forwards run as CUDA-graph replays (`ai/training/graphed_forward.py`). Each replays
the same kernels as eager, so its outputs are bitwise identical, but with one launch instead of
~317. The learner's and the pool opponent's graphs are replayed one after the other, never
concurrently, and all are captured on one stream per cache (`e91b8d2`, `754f9e1`; the module
docstring says why). After those fixes graphs are worth about +1.2-1.5%. `--no-cuda-graphs` falls
back to eager.

Measurements, and what was changed and why, are in
[`research/log/training_throughput_cpu.md`](../research/log/training_throughput_cpu.md).

### Before spending a run on a new advantage estimator

`tools/scripts/advantage_variance_probe.py <checkpoint.pt>` computes every estimator over **one
shared rollout** and reports advantage spread. Minutes of GPU against the hours an arm costs.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/scripts/advantage_variance_probe.py \
    data/checkpoints/<run>/snapshot_<N>steps.pt
```

It exists because sane return targets do not clear an estimator. Per-player GAE beat the default
on every offline return metric -- RMSE 0.551 -> 0.481, outcome correlation 0.861 -> 0.901, exact
telescoping at lambda 1 -- and lost by **520 Elo** at 80M steps, because its advantages carried
28% more variance on identical data. The advantage is what the policy gradient consumes; the
return target is not.

### Reading an A/B

`tools/compare_runs.py <baseline-dir> <arm-dir>` compares two runs at matched step counts. It
prints the configuration diff **before** the metrics, deliberately: the numbers mean nothing until
you know what else differs.

```bash
PYTHONPATH=. .venv/bin/python tools/compare_runs.py \
    data/checkpoints/<baseline> data/checkpoints/<arm>
```

Three things it encodes, each of which was learned by getting it wrong:

* **Observed differences, not just recorded ones.** It compares opponent-pool growth from the
  runs' own logs, because the confound that voided E3-22-28's first attempt was a snapshot cadence
  that *neither* run's metadata recorded. What a run did is evidence; what its metadata claims is
  a claim.
* **A key one run predates is "unverifiable", not a difference.** Otherwise every comparison
  against an older baseline invents confounds and the warning stops meaning anything.
* **Windows, not iterations, and watch the base rate.** A single iteration's `critic_auc` swings
  by more than the effects usually looked for, and AUC is a discrimination score over whatever
  win/loss mix the window contained -- two windows with different base rates are not the same
  problem. Base rates print beside every row, and a material divergence is marked `!`.

```bash
# Two runs that are exactly comparable: identical step budget, one flag apart
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 1.0
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 0.0
```

Because a step budget says nothing about elapsed time, the progress line projects a wall-clock ETA
from the observed rate plus measured overhead, so a step budget can still be aimed at a target
duration:

```
[1,310,720/1,500,000 steps, ETA 130s] It   20 | Steps: 1,310,720 (15,356 st/s) | ...
```

`--eval-max-snapshot-opponents` (default 4) bounds evaluation cost. Each snapshot would otherwise
join the opponent list permanently, making evaluation quadratic in run length until it crowds out
the training it is supposed to measure. Baselines named by `--eval-opponents` are never dropped;
pass `0` for the old unbounded behaviour. Snapshot evaluation runs through the same vectorized
path as `tools/tournament.py`.

### Resuming and branching

`--resume` restores the weights, the optimiser moments, the frozen reference policy and the step
counter, so a run continues rather than restarts. It takes a file, a run directory (its newest
state), or `<run_dir>:<steps>` to branch from a particular snapshot:

```bash
--resume <run-dir>              # continue where it stopped
--resume <run-dir>:160038912    # branch from that snapshot of a run that went further
```

**An unchanged continuation stays in the run's directory.** `--resume <run-dir> --run-name <the
same name>` from the run's newest state appends to that directory -- metrics, TensorBoard,
snapshots -- and adds a leg to its `metadata.json` (`legs`: each leg's commit, flags, description,
`start_steps` and `train_steps`; the top level is the current leg). A different name (a branch), an
earlier state, or an old run's link directory gets a new `<name>_<timestamp>` directory, as does
`--new-directory`. A run whose `run.pid` is still alive is refused rather than written twice.

A resume state is written beside **every** snapshot (`resume_<steps>steps.pt`, several times the
size of the snapshot itself), not only at the run's end, so any point of a run stays branchable.
`--no-resume-every-snapshot` turns that off if disk matters more; a run then keeps only its newest
state and no earlier stretch of it can be re-run.

**Giving a `--seed` that differs from the one the state was written under also re-seeds torch and
numpy**, so the continuation genuinely diverges. Without that, the restore hands back the original
run's action sampling and minibatch order and only the environment deals differ -- half a seed
change, and a seed replicate that understates the variance it exists to measure. The same seed, or
none, restores the stream as before.

#### Naming the run

`--run-name` takes `E<n>-A<n>-R<n>-S<n>` plus the branch points that led to the run
(`E7-A4-R1-S44@4390M+S45`; `research/method/run_nomenclature.md`). The A and R codes are sets of
flags, so they are read off the run rather than chosen:

```bash
tools/scripts/run_codes.py                 # every run's name in the grammar; fails on an uncoded recipe
tools/scripts/run_codes.py --update        # code new architectures/recipes; rewrite research/run_codes.json,
                                           # research/architectures_and_recipes.md and run_name_map.md
tools/scripts/run_codes.py --run <dir>     # does this run's name match the flags it recorded?
```

A new code needs a one-line description in `research/run_codes.json`; a code change that alters
training under the same flags is recorded there as an `override`, since no flag can show it.

#### Splitting the seed

One `--seed` drives four independent sources at once, which is what makes "seed 1 collapses"
unattributable -- the same number picks the starting weights, the rollout sampling, the card deals
and dice, and the opponent draw. Each has an override, and each defaults to `--seed`, so behaviour
is unchanged unless one is passed:

| flag | what it seeds |
|:---|:---|
| `--seed-init` | weight initialisation (and the numpy global stream) |
| `--seed-sampling` | action sampling and minibatch shuffling |
| `--seed-env` | the engine's per-game streams: card deals and dice |
| `--seed-pool` | opponent-pool draws, and which side the learner takes |

`--seed-sampling` works by re-seeding torch immediately *after* the model is built, so everything
before that point is initialisation and everything after is sampling. All four are recorded
resolved in `metadata.json`, so a run states which streams it actually used rather than leaving it
to be inferred from the command.

Note that `--seed-init` also covers `np.random.seed`, but that stream is consumed only by the
warmup-dataset and behavioural-cloning paths (`np.random.choice` / `np.random.randint`); every
other numpy RNG in training is an explicit `default_rng`/`RandomState` instance. On a cold start it
therefore changes nothing.

### Metrics: JSONL and TensorBoard

Every iteration is logged to `<output-dir>/training_metrics.jsonl` and mirrored to TensorBoard
event files in `<output-dir>/tb/`. `--no-tensorboard` disables the mirror; the JSONL is always
written, and a missing or broken `tensorboard` install only prints a warning.

```bash
.venv/bin/python -m tensorboard.main --logdir <run-dir>/tb
```

Four groups, plus the per-opponent evaluations:

| prefix | what it holds |
|:---|:---|
| `progress/` | how fast the run is going: throughput, elapsed time, iteration counter |
| `internal/` | the optimiser's own view — losses, KL, entropy, clip fraction, explained variance, advantage health. Nothing here says whether the agent *plays* well. |
| `endgame/` | what the finished games look like, and the only group with human counterparts: win rate per side, draws, turns, plies, final score, ending mix |
| `strategy/` | is it playing the board well — empty and uncontrolled battlegrounds at turns 5 and 8, salvageability, forced-decision take rates, and the named blunder rates |
| `eval/` | win rate against each fixed baseline, at snapshots only. Strength rather than shape, so it sits outside the four. |

**The x-axis is environment steps, not iterations**, because iteration count depends on
`--num-envs` and rollout length, which would put two step-budgeted runs on different x-axes.
`total_steps` is therefore not a series (it would be the line *y = x*); `progress/iteration` is
logged instead, and its slope is the steps-per-iteration.

**Several lines per chart, not several charts.** TensorBoard draws one line per *run* per chart,
so the pooled series, the per-winner splits and the human references are written as sibling run
directories under the same tag: `.` (the run itself), `won_us`, `won_ussr`, `human_ITS`,
`human_won_us`, `human_won_ussr`. `endgame/turn` therefore carries six lines instead of occupying
six charts. Charts that combine genuinely *different* quantities go through `add_scalars` instead:
`endgame/win_rate` (US / USSR / draw against their human values), `endgame/ending_mix`, and
`strategy/battlegrounds_turn8` (empty against uncontrolled).

`endgame/turn_distribution` and `endgame/ply_distribution` are histograms rather than scalars: a
mean of 6.8 turns is either most games ending near turn 7 or a mixture of turn-3 blowups and
full-length games, and the two are not the same policy.

Human reference lines come from `ai/itsc_reference.py` (completed games from the ITS Junta results
database). Three things deliberately have no line: `mean_victory_points` and `mean_vp_margin` (ITS
does not record the final score), the `defcon1_self` / `defcon1_provoked` split (ITS records the
outcome without the cause, so only the combined `ending_defcon1` is comparable), and everything
under `strategy/`. The ply references are estimates; the turn references are measurements.

Beyond the loss terms, each iteration records `explained_variance` (`1 - Var(G - V) / Var(G)` for
the win-value head — the primary read on whether the critic is learning), advantage-distribution
health (`adv_std`, `adv_std_raw`, `adv_frac_near_zero`), game length (`mean_turn`, `median_turn`,
`mean_ply`, `median_ply`, `episodes_completed`), the ending-reason mix (`ending_frac_*`), which
side won (`ussr_win_rate`, `draw_rate`, `mean_terminal_utility` — US-positive), and
`entropy_fixed_probe`: mean masked policy entropy over a pool of (observation, mask) pairs frozen
at the start of the run, which unlike the on-policy `entropy` cannot be masked by
state-distribution drift.

`mean_ply` / `median_ply` measure game length in the continuous player-slot numeration defined by
[`ai/game_length.py`](../ai/game_length.py): ply 1 is the USSR's turn-1 headline, ply 2 the US's,
ply 3 the USSR's turn-1 AR1, and **154 is a game that played all ten turns out**. Prefer it to
`mean_turn`, which answers "which of the ten" and nothing finer -- a game abandoned at turn 7 AR1
and one that ran to turn 7 AR7 are the same number -- and which carries an artefact at the top of
its range: `finish_end_turn` increments the turn and only then tests `turn <= 10`, so a completed
game terminates holding turn **11** where a human replay log calls it turn 10.

**Per-seat signal, for side collapse.** A collapse is one seat losing nearly every self-play
game. Then its advantages shrink, its policy gradient fades, and the entropy bonus acts almost alone.
So the advantage statistics are also logged before normalisation, split by acting seat
(`adv_mean_us/ussr`, `adv_std_us/ussr`, `adv_n_us/ussr`), next to the learner's policy entropy
per seat (`entropy_us/ussr`). A collapsing seat shows up as its `adv_std_*` falling away from the
other seat's while its `entropy_*` rises.

Two opt-in levers act on that signal. Both are off by default, and off means the draws and updates
of a run without them are unchanged:

* `--seat-balance` (with `--seat-balance-max-frac`, default 0.8) tracks the self-play US win share
  `sp_us`. It sets pressure = min(1, |sp_us - 0.5| / 0.3) and then:
  * puts the learner on the losing seat with probability 0.5 + 0.4 x pressure;
  * raises the fraction of pool (mixed) envs toward the max fraction;
  * draws opponents by PFSP x(1-x) on that seat's own record against each member.

  It logs `opp_seat_sp_us`, `opp_seat_pressure`, `opp_seat_weak_is_us` and
  `opp_seat_learner_on_weak`, and the startup banner reports `seat-balance=on`.
* `--per-seat-adv-norm` normalises each seat's advantages by that seat's own mean and std, instead
  of one shared mean and std. The losing seat's smaller spread then keeps unit scale rather than
  being divided down by the winning seat's.
* `--wolf-seat-weight` ("win or learn fast") scales each seat's PPO surrogate so the winning seat
  learns slowly and the losing seat fast:
  * x is an exponential average of the USSR's win share in pure self-play games. Its memory is
    `--wolf-ema-games`, 2000 games by default.
  * The weights are w_us = 2x^p / (x^p + (1−x)^p) and w_ussr = 2(1−x)^p / (x^p + (1−x)^p),
    with p = `--wolf-power`. At p = 1 that is simply 2x and 2(1−x).
  * `--wolf-scope` sets what the weights scale:
    * `surrogate` (the default, and E4-38) scales the PPO surrogate only. The unweighted entropy
      bonus then pushes the down-weighted seat toward uniform: E4-38's entropy stayed ~1.75 for
      60M.
    * `policy` scales the seat's whole policy objective: surrogate, entropy bonus and KL to π_ref.
      That is a per-seat learning rate, which is how WoLF is defined.
  * The value loss is never weighted.
  * Logged as `wolf_sp_ussr`, `wolf_w_us` and `wolf_w_ussr`, and carried in the resume state.
* P25's loop-closing levers (steps 3j–3l). Each is off at 0, off leaves the update bitwise
  unchanged, and none may be combined with `--wolf-seat-weight`:
  * `--adv-norm-floor c` divides the advantages by `max(batch std, c × EMA of the batch std)`, so
    a faded signal stays small instead of being rescaled to unit noise. The EMA has a memory of
    20M steps and the floor starts at 2M. Not defined with `--per-seat-adv-norm`. Logged as
    `adv_norm_divisor`, `adv_norm_floor_bound` and `adv_std_ema`.
  * `--entropy-ceiling H` is a one-sided per-seat entropy ceiling in nats. Each iteration from 5M
    steps, a seat's entropy coefficient moves by −0.01 × (its rollout entropy − H), clipped to
    [−0.02, `--entropy-coef`]. Below the ceiling it is the fixed bonus, so it never pushes a
    sharpening seat back up. Logged as `ent_coef_us` / `ent_coef_ussr`.
  * `--target-kl k` stops a seat's policy terms (surrogate, entropy, KL to π_ref) for the rest of
    the update once that seat's approximate KL from the rollout policy on a minibatch exceeds `k`.
    The value loss continues. `approx_kl_us` / `approx_kl_ussr` are logged on every run, and
    `kl_stop_frac_*` with the target.
  * The floor's EMA and the per-seat coefficients are carried in the resume state.
* `--entropy-normalize` makes the entropy bonus reward entropy / log(legal count) per decision, the
  fraction of that decision's maximum. A many-option decision (E4.1's ~50-option op-mode nodes)
  then gets no more room than a few-option one. On E4's decision mix the raw entropy is about
  2.1x the normalised one, so `--entropy-coef 0.021` with it matches the default bonus on
  average. The logged entropy stays raw.

At every snapshot it also records the win rate against each fixed baseline, overall and per side
(`eval/win_rate_vs_HeuristicBot`, `..._as_us`, `..._as_ussr`), alongside the decisive-decision and
position diagnostics. Snapshot *opponents* are deliberately excluded: they are renamed every
interval, so each would start a series that stops one interval later.

Three things are deliberately **not** logged, because a series that cannot vary is worse than an
absent one -- it reads as a measurement:

* **Auxiliary losses whose term is switched off.** `defcon_risk_loss` needs `--defcon-coef` and
  `inject_loss` needs `--inject-dataset`; `belief_loss`, `oracle_loss` and `distill_loss` belong to
  architectures that no longer exist. On an ordinary v2 run all five are a flat zero line.
* **Per-start-turn breakdowns when no start pool is in use.** `--start-pool-frac` defaults to 0, so
  every game starts at turn 1 and `game_start1/*` duplicates `game/*` exactly. The `game_start<N>/`
  series are derived on demand and reappear as soon as episodes start at more than one turn.
* `steps_per_sec` is the rate **since the previous iteration**. The lifetime average is kept as
  `steps_per_sec_avg`, and is the one to ignore on a resumed run: its clock is rewound to include
  the previous leg, so it overstates the current rate.

---

## 2. `tools/tournament.py` (Unified Tournament & Head-to-Head Evaluator)
Vectorized tournament and matchup evaluator. Adapts automatically based on the number of models
passed.

### A. 2-Model Head-to-Head Matchup Mode:
```bash
# Evaluate a model against HeuristicBot for 100 games (50 US / 50 USSR) with loss causes
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --models <checkpoint.pt> heuristic \
  --games-per-side 50
```

### B. Multi-Model Round-Robin Tournament Mode:
```bash
# 1,000 games per matchup, round-robin across every checkpoint in a directory
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir <run-dir> \
  --games-per-side 500 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report <run-dir>/tournament_report.md
```

**Pairings are packed** (`--pack-pairs`, default 25). A round robin used to play one pairing at a
time: ~200 games, two ~100-row forwards per step, the GPU at ~35%. Now the agents are split into
blocks of ⌊√pack⌋, and a pack plays all pairings between two blocks, or inside several blocks, in
one engine batch, with one forward per *agent* per step over all its positions
(`BatchMatchRunner.play_packed_matchups`). Each game keeps the deal seed it had before, with the
same seat-swapped copy.

On a 33-player field (528 pairings, 100 games per side):
* tournament time went from 819 s to 250 s (3.3×);
* 547 of 561 pairings were identical game for game, and the largest Elo difference was 0.4. The
  rest is float rounding of near-ties at a different batch composition.

Greedy agents and bots reproduce the old path apart from that rounding. Sampled agents reproduce
it in distribution only. `--pack-pairs 1` plays one pairing at a time, and `--track-choices` or
`--log-games` force it. The one cost that remains is the heuristic bot, which is pure Python per
position.

### C. Multicore: search, and bots that decide in Python
A plain network is batched inside one process and is bound by inference, so it gains little from
more. A `search:` entrant is not: its tree (selection, state clones, engine steps, backups) runs in
Python on one core, and only the leaf evaluation is batched -- the GPU sat at ~4% under a 256-sim
search (`research/log/search_cost_and_coverage.md` §10). The same holds for a bot that decides in
Python (`heuristic_mcts`). `--workers N`
(0 = every core) cuts each matchup into shards of `--shard-pairs` game pairs (default 10; a pair is
one deal played from both sides) and plays them in N processes, one thread each
(`tools/lib/parallel_tournament.py`).

The split plays the same deals as a single process. Every shard loads its agents afresh from their
specs, so nothing an agent carries between games -- a search's own generator, `heuristic_mcts`'s
included -- crosses from one shard into the next, and it seeds every generator its games draw
from: the global ones, and any agent's own through `reseed` (a `search:` entrant's
determinization, chance-node and subsampling streams, an ONNX agent's sampler). So **results depend
on `--shard-pairs`, never on `--workers`** -- `--workers 1` included, which plays the same shards in
one worker process. Omitting `--workers` runs the old one-process path with pairing packing
(`--pack-pairs`); with deterministic agents it plays the same games as the shards, with sampling
ones the same deals but not the same draws. Each worker loads its own copy of every model -- on a
small machine, count the memory -- and so more than one worker is refused a CUDA device: pass
`--device cpu`. If a shard fails, the shards still queued are dropped and the error is raised once
the running ones finish. `--opening` reaches each worker as its specs' `opening:<name>:` prefix.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --models heuristic_mcts data/checkpoints/E5-11-43_560M.onnx --games-per-side 100 \
  --device cpu --workers 0 --output-json data/reports/heuristic_mcts_vs_E5.json
```

### D. ONNX models
A `.onnx` file from `tools/export_onnx.py` -- what the workbench plays, and the only form the
published models on Hugging Face take -- is an entrant like a checkpoint (`OnnxAgent`, ONNX
Runtime on CPU). Its name, observation width and action view come from the export's metadata, and
a file that is not an export, or was made for another observation width, is refused. Greedy, it
plays the same games as its checkpoint (the export is verified to pick the same favourite move);
sampled, it draws from its own generator, so it matches in distribution only.

```bash
hf download mihaild/deepstruggle E5-11-43_560M.onnx --local-dir data/checkpoints
```

### E. Across machines: `--part` and `--pool-parts`
`--part I/N --output-part part.json` plays every N-th shard of the tournament (starting at the
I-th) and writes their raw results and game logs; `--pool-parts part*.json` checks that the files
are one tournament's parts and cover each shard exactly once, merges them, and writes the usual
report, JSON and `--log-games`. Since a shard's games depend only on the shard, the pooled
tournament equals one machine playing all of it -- every part must be given the same entrants,
`--games-per-side` and `--shard-pairs`, and pooling refuses parts that were not. A part writes its
shards and nothing else: `--output-report`, `--output-json` and `--log-games` belong to the
`--pool-parts` run and are refused with `--part`.

```bash
# on machine k of 4
tools/tournament.py --models heuristic_mcts <model.onnx> --games-per-side 200 --device cpu \
  --workers 0 --shard-pairs 1 --part k/4 --output-part parts/part-k.json
# anywhere, once all four are in
tools/tournament.py --pool-parts parts/part-*.json --output-json data/reports/pooled.json
```

---

## 2b. The Elo leaderboard (`tools/leaderboard.py`, `tools/leaderboard_play.py`)

A tournament's Elo is fitted to its own field, so the same model rates differently in two
tournaments. The leaderboard is one scale per engine epoch, kept as records in `leaderboard/`
(reviewed like code, published with it) and fitted on demand:

| file | holds |
|:---|:---|
| `networks.json` | each network: file (relative to `data/`), sha256, `hf` path once published, behaviour `report`, description, `old_names` |
| `players.json` | each player: a network plus how it is run (`tools/lib/player_spec.py`) |
| `epochs.json` | each engine epoch: accepted engine fingerprints, the main players, the anchor and its rating |
| `matches/<epoch>.jsonl` | one line per pairing played: both seats' w/l/d, base seed, engine, commit -- append-only, union-merged |

**A player is a network plus an inference spec.** The id spells the spec: the bare network name is
its greedy policy; anything else is `<network>~<kind>(<field>=<value>,...)` with every field
named, e.g. `E7-A8-R1-S44@6800M~gumbel(sims=256,k=8,fpu=0.2)` or
`...~search(sims=128,determinize=true,node_filter=all,subsample=1,backend=cpp,fpu=0)` or
`...~policy(temperature=0.5)`; bots are `HeuristicBot` / `RandomBot`. A search id names its
defaults too, so it never changes meaning when a default does.

**The fit** is Bradley-Terry with a US-seat term, draws half a win, in two stages: the main
players on their games against each other (anchor pinned), then everyone else against the main
players held fixed (non-main against non-main games included). Adding games without a main player
never moves a main rating, and "main only" / "main + lineage X" are filters over the one fit. A
player no pairing connects to the main players is listed as unrated.

```bash
# register (computes the sha256), then rate against the epoch's main players
PYTHONPATH=. python tools/leaderboard.py add-network E7-A8-R1-S44@6800M \
  data/checkpoints/E7-A8-R1-S44_20261007_162257/snapshot_6800015360steps.pt --description "..."
PYTHONPATH=. python tools/leaderboard.py add-player 'E7-A8-R1-S44@6800M~gumbel(sims=256,k=8)'
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/leaderboard_play.py \
  --epoch E7 --players 'E7-A8-R1-S44@6800M~gumbel(sims=256,k=8,fpu=0.2)' --games-per-side 1000
#   --opponents A B ...   other opponents; --round-robin  every pairing among --players
#   a pairing on record is skipped; --base-seed S adds new deals to it

PYTHONPATH=. python tools/leaderboard.py show --main-only            # or --lineage E7-A8-R1-S44
PYTHONPATH=. python tools/leaderboard.py validate                    # what tests/training checks
```

`leaderboard_play.py` refuses a stale build, an engine fingerprint the epoch does not list, a
network file whose sha256 changed, and uncommitted changes to `ai/ bot/ bindings/ engine/ tools/`
(each record names its commit). The main players are the owner's choice, edited in
`epochs.json`; a new main player is played against every existing one. An engine change opens a
new epoch unless a recorded pairing replays identically, in which case its fingerprint is added.

**The page.** `tools/scripts/build_web.sh` runs `tools/leaderboard.py fit` into
`web/ui/public/leaderboard.json` (git-ignored; stdlib only, so CI needs no torch) and builds
`leaderboard.html` beside the workbench. Each row links the network's behaviour report on GitHub,
its weights on Hugging Face and the workbench with that network loaded; a click shows the
head-to-head results; the view is kept in the link.

## 2c. Publishing to Hugging Face (`tools/publish_hf.py`)

The repo (`mihaild/deepstruggle`) mirrors `data/checkpoints/`: a run directory goes up whole,
an SWA or soup as `_models/<name>.pt`. Resume and opponent-pool states (`resume_*.pt`,
`pool_*.pt`), `run.pid` and `snapshot_final.pt` stay local: snapshots are referred to by step, and
`snapshot_final.pt` is a copy that names none (the leaderboard refuses to register it too).

The workbench runs `.onnx` files (exported and verified by `tools/export_onnx.py`), and a run
holds a snapshot every 10M steps, so only these get one, beside their `.pt`: in a run directory
the **final plain snapshot** (the highest `snapshot_<N>steps.pt`), the **final SWA**
(highest window) and **every file registered in `leaderboard/networks.json`**; and every file
given to `files` -- the **soups** and SWAs under `_models/`. Any other `.onnx` lying in a run
directory is not uploaded. To make another snapshot openable later:
`publish_hf.py files <run-dir>/snapshot_<N>steps.pt`. After an upload, registered networks whose
file went up get their `hf` path in `leaderboard/networks.json` -- commit that change.

```bash
PYTHONPATH=.:build/release python tools/publish_hf.py run data/checkpoints/<run-dir> --dry-run
PYTHONPATH=.:build/release python tools/publish_hf.py files 'data/checkpoints/_models/<name>.pt'
PYTHONPATH=.:build/release python tools/publish_hf.py default '<run-dir>/snapshot_<N>steps.onnx'
PYTHONPATH=.:build/release python tools/publish_hf.py index      # rebuild models.json as the repo is
```

**`models.json`, the repo's catalogue.** Every publish rewrites it in the same commit: each
`.onnx` the repo holds with its commit date, sha256 and size, newest first, and `default` -- the
model a workbench link that names none opens (`default` sets it; without one the page takes the
newest upload). The page reads this one file instead of listing the tree, which costs one request
per 50 files when the dates are wanted. Files uploaded any other way are not in it until `index`
rebuilds it; `index` also carries over a legacy `default.json`'s model, so run it once on a repo
published before the manifest existed. The page's model picker groups the files by directory.
Uploading needs a write token (`hf auth login` or `HF_TOKEN`).

---

## 3. `tools/play_match.py` (Unified Match Runner & Replay Generator)
Plays a match between any pair of agents, supports two distinct checkpoints, provides interactive
CLI terminal play, and writes standardized `.tslog.json` replays for the Web Workbench.

```bash
# 1. Pit two neural checkpoints against each other:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us <checkpoint_a.pt> \
  --ussr <checkpoint_b.pt> \
  --game-id a_vs_b

# 2. Interactive terminal play (human vs bot):
PYTHONPATH=. .venv/bin/python tools/play_match.py --us human --ussr heuristic

# 3. Strategic commentary and regional scoring breakdown:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us strategic --ussr event_heavy --commentary

# 4. Force a named opening on both sides, then let the agents play from there:
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py \
  --agent <checkpoint.pt> --opening human --seed 305 --commentary

# 5. Record what the network believed at every node, and what its critic thought:
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py \
  --us <checkpoint.pt> --ussr heuristic --trace

# View any generated replay in the Web Workbench at:
# http://localhost:8000/?replay=<filename>.tslog.json
```

**`--opening`** replaces the fifteen setup placements with a named opening from
`tools/lib/openings.py` and hands control back to the agents once setup is over, so the replay
shows what a policy does with a board it did not choose. Three openings are defined:

| name | USSR | US |
|:---|:---|:---|
| `human` | +1 East Germany, +4 Poland, +1 Yugoslavia | +4 West Germany, +3 Italy, +2 Iran |
| `ph_west_germany` | +3 Poland, +3 Hungary | +4 West Germany, +3 Italy, +2 Iran |
| `ph_no_west_germany` | +3 Poland, +3 Hungary | +2 Canada, +2 Italy, +3 France, +1 Iran, +1 South Korea |

A scripted placement that is not legal raises rather than falling through to the agent -- a
partly-forced setup is neither opening, and would be reported as one. The same registry backs
`ai/eval/forced_setup.py` and `ai/eval/setup_critic.py` (the critic's value of two openings on
the same deal, per checkpoint), so the replays show the openings their numbers were measured on.

**`--trace`** records, on every step of the replay, the distribution the policy drew its move
from and the critic's reading of the position that move produced — the workbench then shows a
value ribbon under the timeline, a probability chip per log row, and the full distribution for
the selected step. Neural self-play traces by default (`--no-trace` turns it off); a match does
not, because most matches here are bot-vs-bot baselines with no distribution to record.
`--trace-top-k` sets how many legal actions are listed per node; **0, the default, lists every
one of them**, because the workbench puts each probability on the card, mode button or country
it belongs to and a truncated distribution would leave most of the board unlabelled. A non-zero
cap keeps the file smaller, reports the rest of the mass as `p_tail`, and always lists the move
that was played whatever its probability. `--trace-critic-every {step,decision,off}` trades a
gap-free value curve against the extra forward pass on steps the loop settled itself. The probabilities are the model's own
distribution at temperature 1, not the tempered one that was sampled from; the sampling
probability is recorded alongside as `p_chosen_sampled`. Details in `ai/eval/policy_readout.py`.

---

## 3b. `tools/annotate_replay.py` (Post-hoc Policy & Critic Annotation)
Asks a checkpoint what it thinks of a game it may not have played, and writes the answers onto
the replay in the same format `--trace` produces.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/annotate_replay.py \
  --replay data/replays/s160_1.tslog.json \
  --model data/checkpoints/<run>/final.pt --print
```

It re-drives the game from its seed by applying the logged actions, exactly as
`ai/eval/replay_critic.py` does, and **aborts on the first step where the reconstruction stops
matching the replay** — a drifted reconstruction still returns numbers and they still look like
results. A rebuilt engine can change the decision stream with no Python change (invariant 13),
which is the usual way that happens, so the engine fingerprint is written next to the numbers.
`--print` gives a per-step table (probability of the move played, the model's best, entropy,
`v_win` and its step-to-step change); `--limit` stops early for a quick look; `--trace-top-k`
caps the listing as above. The probability
reported is always the one the model assigns to the move **the replay recorded**, never to the
move the model would have made — that is `argmax_idx`, and the two differing is the interesting
case.

---

## 3c. `tools/export_onnx.py` (A Checkpoint for the Browser Workbench)
Writes a checkpoint as ONNX, the format the browser workbench runs (onnxruntime-web), for a
Hugging Face repo or for dropping onto the page. The local workbench server calls the same
`export()` on demand, so there you can pick any `.pt` directly.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/export_onnx.py \
  --checkpoint data/checkpoints/<run>/snapshot_final.pt --out <run>.onnx
```

The file describes itself in ONNX `metadata_props` -- `ts.format`, `ts.obs_size`,
`ts.action_size`, `ts.merged_influence` (the action view, from the run directory as for every
harness), `ts.label`, `ts.checkpoint` + `ts.checkpoint_sha256`, `ts.engine_fingerprint` -- and the
page refuses a model whose observation width is not the engine's. Two things are proved on real
positions before anything is written, and the export is refused (exit 1) otherwise: the value
heads ignore the mask (the page's critic rows pass all ones where Python passes none), and ONNX
Runtime agrees with torch (same favourite move everywhere, probabilities within 1e-3, values
within 1e-4). About 12.6 MB per model.

The browser side is built by `tools/scripts/build_web.sh` (the engine as WebAssembly, then the
page) with Emscripten from `tools/scripts/install_emsdk.sh` (pinned, no root). Rebuild the page
after any engine change: it runs its own copy of the engine, and `tests/web/test_wasm_engine.py`
holds that copy to the native one bit for bit.

---

## 4. `tools/generate_dataset.py` (Vectorized Demonstration Dataset Generator)
Churns out thousands of games in parallel across hundreds of C++ environments, on a
multi-temperature exploration schedule, writing a compressed `.jsonl.gz` dataset for supervised BC
warmup.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \
  --models <checkpoint.pt> \
  --total-games 5000 \
  --batch-size 500 \
  --output-path <dataset.jsonl.gz>
```

The result is a *syntax* prior: it can only distil the policy that generated it, biases included.
It stores `(seed, actions)`, so **regenerate it after any engine change** -- an older set silently
truncates against a changed decision stream.

---

## 5. `tools/inspect_checkpoints.py` (Checkpoint Registry Inspector)
Scans the checkpoint tree and lists every saved model with its file size, modification timestamp,
and the architecture detected from its own weights (`ColdWarNet` V1 or `ColdWarNetV2`). A
checkpoint from a retired architecture is reported as retired: it cannot be run, and
`tools.lib.player_agent.reject_retired_architecture` refuses it rather than loading part of one.

```bash
PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
```

---

## 6. `tools/download_ts_replayer.py` (Human Game Corpus) and the Converter

Human Twilight Struggle games, played by people on the Playdek/Steam app and uploaded to
ts-replayer.fly.dev, turned into engine decisions. The downloader fetches each replay's four JSON
islands once, throttles to one request a second, and skips anything already on disk, so a re-run
costs nothing.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/download_ts_replayer.py
```

It stores into `datasets/ts_replayer` in the shared data tree (`tools/lib/data_root.py`: the main
checkout's `data/`, found from git even inside a worktree), so every checkout and every git
worktree shares one copy -- a worktree has its own empty data directory, and a corpus kept there
would be re-fetched in full for bytes already on the machine. `tools/lib/corpus_paths.py` resolves
the location: `$TS_REPLAYER_CORPUS` first, then the shared data tree. `--out` overrides it.

ts-replayer serves some games under several replay ids, so `distinct_corpus_files()` deduplicates
on content before any split: a duplicate carries several times its weight in behaviour cloning,
and because its ids differ it can otherwise land on both sides of an id-based train/held-out split.

The conversion lives in `tools/lib/` and is *verified*, not merely parsed. Every entry is rebuilt
from the position the log states, driven through the engine as MicroActions, and the resulting
board compared against the log's own next board -- so a mis-parsed entry surfaces as a mismatch on
that entry rather than passing silently into the dataset. Nothing is forced and nothing falls back
to a heuristic approximation: an entry the log does not determine is a failure to diagnose, not a
guess to paper over. Every downloaded game converts; the entries that do not are turns whose
recording stops part way, and every hand solves to exactly the size the rules deal.

- `tools/lib/ts_replayer_parse.py`: the log's grammar -- entries, sections, influence moves, die
  rolls, discards, reveals, headlines, and the country/card name tables.
- `tools/lib/ts_replayer_convert.py`: the driver. Turns each entry into the queue of decisions the
  engine asks for (`pq` for Ops, `eq`/`evq` for events), steps the engine, and reconciles the
  outcome against the log. Also holds the small, individually diagnosed lists of entries the log
  itself gets wrong (`_KNOWN_SCORE`, `_LOG_MISCOUNTED`, `_INVALID_PLAYS`), plus one fault
  recognised by rule rather than listed: where Asia is scored with Shuttle Diplomacy in play and
  the USSR holds Japan, the log keeps a superpower-adjacency bonus the card has removed and pays
  the USSR 1 VP too many (`_shuttle_japan_asia_miscount`). The engine adopts the log's score from
  there on, deliberately: the players were reading the app's score, so that is the position they
  decided against and the one training data must carry. A rule also covers games nobody has
  downloaded yet, where a list cannot.
- `tools/lib/ts_replayer_hands.py`: the hands, which the log never states in full. Both hands for a
  whole game are solved at once as a constraint problem over z3 (MIT), from the rules -- hand size,
  carry-over, spent cards being in the discard pile until a reshuffle, scoring cards that cannot be
  held past a turn, what a trap or an empty hand proves -- with the preference heuristics as soft
  clauses. z3 is optional; without it the converter falls back to per-turn heuristics and
  `Conversion.hands_solved` is False.

Two rules of the road:

- **Never force a decision the log does not state.** The corpus is training data for a model meant
  to learn human play; an invented choice teaches it something no human did.
- **The engine is the reference.** Where the engine and a log disagree, the log is at least as
  likely to be wrong (`_LOG_MISCOUNTED` exists for exactly this), so diagnose before changing
  either -- and engine changes are the user's call.

---

## 7. `tools/build_human_dataset.py` (Human Corpus BC Dataset)

Turns the ts-replayer corpus into behaviour-cloning data -- the only *strategy* prior available,
as against the self-play set's syntax prior.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py
```

Writes a git-ignored, memory-mapped column store (one `.npy` per column plus `meta.json`) under
the dataset directory, read by `ai.training.human_corpus_dataset.HumanCorpusDataset`, whose
`stream_batches` mirrors `WarmupDataset`'s interface.

Observations are stored materialised rather than re-derived from a seed, because a human game has
no seed that replays it -- the dice come from the log and the hands are solved -- so the conversion
is the only thing that reproduces one, at about a second a game. That gives up the self-play
format's forward compatibility, so **rebuild it after any engine change**.

**Value targets are masked on unfinished games.** Roughly half the corpus stops mid-game (the
recording ends, not the game), and those positions have no outcome. The `has_outcome` column is 0
there; a trainer must drop them from the value loss and keep them in the policy loss.

---

## 7b. The Disagreement Bank and Its Human Verdicts

`ai/eval/banks/disagreement_verdicts.jsonl` is a bank of 100 positions from the human corpus,
72 of them with each candidate move marked good or bad by a strong human reviewer (with a
confidence and, often, a note on why). Every row was a human headline, card or play-mode decision
where the human, the raw network and its search did not all choose the same move. Score a model
against the marks:

```bash
export PYTHONPATH=.:build/release
.venv/bin/python tools/scripts/bank_verdicts.py score --agent newest.onnx               # the raw network
.venv/bin/python tools/scripts/bank_verdicts.py score --agent gumbel:newest.pt:32:4     # a searcher
```

It prints the share of positions where the agent's move is marked good, marked bad, or unmarked,
overall and by kind and confidence (`--by`; `--out` keeps the per-position moves). The positions
were chosen *because* the network or its search disagreed with someone, so the rates compare one
model with another; they are not an estimate of how often a model errs in play.
`tests/training/test_disagreement_bank.py` holds the bank to the current engine: every position
loads and asks the decision it was reviewed as, and every mark is a legal move there.

How the bank is made:

```bash
# 1. the network's .pt for the searcher, rebuilt from the published ONNX and checked against it
.venv/bin/python tools/onnx_to_checkpoint.py --onnx newest.onnx --template shallow_E7-02+03+04+05_1200M.pt --out newest.pt
# 2. every disagreement in the corpus (resumable, in parts: a long run can stop on a rare nanobind
#    instance collision, BUGS.md BIND-1). The searcher is reseeded with each game's replay id, so
#    the rows do not depend on the split or on a resume.
.venv/bin/python tools/scripts/disagreement_bank.py --net newest.onnx --search "gumbel:newest.pt:32:4" \
    --part 1/3 --out part1.jsonl.gz --summary part1.json --resume
# 3. how clear each position is: a stronger search's pick and the moves valued over redealt worlds
.venv/bin/python tools/scripts/bank_clarity.py --bank part*.jsonl.gz --checkpoint newest.pt \
    --out clarity.jsonl.gz --emit-playouts 1000 playout_input.jsonl.gz
# 4. paired playouts of the clearest (select / run / pool; .github/workflows/bank_playouts.yml runs
#    `run` over CI runners). A position's pairs are seeded from its row id and the run's --seed,
#    which every part shares; `pool --expect N` names the parts that did not arrive.
.venv/bin/python tools/scripts/bank_playouts.py select --clarity clarity*.jsonl.gz --bank part*.jsonl.gz \
    --min-gap 10 --out playout_input.jsonl.gz
# 5. positions whose hand is over the rules' limit (BUGS.md CONV-1), as an exclusion list -- the
#    "over" rows and untrusted turns only; "missing" also flags correct positions and is reported
.venv/bin/python tools/scripts/bank_hand_check.py --bank part*.jsonl.gz --flags hand_flags.json \
    --untrusted untrusted_turns.json --exclude exclude_ids.json
# 6. the review page's data files: one per pattern, in parts under the 16 MB file limit
.venv/bin/python tools/scripts/disagreement_bank.py --pack part*.jsonl.gz --clarity clarity.jsonl.gz \
    --playouts playouts.jsonl.gz --exclude exclude_ids.json --out review/
```

`tools/scripts/disagreement_review.html` is the review page, published as an artifact with the
`db` capability next to the packed files. A reviewer marks each distinct move good or bad (or the
position Unclear), with a confidence, a note and a bank flag; the page stores this per position id
-- a hash of the position and the decision kind -- so a rebuilt bank keeps the verdicts already
given. Export the page's `verdicts` collection (one `<id>.json` per document) and fold it into the
committed bank:

```bash
.venv/bin/python tools/scripts/bank_verdicts.py collect <exported-verdicts-dir>
```

To read the reviews, open `tools/scripts/disagreement_review.html` straight from disk and load
`ai/eval/banks/disagreement_verdicts.jsonl` with its "Reviews file" picker: each position shows its
three moves, the marks, the confidence and the note, and "Open in workbench" opens the board (set
the workbench address to your own, e.g. `http://localhost:8000/` after `tools/scripts/build_web.sh`
and `web.server.main`). Opened that way the page has no database, so a change made there stays on
the page; the network's probabilities, values and playouts are not in the file and are not shown.

A verdict the page carried to a new id after the CONV-1 fix replaces the one it came from, and a
mark on a move that is no longer legal (a card the fix took out of the hand) is kept apart as
`stale_marks`. The positions are the corpus's converted positions, so the bank must be rebuilt --
and its verdicts carried over by id -- after any change to the converter or the save format.

The network's win chance in a bank row (`v_win`) is `(1 + v_win) / 2` of the value head's output,
which regresses the mover's result on [-1, +1]. Banks packed before that was so read a raw or
sigmoid-squashed value as a probability, so repack them before reviewing from them.

Every bank tool builds E4 masks, so an agent that decides in the merged-influence view is refused
(`tools/lib/corpus_driver.require_e4_view`) rather than handed masks it would misread.

The corpus driver the bank runs on, `tools/lib/corpus_driver.py` (`feed_corpus_game`, `selfplay`,
the `pos=` tokens), also feeds `event_play_census.py`, which counts how each card in hand is used,
and `placement_census.py`, where Ops influence goes, in the human corpus and in a policy's
self-play:

```bash
.venv/bin/python tools/scripts/event_play_census.py --human-corpus --dump human.json
.venv/bin/python tools/scripts/event_play_census.py --checkpoint newest.onnx --games 4096 --dump newest.json \
    --placement-dump newest_pl.json
.venv/bin/python tools/scripts/placement_census.py --human-corpus --dump human_pl.json
.venv/bin/python tools/scripts/placement_census.py --compare human_pl.json NEWEST=newest_pl.json
```

---

## 7c. Search Experiments: Searchers on the Bank, the Rollout Root, Distillation Rounds

The tools behind `research/log/E7_search_depth_and_value.md` and `research/log/E7_gchoice_distill.md`.
Each runs locally and has a CI workflow that shards it over runners.

**Searchers.** `gumbel:<ckpt>:sims:k:fpu:filter:worlds` -- the sixth field, `worlds`
(`BatchedMCTSConfig.gumbel_worlds`), splits each candidate's share of a halving phase over that many
independent draws of the hidden cards and the dice. `rollout:<ckpt>:k:worlds:horizon:rule`
(`ai/search/rollout_root.py`) is the rollout root: the network's top `k` moves, each played in
`worlds` sampled worlds and played out by the network through `horizon` action-round boundaries, the
critic at the leaf, then `argmax`, `z<x>` (the best only if its paired lead over the network's move is
x standard errors) or `kl<t>` (one mirror-descent step). `rollout:<ckpt>:4:16:4:z2` is the measured
player. `name:<label>:<spec>` names an entrant in reports.

**`tools/search_reliability.py`** -- searchers on the search bank: `search` (every position, each
searcher, `--seeds` independent streams), `playouts` (each position's network move and every pick in
paired raw-network continuations), `report` (departure rate, reproducibility, population-weighted
playout gain, merged across runs by position). Workflow `search_reliability.yml`. The offline judge
ranks searchers but games decide: it ranked Gumbel@1,024 above Gumbel@256, which games reverse.

**`tools/value_probe.py`** -- `label` (fresh self-play, a sampled share of decisions, the network's top
moves in paired continuations; workflow `value_probe.yml`), `fit` (heads on the frozen trunk against
the critic, on held-out positions and on the bank), `tune` (the checkpoint with its critic head
fine-tuned on the labels, the policy untouched).

**Distillation rounds.** `tools/generate_search_targets.py --target gchoice` (a Gumbel root at each
`--gumbel-sims` budget) or `--target rollout` (`--rollout-spec`) records the teacher's pick beside the
network's argmax and distribution; `tools/gchoice_targets.py arm` rewrites one set of games into each
arm (`departures`, `gated`, `consensus`, `soft`, `own`), `tools/train.py --mode distill` fine-tunes,
and `tools/gchoice_targets.py check` reports on held-out games where the checkpoint moved. Workflow
`gchoice_distill.yml` runs targets, arms, checks and a greedy round robin end to end.

**Games.** `searcher_tournament.yml` -- a round robin of searchers over one checkpoint (a `.pt` from
the Hugging Face repo, or an export rebuilt as torch), greedy, split over runners.

```bash
export PYTHONPATH=.:build/release
# a searcher in a match
.venv/bin/python tools/tournament.py --models <ckpt.pt> gumbel:<ckpt.pt>:256:8 rollout:<ckpt.pt>:4:16:4:z2 \
    --games-per-side 100 --temperature 0 --workers 0 --device cpu
# searchers on the bank, then the report
.venv/bin/python tools/search_reliability.py search --bank bank.jsonl.gz --model <ckpt.pt> \
    --spec w1=256:8:0.2:all:1 --spec r4=rollout:4:16:4:z2 --seeds 4 --out search.jsonl.gz
.venv/bin/python tools/search_reliability.py playouts --bank bank.jsonl.gz --search search.jsonl.gz \
    --model <ckpt.pt> --pairs 512 --out playouts.jsonl.gz
.venv/bin/python tools/search_reliability.py report --bank bank.jsonl.gz --search search.jsonl.gz \
    --playouts playouts.jsonl.gz --out report.md
# a distillation round with the rollout root as the teacher
.venv/bin/python tools/generate_search_targets.py --checkpoint <ckpt.pt> --target rollout \
    --rollout-spec 4:32:4:z2 --node-filter all --subsample 0.25 --total-games 50 --batch-size 50 \
    --device cpu --seed-offset 10000000 --output-path targets-1.jsonl.gz
.venv/bin/python tools/gchoice_targets.py arm --input targets-*.jsonl.gz --form soft --budget rollout \
    --tau 0.1 --out soft.jsonl.gz
.venv/bin/python tools/train.py --mode distill --warmup-checkpoint <ckpt.pt> \
    --distill-dataset soft.jsonl.gz --distill-epochs 2 --device cpu --output-dir soft.pt
.venv/bin/python tools/gchoice_targets.py check --input heldout.jsonl.gz --base <ckpt.pt> \
    --model soft.pt --budget rollout --out check.json
```

---

## 8. Shared Helpers Library (`tools/lib/`)
Internal simulation, evaluation, and logging modules imported by the CLI tools:
- `tools/lib/player_agent.py`: unified agent loader (`load_agent`) and policy inference wrappers, including `OnnxAgent` for `tools/export_onnx.py` exports.
- `tools/lib/player_spec.py`: a player as a network plus a typed inference spec (policy / gumbel / search / bot), its canonical id and its `load_agent` string; the search defaults live here.
- `tools/lib/leaderboard.py`: the Elo leaderboard's records, their validation and the two-stage fit (stdlib only).
- `tools/lib/__init__.py` re-exports nothing: import from the submodules, so the torch-free ones stay importable without torch.
- `tools/lib/batch_tournament.py`: high-throughput C++ batch tournament runner and Bradley-Terry MLE solver; plays any subset of a matchup's pairs with their own deals, and merges the parts (`merge_matchup_results`).
- `tools/lib/parallel_tournament.py`: splits a tournament into shards played in worker processes (`tools/tournament.py --workers`).
- `tools/lib/tournament_evaluator.py`: diagnostic loss cause classifier (`classify_game_ending_reason`).
- `tools/lib/self_play.py`: single-game trajectory runner; the one writer of `.tslog.json` replays.
- `tools/lib/scoring_formatter.py`: regional scoring calculation formatter.
- `tools/lib/checkpoint_utils.py`: architecture detection and checkpoint discovery utilities.
- `tools/lib/corpus_paths.py`: where the ts-replayer corpus lives, and content-based deduplication.
- `tools/lib/corpus_driver.py`: feeds a policy's self-play or the converted human corpus, one decision at a time, to any tracker; the workbench's `pos=` tokens; the E4-view check the bank tools share.
