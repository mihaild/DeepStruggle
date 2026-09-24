# Twilight Struggle AI CLI Tools (`tools/`)

Standalone developer CLIs for training, tournament benchmarking, match simulation, replay
generation, demonstration dataset creation, and checkpoint inspection. Never invoke the training,
tournament or match machinery from an ad-hoc script -- go through these.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning (NashPG) or supervised demonstration warmup with
live snapshot tournament evaluation.

```bash
# RL training run with blunder-aware rewards, snapshotting every 5M env steps
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --warmup-checkpoint <warmup.pt> \
  --train-steps 160000000 \
  --snapshot-every-steps 5000000 \
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

`--train-steps` is the budget and `--snapshot-every-steps` the snapshot cadence. Give an A/B's
two arms the same value for **both**. The snapshot cadence is not only a reporting knob: the
self-play opponent pool is fed from snapshots, so an arm that snapshots more slowly trains
against a smaller, staler pool. `--snapshot-every-steps` used to be derived from two time
flags, and E3-22-28's first attempt thereby snapshotted every 26.7M steps against its
baseline's ~5M: at 45M steps it had 2 pool opponents where the baseline had 9, and with
`--opponent-frac 0.3` that made it a two-factor experiment. It was thrown away.

### Throughput and CPU

`tools/train.py` and `tools/tournament.py` set `OMP_WAIT_POLICY=PASSIVE` before PyTorch or the
engine load OpenMP. With the default policy, idle workers spin: a training run used ~8.5 cores for
the throughput passive waiting gives on ~1.6. An explicit `OMP_WAIT_POLICY` in the environment
still wins.

The rollout forwards run as CUDA-graph replays (`ai/training/graphed_forward.py`). Each replays
the same kernels as eager, so its outputs are bitwise identical, but with one launch instead of
~317. The learner's graph and the pool opponent's graph overlap on two streams. `--no-cuda-graphs`
falls back to eager.

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

A resume state is written beside **every** snapshot (`resume_<steps>steps.pt`, several times the
size of the snapshot itself), not only at the run's end, so any point of a run stays branchable.
`--no-resume-every-snapshot` turns that off if disk matters more; a run then keeps only its newest
state and no earlier stretch of it can be re-run.

**Giving a `--seed` that differs from the one the state was written under also re-seeds torch and
numpy**, so the continuation genuinely diverges. Without that, the restore hands back the original
run's action sampling and minibatch order and only the environment deals differ -- half a seed
change, and a seed replicate that understates the variance it exists to measure. The same seed, or
none, restores the stream as before.

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

## 8. Shared Helpers Library (`tools/lib/`)
Internal simulation, evaluation, and logging modules imported by the CLI tools:
- `tools/lib/player_agent.py`: unified agent loader (`load_agent`) and policy inference wrappers.
- `tools/lib/batch_tournament.py`: high-throughput C++ batch tournament runner and Bradley-Terry MLE solver.
- `tools/lib/tournament_evaluator.py`: diagnostic loss cause classifier (`classify_game_ending_reason`).
- `tools/lib/self_play.py`: single-game trajectory runner; the one writer of `.tslog.json` replays.
- `tools/lib/scoring_formatter.py`: regional scoring calculation formatter.
- `tools/lib/checkpoint_utils.py`: architecture detection and checkpoint discovery utilities.
- `tools/lib/corpus_paths.py`: where the ts-replayer corpus lives, and content-based deduplication.
