# Twilight Struggle AI CLI Tools (`tools/`)

Standalone CLIs for training, tournaments, matches, replays, datasets and checkpoint inspection.
Never invoke the training, tournament or match machinery from an ad-hoc script -- go through
these. Every CLI's `--help` is the authoritative flag list; this file says what each tool is for,
what its non-obvious flags mean, and where the evidence behind a default lives.

---

## 1. `tools/train.py` (Unified Training Pipeline)

NashPG reinforcement learning, behavioural-cloning warmup (`--mode warmup`) or search-target
distillation (`--mode distill`, §7c), with live snapshot evaluation.

**Derive a launch command from a healthy run, never from an example** (CLAUDE.md invariants
14-15): `tools/scripts/launch_flags.py <recent-healthy-run>` prints what it used, and
`--diff <your-run>` what you changed; run the diff again after launching (it reads
`metadata.json`). The skeleton below only shows the shape:

```bash
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 --run-name E<n>-A<n>-R<n>-S<n> --description "..." \
  --train-steps 160000000 --snapshot-every-steps 10000000 \
  --opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12 \
  ...   # architecture and recipe flags: copy them from the reference run
```

A warm start (`--warmup-checkpoint`) is optional and not what the ladder does.

### Budgeting a run

* **Every budget is in env steps; there is no wall-clock flag.** Steps/s depends on the policy, so
  a time budget biases an A/B in a fixed direction. The progress line still projects an ETA.
* `--train-steps` is the budget, `--snapshot-every-steps` (10M) how often a snapshot is saved and
  evaluated, `--pool-every-steps` (5M) how often the opponent pool grows. Give an A/B's arms the
  same `--pool-every-steps`: a slower-growing pool is a second factor (it voided E3-22-28's first
  attempt).
* **The snapshot interval is a reporting setting only.** Evaluation restores the RNG streams it
  draws from, so a run evaluated every 1M or 2M trains bit for bit the same
  ([`P26_quick_screen.md`](../research/log/P26_quick_screen.md)). A pool member between snapshots
  is written as `pool_<N>steps.pt`; tournaments ignore those.
* `--eval-max-snapshot-opponents` (4) keeps snapshot evaluation from growing quadratically with
  run length; `--eval-opponents` baselines are never dropped; `0` is unbounded.
* TF32 matmuls are on by default (+15% steps/s, no strength cost in a 3-seed A/B, P26); `--no-tf32`
  gives the fp32 of every run before E4-57. Recorded as `tf32` in `metadata.json`.

### Training options

All default to off unless stated, and off leaves a run bit-identical to one without the flag.
Each flag's `--help` has its sub-parameters; the "why" column links the evidence.

**Credit and targets**

| flag | what it does | why / evidence |
|:---|:---|:---|
| `--block-lambda {off,setup,setup-side,same-side}` | GAE λ = 1 inside a decision block. Default `off`; runs from E5-12 used `setup-side`, pass it to reproduce them | `setup-side` added nothing at flat temperature and froze openings; `same-side` −289 ([E5-12](../research/log/E5_12_setup_credit_at_flat_temperature.md), [P4](../research/log/P4_setup_arms.md)) |
| `--setup-mc-credit` (`--setup-mc-min-batch` 512) | setup placements credited with the game result (advantage = result − V(s)) instead of the λ-return; trained in batches once their games end, PPO-clipped; setup rows leave the surrogate and the per-seat KL statistics but keep entropy and KL | the critic over-rates unfamiliar openings 3-4x ([E5-11](../research/log/E5_11_setup_lock_and_critic_views.md)); slow from scratch |
| `--setup-entropy-floor <nats>` (`--setup-entropy-lr`, `--setup-entropy-max-coef`) | adaptive entropy bonus on setup decisions only, zero until setup entropy falls below the floor; logged `entropy_setup`, `setup_ent_coef` | a setup at p ≈ 1 is never resampled, so a bad opening locks (E5-11-43 opens Greece 2 every game). `tools/scripts/setup_oracle.py` measures whether an opening is worse |
| `--setup-script-frac F` (`--setup-script-openings`; needs `--setup-mc-credit`) | a fraction of games set up by a scripted opening (default: the four human variants in `tools/lib/openings.HUMAN_OPENING_MIX`) for both sides, trained as the policy's own (stored log-prob is the network's) | the network learns middlegames of openings it never samples ([E7 shallow setup lock](../research/log/E7_shallow_setup_lock.md)) |
| `--forced-opening <name>` | every game starts after a scripted setup from `tools/lib/openings.py`; the setup stays untrained. Rate such a checkpoint with `tools/tournament.py --opening <name>` or an `opening:<name>:<spec>` entrant; snapshot matches apply it, probes do not | |
| `--vp-potential C` | potential-based VP shaping on top of `--reward-scheme` (Φ = C·VP, Φ(terminal) = 0); the optimal policy is unchanged. `VPPotentialShaping` in `ai/rewards/reward_calculator.py`. A win-only reward is `--reward-scheme terminal --no-blunder-window` | [P30 win-only and VP potential](../research/log/P30_win_only_and_vp_potential.md) |
| `--mode-cf-coef C` (`--mode-cf-subsample` 16, `--mode-cf-playouts`, `--mode-cf-from`) | counterfactual mode credit: at sampled floor decisions every legal option is played to the end by the current network from a cloned state with shared dice; adds the all-actions policy gradient `-C Σ π(a)(Q(a) − Q(taken))`. ~7 s per iteration at k = 16 (`ai/training/mode_cf.py`) | [P31](../research/log/P31_branch_arms_B1_B2.md) |

**Exploration** (`ai/training/show_and_decide.py`)

| flag | what it does | why / evidence |
|:---|:---|:---|
| `--play-mode-temp T` | samples the learner's play-mode decisions at temperature T; the stored log-prob stays the policy's own (uncorrected). Metric `play_mode_event_frac` | [expert review](../research/log/expert_review_E7.md) |
| `--play-mode-floor EPS` (`--floor-scope`, `--floor-rows`, `--floor-from`, `--floor-anneal-from`, `--floor-anneal-steps`) | behaviour policy (1−ε)π + ε·uniform at play-mode decisions and non-country event choices. The stored log-prob stays log π and each floor sample's surrogate is weighted by π_old/μ (≤ 1/(1−ε)). Taking the ratio against μ instead lets the floor only ever raise rare actions (E7-26-44, voided). Metrics `floor_eps`, `floor_row_frac`, `floor_draw_frac` | P31 1a |
| `--seed-scenarios {subs,chernobyl}... --seed-frac F` (`--seed-scenarios-from`) | in a fraction of games the US's first eligible play of the card is forced to its event, as **environment** (`learner = 0`: trains the critic, no policy gradient) | a scripted action trained as the policy's own is adopted whatever it is worth (E7-11-44); P31 1b |
| `--force-applicable-events {wargames,arms_race,one_small_step,wargames_branch}... --force-event-frac F` (`--force-event-credit {own,environment}`, `--force-events-from`) | plays the event with probability F when it is legal and applicable (`show_and_decide.event_applicable`). `own` (default) trains it as the policy's own, `environment` as `--seed-scenarios`. `wargames_branch` is Wargames at DEFCON 2 at any lead | |

**Heads and auxiliary losses** — training-only heads are recorded by their weights, so
checkpoints load anywhere, and `--resume` from a state without the head adds it fresh (trunk,
policy and value restored exactly).

| flag | what it does | why / evidence |
|:---|:---|:---|
| `--aux-ownership W`, `--aux-vp-margin W` (`--aux-sample-frac` 0.1, `--aux-min-batch` 4096) | final control of each country (cross-entropy) and final VP margin /20 (MSE), from the game's end. Probe: `tools/scripts/aux_ownership_probe.py` | [P29 bet 2](../research/log/P29_bet2_ownership_probes.md) |
| `--aux-card-events W` (`--aux-card-sample-frac` 0.0005, `--aux-card-buffer`, `--aux-card-min-batch`, `--aux-card-steps`, `--aux-card-batch`) | per-card head predicting 17 numbers per held card: effective Ops reach (a replica of `Operations::combine_ops`) and what its event would do on this board (`ai/training/card_event_targets.py`). Labelling costs ~10% of time at the default rate. Probe: `tools/scripts/card_board_probe.py --frozen` | [P30 card-board targets](../research/log/P30_card_board_targets.md) |
| `--aux-opp-legality W` (`--aux-opp-legality-frac`, `--aux-opp-legality-min-batch`) | predicts where the **opponent** may place / coup at its next decision, labelled from its own observation then (the current observation already has `can_opp_*`, so a label at the stored state would teach a copy). Metrics include `opp_legal_copy_acc` and `opp_legal_changed_frac` | P31 1d |
| `--ladder-branch-head` | a correction on the branch block (flat 200..219) from the trunk plus a one-hot of the card the decision is about; zero-initialised | branch slots are shared by every card, so a card-blind head learns an average: the trunk predicts Wargames' outcome at 0.98 AUC while P(end) is ~0.2 at any lead ([P31](../research/log/P31_branch_arms_B1_B2.md)) |
| `--ladder-play-mode-head` | the same for the play-mode block (flat 110..114) | pushing one card's event moved every card's (E7-29-44) |
| `--ladder-head-center` / `--no-ladder-head-center` | centres per-entity head features across entities. **On by default** (auto: on for per-entity heads in the E4 view, follows the checkpoint on resume, refused with `--merged-influence`); recorded as `pe_center` | a shift common to every country logit gets no gradient and drifted without limit in the long runs |
| `--z-loss-coef c` | adds `c·mean(logsumexp(logits)²)` (usually 1e-4); `logit_lse_mean`/`_absmax` are logged regardless | both 800M runs diverged through logit-level drift ([E4 long runs](../research/log/E4_long_runs.md)) |
| `--ladder-token-layers L` / `--ladder-token-dim D` | a transformer token path over 84 countries + 110 cards + globals beside the grouped trunk; ~12x the shallow trunk's cost at D = 128, L = 2 | [P30 C1](../research/log/P30_C1_tokens.md) |
| `--obs-features NAME...` | optional observation features: `ops_budget` (a 3-float block appended) and `headline_reveal` (no width: the box-4 holder sees the opponent's revealed headline). **Default since E8: `headline_reveal`**; the list replaces it, and the flag with no names is none (every pre-E8 run). Recorded in weights and metadata; evaluation supports mixed sets (`engine/AGENTS.md` §6a), while training pools require the learner's exact set. A resume, `--resume-weights` or warm start whose weights read another set is refused | [P30 C4](../research/log/P30_C4_ops_budget.md) |

**Optimiser** (P28: [LR schedule](../research/log/P28_step2a_lr_schedule.md), [EMA](../research/log/P28_step2b_ema_weights.md))

| flag | what it does |
|:---|:---|
| `--lr-schedule {constant,step,cosine}` | counted from `--lr-schedule-start` (default: where the run starts, so a resume leaves trained steps alone). `step` walks `--lr-schedule-values` every `--lr-schedule-every`; `cosine` decays to `--lr-min` over `--lr-schedule-span`. Logged as `lr` |
| `--ema-weights <tau>` | EMA of the weights with a time constant of tau env steps; snapshots, pool members and evaluations use the average, the live weights keep training |
| `--compile-update {off,default,max-autotune}` | the PPO update under `torch.compile` (+9-10%, within TF32 of eager); opt-in. Never inductor's CUDA-graph modes: `reduce-overhead` crashed beside the trainer's graphs |

**Per-seat levers** (side collapse; signal: `adv_std_us/ussr` diverging while `entropy_*` rises)

| flag | what it does |
|:---|:---|
| `--seat-balance` (`--seat-balance-max-frac` 0.8) | from the self-play US share: puts the learner on the losing seat more often, raises the pool fraction, draws opponents by PFSP on that seat's record. Logs `opp_seat_*` |
| `--per-seat-adv-norm` | normalises each seat's advantages by its own mean and std |
| `--wolf-seat-weight` (`--wolf-ema-games` 2000, `--wolf-power`, `--wolf-scope {surrogate,policy}`) | win-or-learn-fast weights per seat; `surrogate` (default, E4-38) scales only the surrogate, so the unweighted entropy bonus pushes the down-weighted seat toward uniform; `policy` is a true per-seat learning rate. The value loss is never weighted |
| `--adv-norm-floor c`, `--entropy-ceiling H`, `--target-kl k` | P25's loop-closing levers: an advantage-std floor from its 20M-step EMA, a one-sided per-seat entropy ceiling, a per-seat early stop of the policy terms. Not combinable with `--wolf-seat-weight` |
| `--entropy-normalize` | the bonus rewards entropy / log(legal count); `--entropy-coef 0.021` with it matches the default bonus on E4's decision mix |

### Throughput and CPU

`tools/train.py` and `tools/tournament.py` set `OMP_WAIT_POLICY=PASSIVE` (spinning idle workers
used ~8.5 cores for what passive waiting does on ~1.6; an explicit setting wins). The rollout
forwards replay CUDA graphs (`ai/training/graphed_forward.py`), bitwise identical to eager;
`--no-cuda-graphs` falls back. Measurements:
[`training_throughput_cpu.md`](../research/log/training_throughput_cpu.md).

### Before spending a run on a new advantage estimator

`tools/scripts/advantage_variance_probe.py <checkpoint.pt>` computes every estimator over one
shared rollout. Better return targets do not clear an estimator: per-player GAE won every offline
return metric and lost by 520 Elo at 80M, because its advantages carried 28% more variance.

### Reading an A/B

`tools/compare_runs.py <baseline-dir> <arm-dir>` compares two runs at matched step counts and
prints the configuration diff first. It compares *observed* differences (pool growth from the
logs, not metadata), reports a key one run predates as "unverifiable" rather than a difference,
and reads windows (`--window-steps`) with their win/loss base rates beside each row, since
`critic_auc` over different base rates is not the same problem.

### Resuming and branching

`--resume` restores weights, optimiser moments, π_ref and the step counter. It takes a file, a run
directory (its newest state) or `<run_dir>:<steps>` to branch from a snapshot.

* **An unchanged continuation stays in the run's directory**: same `--run-name`, newest state ->
  appends there and adds a leg to `metadata.json` (`legs`). A different name, an earlier state or
  `--new-directory` gets a new `<name>_<timestamp>` directory. A run whose `run.pid` is alive is
  refused.
* A resume state is written beside every snapshot (`resume_<steps>steps.pt`) so any point stays
  branchable; `--no-resume-every-snapshot` keeps only the newest. `tools/scripts/prune_resume_states.py`
  thins old ones.
* **A `--seed` different from the state's also re-seeds torch and numpy**, so a replicate truly
  diverges; the same seed, or none, restores the stream.

#### Naming the run

`--run-name` is `E<n>-A<n>-R<n>-S<n>` plus branch points (`E7-A4-R1-S44@4390M+S45`;
[`run_nomenclature.md`](../research/method/run_nomenclature.md)). The A and R codes are flag sets,
read off the run rather than chosen:

```bash
tools/scripts/run_codes.py                 # every run's name in the grammar; fails on an uncoded recipe
tools/scripts/run_codes.py --update        # code new architectures/recipes; rewrite research/run_codes.json,
                                           # research/architectures_and_recipes.md and run_name_map.md
tools/scripts/run_codes.py --run <dir>     # does this run's name match the flags it recorded?
```

A new code needs a one-line description in `research/run_codes.json`; a code change that alters
training under the same flags is recorded there as an `override`.

#### Splitting the seed

`--seed` drives four streams; each has an override defaulting to `--seed`, and all four are
recorded resolved in `metadata.json`:

| flag | what it seeds |
|:---|:---|
| `--seed-init` | weight initialisation (and numpy's global stream, used only by the warmup/BC paths) |
| `--seed-sampling` | action sampling and minibatch shuffling (torch is re-seeded right after the model is built) |
| `--seed-env` | the engine's per-game streams: deals and dice |
| `--seed-pool` | opponent-pool draws and which side the learner takes |

### Metrics: JSONL and TensorBoard

Every iteration goes to `<run>/training_metrics.jsonl` and to TensorBoard under `<run>/tb/`
(`--no-tensorboard` drops the mirror; a missing tensorboard only warns):
`.venv/bin/python -m tensorboard.main --logdir <run-dir>/tb`.

| prefix | holds |
|:---|:---|
| `progress/` | throughput, elapsed time, iteration |
| `internal/` | losses, KL, entropy, clip fraction, explained variance, advantage health -- nothing about play quality |
| `endgame/` | win rate per side, draws, turns, plies, final score, ending mix, with human (ITS) reference lines |
| `strategy/` | battleground coverage at turns 5 and 8, salvageability, forced-decision take rates, named blunder rates |
| `eval/` | win rate against each fixed baseline, overall and per side, at snapshots only |

* **The x-axis is env steps**, not iterations (those depend on `--num-envs`).
* Pooled series, per-winner splits and human references are sibling run directories under one tag
  (`.`, `won_us`, `won_ussr`, `human_ITS`, ...), so one chart carries several lines.
* Human references come from `ai/itsc_reference.py`; there is none for VP, for the DEFCON-1 cause
  split, or for `strategy/`.
* `mean_ply` uses the slot numbering of [`ai/game_length.py`](../ai/game_length.py) (154 = all ten
  turns played); prefer it to `mean_turn`, whose completed games read turn 11.
* `entropy_fixed_probe` is entropy on a frozen set of positions, immune to state-distribution
  drift. Per-seat `adv_*_us/ussr` and `entropy_us/ussr` are the side-collapse signal.
* Not logged on purpose: auxiliary losses whose term is off, and `game_start<N>/` breakdowns
  without a start pool. `steps_per_sec` is since the previous iteration; `steps_per_sec_avg`
  overstates a resumed run's rate (its clock includes the previous leg).

---

## 2. `tools/tournament.py` (Unified Tournament & Head-to-Head Evaluator)

```bash
# head-to-head
PYTHONPATH=. .venv/bin/python tools/tournament.py --models <checkpoint.pt> heuristic --games-per-side 50
# round robin over a directory
PYTHONPATH=. .venv/bin/python tools/tournament.py --checkpoint-dir <run-dir> --games-per-side 500 \
  --anchor-model HeuristicBot --anchor-elo 1500.0 --output-report <run-dir>/tournament_report.md
```

A tournament's Elo is fitted to its own field; for a scale that persists, use the leaderboard (§2b).

### A. Packing

Pairings are packed into one engine batch (`--pack-pairs`, default 25), one forward per agent per
step (`BatchMatchRunner.play_packed_matchups`), with each game keeping its deal seed. On a
33-player field: 819 s -> 250 s, Elo within 0.4. `--pack-pairs 1`, `--track-choices` and
`--log-games` play one pairing at a time.

### B. Multicore: search and Python bots

A `search:`/`gumbel:` entrant's tree, and a bot like `heuristic_mcts`, run in Python on one core
(the GPU sat at ~4% under a 256-sim search). `--workers N` (0 = every core) cuts each matchup into
shards of `--shard-pairs` game pairs (default 10) played in N processes
(`tools/lib/parallel_tournament.py`). Every shard loads its agents afresh and seeds every
generator, so **results depend on `--shard-pairs`, never on `--workers`**. Each worker loads its
own models; more than one worker needs `--device cpu`.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --models heuristic_mcts data/checkpoints/E5-11-43_560M.onnx --games-per-side 100 \
  --device cpu --workers 0 --output-json data/reports/heuristic_mcts_vs_E5.json
```

### C. ONNX models

An `.onnx` from `tools/export_onnx.py` (what the workbench and Hugging Face carry) is an entrant
like a checkpoint (`OnnxAgent`, CPU); name, observation width and action view come from its
metadata. Greedy, it plays its checkpoint's games; sampled, only in distribution.
`hf download mihaild/deepstruggle <file>.onnx --local-dir data/checkpoints`.

### D. Across machines: `--part` and `--pool-parts`

`--part I/N --output-part part.json` plays every N-th shard; `--pool-parts part*.json` checks the
parts belong together and cover each shard once, merges them, and writes the report. Every part
needs the same entrants, `--games-per-side` and `--shard-pairs`.

```bash
tools/tournament.py --models heuristic_mcts <model.onnx> --games-per-side 200 --device cpu \
  --workers 0 --shard-pairs 1 --part k/4 --output-part parts/part-k.json     # on machine k of 4
tools/tournament.py --pool-parts parts/part-*.json --output-json data/reports/pooled.json
```

---

## 2b. The Elo leaderboard (`tools/leaderboard.py`, `tools/leaderboard_play.py`)

One Elo scale per engine epoch, kept as reviewed records in `leaderboard/` and fitted on demand:

| file | holds |
|:---|:---|
| `networks.json` | each network: file (relative to `data/`), sha256, `hf` path once published, behaviour `report`, description, `old_names` |
| `players.json` | each player: a network plus how it is run (`tools/lib/player_spec.py`), description, its own `report` |
| `epochs.json` | each engine epoch: its engine `behaviour` digest, main players, anchor and rating, an optional `strong` tier of main players, and the frozen `pre_digest_engines` of records played before digests |
| `matches/<epoch>.jsonl` | one line per pairing: both seats' w/l/d, base seed, engine (source fingerprint), behaviour digest id, commit -- append-only |

**A player is a network plus an inference spec**, spelled in its id: the bare network name is its
greedy policy; otherwise `<network>~<kind>(<field>=<value>,...)` with every field named, e.g.
`...~gumbel(sims=256,k=8,fpu=0.2)`, `...~rollout(k=4,worlds=16,horizon=4,rule=z2)` (§7c),
`...~search(...)`, `...~policy(temperature=0.5)`; bots are `HeuristicBot` / `RandomBot`. A rollout
root whose leaves another network reads (P34's evaluator) names it: `...~rollout(...,leaf=own,leaf_net=<network>)`;
the evaluator is a registered network like any other (its file and sha256 are checked the same way).
`chance=keyed` ties each world's dice and deals to the situation they are drawn in
(`bindings/keyed_chance.hpp`), so the candidates' playouts stay paired longer than on one shared stream.

**The fit** is Bradley-Terry with a US-seat term, draws half a win, in two stages: the main players
among themselves (anchor pinned), then everyone else against them held fixed. Games without a main
player never move a main rating.

```bash
PYTHONPATH=. python tools/leaderboard.py add-network <name> <file.pt> --description "..."
PYTHONPATH=. python tools/leaderboard.py add-player '<name>~gumbel(sims=256,k=8)'   # --report, --replace
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/leaderboard_play.py \
  --epoch E8 --players '<player-id>' --games-per-side 1000    # --opponents ..., --round-robin, --base-seed
PYTHONPATH=. python tools/leaderboard.py show --main-only            # or --lineage E7-A8-R1-S44
PYTHONPATH=. python tools/leaderboard.py validate                    # what tests/training checks
```

`leaderboard_play.py` refuses a stale build, an engine that does not reproduce the epoch's
behaviour digest, a network whose sha256 changed, and uncommitted changes under
`ai/ bot/ bindings/ engine/ tools/`. A report link (a network's or a player's) is always the
behaviour report `tools/scripts/checkpoint_report.py` generated for that exact network or player,
in `research/log/per_checkpoint/` and titled with its id -- never the plan or log it was made for,
nor the searched-leak census written alone (`--games 0`), which has no self-play; `validate` refuses
anything else, and with no report yet the link stays null (a greedy player then shows its network's). A pairing on record is skipped. Main players are the owner's
choice (`epochs.json`).

**An epoch is defined by what the engine does, not by its sources.** `tools/lib/engine_behaviour.py`
plays seeded random games (passing over a move that ends the game at once or takes DEFCON to 2, so
they reach the Late War) and hashes every observation, legal mask and action, in four sections:
`runner` (the batch runner leaderboard games are played on, under both action views and both
feature sets for each side), `state` (the single-position API), `rollout` (`RolloutBatch` under each
stop and leaf setting) and `mcts` (`BatchedSearch`), the searches driven by a stand-in network that
is a hash of the rows it is shown. It takes about 5 seconds. Any build that reproduces an epoch's
digest plays its games -- a refactor, a comment or a change to the web page needs nothing done --
and one that does not is refused, naming the sections that differ: a new epoch if the change is
meant, a bug if not. A new epoch's `behaviour` is what
`PYTHONPATH=.:build/release python tools/lib/engine_behaviour.py` prints; bump
`BEHAVIOUR_VERSION` (`tools/lib/leaderboard.py`) whenever what the digest hashes changes, and
re-take every epoch's digest on one of its engines.

**Strong newcomers play the strong tier** (`--strong`, owner 2026-10-16). A newcomer rated near or above
the top of the main set plays only the epoch's `strong` main players. Against a much weaker opponent a
game carries less information (p(1-p) at 85% is 0.13 against 0.21 at 70%), and the fit misses there:
every strong player scored below its prediction against the weakest main players and above it against
the strongest ([`P33_diagnostics.md`](../research/log/P33_diagnostics.md)), so weak pairings compress
the differences at the top.

**The page.** `tools/scripts/build_web.sh` runs `tools/leaderboard.py fit` into
`web/ui/public/leaderboard.json` and builds `leaderboard.html`: each row links the player's report,
its Hugging Face weights and the workbench with it loaded; a second tab lists the A/R codes from
`research/run_codes.json`. A search or sampled player shows no report until one of its own is
registered (`add-player <id> --replace --report <path>`).

## 2c. Publishing to Hugging Face (`tools/publish_hf.py`)

`mihaild/deepstruggle` mirrors `data/checkpoints/`: a run directory goes up whole, an SWA or soup
as `_models/<name>.pt`. `resume_*.pt`, `pool_*.pt`, `run.pid` and `snapshot_final.pt` stay local.
`.onnx` files go up only for the final plain snapshot, the final SWA, every file in
`leaderboard/networks.json`, and every file given to `files`. After an upload, registered networks
get their `hf` path -- commit that change. Needs a write token (`hf auth login` or `HF_TOKEN`).

```bash
PYTHONPATH=.:build/release python tools/publish_hf.py run data/checkpoints/<run-dir> --dry-run
PYTHONPATH=.:build/release python tools/publish_hf.py files 'data/checkpoints/_models/<name>.pt'
PYTHONPATH=.:build/release python tools/publish_hf.py default '<run-dir>/snapshot_<N>steps.onnx'
PYTHONPATH=.:build/release python tools/publish_hf.py default --player '<leaderboard player id>'
PYTHONPATH=.:build/release python tools/publish_hf.py index      # rebuild models.json from the repo
```

`models.json` is the repo's catalogue, rewritten by every publish: each `.onnx` with date, sha256
and size, `default`, the model a workbench link that names none opens, and `default_search`, how it
plays when the link names no search (a workbench `search=` value). `default <onnx>` sets the model
and keeps the search; `default --player` sets both to a leaderboard player's -- its network's `.onnx`
and its search as the workbench plays it (`player_spec.workbench_search`) -- and the leaderboard page
tags that player HF DEFAULT. The default is the best player the workbench can play. Files uploaded
another way appear after `index`.

---

## 2d. Games against people (`tools/game_stats.py`)

When a game ends in the workbench the page offers to add it to the stats database, with the
player's Playdek login and rating (`web/ui/AGENTS.md` §3.11; the Worker and its setup:
[`web/stats_worker/README.md`](../web/stats_worker/README.md)). A row is the page's submission as
sent: the result, the build's commit and engine fingerprint, the models by sha256, and the compact
log -- seed plus 7 bytes a step, the MicroAction, forced die, who made the move (a click, the
model's suggestion, auto-play), whether the model's readout was on screen, and the settings in
force. The full replay is rebuilt from it, not stored.

```bash
PYTHONPATH=.:build/release python tools/game_stats.py fetch --output data/game_stats/games.json   # needs wrangler login; --local: wrangler dev
PYTHONPATH=.:build/release python tools/game_stats.py show data/game_stats/games.json
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/game_stats.py \
    verify data/game_stats/games.json --write-replays
PYTHONPATH=.:build/release python tools/game_stats.py rating data/game_stats/games.json \
    --ussr-advantage 0 --json data/game_stats/fit.json
```

* `verify` re-drives each game played on this build's engine and checks it reaches the submitted
  final position (`final_hash`), result and summary; `--write-replays` writes the full
  `.tslog.json` (flat in `data/replays/`, each step's description prefixed with who made it). A
  game from another engine is reported with its commit: check that out to verify it. Nothing in
  the database is trusted for being there -- the checks are against accidents and bugs, and
  anyone can post anything.
* **Clean** (recomputed from the log by `tools/lib/game_stats.summarize`, never read from the row):
  one model with one search auto-played one side all game; the human played the other, always in a
  player view, never for the model, never with the model's readout on screen or its suggestion;
  no forced dice, no debug overrides, not from a loaded position. Take-backs are counted
  (`--max-undos` to leave them out), not disqualifying.
* `rating` re-drives and verifies every record in the current engine before fitting each
  **model player** -- model file and search -- on the clean games. Failed verification and
  other-engine records are excluded with counts in the output and JSON. The fit uses Elo's
  logistic on a 400-point scale against each human's self-reported Playdek rating, with
  `--ussr-advantage` the USSR's edge between equal players (not fitted; the default 0 assumes
  none). Playdek does not publish its formula; the 400-point Elo scale is what players report it
  follows. A weak normal prior (`--prior-sd 400` around the mean opponent) keeps a perfect score
  finite. The standard error is about `350 / sqrt(games)` against even opposition and grows when
  the model wins most games; the 95% interval is a bootstrap over the players, so one person's
  many games count as one person's evidence. The ratings people type are noisy and the games
  people choose to submit are not a random sample of the games played -- read a fit with both in
  mind.

---

## 3. `tools/play_match.py` (Unified Match Runner & Replay Generator)

Plays any pair of agents, including interactive terminal play, and writes `.tslog.json` replays
for the workbench (`http://localhost:8000/?replay=<file>.tslog.json`).

```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py --us <a.pt> --ussr <b.pt> --game-id a_vs_b
PYTHONPATH=. .venv/bin/python tools/play_match.py --us human --ussr heuristic
PYTHONPATH=. .venv/bin/python tools/play_match.py --us strategic --ussr event_heavy --commentary
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py --agent <ckpt.pt> --opening human --seed 305
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py --us <ckpt.pt> --ussr heuristic --trace
```

**`--opening`** replaces the setup with a named opening from `tools/lib/openings.py`, then hands
control back. An illegal scripted placement raises. The same registry backs
`ai/eval/forced_setup.py` and `ai/eval/setup_critic.py`.

| name | USSR | US |
|:---|:---|:---|
| `human` | +1 East Germany, +4 Poland, +1 Yugoslavia | +4 West Germany, +3 Italy, +2 Iran |
| `ph_west_germany` | +3 Poland, +3 Hungary | +4 West Germany, +3 Italy, +2 Iran |
| `ph_no_west_germany` | +3 Poland, +3 Hungary | +2 Canada, +2 Italy, +3 France, +1 Iran, +1 South Korea |

**`--trace`** records at every step the distribution the policy drew from (temperature 1; the
tempered sampling probability as `p_chosen_sampled`) and the critic's value of the result; the
workbench shows a value ribbon and per-move probabilities. Neural self-play traces by default
(`--no-trace`), a match does not. `--trace-top-k` (0 = every legal action, the default) and
`--trace-critic-every {step,decision,off}`. Details: `ai/eval/policy_readout.py`.

## 3b. `tools/annotate_replay.py` (Post-hoc Policy & Critic Annotation)

Writes a checkpoint's view of a game it may not have played onto the replay, in `--trace` format.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/annotate_replay.py \
  --replay data/replays/<game>.tslog.json --model <ckpt.pt> --print
```

It re-drives the game from its seed and **aborts on the first step where the reconstruction stops
matching** (a rebuilt engine can move the decision stream); the engine fingerprint is written
next to the numbers. The probability reported is always that of the move the replay recorded;
the model's own pick is `argmax_idx`. `--limit`, `--trace-top-k`.

## 3c. `tools/export_onnx.py` (A Checkpoint for the Browser Workbench)

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/export_onnx.py --checkpoint <ckpt.pt> --out <name>.onnx
```

The file describes itself in ONNX `metadata_props` (`ts.obs_size`, `ts.action_size`,
`ts.merged_influence`, `ts.checkpoint_sha256`, `ts.engine_fingerprint`, ...). Before writing,
it proves the value heads ignore the mask and that ONNX Runtime agrees with torch (same favourite
move, probabilities within 1e-3, values within 1e-4), else exits 1. ~12.6 MB per model. The local
workbench server calls the same `export()` on demand. `tools/onnx_to_checkpoint.py` goes the other
way (a torch checkpoint rebuilt from an export, for searchers). The page itself is built by
`tools/scripts/build_web.sh` (Emscripten from `tools/scripts/install_emsdk.sh`); rebuild it after
any engine change.

---

## 4. `tools/generate_dataset.py` (Self-Play Demonstration Dataset)

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/generate_dataset.py \
  --models <checkpoint.pt> --total-games 5000 --batch-size 500 --output-path <dataset.jsonl.gz>
```

A *syntax* prior: it can only distil the policy that produced it, biases included. It stores
`(seed, actions)`, so **regenerate it after any engine change** -- an old set silently truncates.

## 5. `tools/inspect_checkpoints.py` (Checkpoint Registry Inspector)

Lists every model under `data/checkpoints/` with size, date and the architecture detected from its
weights. A retired architecture is reported as such and refused by
`tools.lib.player_agent.reject_retired_architecture` rather than partly loaded.

---

## 6. `tools/download_ts_replayer.py` (Human Game Corpus) and the Converter

Human games from the Playdek app, uploaded to ts-replayer.fly.dev, turned into engine decisions.
The downloader throttles to one request a second and skips what is on disk.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/download_ts_replayer.py
```

It stores into `datasets/ts_replayer` in the shared data tree (`tools/lib/data_root.py`: the main
checkout's `data/`, found from git even in a worktree), so all checkouts share one copy;
`tools/lib/corpus_paths.py` resolves it (`$TS_REPLAYER_CORPUS` first; `--out` overrides).
`distinct_corpus_files()` deduplicates games served under several ids before any split.

The conversion is *verified*, not merely parsed: each entry is rebuilt from the position the log
states, driven through the engine, and compared against the log's next board. Nothing is forced
and nothing falls back to a guess. Every downloaded game converts; the entries that do not are
turns whose recording stops part way.

- `tools/lib/ts_replayer_parse.py`: the log grammar and name tables.
- `tools/lib/ts_replayer_convert.py`: the driver (entries -> decision queues -> engine ->
  reconciliation), with the small, individually diagnosed lists of entries the log itself gets
  wrong (`_KNOWN_SCORE`, `_LOG_MISCOUNTED`, `_INVALID_PLAYS`) and one fault recognised by rule
  (`_shuttle_japan_asia_miscount`). Where the log's score is wrong the engine adopts it from there
  on: the players decided against the score the app showed.
- `tools/lib/ts_replayer_hands.py`: both hands for a whole game solved at once as a z3 constraint
  problem from the rules, with preference heuristics as soft clauses. Missing z3 or an unsolved
  hand model is a conversion failure before any decisions or callbacks are emitted; a turn absent
  from the model also fails instead of padding a hand. Before each entry the solved hands
  are seated and every other card the engine holds goes back to the **draw deck**, never the
  discard pile: the pile holds only what the log has spent, so the engine reshuffles where the
  humans did, and the discard features and any redeal from a converted position read the real
  pile (`tests/replayer/test_replay_deck_and_discards.py`). A card out of play -- removed from
  the game or in effect -- is never seated: the solver forbids a starred card from the turn
  after its event occurs, whatever is reshuffled, and the converter stops rather than take one
  out of play (`tests/replayer/test_replay_removed_cards.py`).

Two rules: **never force a decision the log does not state** (an invented choice teaches what no
human did), and **the engine is the reference** -- the log is at least as likely to be wrong, so
diagnose before changing either, and engine changes are the owner's call.

## 7. `tools/build_human_dataset.py` (Human Corpus BC Dataset)

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py
```

The only *strategy* prior. Writes a memory-mapped column store read by
`ai.training.human_corpus_dataset.HumanCorpusDataset` (`stream_batches`, as `WarmupDataset`).
Observations are materialised (a human game has no seed that replays it), so **rebuild after any
engine change**. About half the corpus stops mid-game: `has_outcome` is 0 there, and a trainer must
drop those rows from the value loss and keep them in the policy loss.

## 7b. The Disagreement Bank and Its Human Verdicts

`ai/eval/banks/disagreement_verdicts.jsonl`: 100 human-corpus positions where the human, the raw
network and its search did not agree, 72 with each candidate move marked good or bad by a strong
reviewer. Positions were chosen *because* of a disagreement, so the rates compare models; they are
not an error rate in play. `tests/training/test_disagreement_bank.py` holds every position and mark
to the current engine (open item: `BUGS.md` CONV-1 remainder).

```bash
export PYTHONPATH=.:build/release
.venv/bin/python tools/scripts/bank_verdicts.py score --agent newest.onnx              # --by, --out
.venv/bin/python tools/scripts/bank_verdicts.py score --agent gumbel:newest.pt:32:4
```

Building the bank (rebuild after any converter or save-format change; verdicts carry over by
position id):

```bash
# 1. a .pt for the searcher, rebuilt from the published ONNX
.venv/bin/python tools/onnx_to_checkpoint.py --onnx newest.onnx --template <a .pt of that architecture> --out newest.pt
# 2. every disagreement in the corpus; resumable in parts (BUGS.md BIND-1), searcher reseeded per game
.venv/bin/python tools/scripts/disagreement_bank.py --net newest.onnx --search "gumbel:newest.pt:32:4" \
    --part 1/3 --out part1.jsonl.gz --summary part1.json --resume
# 3. clarity: a stronger search's pick and move values over redealt worlds
.venv/bin/python tools/scripts/bank_clarity.py --bank part*.jsonl.gz --checkpoint newest.pt \
    --out clarity.jsonl.gz --emit-playouts 1000 playout_input.jsonl.gz
# 4. paired playouts of the clearest (select / run / pool; bank_playouts.yml runs `run` on CI)
.venv/bin/python tools/scripts/bank_playouts.py select --clarity clarity*.jsonl.gz --bank part*.jsonl.gz \
    --min-gap 10 --out playout_input.jsonl.gz
# 5. positions with an over-limit hand, as an exclusion list
.venv/bin/python tools/scripts/bank_hand_check.py --bank part*.jsonl.gz --flags hand_flags.json \
    --untrusted untrusted_turns.json --exclude exclude_ids.json
# 6. the review page's data files, in parts under the 16 MB limit
.venv/bin/python tools/scripts/disagreement_bank.py --pack part*.jsonl.gz --clarity clarity.jsonl.gz \
    --playouts playouts.jsonl.gz --exclude exclude_ids.json --out review/
```

`tools/scripts/disagreement_review.html` is the review page (an artifact with the `db` capability):
the reviewer marks each move with a confidence and a note, stored per position id. Fold an export
of its `verdicts` collection into the bank with `bank_verdicts.py collect <dir>`. Opened from disk
with the bank file loaded, it shows the marks read-only and links each position to the workbench.
A mark on a move no longer legal is kept as `stale_marks`. A row's `v_win` is `(1 + v)/2` of the
value head; repack banks packed before that. Bank tools build E4 masks and refuse a merged-view
agent (`tools/lib/corpus_driver.require_e4_view`).

The same corpus driver (`tools/lib/corpus_driver.py`) feeds two censuses, over the human corpus or
a policy's self-play:

```bash
.venv/bin/python tools/scripts/event_play_census.py --human-corpus --dump human.json
.venv/bin/python tools/scripts/event_play_census.py --checkpoint newest.onnx --games 4096 --dump newest.json \
    --placement-dump newest_pl.json
.venv/bin/python tools/scripts/placement_census.py --compare human_pl.json NEWEST=newest_pl.json
```

---

## 7c. Search Experiments: Searchers on the Bank, the Rollout Root, Distillation Rounds

The tools behind [`E7_search_depth_and_value.md`](../research/log/E7_search_depth_and_value.md)
and [`E7_gchoice_distill.md`](../research/log/E7_gchoice_distill.md); each has a CI workflow that
shards it over runners.

**Searcher specs.**
* `gumbel:<ckpt>:sims:k:fpu:filter:worlds` -- Gumbel root; `worlds` splits each candidate's share
  over independent draws of hidden cards and dice.
* `rollout:<ckpt>:k:worlds:horizon:rule[:stop][:leaf]` -- the rollout root
  (`ai/search/rollout_root.py`): the network's top `k` moves, each played in `worlds` sampled worlds
  by the network through `horizon` action-round boundaries, the critic at the leaf; then `argmax`,
  `z<x>` (the best only if its paired lead over the network's move is x standard errors) or `kl<t>`.
  `stop` is where a playout ends (`boundaries`, the default; `turn`; `later`). An omitted field
  takes the measured default, so `rollout:<ckpt>` = `rollout:<ckpt>:4:16:4:z2`. Everything but the
  network calls is C++ (`ts.RolloutBatch`, `ts.rollout_candidates`, `ts.rollout_decide`;
  `bindings/rollout_batch.hpp`), shared with the browser's `search=rollout`. On the leaderboard:
  `<network>~rollout(k=4,worlds=16,horizon=4,rule=z2)`.
* `name:<label>:<spec>` names an entrant in reports.

**`tools/search_reliability.py`** -- `search` (searchers on the search bank, `--seeds` streams),
`playouts` (paired raw-network continuations of each pick), `report`. Workflow
`search_reliability.yml`. The offline judge ranks searchers, games decide: it put Gumbel@1,024
above Gumbel@256, and games reverse that.

**`tools/value_probe.py`** -- `label` (fresh self-play positions valued by paired continuations;
`value_probe.yml`), `fit` (heads on the frozen trunk), `tune` (the critic head fine-tuned, the
policy untouched).

**Distillation rounds.** `tools/generate_search_targets.py --target gchoice|rollout` records the
teacher's pick beside the network's; `tools/gchoice_targets.py arm` rewrites games into arms
(`departures`, `gated`, `consensus`, `soft`, `own`); `tools/train.py --mode distill` fine-tunes;
`tools/gchoice_targets.py check` reports where the checkpoint moved on held-out games;
`tools/search_transfer.py` asks whether a student learned the teacher's corrections. Workflow
`gchoice_distill.yml` runs a round end to end; `searcher_tournament.yml` is a round robin of
searchers over one checkpoint.

```bash
export PYTHONPATH=.:build/release
.venv/bin/python tools/tournament.py --models <ckpt.pt> gumbel:<ckpt.pt>:256:8 rollout:<ckpt.pt>:4:16:4:z2 \
    --games-per-side 100 --temperature 0 --workers 0 --device cpu
.venv/bin/python tools/search_reliability.py search --bank bank.jsonl.gz --model <ckpt.pt> \
    --spec w1=256:8:0.2:all:1 --spec r4=rollout:4:16:4:z2 --seeds 4 --out search.jsonl.gz
.venv/bin/python tools/search_reliability.py playouts --bank bank.jsonl.gz --search search.jsonl.gz \
    --model <ckpt.pt> --pairs 512 --out playouts.jsonl.gz
.venv/bin/python tools/search_reliability.py report --bank bank.jsonl.gz --search search.jsonl.gz \
    --playouts playouts.jsonl.gz --out report.md
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

## 7d. The Two-Stage Catastrophic-Blunder Census

Where does a checkpoint throw a game away? `tools/scripts/blunder_census.py screen` finds
candidates cheaply; `tools/scripts/bank_playouts.py validate` settles them expensively, on saved
positions, so validation re-runs without regenerating them.

* **Stage 1, `screen` (local).** Each decision of greedy self-play is screened by the engine's
  exact verdict (`ai/eval/safety.classify_legal_actions`: a forced loss taken or a forced win not
  taken) and by disagreement with a cheap Gumbel root (`gumbel:<ckpt>:32:4`). Hits go to
  `candidates.jsonl.gz` with their exact state (`pos=` token); a random `--control-rate` sample of
  the rest goes to `control.jsonl.gz` so the screen's misses are measured. A `games.jsonl` ledger
  makes it resumable (`--resume`) and splittable (`--part k/N`). Start with `benchmark` (20 games)
  and size the run from its projection.
* **Stage 2, `validate` (CI, `bank_playouts.yml stage=validate`).** Each position is searched by
  Gumbel k=8 @256 over `--search-seeds` runs, and the moves in play are compared in paired
  continuations (`ai/eval/paired_playouts.py`; `--continue-with 256:8` continues with search). The
  verdict keeps exact terminal mistakes (`exact-missed-win`, `exact-forced-loss`) apart from
  estimated regret (`confirmed-regret` at `--min-gap` and `--min-z`, `refuted`, `unresolved`).
  Unresolved rows escalate to 1,024 simulations (`--search-sims 1024 --seed 1 --escalate-from`).
* **`report`** merges the stages: confirmation rate, confirmed blunders per 1,000 decisions (exact
  and estimated, with the control's noise floor), the share the screen found, the share search
  rescues, the worst failures by cause, and each stage's cost.

```bash
export PYTHONPATH=.:build/release
.venv/bin/python tools/scripts/blunder_census.py benchmark --model newest.pt --games 20 --project 2000 --runners 20 --out-dir bench/
.venv/bin/python tools/scripts/blunder_census.py screen --model newest.pt --games 200 --out-dir screen/
.venv/bin/python tools/scripts/bank_playouts.py validate --input screen/candidates.jsonl.gz \
    screen/control.jsonl.gz --model newest.pt --pairs 64 --part 1/20 --out validation-1.jsonl.gz
.venv/bin/python tools/scripts/bank_playouts.py validate --input screen/candidates.jsonl.gz \
    screen/control.jsonl.gz --model newest.pt --pairs 64 --seed 1 --search-sims 1024 \
    --escalate-from validation.jsonl.gz --escalate-cap 100 --out escalation.jsonl.gz
.venv/bin/python tools/scripts/blunder_census.py report --screen screen/candidates.jsonl.gz \
    --control screen/control.jsonl.gz --summary screen/summary.json \
    --validation validation.jsonl.gz --escalation escalation.jsonl.gz --out report.md
```

**The blunder bank (`tools/scripts/blunder_bank.py`).** The census selects and measures regret on
the same pairs, and its control clears the bar almost as often as its candidates, so the bank keeps
only what survives a second measurement: `select` shortlists rows whose best alternative leads by
`--min-gap` at `--min-z`; `confirm` pays every shortlisted move out again from a fresh seed per
judge network (`bank_playouts.yml stage=confirm`; `--continue-with SIMS:K`); `build` keeps rows
every judge confirms (`ai/eval/banks/blunders_E7.jsonl`). `score` puts any player on the bank.
`prove` solves exactly the rows whose game ends this turn (later choices enumerated, dice and
PRNG card draws averaged by inverting SplitMix64, `tools/lib/hidden_draws.py`), and also asks
whether a row is **decided whatever is played**; `settle` drops those rows (`stage=prove`).

```bash
.venv/bin/python tools/scripts/blunder_bank.py select --validation validation.jsonl.gz --min-gap 0.1 --min-z 2 --out shortlist.jsonl.gz
.venv/bin/python tools/scripts/blunder_bank.py confirm --input shortlist.jsonl.gz --model soup.pt --pairs 512 --part 1/20 --out confirm-soup-1.jsonl.gz
.venv/bin/python tools/scripts/blunder_bank.py build --confirm 'soup=soup/*.jsonl.gz' 'r32=r32/*.jsonl.gz' \
    --min-gap 0.1 --min-z 3 --out ai/eval/banks/blunders_E7.jsonl --md blunders.md
.venv/bin/python tools/scripts/blunder_bank.py score --bank ai/eval/banks/blunders_E7.jsonl \
    --player soup=soup.pt g256=gumbel:soup.pt:256:8 --judge soup.pt --pairs 128 --out score.md
.venv/bin/python tools/scripts/blunder_bank.py prove --input ai/eval/banks/blunders_E7.jsonl --budget 2000000 --part 1/37 --out prove-1.jsonl.gz
.venv/bin/python tools/scripts/blunder_bank.py settle --bank ai/eval/banks/blunders_E7.jsonl \
    --prove 'prove/*.jsonl.gz' --out ai/eval/banks/blunders_E7.jsonl
```

---

## 7e. The Search Disagreement Bank (`tools/search_bank.py`)

Where Gumbel search departs from the raw network on raw-play positions, and what the departures
are worth. Stages: `annotate` (raw self-play positions put to the raw network and Gumbel k=4 @16,
k=4 @64, k=8 @256), `select` (a stratified, population-weighted bank), `reference` (k=16 @1024 x3
and equal-budget move values in shared redealt worlds), `oracle` (the best move forced into a
searcher's candidates), `playouts` (paired raw-network playouts) and `report`. CI:
`.github/workflows/search_bank.yml` (takes a published `.pt`, or rebuilds an ONNX export).

* **Where raw play loses, by decision kind:** `playout-select` samples the bank (every position
  with 2+ points of search-valued regret, `--rest` of the others, `--control` agreement controls)
  and `leak` maps the playouts back to the population: the best move's gain over raw's in points a
  game, by decision kind, side, era, DEFCON, card and region. Results:
  [`E7_leak_by_kind_R34.md`](../research/log/E7_leak_by_kind_R34.md).
* **Where a searched player loses:** `annotate --player rollout:4:32:4:z2:earlier:own` has the player
  play, `screen-select` / `confirm-select` judge its moves by playouts alone, and `leak --large 0
  --screen ... --annotate-meta annotate.meta.json` writes the census (reproduction:
  [`E7_searched_leak_R39.md`](../research/log/E7_searched_leak_R39.md)). Beside `leak.md`, `leak.md.json`
  holds the headline numbers with the census's network and player, which
  `tools/scripts/checkpoint_report.py --leak` checks and puts first in that player's report
  (`--games 0` for the census alone: a searched-leak report, which the leaderboard does not link and
  its network's behaviour report does; a later self-play run of the player keeps it, and `--load
  --leak` adds it to one already played; `--index` rewrites the index and the links).
* `event-select` / `event-report`: every action-round play-mode decision with a legal event, ranked
  by raw's measured loss per card.
* `subs-select` / `subs` / `subs-report`: four paired branches at each US Nuclear Subs decision
  (`ai/eval/subs_followup.py`).

```bash
gh workflow run search_bank.yml --ref <branch> -f stage=annotate -f args="--games 1500" -f model='_models/<network>.pt'
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py select \
  --annotated annotate.jsonl.gz --size 8000 --control 2000 --out bank.jsonl.gz
gh release create search-bank-<date> bank.jsonl.gz --prerelease
gh workflow run search_bank.yml --ref <branch> -f stage=reference -f release=search-bank-<date> -f model='_models/<network>.pt'
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py playout-select --bank bank.jsonl.gz \
  --reference reference.jsonl.gz --out playout_input/bank.jsonl.gz
gh release create playout-input-<date> playout_input/bank.jsonl.gz --prerelease
gh workflow run search_bank.yml --ref <branch> -f stage=playouts -f release=playout-input-<date> \
  -f model='_models/<network>.pt' -f args="--pairs 1024"
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py leak --bank bank.jsonl.gz \
  --playouts-input playout_input/bank.jsonl.gz --playouts playouts.jsonl.gz \
  --decisions-per-game <8 x annotated positions / games> --out leak.md
```

---

## 7f. Playout-labelled positions (`tools/value_labels.py`, P34)

The dataset P34's leaf evaluator is fitted on ([plan](../research/plans/P34_search_centric_evaluator.md),
ledger [`research/datasets.md`](../research/datasets.md)). Stages, each shardable with `--part i/n`:
`positions` (self-play of the generator at `--temperature` 0.25, one decision in `--keep` 8, the
raw state, the generator's top 4 legal moves and its own game result; 5% of games held out, by game),
`label` (every position's 32 worlds to the game's end under the safe rule, plus the sibling slice --
`--siblings` 50,000 positions x 4 candidates x 16 shared worlds with each row's leaf -- through
`ts.PlayoutLabels`) and `merge` (`train.npz`, `holdout.npz`, `siblings.npz` and `manifest.json` with
the D code, generator, engine fingerprint, device, seed and sha256s; it prints the ledger row).
Label columns, the mover's side: `score` (mean of 1 / ½ / 0) and `score_se`, `final_vp` and
`turn_vp` (/20), `turn_defcon`, `swing` (per region, Europe .. South America then Southeast Asia, the
mean swing at its next scoring; NaN where no world scored it) and `swing_frac`.

Throughput (RTX 4090, i9-12900KF, A13+R48's SWA): ~1.5M network rows/s steady -- two labellers
alternate so one steps while the GPU evaluates the other, the forward is a CUDA graph over a fixed
16,384-row batch, and observations travel as exact one-byte codes. The engine's observation
extraction is the limit. A label costs ~7,900 rows, so generation 1 (1.1M positions plus the slice)
is ~2 h. Labels are tied to the engine build (raw state bytes; `label` and `merge` refuse a mismatch).

```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/value_labels.py \
  positions --model 'E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M' --games 20000 --out <dir>
PYTHONPATH=.:build/release .venv/bin/python tools/value_labels.py label --dir <dir>
PYTHONPATH=.:build/release .venv/bin/python tools/value_labels.py merge --dir <dir>
```

`exact` (CPU, after `label`; `--part i/n` per process) solves the exact slice: 2,000 positions of turn
10's last action rounds (7 on), each in the labeller's first 4 worlds, by
`tools/scripts/blunder_bank.exact_value` (dice averaged, the opponent's best reply, a 300k-node
budget a world). `merge` then reports `exact.bias`, the mean of (label - exact value) over positions
solved in every world -- what the labels inherit from the generator's greedy follow-up -- with the
solved count, since the solvable positions are the simpler ones.

`label --keyed` plays the worlds under keyed chance (`bindings/keyed_chance.hpp`): the sibling
slice's candidates then correlate at 0.39 within a world instead of 0.22. It is part of the labelling
recipe, so it takes its own D code.

The D code is derived, not chosen: `merge` hands the labelling settings to
`tools/scripts/run_codes.py`'s `labelling_code`, which finds their code under `labellings` in
`research/run_codes.json` or assigns the next one (`--code D1` only checks it).

**Fitting an evaluator on it** (`tools/train.py --mode value-fit`, `ai/training/value_fit.py`): the
network of `--warmup-checkpoint`, fine-tuned (or with `--value-reinit` the same shape from fresh
weights), `v_win` by cross-entropy toward the mean score, `v_vp` toward the final VP, a small
auxiliary head (turn-end VP and DEFCON, region swings; saved beside, not in, the checkpoint), and
a paired loss on the sibling slice's leaves at `--value-sibling-coef` 10. It evaluates on the held-out
games every `--value-eval-every` steps -- cross-entropy against the labels' own entropy, VP error and
sibling pair accuracy, starting with the initial network's (the RL critic's) numbers -- keeps the best
and stops after `--value-patience` evaluations without a better one. `value_fit_best.pt` is an ordinary
checkpoint: register it and play it as `leaf_net=`. Observations travel as exact one-byte codes
(`ts.ObsCodebook`). A value fit takes its network's A code with `value_only` added and its R code
from the value flags, the dataset among them (`run_codes.py`).

```bash
tools/scripts/check_engine_fresh.sh && TRITON_CACHE_DIR=.triton_cache PYTHONPATH=.:build/release \
  .venv/bin/python tools/train.py --mode value-fit --value-dataset <dir> --warmup-checkpoint <rl.pt> \
  --run-name '<RL model>+A<n>-R<n>' --description "..."
```

---

## 8. Other CLIs and the shared library

| CLI | what it is for |
|:---|:---|
| `tools/behavioral_test.py` | scores checkpoints on the encoded behavioural claims (`ai/eval/claims.py`; answer key in [`docs/behavioral_test_claims.md`](../docs/behavioral_test_claims.md)) |
| `tools/generate_policy_targets.py` | two seat specialists play each other; records each one's policy as a distillation target |
| `tools/fetch_acts_corpus.py` | crawls ACTS tournament game journals into `data/` |
| `tools/scrape_card_strategies.py` | builds Markdown card-strategy guides from twilightstrategy.com into `docs/` (git-ignored for their licence) |
| `tools/scripts/launch_flags.py` | a run's effective flags, and `--diff` against another run |
| `tools/scripts/watch_run.py` | the monitor for a launched run (progress, stalls, exit) |
| `tools/scripts/average_weights.py` | uniform weight averages: a run's late snapshots give its **SWA**, branches of one state a **model soup** |
| `tools/scripts/checkpoint_report.py` | a network's behaviour report (`research/log/per_checkpoint/`), with its searched-leak census (`--leak`) |
| `tools/scripts/check_engine_fresh.sh` | exits 0 if the built engine matches the sources, else rebuilds and exits 1 |
| `tools/scripts/check_doc_links.py` | every Markdown link and anchor, resolved the way Obsidian does |

Most other `tools/scripts/*.py` are single-experiment probes; each one's docstring names the log
entry it serves.

`tools/lib/` (imported by the CLIs; `tools/lib/__init__.py` re-exports nothing, so torch-free
modules stay importable without torch):
- `player_agent.py`: `load_agent` and policy wrappers, including `OnnxAgent`.
- `player_spec.py`: a player as a network plus a typed inference spec, its canonical id and its `load_agent` string; search defaults live here.
- `leaderboard.py`: the leaderboard's records, validation and two-stage fit (stdlib only).
- `batch_tournament.py`: the C++ batch tournament runner and Bradley-Terry MLE; plays any subset of a matchup's pairs and merges parts.
- `parallel_tournament.py`: shards a tournament over worker processes.
- `tournament_evaluator.py`: loss-cause classifier (`classify_game_ending_reason`).
- `self_play.py`: single-game runner and the one writer of `.tslog.json` replays.
- `scoring_formatter.py`: regional scoring formatter.
- `checkpoint_utils.py`: architecture detection and checkpoint discovery.
- `corpus_paths.py`, `data_root.py`: where the corpus and the shared data tree live; content deduplication.
- `corpus_driver.py`: feeds self-play or the converted corpus one decision at a time to a tracker; `pos=` tokens; the E4-view check.
- `jsonl.py`: JSON-lines rows, plain or gzipped, and file SHA-256 -- the one reader/writer of bank files.
- `position_bank.py`: what the bank tools share (a position's seed, a move's name, `exact_mistake`, `--continue-with`).
- `hidden_draws.py`: exact outcomes of a step that draws a card from the PRNG (`step_outcomes`).
- `openings.py`: the named openings.
