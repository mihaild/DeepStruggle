# Twilight Struggle AI CLI Tools (`tools/`)

This directory contains standalone, reusable developer CLI tools for training, tournament benchmarking, match simulation, replay generation, demonstration dataset creation, and checkpoint inspection.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning (NashPG) or supervised demonstration warmup with live snapshot tournament evaluation.

```bash
# RL training run with blunder-aware rewards and 20-minute snapshot evaluations
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --warmup-checkpoint data/checkpoints/coldwar_net_v2_warmup.pt \
  --duration-seconds 7200 \
  --snapshot-interval-seconds 1200 \
  --reward-scheme blunder_aware \
  --num-envs 512 \
  --eval-games-per-side 50 \
  --eval-opponents random heuristic \
  --output-dir data/checkpoints/my_new_run
```

### Budgeting a run, and what evaluation costs

`--duration-seconds` budgets **training time only**. Snapshot evaluation and start-pool
refreshes are timed separately and excluded, so the flag means what it says.

For an A/B, budget by steps instead:

```bash
# Two arms that are exactly comparable: identical step budget, one flag apart
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 1.0
PYTHONPATH=. .venv/bin/python tools/train.py --arch v2 --train-steps 60000000 ... --start-pool-frac 0.0
```

A wall-clock budget cannot make two arms comparable, because steps/sec depends on the
policy: the arm whose games run longer has costlier evaluations and gets less training.
That is directional rather than random, and it confounded a real 3-hour A/B, whose arms
finished 1024 and 473 iterations on identical settings. `--train-steps` removes it. Because
a step budget says nothing about elapsed time, the progress line projects a wall-clock ETA
from the observed rate plus measured overhead, so a step budget can still be aimed at a
target duration:

```
[1,310,720/1,500,000 steps, ETA 130s] It   20 | Steps: 1,310,720 (15,356 st/s) | ...
```

`--eval-max-snapshot-opponents` (default 4) bounds evaluation cost. Each snapshot is
otherwise added to the opponent list permanently, making evaluation quadratic in run
length; the final evaluation of a 3-hour run faced 14 opponents and took 957s against a
900s snapshot interval, leaving about one training iteration per interval. Baselines from
`--eval-opponents` are never dropped. Pass `0` for the old unbounded behaviour.

Snapshot evaluation runs through the same vectorized path as `tools/tournament.py`
(~30x the one-game-at-a-time loop it replaced).

### Resuming and branching

`--resume` restores the weights, the optimiser moments, the frozen reference policy and the step
counter, so a run continues rather than restarts. It takes a file, a run directory (its newest
state), or `<run_dir>:<steps>` to branch from a particular snapshot:

```bash
# continue where it stopped
--resume data/checkpoints/arm_D_cont_80to160
# branch from the 160M snapshot of a run that went further
--resume data/checkpoints/arm_D_cont_80to160:160038912
```

A resume state is written beside **every** snapshot (`resume_<steps>steps.pt`, 51 MB against the
snapshot's 13), not only at the run's end, so any point of a run stays branchable. Turn it off with
`--no-resume-every-snapshot` if disk matters more than that; a run then keeps only its newest state
and no stretch of it can be re-run from later, which is how one arm came to have no branch point at
160M and could not be replicated there at all.

**Giving a `--seed` that differs from the one the state was written under also re-seeds torch and
numpy**, so the continuation genuinely diverges. Without that the restore would hand back the
original run's action sampling and minibatch order and only the environment deals would differ --
half a seed change, and a seed replicate that understates the variance it exists to measure. The
same seed, or none, restores the stream as before.

Every iteration is logged to `<output-dir>/training_metrics.jsonl` and mirrored to TensorBoard
event files in `<output-dir>/tb/` (`--no-tensorboard` disables the mirror; the JSONL is always
written, and a missing/broken `tensorboard` install only prints a warning). Watch a live run with:

```bash
.venv/bin/python -m tensorboard.main --logdir data/checkpoints/my_new_run/tb
```

### The TensorBoard layout

Four groups, plus the per-opponent evaluations:

| prefix | what it holds |
|:---|:---|
| `progress/` | how fast the run is going: throughput, elapsed time, iteration counter |
| `internal/` | the optimiser's own view — losses, KL, entropy, clip fraction, explained variance, advantage health. Nothing here says whether the agent *plays* well. |
| `endgame/` | what the finished games look like, and the only group with human counterparts: win rate per side, draws, turns, plies, final score, ending mix |
| `strategy/` | is it playing the board well — empty and uncontrolled battlegrounds at turns 5 and 8, salvageability, forced-decision take rates, and the named blunder rates |
| `eval/` | win rate against each fixed baseline, at snapshots only. Strength rather than shape, so it sits outside the four. |

**The x-axis is environment steps, not iterations.** Every experiment here is budgeted and
compared by `--train-steps`, while iteration count depends on `--num-envs` and rollout length —
so indexing by iteration put two directly comparable arms on different x-axes. `total_steps` is
therefore no longer a series (it would be the line *y = x*); `progress/iteration` is logged
instead, and its slope is the steps-per-iteration.

**Several lines per chart, not several charts.** TensorBoard draws one line per *run* per chart,
so the pooled series, the per-winner splits and the human references are written as sibling run
directories under the same tag. `endgame/turn` therefore carries six lines — pooled, games the
US won, games the USSR won, and a human line for each — instead of occupying six charts. The
runs are `.` (the run itself), `won_us`, `won_ussr`, `human_ITS`, `human_won_us`,
`human_won_ussr`. Charts that combine genuinely *different* quantities go through
`add_scalars` instead: `endgame/win_rate` (US / USSR / draw against their human values),
`endgame/ending_mix`, and `strategy/battlegrounds_turn8` (empty against uncontrolled).

`endgame/turn_distribution` and `endgame/ply_distribution` are histograms rather than scalars:
a mean of 6.8 turns is either most games ending near turn 7 or a mixture of turn-3 blowups and
full-length games, and only the second resembles the human corpus.

Human lines come from `ai/itsc_reference.py` — 44,136 completed games from the ITS Junta
results database. Three things deliberately have no line: `mean_victory_points` and
`mean_vp_margin` (ITS does not record the final score), the `defcon1_self` / `defcon1_provoked`
split (ITS records the outcome without the cause, so only the combined `ending_defcon1` can be
compared), and everything under `strategy/`. The ply references are estimates while the turn
references are measurements — see `research/metrics.md` §1.5.1.

Not charted, but kept in the JSONL: `total_steps` (it is the axis) and `mean_terminal_utility`
(exactly `us_win_rate − ussr_win_rate`, both of which are charted). The per-start-turn series
(`game_start{N}/`) are not pre-registered — they exist only under mid-game start sampling, which
no run uses, and are derived on demand so they still appear if it is turned back on.

**Cost.** Measured: ~21 µs and ~57 bytes per scalar write. A 160M-step run logs ~2,442 points,
so one extra series costs 0.05 s and 0.14 MB over an entire run, and a hundred more would cost
~5 s and ~14 MB against a 10,521-second run. The budget is not the constraint; how many charts
a person can read is.

Beyond the loss terms, each iteration records `explained_variance` (`1 - Var(G - V) / Var(G)` for
the win-value head — the primary read on whether the critic is learning), advantage-distribution
health (`adv_std`, `adv_std_raw`, `adv_frac_near_zero`), game length (`mean_turn`, `median_turn`,
`mean_ply`, `median_ply`, `episodes_completed`), the ending-reason mix (`ending_frac_*`), which side won (`ussr_win_rate`,
`draw_rate`, `mean_terminal_utility` — US-positive), and `entropy_fixed_probe`: mean masked policy
entropy on a pool of ~2,000 (observation, mask) pairs frozen at the start of the run, which unlike
the on-policy `entropy` cannot be masked by state-distribution drift.

`mean_ply` / `median_ply` measure game length in the continuous player-slot numeration defined by
[`ai/game_length.py`](../ai/game_length.py): ply 1 is the USSR's turn-1 headline, ply 2 the US's,
ply 3 the USSR's turn-1 AR1, and **154 is a game that played all ten turns out**. Prefer it to
`mean_turn` when comparing lengths. The turn counter answers "which of the ten" and nothing
finer, so a game abandoned at turn 7 AR1 and one that ran to turn 7 AR7 are the same number, and
it carries an artefact at the top of its range: `finish_end_turn` increments the turn and only
then tests `turn <= 10`, so a completed game terminates holding turn **11** while a human replay
log numbers that same game turn 10. Reference points, self-play at temperature 0.1: RandomBot 40,
the 80M arms 99-107, HeuristicBot 115, and human play ~119 (the 44,136 completed games of the
ITS results database; the ts-replayer corpus gives 142, but its finished subset is biased long
-- see `research/metrics.md` §1.5.1).

At every snapshot it also records the win rate against each fixed baseline, overall and per side
(`eval/win_rate_vs_HeuristicBot`, `..._as_us`, `..._as_ussr`), alongside the decisive-decision and
position diagnostics. Snapshot *opponents* are deliberately excluded: they are renamed every
interval, so each would start a series that stops one interval later.

Three things are deliberately **not** logged, because a series that cannot vary is worse than an
absent one — it reads as a measurement:

* **Auxiliary losses whose term is switched off.** `belief_loss`, `oracle_loss` and `distill_loss`
  are v4-only, `defcon_risk_loss` needs `--defcon-coef`, `inject_loss` needs `--inject-dataset`.
  On an ordinary v2 run all five were a flat zero line for the whole run.
* **Per-start-turn breakdowns when no start pool is in use.** `--start-pool-frac` defaults to 0, so
  every game starts at turn 1 and `game_start1/*` duplicated `game/*` exactly. They reappear
  automatically when episodes actually start at more than one turn.
* `steps_per_sec` is the rate **since the previous iteration**. The lifetime average is kept as
  `steps_per_sec_avg`, and is the one to ignore on a resumed run: its clock is rewound to include
  the previous leg, so a run doing 7,123 steps/s can report 9,930.

---

## 2. `tools/tournament.py` (Unified Tournament & Head-to-Head Evaluator)
Fast vectorized tournament and matchup evaluator (300–800 games/sec). Adapts automatically based on the number of models passed:

### A. 2-Model Head-to-Head Matchup Mode:
```bash
# Evaluate model against HeuristicBot for 100 games (50 US / 50 USSR) with loss causes
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --models data/checkpoints/.../snapshot_3602s.pt heuristic \
  --games-per-side 50
```

### B. Multi-Model Round-Robin Tournament Mode:
```bash
# Run a 1,000-game-per-matchup round-robin tournament across all checkpoints in a directory:
PYTHONPATH=. .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/run_v3_20260827_205207 \
  --games-per-side 500 \
  --anchor-model HeuristicBot \
  --anchor-elo 1500.0 \
  --output-report data/checkpoints/run_v3_20260827_205207/massive_tournament_report.md
```

---

## 3. `tools/play_match.py` (Unified Match Runner & Replay Generator)
Plays matches between any pair of agents, supports two distinct checkpoints, provides interactive CLI terminal play, and generates standardized `.tslog.json` replays for the Web Workbench.

```bash
# 1. Pit two different neural checkpoints against each other:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us data/checkpoints/run_v3_20260827_205207/snapshot_3602s.pt \
  --ussr data/checkpoints/run_v2_blunder_aware_9h/snapshot_21601s.pt \
  --game-id v3_vs_v2

# 2. Interactive Terminal Play (Human vs AI Bot):
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us human \
  --ussr heuristic

# 3. Rich Strategic Commentary & Regional Scoring Breakdown:
PYTHONPATH=. .venv/bin/python tools/play_match.py \
  --us strategic \
  --ussr event_heavy \
  --commentary

# View any generated replay in the Web Workbench at:
# http://localhost:8000/?replay=<filename>.tslog.json
```

---

## 4. `tools/generate_dataset.py` (Vectorized Demonstration Dataset Generator)
Charns out thousands of games in parallel using 500 C++ environments and multi-temperature exploration schedules, dumping compressed `.jsonl.gz` datasets for supervised BC warmup training.

```bash
PYTHONPATH=. .venv/bin/python tools/generate_dataset.py \
  --total-games 5000 \
  --batch-size 500 \
  --output data/datasets/warmup_<generator>_<n>.jsonl.gz
```

---

## 5. `tools/inspect_checkpoints.py` (Checkpoint Registry Inspector)
Scans `data/checkpoints/` and displays all saved models, file sizes, modification timestamps, and detected network architectures (`ColdWarNet` V1, V2, or V3).

```bash
PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py
```

---

## 6. `tools/download_ts_replayer.py` (Human Game Corpus) and the Converter

> Every discrepancy found between these logs and the engine -- score reconciliation, the Shuttle
> Diplomacy/Japan log fault, what the hand solver may and may not supply, and what the 300 files
> actually contain -- is recorded in
> [`research/experiments_replayer_conversion.md`](../research/experiments_replayer_conversion.md).

Human Twilight Struggle games, played by people on the Playdek/Steam app and uploaded to
ts-replayer.fly.dev, turned into engine decisions. The downloader fetches each replay's four
JSON islands once and caches them under `data/datasets/ts_replayer/<id>.json.gz`; it throttles
to one request a second and skips anything already on disk, so a re-run costs nothing.

```bash
PYTHONPATH=. .venv/bin/python3 tools/download_ts_replayer.py --out data/datasets/ts_replayer
```

The conversion lives in `tools/lib/` and is *verified*, not merely parsed. Every entry is
rebuilt from the position the log states, driven through the engine as MicroActions, and the
resulting board compared against the log's own next board -- so a mis-parsed entry surfaces as
a mismatch on that entry rather than passing silently into the dataset. Nothing is forced and
nothing falls back to a heuristic approximation: an entry the log does not determine is a
failure to diagnose, not a guess to paper over. Current state of the corpus: **300 of 300
games convert in full**, 29,820 of 30,620 entries (97.4%), 144,844 decisions.

- `tools/lib/ts_replayer_parse.py`: the log's grammar -- entries, sections, influence moves,
  die rolls, discards, reveals, headlines, and the country/card name tables.
- `tools/lib/ts_replayer_convert.py`: the driver. Turns each entry into the queue of decisions
  the engine asks for (`pq` for Ops, `eq`/`evq` for events), steps the engine, and reconciles
  the outcome against the log. Also holds the small, individually diagnosed lists of entries
  the log itself gets wrong (`_KNOWN_SCORE`, `_LOG_MISCOUNTED`, `_INVALID_PLAYS`), plus one
  fault recognised by rule rather than listed: where Asia is scored with Shuttle Diplomacy in
  play and the USSR holds Japan, the log keeps a superpower-adjacency bonus the card has
  removed and pays the USSR 1 VP too many. The engine takes the log's score and it is
  adopted from there on (`_shuttle_japan_asia_miscount`). The log's number is kept on purpose:
  the players were reading the app's score, so that is the position they decided against and
  the one training data must carry. It holds for 2 of the 300 games downloaded so far, and a
  rule covers games nobody has downloaded yet where a list cannot.
- `tools/lib/ts_replayer_hands.py`: the hands, which the log never states in full. Both hands
  for a whole game are solved at once as a constraint problem over z3 (MIT), from the rules --
  hand size, carry-over, spent cards being in the discard pile until a reshuffle, scoring cards
  that cannot be held past a turn, what a trap or an empty hand proves -- with the preference
  heuristics as soft clauses. Every hand in the corpus comes out exactly the size the rules
  deal (4,464 of 4,464). z3 is optional; without it the converter falls back to per-turn
  heuristics and `Conversion.hands_solved` is False.

The entries that do not convert are turns whose recording stops part way: 731 of the 800 are
genuine fragments (the log cuts off mid-turn and that turn's card lists are short to match),
and the other 69 are five games whose file ends on an announced-but-unwritten round.

Two rules of the road, both learned the hard way:

- **Never force a decision the log does not state.** The corpus is training data for a model
  meant to learn human play; an invented choice teaches it something no human did.
- **The engine is the reference.** Where the engine and a log disagree, the log is at least as
  likely to be wrong (`_LOG_MISCOUNTED` exists for exactly this), so diagnose before changing
  either -- and engine changes are the user's call.

---

## 8. `tools/build_human_dataset.py` (Human Corpus BC Dataset)

Turns the ts-replayer corpus into behaviour-cloning data.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/build_human_dataset.py
```

Writes `data/datasets/human_corpus/` (~1.2 GB, git-ignored): one memory-mapped `.npy` per column
plus `meta.json`, read by `ai.training.human_corpus_dataset.HumanCorpusDataset`, whose
`stream_batches` mirrors `WarmupDataset`'s interface.

Observations are stored materialised rather than re-derived from a seed, because a human game has
no seed that replays it -- the dice come from the log and the hands are solved -- so the conversion
is the only thing that reproduces one, at about a second a game. That gives up the self-play
format's forward compatibility, so **rebuild it after any engine change**.

**Value targets are masked on unfinished games.** Roughly half the corpus stops mid-game (the
recording ends, not the game), and those positions have no outcome. The `has_outcome` column is 0
there; a trainer must drop them from the value loss and keep them in the policy loss. Current
build: 280 games, **144,844 samples, 84,073 with a value target**, 0 conversion failures.

---

## 7. Shared Helpers Library (`tools/lib/`)
Contains internal simulation, evaluation, and logging modules imported by the CLI tools:
- `tools/lib/player_agent.py`: Unified agent loader (`load_agent`) and policy inference wrappers.
- `tools/lib/batch_tournament.py`: High-throughput C++ batch tournament runner and Bradley-Terry MLE solver.
- `tools/lib/tournament_evaluator.py`: Diagnostic loss cause classifier (`classify_game_ending_reason`).
- `tools/lib/self_play.py`: Single-game trajectory runner.
- `tools/lib/scoring_formatter.py`: Regional scoring calculation formatter.
- `tools/lib/checkpoint_utils.py`: Architecture detection and checkpoint discovery utilities.
