# Twilight Struggle AI CLI Tools (`tools/`)

Standalone developer CLIs for training, tournament benchmarking, match simulation, replay
generation, demonstration dataset creation, and checkpoint inspection. Never invoke the training,
tournament or match machinery from an ad-hoc script -- go through these.

---

## 1. `tools/train.py` (Unified Training Pipeline)
Launches neural network reinforcement learning (NashPG) or supervised demonstration warmup with
live snapshot tournament evaluation.

```bash
# RL training run with blunder-aware rewards and 20-minute snapshot evaluations
TRITON_CACHE_DIR=.triton_cache PYTHONPATH=. .venv/bin/python tools/train.py \
  --arch v2 \
  --warmup-checkpoint <warmup.pt> \
  --duration-seconds 7200 \
  --snapshot-interval-seconds 1200 \
  --reward-scheme blunder_aware \
  --num-envs 512 \
  --eval-games-per-side 50 \
  --eval-opponents random heuristic \
  --output-dir <run-dir>
```

### Budgeting a run

`--duration-seconds` budgets **training time only**: snapshot evaluation and start-pool refreshes
are timed separately and excluded, so the flag means what it says.

For an A/B, budget by steps instead -- `--train-steps`. A wall-clock budget cannot make two arms
comparable, because steps/sec depends on the policy: the one whose games run longer has costlier
evaluations and so gets less training, which biases the comparison in a fixed direction rather
than a random one.

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

# View any generated replay in the Web Workbench at:
# http://localhost:8000/?replay=<filename>.tslog.json
```

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
PYTHONPATH=. .venv/bin/python tools/download_ts_replayer.py
```

It caches into `~/.cache/ts_ai/ts_replayer` (`$XDG_CACHE_HOME` honoured), **not** into the
repository, so every checkout and every git worktree shares one copy -- a worktree has its own
empty data directory, and a corpus kept there would be re-fetched in full for bytes already on the
machine. `tools/lib/corpus_paths.py` resolves the location: `$TS_REPLAYER_CORPUS` first, then an
existing in-repo copy for checkouts that predate this, then the shared cache. `--out` overrides it.

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
