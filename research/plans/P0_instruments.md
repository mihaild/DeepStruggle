# P0 — Instruments

**Status:** queued (third revision 2026-09-11, after the single-layout refactor)
**Gate:** none. Everything after this is read through these, so they come first.
**Needs approval:** none. No engine, bindings, observation or trainer change remains in it.

## Goal

Four measurements that the later steps are judged by, taken once per checkpoint so every later
arm has a before/after. Two are new probes, two are existing instruments pointed at new states.

The baseline is **not** one checkpoint. A probe number is only comparable inside a
(layout, engine, step-budget) cell, so the baseline is a table (*Procedure*).

## Why

The goal is "no simple mistakes", and the mistakes §25 actually names are not measured by
anything in `ai/eval/`:

- setup that ignores Poland / West Germany;
- **the one the owner calls the gap to a mediocre human: H2 @160M can play a tactic when the card
  is in front of it and cannot choose which card to spend it on.** Holding two cards it must not
  play, it spent UN Intervention on the one that could have spaced itself, and was left having to
  play the other.

Voice of America exposure was a fifth probe here and has been **moved to `reserve.md`**: it is an
accurate description of a real mistake, but the model does not yet contest battlegrounds at all,
and a probe measuring how it defends a foothold it never takes measures nothing.

Elo does not move for any of these (`experiments.md` §4.4: a behaviour can be worth two points of
win rate and still be the thing a human notices first). And P2 spends an arm on chance-aware
targets; whether that is worth an arm depends on how much of the return variance is chance in the
first place, which nobody has measured.

## What is true now, and what it does to this plan

This file was written against v2.2 and arm G. Both are gone. The reset:

- **The starred-card bug** — a starred card spent for Operations was deleted from the game, for
  380 of the repo's 389 commits (`cff2344`, `25d9b70`) — changes the decision stream. Every Elo
  and every probe number in §1–§24 is on a different ladder.
- **There is one observation layout, v2.3 (3824).** legacy, v2.1 and v2.2 are gone along with
  every argument that named one. Arms A–G cannot be loaded at all, so the first version of this
  file's entire baseline table is void.
- **The current arms are H (80M), H2 (80/160/240M) and I (80M).** H2 @240M is the strongest at
  93.0% against the anchor (§25.1).
- `temp_cards` is gone (`49ed564`); the chance node is named (`ctx().pending_roll_type`,
  `ctx().roll_actor`) (`712bce4`). Probe 3 below used `temp_cards[1]` and no longer can.
- Europe Control is its own recorded ending (`430ba9b`), and game length is measured in **plies**,
  not turns (`53f9c1c`).
- `../log/variance_and_noise.md` retires the claim that the corrected engine lengthens games — H2 does not
  replicate H's game shape. Nothing here leans on it.

**Two things in the previous revisions survive unchanged**, and they are what probes 3 and 4 are
built on:

1. The batch runner still drains chance inside C++ *whether or not* `auto_advance` is set — now
   documented in `../method/running_experiments.md` (*the Python chance-drain loop is not worth moving into C++*) — so
   the pre-deal node is still not reachable through it, and probe 3 still needs its own driver.
2. The nested variance decomposition still needs no engine or bindings change: `rng_state` is
   still read/write, `set_card_location` is still there, and `get_state` still returns a mutable
   `reference_internal`.

The third — seven layout-blind call sites in `ai/eval` — is **fixed**, by removing the layout
argument rather than by converting them. See §1.

## Change

### 1. Make the instruments read the observation the checkpoint was trained on — **mostly done**

This was the largest item in the plan and most of it has since landed, by a route that removes
the failure rather than guarding it: **there is one observation layout, and no argument that
names one.** `ts.extract_observation(state, perspective)` and `ts.VectorizedBatchRunner(n, seed)`
take no `layout`; `TsVectorizedEnv` takes no `layout` or `obs_flags`. The seven layout-blind
`ai/eval` call sites this file used to list are correct by construction — there is nothing left
for them to default wrongly.

The other half landed with it. Every architecture now checks its input width in
`extract_features`; `bindings.ts_env.check_obs_width` checks a model against the engine before a
probe runs, and `ai.models.coldwar_net_v2.check_checkpoint_layout` refuses a checkpoint from a
retired layout by its own weights. `tests/training/test_probe_observation_layout.py` and
`tests/training/test_layout_forward.py` hold both directions.

**What is left of this step:**

* `ai/eval/handle.py` — still worth having, but smaller than specified: an `EvalHandle` now
  carries only `(model, device, name, steps, base_commit)`, because layout and flags are no
  longer per-checkpoint facts. It is a convenience for the probe suite's row labelling, not a
  correctness measure. Build it in step 7 with the CLI, not ahead of everything else.
* `ai/eval/rollout.py` (below) — unchanged, and still required for probe 3.

### 2. The probes

**Probe 1 — setup.** `ai/eval/setup_probe.py`.

`init_new_game` deals hands and then leaves the state in `Phase::SETUP` at a `POINT_NODE`
(`state_machine.cpp:148-156`): USSR places 6 in Eastern Europe, then US 7 in Western Europe, then
US 2 bonus in any country it already occupies. The probe is exactly 15 batched forwards from a
fresh runner — no rollout at all.

- n = 2,000 seeds (four batches of 512), temperature 0.1, matching self-play.
- Report: the 84-wide placement histogram per side; **P(Poland ≥ 3 | USSR)** (`cid 15`);
  **P(West Germany ≥ 4 | US)** (`cid 7`); the distribution over distinct 6- and 7-placements
  (top 10 plus a coverage count); placement entropy per side. Every rate as a Wilson band via
  `ai/stats.py`, per `71739f9` — a point estimate on 2,000 seeds invites a comparison the noise
  does not support.
- *Decided:* **yes, condition on Europe Scoring (`cid 2`) in hand.** It is free — the deal
  precedes setup, so `state.get_card_location(2)` is readable at the setup node — reported as a
  three-way split (USSR hand / US hand / deck) with each cell's n printed. It is the one hand
  feature that plausibly changes the answer, and a flat split is itself the result.
- Human yardstick: read the **first log entry's board** straight out of `ts_replayer_parse`, not
  out of a conversion, so every corpus game counts including the ~half whose recording stops
  (`ts_replayer_convert.py:1508-1512`). **Caveat to print with the number:** bid/handicap
  influence is folded into those setup totals (`ts_replayer_convert.py:199-206`), so the human
  Poland and West Germany figures are inflated relative to a no-bid game; split the yardstick by
  bid = 0 vs bid > 0.

**Probe 2 — chance-variance decomposition.** `ai/eval/chance_decomposition.py`.

The original design replayed forced dice through the batch runner. The runner drains chance in
C++ unconditionally and takes no forced rolls, so that route needs a bindings change and an
owner approval — and it is not needed. A nested decomposition by the law of total variance gets
the same shares from two controls that are already exposed:

- *Deals fixed* — at each turn boundary, force the newly dealt cards to the ones a recorded
  reference rollout drew, via `set_card_location` on the state `get_state(i)` returns.
- *Dice free / fixed* — `state.rng_state` is read/write; a fresh value re-rolls everything
  downstream.

Four conditions per saved state, nested deal ⊃ dice ⊃ policy:

| # | deals | dice | temperature | gives |
|:--|:--|:--|:--|:--|
| A | reference | reference | 0 | reproducibility check: variance must be exactly 0 |
| B | reference | free | 0 | `Var_dice` |
| C | reference | free | 0.1 | `Var_dice+policy` |
| D | free | free | 0.1 | `Var_total` |

Shares of `Var_total`: dice = `Var_B / Var_D`; policy = `(Var_C − Var_B) / Var_D`;
deal = `(Var_D − Var_C) / Var_D`. The outcome is the terminal utility from the saved state's
mover's perspective; variance is taken within a state over replicates, averaged over states, with
a bootstrap CI over states.

- *Decided:* **N = 16 replicates, 1,000 saved states** over turns 3–9, sampled with
  `position_diagnostics`' own sampler and filtered by `is_salvageable` — a decided position has
  no variance to decompose and would drag every share toward zero. 16 is enough for a share
  against a coarse 20%/40% threshold and not enough for a small difference between checkpoints;
  the decision rule only asks for the former.
- *Cost, measured not guessed:* 128 envs ran at ~9,200 env-steps/s on 4 CPU threads — 128 full
  games in ~10 s. 1,000 × 16 × 4 = 64,000 partial rollouts is **~1.5 h on 4 CPU threads**, less
  with more, and **it never touches the GPU**, so it runs beside a training job.
- *Methodological caveat, decided before running:* a forced draw can be unavailable, because the
  counterfactual rollout may already have played, discarded or removed a card the reference drew.
  Rule: force the reference's draw where the card is still in the counterfactual's draw deck,
  otherwise draw normally, and **report the fraction of forced draws that could not be honoured**
  beside every share. Below ~90% fidelity the deal share is a lower bound and must be reported
  as one.

**Probe 3 — pre-deal calibration.** Extend `ai/eval/critic_calibration.py`.

- New parameter `sample_nodes: Literal["decision", "pre_deal", "both"] = "decision"`, and the
  measurement moves onto `ai.eval.rollout`, without which the node is invisible.
- The pre-deal afterstate is the chance node with `ctx().pending_roll_type == RollType.TURN_CLEANUP`
  — a named field since `712bce4`; the `temp_cards[1]` this file cited in its first revision no
  longer exists. `decision_player` is `NONE` there, and `critic_calibration.py:120`'s
  `if side == 0: continue` is exactly what drops it, so evaluate **both perspectives** and store
  both rows.
- Sizing: chance nodes are 10.1 per game and TURN_CLEANUP is 36% of them (`../method/running_experiments.md`), so ~3.6
  per game — 500 games gives ~1,800 nodes, ~3,600 rows.
- **`classify_ending` must be replaced, not extended.** It predates the corrected engine and
  cannot see Europe Control, wargames, held scoring, or self-inflicted versus provoked DEFCON-1 —
  the taxonomy the trainer already emits as `ending_frac_*`. Reuse that taxonomy so probe 3 and
  the training logs name the same endings.
- Report the calibration buckets, Brier and `corr(v_win, realised)` **split by node kind**, so
  P2's "the critic prices the board before it sees the deal" claim has a curve to move.

**Probe 4 — disposing of a card you must not play.** `ai/eval/sequencing.py`. The reason this
step has a fourth probe: §25 names turn sequencing as *the* gap to a mediocre human, says "it is
what the P0 probes and P4 exist to measure", and none of probes 1–3 measure it.

**The position it comes from** (`h2_160M_selfplay_20260405`, turn 10, USSR to play), read
correctly this time. The USSR held UN Intervention, Tear Down this Wall and Grain Sales to
Soviets. **Both US cards are DEFCON-suicide here** — that is not the distinction. The distinction
is the *exit* each one has:

| card | ops | at this space box (3+ required) |
|:---|---:|:---|
| Tear Down this Wall (`cid 96`) | 3 | **spaceable** — can be spent on the space track, event never fires |
| Grain Sales to Soviets (`cid 67`) | 2 | **not spaceable** — too few Ops for the next box |

A card you must not play has exactly three exits: space it, run it through UN Intervention, or
hold it at end of turn. Tear Down this Wall has all three; Grain Sales has two. So UN Intervention
— the scarce one — belongs on the card with fewer exits. The model spent it on Tear Down this
Wall, which could have spaced itself, and was then left holding Grain Sales with the hand to
empty. Correct play: **space Tear Down this Wall**, then either run UN Intervention on Grain
Sales, or play UN Intervention for its own Ops and hold Grain Sales at end of turn.

**And UN Intervention costs a card you may not know you are spending.** Its event plays an
opponent card for Operations *in the same action round*, so it consumes two cards in one AR —
which is exactly the one card you would otherwise have held back at end of turn. Using it does
not just spend the card; it removes the "hold it" exit for everything else in the hand. That is
what turned a recoverable position into a forced Grain Sales.

Three measures, in order of how little judgement they need:

1. **UN Intervention spent on a card that was not the suicide card.** The hand contains a
   DEFCON-suicide card (`ai/eval/blunders.py`'s `defcon_suicide_cards(state, player)`, which is
   already position-aware — Nuclear Subs, coupable influence, Star Wars, Five Year Plan), UN
   Intervention (`cid 32`) is in hand and legal, and it is spent on some other card. No judgement
   about the best line: the problem was in hand, the tool was in hand, the tool went elsewhere.
2. **UN Intervention spent on the spaceable one of two.** The hand holds two DEFCON-suicide
   cards, one spaceable and one not, and UN Intervention goes to the spaceable one — the §25
   error exactly. Strictly a subset of (1) and the sharpest single number, because the
   alternative is not a matter of taste.
3. **The suicide card is played raw anyway**, with UN Intervention still in hand — the outcome
   the first two are upstream of. Already counted by
   `blunders.py`'s `defcon_suicide_with_alternative`; report it here beside them so the chain is
   visible in one place.

All three as Wilson bands with their opportunity counts, over 500 self-play games.

**Spaceability is read from the engine, not reimplemented.** `SpaceRace::can_attempt_space` is
not exposed to Python, and it should not be added just for this: step the state to the card's
`SELECT_PLAY_MODE` node (`ai/eval/positions.py:step_to_play_mode`) and read whether flat action
112 (`PLAY_MODE_START + PlayMode.SPACE`) is legal. That is the engine's own rule, including the
Ops modifiers `can_attempt_space` folds in — printed Ops is not the test.

**Constructed positions**, added to `ai/eval/claims.py` so they run inside the behavioural suite:
the §25 hand at a space box requiring 3, asserting the policy does not put UN Intervention on
Tear Down this Wall; and a control where both suicide cards are unspaceable, where spending UN
Intervention on either is fine and the claim is only that it is spent on one of them.

Baselines here, not gates. These are the numbers a later strategy arm has to move.

**Built** as `ai/eval/sequencing.py` (`fa8be41`), with two rules rather than three:
`un_intervention_off_target` and `un_intervention_on_spaceable`, counted through `BlunderCounts`
so the Wilson bands, the examples and the metric keys are the existing machinery rather than a
second copy of it.

Two things came out of building it, and both survive the implementation.

*Spaceability has to be asked one decision earlier than it is used.* At the companion node a card
action means "UN Intervention takes this one", so stepping it there answers a different question
and reports every card as unspaceable. The probe keeps the last ordinary card-selection node per
environment and asks there, where playing the card is still the live question. Two independent
implementations of this probe both hit that, one of them scoring zero opportunities on a
hand-built position that contained one before it was found.

*`blunders.defcon_suicide_cards` does not list Tear Down this Wall*, though the §25 position turns
on it: the US free coup in Europe at DEFCON 2 drops DEFCON, and the USSR, as the phasing player
that spent it for Ops, loses. The probe uses that function as the single definition of "a card you
must not play here" rather than keeping a second list, so **the position the probe was written
from is not one the probe would currently flag**. Whether the list should gain the card is a
question about `blunders.py` and the `defcon_suicide_with_alternative` rate it already publishes,
so it is recorded here rather than changed. `tests/training/test_sequencing_probe.py` uses Duck
and Cover -- 3 Ops, US-associated, on the list -- so the spaceable/unspaceable pair is structurally
identical.

**First numbers, 300 games a checkpoint**, from a run of the two rules plus a third measure that
was tried and is *not* in the merged implementation (a card that must not be played is played for
Operations anyway, with UN Intervention still in hand):

| | H2 @240M | H2 @480M |
|:---|---:|---:|
| UN Intervention spent off target | 12.5% (9/72) | 14.1% (11/78) |
| spent on the spaceable one of two | 10.0% (1/10) | 20.0% (1/5) |
| *(tried, not merged)* suicide card played raw | 38.7% (12/31) | 77.8% (14/18) |

**The second rule is the sharpest and is not readable at this sample size.** It is the §25 error
exactly, and it arose ten times in 300 games at 240M and five at 480M -- bands of [1.8, 40.4] and
[3.6, 62.4]. Either the baseline row runs thousands of games or the configuration is simply rare
in self-play; decide which before quoting it.

The first rule reads the other way round and is worth saying plainly: when UN Intervention *is*
spent and a card that must not be played is in hand, it goes to that card 86-88% of the time. The
model is not misdirecting the tool. The third measure, if it is ever added, is the one that says
what happens instead -- it joins up with `dcedb8a`'s finding that 82.7% of DEFCON-1 endings are
provoked -- but its bands touch at one point over 31 and 18 opportunities, so it is suggestive
and nothing more.

### 3. Existing instruments, on the same checkpoints

`battleground_value.py` (§12.1 perturbation probe), `position_diagnostics.py` (empty
battlegrounds at turn 8, ply distribution, DEFCON-1 share), forced-win take rate as a floor. All
three can be trusted now: `position_diagnostics` was fixed by `6220f32`, and the perturbation
probe by the single-layout refactor. Any number any of them produced on a v2.x checkpoint before
those two changes is void.

### 4. One CLI

`tools/probe_suite.py --models <paths and baselines> --probes setup disposal chance
calibration existing --output-json <path> --device cpu`, one row per model. One entry point, so a
later arm reports the same numbers by running the same command.

## Implementation order

Seven commits. Each one is separately reviewable and separately revertible, and each has an
acceptance check that fails loudly if it did not work. Nothing measures anything until step 2 has
landed, which is why it is not last.

**0. Build the engine in a worktree of your own** — not code, but nothing runs without it.

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
tools/scripts/check_engine_fresh.sh          # rebuilds and stamps, exit 1; rerun for exit 0
```

*Accept when:* the script exits 0, `ts.OBS_SIZE == 3824`, and
`NeuralAgent.from_checkpoint('data/checkpoints/arm_H2_cont_160to240/snapshot_final.pt')` loads.
~10 min, mostly compile. pytest refuses to run against an unstamped build, so this is not
skippable.

**1–2. The layout work — landed.** One layout, no `layout` argument anywhere, width guards on
every architecture, and `check_obs_width` / `check_checkpoint_layout` refusing a retired
checkpoint by width rather than letting it misread. The probes are correct by construction rather
than by conversion. What remains of `EvalHandle` is row labelling and belongs to step 7.

**3. `ai/eval/rollout.py`**, with `tests/training/test_rollout_driver.py`. It now gates one
probe rather than two, so it can equally follow step 4 — take it whenever probe 3 comes up.

*Accept when:* driven greedily from the same seeds with chance drained the default way, the
driver's terminal utilities and action streams are **identical** to `VectorizedBatchRunner`'s over
64 games. That equivalence is the whole warrant for using it in probe 3, so it is the test
that matters; a driver that merely "looks right" reintroduces the 25-point disagreement
`../log/measurement_bugs.md` records from the last time a single-state path diverged from the
batched one. ~2 h.

**4. Probes 1 and 4 — landed.** setup and card disposal. `ai/eval/setup_probe.py`,
`ai/eval/sequencing.py`, additions to `ai/eval/claims.py`, and the corpus setup reader. These two
come first among the probes because they are the cheapest to run (15 batched forwards, and 500
games), they need only the C++ runner, and they produce the numbers that gate P4 — the step most
likely to jump the queue.

*Accept when:* `random` places roughly uniformly over the legal setup countries and `heuristic`
does not, which is the sanity check that the probe reads placements rather than noise; the human
yardstick is stable across a re-run of the corpus reader; and the disposal probe reproduces the
§25 position as a positive — replay `h2_160M_selfplay_20260405` to turn 10 and confirm it is
flagged, since a probe that does not catch the case it was written from catches nothing. ~4 h.

**5. Probe 3** — pre-deal calibration. Needs step 3's `observe_node`. Includes replacing
`classify_ending` with the trainer's `ending_frac_*` taxonomy.

*Accept when:* the ending distribution the probe reports on H2 @240M self-play reproduces §25.1's
table (mean ply ~105, DEFCON 1 ~34%, Europe Control 0.0%) — an independent path onto numbers that
are already published is the cheapest available check that the new taxonomy is wired correctly.
~3 h.

**6. Probe 2** — chance decomposition, including the draw-fidelity metric. Heaviest to write and
the only one with a real runtime.

*Accept when:* condition A returns **exactly zero** variance. If it does not, the replay is not
deterministic and every share below it is meaningless, so this is a hard gate rather than a
diagnostic. ~3 h to write, ~1.5 h to run on CPU.

**7. `tools/probe_suite.py`, the baseline run, and the write-up.** One CLI over all five probes
plus the existing instruments; run it over the *Procedure* table; write `experiments.md` §26 with
the baseline rows; add the columns to the standard eval row in `../method/measurement_tiers.md`; `git rm` this file
and drop its row from `plans/README.md`.

Steps 1–2 have landed. Step 3 is the one left that has to be right; 4–6 are independent of each
other and can land in any order, or in parallel. Total ~1.5 days including the write-up, none of
it on the GPU.

## Procedure

### Which checkpoints

Rate the last four snapshots of each run and report the median (`../log/variance_and_noise.md`); all four arms
below have a full 5M-step snapshot series.

| row | checkpoint | layout | steps | base commit |
|:---|:---|:---|---:|:---|
| v2.3 baseline | `arm_H_v23_corrected/snapshot_final.pt` | v2.3 | 80M | `32902a3` |
| v2.3 seed B | `arm_H2_v23_seedB` @ 80M and 160M | v2.3 | 80M / 160M | `32902a3` |
| **strongest** | `arm_H2_cont_160to240/snapshot_final.pt` | v2.3 | 240M | `71739f9` |
| KL off | `arm_I_no_kl/snapshot_final.pt` | v2.3 | 80M | `c391f2e` |
| floor | `heuristic`, `random` | — | — | — |
| yardstick | human corpus; ITS results for game shape | — | — | — |

**The three base commits do not split the ladder — checked, not assumed.** The only engine or
bindings source to change between `32902a3` and today is `scoring.cpp`, and the change sets
`effect_bits::EUROPE_CONTROL_WIN` at a point where the game is already over: same VP, same phase,
no decision affected. So H, H2, H2-continued and I share a decision stream and are mutually
comparable, even though `check_engine_fresh.sh`'s content hash differs across them. Record each
row's `base_commit` in the output anyway — the field exists on every run's `metadata.json` now,
and it is what makes this checkable next time instead of arguable.

Comparisons that are licensed: H @80M vs H2 @80M (seed replication, known to be worth ~nothing —
4.7 Elo); H2 @80M vs @160M vs @240M (budget); **I @80M vs H2 @80M (the KL term, the one arm I is
running to answer)**. Not licensed: anything against arms A–G, on either ladder.

### Arm I

`arm_I_no_kl` — v2.3, cold start, `--eta 0` (NashPG KL penalty off), 80M steps, seed 20260921,
otherwise H2's recipe, so the only difference from H2 @80M is the KL term. **Finished**, and its
row can be taken from its last four snapshots like any other.

Probes run on **CPU** and never write into a run directory; output goes to
`research/probe_baseline_20260911/`.

### Recording

Commit the probes with the baseline table in `experiments.md` in the same change, per this
directory's maintenance rule, and add the numbers to the standard eval row in `../method/measurement_tiers.md`. Then
`git rm` this file and delete its row from `plans/README.md`.

## Measure

Per row: `P(Poland ≥ 3 | USSR)`, `P(West Germany ≥ 4 | US)`, setup entropy; dice / deal / policy
shares of return variance with the draw fidelity; pre-deal vs decision Brier and correlation; the
three card-disposal rates and the constructed-position pass rate; plus the existing instruments.
Rates as Wilson bands. No Elo.

## Decision rule

- Setup probe well below the human rate (expected): P4 stays in the queue and may move ahead of
  P2.
- **Dice + deal share = 1 − policy share.** `≥ ~40%`: P2 runs. `≤ ~20%`: P2 is demoted to reserve
  and its arm goes to P4/P5. Between: P2 keeps its place but behind P4.
- Card disposal: baseline only. If the constructed positions pass **and** the self-play rates are
  high, the model knows the tactic and cannot time it, which argues for P4's macro-action credit
  over P6's capacity. If the constructed positions also fail, it does not know the tactic, and
  that is a different problem — closer to P1's value head than to P4.

## Follow-ups

- Every later step reports these numbers. Add them to the standard eval row in `../method/measurement_tiers.md`.
- ~~Queue the BC warmup layout fix~~ — **fixed by the single-layout refactor.** `WarmupDataset`
  and `tools/generate_dataset.py` re-extracted at the legacy default and passed no layout down;
  with one layout there is no default to be wrong. Arms H, H2 and I were cold starts and were
  never affected.
- If §1.3's width assertion fires anywhere outside the tests, that call site was measuring noise;
  note where in the log entry.

## Runs

(none yet)
