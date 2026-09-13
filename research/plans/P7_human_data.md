# P7 — Human data: several thousand strong games, one perspective each

**Status:** queued (conversion can start now; the arms wait for positions to exist)
**Gate:** conversion stage has no gate. Arm 7b (start pool) needs the converted positions;
arm 7c (critic targets) needs 7b's result; arm 7d (the §22 replication) needs only the dataset.
Nothing here blocks P0–P2, which run alongside.
**Needs approval:** none for training; the conversion follows the standing rule for
`tools/lib/ts_replayer_*` — a decision the log does not determine is a bug to diagnose, not a
gap to fill (`AGENTS.md` §4 invariant 11).

## Goal

Use a corpus of ~3,000 high-level games — recorded from one player's perspective, so that
player's hand is known and the opponent's is not — in the order of **distance from the policy
gradient**: instruments first, state distribution second, critic third, policy last and only
after one arm has discriminated whether the earlier failures were the data or the mechanism.

## Why

- The two settled negatives — BC warm-up washes out within 2M steps (`experiments.md` §9.1),
  and an imitation term mixed into RL costs 157–236 Elo (§22) — both used ~266 games of
  uncertain quality and both put the human signal into the **policy loss**, where it is
  fragile: the policy is asked to reproduce moves at positions it cannot follow up from.
  Neither result says anything about human data used to choose *what states the agent trains
  on* or *what the critic believes*.
- Those are the uses that match the measured failures: the critic never sees held battlegrounds
  and so never learns their worth (`ai/eval/battleground_value.py`); games end at turn ~6 so
  late boards are never priced (§18.2); the critic is pessimistic on human boards by 7–10
  points (§17). Human positions are exactly the boards self-play does not reach.
- ~3,000 games × ~175 own-perspective decisions ≈ 0.5–1M labelled decisions. Not enough to
  clone a 12M-parameter policy into a standalone player (AlphaGo's SL policy used 30M
  positions), but plenty at the *card* level — headline choice, what to space, when to score —
  where the decision space is small and strong players agree; and 10× more positions for every
  instrument.

## Change

### 7a. Conversion and instruments (no GPU; the part that is "non-trivial")

Source: presumably the ACTS tournament journals (`tools/fetch_acts_corpus.py` is an existing
crawler into `data/acts_games.sqlite`); the ts-replayer pipeline (`tools/lib/ts_replayer_*`,
`tools/build_human_dataset.py`) is the converter. One-perspective logs mean:

- the recorder's hand is known; the opponent's hand is reconstructed by the z3 solver as now.
  Where a hand is under-determined, the position is still usable for 7b (the engine deals fresh
  after a turn boundary anyway) and for the critic (7c); it is *not* usable for imitation of
  the opponent's decisions — do not sample a hand and pretend the decision was observed;
- every log/engine mismatch is diagnosed, not patched. Budget real time for this: 3,000 games
  will surface more mismatches than 266 did. Run `-m corpus_full` before merging any converter
  change (`CLAUDE.md`);
- *decide before running:* what "high-level" means for the split in the next item — tournament
  round, rating, or a named-player list.

Instruments, all existing, re-run on the new corpus with a **strength split**:
`ai/eval/agreement.py` (agreement rate, `../method/human_play.md`), `critic_calibration.py`,
`battleground_value.py` on real contested positions, and the P0 setup and VOA yardsticks. The
split is the point: if agreement with *strong* players tracks Elo across the existing
checkpoint lineages and agreement with the old corpus does not, the instrument is validated
and the old corpus is explained in one measurement.

### 7b. Human positions as a start pool (state distribution, not policy)

`ai/training/start_pool.py` already captures positions at the last decision before a turn
rolls over, reseeds the PRNG so the next deal is fresh, and filters by `is_salvageable`. Human
turn-boundary positions fit that contract: the board, VP, DEFCON, tracks and discard/removed
piles are public; the one held card per side comes from the reconstruction (or the position
is skipped if it is not determined). Add a loader for converted positions and a flag for the
human share.

Allocation is what sank the earlier mid-game-start arm (50% of envs pulled from the opening;
`experiments.md`, brief §3): here the human share is **5–10% of envs, turns ≥ 4 only**, the
rest of the mix unchanged at turn 1. The agent chooses every move itself; nothing is imitated.

### 7c. Critic-only auxiliary loss on human outcomes

Value targets from the game result on human-distribution boards, from the recorder's
perspective (their observation is exactly constructible), added to the value loss only —
`ai/training/nash_pg.py`, alongside the existing `--inject-*` path but with the policy term
off. Uses P1's categorical target if adopted. Only games whose recording reaches the result
(the old corpus had value targets masked on half of it; check the share here).

### 7d. The discriminator: replicate §22 with the new corpus

The existing `--inject-dataset --inject-every --inject-weight` path, unchanged, at the §22
setting, with the new dataset. One arm. If it still costs Elo at 10× better data, the
mechanism is at fault and only the anchored forms in reserve are worth trying; if it now helps,
the old corpus was at fault and simpler imitation is back on the table. Run it *after* 7b so
the result is read against the right baseline, and never adopt it — it is a measurement.

## Procedure

- 7a: engineering time, then one instruments row per existing checkpoint lineage.
- 7b: 1 arm × 2 seeds × 80M vs the current baseline; confirm at 240M if adopted.
- 7c: 1 arm × 2 seeds × 80M on top of 7b's winner.
- 7d: 1 arm × 2 seeds × 80M, matched to the §22 setting.

## Measure

7a: agreement-vs-Elo correlation by strength split; calibration on human boards before/after.
7b: empty battlegrounds at turn 8 and final-turn distribution (the state-distribution effect),
then Elo *from the opening* separately from Elo overall — the earlier arm won the resumed
regime and lost the opening, and the two must be read apart. 7c: calibration on human boards
*and* on self-play states (the regression to watch for), then Elo. 7d: Elo only; it is a
discriminator.

## Decision rule

- 7b adopt if turn-8 empty battlegrounds fall and opening Elo does not (−20 bound). Kill if
  opening Elo drops — then the share is too high; try 5% once before killing.
- 7c adopt if human-board calibration improves without self-play calibration regressing, at
  Elo not worse than −20.
- 7d: never adopted; its sign decides which reserve entry (human KL anchor vs. plain
  imitation) is promoted, if any.

## Follow-ups

- 7d says "mechanism": promote the **card-level human KL anchor** from reserve.
- 7d says "data": promote **advantage-filtered imitation** from reserve.
- 7b adopted and P3 finds search helps: **expert iteration on human positions** (reserve).
- Reconstructed opponent hands are belief-head targets on human play (reserve, with P5).
- If P4 does not fix setup: the human setups are the fallback anchor (reserve) — yardstick
  first, prior second.

## Runs

(none yet)
