# Bookkeeping — how to record an experiment so it can be found again

Written on 2026-09-16, after the record failed a specific test: *"we had a lot of comparison
between pooled and non-pooled arms, and it is hard to find now."* It was hard to find because the
evidence sat in one log file, two plan files, three git-ignored tournament reports and nine
`metadata.json` descriptions, and no document anywhere held the answer. Two of the arms were not
in any table at all, and one word — "pool" — named four unrelated mechanisms.

The rule that follows from that is short: **an experiment is not recorded until the question it
answers has a row.** Recording the arm is not enough. Arms are found by people who already know
what they are looking for.

## The four places, and what goes in each

| | file | discipline |
|:---|:---|:---|
| 1 | [`../runs.md`](../runs.md) | one row per **arm** — what it varied, seeds, budgets, directory, and a link to the writeup |
| 2 | [`../questions.md`](../questions.md) | one row per **question** — the arms that bear on it, and the verdict |
| 3 | `../log/<programme>.md` | the experiment: setup, numbers, prediction, outcome, caveats. Append-only |
| 4 | `../findings/engine/<topic>.md` or `../findings/training/<topic>.md` | what is true now, cited to 3. Rewritten in place |

**Which of the two findings directories.** Ask what would invalidate the claim. If a rules fix, a
mask change or a broken probe would delete it, it is **engine** knowledge and it is pinned to a
commit. If only a better-powered replication would move it, it is **training** knowledge and it is
a difference measured between two arms inside one engine. The cases where that test does not
decide are listed in [`../findings/README.md`](../findings/README.md), and the assumption the split
rests on is [`what_survives_an_engine_change.md`](what_survives_an_engine_change.md).

1 and 2 are indexes and cost a minute each. 3 is where the work goes. 4 exists only when a
question has an answer worth stating without its narrative.

## Every measurement lands in `log/` first

**A number is recorded when it is in `../log/`. Everywhere else links to it.** No other document
may hold the only copy of a measurement — not `findings/`, not `runs.md`, not `checkpoints.md`,
not a plan, and certainly not a commit message.

This is stricter than "write a log entry when the arm finishes", and it is stricter for a reason:
on 2026-09-16 three measurements were produced and written straight into `findings/` and
`checkpoints.md` — the per-player GAE tournament, the PFSP tournament, and the base-rate
sensitivity analysis behind both. Each was a *verdict* in the right place. None had the setup, the
sample size, the instrument or the caveats anywhere, because those belong to the experiment and
`findings/` is not where experiments live. A reader asking "how was that 520 Elo obtained, over how
many games, against which snapshots" had nowhere to go.

What "measurement" covers here is broad, and deliberately so:

* a tournament or any Elo number;
* a probe, sweep or diagnostic run, including ones that returned nothing;
* a statistic computed from `training_metrics.jsonl` — a windowed comparison, a correlation, a
  variance estimate;
* a sensitivity or robustness check on any of the above;
* a count taken from the corpus, the engine or the logs to settle a question.

The log entry carries: what was asked, how it was measured, **the instrument and the sample size**,
the result with an error bar, the verdict, and what would overturn it. `findings/` then states the
conclusion in a sentence or two and links back. If the conclusion later changes, `findings/` is
rewritten and the log entry is not — that asymmetry is the whole point of the split, and it only
works if the numbers are in the half that never moves.

**The practical test.** Open the `findings/` claim and follow its link. If you cannot reach the
sample size and the instrument in one hop, the measurement is not recorded yet.

**Where in `log/`.** One file per programme, as the index describes. A measurement that belongs to
an existing programme is appended to that programme's file; a new programme gets a new file and a
row in [`../log/README.md`](../log/README.md). A measurement large enough to be looked up on its own
— a full round-robin matrix, a corpus census — gets its own file beside the programme's, linked
from it.

## Before launching

**Register the prediction.** In the plan file or in the arm's `--description`, say what the probes
and the Elo should do. It costs a minute and it is the difference between a result and a
rationalisation.

**Name the endpoint, and say which instrument produces it.** Write down the statistic, the
sample size, and the file it will be read from. The pooling replication pre-registered *mean
|ussr_win_rate − 0.5| over the final 40M*, the arms landed, and the number was never computed —
what got reported instead was external side balance from a tournament, a different instrument
chosen after the fact. That is exactly the failure pre-registration exists to prevent, and
naming the instrument as well as the statistic is what would have caught it.

**If `engine/` changed since the control was trained, measure the decision stream before claiming
the pair is matched.** Build both engines, replay fixed seeds under a deterministic walk and under
the arms' own policies, and diff the decision type, the chosen action and the legal mask's hash at
every step. It takes hours; the alternative is re-running the control. Record the verdict in the
arm's row in [`../runs.md`](../runs.md) — the two worked examples are the `ROLL_DIE` pair and P14,
both under *E3-21*.

**Name the mechanism unambiguously.** Before reusing a word, grep for it. "Pool" already meant an
opponent pool, a start-state pool, a pooled head-to-head rating protocol, and global average
pooling inside the network. If the word is taken, the new thing needs a different one, and the
disambiguation belongs at the top of its findings file.

**Write the `--description` for a stranger.** `tools/train.py --description` is the only record
that survives if nothing else is written, and it is frequently the only record that exists. Say
which programme the arm belongs to, what its matched control is, and what it is judged on.
`E3-17-24`'s description is the model: it states the comparison, the confound it fixes, and why.

## While running

Add the row to [`../runs.md`](../runs.md) **when the run is launched**, not when it finishes. A
run that dies half-way still consumed GPU-hours and still has a directory, and an unlisted
directory is how `E3-18` and `E3-19` went missing from a table whose whole purpose is to identify
a directory unambiguously.

## When it finishes

1. Write the log entry: question, setup, result with sample size and an error bar, verdict,
   caveats. A negative or inconclusive result gets the same treatment as a positive one.
2. Fill in the `result written up in` cell of the arm's row. If there is nothing to link, write
   **no writeup** in the cell. An honest "no writeup" is a useful signal; an empty cell is not.
3. Add or update the row in [`../questions.md`](../questions.md). If the arm was run for a
   question that already has a row, the verdict is what changes — including to **open** if the
   result withdraws an earlier one.
4. If the answer is now stable enough to state without its narrative, put it in
   `../findings/`, and link back to the log rather than re-telling the experiment.

## Do not leave a result in `data/checkpoints/`

`data/` is git-ignored. A `report.md` written there by `tools/tournament.py` can be deleted by
anyone clearing disk space and no commit will record that it existed. Three of the five pooling
results were in that state when this file was written: `arena80_160/report.md`,
`arena_p12/report.md` and `arena_320/report.md`. **Copy the numbers that carry a claim into the
log entry**, name the report file, and mark it untracked so a later reader knows the tracked copy
is the only one.

The same applies to a tournament whose model labels are positional. `pool_comparison_tournament.md`
names its entrants `snapshot_final#1` through `#4` and nothing anywhere says which arm each was;
it is unusable. Stage checkpoints under readable symlink names before rating them, the way
`arena_p12/` does.

## The test to apply before calling it recorded

Ask the question out loud — "what did we learn about X?" — and try to answer it in **one lookup**
from [`../questions.md`](../questions.md). If the answer requires opening three files and
reconciling them, the row is not written yet.
