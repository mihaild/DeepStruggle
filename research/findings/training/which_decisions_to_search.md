# Searching only card and play-mode decisions was the wrong scoping

**Written 2026-09-17**, after P15-X4a distilled a searcher into the policy at card/play-mode nodes
only and X4b ran its first arm under the same restriction.

## The claim that was inherited

P3 scoped determinized search to `SELECT_CARD` and `SELECT_PLAY_MODE` on the argument that those
are *"the decisions that matter"*, and P15's X4a and X4b specifications carried that scoping
forward without re-examining it. Every search number this project produced under that filter
inherits it.

## Why it does not hold

**Where to put influence is a first-class strategic decision, not a detail of executing a card.**
Choosing to play a card for operations rather than its event is one decision; the placement that
follows decides which country flips, whether a battleground is contested, whether a region's
domination threshold moves, and what the opponent's coup or realignment options are next turn. A
plan expressed as "play this card for ops" is not yet a plan — the ops have to land somewhere, and
where they land is most of the content.

Treating that as beneath the searcher's attention takes the decision with the widest branching in
the game and hands it to the raw policy, unexamined.

## What the measurement says

[`log/P15_X4a_where_the_search_signal_is.md`](../../log/P15_X4a_where_the_search_signal_is.md)
censused **all 85,113 decisions of 200 games** with the searcher answering every one:

| type | n | top-1 agree | KL(search ‖ policy) | mean legal |
|:---|---:|---:|---:|---:|
| **POINT_NODE** | 33,420 | 92.9% | **0.1239** | **26.4** |
| SELECT_CARD | 21,058 | 90.5% | 0.0431 | 5.6 |
| SELECT_PLAY_MODE | 15,362 | 96.6% | 0.0192 | 2.1 |
| SELECT_OP_MODE | 10,682 | 94.8% | 0.0319 | 2.9 |
| CHOOSE_TIMING_BRANCH | 4,179 | 95.7% | 0.0175 | 2.0 |
| CHOOSE_BRANCH | 412 | 97.8% | 0.0189 | 2.6 |

Weighting each type's KL by how often it occurs:

| | share of total CE signal |
|:---|---:|
| **POINT_NODE** | **71.8%** |
| SELECT_CARD + SELECT_PLAY_MODE — what P3's filter searches | 20.9% |
| everything else | 7.3% |

`SELECT_PLAY_MODE` averages **2.1 legal actions**. A node with two options cannot carry an edge;
searching it 96 times answers a question the mask has already answered. Meanwhile placements
average 26.4 legal actions and reach 82.

**So the filter excluded roughly three quarters of the available signal, and spent its budget on
the decisions with least to choose between.**

## The measured consequence, so far: smaller than the signal share suggests

Offline distillation from the same source checkpoint, same recipe, same 2 epochs at 1e-4, rated at
temperature 0 in one field (`/workspace/data/tournaments/P15_X4a_allnodes/`):

| model | Elo | vs source |
|:---|---:|---:|
| distilled, **all nodes** | 1565.6 | +35.6 |
| distilled, card/play-mode | 1555.1 | +25.1 |
| `source_200M` | 1530.0 | — |

All-nodes is ahead by **10.5 Elo**, head to head **51.6%** over 1,000 games — about one standard
error, so **not a significant difference**.

That is worth stating plainly rather than dressing up: the scoping was wrong on the merits and
wrong on the signal census, but **fixing it did not, in this one-shot offline test, produce the
large gain the 71.8% figure might lead one to expect.** Elo is not linear in nats, and this is the
evidence that the two do not track.

Two caveats on that near-tie, one of which points the wrong way:

* The all-nodes set came from **200 games** against the card/play-mode set's **400** — half the
  state diversity, a confound working *against* the arm under test. A games-matched 400-game
  replication is running.
* One seed, two epochs, one learning rate, one source checkpoint.

## What this changes

* **X4b runs with `--search-node-filter all`**, decided on the census before the arm was spent.
  That arm beat its step-matched control by +165.6 Elo at 20M and is continuing to 80M.
* Any search result recorded under `card_playmode` should be read as a **lower bound** on what a
  searcher can do here, not as that searcher's strength.
* The open question is no longer *whether* placements matter — the census settles that — but why
  the offline one-shot gain from including them is so much smaller than their signal share. The
  most likely answer is that a single round of CE cannot exploit a 26-way distribution the way it
  exploits a 5-way one, and that the continuous form is where placements pay. X4b at 80M is the
  test.

## See also

* [`log/P15_X4a_where_the_search_signal_is.md`](../../log/P15_X4a_where_the_search_signal_is.md) — the census
* [`log/P15_X4a_distillation.md`](../../log/P15_X4a_distillation.md) — the original card/play-mode round
* [`log/P15_X4b_search_during_rl.md`](../../log/P15_X4b_search_during_rl.md) — the continuous form
