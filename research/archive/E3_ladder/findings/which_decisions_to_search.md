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

## The measured consequence: in one-shot offline distillation, almost none

Offline distillation from the same source checkpoint, same recipe, 2 epochs at 1e-4, rated at
temperature 0 in one field (`/workspace/data/tournaments/P15_X4a_allnodes/`, 500 games a side):

| model | targets | games | Elo | vs source |
|:---|---:|---:|---:|---:|
| distilled, **all nodes** | 176,122 | 400 | **1572.5** | +42.9 |
| distilled, all nodes | 85,113 | 200 | 1566.6 | +37.0 |
| distilled, card/play-mode | 75,592 | 400 | 1560.3 | +30.7 |
| `source_200M` | — | — | 1529.6 | — |
| `anchor_280M` | — | — | 1500.0 | |

The first and third rows are **games-matched**: 400 self-play games each, differing only in which
decisions the searcher answered. All-nodes gets 2.3× the targets from the same games, which is
intrinsic to the treatment rather than a confound.

**And they are a dead heat.** Head to head, all-nodes-400 against card/play-mode is **49.4%** —
it loses marginally, over 1,000 games, while sitting 12.2 Elo higher in the table. Every distilled
variant lands between +31 and +43 over the source.

So the scoping was wrong on the merits and wrong on the signal census, and **fixing it bought
nothing measurable in the one-shot offline setting.** That is the honest result, and it is not the
one the 71.8% figure predicts.

### One explanation ruled out: the searcher is not dodging operations

The obvious suspect is that the searcher places influence badly, therefore underrates the whole
ops branch, therefore avoids it — which would make its play-mode targets biased and worthless to
distil. Measured and **refuted**, in the direction that matters:
[`log/P15_search_play_mode_bias.md`](../../log/P15_search_play_mode_bias.md). Over 18,097
play-mode decisions with both options legal, search shifts **toward** ops (+2.16 pp) and away from
EVENT (−1.08 pp) and SPACE (−1.08 pp) — it takes both of the available escape routes from a
placement decision *less* often than the raw policy does.

### Why the signal share does not convert into Elo here

The census measures the *gradient available*; Elo measures what one pass of cross-entropy at
1e-4 actually installs. The likely explanation is a ceiling on the method rather than on the
signal: two epochs of soft CE can shift a 5-way distribution toward its teacher, but a 26-way
placement distribution needs far more capacity moved for the argmax to change, and top-1 agreement
at `POINT_NODE` was already 92.9%. The disagreement is spread thinly across many low-probability
placements, which is exactly the shape CE moves slowly.

This is a concrete instance of the standing caution that **Elo is not linear in nats**, and it is
worth remembering the next time a diagnostic is used to predict a result rather than to explain
one. The census correctly located the signal; it did not predict the payoff.

## What this changes

* **X4b runs with `--search-node-filter all`**, decided on the census before the arm was spent.
  That arm beat its step-matched control by +165.6 Elo at 20M and is continuing to 80M. **That is
  not evidence that `all` beats `card_playmode` during RL** — the control had search off
  altogether, and no `card_playmode` X4b arm has been run. The offline result above says the two
  filters tie in one-shot distillation; whether they tie continuously is untested.
* Any search result recorded under `card_playmode` should be read as a **lower bound** on what a
  searcher can do here, not as that searcher's strength.
* The open question is no longer *whether* placements carry the signal — the census settles that —
  but whether any method converts that signal into strength. One-shot CE does not. The remaining
  candidate is the continuous form, where the teacher improves with the student and a placement
  distribution gets revisited thousands of times rather than twice. **The experiment that would
  settle it is an X4b arm at `card_playmode`, matched against the `all` arm** — without it, X4b's
  +165.6 Elo is evidence for search-during-RL, not for this scoping fix.

## See also

* [`log/P15_X4a_where_the_search_signal_is.md`](../../log/P15_X4a_where_the_search_signal_is.md) — the census
* [`log/P15_X4a_distillation.md`](../../log/P15_X4a_distillation.md) — the original card/play-mode round
* [`log/P15_X4b_search_during_rl.md`](../../log/P15_X4b_search_during_rl.md) — the continuous form
