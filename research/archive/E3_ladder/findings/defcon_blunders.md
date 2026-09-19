# The DEFCON-1 blunder — what the critic cannot see, and what windowing does about it

The provoked DEFCON-1 loss is this project's clearest named failure: the agent plays an opponent
card for Operations, the event fires, DEFCON falls to 1 on its own turn and it loses. **Update
discipline: rewritten in place.** The traces, the replays and the arms are in
[`../../log/P9_architecture.md`](../../log/P9_architecture.md); the queued follow-up is
[`../../plans/P8_teach_the_defcon_conjunction.md`](../../plans/P8_teach_the_defcon_conjunction.md).

Filed under `training/` — what the critic represents and what a reward window costs — but resting
on an engine-side taxonomy: the provoked/self-inflicted split is read off the engine's own
`DEFCON_SUICIDE_PROVOKED` flag, and a sibling ending metric was wrong for a rules reason (a
turn-10 Wargames counted as final scoring,
[`../../log/measurement_bugs.md`](../../log/measurement_bugs.md)). If the DEFCON rules move, the
behavioural percentages move with them; the −68 Elo verdict on windowing does not.

## The critic never sees it coming, at any node

Traced through the `h2_480M_provoked_*` replays, replaying their recorded flat actions through a
fresh engine on the replay's own seed and reading `v_win` from the **losing** side's perspective.
Every value is read on the state *before* that node's action is applied, so the cost of a single
choice is the difference between the node where it is made and the node after it.

| game | v at the card node | delta from selecting the fatal card | delta from choosing OPS |
|:---|---:|---:|---:|
| 7107 Star Wars | −0.770 | +0.004 | +0.006 |
| 7115 Lone Gunman | −0.052 | −0.003 | −0.004 |
| 7118 Grain Sales | −0.691 | −0.008 | +0.007 |
| 7119 Grain Sales (unspaceable) | −0.711 | −0.007 | −0.014 |

**It never reacts.** Selecting the fatal card is worth at most 0.008 to the critic and committing
it to Operations at most 0.014; two of the eight deltas point the wrong way. At the last decision
before DEFCON 1 the critic still reports −0.40 to −0.75 rather than anything near −1, and in 7115
the losing side sits at −0.05 — essentially even — four micro-actions from death.

This is flat from 5M steps to 80M and still present at 480M, so it is not data scarcity. No
architecture change measured so far has moved it.

Two consequences:

* **It is not only a credit-assignment problem.** Sharpening the policy's credit will not by
  itself teach a conjunction the critic cannot represent — but that is the argument *for*
  windowing the provoked case, not against it. Inside a blunder window the advantage is
  `-1 - v_t`, which never consults the critic, so the window is precisely the mechanism that
  works when the critic is blind. Outside one, the −1 has to flow back through a value function
  that prices the position at −0.7 and rising, and is absorbed rather than attributed.
* **It raises the value of an auxiliary DEFCON-risk head** — a direct supervised signal for what
  `v_win` demonstrably does not encode. Its label is the same `defcon_blunder` flag that excludes
  provoked endings, so the one-line label fix is a prerequisite for either.

## Windowing moves exactly the class it targets, and costs 68 Elo

`--window-provoked-defcon` (arm E3-08) credits a provoked DEFCON-1 to the player who played the
card, using the existing turn-scoped blunder window, instead of letting the −1 propagate back as
an ordinary loss. Two seeds, 80M steps, against `p1_scalar_nofilter` at 80M — same recipe,
filtering off in both, the window the only difference.

| ending | control | window (seed 21) | delta |
|:---|---:|---:|---:|
| DEFCON-1 total | 49.4% | 27.0% | −22.4 |
| — provoked | 36.2% | 13.7% | **−22.5** |
| — self-inflicted | 13.1% | 13.3% | **+0.1** |
| final scoring | 6.9% | 16.7% | +9.8 |
| 20 VP | 39.6% | 53.9% | +14.3 |

**The self-inflicted share does not move.** The intervention targets provoked endings alone and
provoked endings alone changed, by 62% of their own value, while the neighbouring class in the
same metric family stayed put to a tenth of a point. That rules out the reading that the arm
simply made every DEFCON-1 rarer by playing timidly, and it replicates on the second seed
(provoked 10.7%, self-inflicted 12.1%).

Games also got longer — mean ply 93.4 to 103.1 — moving toward the human distribution, where
about 30% of games reach final scoring against this control's 7%.

**And it loses.** Pooled over sixteen snapshot pairings per seed against the control: seed 21
41.6% [39.9, 43.3] = **−59 Elo**, seed 22 39.1% [37.4, 40.8] = **−77 Elo**, pooled 40.3%
[39.1, 41.5] = **−68 Elo**. Two-seed confirmed.

The anchor metric disagreed with all of this — the two seeds sit 10.4 anchor points apart and
straddle the control — which is one of the cases that retired `HeuristicBot` as a rating
instrument ([`../../method/running_experiments.md`](../../method/running_experiments.md)).

## Where this leaves the question

The behavioural target is reachable and the current mechanism is too blunt to be worth its cost:
the turn-scoped window is **5.6× wider than the mistake it credits**, so it penalises every other
decision in the turn as well. `--window-provoked-defcon` is **not adopted**. The queued step is a
narrower window plus the auxiliary risk head, which is
[`../../plans/P8_teach_the_defcon_conjunction.md`](../../plans/P8_teach_the_defcon_conjunction.md).
