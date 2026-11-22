# P57 — why the belief is overconfident against people, tested (owner, 2026-11-22)

**Proposed, owner's request ("S-2 -- plan testing it").** [P35's site probe](../log/P35_belief_vs_people.md)
found the belief root's belief about a *person's* hand worse than the uniform prior (log loss −2.7% R34 soup,
−3.3% B6 soup; joint bits −0.47 / −0.53), while it gains +12.4% / +9.1% on each network's own self-play. Its log
names a cause it did not test: that the point-by-point decisions (every Influence point, every coup target)
carry most of the misleading evidence -- P35's earlier human-corpus number (+0.8%) conditioned on card and
play-mode choices only. The 2026-11-22 audit (S-2) added a second candidate: the temperature set is one set for
every decision type, at most T = 8, with no "this player is not my policy" component, so one well-matched kind
of decision (point placements, where people and the network agree often) pulls the weight onto low T for all.

## Arms -- one probe, paired positions

All on `ai/eval/belief_site_probe.py`, the same 307 clean site games of 2026-11-17 (or the latest export, re-run
for every arm), both networks (the R34 soup and B6's soup), 2,048 particles, box 4 on, and each network's own
self-play (256 games) as the control the change must not break:

| arm | what it changes | build |
|:---|:---|:---|
| **A** | today: every decision (`all_decisions=True`) | none |
| **B** | choices only: card and play-mode decisions (`all_decisions=False`, which `ts.BeliefTracker` already takes) | a `--choices-only` flag on the probe |
| **C** | every decision, plus a uniform component: the likelihood mixed with the uniform move with weight ε (0.05, 0.2), i.e. a T = ∞ model in the average | a mixture weight in `BeliefTracker` (one float, default 0 = today) |
| **D** | every decision, a temperature weight per decision type (choices / points / other), each averaged on its own evidence | the tracker keeps three evidence sums instead of one -- only if B or C shows the effect |

## Measures and the rule

Per arm and network, P35's measures with game-bootstrap standard errors: log-loss gain against the prior,
bits per position, hit rate and lift, joint bits, and the temperature weights; on people and on self-play.

* **The hypothesis is confirmed** if B's people log-loss gain is at least 2 standard errors above A's (paired),
  with joint bits ≥ 0.
* **A change is worth adopting in the root** if, on people, its log-loss gain is ≥ 0 and its lift is not below
  A's, and on self-play its gain falls by less than a third of A's (+12.4% / +9.1%). The root plays people on
  the site, so a belief no worse than the prior there beats one tuned to self-play.
* Then one leaderboard match decides deployment: the adopted belief root against today's on E9, 1,000 deals a
  side -- self-play strength must not drop by more than its error; the site's games are the reason, not Elo.

Also counted in every arm: decisions where every sampled hand gives the person's move zero probability (the
−1e9 path, audit S-1). The owner ruled them impossible in a game; on the site a non-zero count would mean the
record or the seating disagrees with the person's real move, which is a finding in itself.

## Cost

CPU only. A is P35's run (about 40 minutes a network on 8 threads); B is the same cost; C two ε values; D only
after B/C. No training, no GPU. A and B first, then C, reported before D is built.
