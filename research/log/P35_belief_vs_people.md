# The belief root against people: how well it guesses a person's hand (2026-11-17)

**Conclusion.** Scored exactly as it runs (`ts.BeliefTracker`: 2,048 hands, every opposing decision of the turn,
the temperature model-averaged, box 4), the belief about a *person's* hand on the site's games is **worse than
the uniform prior** in log loss (R34 soup −2.7%, B6 soup −3.3%, ±0.3) and in the true hand's probability
(−0.47 / −0.53 bits), in every seat and stage of the war -- while on each network's own self-play it gains +12.4%
/ +9.1% (+2.1 / +1.5 bits). It still *ranks* the person's cards a little better than random: a dealt world holds
24.6-24.9% of the person's real cards against the prior's 22.3% (lift +2.9 / +3.3% of the way to the oracle;
self-play +9.8 / +7.1%). So against people the belief points the right way but is overconfident, and the
averaged temperature does not flatten it enough. B6 reads people as playing like itself (its temperature weight
sits on T 1-1.5, R34's on T 2), and pays for it: paired on the same positions it ranks slightly better than R34
(+0.38 lift, s.e. 0.08) and is slightly worse calibrated (−0.6% log loss, s.e. 0.24).

Worse than P35's human-corpus number (+0.8% for the averaged belief), which conditioned on card and play-mode
choices only; here every decision is conditioned on, as the root does. Untested: that the point-by-point
decisions carry most of the misleading evidence (a choices-only run would say).

## The question

The owner: how well does the belief root, for `E8.1-A8-R132-S45@11160M+(S45,46,47)@11520..11600M` (B6's soup)
and `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M` (the R34 soup), predict a person's hand, compared
with how well it guesses in self-play? First a design and a measure.

## Setup

`ai/eval/belief_site_probe.py`. Games: the site's export of 2026-11-17 (`data/sql_games6.txt`, 420 games),
each re-driven on this build (engine `82878d47`): the person seated with the person's rule bits, the model with
its own (the page's seating), else unseated (an older page); kept when it ends on the submitted position. 307
clean games kept (5 not one person against one model from the deal, 56 not clean, 52 not replaying here: older
engines).

At every decision of the AI's side where the person holds unseen cards, the tracker -- fed every decision of
the game, a person's move recorded as the flat action that decodes to it, as the page does -- gives its belief;
55,156 positions. **Either network models the person on every game** (the belief needs only the record and the
network's likelihoods; B6's soup played only 4 of them), so the two see the same positions and, the draws being
keyed by seed and position, the same hands: their difference is paired. The reference is each network's own
self-play (256 games, the policy at T = 1, both seats in its feature set), scored the same way at every
decision of either side (~100,000 positions); there the likelihood is exact.

Measures, per position, over the decider's hidden pool (the opponent's unseen hand and the deck):

* **log loss** per pool card against the prior hand / pool (P35's number), and the same as bits per position;
* **hit rate**: the expected share of a dealt world's hand that the opponent really holds; prior hand / pool,
  oracle 1. **Lift** = (hit − prior) / (1 − prior), the way from random worlds to the oracle root's;
* **joint bits**: log2 p_belief(true hand) / p_prior(true hand) = log2 Σ_T w_T L_T(h*) / Z_T. With an exact
  model its expectation is the information the record carries about the hand (≥ 0); below 0 the belief
  misleads. Z_T is the particles' mean likelihood, biased low when few hands carry the weight (reads high).

Checks: the marginals rebuilt from the likelihoods equal the tracker's dealt worlds within resampling noise
(max 0.007 over 367 beliefs, 50,000 draws each); the true hand's likelihood equals the policy's own
log-probability of each recorded move (max 3e-5 over 5,571). Joint bits are missing at ~1% of beliefs (540 / 592)
where the probe's mirror of the record differs from the tracker's (box 4's revealed headline). Standard errors
by bootstrap over games.

## Results

| belief's model | opponent | positions | log-loss gain | bits / position | hit (prior → belief) | lift | joint bits | T weights 1 / 1.5 / 2 / 3 / 5 / 8 |
|:---|:---|---:|---:|---:|:---|---:|---:|:---|
| R34 soup | people | 55,156 | **−2.7%** ±0.3 | −0.43 | 22.3 → 24.6% | +2.9% | **−0.47** ±0.05 | .01 / .16 / .63 / .17 / .02 / .01 |
| R34 soup | its self-play | 101,336 | +12.4% ±0.1 | +1.97 | 22.0 → 29.7% | +9.8% | +2.08 | .89 / .09 / .01 / 0 / 0 / 0 |
| B6 soup | people | 55,156 | **−3.3%** ±0.3 | −0.52 | 22.3 → 24.9% | +3.3% | **−0.53** ±0.05 | .42 / .51 / .06 / .01 / 0 / 0 |
| B6 soup | its self-play | 97,677 | +9.1% ±0.1 | +1.46 | 21.9 → 27.4% | +7.1% | +1.52 | .91 / .07 / .01 / 0 / 0 / 0 |

Self-play reweighted to the site's mix of (side, stage): R34 +12.3% / lift +9.8%, B6 +9.0% / +7.1% -- the
difference is not the positions' mix. 94% of positions in both sources condition on at least one opposing
decision (mean ~15); median effective hands 659 (people) / 397 (self-play) for R34, 589 / 603 for B6.

By the person's side and stage (log-loss gain, lift):

| | US early | US mid | US late | USSR early | USSR mid | USSR late |
|:---|---:|---:|---:|---:|---:|---:|
| R34 soup, people | −5.1%, +3.5 | −2.1%, +3.0 | −4.6%, +2.2 | −0.9%, +2.3 | −1.2%, +3.1 | −2.3%, +3.0 |
| B6 soup, people | −3.3%, +3.8 | −2.6%, +3.0 | −7.9%, +2.4 | −1.1%, +3.5 | −2.8%, +3.5 | −4.2%, +3.5 |
| R34 soup, self-play | +12.9%, +11.1 | +11.2%, +8.7 | +9.0%, +6.9 | +12.5%, +10.4 | +14.0%, +10.6 | +14.4%, +10.8 |
| B6 soup, self-play | +9.2%, +7.7 | +8.1%, +6.2 | +8.2%, +6.1 | +8.9%, +7.4 | +10.2%, +7.5 | +10.5%, +7.9 |

Paired, B6 minus R34 on the site positions: log-loss gain −0.59 (s.e. 0.24), lift +0.38 (0.08), joint bits
−0.06 (0.04).

## Caveats

* What the root does with a world is not a log loss: a world dealt with 2.5 more points of the person's real
  cards may help the search even while the belief's probabilities are overconfident. Whether the belief root
  wins or loses against people is a playing question this probe does not answer.
* The site's people are a few dozen players, mostly against the R34 soup, many games ending early.
* Self-play is the policy at T = 1, not the searched player; against a searched opponent the network's
  likelihood is not exact either (P35: the same edge against the R34 soup's root).

Data: `data/reports/belief_site_probe_{r34soup,b6soup}.npz` (per-position rows; `--summarize` prints the tables).
