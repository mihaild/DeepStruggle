# One offline round of Gumbel-choice distillation on the heads soup (2026-10-10): the policy moves toward search's picks and gets slightly weaker

**Question.** Gumbel k=8 @256 is worth about +55-65 Elo on the heads soup
([`P31_branch_arms_B1_B2.md`](P31_branch_arms_B1_B2.md), [`E7_gumbel_headroom.md`](E7_gumbel_headroom.md)),
and every attempt to train that gain into the network has come out level or worse: visit-count
search targets as a fine-tune (E7-93/94-45), the target-forms probe (no form at 32-64 simulations
better than the prior), playout policy gradient, and R18 from the plateau (level in 310M, entropy
0.30 -> 0.5). The search-disagreement bank ([its note on the fork](https://github.com/JamesYouL2/DeepStruggle/blob/exp/search-bank/research/log/E7_search_disagreement_bank.md)) recommended the one form
not yet trained: distil Gumbel's own pick, only where it departs from the network. Does the network
absorb it, and is it then stronger?

**Answer.** It absorbs a little of it, and is not stronger. After one offline round over 90k
departures the distilled network plays Gumbel's choice at 14% of held-out departures (control 1%),
and loses to the base **49.1% ± 0.65** (−6 Elo), exactly as it loses to the no-signal control. The
@64 targets are worse (48.6%), the margin-gated ones in between (49.5%). The pre-registered bar --
+1.5 points over both the base and the control -- lies outside the 95% interval. **Distilling search's
choices, generically or filtered by the root's own margin, is not a way to bank search's gain on this
network.**

## Method

Run on the fork (JamesYouL2/DeepStruggle, branch `exp/gchoice-distill`): CI run `38007091505`
(workflow `gchoice_distill.yml`, commit `e616f7c`, engine `6c036593`), all on CPU runners. Tools: `tools/generate_search_targets.py --target gchoice`,
`tools/gchoice_targets.py` (arm, check), `tools/train.py --mode distill` (`tools/README.md` §7c).

* **Base.** `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt`, the SWA-made heads soup (sha256
  `91a43a8b089c…`).
* **Targets.** 4,000 games of the base against itself at temperature 0.2 (20 runners x 200, disjoint
  seeds). At a random quarter of decisions with two or more legal moves -- 482,091 positions -- a
  noise-free Gumbel root (the `gumbel:` tournament searcher, k = 8, first-play urgency 0.2) chose at
  256 and at 64 evaluations, recorded with every candidate's evaluations and value beside the
  network's argmax and distribution. Gumbel @256 departs from the argmax at 18.6% of them (89,793),
  @64 at 11.7% (56,273); in a 16-game sample @64 picked @256's departure only 29% of the time.
* **Arms**, all on the same positions, 2 epochs at lr 1e-4 (soft cross-entropy on the policy only,
  P15-X4a's trainer):
  * `departures@256`: a one-hot on @256's choice where it departs, the network's own distribution
    where it agrees (zero gradient at the start: the agreeing positions hold the policy, they do not
    sharpen it);
  * `departures@64`: the same with @64's choice;
  * `gated@256`: a one-hot only where the root's own margin Q(choice) − Q(argmax) is 0.04 or more in
    win probability (22,622 of the 89,793; the bank found margin a weak gate -- the top quarter of
    departures by margin held ~40% of their playout gain);
  * `own`: the network's own distribution everywhere -- no signal, the same optimiser steps.
* **Checks.** 100 held-out games on other seeds (12,537 searched positions, 2,361 @256
  departures): where search departs, how often is the arm's argmax now its choice; where it agrees,
  how often did the argmax change, and the KL from the base.
* **Strength.** The base and the four arms in one round robin, greedy (temperature 0, since the arms'
  entropies differ), 3,000 games a side per pairing (± 0.65 points per pairing).
* **Pre-registered rule** (before the data): departures@256 beats both the base and the control by
  1.5 points or more, with the checks showing movement at departures and little elsewhere ->
  propose online Gumbel-choice targets; level with the control -> stop distilling, spend on
  deploy-time search.

## Results

**Where the arms moved** (held-out, Gumbel @256's departures and agreements):

| arm | departures: argmax now the choice | p(choice) base → arm | agreements: argmax changed | KL(base ‖ arm) | entropy (base 0.316) | train top-1 with target |
|:---|---:|:---|---:|---:|---:|---:|
| departures@256 | **14.3%** | 0.153 → 0.208 | **3.8%** | 0.041 | **0.416** | 80.7% |
| departures@64 | 8.2% | 0.153 → 0.179 | 1.8% | 0.011 | 0.344 | 88.6% |
| gated@256 | 6.6% | 0.153 → 0.176 | 1.8% | 0.009 | 0.357 | 93.2% |
| own (control) | 1.4% | 0.153 → 0.153 | 0.4% | 0.001 | 0.316 | 99.3% |

Departures@64 against its own @64 targets: 13.4% now its choice, 0.201 → 0.238.

**Strength** (score with draws as half, row against column, 6,000 games each):

| arm | against the base | against the control |
|:---|---:|---:|
| departures@256 | **49.1% ± 0.65** (−6 Elo) | 49.1% (−6) |
| departures@64 | 48.6% (−10) | 48.6% (−10) |
| gated@256 | 49.5% (−4) | 49.8% (−2) |
| own (control) | 49.8% (−1.5) | -- |

Every arm is weaker as US than as USSR by about as much as the base (USSR − US +3.9 to +5.5 points);
no seat-specific effect. Report and checks: `data/reports/gchoice_distill/38007091505/`
(artifact `gchoice-distill-38007091505`).

## Reading

* **The rule's verdict is "stop".** departures@256 is level with the control at best, and its 95%
  interval (up to ~50.4%) excludes the +1.5-point bar. Search's +8 points a game are nowhere near.
* **The network cannot make search's departures its argmax.** On its own training positions it
  matches the target 80.7% of the time; with agreements 81.4% of the targets and 3.8% of them moved,
  that leaves about one departure in seven fitted -- the held-out rate (14.3%), so this is not
  overfitting but an inability to fit. What it does instead is spread probability onto them: entropy up 0.10, and 3.8% of the positions where search agreed with it changed their move.
  That is R18's broadening (0.30 -> 0.5) in miniature, by a different target and a different trainer.
* **Strength falls with the noise of the target, not with the amount of movement.** The @64 arm moved
  less than @256 and lost more; the gated arm moved least and lost least. A Gumbel root's departure
  at these budgets is mostly a near-tie resolved by its own noise (the two budgets agree on 29% of
  @256's departures), and a one-hot on a near-tie teaches "be less sure here", which costs a greedy
  player.
* **What this does not rule out.** A denoised target -- a departure only where independent searches
  agree, or one confirmed by paired playouts -- is a different and much smaller set, and this run's
  targets can be reused for it (`targets_from=38007091505`) without searching again. Nor does it say
  anything about value-target training. It does say that the generic forms, offline and online, are
  exhausted on this network.
* **So the gain of search is best banked by searching.** Gumbel k=8 @256 is +55-65 Elo at play time
  on this network; the census's one recurring blunder (the last influence placement of turn 10) wants
  an exact solver, not training.

## Addendum (2026-10-10): the consensus arm -- denoised targets do not help either

The one form "What this does not rule out" named: a departure only where the @64 root, searched on
its own streams, picks the same move as @256 (`consensus:256:64`; 24,986 of the 89,793 departures).
Same targets (`targets_from=38007091505`), same trainer, run `38015306195`, with a retrained
departures@256 and the no-signal control in the same field (3,000 games a side per pairing, greedy):

| arm | against the base | against the control | held out: argmax now the choice | agreements changed | entropy |
|:---|---:|---:|---:|---:|---:|
| consensus@256x64 | 49.5% ± 0.65 | **50.0%** | 5.5% | 1.0% | 0.327 |
| departures@256 (retrained) | 49.5% | 49.6% | -- | -- | -- |
| own (control) | 49.7% | -- | -- | -- | 0.316 |

Level with the control: the consensus subset moves the policy less and costs less, and gains
nothing. Agreement between two searches does not make a departure learnable or useful here --
two k=16 @1,024 roots reproduce each other's departure only ~47% of the time on the search bank,
so agreement between two noisier roots selects little signal. The distillation thread is closed;
the follow-up is the search itself ([`E7_search_depth_and_value.md`](E7_search_depth_and_value.md),
which also distils the stronger rollout root -- with the same result).

## Replicate

```bash
gh workflow run gchoice_distill.yml --ref main          # the defaults are this run
gh run download <run-id> -n gchoice-distill-<run-id> -D data/reports/gchoice_distill/<run-id>
# new arms on the same targets:
gh workflow run gchoice_distill.yml --ref main -f targets_from=<a targets run of this repository> -f arms='...'
```
