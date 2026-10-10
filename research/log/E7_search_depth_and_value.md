# Where the E7 network's limits are: search gains from depth, the critic cannot choose, and rollout labels sharpen it (2026-10-10)

**Question.** Where does the E7 network still fall short, and where should it go next? Three
measurements on one shared footing -- the search-disagreement bank's 2,300 raw-play positions of
`E7line_swa_4720-4800M` with their population weights ([the bank's note on the fork](https://github.com/JamesYouL2/DeepStruggle/blob/exp/search-bank/research/log/E7_search_disagreement_bank.md); its
positions are fork release `search-bank-playouts-20261007`) -- then games. All runs were made on the
fork's CI (JamesYouL2/DeepStruggle); the run ids below are its.

1. **Chance or depth?** Gumbel@256's departures from the network reproduce across seeds only 49% of
   the time. Is that noise from sampling the hidden cards and the dice (one draw per halving phase;
   a move's dice rolled once when the tree expands it), and does averaging over more draws fix it?
2. **Is the critic the bottleneck, and is it fixable?** Can a value head on the network's own frozen
   trunk, trained on rollout labels, rank sibling moves better than the critic?
3. **Can rollouts stand in for depth?** An Ataraxos-style root -- the network's top moves played out
   by the network in many sampled worlds to a horizon, then the critic, then a cautious choice.

**Answer, in one paragraph.** Search's gain comes from **depth**, not from averaging over chance:
spreading Gumbel's budget over more worlds makes its picks reproducible (49% -> 96%) and *worse*
(the 64-world root, which reads the critic one move ahead, gains nothing), while four times the
budget in one world gains most. The **critic cannot choose between sibling moves on its own**: a
root that plays whichever child the critic prefers loses 170 points a game against the network's
own move. But the trunk knows more than the critic uses: a head trained on rollout labels **doubles
the correlation** of predicted with true sibling differences on held-out positions (0.25 -> 0.59)
and lifts clear-pair accuracy 71% -> 76% -- a training-signal limit, not a representation one -- yet
helps nowhere on the positions search disputes. **Rollouts through 2-4 action rounds** with a
cautious rule are the best searcher offline -- and in games by a wide margin: **+84 Elo over
Gumbel@256** (61.8% +- 1.4) and +177 over the network, while Gumbel itself gets *worse* with four
times the budget (42.0% against Gumbel@256, -56 Elo).

## 1. Worlds against depth (`gumbel_worlds`)

`BatchedMCTSConfig.gumbel_worlds = m` splits each candidate's share of a halving phase over min(m,
share) independent draws of the hidden cards and the dice, the candidates of a world sharing its
draw (`ai/search/gumbel_root.py`; picks at m = 1 identical to the previous code on 64 positions x 2
seeds). `tools/search_reliability.py`: every bank position searched four times per searcher on
independent streams, every pick played out in 512 paired raw-network continuations (CI
`38016734450`).

| Gumbel k=8 | evals | departs | departure reproduced | gain per game | vs w1 @256 |
|:---|---:|---:|---:|---:|---:|
| 1 world per phase @256 (today) | 252 | 22.3% | 49.3% | +28.8 +- 13.5 | -- |
| 4 worlds @256 | 252 | 18.3% | 61.5% | +31.8 | +3.0 +- 10.4 |
| 16 worlds @256 | 252 | 19.3% | 88.5% | +15.0 | -13.8 +- 14.0 |
| 64 worlds @256 | 252 | 21.9% | 96.5% | +3.2 | -25.5 +- 15.5 |
| 1 world @1,024 | 1,019 | 33.3% | 39.4% | **+57.9** | **+29.1 +- 14.8** |
| 16 worlds @1,024 | 1,019 | 27.8% | 78.3% | +40.1 | +11.3 +- 16.6 |

Gain is the pick's playout score minus the network move's, population-weighted, summed over a game's
decisions (this sum overstates tournament points 3-4x, [the bank's note](https://github.com/JamesYouL2/DeepStruggle/blob/exp/search-bank/research/log/E7_search_disagreement_bank.md); read it for
ranking). Spreading over worlds shortens each sub-search; at 64 worlds a candidate gets
about one evaluation per world -- the critic one move ahead -- and that root is stable and empty.
Four times the budget in one world departs more, reproduces less, and gains most.

## 2. The critic, and what rollout labels add (`tools/value_probe.py`)

**Labels** (CI `38017894445`): 960 fresh self-play games of the model (seeds disjoint from the bank),
a 2.5% sample of decisions, the network's top 3 moves played out in 256 paired continuations:
12,021 positions. Sibling differences average 3.9 points against a median standard error of 3.1;
where the labels separate two moves (14% of pairs) the network's own move is the better one 80% of
the time.

**Fit.** Children of each labelled move in 4 sampled worlds, observed as the player to act there
(what a search leaf reads), their trunk features h (480 floats) frozen. Heads trained on absolute
values plus a 10x-weighted paired-difference loss; 80% of positions to train, 20% held out.

| value of the position after each move | held-out: picks the clearly better of two (640 pairs) | corr with labelled difference (all pairs) | disattenuated | on the bank |
|:---|---:|---:|---:|---:|
| critic, untouched | 71.4% | +0.25 | +0.27 | 53.3% |
| critic head fine-tuned on rollout labels | **76.2%** | +0.42 | +0.45 | 50.7% |
| fresh MLP on frozen h | 72.2% | **+0.59** | **+0.64** | 52.4% |
| fresh MLP, paired loss only | 71.9% | +0.57 | +0.62 | 50.6% |

Learning curve (fresh MLP): accuracy flat at 71-72% from 2,400 to 9,600 positions, correlation rising
(0.40 -> 0.50 -> 0.59).

* **The bank is adversarial for any critic-like judge.** Its positions are where critic-driven search
  disagreed with the network, so its pairs are moves the critic liked; every head sits near 50% there.
  The bank's stored "bare critic" (a root plus one simulation) reads 61% on those pairs and the
  critic of the child alone 53%, against 71% on fresh positions: read the bank's sibling accuracy
  as a floor, not a rate.
* **A training-signal limit on typical positions.** The trunk carries sibling information the critic
  head does not read out: a small head on frozen h, trained on 10k rollout-labelled positions,
  more than doubles the correlation. Rollout targets are a usable value-training signal.
* **No help on the hard positions.** On the bank's disputed positions nothing improves: what decides
  those is not in a one-move-ahead reading of these features.

## 3. A rollout root (`ai/search/rollout_root.py`)

The network's top 4 moves; in each of W worlds (fresh hidden cards and dice, shared by the
candidates) each is played and the network plays both sides greedily through H action-round
boundaries (a rollout move that loses on the spot is replaced by the player's next-best); the critic
reads the leaf. Rules: `argmax`; `z2` -- the best mean value only if its paired lead over the network's
move is 2 standard errors; `kl<t>` -- argmax of log pi + Q/t. Same harness and games as section 1
(CI `38018140895`).

| rollout root (k = 4) | network rows | departs | reproduced | gain per game | vs Gumbel@256 |
|:---|---:|---:|---:|---:|---:|
| H0, 64 worlds, argmax (the critic's favourite child) | 238 | 42.9% | 96.7% | **-170.5 +- 58.3** | -199 +- 57 |
| H1, 32 worlds, z2 | 445 | 30.8% | 90.0% | +20.3 | -8.5 +- 33.9 |
| H2, 32 worlds, z2 | 904 | 26.5% | 79.0% | +66.5 | +37.7 +- 24.5 |
| H2, 32 worlds, argmax | 904 | 41.2% | 79.4% | +39.7 | +10.9 +- 31.8 |
| H2, 32 worlds, kl0.05 | 904 | 3.6% | 81.6% | +24.1 +- 6.1 | -4.6 +- 12.6 |
| H4, 16 worlds, z2 | 910 | 17.6% | 62.1% | **+75.0** | **+46.2 +- 21.2** |

* **The critic as a chooser is a disaster**: one move ahead, averaged over 64 worlds, it departs from
  the network at 43% of decisions and loses 0.85 points per departure.
* **Depth by rollout works offline**: 2-4 action rounds and the cautious z rule match or beat Gumbel
  at 1,024 evaluations for the same network rows (CPU time about twice: rollouts step the engine).
  The KL rule at t = 0.05 departs rarely and is right 75% of the time when it does -- too timid.
* **Caveat:** the judge is raw-network continuations, which the rollout root's own rollouts share,
  so this flatters it. Games decide.

## 4. Games

**The rollout-trained critic in search** (CI `38025151765`, `searcher_tournament.yml`; the critic head
re-tuned on CI from the same labels -- `tools/value_probe.py tune`, all 12,021 positions, final loss
reproducing the local run -- and swapped into the checkpoint, the policy untouched; 1,000 games a
side per pairing, greedy):

| pairing | score | Elo |
|:---|---:|---:|
| Gumbel@256 against the network | **62.1% +- 1.1** | +86 |
| Gumbel@256 over the rollout-trained critic, against the network | 58.4% +- 1.1 | +59 |
| rollout-trained critic against the original, both Gumbel@256 | 49.3% +- 1.1 | -5 |

**The better-ranking critic does not make search stronger** -- level head to head, and it keeps less
of search's gain against the network (2.4 SE). What it learned, ranking a decision's children on
typical positions, is what search already gets by looking deeper; and the head now reads every leaf
of a tree after being fitted only on positions one move from a decision.

**The rollout root against Gumbel** (CI `38025925220`; the network, Gumbel k=8 @256 and @1,024, and
the rollout root k=4, 16 worlds, 4 boundaries, z2; 600 games a side per pairing, greedy):

| pairing | score | Elo |
|:---|---:|---:|
| **rollout root against the network** | **73.4% +- 1.3** | **+177** |
| **rollout root against Gumbel@256** | **61.8% +- 1.4** | **+84** |
| rollout root against Gumbel@1,024 | 69.1% +- 1.3 | +140 |
| Gumbel@256 against the network | 62.5% +- 1.4 | +89 |
| Gumbel@1,024 against the network | 52.8% +- 1.4 | +19 |
| Gumbel@1,024 against Gumbel@256 | **42.0% +- 1.4** | **-56** |

Field Elo (net 1,429): Gumbel@1,024 1,456, Gumbel@256 1,515, rollout root 1,600. The rollout root wins
in both seats against every opponent (against the network 69% as US, 78% as USSR). Cost: ~910
network rows per decision (3.6x Gumbel@256) and ~8x its CPU time in this Python implementation.

* **Gumbel's budget curve turns down.** 256 was worth +12 +- 3 over 64 (`E7_gumbel_headroom.md`);
  1,024 is worth -56 against 256 in games -- although the offline playout judge ranked it above 256
  (+29 +- 15 a game). Two mechanisms fit, neither yet isolated: a determinized tree optimises both
  sides' replies against the hidden cards and the single dice roll each edge was expanded with, so a
  deeper tree trusts more lucky draws (the classic PIMC overfit, and section 1's evidence that a
  move's roll is never re-drawn); and a searcher that re-plans each micro-decision in new worlds
  (39% reproducible at 1,024) plays incoherent sequences -- the points of one placement from
  different plans -- which a per-decision judge continuing the network's own plan cannot see.
* **The rollout root avoids both by construction.** In its rollouts each side is the network acting on
  its own observation -- honest play, nothing maximised over a sampled draw -- the dice are redrawn in
  every world, and the z rule leaves the network's move in place unless another is clearly better,
  so consecutive decisions stay on the network's plan unless the evidence says otherwise.
* **The offline judge is not enough on its own.** It ranked the rollout root first, correctly, but
  Gumbel@1,024 above Gumbel@256, wrongly. Search variants are ranked by games.

**On the frontier network** -- the heads soup `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt`, used as
published (CI `38042458244`; the provenance's "rebuilt with" line is the workflow's template echo, the
.pt was taken as it is); the rollout root at 4 and at 2 boundaries, 16 worlds; 500 games a side:

| pairing | score | Elo |
|:---|---:|---:|
| Gumbel@256 against the soup | 58.2% +- 1.6 | +57 |
| **rollout root, 4 boundaries, against the soup** | **70.2% +- 1.5** | **+148** |
| rollout root, 2 boundaries, against the soup | 64.1% +- 1.5 | +101 |
| **rollout root, 4 boundaries, against Gumbel@256** | **63.7% +- 1.5** | **+97** |
| rollout root, 2 boundaries, against Gumbel@256 | 57.6% +- 1.6 | +53 |
| 4 boundaries against 2 | 55.1% +- 1.6 | +35 |

Field Elo (soup 1,423): Gumbel@256 1,477, rollout H2 1,530, rollout H4 1,570. It holds on the best
network, by more than on the bank's model, and in both seats (against Gumbel@256 the H4 root scores
66.6% as US and 61.0% as USSR). The longer horizon is better; the leaderboard's Gumbel k8 @256 on this
soup (57.9% against its network) is reproduced here (58.2%).

## 5. Distilling the rollout root (2026-10-10): the same failure as Gumbel's

Would training on the rollout root fail too? One offline round, the gchoice
pipeline with the rollout root as the teacher (`generate_search_targets.py --target rollout`, k = 4, 32
worlds, 4 boundaries, z2) on the heads soup's own self-play (CI `38061318714`: 20 x 50 games, a quarter
of decisions, temperature 0.2; 121,262 positions, the teacher departing at 19,990 = 16.5%). Arms on the
same positions, 2 epochs at lr 1e-4: a one-hot on the teacher's pick where it departs; a **soft** step,
pi(a) exp((Q(a) - Q(argmax)) / tau), at tau 0.05 and 0.1 (Q the mover's rollout value in [-1, 1]); the
network's own policy (control). Pre-registered: an arm beats the base and the control by 1.5 points,
with the teacher's departures learned well above Gumbel's 1 in 7 and no entropy blow-up.

| arm | held out: teacher's departures now played (961) | agreements changed (4,832) | entropy (base 0.306) | vs base | vs control |
|:---|---:|---:|---:|---:|---:|
| departures (one-hot) | **10.0%** | **10.7%** | **0.458** | **48.1% +- 0.65** | 48.7% |
| soft, tau 0.05 | 0.8% | 0.8% | 0.308 | 50.2% | 50.4% |
| soft, tau 0.1 | 0.5% | 0.5% | 0.306 | 50.3% | 50.2% |
| own (control) | 0.2% | 0.6% | 0.306 | 49.4% | -- |

* **Fails by the rule.** The one-hot arm learns 1 in 10 of the teacher's departures (Gumbel's: 1 in 7),
  about 14% even on its own training positions, disturbs three times as many agreements as Gumbel's
  did (10.7% against 3.8%), broadens more (+0.15 against +0.10) and loses 13 Elo. The soft arms barely
  move (p(choice) 0.069 -> 0.072) and are level.
* **A stronger, cleaner teacher does not help, because the teacher was never the bottleneck.** The
  rollout root's departures go to moves the network gives 0.07 (Gumbel's: 0.15) -- further from what
  it expresses -- and they are the products of four action rounds of consequences that a forward
  pass does not compute. Every teacher tried (visit counts, Gumbel's pick, gated, consensus, and now
  rollouts) meets the same wall: the network cannot represent search's corrections at the positions
  where search makes them, and pushing it there costs more than it gains.

## Reading: where the limits are, and where to go

**Where the limits are.**

1. **The critic cannot choose moves; depth must.** One move ahead it loses 170 points a game against
   the network's own choices, and averaging it over chance does not help. Every searcher that works
   puts the critic several decisions downstream.
2. **The critic's training signal is weak where moves are close, but fixing it alone does not help
   search.** Rollout labels double its sibling correlation on typical positions; in search that buys
   nothing (49.3% head to head). The positions search disputes are not resolved by a better
   one-move reading.
3. **Determinized tree search is the limit of today's search.** Gumbel/PUCT in sampled worlds gains
   +85-90 Elo at 256 evaluations and loses ground beyond it. Honest rollouts over many worlds with a
   cautious rule gain twice as much.
4. **Search does not distil** ([`E7_gchoice_distill.md`](E7_gchoice_distill.md) and section 5): the policy moves
   toward search's picks without fitting them and broadens; Gumbel's picks (generic, gated,
   consensus) and the rollout root's (one-hot or a soft KL step) are all level or worse.

**Where to go.**

* **Deploy the rollout root as the strongest player** -- the leaderboard's searcher and the
  workbench's: confirmed on the heads soup (+97 Elo over Gumbel@256 there). Tune it by games, not by the
  offline judge: worlds, horizon, k, the z threshold, and the KL rule at a larger t.
* **Make it fast.** Its rollouts step the engine one Python call per state; batched C++ rollouts
  (`VectorizedBatchRunner` already steps many states) would cut most of its 8x time.
* **Do not spend GPU on search-target training or on value-head fine-tuning for search** -- both are
  measured null here.
* **Search does not compound into the network here** -- not Gumbel's picks, not the rollout root's
  (section 5). The gain is banked by searching at play time; improving the network itself goes
  through RL and representation (the A8 heads are the precedent: Wargames was fixed by giving the
  policy a card-conditioned readout, not by showing it the move).
* **Longer horizons next:** 4 boundaries beat 2 by +35 on the soup; 6-8 (or to the turn's end) is the
  obvious next point, with the time cost measured.

## Replicate

The runs above were made on the fork, branch `exp/chance-search`; with this change the same
workflows run here (the bank's positions are read from the fork's release, `release_repo`).

```bash
# 1 and 3: searchers on the bank (reproducibility + paired playouts), and a merged report
gh workflow run search_reliability.yml --ref main                    # Gumbel worlds
gh workflow run search_reliability.yml --ref main \
  -f specs="h0=rollout:4:64:0:argmax h2z2=rollout:4:32:2:z2 h4z2=rollout:4:16:4:z2"
PYTHONPATH=. python tools/search_reliability.py report --bank bank.jsonl.gz \
  --search run1/search-*.jsonl.gz run2/search-*.jsonl.gz --playouts run1/playouts-*.jsonl.gz \
  run2/playouts-*.jsonl.gz --out combined.md
# 2: rollout labels on CI, the probe locally
gh workflow run value_probe.yml --ref main
PYTHONPATH=.:build/release python tools/value_probe.py fit --model E7line_swa_4720-4800M.pt \
  --labels labels/label-*.jsonl.gz --bank bank.jsonl.gz --bank-playouts full/playouts.jsonl.gz \
  --out probe.md
# 4: games
gh workflow run searcher_tournament.yml --ref main \
  -f entrants="name:net:{BASE} name:gumbel256:gumbel:{BASE}:256:8 name:rollout-h4z2:rollout:{BASE}:4:16:4:z2"
```
