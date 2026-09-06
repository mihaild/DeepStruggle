# E2 -- the human corpus on this engine

Measured by `ai/eval/human_corpus.py` over every game in `data/datasets/ts_replayer/`. Read alongside `research/experiments.md` 4.2, 4.4, 4.5 and 4.6, whose definitions and table shapes these follow.

## 0. Corpus accounting

| | count |
|:---|---:|
| files read | 300 |
| files that would not load | 0 |
| cached files holding no turns at all | 9 |
| games skipped by the converter | 9 |
| games whose conversion failed | 0 |
| games converted | 282 |
| ... of which reached a terminal state (`game_ended`) | 131 |
| ... of which are fragments (log stops first) | 151 |
| decisions classified | 144844 |
| emitted samples excluded: chance nodes | 0 |
| emitted samples excluded: no legal action in mask | 0 |

- skipped, replay 196: handicap US +5: the engine sets up US +2
- skipped, replay 222: handicap US +1: the engine sets up US +2
- skipped, replay 225: handicap US +1: the engine sets up US +2
- skipped, replay 230: handicap US +1: the engine sets up US +2
- skipped, replay 250: handicap US +3: the engine sets up US +2
- skipped, replay 257: handicap US +1: the engine sets up US +2
- skipped, replay 273: handicap US +1: the engine sets up US +2
- skipped, replay 287: handicap US +1: the engine sets up US +2
- skipped, replay 303: handicap US +1: the engine sets up US +2
- empty cached files, replays 90, 214, 215, 216, 217, 218, 254, 298, 309: `all_turns`,
  `unique_turns`, `hands` and `stats` are all empty in the cached download, so there is no
  game in the file to convert. Not a converter failure; verified by reading the files
  directly. (The first run of this measurement labelled these nine "conversion failed",
  which was wrong; `ai/eval/human_corpus.py` now separates them. No other number in this
  report is affected -- they contribute zero decisions and zero turn boundaries.)

**No game that had a log failed to convert.** 291 files hold a game, 9 of those are skipped
because the human match used a handicap the engine does not set up, and the remaining 282
converted -- consistent with `tools/README.md` §6.

## 1. Human game arc

**US win rate over finished games: 48.9%** (64 US / 65 USSR / 2 draws, N = 131 finished; 151 fragments have no outcome and are excluded from this rate).

Mean victory_points at each turn boundary (US-positive), against 4.6's agent rows:

| turn | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (4.6) | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |
| K=40 (4.6) | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |
| **human** | +0.21 | -1.09 | -1.58 | -1.80 | -1.34 | -0.19 | -0.62 | -0.01 | +2.36 |
| human N | 267 | 253 | 235 | 210 | 193 | 167 | 145 | 107 | 94 |

US win rate by how long the game ran (finished games only):

| | ends turns 1-4 | ends turns 5-6 | ends turns 7-10 |
|:---|---:|---:|---:|
| control (4.6) | 44.8% | -- | 39.5% |
| K=40 (4.6) | 38.7% | -- | 42.0% |
| **human** | n/a | 28.6% | 50.0% |
| human N | 0 | 7 | 124 |

## 2. Human forced-win behaviour

| | value |
|:---|---:|
| decisions classified | 144844 |
| instant-win opportunities | 149 |
| share of all decisions | 0.103% |
| games with at least one | 95 of 282 (33.7%) |
| taken | 71 |
| **take rate** | **47.7%** |
| US opportunities / taken | 93 / 45 |
| USSR opportunities / taken | 56 / 26 |
| declines | 78 |
| declines in games with a settled outcome | 73 |
| ... decliner won anyway | 58 |
| **cost of declining** | **20.5%** |
| declines in fragments (outcome unknown, excluded) | 5 |

## 3. Critic calibration on human positions

Checkpoint `/workspace/data/checkpoints/dec_turns40/snapshot_final.pt` (`snapshot_final`), value head `v_win`, over 84073 human decisions in the 131 finished games. Actual outcome is +1 if the mover won that game, -1 if they lost.

| v_win bin | N | mean predicted | mean actual | gap (pred - actual) |
|:---|---:|---:|---:|---:|
| [-1.00, -0.75) | 693 | -0.77 | -0.77 | -0.01 |
| [-0.75, -0.50) | 11951 | -0.61 | -0.43 | -0.18 |
| [-0.50, -0.25) | 18428 | -0.37 | -0.15 | -0.22 |
| [-0.25, +0.00) | 15458 | -0.13 | -0.03 | -0.10 |
| [+0.00, +0.25) | 15987 | +0.13 | +0.17 | -0.04 |
| [+0.25, +0.50) | 14945 | +0.36 | +0.23 | +0.13 |
| [+0.50, +0.75) | 6431 | +0.59 | +0.54 | +0.05 |
| [+0.75, +1.00] | 180 | +0.79 | +0.99 | -0.20 |
| **all** | 84073 | -0.06 | +0.01 | -0.07 |


---

## 4. Verdict on the gating question

**Yes -- the human corpus reproduces the standard arc on this engine, including the US
late-war recovery that 4.6 and 4.7 could not get out of any agent.**

The human VP curve has the shape the game is supposed to have and neither agent row has:

| turn | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control (4.6) | +0.15 | -1.02 | -1.87 | -3.03 | -3.58 | -3.91 | -3.11 | -2.49 | -4.47 |
| K=40 (4.6) | +0.01 | -0.91 | -2.26 | -1.96 | -2.20 | -2.31 | -1.95 | -1.68 | -2.18 |
| **human** | +0.21 | -1.09 | -1.58 | -1.80 | -1.34 | -0.19 | -0.62 | -0.01 | **+2.36** |

* The **USSR early lead is reproduced** and is about the same size as the agents': VP turns
  negative at turn 3 and troughs near -1.8 at turn 5. So the Early War asymmetry is not an
  agent artifact.
* The **US recovery is present and completes**: the human curve climbs monotonically from the
  turn-5 trough, crosses zero around turn 9 and ends the Late War at **+2.36**. The control
  ends at -4.47 and even K=40, the best agent, only claws back to -2.18 and never reaches
  parity at any turn.
* **Overall balance**: humans win 48.9% as US (64/65/2 over 131 finished games, 95% CI
  +/-8.6pp) -- statistically indistinguishable from even, and matching real Twilight Struggle.
  Agents on the same engine sit at 35.0-40.2% (4.5).
* **Long games do not favour the USSR** for humans: 50.0% US in games ending turns 7-10
  (n=124, +/-8.8pp), against the control's 39.5%.

**Therefore the engine is not the cause.** A rules bug favouring the USSR that only shows in
long games -- the hypothesis 4.7 left open -- would have to suppress the US late war for
humans too, and it does not. The 60-65% USSR skew in 4.5 and the missing recovery in 4.6 are
properties of our agents, not of the simulation they run in.

### What these numbers cannot establish

* **They do not clear the engine of every rules bug**, only of one large enough to erase the
  US late-war edge. A defect that humans route around, or one worth a point or two, would not
  show here.
* **They are not a matched comparison.** Human games and agent self-play games differ in
  everything at once (skill, length distribution, ending mix). The claim supported is "the
  arc is reachable on this engine", not "agents and humans were measured under equal
  conditions".
* **Turn-boundary populations shift**, exactly as 4.6 warns, and they shift for a *different*
  reason here. In the agent rows a game leaves the sample by ending -- and US wins end early
  on 20 VP -- so the surviving population skews USSR-favourable. In the human rows most of
  the attrition (267 -> 94) is the ts-replayer recording stopping mid-game, not the game
  ending: 151 of the 282 converted games are fragments. Recording truncation is *plausibly*
  outcome-neutral where an early 20 VP win is not, which if anything makes the human late-war
  numbers less USSR-biased than the agent ones -- but that neutrality is an assumption this
  measurement did not test.
* **The turn 1-4 bucket is empty for humans** (no human game in the corpus ended before turn
  5), so 4.6's short-game row has no human counterpart. The 5-6 bucket holds 7 games and its
  28.6% is noise (+/-33pp).
* **Nothing here says why** the agents fail to convert the late war. It rules out the engine;
  it does not distinguish among the remaining explanations (board presence by turn 8, critic
  myopia, the terminal-only objective).

## 5. Reading the other two measurements

**Forced wins (§2).** Humans take 47.7% of instant wins (71 of 149, +/-8.0pp) -- *below* every
agent configuration (control 81.2%, K=40 72.4%). Per 4.4 this is not a quality score, and the
human corpus makes that case more strongly than the agent data did: of the 73 declines whose
game the log settles, the decliner **won anyway 58 times**, so declining cost the game 20.5%
of the time (+/-9.3pp) -- the same order as the control's 18.2% and worse than K=40's 4.5-10.6%.
Humans who are 48.9%-balanced overall decline forced wins at twice the agents' rate; take rate
therefore cannot be read as strength in either direction.

Two structural notes:

* Opportunities are **rarer** in human games than agent ones: 0.103% of decisions against
  0.175-0.196% in 4.2. Human games reach a decisive position less often, which fits games that
  are longer and closer.
* The US/USSR opportunity split is **93 / 56**, the opposite tilt to the control's 92 / 158.
  This is the same finding as the win rate, from the direction 4.5 used: on this engine, in
  human hands, it is the *US* that gets more winning chances.

**Critic calibration (§3).** The 4.2 finding does not reproduce as stated. Over 84,073 human
decisions in finished games, `dec_turns40/snapshot_final.pt` is if anything *pessimistic* on
average (mean predicted -0.06 against mean actual +0.01), and the largest miscalibrations are
in the negative bins: at a predicted -0.37 the mover actually went on to score -0.15, and at
-0.61 they scored -0.43. The critic overstates how lost a losing-looking human position is.

The one band where it is optimistic in the 4.2 direction is **[+0.25, +0.50)**: predicted
+0.36 against an actual +0.23, over 14,945 decisions. That is the band 4.2's missed forced
wins sit in ("+0.626 when missed against +0.419 when taken"), so the effect is visible on
human positions too -- but it is confined to that band rather than being a global optimism,
and it is smaller (0.13) than the pessimism in the negative bins (0.18-0.22).

Caveat on interpretation: `v_win` predicts the outcome under *the agent's own* continuation,
while the actual outcome here is what the *humans* went on to do. A gap can therefore mean the
critic is wrong about the position, or that human play from that position differs from the
agent's -- these numbers cannot separate the two. The measurement is restricted to the 131
finished games; the 151 fragments have no outcome and are excluded rather than imputed.
