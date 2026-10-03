# Does South Korea's control move the soup's KAL-007 headline? (2026-10-03)

Owner: in self-play of the shallow_E7-02+03+04+05_1200M soup, when the US holds Soviets Shoot Down
KAL-007, is its decision to headline it affected by control of South Korea? Flip control with the
smallest change of US influence there and compare P(headline KAL-007) in four scenarios: natural
with/without US control, and the same positions flipped.

KAL-007's event degrades DEFCON by 1 and gives the US 2 VP; only if the US controls South Korea
does the US also get to place influence or realign with the card's 4 Ops. So control is what
makes the headline worth more than its 2 VP.

**Method.** `tools/scripts/kal007_sk_control.py`, 8,192 greedy self-play games (seed 56,000,
batches of 1,024). At every US headline card choice with KAL-007 in the US hand (2,035 positions),
read the policy's probability of the KAL-007 slot in the position as played, and on a copy where
only South Korea's US influence changes: controlled (US ≥ USSR + 3, stability 3) → US = USSR + 2;
not controlled → US = USSR + 3. The probe asserts that its own observation equals the one the env
played from, that the engine's `is_controlled_by` agrees before and after the flip, and that the
headline mask is unchanged by the flip. The engine is never stepped from the edited copy. The
headline is chosen sequentially but the opponent's committed card is hidden from the observation.
DEFCON is 3 in 99% of the positions (a headline is never at DEFCON 2, since DEFCON improves at
every turn's end).

## Result

| scenario | positions | mean P(headline KAL-007) | median | P > 0.5 |
|:---|---:|---:|---:|---:|
| natural, US controls SK | 1,193 | 0.551 | 0.707 | 56% |
| natural, US does not control SK | 842 | 0.437 | 0.258 | 44% |
| flipped: controlled → not (US = USSR + 2) | 1,193 | 0.521 | 0.606 | 53% |
| flipped: not → controlled (US = USSR + 3) | 842 | 0.491 | 0.493 | 50% |

Paired change in the same positions (flipped − natural):

| flip | positions | mean Δ | median Δ | Δ < 0 | Δ > 0 | \|Δ\| > 0.1 | greedy choice changes |
|:---|---:|---:|---:|---:|---:|---:|---:|
| lose control | 1,193 | −0.030 | −0.002 | 86% | 13% | 12% | 43 (3.6%) |
| gain control | 842 | +0.054 | +0.002 | 18% | 82% | 23% | 56 (6.7%) |

("Greedy choice changes" counts positions where P crosses 0.5.)

**Reading.**

* **The soup does read South Korea's control, in the right direction, but weakly.** The sign is
  right in 86% / 82% of positions. The size is small: the median position moves by 0.002, and
  the means (−0.03 / +0.05) come from a minority. The effect sits where the decision is open:
  among positions with natural P in (0.1, 0.9), losing control moves P by −0.085 on average
  (median −0.063, n = 333) and gaining it by +0.126 (median +0.116, n = 217). Where the soup is
  already sure (P near 0 or 1, about 70% of positions) control hardly matters to it.
* **Most of the natural gap is not control.** Positions where the US controls South Korea are
  headlined 0.551 against 0.437 without, a gap of 0.11. Flipping control in the same positions
  moves P by only 0.03–0.05, so most of that gap comes from what else differs between the two
  kinds of position. Gaining control closes about half the gap (0.437 → 0.491); losing it closes
  about a quarter (0.551 → 0.521).
* **What the positions look like.** Controlled: mostly US 3 / USSR 0, the US opening of South
  Korea untouched (658 of 1,193), so the flip there is one influence. Not controlled: mostly
  the USSR in control (USSR 3 / US 0, 235; USSR 4 / US 1, 83), so gaining control is usually a
  6-influence change. That edit is larger and also takes a battleground from the USSR, so the
  "gain" arm is not a pure control toggle in its other consequences: Asia standing, adjacency
  for Korean War-type rolls. The per-size breakdown in the tool output shows the 6-influence
  group (577 positions) at +0.055, the same as the arm overall.
* Turn split: about 40% at turn 8, 32% at turn 9, 28% at turn 10, the same in both arms.

**Not measured here:** whether headlining KAL-007 *without* control is actually a mistake (it is
still 2 VP and a DEFCON drop that can pin the USSR's headline response), and how the value head
reads the same flip. Either is the natural next step if the owner wants it.

Data: `data/eval/kal007/soup_8192.json` (per-position records; `--load` re-tabulates).
