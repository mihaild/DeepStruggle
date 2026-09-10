# P6 — Global attention over country and card tokens

**Status:** queued
**Gate:** the §12.1 perturbation probe (`ai/eval/battleground_value.py`) is *still* flat — the
value does not respond to Poland's influence changing together with Europe Scoring's status —
after P1 and P2 have fixed the target. Architecture is last because `experiments.md` §5 found
capacity is not the bottleneck on v2, and §24 confirmed it from the other side: the one +90 Elo
jump in the project came from telling the same network more (layout v2.2), not from making it
bigger. A backbone change on a bad target measures nothing.
**Needs approval:** none (model only). The observation layout is unchanged — this re-encodes
the same floats, which is what keeps it out of the `CLAUDE.md` observation rule.

## Goal

Give every country a path to every other country and to every card in one layer, instead of
three hops on the map graph and then a pooled vector.

## Why

Interactions in this game are global: Poland's value depends on the Europe scoring card,
DEFCON, the VP track and what the US holds; the empty-battleground failure is a failure to
price far-apart regions against each other. The current backbone is a 3-layer graph convolution
over the 84-country adjacency (`ideas_and_plans.md` §3), so Europe and South America only meet
in the global pooling. Ataraxos and AlphaStar encode the board with a transformer for exactly
this reason (`references.md` §4). This is the one architecture change with a specific mechanism
behind it, which is why it is queued at all.

## Change

`ai/models/coldwar_net.py`: an encoder-only transformer over 84 country tokens + 110 card
tokens + a small set of global-scalar tokens, replacing the GNN + card MLP + fusion; heads
unchanged (P1's categorical head, the policy head over the 212-dim space, belief/oracle heads
if P5 adopted). *Decide before running:* width and depth such that forward cost stays within
~1.5× of v2 — the 4090 budget is the constraint, and the v3/v4 variants that exist in the code
were never well trained (`next_step_brief.md` §1), so do not assume their sizes are right. Keep
the map adjacency as an attention bias or a learned positional feature rather than dropping it.

## Procedure

1 arm × 2 seeds × 80M against the baseline at *matched steps*; wall-clock will be worse. Confirm
at 240M only if the probe moves — an architecture arm that is Elo-neutral and probe-neutral at
80M is dead.

## Measure

Perturbation probe first (this is the arm's whole reason), input ablation
(`ai/eval/input_ablation.py`) to see whether cards are actually attended to, empty battlegrounds
at turn 8, then Elo at matched steps and per wall-clock hour.

## Decision rule

- Adopt if the perturbation probe becomes responsive and Elo at matched steps is not worse
  than −20; accept a wall-clock cost up to ~1.5×.
- Kill if the probe does not move: then the value target, not the receptive field, still owns
  the failure, and the next arm is in reserve (shaping).

## Follow-ups

- Adopted: the card tokens make the belief head (P5) and a history encoder natural extensions;
  queue as reserve items with triggers, not as steps.

## Runs

(none yet)
