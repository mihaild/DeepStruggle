# P1 — Categorical value head + advantage filtering

**Status:** implemented, not yet run. `--categorical-value` and `--adv-filter-quantile`.
**Gate:** P0 probes exist (so the arms can be read by something other than Elo). The gate is
on *reading* the arms, not on running them — the four cells can train while P0 lands.
**Needs approval:** none (model and trainer only).

## Goal

Replace the two scalar value heads with one categorical distribution over final VP, trained by
cross-entropy; and drop low-|advantage| samples from the policy update. Two factors, screened as
a 2×2 so each is attributable.

## Why

- The critic prices battleground control as a step function with partial progress at zero
  (`experiments.md` §12), is over-pessimistic on human boards by 7–10 win-rate points (§17), and
  is over-optimistic exactly where forced wins get declined (§4). With λ = 0.98 the value targets
  are mostly critic values, so every one of these defects is copied straight into the advantages.
- Outcomes in a chance-heavy game are multimodal (coup hits or misses; the scoring card is or is
  not in the hand). MSE on a scalar regresses to the mean of the modes, a value that is never
  observed; cross-entropy on a distribution does not. Ataraxos, MuZero and Stochastic MuZero all
  use categorical heads for this reason (`references.md` §1, §3).
- Advantage filtering is reported at 2.5× wall-clock with a better asymptote in Ataraxos, costs
  nothing, and on a 4090 that is the difference between screening at 80M and screening at 200M.

## Change

`ai/models/coldwar_net.py`: `val_win_head` + `val_vp_head` → one head with 41 atoms over final
VP in [−20, +20] (the engine already normalises abrupt endings — 20 VP, DEFCON 1 — to ±20, so
the support is exact and the terminal utility is `sign`). Downstream consumers keep their
interface: `v_win = E[sign(VP)]` (or P(VP>0) − P(VP<0)), `v_vp = E[VP]`, both computed from the
distribution, so `nash_pg.py`, the tournament code and every probe run unchanged.

`ai/training/nash_pg.py` value loss (currently `mse(v_win) + vp_coef · mse(v_vp)`): cross-entropy
against the λ-return projected two-hot onto the atoms. *Decide before running:* whether the
target is the realised final VP (pure MC, simplest, noisiest) or the λ-return in VP units
projected two-hot (what Ataraxos does; needs the VP-valued return the buffer already computes
for `v_vp`). Recommend the latter, since P2 will want a bootstrapped target anyway.

Advantage filtering: drop samples with |A| below a quantile threshold (*decide before running:*
Ataraxos's setting, or the bottom 50% as a first guess) from the policy loss only; value loss
sees all samples. One flag, default off.

Everything else fixed at the current baseline recipe: the engine's one observation layout,
`blunder_aware`, K = 40, `eta` 0.1, cold start, no human data. (Written as "the arm F recipe:
v2.2 layout, `staged_cards` off" — both are gone. There is one layout and no engine flags, and
arm F cannot be loaded, so **the (scalar, no filter) cell has to be run fresh as the control**
rather than reused from F/F2 as this document originally assumed.)

`eta` stays at 0.1. Arm I removed it entirely and cost 169 Elo — not because the game is
intransitive, which it measurably is not, but because the penalty is doing stability work
(`experiments.md` §26). It is not a factor to vary here.

## Procedure

2×2: {scalar, categorical} × {no filter, filter}, 2 seeds each, 80M steps — 8 runs, ~12 h.
All four cells run fresh. The original plan reused arms F and F2 as the (scalar, no filter)
control; they are v2.2 checkpoints and cannot be loaded, so that shortcut is gone. Arm H2's 80M
snapshot is the nearest existing point of reference but was trained on a different recipe line,
so it is context and not the control.
Confirm the winning cell at 2 × 240M (~9 h).

## Measure

Report for every cell, alongside the list below: Elo against the **anchor** as well as
head-to-head against the control, and the per-side split (`game_won_us/`, `game_won_ussr/`). The
480M leg beat its own past by +65 Elo while *losing* ground to HeuristicBot, and its USSR win
rate climbed to 72% against a human 49.9% (`experiments.md` §27) — a cell can look like progress
on one of those and not the others.

In this order: calibration curve (decision states and pre-deal states); §12.1 perturbation probe —
does the value respond monotonically to influence short of control; empty battlegrounds at turn 8;
setup probe; forced-win take rate as a floor; then Elo (last four snapshots) and head-to-head
against the control.

## Decision rule

- Adopt categorical if the perturbation probe becomes monotone and calibration improves, and
  Elo is not worse by more than the between-run SD (~20). It is a target-quality change; it is
  allowed to be Elo-neutral at 80M.
- Adopt filtering if it is Elo-neutral-or-better at matched steps and faster in wall-clock; kill
  it if it costs more than 20 Elo at matched steps (it changes the effective batch).
- If categorical is worse on the probes, do not tune the atom count — look at whether the
  two-hot target is being computed from the right perspective (the alternating-sign bug class
  in `metrics.md` §1).

## Follow-ups

- Adopted: this becomes the baseline recipe for P2–P6. Write the new recipe line into
  `experiments.md` and update the control in every queued step.
- The 51-atom "VP & end types" head in `ideas_and_plans.md` §3 (extra atoms for the ending type)
  is a possible extension if DEFCON-1 endings stay mispriced; queue it in reserve, not here.

## Runs

(none yet)
