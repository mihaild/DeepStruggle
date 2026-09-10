# P5 — Oracle critic (and the belief head that rides with it)

**Status:** queued
**Gate:** after P2, so a deal-side variance reducer is not confounded with the deal-side
bootstrapping change. If P2 is demoted, run this in its slot.
**Needs approval:** none — the code exists.

## Goal

Measure the privileged critic that is already implemented and has never appeared in the
experiment log: `oracle_loss_coef = 0.25` and `belief_loss_coef = 0.10` in
`ai/training/nash_pg.py` (`get_batches_with_oracle`, `forward_all(obs, mask, opp_hands)`,
`evaluate_oracle`, `predict_belief`; oracle loss `mse(oracle_val, ret_win)`, distillation
`mse(v_win, oracle_val.detach())`, belief loss `BCE(pred_belief, opp_hands)`).

## Why

- In a game with weak hidden information and heavy chance, the oracle critic is mainly a
  **deal-side variance reducer**: a critic that sees both hands has the hand luck removed from
  its target, and the public critic distils from it (Suphx, `paper_suphx_mahjong.md`). That is a
  different mechanism from the one Suphx needed it for (Mahjong's hidden information is strong),
  and it is the mechanism that fits here.
- The belief head is the cheap version of what the VOA-awareness goal needs: P(VOA in the
  opponent's hand) is one of its 110 outputs. Whether it is any good has never been measured.
- It is the only arm in the queue that costs zero implementation time.

## Change

None to code, if the paths still run on the current layout (v2.1 changed the observation; the
oracle path consumes `opp_hands` separately, so check it, and check that `forward_all` exists on
the categorical head from P1). *Decide before running:* the two coefficients — keep the defaults
for the screen; do not tune on the screen.

Everything else fixed at the current baseline.

## Procedure

1 arm × 2 seeds × 80M: baseline + oracle + belief, against the baseline. If it helps, a second
pair with the belief loss off, to attribute the effect. Confirm the winner at 240M.

## Measure

Pre-deal calibration and the ordinary calibration curve (the oracle's own curve and the public
critic's); belief head quality — AUC of `pred_belief` against the true opponent hand, and
specifically P(VOA in hand) when it is; the VOA-exposure probe; then Elo.

## Decision rule

- Adopt if the public critic's calibration improves with Elo not worse than −20.
- If the belief AUC is near chance, the head is not learning from the BCE alone — log it and
  queue "belief head with a real target weight" in reserve rather than tuning here.
- If Elo drops by more than 20: the distillation term is pulling the public critic toward
  values it cannot know from its own inputs (the classic oracle failure); halve
  `oracle_loss_coef` once, and if that does not recover it, kill.

## Follow-ups

- VOA-exposure probe still at baseline after adoption → the belief head is learned but the value
  does not use it; that is a P6-style architecture question (does the fused vector carry the
  belief?), not a loss-weight question.
- Oracle adopted + P2 adopted → both deal-side reducers are in; the remaining chance is the
  policy's own sampling, which Q-boosting proper (an action-value critic, `references.md` §3)
  would address. Reserve.

## Runs

(none yet)
