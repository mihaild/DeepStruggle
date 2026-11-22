# Research Record, Bookkeeping & Methodological Notes

Audit at commit `fb9b817` (2026-11-22).

---

## Documentation Drift & Contradictions

### R-1. Invariant 14 contradicts the active flagship recipe
- **Locations:** [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md)
- **Issue:** Invariant 14 still states that every RL run needs `--opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12` to prevent runaway collapse. However, P47 (`R132` / B6) showed the E3 spread pool spent ~50% of pool games against weak early snapshots (475M–5B), creating 8–13 pp handover style bias and locking bad setups, while pure self-play (`--opponent-frac 0`) had no collapse to 12.4B and reached J1's 13.6B strength in ~6B steps. Every recent flagship run (`R132`, `R139`, `R155`, `R164`–`R170`) runs `--opponent-frac 0`.
- **Impact:** Any agent following Invariant 14 verbatim when launching a new run will silently reintroduce the E3 spread pool that the research program abandoned.

### R-2. Core tracking files are far behind `HEAD`
- **[runs.md](../runs.md):** Stops around `R123` (missing `R124`–`R170`, including B6, `R138`, `R139`/`R140`, `R141`–`R163`, `R164`, and `R165`–`R170`), and many completed Oct/Nov rows still read `**Running**`.
- **[checkpoints.md](../checkpoints.md):** Still lists only the E7 leaderboard — no E8 / E8.1 standings at all, even though B6 soup (`1774` belief / `1732` raced 64-world / `1573` greedy) and J1 (`1704`) are live in `leaderboard/players.json` and `leaderboard/matches/E8.jsonl`.
- **[plans/README.md](../plans/README.md):** Still lists P45, P46, P47, and P48 as proposed/queued despite P46 and P47 being largely complete and P48a+P48b (`R138`) having been run.
- **Unlogged recent plans/runs (`P51`–`P55`, `R138`–`R169`):** `P51` (`--rollout-halves`, reverted), `P52`/`P52b` (`--update-bf16` / `--bf16-fp32-heads`), `P53`–`P55` (`R165`–`R169`), and `R138`–`R163` have no plan or log writeups in `research/plans/` or `research/log/` — their results exist only in git commit messages and `research/run_codes.json`.

---

## Methodological & Experimental Risks in Recent Runs

### R-3. Single-seed (S45) chain across major base pivots
- **Context:** `B6` from scratch (`R132`), `R139` (low LR `1e-4`), `R155` (post-fix rebuild), `R164` (P50 per-player GAE), `R165`–`R169` (P53–P55), and `R170` (P56) all rely on a single seed (`S45`), despite the repo's established seed-variance rule ([measurement_pitfalls.md](../method/measurement_pitfalls.md): single-leg gain carries ±16 Elo, single snapshot oscillates ~30 Elo).
- **Specific risk in `P55` → `P56`:** `R169` (`P55`) became the base checkpoint for `P56` (`R170`) with `--afterstate-value-coef 1.0 --afterstate-frac 1.0 --predeal-value-coef 1.0` without a logged strength read vs plain `B6`/`R155` at matched steps, even though [training.md](training.md) (T-2) shows `predeal_value_coef = 1.0` applies ~150–250× per-row gradient weight to ~60–120 pre-deal rows per minibatch and triples total critic weight on the shared trunk. (P56's planned `C'` seed replicate is a good step; `R169` itself should also be checked against `B6`/`R155`.)

### R-4. Experimental interpretations invalidated by code bugs found in this audit
Several negative or inconclusive past experiments were affected by silent code defects and should not be treated as settled negatives until re-tested after fixes:
1. **P48b (`R138`, restart pool):** `RestartPool.provide()` wiped `v_now` before `complete()` read it ([training.md](training.md) T-6), so the adaptive disagreement put-back rule was never actually tested on rollout critic error.
2. **P38 / P39 / P41 wider `TokenTransformer` (`--mup`) runs:** `--mup` was a no-op on `TokenTransformer` (`token_dim` not varied in `base_shapes_for`), `TokenBlock` lacked $1/d_h$ attention and $1/\sqrt{2L}$ residual scaling, `_run_compiled` fell back to eager past Dynamo's 8-entry cache ([models.md](models.md) M-1, M-5), and those runs trained under the style-biased E3 pool.
3. **P35 belief root vs humans ([P35_belief_vs_people.md](../log/P35_belief_vs_people.md)):** Overconfidence (`-3.3%` LL gain) stems from BMA locking low $T$ via `POINT_NODE` agreement with no $T = \infty$ prior option, compounded by float32 `-1e9` accumulator freezing ([search_and_belief.md](search_and_belief.md) S-1, S-2).
4. **Online distillation (`online_distill.py`) & P44 Step 2:** `online_distill.py:85` scaled `v_vp` loss down by 400× ([training.md](training.md) T-16).
5. **P49b per-region & ending exploiters (`R145`–`R163`):** Clamped ±20 VP Final Scoring was misclassified as `"20 VP"` (dropping `final_share_us` and mis-crediting `final_scoring`/`20vp` exploiters), and Europe Control's `±20 - prev_vp` teleport distorted `RegionTargetedReward.scored_vp` ([training.md](training.md) T-10, T-13).
