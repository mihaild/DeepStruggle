# Code & Research Audit Findings (2026-11-22)

Read-only audit of the codebase and research record at commit `fb9b817` (`main`, 2026-11-22). Every finding below was verified directly against source files; no code was modified and no runs were started.

## Index by Area

| Area | File | Highlights |
|:---|:---|:---|
| C++ engine & observations | [engine.md](engine.md) | `OLYMPIC_GAMES` sponsor flip, `STAR_WARS` US-executed USSR wars, Grain Sales interactions (`WWBY`, `MISSILE_ENVY`, `allow_early_stop`, FYP self-draw, stale headline IDs), `fire_owed_event` stale `start_influence_nodes`/`op_mode`, `PEEKED_TEMP` leak to USSR, P56 headline blindness, macro/canonical/held-scoring edge cases |
| Training loop, pools & rewards | [training.md](training.md) | P56 missed rolls after forced steps, P55/P56 ~150–250× pre-deal row weighting, P48b `RestartPool` `v_now` wipe + `KeyError` crash + Turn-2 leak, `RolloutBuffer` uninitialized GPU memory (`NaN`), `--no-decay-norms-biases` optimizer moment scrambling, `ending_reason` misclassifying ±20 VP Final Scoring, `BlunderAwareRewardCalculator` USSR 0-penalty bug, `online_distill` 400× `v_vp` bug, `OpponentPool` bugs |
| Model architectures & scaling | [models.md](models.md) | `--mup` no-op on `TokenTransformer` / `entity` / widened heads + `0.25×` `SphereBlock.w2` penalty + coupled decay, `TokenTransformer` Dynamo cache exhaustion & unmasked empty slots, forward/reload crashes, `pe_macro`/`pe_card`/`head_norms` wiring gaps, `SphereTrunk` non-RL/EMA breaks, `--update-bf16` mismatches |
| Search, belief & bindings | [search_and_belief.md](search_and_belief.md) | Float32 `-1e9` freezing belief BMA permanently, belief BMA overconfidence vs humans (`POINT_NODE` locking low $T$), `chance=keyed` USSR `-1` bug still active on leaderboard/browser, `RolloutBatch` 1-world `Rule::Z` + suicide-guard keyed/Ortega/Che bugs + `kPredeal` stopping before Final Scoring, mixed-rule root mask mismatches, `BatchedSearch::set_buffers` UAF, `BeliefTracker` null-deref/stale hands, `BIND-1` status |
| Research record & methodology | [research_record.md](research_record.md) | Invariant 14 contradicting pure self-play flagship lineage, stale `runs.md` / `checkpoints.md` / `plans/README.md`, unlogged P51–P55 & R138–R169 runs, single-seed (S45) chain on recent base pivots |

> **Engine change rule:** Per [AGENTS.md](../../AGENTS.md) Invariant 11, all items in [engine.md](engine.md) require explicit owner approval before any edit under `engine/`.
