# The arm registry — every arm, and where its result is written

One row per arm. The row says what the arm varied, at which seeds and budgets, which directory
under `data/checkpoints/` holds it, and **where the result is written up**. Where there is no
writeup the row says *no writeup* rather than leaving the cell blank: an arm that was run and
never reported is a fact about this project, and hiding it behind an empty cell is how an arm
gets re-run.

The naming scheme itself — what `E3-20-28-160M` means, when the engine letter bumps — is
[`method/run_nomenclature.md`](method/run_nomenclature.md). If you are looking for *what we
concluded* about some question rather than for a particular arm, start from
[`questions.md`](questions.md) instead; this file is indexed by arm, and most questions span
several.

**Update discipline: rewritten in place.** A row is edited when an arm gains a budget, a seed or
a writeup. The per-arm notes below the tables are the exception — a prediction registered before
a run and the outcome recorded against it are history, and those are not edited once written.

**How to add a row:** [`method/bookkeeping.md`](method/bookkeeping.md).

**Two caveats about the evidence behind this file.**

* `data/checkpoints/` is git-ignored. Every `directory` cell names something that can be deleted
  without any commit recording it, and several results cited below exist *only* as a
  `report.md` in such a directory. Those are marked **untracked**.
* `metadata.json` is authoritative for flags and seeds and is not authoritative for *purpose*:
  the `description` field is free text written at launch, and in two cases (E3-20-22, E3-19-22)
  it names a different programme number from the one the plan files use.

---

## E3 — the current engine

Observation v2.3, `blunder_aware`, K=40, `eta` 0.1, 512 envs, cold start unless noted.
"Varied" is what distinguishes the row from the control, E3-01.

| # | arm | varied | seeds | budgets | directory | result written up in |
|---:|:---|:---|:---|:---|:---|:---|
| 01 | control | — | 21 | 80M, 160M, 240M | `p1_scalar_nofilter` | [findings/architecture.md](findings/architecture.md); [log/P9_architecture.md](log/P9_architecture.md) |
| 02 | categorical head, target in normalised units | 21 | **VOID** | `..._VOID_unscaled_target` | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) — died to a units bug |
| 03 | categorical head, `v_vp` in real VP | 21 | **VOID** | `..._VOID_vp_scale` | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) — died to a scale bug |
| 04 | categorical head, `--vf-coef` sweep | 21 | **VOID** | `..._VOID_vf_coef` | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) |
| 05 | categorical head, derived `v_win` | 21 | **VOID** | `..._VOID_derived_baseline` | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) |
| 06 | categorical head, `--value-dist-coef 0.02` | 21 | 80M | `p1_categorical_nofilter` | [findings/architecture.md](findings/architecture.md) — −28 Elo |
| 07 | `--adv-filter-quantile 0.5` | 21, 22 | 80M, 160M | `p1_scalar_filter_seed2026092*` | [plans/P1](plans/P1_categorical_value_advantage_filtering.md) — +25 Elo, +24 at 160M |
| 08 | `--window-provoked-defcon` | 21, 22 | 80M | `p1_window_provoked_seed2026092*` | [findings/defcon_blunders.md](findings/defcon_blunders.md) |
| 09 | `--arch mlp`, backbone control | 21, 22 | 80M | `p1_mlp_backbone_seed2026092*` | [log/P9_architecture.md](log/P9_architecture.md) — −110 Elo |
| 10 | `--identity-dim 16` | 21, 22 | 80M, 160M | `p1_identity_seed2026092*` | [findings/architecture.md](findings/architecture.md) — +107/+115 Elo |
| 11 | `--arch mlp --drop-static` | 21, 22 | 80M | `p1_mlp_static_seed2026092*` | [log/P9_architecture.md](log/P9_architecture.md) — −40/−0 Elo, not adopted |
| 12 | `--self-transform` (on identity) | 21, 22 | 80M | `E3-12-2*_<ts>` | [findings/architecture.md](findings/architecture.md) — +53/+83 Elo |
| 13 | `--self-transform --attn-readout 64` | 21, 22 | 80M | `E3-13-2*_<ts>` | [findings/architecture.md](findings/architecture.md) — −14/−23, rejected |
| 14 | `--self-transform --per-entity-heads 64` | 21, 22 | 80M | `E3-14-2*_<ts>` | [findings/architecture.md](findings/architecture.md) — −219 Elo, rejected |
| 15 | as 14 but **residual** | 21, 22 | 80M, 160M | `E3-15-2*_<ts>` | [findings/architecture.md](findings/architecture.md) — +181 over 12 |
| 16 | E3-15 recipe, `--graph-layers 1` | 21 | 80M | `E3-16-21_<ts>` | [findings/seed_variance.md](findings/seed_variance.md) — **withdrawn** |
| 17 | E3-15 recipe, `--graph-layers 0` | 21, 22, 24, 25, 26 | 80M, 160M, 320M | `E3-17-2*` | [findings/seed_variance.md](findings/seed_variance.md), [findings/pooling.md](findings/pooling.md) |
| 18 | E3-17-22 continued 160M → 240M, unchanged | (22) | 240M | `E3-18-22` | **no writeup** — see note |
| 19 | pooled opponents from a mid-run resume, frac 0.30 | (22), 23 | 160M | `E3-19-22`, `E3-19-23` | [findings/pooling.md](findings/pooling.md) — partly; see note |
| 20 | pooled opponents from scratch (`--opponent-self-pool`, frac 0.3, size 12) | (22), 27, 28, 29 | 160M, 320M | `E3-20-2*` | [findings/pooling.md](findings/pooling.md), [log/seed_variance_and_pooling.md](log/seed_variance_and_pooling.md) |
| 21 | E3-20 recipe + `--same-perspective-bootstrap` | 28 | 160M | *(none on disk)* | **not run** — registered only; see note |

Seed `21` is 20260921, `22` is 20260922, and so on — `28` is 20260928. A seed in parentheses
means the run was launched with `seed=None`: the last two digits are the arm's label, not a
seed that was set. See *E3-20 and the unseeded arms*.

Continuation legs use a fresh sampling seed so they diverge rather than replay: E3-01-21's legs
used 20260931 and 20260941, and E3-10's use 20260951 and 20260952. Those are properties of a leg
rather than of the configuration, so the registry records them and the name does not.

The four VOID rows are numbered rather than omitted. They exist on disk, they consumed 6.1 of one
session's 22 GPU-hours, and a registry whose purpose is to identify a directory unambiguously
cannot leave four directories out of it.

## E2 — the corrected engine, before the mandatory-choice fixes

| # | arm | varied | seeds | budgets | directory | result written up in |
|---:|:---|:---|:---|:---|:---|:---|
| 01 | arm H | baseline on the corrected engine + v2.3 | — | 160M | `arm_H_v23_corrected` | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| 02 | arm H2 | second seed of H | 21 | 160M, 240M, 480M | `arm_H2_v23_seedB`, `arm_H2_cont_160to240`, `arm_H2_cont_240to480` | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) |
| 03 | arm I | `eta` 0 — NashPG KL penalty off | 21 | 80M | `arm_I_no_kl` | [log/corrected_engine_arms_H_I.md](log/corrected_engine_arms_H_I.md) — −169 Elo |

`E2-02-21` is one lineage across four budgets, so the strongest checkpoint in the project is
**`E2-02-21-480M`**, and it is the standing rating anchor.

## E1 and older — the lettered arms

Pre-starred-card fix. **Not runnable**: the observation layouts they used are deleted and
`check_checkpoint_layout` refuses their checkpoints by width. Kept because a great deal of the
log is written in their names.

| arm | varied | budgets | directory | result written up in |
|:---|:---|:---|:---|:---|
| A / B | BC warmup from self-play vs from the human corpus | 80M | `run_v2_20260906_153116_e3_{synth,human}` | [log/P7_human_bc_warmup.md](log/P7_human_bc_warmup.md) |
| D | legacy layout, cold start, no injection, K=40 | 80M–320M | `arm_D_*` | [log/observation_layout.md](log/observation_layout.md) |
| D′ | D replicated at a new seed | 80M | `arm_Dp_cold_noinject_seed5` | [log/variance_and_noise.md](log/variance_and_noise.md) |
| E | observation v2.1 — card tracking, no dead history | 80M–320M | `arm_E_*` | [log/observation_layout.md](log/observation_layout.md) — neutral |
| F / F2 | observation v2.2 — decision context in | 80M, 160M | `arm_F_v22_cold`, `arm_F2_v22_cold_seedB`, `arm_F_cont_80to160` | [log/observation_layout.md](log/observation_layout.md) — +91.7 Elo |
| G / G2 | v2.2 + engine flag `staged_cards` | 160M, 320M | `arm_G_v22_staged`, `arm_G2_v22_staged_seedB` | [log/observation_layout.md](log/observation_layout.md) — flag not demonstrated |
| var A/B/C/C2/C3 | warm start vs cold start vs synth-only, for variance | 80M, 160M | `var_*` | [log/variance_and_noise.md](log/variance_and_noise.md) |

## Named A/B pairs from before the scheme

These predate `--run-name` entirely. Their directories are the only names they have.

| pair | question | directories | result written up in |
|:---|:---|:---|:---|
| blunder window | confine an unprovoked blunder's penalty to its own turn | `abw_window_{on,off}`, `bw_{window,nowindow}_ref_ent` | [log/early_training_signal.md](log/early_training_signal.md) §2.1 — retained, evidence needs re-checking |
| decisive-transition priority | `--priority-alpha` | `dec_prio_{on,off}` | **no writeup** — [log/early_training_signal.md](log/early_training_signal.md) §2.2 records that the conclusion was never captured |
| decisiveness K | K=20 vs K=40 length-scaled reward | `dec_turns20`, `dec_turns40` | [log/agent_deficiencies_and_decisiveness.md](log/agent_deficiencies_and_decisiveness.md) |
| **start-pool A/B (first)** | `--start-pool-frac` — resume self-play from saved mid-game positions | `sp_pool_{on,off}` | [log/early_training_signal.md](log/early_training_signal.md) §3.1 — **withdrawn, confounded** |
| **start-pool A/B (second)** | the same, step-budgeted at 78M | `sp2_pool_{on,off}` | [log/early_training_signal.md](log/early_training_signal.md) §3.2–3.3 — settled negative |
| injection dose | human-data injection every 1 / 4 / 16 iterations | `inj_every*`, `inj_none` | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |
| injection × human init | human warm start with and without injection | `hum_inj1`, `hum_none` | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |
| injection weight grid | weight × frequency, 8 cells | `g_control`, `g_w*` | [log/P7_human_injection_and_its_cost.md](log/P7_human_injection_and_its_cost.md) |
| slice / reference ablation | `abl_slice_ref`, `abl_noslice*` | `abl_*` | **no writeup** — metadata says only "ablation arm <name>" |
| V4 oracle-critic fine-tunes | privileged oracle critic distillation | `run_v4_2026082*` | **no writeup in `research/`** — predates the log |

`sp_pool_*` and `sp2_pool_*` are the **start-state** pool, not the opponent pool. The two are
unrelated mechanisms with opposite verdicts; see [`findings/pooling.md`](findings/pooling.md).

---

## Per-arm notes

Predictions registered before a run, and the outcomes recorded against them. **Not edited once
written** — a prediction that failed is the most useful thing on this page.

### E3-12 and E3-13 — the graph self-transform and the attention read-out

**E3-12 and E3-13 carry `--identity-dim 16`, and their control is E3-10, not E3-01.** Identity
is part of the recipe as of the identity arms ([`log/P9_architecture.md`](log/P9_architecture.md)),
so an arm that omitted it would be measuring identity again.

Both target what `log/P9_architecture.md` localised (*exact influence and control, read off the
trunk per country*): a country's exact influence is recoverable from its own raw observation slots
97% of the time, from its post-GraphConv token 63%, and from the pooled 512-float trunk
essentially never. E3-12 gives each graph layer a second weight matrix applied to the node itself,
so a country need not be averaged with its neighbours; E3-13 adds an attention read-out at the end
of the trunk, where the state vector queries the 84 country and 110 card tokens — each
concatenated with its raw slots, because the token is itself already damaged — and folds the
result back into the trunk.

Predictions were recorded before the runs, so the result could not be read backwards: E3-12
should lift the `gconv` rungs and leave the trunk near zero; E3-13 should lift the trunk toward
the token's 63%. Elo may not move at all, and that would itself be the finding.

**Outcome: the first held, the second failed, and Elo moved.** E3-12 lifted `gconv1` recovery from
60% to 93–94% on both seeds and is worth **+53 and +83 Elo**. E3-13's read-out left the trunk
where E3-12 already put it and cost the whole gain, **−14 and −23**. A single-query read-out is
still a pooling operation, so it could not have done otherwise. `--self-transform` is adopted;
`--attn-readout` is not. See `log/P9_architecture.md`, *fixing the graph layer works*.

### E3-14 — per-entity policy heads, built wrong

**E3-14** keeps E3-12's adopted self-transform and replaces the dense policy head for the actions
that name an entity. A card's logit is computed from that card's own token and raw slots, and a
country's from that country's, each conditioned on a projection of the trunk; the 18 actions that
name no entity — play mode, timing, op mode, branch, confirm — stay dense.

It follows directly from the attention read-out's negative result. The read-out failed because a
single-query read-out is still a pooling operation: one query over 84 countries returns one
weighted average. The token holds 89–91% of the recoverable exact influence and the pooled trunk
holds 6–14%, so the remaining move is to stop routing the board through the trunk at all.

Prediction, recorded before the runs: **the trunk ladder should not move** — nothing here changes
what the trunk holds — and the Elo should, because the information now reaches the decision by
another path. If Elo does not move while a country's logit demonstrably tracks its own influence
(`tests/training/test_arch_variants.py`), then exact per-country influence is not what the policy
was missing, and the self-transform's +53/+83 came from somewhere else in the representation.

**Outcome: both halves of the prediction were wrong.** The trunk moved *up* (battleground
recovery 26.9% against E3-12's 14.3% and 6.1%) and Elo collapsed to **−219** against E3-12-21 —
below even the E3-10 control. The cause is in the implementation: `pe_trunk` projects the trunk
512 → 64 before the heads see it, so every action logit lost seven eighths of its view of the
situation to gain its own country's detail. The idea is untested; this build of it is refuted.
The residual form (`logit = dense + correction`) is what to try. See `log/P9_architecture.md`,
*per-entity policy heads, built wrong*.

### E3-15 — the same idea as a residual

**E3-15** is E3-14 rebuilt as a residual: `logit = dense + per-entity correction`, with the
correction's output layers zero-initialised so the network *starts* as the dense baseline exactly.
E3-14 replaced the dense head, which made a 64-float projection the only path from the trunk to
any logit and cost 219 Elo; here the full trunk still reaches every logit through `base`, and the
narrow context limits only how much situation the correction itself can see.

Predictions, recorded before the runs:

* It cannot start worse than the dense baseline, so a repeat of E3-14's collapse would mean the
  correction *learns* something harmful rather than that it begins from a bad place —
  a different and more interesting failure.
* The floor is E3-12, which it begins as. The question is only whether the correction adds.
* No strong prediction on the trunk ladder. E3-14 moved it up while halving play, which was the
  session's clearest demonstration that a representation probe moving the right way is not
  evidence a change helped, so the ladder is recorded and not used to judge the arm.

### E3-16 and E3-17 — the depth of the map graph

**E3-16 and E3-17** vary only the depth of the map graph, on the E3-15 recipe, at seed 21 —
the same seed as `E3-15-21`, so each is a paired comparison against it and depth is the only
difference.

The motivation is that adjacency's *mechanical* uses are already precomputed per country in the
observation: placement legality and coup legality are board slots, and the realignment modifier is
another. The graph does not derive them. What it adds is strategic reasoning about
neighbourhoods, and the map's edges are plainly unequal — Colombia bridges two regions, Benelux
to West Germany rarely decides anything — and some are live only while a particular event is in
force. None of that is what a degree-normalised convolution computes.

Predictions, recorded before the runs:

* **1 layer matches 2.** The second convolution has lost 5–9 points of per-country influence in
  five arms out of five and no measured gain offsets it.
* **0 layers is the real test and has never been run.** The MLP arm removed every structured
  encoder at once, so the graph alone has never been isolated. If 0 also matches, adjacency is
  not earning its place in this architecture.
* Both should be **faster** than 2 layers, which is the practical point: a cheaper backbone makes
  every later experiment cheaper.
* The `gconv1`/`gconv2` rungs of the trunk ladder are not comparable across depths — with one
  layer or none they tap the same tensor — so only `raw` and `trunk` are read across arms.

**Outcome as first recorded, at 80M and seed 21.** Anchored on E3-17, which has no adjacency at
all: two layers −59 Elo, one layer −84, and the no-graph arm is also the fastest at 13,863
steps/s against 12,159. Read at the time as *removing the map graph is not neutral but positive*.

**That reading is withdrawn.** A later tournament rated both seeds of E3-15 and E3-17 in one
field and the sign of the comparison reverses with the seed: matched on seed 21 E3-17 is much
better, matched on seed 22 E3-15 is. Seed spread is ~95 Elo, an order of magnitude above the
1.8pp binomial error on the pairings. The depth question is **open**, and answering it needs
more seeds per arm, not more games per pairing. See
[`findings/seed_variance.md`](findings/seed_variance.md) and `log/P9_architecture.md`, *seed
variance is ~95 Elo*. The throughput figure is unaffected — it is not a play-strength claim.

E3-17 keeps a per-country encoder, so every country is still encoded from its own observation
slots and identity embedding. What is gone is *only* the adjacency mixing. It is the recipe every
E3-18 through E3-21 arm is built on, which is why an open architecture question sits underneath
the whole pooling programme.

### E3-11 — dropping the constant observation slots

E3-11 is an ablation *of* E3-09 rather than of the control: it drops the 1,364 observation slots
that never vary (35.7% of the input, 6.6M parameters against E3-09's 8.0M), which a positional
reader should not need because it recovers entity identity from the offset. It is the row most
likely to be misread, so what it measured is written next to it: **−40 and −0 Elo** against E3-09
across its two seeds, a mean inside a seed spread of 40, at identical throughput (64.4k vs
65.5k steps/s). The static mask was verified independently — 0 of the 1,364 claimed-constant
dimensions take a second value across 16,800 states, counted by distinct values rather than by
standard deviation. Not adopted.

### E3-18 and E3-19 — the rows that were called missing

An earlier version of this registry said: *"Attempts 18 and 19 are not recorded here. Runs named
`E3-20-*` exist on disk and the table jumped from 17 to 20; the two missing rows were never
written down and are not reconstructed speculatively."*

They were written down — in `metadata.json`, which the registry had not been read against:

* **E3-18-22** — `data/checkpoints/E3-18-22/metadata.json`, base commit `3df7566d`: *"P10 exp1:
  continue E3-17-22 160M→240M unchanged; does critic AUC and adv_std recover without
  intervention"*, resumed from `E3-17-22.../resume_160038912steps.pt`. That is P10's experiment 1,
  the do-nothing control. **Its result is not written up anywhere in `research/`.** Its final
  checkpoint appears in the untracked `data/checkpoints/arena_heads/report.md` under the label
  `H64-E3-18-22`, and `log/seed_variance_and_pooling.md` notes only that it has no 80M snapshot
  and was excluded from the US-decay table.
* **E3-19-22** — same commit: *"P10 exp4: pooled frozen-opponent sampling, 30% of envs, 8
  snapshots spanning E3-17-22, learner resumed from 80M, sides alternating. Judged on
  `adv_std_raw` and critic AUC/Brier staying up, not on the US recovering."*
* **E3-19-23** — *"Seed replicate of E3-19-22: pooled frac=0.30 from 90M (80M resume was pruned),
  seed 20260923, on a rented 4090. Matched control is E3-17-22 90M-160M."* The resume point
  differs from E3-19-22's by 10M steps because the 80M state had been deleted, so the two
  "replicates" are not started from the same place.

E3-19 is the pool applied to a run already in progress; E3-20 is the pool from scratch. They are
different experiments and were numbered as such. See [`findings/pooling.md`](findings/pooling.md).

### E3-20 and the unseeded arms

`E3-20-22/metadata.json` records `seed: null` and describes the run as **P11**, not P12: *"P11:
pooled opponent sampling from scratch, growing self-pool at frac=0.30, capacity 12, to 160M.
Tests prevention rather than repair — the pool contains only this run's own history."*

`E3-17-24`'s own metadata states the consequence plainly: *"E3-17-22 (no pool) ran `--seed
20260922` while E3-20-22 (pool) ran `seed=None`, i.e. a different env stream AND an unseeded net,
so the two differ in more than the pool."* The `-22` in `E3-20-22` is therefore a label, not a
seed, and the original pooled-vs-no-pool pair was **unpaired**. The replication arms (24–29) each
set `--seed` explicitly, which is what makes them a replication rather than a repetition.

An earlier registry listed seed `21` for E3-20. **No `E3-20-21` directory exists on disk** and no
document describes one; the entry is unsubstantiated and has been dropped rather than carried
forward.

### E3-21 — the value-bootstrap arm

**E3-21 is the value-bootstrap arm.** It copies E3-20-28 flag for flag, seed included, and
adds `--same-perspective-bootstrap`, which takes the value target from `V(s_{t+1}, p_t)` —
the result as the player who just moved sees it — instead of negating the next step's
value. The negation assumes `V(s, me) = -V(s, opponent)`, which holds under perfect
information and not here: measured over 227 positions, `v_US + v_USSR` has mean absolute
0.144 where the identity requires 0. E3-20-28 is therefore a matched baseline at the same
seed. See [`log/search_cost_and_coverage.md`](log/search_cost_and_coverage.md) §8.

**No `E3-21-28` directory exists under `data/checkpoints/`.** The arm is registered (commit
`9f47c3e`) and, as far as this record shows, has not been run.

One caveat on the pairing, for when it is: `b6874af` and `9f78026` changed `engine/` after
E3-20-28 was trained. The training decision stream is unchanged — no `ROLL_DIE` node is ever
handed to an agent (0 in 879 single-env and 12,800 vectorized env-steps) and the runner always
forces die 0 — so the engine letter stays E3 and the pair is comparable. That is a measured
claim, not an assumption, and it is the reason the letter was not bumped.
