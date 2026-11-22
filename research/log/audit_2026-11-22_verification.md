# Verification of the 2026-11-22 external audit

The audit is in `/workspace/research/gemini_findings/` (untracked; written by another model against
commit fb9b817). Every finding was checked against experimental-main, whose engine, models and search
code equal fb9b817's (only `ai/training/afterstate_critic.py` and the P56 bindings changed since).
Four independent checks, one per area, most verdicts backed by a repro script driven through
`ts_engine` or the Python modules (scripts in the session's scratch directory, `tmp/audit/`).
Nothing was edited and no run was started; every engine or observation fix needs the owner.

**Overall: almost nothing is fabricated, much is overstated, and most findings need flags or card
combinations no current run uses.** The handful that touch current training or deployed players are
listed first.

## What touches current training or deployed players

| ID | verdict | what it is | proposed fix |
|:---|:---|:---|:---|
| E-9 | TRUE (obs) / FALSE (mask) | `fire_owed_event` leaves the phasing side's `start_influence_nodes` and `op_mode` from the Ops it just finished: the "can I place here" board slot is wrong in 16 countries after e.g. CIA Created Ops-first -- 2.4% of decisions, 12 of 20 random games; op-mode slot reads COUP during a Decolonization placement. The mask is right (`begin_op_mode` retakes the snapshot). | engine (owner): clear both in `fire_owed_event`. Changes the base observation's content for every network. |
| E-2 | TRUE | Star Wars fires the chosen card's Event as the US (`card_dispatcher.cpp:1244`): Korean War / Arab-Israeli War give the US the war and its +2 Military Ops. A policy can learn it. | engine (owner): fire as the card's side; decide Flower Power's charge for a war through Star Wars (card text says it applies). |
| E-16 | TRUE | After the first headline pick the picker's observation is bit-identical for every card (the card stays `HAND_*`): P56's afterstate cannot tell headline picks apart. At lambda_opp 1 (T1) the node telescopes away; at 0.8 (A8) the pick's own credit is shrunk with no d1 to replace it. | trainer (no engine change): no afterstate node for the first headline pick. Showing the committed card is an observation change (owner). |
| T-2 | PARTLY | Each extra critic is a mean over its own rows: at coefficient 1.0 a pre-deal row weighs ~31x a decision row, an afterstate row ~2.7x (the audit's 150-250x is wrong); each term carries the main value loss's total weight, so the value terms sum to ~3x vf_coef. Contradicts the flag's help text ("rows weighted by this x --vf-coef"). R169 was judged not weaker with it. | decide the meaning: document "coef = share of the total", or normalise per row (a new R code). |
| T-1 | TRUE | P56 node: a forced single-choice step between the move and its die makes the walk start AFTER_DECISION, so the roll's luck stays in the advantage (1.3% of roll decisions on random play); a two-roll walk removes only the first roll's luck. No bias at lambda_opp 1, less noise removed. | trainer: take the first BEFORE_ROLL row with outcomes in the decision's walk (d1 = average - V_t), each roll's luck dropped. |
| T-4(2) | TRUE, small | Per-player GAE bootstraps the side not to move at the rollout boundary from the negated opponent value (`cV[1] = -LV[e]`): one tail row per env per side per rollout. Same in P50. | value the last observation from both sides and pass one value per side. |
| E-14 | TRUE (critic only) | Our Man in Tehran's `PEEKED_TEMP` cards show in the USSR's observation; the USSR never decides there, but the afterstate capture records exactly that observation. | observation (owner): show the peeked slot to the US only. |
| E-1 | TRUE | Olympic Games' sponsor read from `phasing_player` at the roll: wrong after a USSR Grain Sales draws it for the US (USSR won 81% of rolls instead of the sponsor). | engine (owner): store the sponsor before the roll. |
| S-7 | TRUE | The rollout search's suicide guard misses Che / Ortega coups (op_mode is not COUP at their target choice), CMC coups at DEFCON > 2, and in keyed mode sees other dice than the playout. All leaderboard rollout players; rare. | bindings: guard those target choices; guard on the CMC flag, not DEFCON alone; re-key before each chance step. |
| S-6 | TRUE | z2 switches on one valued world (standard error 0) -- reachable with Cambridge Five or candidates needing different scoring cards. | require >= 2-4 valued worlds before switching. |
| S-9.2 | TRUE, silent | `entry:` / `sparse:` wrappers expose no `model` / `obs_features`: an E8.1 network inside plays its unsearched moves under E8 rules. | delegate `model`, `obs_features`, `observe`, `new_games`, `select_actions_games` to the inner agent. |
| S-1 | PARTLY | Belief: -1e9 for moves no hand allows, accumulated in float32 at turn fold; temperatures round to 64-nat steps and later turns' evidence vanishes (not "frozen at 1/6" as claimed). Fires on rare trap escapes, and always against merged/macro opponents (none on the board). | drop decisions that rule out every hand; store evidence in double; refuse or decode non-E4 opponents. |
| S-5 | PARTLY | keyed still the default (web, every leaderboard keyed player); keyed2 never measured. Deliberate and documented (861c956); the audit overstates which fields drop. | one keyed2-vs-keyed match, then switch defaults if not worse. |
| R-1 | TRUE | Invariant 14 (an opponent pool for every run) contradicts every flagship run since B6 (pure self-play, owner's go 2026-11-15); `watch_run.py` reports the missing pool as "the failure itself". | owner: rewrite invariant 14; make `watch_run.py` accept a deliberate `--opponent-frac 0`. |

## Engine (`engine.md`) -- every fix needs the owner

| ID | verdict | note |
|:---|:---|:---|
| E-1, E-2, E-9 | TRUE | above |
| E-3 | TRUE | We Will Bury You's check at play-mode choice ignores who is phasing: under a USSR Grain Sales the US clears the debt by playing the drawn card as Event (0 VP); rare. Fix: require US phasing at top level. |
| E-4 | TRUE | `clear_force_for_card`'s extra Missile Envy clause matches any card: an owed Missile Envy cleared by a Grain Sales draw played for Ops; rare. Fix: drop the clause. |
| E-5 | TRUE | Five Year Plan moves the discarded card before its Event: Shuttle Diplomacy in discard while active; rare. Fix: run the Event, then move. |
| E-6 | TRUE | Grain Sales offers Five Year Plan itself from discard (rare); Latin American Debt Crisis counts itself under Containment (one extra forced prompt, trivial). Fix: a hand-scan helper skipping every frame's resolving/pending card. |
| E-7 | TRUE | Headline card ids kept all turn: a re-drawn headline card can stay in hand / its Event not fire (Missile Envy reproduced for real); very rare. Fix: clear at headline end. |
| E-8 | PARTLY | Grain Sales' `allow_early_stop` leaks a DONE slot at Olympic Games / Wargames, but it duplicates an existing legal branch -- not "illegal / DEFCON suicide". Fix: reset at play-mode commit. |
| E-10 | TRUE | Flower Power not charged for a US war headline; rare. |
| E-11 | TRUE | UN Intervention fires via Missile Envy during the headline; rare. Fix: skip it there as for Star Wars / Grain Sales. |
| E-12 | TRUE (reading) | Missile Envy and trap blocks return before AR8's optional pass; very rare, rules ambiguous. |
| E-13 | PARTLY | the decline path exists but is practically unreachable. |
| E-14, E-16 | TRUE | above |
| E-15 | TRUE (latent) | optional blocks use the unresolved perspective; no caller passes NONE. |
| E-17 | TRUE | canonical-order arms only: a forced single-choice decision surfaces. |
| E-18 | TRUE | P42 macro view only: macro spans removal steps. |
| E-19 | TRUE (low) | ending classification for `blunder_aware` (not the default reward). |
| minor x3 | TRUE, negligible | Missile Envy path contexts, Summit dice convention, memcmp padding. |

## Training (`training.md`)

| ID | verdict | fires on current runs? | note / fix |
|:---|:---|:---|:---|
| T-1, T-2, T-4(2) | see above | yes | |
| T-3 | TRUE (probe) | no (`--per-seat-adv-norm` unused) | `empty_like` leaves a seat with <= 1 row uninitialised (NaN on CUDA possible). Fix: `zeros_like`, global stats fallback. |
| T-4(1) | TRUE | no | per-player GAE ignores slice-turn-boundaries / blunder windows; refuse or pass `turns`. (3) mostly false. |
| T-5 | (1)(2)(4)(5) TRUE, (3) FALSE, (6) PARTLY | no (critic prior used only in R127-R131) | refuse critic-prior with the extra critics, same-perspective bootstrap, or no pool. |
| T-6 | TRUE | no | `RestartPool.provide()` wipes `v_now` before `complete()` reads it: **R138's (P48b) disagreement / critic metrics are invalid**. Fix: snapshot the value at completion. |
| T-7, T-8, T-9 | TRUE | no | restart pool KeyError when no whole game ends; turn-2 restarts leak; forced-opening silently drops the restart pool. |
| T-10, T-13 | TRUE | no | ending classifier calls a clamped +-20 Final Scoring "20 VP"; Europe Control's jump booked as Europe VP -- **distorts the region / ending exploiters (R156-R162)**. Fixing T-10 changes ending-mix metrics and game stats. |
| T-11, T-12 | TRUE | no | blunder-aware held-scoring reward cannot express loser -1 / winner 0; VP-potential wrapper hides the region reward. |
| T-14 | TRUE (probe) | no | `--no-decay-norms-biases` resume can swap optimizer moments. Fix: order by `named_parameters()`, guard by name. |
| T-15 | TRUE, loud | no | |
| T-16 | TRUE | no | online-distill v_vp loss 400x too small (extra `/20`). |
| T-17, T-18 | TRUE (one sub-item unreachable each) | pooled runs only | notably `--seed-sampling/env/pool` ignored on resume when `--seed` is unchanged. |

## Models (`models.md`) -- none changes what an A8 run computes

`--mup` gives A8 no rate multipliers at all (by design), bf16 is off.

| ID | verdict | note |
|:---|:---|:---|
| M-1 token/entity no-op | TRUE, moot | no run ever changed those widths from the base. |
| M-1 sphere w2 x0.25 | TRUE | the expansion factor read as width: **A30/A32 sphere runs under `--mup` trained w2 4x slower**. Fix `base_shapes_for` before revisiting spheres. |
| M-1 coupled decay | TRUE (convention) | no effect on A8. |
| M-2, M-3.1, M-3.2 | TRUE, loud | unused combinations / refusals. |
| M-3.3 | TRUE, **silent** | card-lookup factorisation and head count hard-coded on reload: a non-default checkpoint would misload. Fix: store them as buffers. |
| M-4 | PARTLY | head registries incomplete (`pe_macro`, `pe_card`, ...); the head_center claim is false. Fix: one registry. |
| M-5.1, M-5.3 | FALSE | no Dynamo cache exhaustion (4 graphs in 60 calls); RMSNorm works under autocast. |
| M-5.2 | TRUE (design) | duplicate empty slots encode hand size. |
| M-6 | TRUE, low | sphere projection only in the main optimizer. |
| M-7 | PARTLY | under `--update-bf16`: old log-probs fp32, new bf16 -- may explain part of P53's cost; fix before bf16 is retried. Mask padding crash for macro + injector. |

## Search and belief (`search_and_belief.md`)

| ID | verdict | note |
|:---|:---|:---|
| S-1, S-5, S-6, S-7, S-9.2 | see above | |
| S-2 | PARTLY / not shown | the cause is the P35 log's own untested hypothesis; run the choices-only probe (`all_decisions=False`). |
| S-3 | TRUE, rare | belief count test and evidence dropped on record shrink. |
| S-4 | TRUE, documented | probe ignores Space Race box 4 (P35 records it). |
| S-8 | TRUE by design | `stop=predeal` leaves turn-10 final scoring to the critic; no leaderboard player uses it. |
| S-9 other | mixed | 9.1 loud and narrow, 9.3 loud, 9.4 latent, 9.5 information loss only. |
| S-10, S-11 | TRUE as API hazards | no caller triggers them; add keep_alive / input validation. |
| S-12 | TRUE | `ctx_stack_depth` setter unchecked; BIND-1 probably fixed by db5f79e. |

## Research record (`research_record.md`)

| ID | verdict | note |
|:---|:---|:---|
| R-1 | TRUE | above |
| R-2 | TRUE | `runs.md` stops at R123 (16 rows still "Running"); `checkpoints.md` has no E8/E8.1 standings; `plans/README.md` still shows P45-P48 as proposed or queued; P51-P55 have no write-up (results only in commits, `run_codes.json` and `data/reports`). |
| R-3 | PARTLY | the single-seed (S45) chain is true; "R169 never read against B6 at matched steps" is false -- P55 read it every 400M (52.7 / 47.5 / 47.6 / 53.8 / 50.3%, `data/reports/p55_critics_vs_b6_*`), unwritten in `research/`. |
| R-4 | follows the code verdicts | P48b (T-6) and the region / ending exploiters (T-10, T-13) are affected; the P38-P41 token-transformer claim rests on M-1/M-5, of which the cache claim is false and the muP one moot; P35's overconfidence cause is not shown (S-2); online distill (T-16) true. |

## The second external pass: the fixes checked (`fix_verification.md`)

A second pass by the same model, [`fix_verification.md`](../gemini_findings/fix_verification.md), checked the fix
commits on `main` through `7b8671b` against every finding above (regressions V-1 to V-3, incomplete fixes V-4 to V-9,
leftovers V-10). Checked TRUE and fixed on the `fix-misc` branch (2026-11-22, owner's go):

| ID | what it is | fix |
|:---|:---|:---|
| V-8 | Europe Control decided at final scoring goes through no Event, so the region reward booked 0 for it (and the ending reads "Europe Control", so final scoring's share was skipped too): a Europe exploiter's win there paid `control_win` instead of the cap | `ai/rewards/region_reward.py` books +/-100 itself (`final_scoring_vp_us`); real-game tests to every ending |
| V-10 schema | `headline_first_owner` (E9) missing from `GameStateDict`; the golden schema is generated from the TypedDicts, so it could not notice -- nor the older drift in `die_roll`, `decision_context`, `legal_actions`, `hands` | the TypedDicts carry every engine key (save dicts too); a test holds engine keys to typed keys, at run time and in the source |
| V-10 S-7 | the guard's fixes (CMC at any DEFCON, Che / Ortega target choices, keyed probe dice) had no tests | `tests/bindings/test_rollout_batch.py`, one per case, the dice under keyed, keyed2 and keyedsym |
| V-10 R-2 / R-4 | log index, two commit references, `runs.md` notes (T-10 / T-13 on the region and ending exploiters, M-1 on the sphere muP runs), the training diagram's pool | this record |

The log's in-game Europe Control line also read "US +100 VP" beside the score's move to 20; it now says the win.
