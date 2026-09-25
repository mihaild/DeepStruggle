# P24 — an AlphaStar-style league, without a supervised starting point

**Status:** proposed, not started. Gated on the slow-π_ref replicate. [P25](../archive/E4_ladder/plans/P25_collapse_robustness.md) closed without a lever: collapse on the fixed pool is a recoverable delay. P23 (E4.1) is parked. **New constraint (2026-09-24):** a `--warmup-checkpoint` start (fresh optimizer, π_ref and pool) dissolves the US seat within 5M, even in the E4 view ([`../log/P23_E4_1_ab.md`](../log/P23_E4_1_ab.md)). Resetting an exploiter to an early snapshot is exactly such a start, so find out which reset does it before building the league.
**Needs approval:** none of it touches `engine/` or the observation. It needs one trainer change
(a pool that grows from other runs' directories) and a driver built on `tools/train.py`.

## Goal

Stop the late-training drift and oscillation by giving the main agent opponents that go looking
for its weaknesses, rather than only its own past selves.

## Why

* **The failure is a drift into one-seat specialisation, not an orbit** (E3,
  [`../archive/E3_ladder/plans/P15_breaking_the_cycle.md`](../archive/E3_ladder/plans/P15_breaking_the_cycle.md)
  X3), and E4 shows the same thing: side collapses are common and recoverable
  ([`../log/E4_collapse_is_recoverable.md`](../log/E4_collapse_is_recoverable.md)), the setup
  preference flips every 5–25M ([`../log/P21_M2d_setup_west_germany.md`](../log/P21_M2d_setup_west_germany.md)),
  and late changes that loosen the update collapse ([`../log/E4_late_dynamics.md`](../log/E4_late_dynamics.md)).
* **A pool of past selves cannot punish a weakness it shares.** Every pool member descends from
  the same lineage. When the current policy stops contesting West Germany, no member was trained
  to take advantage of that. The goal probes show weaknesses of exactly this kind: forced wins
  taken 59–70% of the time, and the US opening at 25% against the human 57%
  ([`../log/E4_goal_probes.md`](../log/E4_goal_probes.md)).
* **What E3 already ruled out:** PFSP weighting over self-snapshots (null), and one exploiter
  against a frozen target that had no weakness left to exploit (X1). A league differs from both:
  exploiters train against the *moving* main agent, and are reset so they keep searching.

## Design

AlphaStar's league has three roles. It resets its exploiters to a supervised policy, which keeps
them human-like and diverse. **We have no such policy**: our human corpus is 266 games, and a
policy cloned from it washed out within 2M RL steps on E3. So the reset target is the design
question.

| role | trains against | reset | adds to the league |
|:---|:---|:---|:---|
| **main agent** | the whole league (its own snapshots + every exploiter's), PFSP-weighted | never | every 5M |
| **main exploiter** | the main agent's latest snapshots only (`--opponent-frac 1.0`) | when it beats the main agent >70% on both seats, or after 40M steps | each snapshot while it is still improving against the main |
| **league exploiter** | the whole league, PFSP | with probability 0.25 at each 40M boundary | every 5M |

**Reset targets without supervision, in order of preference:**

1. **An early snapshot of the main agent** (~40M): competent at the basics, before the late drift.
   It is the closest analogue to AlphaStar's supervised policy, as a competent generalist that
   is not yet specialised.
2. **Scratch**: maximally diverse, but pays ~40M steps to reach competence each time. On one
   GPU that is too slow for a main exploiter; possibly right for a league exploiter.
3. **Human BC** (P7): human-shaped (sane setup, contested battlegrounds), which is exactly the
   diversity self-play lacks. It washes out under RL, but as a *reset point* that matters less,
   since it is re-applied at each reset. Worth one arm once 1 and 2 are measured.

**Per-seat exploiters.** E3 X1 found a locked seat degenerates from 50% to 7% in 40M without
gradient, and our collapses are seat-specific. So the main exploiter trains **both** seats
(alternating, as now) but is **scored per seat**, and resets on either seat's success.

**Per-seat main agents — the asymmetric-game precedent (added 2026-09-23).** AlphaStar did not
train one main agent for all races; it trained **one main agent per race**, each with its own
exploiters, because the roles are different games. Twilight Struggle's seats are at least as
different, every collapse in the E4 census loses the same seat, and the P25 bench shows the
winning seat keeps sharpening inside a shared network while the losing seat is still learning
(E4-36-03: USSR entropy 1.65 → 0.9, US held at 1.75). One main agent for both seats is therefore
a *choice* this plan should test rather than assume. Two forms, in order of cost: **per-seat
adapters or heads on a shared trunk** (the reserve entry *per-side capacity*, promoted here as a
league design question), and **two main agents, one per seat**, each training only its seat
against a league that contains the other. The second halves each agent's data and doubles
parameters, so it runs only if the first is insufficient. The exploiters are then per-seat by
construction: a US exploiter attacks the USSR main, and vice versa. Judged as the rest of this
plan: per-seat strength against frozen anchors at matched wallclock.

## Change

* **Trainer (small):** `--league-dirs <dir> ...`, a pool that rescans other runs' directories each
  snapshot interval and adds their new snapshots. Each member keeps its own action view
  (`tools/lib/action_view.py`, P23) and its source run, for per-member PFSP statistics.
  `--opponent-checkpoints` today is fixed at launch, which is the only missing piece.
* **Driver:** a script that launches the main agent and the exploiters as ordinary `tools/train.py`
  runs sharing `/workspace/data/checkpoints`, checks exploiter success with `tools/tournament.py`
  every 10M, and relaunches an exploiter from its reset target. Everything goes through the CLIs
  (invariant 9); the league is the set of directories.
* **Names:** one attempt per role, e.g. `E4.1-10-03` main, `E4.1-11-03-<k>` the k-th main
  exploiter.

## Procedure

1. **Budget realism first.** On one 4090, three concurrent learners share ~45k steps/s, so the
   main agent gets ~1/3 of it. The comparison is at **matched main-agent steps and matched
   wallclock**, against a plain main arm with identical flags otherwise. The wallclock comparison
   is the one that decides adoption, since the exploiters cost the main agent throughput.
2. **Stage A (cheapest informative):** main + one main exploiter (reset to the main's 40M),
   160M main-agent steps, seed 3, against the plain run at matched main steps.
3. **Stage B:** add a league exploiter if A shows exploitable weakness (the exploiter reaches
   >60% against the main on either seat).
4. **Seed 5 replicate** of whatever wins.

## Measure

* **The exploiter's win rate against the main agent, per seat, over time.** If the exploiter
  never finds anything (<55%), the main has no easily found weakness, and the league is
  paying for nothing. That was E3 X1's outcome and is the first thing to watch.
* The goal probes on the main agent (setup, forced wins, battlegrounds) every 20M.
* Setup-preference flip rate (`setup_critic`) and side-balance oscillation, the measures the
  late-dynamics arms were judged on.
* Elo last, per seat, against both same-config baselines.

## Decision rule

Adopt the league for the long run if the main agent beats the plain arm **at matched wallclock** on
both seats at 160M, on both seeds. If it wins only at matched main steps, it is a better use of
steps but not of the GPU, and the long run should prefer plain training until throughput (P18) is
fixed.

## Follow-ups

* PSRO-lite meta-Nash sampling (reserve) if PFSP over a league is still neutral.
* Search distillation (E4-28) *inside* the exploiters: an exploiter trained with search targets
  is a stronger adversary per step, and the main agent never pays the search cost itself.
