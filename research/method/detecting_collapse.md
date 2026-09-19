# Detecting a seat collapse from internal metrics

Win rate against a frozen opponent is the ground truth, but it arrives only at snapshot boundaries
— every 5M steps — and costs a hundred games a side. This asks what the *training loop's own*
metrics can tell you in between, and the answer is narrower than it looks.

**Measured 2026-09-19** on the one paired dataset this project has: `E4-01-01` collapsed and
`E4-02-01` did not, on the same engine, architecture, reward and seed policy, **differing only in
the opponent pool**. So a metric that separates them is separating collapse from health, not two
unrelated runs.

## 1. Dynamics metrics do not separate the two runs at all

Windowed means, collapsed / healthy, as a ratio. 1.00 means indistinguishable.

| metric | ≤40M | ≤80M | ≤120M | ≤150M | ≤185M |
|:---|--:|--:|--:|--:|--:|
| `adv_std_raw` | 1.07 | 0.99 | 1.01 | 0.95 | 0.88 |
| `critic_auc` | 1.03 | 0.99 | 1.02 | 0.99 | 0.98 |
| `critic_brier_skill` | 1.25 | 0.96 | 1.11 | 0.93 | 0.88 |
| `critic_base_rate` | 0.89 | 0.99 | 0.94 | 1.02 | 1.06 |
| `mean_turn` | 0.95 | 0.98 | 0.96 | 1.05 | 0.98 |
| `ending_frac_defcon1_won_ussr` | 1.09 | 1.08 | 1.14 | 0.92 | 0.88 |

Through 150M — 78% of the budget, and 30M steps *after* the collapsed run's US seat had already
halved — **nothing here is outside normal run-to-run variation.** The divergence only appears in
the last window, by which point the collapse is plain in the win rate anyway.

This is the central negative result. Watching `adv_std_raw`, the critic quality metrics or
`mean_turn` for an early warning does not work.

## 2. Absolute thresholds fail in both directions

First step at which a rule fires, on each run:

| rule | collapsed | healthy |
|:---|:---|:---|
| `us_win_rate` trailing-20 < 0.10 | 175M | **20M — false positive** |
| `us_win_rate` trailing-50 < 0.15 | 182M | **19M — false positive** |
| `us_win_rate` trailing-50 < 0.10 | 182M | never |
| `us_win_rate` trailing-100 < 0.15 | 183M | never |
| `adv_std_raw` trailing-100 < 0.18 | 182M | never |
| `adv_std_raw` trailing-100 < 0.15 | 184M | never |
| `critic_brier_skill` trailing-100 < 0.10 | 7M | **7M — false positive** |
| `critic_base_rate` trailing-200 > 0.85 | never | never |
| `mean_turn` trailing-200 < 5.0 | never | never |

Every rule is either **too late to act on** (182–184M, when the run was stopped at 184M) or
**fires on the healthy run too**. Short windows are noisy; long windows never reach the threshold
because the collapse occupies only the last ~4% of iterations.

The false positives have a cause worth knowing: **`us_win_rate`'s absolute level depends on
configuration.** With `opponent_frac 0.3` and alternating sides, 30% of those games are against
pool snapshots rather than the current policy, so a healthy pooled run sits near 0.33 where an
unpooled one sits near 0.40. An absolute threshold tuned on one is wrong for the other.

## 3. What does work: decline relative to the run's own baseline

| rule | collapsed | healthy |
|:---|:---|:---|
| `us_win_rate` trailing-50 below **50% of that run's own 20–60M mean** | **148M** | **never** |
| `adv_std_raw` trailing-50 below 50% of own baseline | 183M | never |
| `mean_turn` trailing-50 below 50% of own baseline | never | never |

**`us_win_rate` against its own baseline fires at 148M with no false positive** — 36M steps, about
15% of budget, before the run was actually stopped. It works precisely because it is
self-normalising: it does not care that the pooled and unpooled runs sit at different absolute
levels, only that one of them fell by half from where it started.

This is the rule to implement. `adv_std_raw` gives the same answer 35M later, which is
corroboration rather than warning.

## 4. The configuration check beats every dynamics metric

```
opp_pool_size present in the first metrics row:
  E4-01-01 (collapsed)   False
  E4-02-01 (healthy)     True  (= 1.0, seeded)
```

Available at **iteration 1**, binary, no threshold to tune. Both collapses this project has seen
were pool starvation, so the precondition is worth more than any downstream signal — check it
before the run has done anything, and again once snapshots start landing to confirm the pool is
*growing* (1 → 2 → 3 → …) and that `opp_win_rate_mean` stays off 1.0.

It only catches this one cause. But this one cause is 2 for 2.

## 5. What E3 already knew

From the archived ladder ([`../archive/E3_ladder/README.md`](../archive/E3_ladder/README.md)).
Several of these were written down before E4 and rediscovered the hard way anyway, which is itself
the argument for this page.

**`critic_base_rate` cannot name a side, by construction.** It is `max(p, 1−p)`
(`critic_tracker.py:140`) — the majority-class rate, always ≥ 0.5 and carrying **no direction**.
As a magnitude it is equally consistent with one side collapsing and with both sides improving at
different rates. `P15_control_per_seat.md` recorded that the answer it appeared to give was
**backwards**. A threshold on it fired three times during E4-02-01 and was wrong every time.

**Per-seat rating against a frozen anchor is the confirmatory test**, and the only measure that
distinguishes a collapsed seat from uneven improvement. It is not an internal metric and it is not
cheap; treat §3's rule as the trigger and this as the confirmation.

**No live training metric rates an arm past ~120M** (`P15_X0_frozen_anchors.md`). Consistent with
§1: the in-run instruments saturate, so late-run health has to be measured from outside.

**The collapse *fingerprint*, once it has happened**: `mean_turn` ≈ 4, ~60% of wins ending in
DEFCON 1, the policy beating its available opponent ~99%. X4b read 3.82 / 60.8% / 99.7%; E4-01-01
read 4.29 / 58.9% / 99.4%. Note §1 and §2 show these do **not** work as early warnings — a
trailing-200 mean of `mean_turn` never drops below 5.0 even on the collapsed run. This is how you
confirm a diagnosis, not how you catch one.

**The KL term can dominate the policy gradient** — 300× on half of `E3-31-28`'s iterations
(`P15_kl_domination.md`). An internal signal worth watching in its own right, and not one E4 has
instrumented.

**One network can hold both seats** (`P15_two_seat_capacity.md`), so a seat collapse is a training
pathology and never a capacity limit. That is what justifies treating it as a bug to find rather
than a trade-off to accept.

## 6. The protocol, cheapest first

1. **Before the run does anything** — confirm the startup banner says `[opponent pool] ... frac=0.3`
   and that `opp_pool_size` appears in the first metrics row. Diff the intended flags against a
   recent healthy run's `metadata.json` (`CLAUDE.md` invariant 14).
2. **Every few snapshots** — `opp_pool_size` growing to capacity, `opp_win_rate_mean` well below
   1.0, `opp_pool_span_m` rolling forward.
3. **Continuously** — `us_win_rate` trailing-50 against the run's own 20–60M baseline; alarm at
   50%. Do *not* use an absolute threshold.
4. **On an alarm** — confirm with a per-seat frozen-opponent eval before acting. Nothing in §1
   is worth acting on alone.

## 7. What this does not establish

**One collapse, one control.** The §3 rule is derived from the same pair it is validated on, so its
50% cut is fitted, not tested out of sample. It should be re-checked against the next collapse
rather than trusted as calibrated.

**Only one failure mode.** Both instances were opponent-pool starvation. A collapse from another
cause — KL domination, a reward bug, a bad resume — may present entirely differently, and §1's
negative result does not transfer to it.

**`us_win_rate` is configuration-dependent.** Its meaning changes with `opponent_frac`; the
self-normalising rule survives that, an absolute one does not.
