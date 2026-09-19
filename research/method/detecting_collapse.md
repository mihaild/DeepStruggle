# Detecting a seat collapse from internal metrics

Win rate against a frozen opponent is ground truth but arrives only every 5M steps, and **side
balance is not a collapse metric at all** — it cannot distinguish one side collapsing from both
sides improving unevenly. This page asks what the training loop's *own* metrics say, and the
answer is that the policy-update family says it almost immediately while the critic family says
nothing at all.

**Measured 2026-09-19** on the one paired dataset this project has: `E4-01-01` collapsed and
`E4-02-01` did not, same engine, architecture, reward and seed policy, **differing only in the
opponent pool**. All 74 metrics common to both runs were swept and ranked by standardised
difference, in windows *before* the collapse shows in outcomes.

> An earlier version of this page concluded that no internal metric separates the two runs. That
> was a selection effect: it hand-picked a dozen metrics and never tested entropy, the PPO ratio
> statistics or the clip fraction. The sweep below replaces it.

## 1. The policy-update family separates almost immediately

First step at which a rule fires, trailing means, no false positives on the healthy run:

| rule | collapsed | healthy |
|:---|---:|---:|
| `logratio_max` trailing-50 < 2.0 | **3M** | never |
| `ratio_negadv_max` trailing-50 < 6.0 | **3M** | never |
| `ending_frac_wargames` trailing-50 > 0.005 | 15M | never |
| `clip_frac` trailing-50 < 0.22 | 69M | never |
| `clip_frac` trailing-100 < 0.20 | 79M | never |

`logratio_max` and `ratio_negadv_max` fire at **3M steps** — about 1% of budget, and 180M before
the run had to be killed.

What they mean: both measure how far the policy moves in an update. The collapsed run's maximum
log-ratio sits at 1.71 against the healthy run's 2.72, and its max ratio on negative-advantage
samples at 3.9 against 15.0. **The policy is barely moving.** `clip_frac` says the same more
slowly — 0.17–0.20 against 0.24–0.32, so far fewer updates are large enough to hit the PPO clip.

## 2. Entropy: the signature is *failure to sharpen*, not entropy collapse

Trailing-100 entropy across each run:

```
COLLAPSED  1.68 1.46 1.46 1.39 1.40 1.41 1.40 1.42 1.39 1.42 1.40 1.46 1.46 1.45 1.44
HEALTHY    1.50 1.13 1.18 1.06 1.08 1.12 1.03 0.99 1.08 1.18 1.21 1.28 1.26 1.32 1.23
```

Both start high. The healthy policy **descends to ~0.99** as it learns to commit, then drifts
back up. The collapsed policy **plateaus at ~1.40 and never sharpens**. It is the opposite of the
intuition that collapse means entropy loss — here the failing run is the one that stays uncertain.

Standardised difference on entropy: **d = 2.51 early, 5.62 mid, 3.83 late** — one of the largest
separations of any metric, and present from the first window.

As a rule, the level is what matters, not the trend:

| rule | collapsed | healthy |
|:---|---:|---:|
| `entropy` trailing-100 > 1.30 at any point after 40M | **40M** | 136M |

Not a permanent separator — the healthy run's entropy does climb past 1.30 eventually — but a
clean one through the 40M–130M window, which is when you need it. **Minimum entropy reached by
100M** is the crisper statistic: healthy 0.99, collapsed 1.39.

## 3. Critic quality says nothing. Explicitly nothing.

Mean values, collapsed vs healthy:

| metric | 0–60M | 60–120M | 120–160M |
|:---|:---|:---|:---|
| `critic_auc` | 0.749 / 0.739 | 0.797 / 0.787 | 0.821 / 0.829 |
| `critic_brier_skill` | 0.185 / 0.169 | 0.261 / 0.243 | 0.304 / 0.322 |
| `explained_variance` | 0.775 / 0.809 | 0.859 / 0.857 | 0.889 / 0.882 |
| `value_loss` | 0.055 / 0.051 | 0.051 / 0.052 | 0.047 / 0.050 |

**The collapsing run's critic is as good as the healthy one's, and by some measures better**, all
the way to 160M. That is not a paradox: the critic's job is to predict the outcome of whatever
distribution it is shown, and a degenerate distribution is *easy* to predict. Critic quality
measures the critic, not the policy, and cannot be used as a health signal for the policy.

`critic_base_rate` is worse than uninformative. It is `max(p, 1−p)` — the majority-class rate,
always ≥ 0.5, carrying **no direction** — so it cannot name a side. E3 recorded that the answer it
appeared to give was *backwards* (`archive/E3_ladder/log/P15_control_per_seat.md`). A threshold on
it fired three times during E4-02-01 and was wrong every time.

## 4. Two metrics that separate strongly and must not be used

`steps_per_sec_avg` (d = 319) and `episodes_completed` (d = 3.6) are the top of the raw ranking
and are **configuration artefacts, not health signals**. The unpooled run is simply faster and
completes more episodes per iteration because 30% of the healthy run's games go to pool opponents.
They would separate these two runs perfectly and tell you nothing about any other pair.

## 5. Why this story is coherent

Everything above is one causal chain, which is the reason to trust it over any single correlation:

> no opponent diversity → low advantage variance → tiny policy updates (`logratio_max`,
> `ratio_negadv_max`, `clip_frac` all depressed) → the policy never sharpens (`entropy` plateaus
> high) → it never commits to a seat's strategy → one seat drifts and the other exploits it

The critic sits outside that chain entirely, which is why it is unaffected.

## 6. The protocol

1. **Iteration 1** — `opp_pool_size` present in the metrics at all. Absent in the collapsed run,
   present in the healthy one. Binary, no tuning, catches the cause rather than the symptom.
2. **From ~3M** — `logratio_max` and `ratio_negadv_max` trailing-50. A policy that is not moving
   is the earliest honest signal that something is wrong.
3. **From ~40M** — entropy has not descended. If trailing-100 entropy is still above ~1.3 when a
   healthy run would be near 1.0, the policy is not committing.
4. **Continuously** — `clip_frac`, as the slower confirmation of (2).
5. **Never** — critic quality, `critic_base_rate`, side balance, or `steps_per_sec`.
6. **To confirm before acting** — per-seat win rate against a frozen opponent. Slow and expensive,
   but it is the only ground truth.

## 7. What this does not establish

**One collapse, one control.** Every threshold here is fitted on the pair it is validated against.
The *ordering* of the metrics is more trustworthy than any specific cut.

**One failure mode.** Both collapses this project has seen were opponent-pool starvation. A
collapse from KL domination, a reward bug or a bad resume may present differently — though §5's
chain suggests the policy-update family would still be the place to look, since it measures
whether learning is happening at all.

**Absolute entropy levels are architecture- and action-space-specific.** 1.0 versus 1.4 means
something for a 220-wide action space and v2; it is not a portable constant.
