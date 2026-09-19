# P21 arm M2d — the country head alone buys nothing, and collapses

**2026-09-19. Seed 1 complete; seeds 3–6 pending.** M2d is M2 with `pe_card` removed and
`pe_country` kept — otherwise identical, 3.185M parameters against M2's 3.190M, same trunk, same
seed policy.

| arm | Elo | steps/s | USSR all | US all | gap |
|:---|---:|---:|---:|---:|---:|
| `E4-03-01@80M` — anchor | 2146.1 | 11,732 | 79.0% | 80.7% | −1.7 |
| `M2@160M` | 2129.0 | 36,210 | 80.3% | 75.7% | +4.7 |
| `M2@80M` | 2095.2 | 35,744 | 76.8% | 72.2% | +4.7 |
| `M1@80M` | **1776.3** | 59,561 | 40.2% | 35.8% | +4.3 |
| **`M2d@80M`** | **1776.0** | 42,059 | 55.0% | **21.0%** | **+34.0** |
| `E4-04-01@80M` — defaults | 1710.1 | 14,057 | 34.0% | 27.7% | +6.3 |

## The country head alone recovers none of M2's +362

**M2d rates 1776.0 against M1's 1776.3** — within 0.3 Elo of the rung below it, while M2 with both
heads rates 2095.2. Against M2 directly it wins 26.0% as USSR and **2.0% as US**.

**This refutes the prediction.** [`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)
argued that `pe_card` should be near-inert because it cannot distinguish 95 of 110 cards without
identity, while `pe_country` can address a country through its dynamic slots — so `pe_country`
should carry the gain. The opposite happened.

The reasoning was incomplete rather than simply wrong. `pe_card` addresses *types*, and
**type-level information is what card timing needs**: Ops value, is-scoring, one-time, era and
where the card currently is. Deciding *when* to play a scoring card is a type-level judgement that
never requires knowing it is Asia Scoring rather than Europe Scoring. Card identity may matter far
less than card properties.

## It is the first collapse observed on E4

`us_episode_frac` fell monotonically from 0.68 to 0.0000, finishing at 0.035, with 12% of rows
below 0.02 — where M2 with both heads never dropped below 0.02 once in 1,221 rows. The result is a
**+34.0 pp side gap**, the largest anywhere in the ladder, and **−319 Elo** against M2.

The degeneracy signature tracks it:

| steps | `adv_std_raw` | `explained_variance` | `us_episode_frac` |
|---:|---:|---:|---:|
| 0M | 0.507 | −0.105 | 0.678 |
| 24M | 0.203 | 0.937 | 0.146 |
| 48M | 0.090 | 0.991 | 0.030 |
| 60M | 0.108 | 0.987 | 0.015 |

Advantage variance falls as the critic's `explained_variance` climbs to **0.99** — the critic
looks *excellent* exactly as the policy degenerates, because a one-sided policy is a trivially
easy prediction problem. That is
[`../method/detecting_collapse.md`](../method/detecting_collapse.md)'s "critic quality measures the
critic, not the policy" with a live instance.

The absolute `adv_std_raw < 0.01` threshold never fired (minimum 0.061), so that rejected
signature stays rejected. But the *direction* — advantage variance down, explained variance to
1.0, side fraction to 0 — now co-occurs with one-sidedness in a second independent arm.

## The confound, and what resolves it

M2d may sit at M1's level **because the country head is useless**, or **because this seed
collapsed and the collapse destroyed its rating**. One arm cannot separate them.

Seeds 3–6 at 80M are queued. If all four collapse, removing `pe_card` *causes* it and 1776.0 is
the mechanism's real value. If some do not, there are uncollapsed country-only arms to rate, which
separates the two explanations.

## Why this collapse is worth keeping

E3's collapses appeared at 200M+ steps in arms that were expensive to re-run, which is why they
were studied from logs rather than experiment. **This one reproduces at 80M in about 30 minutes on
a 3.2M-parameter network.** Every arm sets `--seed` explicitly and records it in `metadata.json`,
so `E4-08-01` is exactly reproducible and `tools/scripts/launch_flags.py` reconstructs its command.

A collapse that can be summoned on demand for half an hour of GPU is a far better object of study
than one observed once. If the seed sweep confirms the rate, this configuration becomes the
repository's first controllable instance of the failure mode.
