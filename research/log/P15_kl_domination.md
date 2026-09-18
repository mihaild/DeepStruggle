# E3-31-28's decline: the KL term is 300x the policy gradient on half its iterations

**Measured 2026-09-18.** With the severe X4b collapse explained as pool starvation
([`P15_X4b_collapse_is_pool_starvation.md`](P15_X4b_collapse_is_pool_starvation.md)), the
remaining unexplained result is `E3-31-28`: a **healthy** pool of 5, mean win rate 0.574, and
still a decline from 80.8% to 57.0% against `p28_280M` between 20M and 25M, driven by the USSR
seat (75.0% → 39.0%).

Its `kl_div` reaches **62.8** where the non-declining replay sits at 0.033. This is that number.

## The KL is bimodal, not a sawtooth

`kl_div` is a proper mean over every minibatch in the iteration, and π_ref refreshes only every
5M steps, so neither aggregation nor the refresh schedule can produce variation between adjacent
iterations. It does anyway — from 20.25M it alternates between ordinary and enormous:

| steps | kl_div | | steps | kl_div |
|---:|---:|---|---:|---:|
| 20,185,088 | 0.209 | | 20,774,912 | **34.99** |
| 20,250,624 | **1.758** | | 20,840,448 | **20.51** |
| 20,316,160 | **12.08** | | 20,905,984 | 0.154 |
| 20,381,696 | **47.46** | | 20,971,520 | **28.73** |
| 20,447,232 | **14.01** | | 21,037,056 | **46.52** |
| 20,512,768 | **24.35** | | 21,102,592 | 0.153 |
| 20,578,304 | 0.092 | | 21,168,128 | **44.89** |
| 20,643,840 | **35.87** | | 21,233,664 | **26.02** |
| 20,709,376 | 0.124 | | 21,299,200 | **32.63** |

Before 20.25M every value is between 0.05 and 0.22. After it, roughly a fifth of iterations stay
there and the rest are two orders of magnitude higher. Max over the run: **193.7** at 23.2M.

## What that does to the update

The optimised objective is

```python
policy_loss = ppo_loss + self.eta * kl_div - self.ent_coef * own_entropy + self.search_ce_coef * search_ce
```

with `eta = 0.1`. On a high-KL iteration the KL term contributes **0.1 × 47 = 4.7**, against a PPO
surrogate whose reported magnitude at those same iterations is **0.005 – 0.036** — a factor of
roughly **130 to 900**. On those iterations the update is very nearly pure "return to the
reference policy", and the advantage signal the arm exists to follow is numerically irrelevant.

That is a sufficient mechanism for a policy that stops improving and then slides, and it starts at
20.25M, which is where the decline starts.

## Why nobody saw it: `policy_loss` does not report the policy loss

```python
policy_loss_accum += ppo_loss.item()      # line 889
...
"policy_loss": policy_loss_accum / max(1, num_updates),
```

The logged `policy_loss` is the **PPO surrogate alone**. The KL, entropy and search-CE terms are
all in the tensor that gets differentiated and none of them are in the number that gets written
to `training_metrics.jsonl`. So the quantity actually being minimised has never been logged, and a
regulariser growing to 300x the surrogate shows up nowhere except in `kl_div` itself — which was
read as "the policy is drifting", the symptom, rather than "the update is now almost entirely
regularisation", the cause.

## A related inconsistency: the KL covers the opponent's decisions

```python
kl_div = torch.sum(cur_p * (cur_log_p - ref_log_p), dim=-1).mean()
own_entropy = (cur_entropy[b_learner > 0.5].mean() ...)
```

The surrogate and the entropy bonus are both restricted to `b_learner > 0.5`. The KL is not: it
is averaged over **every** row in the batch, including decisions made by a frozen pooled opponent.
The comment beside `own_entropy` states the reason that is wrong for entropy — "an entropy bonus
on a frozen opponent's choices would push the learner's policy toward states it did not choose to
be in" — and the same argument applies unchanged to a KL penalty.

This is also the most natural explanation for the *bimodality*. 30% of environments draw a pooled
opponent, the draw varies per iteration, and `E3-31-28`'s pool spans 20M steps; an iteration that
happens to draw distant snapshots is evaluated at states far from anything π_ref was fit on.
Nothing here proves that link — it is the hypothesis the numbers suggest, not a measured result.

## Status and what would settle it

**Not established.** `E3-31-28` and the non-declining `E3-34-28` differ in three things at once —
reference interval (5M vs 200k), pool size (5 vs 12), and from-scratch vs resumed — so the
contrast is suggestive only. What is established is arithmetic, and it does not depend on the
comparison: at `kl_div` 47 and `eta` 0.1 the regulariser is two to three orders of magnitude
larger than the surrogate it is regularising.

Three things follow, cheapest first:

1. **Log the loss that is optimised.** `policy_loss` should report the assembled `policy_loss`, or
   each term should be logged separately. This is a measurement fix and costs nothing.
2. **Decide whether the KL belongs on opponent rows**, and make it consistent with the surrogate
   and entropy either way. This changes the objective, so it is the owner's call.
3. **Then test the mechanism**: one arm at `ref_update_freq` 5M with the KL restricted to learner
   rows, against `E3-31-28` itself. If the bimodality is the pooled-opponent draw, it disappears.

Neither 2 nor 3 should be done while `E3-33-30` and `E3-35-28` are in flight on the same
objective.

---

## Appendix: the search_ce spikes, diagnosed (2026-09-18)

Chased while reading this code, and **not** a cause of any collapse — it occurs at the same rate
in the arm that does not decline, which rules it out cleanly.

`search_ce` exceeds 100 in 30–34% of iterations on every search arm measured (`E3-31-28` 34.3%,
`E3-34-28` 30.3%, `E3-35-28` from its first row), reaching 351,015. A 212-way softmax cannot
produce that: its maximum cross-entropy is log(212) = 5.36.

The mask fill is `-1e9`, so target mass *e* on a masked action costs *e* × 10⁹. The observed
117,306 / 90,744 / 43,244 imply ~1.2e-4 / 9.1e-5 / 4.3e-5 of misplaced mass.

**Cause.** `_search_targets` calls `BatchedMCTS.run()` and wrote visits straight into the target,
on the strength of a docstring asserting that BatchedMCTS filters against the caller's mask. It
does — on the *agent* path, not on `run()`. The search is determinized, and `batched_mcts.py`
says so where it solves this for an acting agent: *"a determinized search can legitimately return
an action that is illegal in the real state, because in this game the legal SET itself can depend
on hidden information"*, so there "the search proposes and the true mask disposes". Nothing
disposed here.

The arithmetic corroborates: one stray visit out of 64 simulations is 0.016 of a row, and diluted
over the searched rows of a batch that is ~1e5 of mean CE.

**Impact on training: negligible.** The CE gradient is (π − p_target), so a masked action
contributes ~1e-4, and `search_ce_grad_frac` stays flat through every spike. What it cost was the
metric — a third of the rows unusable — and what it warns about is the silent case: the identical
misalignment on an action legal in both the determinization and the real state produces an
ordinary CE. That is exactly how the earlier search-target off-by-one survived.

Fixed: illegal visits dropped, target renormalised over the survivors, a row with no legal visit
left untargeted. `search_dropped_visit_frac` and `search_dropped_row_frac` now report what the
mask rejected — where a **small nonzero value is the correct reading** and zero would mean the
filter had stopped running. Takes effect from the next launch; `E3-35-28` is running the old code.

---

**Item 1 done, same session (2026-09-18).** `policy_loss_total` and `kl_term` are now logged as
new keys — new rather than a redefinition of `policy_loss`, so every series recorded to date stays
comparable and the two arms in flight are undisturbed. Items 2 and 3 remain the owner's call.

---

## The search targets do not flatten (2026-09-18) — the feedback loop is dead

The decline's signature is the policy's entropy **rising** while strength falls. Two readings, with
opposite fixes: either the searcher's distribution flattens and the CE term is actively teaching
that, or the policy flattens on its own and the targets are bystanders.

`tools/scripts/search_target_entropy.py` over `E3-35-28`'s snapshots, spanning the 73.6% era and
the 58.2% era — 48 positions held **fixed** across every snapshot, so this measures the network's
effect on search sharpness and not a drift in which positions the arm reaches, 64 simulations:

| steps | mean target H | median | top-1 share | policy entropy |
|---:|---:|---:|---:|---:|
| 28.31M | 0.715 | 0.578 | 0.733 | 0.774 |
| 28.77M | 0.725 | 0.636 | 0.733 | — |
| 29.29M | 0.742 | 0.748 | 0.725 | — |
| 29.75M | 0.713 | 0.531 | 0.742 | — |
| 30.28M | **0.693** | 0.578 | 0.727 | **0.947** |

**Flat.** Target entropy sits in 0.693–0.742 with no trend and the top-1 share is pinned at
0.725–0.742, while the policy's own entropy climbs 0.774 → 0.947 over the same span.

So the loop is dead as an explanation. The searcher keeps producing targets just as decisive as
before; the policy drifts away from them anyway. **The CE term is failing to prevent the decline,
not causing it** — and removing or reweighting it, which was the obvious response to the loop
hypothesis, would not address this.

### What that leaves

If the targets stay sharp and the policy flattens, something is out-competing the CE term in the
update, and the logged gradient share says it is being out-competed badly: `search_ce_grad_frac`
falls from **0.21–0.32** before the decline to **0.03–0.17** after, while `kl_div` over the same
span rises from ~0.07–0.13 to spikes of 2.4, 3.6 and 7.5.

That is the same shape as this entry's main finding on `E3-31-28`, one order of magnitude smaller:
the KL regulariser growing until it crowds the policy-improvement terms out of the update, and the
policy consequently drifting toward a stale reference rather than toward its targets.

**Stated as the hypothesis it is.** The per-iteration coupling is *not* clean — 29.82M pairs
`kl_div` 2.36 with `grad_frac` 0.077, but 30.21M pairs `kl_div` 3.62 with `grad_frac` 0.171. What
is clean is the level shift in both series across the decline. Establishing the mechanism needs
the arm suggested above: KL restricted to learner rows, or η annealed, against this arm as control.
`policy_loss_total` and `kl_term` are now logged, so the next run can be read on the objective
that is actually optimised rather than on the surrogate alone.
