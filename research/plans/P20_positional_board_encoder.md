# P20 — the pooled board path is unmotivated without a graph

**Status: proposed, not launched.** Raised by the owner, 2026-09-19: *"Without graph, shouldn't we
just feed raw country features into trunk?"*

## The claim

At `graph_layers = 0` the board reaches the trunk like this:

```
board (B, 84, 26) ++ identity  →  board_fc, SHARED across countries  →  h_board (B, 84, 64)
                                  →  mean + max over the 84  →  board_proj  →  e_board (256)
```

Mean and max are **symmetric**, so this discards which country contributed what. That is the
right inductive bias for an *exchangeable, variable-sized* set. The board is neither: it is a
fixed, known, ordered list of 84 countries, and position is free information that the observation
already encodes perfectly. Pooling throws it away, and `country_identity` then spends 1,344
parameters buying part of it back
([`../findings/training/country_identity_without_graph_conv.md`](../findings/training/country_identity_without_graph_conv.md)).

With a graph there is at least a reason to work in shared-weight token space: `gconv` needs
per-node tokens to pass messages over `norm_adj`. **With `graph_layers = 0` that reason is gone**
and the shared-then-pool structure is left doing the harm without the benefit.

## The record already says pooling is the bottleneck

| read from | fraction of the country-influence gap closed |
|:---|---:|
| pre-pooling token | **~89–94%** |
| the 512-float trunk | **~14%** |

And the obvious repair was tried *in a form that could not work*:

> **"Can an attention read-out fix the pooled trunk?" — E3-13 — settled negative, −14/−23 Elo; a
> single-query read-out is still pooling.** ([`../questions.md`](../questions.md))

That is why `attn_readout` is 0 in every arm. It tested a *different* pooling, not the absence of
pooling. **A positional (non-pooled) board path into the trunk has never been tried at all** — not
in V2, and not in V1, which pools the same way.

## What the per-entity heads do and do not cover

`pe_country` routes around the pool, but only for actions that *name a country*. Everything else
still reads the ~14% trunk:

* the resolution node, op mode, branch, confirm, DEFCON and region actions — the `gap_mid` and
  `gap_end` blocks are dense-only by construction;
* **the value head, which has no per-entity route at all.** The critic sees the board only through
  the pooled summary.

That last point deserves an experiment of its own: the standing open finding is that *"the critic
does not see a provoked DEFCON-1 coming at any node, and no architecture change so far has moved
that"* ([`../archive/E3_ladder/findings/architecture.md`](../archive/E3_ladder/findings/architecture.md)).
A critic reading a summary that holds ~14% of per-country influence is a candidate explanation
that has not been ruled out.

## The architecture already flattens where it decided order matters

The history encoder is `Conv1d → Flatten → Linear(16*32, 128)` — 65,664 parameters, position
preserved. The board is the inconsistent case, not the flat proposal.

## The proposal

**Add** a positional board embedding alongside the pooled one, concatenated into `fusion_in`.
Add, not replace: the shared token path still feeds `cross_attn` and `pe_country`, and the pooled
summary is cheap and may still carry useful aggregate signal.

| variant | added parameters | vs a 3.1M model |
|:---|---:|---:|
| flatten raw slots, `Linear(84*26, 256)` | 559,360 | +18% |
| flatten tokens at width 16 | 344,320 | +11% |
| flatten tokens at width 32 | 688,384 | +22% |
| flatten tokens at width 64 | 1,376,512 | +44% |

The current whole board path is 36–39k parameters, so this is a real increase — but the model is
3.1M against architectures that solve comparable games with far more, and
[`P6`](P6_attention_backbone.md) notes Ataraxos and AlphaStar both encode the board
positionally with a transformer for exactly this reason.

**Start with flatten-tokens-at-16 (+11%).** It keeps the per-country nonlinearity — a flat map of
the raw slots must relearn "high influence in a battleground is good" 84 times, where the shared
encoder learns it once — while restoring position.

## Honest counter-arguments

* **Capacity is reportedly not the bottleneck on v2** ([`P6`](P6_attention_backbone.md)). True, and
  this proposal is not about capacity: a pooled summary destroys information that no amount of
  downstream capacity can recover. But it does mean "more parameters helped" would be the wrong
  reading of a positive result, and the control has to be a parameter-matched arm, not the
  current baseline.
* **Weight sharing is a real generalisation benefit** and flattening gives it up for the flat
  path. Adding rather than replacing keeps both.
* **One seed decides nothing here.** Seed spread is ~95 Elo, larger than most architecture effects
  in this record, and two of this project's architecture conclusions have already been withdrawn
  for exactly that reason.

## Ordering

Behind [P19](P19_architecture_ab.md), which asks whether the late-E3 architecture is worth carrying
at all on this engine. If P19 says the bundle is not stronger, P20 becomes *more* interesting
rather than less: it is a different explanation for why the board does not reach the decision.
