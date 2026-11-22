# Two throughput levers: the rollout in two halves, and the update in bf16 (2026-11-20/21)

**Conclusion.**

* **P51, the rollout in two halves of the environments, is slower and was reverted.** On R155's recipe it cost
  84.1k -> 71.6k steps/s with both extra critics and 105.0k -> 95.0k without. The rollout is CPU-bound: the
  network pass and sampling are 0.33 ms of a ~3 ms rollout step, so the overlap can hide at most ~0.3 ms, while
  the second env-step call costs more than that.
* **P52, the update's forwards under bf16 autocast (`--update-bf16`), is fast and changes the gradient.** Alone it
  is worth about as much as compiling the update (+16% against +13% from scratch), but on a mature network's
  rollout rows it moves the taken action's log-prob by 0.025 on average and puts 4.3% of rows outside PPO's 0.1
  clip band before any learning, with the gradient 31% off eager (cosine 0.956). Heads kept in fp32
  (`--bf16-fp32-heads`, P52b) cut the shift to 0.0115 and the out-of-band rows to 0.95%; the gradient is still
  21% off (cosine 0.978). Compile, by contrast, is 0.13% off (cosine 0.999999).
* Both flags stay opt-in. bf16 was then tried in training ([P53](P53_compile_bf16.md), with compile) and alone
  ([P54](P54_compile_alone.md), stopped early); it is postponed, compile is adopted.

**Instrument.** Timings in place on R155's recipe (`E8.1-A8-R132-S45@11600M+R139@12400M+R155`) from its
14,000M state, 1,024 envs x 128, 12M steps, steady medians; the gradient comparison on rollout rows of the same
mature network. Numbers from the commits that built each flag (no report file): `933d353` (P51, reverted by
`0959727`), `1874ce7` (P52), `2c333b4` (P52b), `1a6ebc5` (the from-scratch probes).

## P51 -- the rollout in two halves (`--rollout-halves`, reverted)

The GPU's network pass and sampling for one half of the envs were queued while the engine stepped the other
(`step_flat_some`, `env.step(only=)`); same envs, buffer, minibatch and update. Tests held (two half steps equal
one full step over 1,500 steps; stored log-probs and values the network's own).

| R155's recipe, 12M steps | plain | `--rollout-halves` |
|:---|---:|---:|
| both extra critics | 84.1k steps/s | 71.6k |
| neither | 105.0k | 95.0k |

Entropy and explained variance level. The second env-step call per step raised the wrapper's time from 19.3 to
31.4 s, and two half passes cost 0.5 ms against one full pass's 0.33 ms. Reverted the same day.

## P52 -- the update in bf16 (`--update-bf16`, opt-in)

The learner's per-minibatch forward, the reference's precomputed logits and the extra critics' value rows run
under `torch.autocast(bfloat16)`; outputs are cast back to fp32 before any loss; the rollout (CUDA graphs, stored
log-probs) stays fp32.

| throughput (steps/s) | plain | `--compile-update max-autotune` | `--update-bf16` | both |
|:---|---:|---:|---:|---:|
| R155's recipe from 14,000M, both extra critics | 92.3k | 113.3k | 114.1k | 134.4k |
| R132 from scratch, 25M steps | 114.1k | 129.1k (+13%) | 132.0k (+16%) | 166.7k |

Entropy, explained variance and value losses were level across the four in the 12M-step timing; that is not a
strength test.

**What bf16 does to the update** (rollout rows of a mature network, before any learning step):

| | taken action's log-prob shift (mean) | rows outside the 0.1 clip band | gradient off eager | cosine to eager |
|:---|---:|---:|---:|---:|
| `--update-bf16` | 0.025 | 4.3% | 31% | 0.956 |
| `--update-bf16 --bf16-fp32-heads` (P52b) | 0.0115 | 0.95% | 21% | 0.978 |
| `--compile-update max-autotune` | -- | -- | 0.13% | 0.999999 |

The compiled update's differences from eager are unbiased across batches: compile changes the sample, not the
direction. bf16 rounds the logits PPO's ratio reads; fp32 heads halve the log-prob shift and cut the rows pushed
out of the clip band by about four, but leave most of the gradient error.

**Caveat** (audit M-7, [`audit_2026-11-22_verification.md`](audit_2026-11-22_verification.md)): under
`--update-bf16` the old log-probs are fp32 and the new ones bf16, which may explain part of P53's cost; it is to
be fixed before bf16 is retried. `--bf16-fp32-heads` has not been trained with.
