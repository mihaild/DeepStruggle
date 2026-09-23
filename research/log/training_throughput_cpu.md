# Training throughput: the CPU goes to PyTorch's thread pool, the step rate to the main thread

**2026-09-23. A diagnosis only; nothing has been changed yet.** The owner noticed that `tools/train.py`
uses far more CPU than an engine running at more than 1M steps/s on one core can explain.

## What was measured

On a live run, E4-39-03 (λ 0.99 M2d, 512 envs, one of two runs on the machine):

* **Throughput.** Each process makes ~19–20k env steps/s. That is ~40 vector steps of all 512 envs
  per second, or ~25 ms per step. At 1M steps/s the engine accounts for about 0.5 ms of that.
* **CPU.** 24,540 CPU-seconds in 3,211 s of wall time, an average of 7.6 cores:
  * the main thread runs at 75–90% of one core;
  * **16 worker threads each run at ~35–43%, continuously,** with near-identical accumulated times
    (~24 CPU-minutes each after ~50 minutes). Together that is about 6.4 cores.
* **Whose threads.** The 16 workers are PyTorch's intra-op OpenMP pool (`at::get_num_threads() = 16`,
  `OMP_NUM_THREADS` unset). The engine and the batch runner are single-threaded: `engine/` and
  `bindings/` contain no OpenMP and no `std::thread`.
* **Inference is on the GPU.**
  * Each process holds ~2.9 GB of GPU memory.
  * Pool opponents and eval agents are moved to the training device (`generic_trainer.py:1757`,
    `:2124`, `:932`, `:1060`) and called on GPU tensors (`nash_pg.py:504`).
  * GPU utilisation is ~24%.
* **Not a regression in cost per step.** The machine's total step rate is about constant however
  many runs share it: E4-27-03 alone did 38.3k/s (the code before this session), two E4-39 runs do
  ~38.4k/s together, and five bench runs did ~34.9k/s. The share a single process gets moves with
  how many processes there are.

## Reading

* **The pool is mostly spinning.** The per-step CPU tensor work (converting observations, masks and
  rewards before the copy to the GPU) is small. Each operation still enters an OpenMP parallel
  region, and libgomp workers spin-wait between regions. That is ~6 cores per process of
  near-useless CPU.
* **The step rate is bound by the main thread,** not the engine and not the GPU. The ~25 ms per
  vector step goes to:
  * per-step Python loops over the 512 envs;
  * building the observations (512 × 3,824 floats) and copying them to the GPU, ~7.8 MB per step;
  * the pool opponent's forward pass;
  * Python overhead around the GPU calls.

## Proposed, not done

1. Cap PyTorch's CPU threads in the trainer (e.g. `torch.set_num_threads(2)`), then compare steps/s
   and CPU against an uncapped run.
2. Profile the main thread on a short run (cProfile, or py-spy if it is added to the environment).
   Then remove the per-env Python loops and the host-side copies. The engine could supply more than
   10× the current rate.
