# Method — how to run an experiment and how to trust the number

`log/` and `findings/` hold what was learned. This directory holds how to learn it without fooling
yourself, which is kept separate because it is read at a different moment: before a run, and again
when a result looks too good.

| file | what it is for |
|:---|:---|
| [`bookkeeping.md`](bookkeeping.md) | how an experiment must be recorded to stay findable — the rule that an experiment is not recorded until the *question* it answers has a row |
| [`run_nomenclature.md`](run_nomenclature.md) | the naming scheme, `<engine>-<attempt>-<seed>-<steps>`, and when the engine letter bumps |
| [`running_experiments.md`](running_experiments.md) | launching, resuming and monitoring a run |
| [`measurement_pitfalls.md`](measurement_pitfalls.md) | the checklist to work through before trusting a number |
| [`measurement_tiers.md`](measurement_tiers.md) | how much evidence a claim needs before it counts as settled |
| [`what_survives_an_engine_change.md`](what_survives_an_engine_change.md) | the assumption that relative training results cross an engine revision while absolute ratings do not — stated as an assumption, with its evidence and its gaps |
| [`human_play.md`](human_play.md) | playing the workbench by hand, and what to look for |
| [`references.md`](references.md) | external sources, and the rules texts the engine implements |

The companion to this directory is [`../log/measurement_bugs.md`](../log/measurement_bugs.md):
every instrument that reported confident nonsense, written up in full. The checklist here is the
short form; that file is the evidence for why each item is on it.
