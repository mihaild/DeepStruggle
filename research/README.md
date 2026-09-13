# research/

The project's research record. **None of this is published** — `tools/scripts/publish_snapshot.sh`
strips the whole directory, and `check_public_hygiene.sh` fails the publish if anything in the
public tree so much as cites it. That is deliberate: the public repository is code, and this is
the reasoning behind the code.

## The one rule: file by update discipline

Documents here differ in *when you may rewrite a line*, and that is what decides where a thing
lives. Topic is a bad filing dimension — topics drift and a document ends up filed twice.

| directory | discipline | the question it answers |
|:---|:---|:---|
| [`log/`](log/) | **append-only** — never rewritten | what did we try, and why do we believe it? |
| [`findings/`](findings/) | **rewritten in place** — always current | what is true now? |
| [`method/`](method/) | **living reference** | how do we measure this? |
| [`plans/`](plans/) | forward-looking; becomes a log once run | what should we try next? |
| [`papers/`](papers/) | static | what does the literature say? |
| [`archive/`](archive/) | frozen | superseded material, kept for history |

**The test, when you are unsure:** a statement in here turns out to be wrong. Do you *edit* it, or
*correct it underneath*?

* Correct it underneath → `log/`. Dead ends, abandoned arms, retracted claims and wrong
  predictions all stay, permanently. That is the only thing a log is for: without them you cannot
  answer "did we already try that, and why did we stop?"
* Edit it → `findings/` or `method/`.

Two rules keep the first two from collapsing into one another:

1. **`findings/` states conclusions and links to the log. It never re-tells the experiment.** If a
   findings file starts accumulating narrative, it is drifting into log territory and should be
   cut back.
2. **`log/` is never tidied.** Editing a log entry to match current belief destroys the only
   property it has.

## log/ — one file per programme

Not per engine era, and not per arm. An era would give one enormous file again as soon as the
engine settles; an arm is too granular, and every interesting comparison spans several. A
*programme* — one investigated question over several arms — is naturally bounded and is what
[`plans/`](plans/) already numbers. Where a programme has a plan, the log carries its `PN` prefix,
so `plans/PN` proposes and `log/PN` records. The engine era lives inside each entry, in the run's
short name (`E3-15-21-80M`), which means a programme that straddles an engine bump stays in one
file.

| file | programme |
|:---|:---|
| [`early_training_signal.md`](log/early_training_signal.md) | blunder window, decisive-transition priority, start-pool sampling |
| [`agent_deficiencies_and_decisiveness.md`](log/agent_deficiencies_and_decisiveness.md) | what the control cannot do; length-scaled reward; forced wins |
| [`engine_reanchor_and_human_control.md`](log/engine_reanchor_and_human_control.md) | re-establishing the ladder after the engine fixes; the human corpus as a strong-player control |
| [`P7_human_bc_warmup.md`](log/P7_human_bc_warmup.md) | human data as initialisation |
| [`P7_human_injection_and_its_cost.md`](log/P7_human_injection_and_its_cost.md) | human data as continuous injection, and the ablation that killed it |
| [`P7_replayer_conversion.md`](log/P7_replayer_conversion.md) | converting the human game logs to engine decisions |
| [`critic_positional_value.md`](log/critic_positional_value.md) | board and hand perturbation probes on the critic |
| [`critic_vs_policy_160M.md`](log/critic_vs_policy_160M.md) | counterfactual rollouts; why the run will not invest positionally |
| [`observation_layout.md`](log/observation_layout.md) | the observation layouts and what each was worth |
| [`corrected_engine_arms_H_I.md`](log/corrected_engine_arms_H_I.md) | the corrected-engine ladder |
| [`P9_architecture.md`](log/P9_architecture.md) | identity embeddings → self-transform → per-entity heads → removing the map graph |
| [`measurement_bugs.md`](log/measurement_bugs.md) | every instrument that reported confident nonsense, in full |
| [`variance_and_noise.md`](log/variance_and_noise.md) | run-to-run variance, and how much of a rating is just where you stopped |

## findings/ — short, current, cited

| file | |
|:---|:---|
| [`architecture.md`](findings/architecture.md) | what the network is now, and what each change was worth |

## method/ — how to measure

| file | |
|:---|:---|
| [`measurement_pitfalls.md`](method/measurement_pitfalls.md) | **read this before trusting a number.** The checklist, distilled from every measurement bug we have made |
| [`running_experiments.md`](method/running_experiments.md) | how to run an arm and how to rate one |
| [`measurement_tiers.md`](method/measurement_tiers.md) | which probe runs during training, per snapshot, and once per arm |
| [`run_nomenclature.md`](method/run_nomenclature.md) | `<engine>-<attempt>-<seed>-<steps>`, and the table of every arm |
| [`human_play.md`](method/human_play.md) | the human corpus: game length, side asymmetry, the agreement measure |
| [`references.md`](method/references.md) | the literature behind the plan ordering |

## Two habits that this record exists to enforce

**Register the prediction before the run.** Write down what the probes and the Elo should do, in
`plans/` or `run_nomenclature.md`, *before* launching. It costs a minute and it is the difference
between a result and a rationalisation. The most useful entries in `log/P9_architecture.md` are
the two where the prediction failed.

**A probe only tests understanding if its answer key is absent from the observation.** This
observation is unusually rich in precomputed per-country facts, and more than one probe here has
measured a copied input before anyone noticed. `method/measurement_pitfalls.md` lists the rest.

## Size

Split a file at about 400 lines. A document nobody can read in one sitting stops being read, and
then the claims in it stop being checked — which is exactly how a 1,900-line file came to carry
stale test counts, a stub package described as a single file, and a module that no longer existed.
