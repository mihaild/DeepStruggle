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
| [`log/`](log/README.md) | **append-only** — never rewritten | what did we try, and why do we believe it? |
| [`findings/`](findings/README.md) | **rewritten in place** — always current | what is true now? |
| [`method/`](method/README.md) | **living reference** | how do we measure this? |
| [`plans/`](plans/README.md) | forward-looking; becomes a log once run | what should we try next? |
| [`papers/`](papers/README.md) | static | what does the literature say? |
| [`archive/`](archive/README.md) | frozen | superseded material, kept for history |

Two files sit at the top level because they are indexes into everything below, and an index in a
subdirectory is one nobody opens:

| file | |
|:---|:---|
| [`runs.md`](runs.md) | **the arm registry** — one row per arm: what it varied, seeds, budgets, checkpoint directory, and where its result is written up (or that it is not) |
| [`checkpoints.md`](checkpoints.md) | **the checkpoint catalogue** — which model to load and what it is worth: every rated checkpoint with its Elo and the field that measured it, the pooled-vs-unpooled verdict at 160M, and the directory-naming traps |
| [`questions.md`](questions.md) | **the question index** — one row per question: which arms bear on it and what the verdict is. Start here when you know what you want to know but not which arm measured it |

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
[`plans/`](plans/README.md) already numbers. Where a programme has a plan, the log carries its `PN` prefix,
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

## findings/ — short, current, cited, and split by what can invalidate it

Two kinds of knowledge live here and they answer to different things, so they are filed apart.
The index is [`findings/README.md`](findings/README.md), which also lists the cases where the
boundary is not clean.

| directory | | invalidated by |
|:---|:---|:---|
| [`findings/engine/`](findings/engine/README.md) | **(A)** what the simulator does — rules, mask semantics, revision boundaries, the instruments | a defect found, a rules fix, a probe caught lying |
| [`findings/training/`](findings/training/README.md) | **(B)** what a training choice is worth — architecture, reward, pooling, value targets, seed variance | a better-powered replication, more seeds |

| file | |
|:---|:---|
| [`engine/engine_revisions.md`](findings/engine/engine_revisions.md) | which game each arm was trained on: E1/E2/E3, what each boundary changed, what it invalidated |
| [`engine/engine_change_decision_stream.md`](findings/engine/engine_change_decision_stream.md) | P14 changed the legality code and moved no decision in 385,812 steps; how to establish that for the next change |
| [`training/architecture.md`](findings/training/architecture.md) | what the network is now, and what each change was worth |
| [`training/pooling.md`](findings/training/pooling.md) | pooled vs non-pooled opponents: every comparison, the current verdict, and the four unrelated things this project calls "pool" |
| [`training/seed_variance.md`](findings/training/seed_variance.md) | the error bars a comparison has to clear, and the two results they have withdrawn |
| [`training/defcon_blunders.md`](findings/training/defcon_blunders.md) | the critic cannot see a provoked DEFCON-1 at any node; what windowing does and what it costs |

**The split rests on an assumption, and the assumption is written down.** The owner's: *engine
changes make old arms incomparable with new, but after the starred events change they do not
affect the relative effect of training approaches.* What that licenses and what it does not is
[`method/what_survives_an_engine_change.md`](method/what_survives_an_engine_change.md). Read it
before carrying any number across an engine revision.

## method/ — how to measure

| file | |
|:---|:---|
| [`measurement_pitfalls.md`](method/measurement_pitfalls.md) | **read this before trusting a number.** The checklist, distilled from every measurement bug we have made |
| [`running_experiments.md`](method/running_experiments.md) | how to run an arm and how to rate one |
| [`measurement_tiers.md`](method/measurement_tiers.md) | which probe runs during training, per snapshot, and once per arm |
| [`run_nomenclature.md`](method/run_nomenclature.md) | `<engine>-<attempt>-<seed>-<steps>`: what each field means and when the engine letter bumps. The arms themselves are [`runs.md`](runs.md) |
| [`what_survives_an_engine_change.md`](method/what_survives_an_engine_change.md) | **the assumption behind the (A)/(B) split** — what transfers across an engine revision, what does not, and the evidence for and against |
| [`bookkeeping.md`](method/bookkeeping.md) | **how to record an experiment so it can be found again** — what goes in `runs.md`, `questions.md`, the log and `findings/`, and in what order |
| [`human_play.md`](method/human_play.md) | the human corpus: game length, side asymmetry, the agreement measure |
| [`references.md`](method/references.md) | the literature behind the plan ordering |

## Two habits that this record exists to enforce

**Register the prediction before the run.** Write down what the probes and the Elo should do, and
which instrument will produce the number, in `plans/` or in the arm's note in
[`runs.md`](runs.md), *before* launching. It costs a minute and it is the difference between a
result and a rationalisation. The most useful entries in `log/P9_architecture.md` are the two
where the prediction failed. The full procedure is
[`method/bookkeeping.md`](method/bookkeeping.md).

**Index by question, not only by arm.** An arm is found by someone who already knows its name.
Every result also gets a row in [`questions.md`](questions.md), because "what did we learn about
pooled versus non-pooled?" is the form the question is actually asked in, and answering it used
to take three files and a `git log`.

**A probe only tests understanding if its answer key is absent from the observation.** This
observation is unusually rich in precomputed per-country facts, and more than one probe here has
measured a copied input before anyone noticed. `method/measurement_pitfalls.md` lists the rest.

## Size

Split a file at about 400 lines. A document nobody can read in one sitting stops being read, and
then the claims in it stop being checked — which is exactly how a 1,900-line file came to carry
stale test counts, a stub package described as a single file, and a module that no longer existed.
