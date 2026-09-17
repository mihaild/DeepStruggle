# P16 — Policy probabilities and critic values on the replay

**Status:** **done** — phases 1–3 landed; phase 4 (live web games) dropped by the owner.
**Gate:** none — CPU/engineering, Tier 2. Independent of the P15 arms, and useful while they run.
**Needs approval:** none for the engine, the bindings or the observation. It adds *optional* replay
fields, one module, one CLI and a workbench panel. The owner's decisions are recorded in
[Decide before running](#decide-before-running).

> **What landed.** `ai/eval/policy_readout.py`; the trace written by `tools/lib/self_play.py`
> (on by default) and `tools/play_match.py --trace`; `tools/annotate_replay.py` for the post-hoc
> path; the value ribbon, probability chips and readout panel in the workbench. Measured on a
> 189-step CPU game: the file grows **6.21 → 6.48 MB (+4.3%)** and generation takes
> **0.55 s → 1.55 s** with `--trace-critic-every step`, which is the honest price of two critic
> forwards on steps that previously ran no network at all; `decision` removes most of it. The
> action sequence is identical with the trace on and off under a fixed torch seed.

## Goal

Every step of a generated replay carries, next to the action that was played:

1. **the distribution the policy assigned over the legal actions at that node** — what it believed,
   not only what it did; and
2. **the critic's reading of the position** at every recorded step, both perspectives.

And the workbench shows both while scrubbing: a value ribbon under the timeline, a probability chip
on each log row, and a per-step panel listing the top legal actions with their probabilities.

## Why

* The blunder counter (`ai/eval/blunders.py`, wired into `tools/lib/self_play.py`) says a mistake
  happened. It cannot say whether the policy *nearly avoided it*. `p_chosen = 0.02` on a blunder the
  mode would have dodged is a sampling-temperature problem; `p_chosen = 0.81` is a policy problem.
  Those have different fixes and today they look identical in a replay.
* The critic trace over a real game is the missing half of the stall diagnosis. "The critic
  degenerates toward the base rate" is currently a training-metric claim
  ([`README.md`](README.md), the binding constraint). A per-step `v_win` curve over a played game
  makes it a property of an artefact you can point at, and localises *where* on the board the
  degeneracy shows.
* The machinery for the critic half already exists but only post-hoc and only offline:
  `ai/eval/replay_critic.py` re-drives a replay from its seed and evaluates both heads from both
  perspectives, with `verify()` guarding the reconstruction. It was used once, by hand, for
  [`../log/europe_control_and_held_scoring.md`](../log/europe_control_and_held_scoring.md). Making
  the trace part of the replay makes that reading cheap, repeatable and visible.
* It costs nothing to compute at generation time. `sample_action` already runs the forward pass,
  already has the masked logits and already has both value heads — and throws all but the sampled
  index away.

## Two sources, and which one is authoritative

**Inline (the primary path).** The numbers are recorded by the generator that played the game, so
they are the exact model, the exact legal mask, and the exact temperature that produced the move.
Nothing has to be reconstructed and nothing can drift.

**Annotated (the second path, `tools/annotate_replay.py`).** Re-drive a replay from `(seed, actions)`
and ask *a* checkpoint what it thinks. Needed for two cases inline cannot serve: replays already on
disk, and asking a *different* checkpoint about a game it did not play (what does the 320M snapshot
think of the 80M snapshot's turn 4?) — a genuinely different and useful question, so the mode is
recorded in the file rather than left implicit.

The annotator's risk is the one `CLAUDE.md` invariant 13 names: a rebuilt engine can change the
decision stream with no Python change, and a `(seed, actions)` reconstruction then truncates
silently. Two guards, both existing: `verify()` at **every** step (refuse, do not warn), and the
engine fingerprint from `tools/lib/engine_fingerprint.py` written into the trace metadata, so a
trace can be told from the build that produced it.

## Design decisions

1. **Probabilities are the model's own distribution, at T = 1** — `softmax(masked_logits)`, not the
   tempered distribution that was sampled from. Self-play runs at T = 0.3, and "what did the policy
   believe" and "how likely was this sample" are different questions. Both are recorded:
   `p_chosen` (T = 1) and `p_chosen_sampled` (at the sampling temperature). `temperature` is on the
   step, because it may vary by caller.
2. **One forward per decision node.** The readout reuses the pass `sample_action` already makes; it
   does not add a second.
3. **The critic is read at every recorded step, from both perspectives,** on the state *after* the
   action — the state the replay snapshot shows, so the number the UI displays belongs to the board
   the UI displays. Both perspectives because `v_win_us + v_win_ussr` is pure model error for a
   zero-sum critic, it is already an established instrument in `replay_critic.evaluate`, and the
   second forward is cheap. Reuse `evaluate()` rather than writing a second copy of it.
4. **The pre-action critic value goes on the policy block** (`v_win`, `v_vp` from the acting
   player's perspective), because it comes free from the same forward and it is what makes
   "the critic thought +0.3, then the move it chose left it at −0.1" readable on one row.
5. **Bounded size, and no `None`-shaped surprises.** The full 212-vector is not stored by default;
   the legal actions sorted by descending probability, capped at `top_k`, floored at `p_floor`, plus
   the residual mass as `p_tail`. The chosen action is always present in `top` even if it falls
   below the floor — otherwise the interesting case (a 0.001 blunder) is the one that gets dropped.
6. **Optional everywhere.** Every new field sits in a `total=False` TypedDict and every reader
   treats it as absent-by-default: heuristic bots have no distribution, human games have no model,
   and old replays have no trace. A UI that hides its panel when the key is missing is the whole
   compatibility story.
7. **It does not go out over the wire during a live game.** A policy distribution over the bot's
   legal actions is a read on the bot's hand, which is exactly what `observation_b64`'s per-role
   emission exists to withhold (`web/server/session.py::_observation_for`). The trace is written
   into the saved replay, which is a post-game artefact, and is never put in a `STATE_UPDATE`.

## The data

New in `web/server/replay_types.py` (and added to `REPLAY_TYPES` in
`tests/bindings/test_replay_schema_golden.py`, then the golden file regenerated):

```python
class PolicyChoiceDict(TypedDict, total=False):
    idx: int       # flat action index, 0..211
    p: float       # probability under the model's own (T=1) distribution
    name: str      # ActionEncoder.get_action_name at the PRE-action state

class ReplayPolicyDict(TypedDict, total=False):
    source: str               # "policy" | "forced" | "scripted" | "bot" | "human"
    temperature: float
    n_legal: int
    chosen_idx: int
    p_chosen: float           # T = 1
    p_chosen_sampled: float   # at `temperature`
    argmax_idx: int
    p_max: float
    entropy: float            # nats, T = 1
    top: List[PolicyChoiceDict]
    p_tail: float             # mass not in `top`
    v_win: float              # acting player's perspective, PRE-action state
    v_vp: float

class ReplayCriticDict(TypedDict, total=False):
    v_win_us: float
    v_win_ussr: float
    v_vp_us: float
    v_vp_ussr: float
    win_residual: float       # v_win_us + v_win_ussr -- zero for a consistent critic
    vp_residual: float
    at: str                   # "after" -- which state these were read on

class ReplayTraceMetaDict(TypedDict, total=False):
    mode: str                 # "inline" | "annotated"
    model_us: str             # checkpoint path or agent label
    model_ussr: str
    checkpoint_sha256_12: str # of the annotating checkpoint, or of both when they differ
    arch: str                 # "ColdWarNetV2" etc, from the weights
    temperature: float
    top_k: int
    p_floor: float
    engine_fingerprint: str   # tools.lib.engine_fingerprint.fingerprint()
```

`ReplayStepDict` gains `policy: ReplayPolicyDict` and `critic: ReplayCriticDict`;
`ReplayMetadataDict` gains `trace: ReplayTraceMetaDict`.

**Size.** A current replay is ~33 KB *per step* — `s160_1.tslog.json` is 8.8 MB over 268 steps,
because every step carries a whole `state_snapshot`. A `top_k = 12` trace is ~400–600 bytes per
step: **under 2%**. There is no size argument for storing less, and there is one for not storing the
full 212-vector (mostly masked-away zeros).

## Where the code goes

1. **`ai/eval/policy_readout.py` (new, ~70 lines).**
   `read_policy(model, obs_t, mask_t, temperature, deterministic) -> tuple[int, ReplayPolicyDict]`.
   One `model.forward(obs, mask)` — which returns `(masked_logits, v_win, v_vp)` on **both** V1 and
   V2, so no per-architecture branch — then the T = 1 softmax for the reported probabilities and the
   tempered `Categorical` for the sample.
   It does **not** replace `sample_action`, which is on the training hot path and must not be
   touched for a diagnostic. It duplicates its three lines of sampling, and a **seeded equivalence
   test** (same `torch.manual_seed`, same obs/mask/temperature → same action index, same log-prob)
   pins the duplicate so it cannot drift. Sits next to `replay_critic.py` because it is the same
   kind of thing: reading numbers off a model for a recorded game.
2. **`tools/lib/self_play.py`.** `PolicySource.choose` keeps the readout for the node it just
   decided; `_record` attaches it as the step's `policy` and calls
   `replay_critic.evaluate(model, after)` for `critic`. Forced steps (played by
   `SettlePolicy.RECORD_FORCED`, which is most of them) get `source: "forced"`, `n_legal: 1`,
   `p_chosen: 1.0` and no forward; scripted-opening steps get `source: "scripted"`. The critic is
   still read on those steps by default — a value curve with holes in it is much harder to read than
   one without, and the cost is one forward.
   New flags on the function and on `tools/play_match.py`: `--trace/--no-trace` (default on for
   `generate_self_play_replay`), `--trace-top-k`, `--trace-critic-every {step,decision}`.
3. **`tools/play_match.py`.** Same recorder change. The readout has to come from the agent, so
   `NeuralAgent` grows a `last_readout: Optional[ReplayPolicyDict]` set in `select_action`; every
   other agent leaves it `None` and the step simply has no `policy` block. This is the path that
   produces a trace for a *heuristic vs neural* game, which is where the interesting disagreements
   are.
4. **`tools/annotate_replay.py` (new CLI).**
   `--replay X.tslog.json --model ckpt.pt [--out Y.tslog.json] [--print]`. Uses
   `replay_critic.load_actions` / `replay_to`, calls `verify()` at every step and **aborts** on the
   first mismatch naming the step (never "continues with a warning" — a drifted reconstruction
   still returns numbers and they still look like results). Writes `trace.mode = "annotated"`.
   `--print` gives the terminal table: step, turn/AR, player, description, `p_chosen`, `p_max`,
   entropy, `v_win_us`, `Δv_win`. Documented in `tools/README.md`.
5. **Web UI (`web/ui/src/`).**
   * `replay_controls.ts`: `ReplayStep` gains the two optional fields.
   * **Value ribbon** — a new ~40px inline-SVG strip directly under `#rep-timeline-slider`:
     `v_win_us` across all steps, zero line, the current step marked, VP-change steps ticked,
     click-to-seek (the slider and the ribbon share one index). This is the "critic prediction at
     every moment" view; it is also where a base-rate-collapsed critic becomes obvious at a glance.
   * **Log rows** (`main.ts::renderLogStream`, the replay branch): a `p=0.42` chip coloured by
     probability, plus a `Δv` badge when the critic moved more than a threshold across that step.
     Both hidden when the step has no trace.
   * **Probabilities on the choices themselves** (landed 2026-09-17, replacing the bar list):
     each card in hand, each HUD mode/target button and each country on the map carries the
     probability of choosing it, with the move that was played outlined. The board on screen is
     the position *after* step N, so it is the node step N+1 was decided at — the numbers come
     from step N+1's policy, while the critic numbers describe the position itself and come from
     step N. Pairing them the other way would label a hand that no longer holds the card played.
     Flat indices are mapped back to cards/countries through `GET /api/metadata/action_space`
     rather than a duplicated offset table; `tests/web/test_action_space_metadata.py` replays a
     real game through that mapping.
   * **Readout panel** — a card in the right rail beneath `#decision-panel`: the critic's
     prediction for the position on screen, **US and USSR, `v_win` and `v_vp`, to five decimals**,
     with the zero-sum residual under them, then a one-line summary of the decision whose
     probabilities are on the board.
     Inline SVG/CSS only, no new dependency — the repo already draws its map by hand.
6. **Deferred: `web/server/session.py` + `web/bot_client.py` (Phase 4).** A live web game's bot
   moves arrive over the websocket, so the server cannot compute the acting bot's distribution. The
   bot client would have to send its readout alongside the action and the session store it on the
   step — writing it to the replay only, never into a broadcast (decision 7). Worth doing only if
   human-vs-bot replays are something we want to read this way; it is the one piece that touches the
   live protocol, so it is last and separable.

## Procedure

| phase | what | done when |
|:---|:---|:---|
| 1 | schema + `policy_readout.py` + `self_play.py` + `play_match.py` | a generated replay carries a trace; reproduction tests unchanged |
| 2 | `tools/annotate_replay.py` + tests | an existing replay can be annotated; a deliberately wrong state is refused |
| 3 | UI: ribbon, chips, panel | scrub a real replay and read both numbers at every step |
| 4 | live web games (optional) | decide after phase 3 |

## Measure

* **The decision stream must not move.** This is the one real risk: if the readout consumes torch
  RNG differently from `sample_action`, the same seed produces a *different game*, and everything
  that compares a replay to a training rollout silently changes meaning.
  `tests/replayer/test_replay_reproduces.py` and `tests/replayer/test_replay_matches_training.py`
  are the acceptance gate, plus the seeded equivalence test from §1 above.
* New tests: probabilities over `top` plus `p_tail` sum to 1 ± 1e-5; `chosen_idx` equals the step's
  `flat_action_idx`; `top` is descending and every entry is legal in the reconstructed pre-action
  state; forced steps are marked and carry no `top`; a replay generated with `--no-trace` is
  byte-identical to today's output for the same seed.
* `tests/bindings/test_replay_schema_golden.py` regenerated deliberately
  (`python tests/bindings/test_replay_schema_golden.py`), with the new types added to
  `REPLAY_TYPES` first.
* Overhead, measured not assumed: wall time of `generate_self_play_replay` with and without the
  trace, and the file-size delta, quoted in the commit.
* `pyrefly check ai tools tests web bindings` at 0 errors.
* Docs: `web/server/AGENTS.md` (the on-disk replay contract), `tools/README.md` (the new CLI), and
  root `AGENTS.md` §4 where the self-play recorder's contract is stated.

## Decide before running

*Answered by the owner on 2026-09-16, and built that way.*

1. **`top_k` and `p_floor`** — first 12 and 1e-3, then **revised to 0 and 0.0 (list every legal
   action)** on 2026-09-17, when the display moved onto the choices themselves: a number painted
   on each card, mode button and country has no list-length problem, and a truncated
   distribution would leave most of the board unlabelled. A non-zero `--trace-top-k` still caps
   it; `--trace-full` is gone, being the default. The widest node is an 84-way placement, ~4.5 KB
   against a step that already carries a ~33 KB snapshot.
2. **Critic on forced steps** — yes, `--trace-critic-every step` is the default, for a
   continuous curve; `decision` halves the forwards if generation time turns out to matter.
3. **Phase 4** (live web games) — **dropped.** Nothing in `web/server/session.py` or
   `web/bot_client.py` was touched, so a human-vs-bot game records no trace. The design note in
   §6 above stands if it is ever wanted.

## Follow-ups

* A `p_chosen` histogram over a set of replays is a cheap read on policy collapse that needs no
  training run, and pairs directly with the entropy training already logs.
* The blunder counter can report the policy's own probability on each blunder it counts —
  "committed / chances" becomes "committed / chances / mean belief", which distinguishes a bad
  policy from a hot sampler without a new run.
* Annotating one game with several snapshots of one lineage gives a value-trace-over-training view
  on a fixed position set, which is the shape of instrument P15-X0 wants.

## Runs

(none — this is engineering, not an arm)
