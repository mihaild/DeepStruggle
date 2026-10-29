#!/usr/bin/env python3
"""Architecture and recipe codes, derived from what each run recorded, and every run's name.

A run is named `E<n>-A<n>-R<n>-S<n>` plus the branch points that led to it
(`research/method/run_nomenclature.md`). A and R are codes in `research/run_codes.json`, and a code
*is* a set of `tools/train.py` flags: the architecture flags (the network's shape, its input and
its heads) for A, every other training setting for R. Bookkeeping -- where files go, how often to
snapshot, evals, the budget, the seeds -- is neither. Nothing is assigned by hand, so two runs
with one code cannot quietly differ in a flag, which is how lineage-critical flags drifted four
times on the E4 ladder (CLAUDE.md, invariant 15).

    tools/scripts/run_codes.py                 # every run's name; fails on a recipe with no code
    tools/scripts/run_codes.py --update        # give new architectures/recipes the next codes,
                                               # rewrite run_codes.json and run_name_map.md
    tools/scripts/run_codes.py --run <dir>     # one run: its codes, its name, and whether the
                                               # name it was launched under says the same
    tools/scripts/run_codes.py --link          # a <new name>_<timestamp> directory of links for
                                               # every old-scheme run, and _models/<label> links

A value fit (`--mode value-fit`, P34's evaluators) takes its network's A code with `value_only`
added -- the same shape read only for its value is a different thing to name -- and its R code from
the value-fit flags, the dataset among them. A policy fit (`--mode policy-fit`, P36c's students) is
the same with `policy_only`, and an action-value fit (`--mode q-fit`) with `q_only`. A labelling recipe (`D`, tools/value_labels.py) is the
same idea for datasets: `labelling_code` maps a dataset's labelling settings to its code under
`labellings`, assigning the next one for a new set.

What a flag cannot show is recorded by hand under `overrides` in the JSON: a code change that
altered training with the flags unchanged (E7-12-44 centred the setup credit), a league whose
exploiter was not training, a run the registry voids. A recipe's description is written there
too; this tool fills in the flags.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.scripts import launch_flags as lf  # noqa: E402
from ai.training.run_name import (LEGACY_RUN_NAME_RE, RunNameError, RunPath, is_run_name,  # noqa: E402
                                  parse_label, parse_run_name)
from tools.lib.data_root import checkpoints_dir  # noqa: E402

CODES_JSON = os.path.join(_ROOT, "research", "run_codes.json")
MAP_MD = os.path.join(_ROOT, "research", "run_name_map.md")
CODES_MD = os.path.join(_ROOT, "research", "architectures_and_recipes.md")

#: The network: its shape, what it reads and its policy heads. An auxiliary head trained only as a
#: target (aux_card_events, aux_opp_legality) is recipe: the policy never reads it.
ARCH_FLAGS = frozenset({
    "arch", "identity_dim", "drop_static", "categorical_value", "obs_features", "self_transform",
    "graph_layers", "per_entity_heads", "attn_readout",
})
#: Neither network nor training: where files go, cadence, evals, budget, seeds, other modes.
#: `merged_influence` is the engine's minor version (E4.1, P23), not a recipe.
BOOKKEEPING = frozenset({
    "mode", "resume", "resume_every_snapshot", "resume_every_steps", "snapshot_every_steps",
    "train_steps", "eval_max_snapshot_opponents", "eval_opponents", "eval_games_per_side",
    "post_tournament", "post_tournament_models", "post_tournament_games", "output_dir",
    "run_name", "description", "device", "tensorboard", "seed", "seed_init", "seed_sampling",
    "seed_env", "seed_pool", "warmup_checkpoint", "warmup_dataset", "distill_dataset",
    "distill_epochs", "distill_lr", "bc_epochs", "curriculum_switch_fraction", "no_cuda_graphs",
    "merged_influence", "new_directory", "turn_credit_workers", "search_workers", "resume_widen", "resume_grow_sphere",
    "resume_grow_tokens",
    # a value fit's shape change: the network it builds is recorded as ladder_config, the A code's
    "value_arch", "policy_arch",
    # two arms taking turns on one machine: the wall clock, not the training
    "phase_lock",
    # the token path in checkpointed passes: memory, not the function
    "token_chunk",
})
#: Settings that do nothing while their switch is off: dropped then, so a stray value cannot make
#: two identical recipes look different. (switch, its off value, the settings it gates)
_GATED: Tuple[Tuple[str, Any, Tuple[str, ...]], ...] = (
    ("aux_card_events", 0.0, ("aux_card_sample_frac", "aux_card_min_batch", "aux_card_buffer",
                              "aux_card_steps", "aux_card_batch")),
    ("lr_schedule", "constant", ("lr_schedule_start", "lr_schedule_every", "lr_schedule_values",
                                 "lr_schedule_span", "lr_min")),
    ("setup_mc_credit", False, ("setup_mc_coef", "setup_mc_min_batch")),
    ("mup", False, ("mup_base_proj_dim", "mup_base_hidden_dim")),
    ("setup_script_frac", 0.0, ("setup_script_openings",)),
    ("setup_entropy_floor", 0.0, ("setup_entropy_lr", "setup_entropy_max_coef")),
    ("play_mode_floor", 0.0, ("floor_scope", "floor_from", "floor_anneal_from",
                              "floor_anneal_steps", "floor_rows")),
    ("force_applicable_events", None, ("force_event_frac", "force_events_from",
                                       "force_event_credit")),
    ("seed_scenarios", None, ("seed_frac", "seed_scenarios_from")),
    ("forced_openings_frac", 0.0, ("forced_openings_us", "forced_openings_ussr")),
    ("search_ce_coef", 0.0, ("search_sims", "search_subsample", "search_node_filter",
                             "search_gumbel_k", "search_prior_temperature",
                             "search_teacher_checkpoint", "search_rollout_horizon", "search_rollout_k", "search_teacher_policy",
                             "search_rollout_worlds", "search_rollout_stop", "search_rollout_leaf_side")),
    ("teacher_coef", 0.0, ("teacher_checkpoint", "teacher_seat")),
    ("active_search_spec", None, ("active_search_frac", "active_search_ce_coef", "active_search_gate_every_steps",
                                  "active_search_gate_games", "active_search_gate_threshold",
                                  "active_search_gate_only", "active_search_refresh_steps",
                                  "active_search_refresh_steps_late", "active_search_late_from")),
    ("searched_opponent_spec", None, ("searched_opponent_share", "searched_opponent_scope",
                                      "searched_opponent_refresh_steps", "searched_opponent_batch",
                                      "searched_opponent_max_wait")),
    ("mode_cf_coef", 0.0, ("mode_cf_subsample", "mode_cf_playouts", "mode_cf_from")),
    ("turn_credit_coef", 0.0, ("turn_credit_budget", "turn_credit_worlds", "turn_credit_beta",
                               "turn_credit_k", "turn_credit_floor", "turn_credit_nested_p",
                               "turn_credit_nested_share", "turn_credit_narrow_share", "turn_credit_cap",
                               "turn_credit_steps", "turn_credit_batch", "turn_credit_buffer")),
    ("aux_opp_legality", 0.0, ("aux_opp_legality_frac", "aux_opp_legality_min_batch")),
    ("league_dirs", None, ("league_frac", "league_pool_size")),
    ("opponent_pfsp", False, ("opponent_pfsp_weighting", "opponent_pfsp_uniform_mix")),
    ("aux_ownership", 0.0, ()),
    ("ladder_logit_cap", 0.0, ("ladder_logit_cap_grad", "ladder_logit_cap_leak",
                                     "ladder_logit_cap_leak_up")),
    ("logit_gap_coef", 0.0, ("logit_gap",)),
)
#: A second switch that keeps a gate open: the searcher also runs for its values alone (P32 T2).
_GATE_ALSO: Dict[str, Tuple[str, Any]] = {"search_ce_coef": ("search_value_beta", 0.0)}

_RUN_DIR_RE = re.compile(r"^(?P<name>[^_]+)_(?P<ts>\d{8}_\d{6})(?P<void>_VOID)?$")
_RESUME_STEP_RE = re.compile(r"resume(?:_paused)?_(\d+)steps\.pt$")
#: A run started from a SWA of its parent (a P34 value fit): it branches where the window ends.
_SWA_STEP_RE = re.compile(r"swa_\d+-(\d+)M\.pt$")


def category(dest: str) -> str:
    if dest in ARCH_FLAGS or dest.startswith("ladder_"):
        return "A"
    return "B" if dest in BOOKKEEPING else "R"


def _off(v: Any, off: Any) -> bool:
    return v in (None, [], (), "", off) or (off is None and not v)


class Run:
    """One run directory: its old name, its parent, and its A/R signatures."""

    def __init__(self, path: str, meta: Dict[str, Any], flags: Dict[str, Any]) -> None:
        self.path = path
        self.dir = os.path.basename(path)
        m = _RUN_DIR_RE.match(self.dir)
        if not m:
            raise ValueError(f"{self.dir} is not <name>_<timestamp>")
        self.name0 = m.group("name")
        self.ts = m.group("ts")
        self.void = bool(m.group("void"))
        self.meta = meta
        self.flags = flags
        self.parent: Optional[Run] = None
        self.branch_step = 0
        self.end_step = int(meta.get("train_steps") or 0)

    @property
    def seed(self) -> str:
        if self.meta.get("mode") in ("value-fit", "policy-fit", "q-fit", "online-distill"):
            return str(self.meta.get("seed"))      # a supervised fit has one seed
        s = {self.meta.get(k) for k in ("seed_env", "seed_sampling", "seed_pool")}
        if len(s) != 1:
            raise ValueError(f"{self.dir}: env/sampling/pool seeds differ {s}; S cannot name one")
        return str(s.pop())

    @property
    def engine(self) -> str:
        """From the name it was launched under: the last E of a new-grammar name (a branch can
        change it), the prefix of an old one, whose merged-influence view is the minor version."""
        if not LEGACY_RUN_NAME_RE.match(self.name0) and is_run_name(self.name0):
            return parse_run_name(self.name0).fields()["E"]
        m = re.match(r"^E(\d+)", self.name0)
        if not m:
            raise ValueError(f"{self.dir}: no engine in its name")
        minor = ".1" if self.flags.get("merged_influence") else ""
        return m.group(1) + minor

    def lineage_names(self) -> List[str]:
        out, r = [], self
        while r is not None:
            out.append(r.name0)
            r = r.parent
        return out


def is_link_dir(d: str) -> bool:
    """A new-name directory made by `--link`: its files are symlinks into the run's own directory,
    so it is an alias of that run, not a run."""
    return os.path.islink(os.path.join(d, "metadata.json"))


def link_runs(runs: List[Run], names: Dict[str, RunPath], root: str) -> List[str]:
    """Give every run launched under an old-scheme name a `<new name>_<its timestamp>` directory
    of relative symlinks to its files. Idempotent: an existing link directory gains the files its
    run has written since (an SWA, a later snapshot), and nothing is ever removed or overwritten."""
    made: List[str] = []
    for r in runs:
        if r.dir not in names or not LEGACY_RUN_NAME_RE.match(r.name0):
            continue
        target = os.path.join(root, f"{names[r.dir].text()}_{r.ts}")
        if os.path.exists(target):
            meta = os.path.join(target, "metadata.json")
            if not (is_link_dir(target)
                    and os.path.realpath(meta) == os.path.realpath(os.path.join(r.path, "metadata.json"))):
                raise ValueError(f"{target} exists and is not the link directory of {r.dir}")
        else:
            os.makedirs(target)
            made.append(os.path.basename(target))
        for entry in sorted(os.listdir(r.path)):
            link = os.path.join(target, entry)
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", r.dir, entry), link)
    return made


def link_models(codes: Dict[str, Any], root: str) -> List[str]:
    """`_models/<label>.<ext>`: soups and SWAs kept outside run directories, named by their label
    (`codes["models"]` maps each file to it). The label parses, so tools name the model by it."""
    made: List[str] = []
    out = os.path.join(root, "_models")
    os.makedirs(out, exist_ok=True)
    for rel, label in sorted(codes.get("models", {}).items()):
        parse_label(label)
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            raise ValueError(f"model file {rel} is missing")
        link = os.path.join(out, label + os.path.splitext(rel)[1])
        if os.path.lexists(link):
            if os.path.realpath(link) != os.path.realpath(src):
                raise ValueError(f"{link} already names another file")
            continue
        os.symlink(os.path.join("..", rel), link)
        made.append(os.path.basename(link))
    return made


def _resume_source(src: str) -> Tuple[str, str]:
    """(run directory, state file) of a recorded `resumed_from`: a state file, a run directory
    (its end state), or `<run_dir>:<steps>` (that step's state) -- the forms `--resume` takes."""
    head, sep, tail = src.rpartition(":")
    if sep and tail.isdigit():
        return head.rstrip("/"), f"resume_{tail}steps.pt"
    if os.path.isdir(src) or not src.endswith(".pt"):
        return src.rstrip("/"), "resume_state.pt"
    return os.path.dirname(src), os.path.basename(src)


def load_runs(root: str) -> List[Run]:
    runs: List[Run] = []
    for d in sorted(glob.glob(os.path.join(root, "E*_*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "metadata.json")):
            continue
        if not _RUN_DIR_RE.match(os.path.basename(d)) or is_link_dir(d):
            continue
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        runs.append(Run(d, meta, lf.non_default(d)))
    by_dir = {r.dir: r for r in runs}
    for r in runs:
        # A run continued in place keeps every leg; where it came from is its first leg's resume.
        src = (r.meta.get("legs") or [r.meta])[0].get("resumed_from")
        if not src:
            continue
        src_dir, src_file = _resume_source(src)
        pdir = os.path.basename(src_dir)
        alias = os.path.join(root, pdir)
        if is_link_dir(alias):            # resumed through a new-name link directory
            pdir = os.path.basename(os.path.dirname(os.path.realpath(os.path.join(alias, "metadata.json"))))
        parent = by_dir.get(pdir)
        if parent is None:
            raise ValueError(f"{r.dir} resumed from {src}, which is not a run directory here")
        r.parent = parent
        m = _RESUME_STEP_RE.search(src_file)
        swa = _SWA_STEP_RE.search(src_file)
        # A resume from the run directory (its newest state) branches where that state stood, which the
        # branch records as start_steps -- not the parent's budget, which a parent stopped early never reached.
        recorded = (r.meta.get("legs") or [r.meta])[0].get("start_steps")     # the first leg's, never the latest
        r.branch_step = (int(m.group(1)) if m else int(swa.group(1)) * 1_000_000 if swa
                         else int(recorded) if recorded else parent.end_step)
    runs.sort(key=lambda r: r.ts)
    return runs


def arch_sig(r: Run) -> Dict[str, Any]:
    sig = {k: v for k, v in r.flags.items() if category(k) == "A"}
    if r.meta.get("mode") == "value-fit":
        sig["value_only"] = True
    if r.meta.get("mode") == "policy-fit":
        sig["policy_only"] = True
    if r.meta.get("mode") == "q-fit":
        sig["q_only"] = True
    return sig


def _run_of(checkpoint: str) -> Optional[str]:
    """The run name of the directory a checkpoint sits in, or None."""
    m = _RUN_DIR_RE.match(os.path.basename(os.path.dirname(checkpoint)))
    return m.group("name") if m else None


def recipe_sig(r: Run, overrides: Dict[str, Any]) -> Dict[str, Any]:
    sig = {k: v for k, v in r.flags.items() if category(k) == "R"}
    for switch, off, gated in _GATED:
        also = _GATE_ALSO.get(switch)
        if _off(sig.get(switch), off) and (also is None or _off(sig.get(also[0]), also[1])):
            for g in gated:
                sig.pop(g, None)
    if sig.get("opponent_checkpoints") == []:
        sig.pop("opponent_checkpoints")
    # A frozen opponent taken from the run's own lineage is a role, not a file.
    lineage = set(r.lineage_names()[1:])
    if sig.get("opponent_checkpoints"):
        sig["opponent_checkpoints"] = [
            "<parent snapshot at the branch point>" if _run_of(p) in lineage else p
            for p in sig["opponent_checkpoints"]]
    if sig.get("league_dirs"):
        own = set(r.lineage_names())
        sig["league_dirs"] = ["<this lineage's league>" if os.path.basename(p.rstrip("/")) in own
                              else p for p in sig["league_dirs"]]
    note = overrides.get(r.dir, {}).get("recipe_note")
    if note:
        sig["_note"] = note
    return sig


def _key(sig: Dict[str, Any]) -> str:
    return json.dumps(sig, sort_keys=True, default=str)


def load_codes() -> Dict[str, Any]:
    if not os.path.exists(CODES_JSON):
        return {"architectures": {}, "recipes": {}, "overrides": {}, "runs": {}}
    with open(CODES_JSON, encoding="utf-8") as f:
        return json.load(f)


def labelling_code(sig: Dict[str, Any], update: bool) -> Optional[str]:
    """The D code of a labelling recipe (tools/value_labels.py's settings), or None. With `update` a
    new recipe gets the next code, written to run_codes.json at once."""
    codes = load_codes()
    table = codes.setdefault("labellings", {})
    code = _lookup(table, sig)
    if code is None and update:
        code = f"D{1 + max((int(c[1:]) for c in table), default=0)}"
        table[code] = {"description": "", "flags": sig}
        with open(CODES_JSON, "w", encoding="utf-8") as f:
            json.dump(codes, f, indent=1, sort_keys=False, default=str)
            f.write("\n")
    return code


def _lookup(table: Dict[str, Any], sig: Dict[str, Any]) -> Optional[str]:
    k = _key(sig)
    for code, entry in table.items():
        if _key(entry["flags"]) == k:
            return code
    return None


def assign(runs: List[Run], codes: Dict[str, Any], update: bool) -> Tuple[Dict[str, RunPath], List[str]]:
    """Each run's name, and the signatures that have no code (assigned when `update`)."""
    overrides = codes.get("overrides", {})
    missing: List[str] = []
    names: Dict[str, RunPath] = {}
    for r in runs:
        if r.void or overrides.get(r.dir, {}).get("void"):
            continue
        fields: Dict[str, str] = {"E": r.engine, "S": r.seed}
        for key, table_name, sig, letter in (
                ("A", "architectures", arch_sig(r), "A"),
                ("R", "recipes", recipe_sig(r, overrides), "R")):
            table = codes.setdefault(table_name, {})
            code = _lookup(table, sig)
            if code is None:
                if not update:
                    missing.append(f"{r.dir}: no {table_name[:-1]} code for {_key(sig)}")
                    code = "?"
                else:
                    n = 1 + max((int(c[1:]) for c in table), default=0)
                    code = f"{letter}{n}"
                    table[code] = {"description": "", "first": r.dir, "flags": sig}
            fields[key] = code[1:] if code != "?" else "0"
        if r.parent is None or r.parent.dir not in names:
            root = tuple((k, fields[k]) for k in ("E", "A", "R", "S"))
            names[r.dir] = RunPath(root)
        else:
            parent = names[r.parent.dir]
            branch_m = round(r.branch_step / 1e6)
            names[r.dir] = parent.branch(branch_m, fields)
    _replicates(runs, names)
    return names, missing


def _replicates(runs: List[Run], names: Dict[str, RunPath]) -> None:
    """Two from-scratch runs with one name are same-seed replicates: the later ones get -2, -3."""
    seen: Dict[str, List[str]] = {}
    for r in runs:
        if r.dir in names and r.parent is None:
            seen.setdefault(names[r.dir].text(), []).append(r.dir)
    replicate: Dict[str, int] = {}
    for _, dirs in seen.items():
        for i, d in enumerate(dirs[1:], start=2):
            p = names[d]
            names[d] = RunPath(p.root, p.segments, i)
            replicate[d] = i
    # A branch of a replicate carries the index on its root (`E8-A8-R1-S44-2@10010M+R87`); its name was built
    # from the parent's before the index was known, so it is added here, down every branch.
    by_dir = {r.dir: r for r in runs}
    for r in runs:
        if r.parent is None or r.dir not in names:
            continue
        top = r
        while top.parent is not None and top.parent.dir in by_dir:
            top = by_dir[top.parent.dir]
        if top.dir in replicate:
            p = names[r.dir]
            names[r.dir] = RunPath(p.root, p.segments, p.index, replicate[top.dir])


def write_map(runs: List[Run], names: Dict[str, RunPath], codes: Dict[str, Any]) -> None:
    rows = []
    for r in runs:
        void = r.void or codes.get("overrides", {}).get(r.dir, {}).get("void")
        if void:
            new = "void -- not named"
        else:
            new = f"`{names[r.dir].text()}`"
        span = f"{r.branch_step / 1e6:,.0f} → {r.end_step / 1e6:,.0f}M"
        rows.append(f"| `{r.dir}` | {new} | {span} |")
    lines = [
        "# Run names: the old scheme mapped onto E-A-R-S",
        "",
        "Generated by `tools/scripts/run_codes.py --update` from each directory's `metadata.json`;",
        "do not edit by hand. The scheme is [`method/run_nomenclature.md`](method/run_nomenclature.md),",
        "the codes are [`architectures_and_recipes.md`](architectures_and_recipes.md). Directories keep",
        "their old names -- research cites them by path -- and each also has a `<name>_<timestamp>`",
        "directory of symlinks under the new name (`run_codes.py --link`), so tools and resolvers take",
        "the new names. Soups and line SWAs kept outside run directories are linked as",
        "`data/checkpoints/_models/<label>.pt`. Several directories with one name are legs of one run",
        "(a continuation that changed nothing keeps its name).",
        "",
        "| directory | name | steps |",
        "|:---|:---|:---|",
    ] + rows + [""]
    with open(MAP_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _flags_text(flags: Dict[str, Any], base: Optional[Dict[str, Any]] = None) -> str:
    """A code's flags; against `base` (A1 or R1) only what differs, a dropped flag as `(default)`."""
    body = {k: v for k, v in flags.items() if k != "_note"}
    if base is not None:
        b = {k: v for k, v in base.items() if k != "_note"}
        changed = {k: v for k, v in body.items() if b.get(k, object()) != v}
        dropped = [k for k in b if k not in body]
        parts = [f"`{p}`" for p in lf.as_flags(changed)] + \
            [f"`--{k.replace('_', '-')}` (default)" for k in sorted(dropped)]
        text = " ".join(parts) or "*(as the base)*"
    else:
        text = " ".join(f"`{p}`" for p in lf.as_flags(body)) or "*(every flag at its default)*"
    return text + (f" -- *{flags['_note']}*" if "_note" in flags else "")


def recorded_elsewhere(codes: Dict[str, Any], runs: List[Run]) -> Dict[str, str]:
    """The named runs in run_codes.json whose directories are not under this checkpoint root: runs made on
    another machine (CI's fits, a teammate's box), recorded by an update there. Kept, since nothing here can
    re-derive them; a run deleted for good is removed from the JSON by hand."""
    local = {r.dir for r in runs}
    return {d: n for d, n in codes.get("runs", {}).items() if d not in local}


def write_codes_md(codes: Dict[str, Any], names: Dict[str, RunPath], runs: List[Run],
                   elsewhere: Optional[Dict[str, str]] = None) -> None:
    used: Dict[str, List[str]] = {}
    for r in runs:
        if r.dir in names:
            f = names[r.dir].fields()
            used.setdefault("A" + f["A"], []).append(r.name0)
            used.setdefault("R" + f["R"], []).append(r.name0)
    for d, name in (elsewhere or {}).items():
        try:
            f = parse_run_name(name).fields()
        except RunNameError:
            continue
        launched = re.sub(r"_\d{8}_\d{6}$", "", d)
        used.setdefault("A" + f["A"], []).append(launched)
        used.setdefault("R" + f["R"], []).append(launched)
    out = [
        "# Architectures and recipes",
        "",
        "The A and R codes of the run names ([`method/run_nomenclature.md`](method/run_nomenclature.md)).",
        "A code is a set of `tools/train.py` flags, derived from the runs' own `metadata.json` by",
        "`tools/scripts/run_codes.py`, which also writes this file; the descriptions are kept in",
        "[`run_codes.json`](run_codes.json). An architecture is the network -- its shape, its input",
        "and its policy heads; a recipe is everything else that trains it. Flags are listed against",
        "the CLI's defaults: A1 and R1 in full, every other code as its difference from them.",
        "A default that changed at an engine epoch is read as the run's own epoch had it: since E8 a",
        "network reads `headline_reveal` unless its code lists `--obs-features`, so an A code names",
        "the same flags on E7 and E8 and the E letter says which view it reads.",
        "",
        "## Architectures",
        "",
        "| code | what it is | flags | runs (old names) |",
        "|:---|:---|:---|:---|",
    ]
    base_a = codes["architectures"].get("A1", {}).get("flags")
    for code, e in sorted(codes["architectures"].items(), key=lambda kv: int(kv[0][1:])):
        runs_s = ", ".join(sorted(set(used.get(code, [])), key=lambda n: n))
        txt = _flags_text(e["flags"], None if code == "A1" else base_a)
        out.append(f"| **{code}** | {e['description']} | {txt} | {runs_s} |")
    out += ["", "## Recipes", "", "| code | what it is | flags | runs (old names) |",
            "|:---|:---|:---|:---|"]
    base_r = codes["recipes"].get("R1", {}).get("flags")
    for code, e in sorted(codes["recipes"].items(), key=lambda kv: int(kv[0][1:])):
        runs_s = ", ".join(sorted(set(used.get(code, [])), key=lambda n: n))
        txt = _flags_text(e["flags"], None if code == "R1" else base_r)
        out.append(f"| **{code}** | {e['description']} | {txt} | {runs_s} |")
    out.append("")
    with open(CODES_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true",
                    help="assign codes to new signatures and rewrite the JSON and both tables")
    ap.add_argument("--run", metavar="DIR", help="one run: its codes and name")
    ap.add_argument("--link", action="store_true",
                    help="create each old-scheme run's <new name>_<timestamp> directory of links, "
                         "and _models/<label> links for the soups and SWAs in run_codes.json")
    ap.add_argument("--root", default=None, help="checkpoint root (default: the shared data tree)")
    a = ap.parse_args()
    root = a.root or checkpoints_dir()
    runs = load_runs(root)
    codes = load_codes()
    names, missing = assign(runs, codes, a.update)
    if a.link:
        if missing:
            for line in missing:
                print("MISSING", line)
            return 1
        made = link_runs(runs, names, root)
        models = link_models(codes, root)
        print(f"{len(made)} run directories and {len(models)} model links created under {root}")
        return 0
    if a.run:
        d = os.path.basename(a.run.rstrip("/"))
        if d not in names:
            print(f"{d}: void or not found")
            return 1
        name = names[d].text()
        launched = next(r.name0 for r in runs if r.dir == d)
        print(f"{d}\n  name {name}")
        if is_run_name(launched) and not LEGACY_RUN_NAME_RE.match(launched):
            ok = parse_run_name(launched).text() == name
            print(f"  launched as {launched}: {'agrees' if ok else 'DISAGREES -- its flags say ' + name}")
            return 0 if ok else 1
        return 0
    for line in missing:
        print("MISSING", line)
    if a.update:
        elsewhere = recorded_elsewhere(codes, runs)
        here = {d: p.text() for d, p in names.items()}
        codes["runs"] = {d: here.get(d, elsewhere.get(d)) for d in codes.get("runs", {})
                         if d in here or d in elsewhere}
        codes["runs"].update({d: n for d, n in here.items() if d not in codes["runs"]})
        with open(CODES_JSON, "w", encoding="utf-8") as f:
            json.dump(codes, f, indent=1, sort_keys=False, default=str)
            f.write("\n")
        write_map(runs, names, codes)
        write_codes_md(codes, names, runs, elsewhere)
        print(f"wrote {os.path.relpath(CODES_JSON, _ROOT)}, {os.path.relpath(MAP_MD, _ROOT)}, "
              f"{os.path.relpath(CODES_MD, _ROOT)}")
    else:
        for r in runs:
            if r.dir in names:
                print(f"{r.dir:42s} {names[r.dir].text()}")
    return 1 if missing and not a.update else 0


if __name__ == "__main__":
    raise SystemExit(main())
