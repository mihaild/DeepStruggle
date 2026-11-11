"""The run-name grammar: engine, architecture, recipe, view and seed, and the path a run took.

`research/method/run_nomenclature.md` is the specification; this module is its one parser. A run
is named by where it started and every change made to it since:

    E7-A4-R1-S44                       engine 7, architecture 4, recipe 1, seed 44, from scratch
    E7-A4-R1-S44@4390M+S45             that run's 4,390M state, continued under seed 45
    E7-A4-R1-S44@4390M+A7-R10          ... with architecture 7 and recipe 10 from 4,390M
    E7-A4-R1-S44-2                     a same-seed replicate of the first
    E7-A4-R1-S44-2@4390M+R10           a branch of that replicate (its index stays with the root)
    E8-A35-R102-V1-S44                 view 1 (V: how observations and actions are represented); V0, the
                                       default view, is never written


`@<n>M` is always an **absolute** step on the lineage's own clock, never a length. In a run's
name every `@` is a branch point; a continuation that changes nothing keeps the name. A field in a
change list restates only what changed, in the order E, A, R, V, S.

A *label* names a model, so it adds a selector -- one snapshot or a range averaged into an SWA --
and may put a group of alternatives after the last branch point, which is a soup:

    E7-A4-R1-S44@4800M                         one snapshot
    E7-A4-R1-S44@4720..4800M                   the SWA of that range
    E7-A4-R1-S44@4390M+(S44,45,46)@4800M       soup of three branches of the 4,390M state
    E7-A2-R2-S5@800M+R3@1200M+(S6,R4-S7)@1500..1600M
    (E7-A4-R1-S44,E7-A4-R23-S44)@1200M         soup of two runs with no common state

In a group, an item that changes nothing (`S44` on a seed-44 parent) is the parent's own
continuation, and a bare number repeats the previous item's letter (`S44,45,46`). A selector after
a group applies to every item. Parentheses, not braces: the shells expand `{a,b}` silently, while
an unquoted `(` is a syntax error in bash and zsh -- loud, never wrong.

Names in the older `E<engine>-<attempt>-<seed>` scheme stay valid for the directories that carry
them (`LEGACY_RUN_NAME_RE`); they are mapped onto this grammar in `research/run_name_map.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Tuple

#: The scheme until 2026-10-07: `E<engine>-<attempt>-<seed>[-<replicate>][-<M>M.<seed>]...`.
LEGACY_RUN_NAME_RE: Final = re.compile(
    r"^E\d+(?:\.\d+)?-\d{2}-\d{2}(?:-\d+)?(?:-\d+M\.\d{2})*$")

#: The fields, in the order a name and a change list give them. V (the view, owner 2026-11-07) is optional:
#: a name without it is V0, the default view, and V0 is never written in a root.
FIELDS: Final = ("E", "A", "R", "V", "S")
_FIELD_RE: Final = re.compile(r"^(E)(\d+(?:\.\d+)?)$|^([ARVS])(\d+)$")
#: A field's value when a name does not give it.
_DEFAULTS: Final = {"V": "0"}
_STEP_RE: Final = re.compile(r"^(\d+)([Mk])$")


class RunNameError(ValueError):
    """A string that is not a name in the grammar, with what is wrong with it."""


def _field(tok: str) -> Tuple[str, str]:
    m = _FIELD_RE.match(tok)
    if not m:
        raise RunNameError(f"{tok!r} is not a field (E<n>[.<m>], A<n>, R<n>, V<n> or S<n>)")
    return (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))


def _millions(tok: str, what: str) -> int:
    """`4390M` -> 4390; the step of a branch point, in millions."""
    m = _STEP_RE.match(tok)
    if not m or m.group(2) != "M":
        raise RunNameError(f"{what} {tok!r} is not <n>M (absolute steps in millions)")
    return int(m.group(1))


@dataclass(frozen=True)
class Segment:
    """`@<step>M+<changes>`: from this run's state at `step` million, with `changes` applied."""
    step: int
    changes: Tuple[Tuple[str, str], ...]

    def text(self) -> str:
        return f"@{self.step}M+" + "-".join(f"{k}{v}" for k, v in self.changes)


@dataclass(frozen=True)
class RunPath:
    """A run: a from-scratch root and the branch points that led to it."""
    root: Tuple[Tuple[str, str], ...]          # (("E","7"),("A","4"),("R","1"),("S","44")); V only when not 0
    segments: Tuple[Segment, ...] = ()
    index: Optional[int] = None                # same-seed replicate (or league generation)
    #: The root's own replicate index when the run branches from a replicate (`E8-A8-R1-S44-2@10010M+R87`):
    #: it belongs to the root and is written there, before the branch points.
    root_index: Optional[int] = None

    def text(self) -> str:
        s = "-".join(f"{k}{v}" for k, v in self.root)
        s += (f"-{self.root_index}" if self.root_index is not None else "") + "".join(g.text() for g in self.segments)
        return s + (f"-{self.index}" if self.index is not None else "")

    def fields(self) -> Dict[str, str]:
        """E, A, R, V and S as they stand at the end of the path (V "0" where no name gave it)."""
        cur = dict(_DEFAULTS)
        cur.update(dict(self.root))
        for seg in self.segments:
            cur.update(dict(seg.changes))
        return cur

    def last_step(self) -> int:
        return self.segments[-1].step if self.segments else 0

    def branch(self, step: int, changes: Dict[str, str]) -> "RunPath":
        """This run's state at `step` million with `changes`; unchanged fields are dropped, and a
        branch that changes nothing is the run itself (a continuation keeps its name)."""
        cur = self.fields()
        diff = tuple((k, changes[k]) for k in FIELDS if k in changes and changes[k] != cur[k])
        if not diff:
            return self
        if step <= self.last_step():
            raise RunNameError(f"branch at {step}M is not after the last branch point "
                               f"({self.last_step()}M) of {self.text()}")
        # A branch of a from-scratch replicate keeps the replicate's index on its root.
        root_index = self.root_index if self.segments else self.index
        return RunPath(self.root, self.segments + (Segment(step, diff),), None, root_index)


@dataclass(frozen=True)
class Selector:
    """`@<n>M` (one snapshot) or `@<lo>..<hi>M` (the SWA of every snapshot in the range)."""
    lo: str
    hi: Optional[str] = None

    def text(self) -> str:
        return f"@{self.lo}" if self.hi is None else f"@{self.lo.rstrip('Mk')}..{self.hi}"


@dataclass
class Label:
    """A model: one or more ingredient runs, and which of their snapshots."""
    ingredients: List[RunPath] = field(default_factory=list)
    selector: Optional[Selector] = None

    @property
    def is_soup(self) -> bool:
        return len(self.ingredients) > 1


# ---------------------------------------------------------------------------------------------
# tokenizing: '@', '+', '(', ')', ',' and '..' are structure; '-' separates fields and an index.

def _split_top(s: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise RunNameError(f"unbalanced ')' in {s!r}")
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if depth:
        raise RunNameError(f"unbalanced '(' in {s!r}")
    out.append(cur)
    return out


def _changes(text: str, prev_letter: Optional[str] = None) -> Tuple[Tuple[Tuple[str, str], ...], Optional[int]]:
    """`R4-S7` -> ((R,4),(S,7)); a trailing bare number is an index; in a group a lone bare
    number repeats the previous item's letter (`45` after `S44`)."""
    toks = text.split("-")
    index = None
    if len(toks) > 1 and toks[-1].isdigit():
        index = int(toks.pop())
    if len(toks) == 1 and toks[0].isdigit() and prev_letter is not None:
        toks = [prev_letter + toks[0]]
    pairs = tuple(_field(t) for t in toks)
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise RunNameError(f"{text!r} changes a field twice")
    if keys != sorted(keys, key=FIELDS.index):
        raise RunNameError(f"{text!r} is not in E, A, R, V, S order")
    return pairs, index


def _root(text: str) -> RunPath:
    toks = text.split("-")
    index = None
    if len(toks) in (5, 6) and toks[-1].isdigit():
        index = int(toks.pop())
    if len(toks) not in (4, 5):
        raise RunNameError(f"{text!r} is not E<n>-A<n>-R<n>[-V<n>]-S<n>[-<replicate>]")
    pairs = tuple(_field(t) for t in toks)
    if tuple(k for k, _ in pairs) not in (("E", "A", "R", "S"), FIELDS):
        raise RunNameError(f"{text!r}: a root names E, A, R, V (optional) and S, in that order")
    if dict(pairs).get("V") == "0":
        raise RunNameError(f"{text!r}: V0 is the default view and is not written")
    return RunPath(pairs, (), index)


def _no_restated(run: RunPath, pairs: Tuple[Tuple[str, str], ...], part: str) -> None:
    cur = run.fields()
    same = [f"{k}{v}" for k, v in pairs if cur[k] == v]
    if same:
        raise RunNameError(f"'@{part}' restates {', '.join(same)}: a change list names only "
                           "what changed, and a continuation keeps the run's name")


def _path(text: str) -> RunPath:
    """A run path with no group and no selector: `root(@<n>M+<changes>)*`."""
    if "(" in text or ")" in text or "," in text or ".." in text:
        raise RunNameError(f"{text!r}: a run's name has no group or range (those name models)")
    parts = text.split("@")
    run = _root(parts[0])
    for part in parts[1:]:
        if "+" not in part:
            raise RunNameError(f"'@{part}' in a run's name must be a branch point, @<n>M+<changes>")
        step_s, ch = part.split("+", 1)
        if run.index is not None and run.segments:
            raise RunNameError(f"{text!r}: a replicate index ends a name (only a from-scratch replicate, "
                               "the root, can be branched from)")
        pairs, index = _changes(ch)
        step = _millions(step_s, "branch point")
        _no_restated(run, pairs, part)
        nxt = run.branch(step, dict(pairs))
        run = RunPath(nxt.root, nxt.segments, index, nxt.root_index)
    return run


def parse_run_name(text: str) -> RunPath:
    """The name a run is launched under and its directory carries. Raises RunNameError."""
    return _path(text)


def is_run_name(text: str) -> bool:
    """True for a name `tools/train.py --run-name` accepts: this grammar, or the legacy scheme."""
    if LEGACY_RUN_NAME_RE.match(text):
        return True
    try:
        _path(text)
        return True
    except RunNameError:
        return False


def _selector(text: str) -> Selector:
    if ".." in text:
        lo, hi = text.split("..", 1)
        if not lo.isdigit():
            raise RunNameError(f"range '{text}' is not <lo>..<hi>M")
        _millions(hi, "range end")
        if int(lo) >= int(hi[:-1]):
            raise RunNameError(f"range '{text}' is empty")
        return Selector(lo + "M", hi)
    m = _STEP_RE.match(text)
    if not m:
        raise RunNameError(f"selector '@{text}' is not @<n>M or @<lo>..<hi>M")
    return Selector(text)


def _expand(prefix: RunPath, step: int, group_body: str) -> List[RunPath]:
    """The ingredients of `prefix@<step>M+(<items>)`."""
    out: List[RunPath] = []
    prev: Optional[str] = None
    for item in _split_top(group_body, ","):
        if not item:
            raise RunNameError("an empty group item: restate the parent's value instead "
                               "(S44 for the seed-44 parent's own continuation)")
        head, *rest = _split_top(item, "@")
        pairs, index = _changes(head, prev)
        if index is not None:
            raise RunNameError(f"group item {item!r} carries an index")
        prev = pairs[0][0]
        cur = prefix.fields()
        same = [k for k, v in pairs if cur[k] == v]
        if same and len(same) != len(pairs):
            raise RunNameError(f"group item {item!r} restates {same} beside a change; only a "
                               "whole item may restate (the parent's own continuation)")
        run = prefix.branch(step, dict(pairs))
        tail = "@" + "@".join(rest) if rest else ""
        out.extend(_continue(run, tail))
    return out


def _continue(run: RunPath, tail: str) -> List[RunPath]:
    """Apply `(@<n>M+<changes>)*` and at most one trailing group to `run`."""
    if not tail:
        return [run]
    parts = _split_top(tail[1:], "@")
    for i, part in enumerate(parts):
        step_s, ch = part.split("+", 1) if "+" in part else (part, "")
        step = _millions(step_s, "branch point")
        if ch.startswith("("):
            if not ch.endswith(")") or i != len(parts) - 1:
                raise RunNameError(f"a group closes a name: '@{part}'")
            return _expand(run, step, ch[1:-1])
        pairs, index = _changes(ch)
        if index is not None:
            raise RunNameError("an index has no place inside a model's label")
        _no_restated(run, pairs, part)
        run = run.branch(step, dict(pairs))
    return [run]


def parse_label(text: str) -> Label:
    """A model: `<run path or group>[@<selector>]`. The selector applies to every ingredient."""
    body, sel = text, None
    parts = _split_top(text, "@")
    if len(parts) > 1 and "+" not in parts[-1]:
        sel = _selector(parts[-1])
        body = "@".join(parts[:-1])
    if body.startswith("("):
        if not body.endswith(")"):
            raise RunNameError(f"{text!r}: a top-level group is the whole name")
        runs = [_path(item) for item in _split_top(body[1:-1], ",")]
    else:
        head, *rest = _split_top(body, "@")
        root = _root(head)
        runs = _continue(root, "@" + "@".join(rest) if rest else "")
    lab = Label(runs, sel)
    _check_soup(lab, text)
    return lab


def _check_soup(lab: Label, text: str) -> None:
    """A soup averages weights, so its ingredients must share the architecture; one engine keeps
    the model's game defined. Duplicates are a typo."""
    if not lab.is_soup:
        return
    names = [r.text() for r in lab.ingredients]
    if len(set(names)) != len(names):
        raise RunNameError(f"{text!r} lists an ingredient twice")
    for key, why in (("A", "weights of different architectures cannot be averaged"),
                     ("E", "a soup across engines plays no one game"),
                     ("V", "a soup across views decides in no one view")):
        vals = {r.fields()[key] for r in lab.ingredients}
        if len(vals) > 1:
            raise RunNameError(f"{text!r} mixes {key}{sorted(vals)}: {why}")


def shares_a_state(lab: Label) -> bool:
    """True when every ingredient of a soup descends from one common branch point -- the case
    weight averaging is known to work in. Ingredients trained apart from scratch are not aligned,
    and their average is usually broken."""
    if not lab.is_soup:
        return True
    roots = {r.root for r in lab.ingredients}
    if len(roots) > 1:
        return False
    return any(r.segments for r in lab.ingredients)
