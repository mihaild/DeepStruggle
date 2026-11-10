"""What a player is: a network plus the way it is run.

The same weights make many players -- greedy, sampled at a temperature, under PUCT search, under a
Gumbel root at some budget, under a rollout root -- and each is a different strength. The agent
strings `load_agent` takes (`temp:0.5:<ckpt>`, `gumbel:<ckpt>:256:8`, `rollout:<ckpt>:4:16:4:z2`,
`search:<ckpt>:128:determinize:all::cpp:0`) say this positionally, with the defaults in code, so the
same string can come to mean a different player when a default moves. Here each way of running a
network is a typed spec with named fields, and a player's id spells it out:

    E7-A4-R1-S44@4800M                                   greedy policy (the plain network name)
    E7-A4-R1-S44@4800M~policy(temperature=0.5)           sampled
    E7-A8-R1-S44@6800M~gumbel(sims=256,k=8,fpu=0.2)      Gumbel root search
    E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2)   rollout root
    E7-A8-R1-S44@6800M~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own)
    E7-A8-R1-S44@6800M~rollout(k=4,...,leaf=own,leaf_net=<evaluator>)   two networks (P34)
    E7-A8-R1-S44@6800M~search(sims=128,determinize=true,node_filter=all,subsample=1,backend=cpp,fpu=0)
    HeuristicBot                                         a rule-based bot, no network

A search spec's id names every field, defaults included, so it never depends on what a default is
today. Only the greedy policy is abbreviated, to the bare network name, because that is the player
everyone means by a checkpoint's name. The one exception is a field added after ids using its kind were
published (`LATE_FIELDS`): it is named only when it differs from the behaviour those ids already had, so
every existing id and stored spec keeps meaning the same player.

Standard library only: tools/lib/leaderboard.py imports this, and it runs on CI without torch.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, Final, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Policy:
    """The network's own policy: argmax at temperature <= GREEDY_TEMPERATURE (the tournament's greedy
    cut-off), otherwise sampled at `temperature`."""

    temperature: float = 0.0


@dataclass(frozen=True)
class Gumbel:
    """A noise-free Gumbel root (ai/search/gumbel_root.py): the `k` most probable moves,
    sequential halving over `sims` network evaluations, first-play urgency reduction `fpu`."""

    sims: int = 256
    k: int = 8
    fpu: float = 0.2


@dataclass(frozen=True)
class Rollout:
    """A rollout root (ai/search/rollout_root.py): the `k` most probable moves, each played out by
    the network in `worlds` sampled worlds through `horizon` action-round boundaries, the critic at
    the leaf, then `rule` -- `argmax`, `z<x>` (the best only on an x-SE paired lead over the
    network's move) or `kl<t>`."""

    k: int = 4
    worlds: int = 16
    horizon: int = 4
    rule: str = "z2"
    #: Where a playout stops (ai.search.rollout_root.STOPS) and whose side its leaf is read from
    #: (LEAF_SIDES): "boundaries" and "mover" are the root every rollout id before them meant.
    stop: str = "boundaries"
    leaf: str = "mover"
    #: The registered network that reads the leaves (P34's evaluator); "" = the root's own network.
    leaf_net: str = ""
    #: How a world's randomness is shared by its candidates: "shared" (one dice stream, the measured
    #: root), "keyed" (each draw tied to its situation, bindings/keyed_chance.hpp) or "keyed2" (keyed
    #: with the situation's player fields narrowed, so a USSR side no longer hides the turn; "keyed"
    #: keeps the original packing every keyed player was measured with).
    chance: str = "shared"
    #: P35 3b: the worlds dealt from the policy-likelihood belief over the opponent's hand with this
    #: many importance-sampled hands, the likelihood temperature model-averaged from the opponent's own
    #: choices (bindings/belief.hpp); 0 = uniform redeals, the root every rollout id before it meant.
    belief: int = 0
    #: Every candidate's horizon counted from the end of the mover's own action round
    #: (bindings/rollout_batch.hpp's Config::aligned), the headline one boundary; false = the count every
    #: rollout id before it meant.
    aligned: bool = False
    #: Space Race box 4 (Config::box4): the opponent's headline, chosen first and revealed to the side holding
    #: Man in Earth Orbit alone, kept in every world; false = forgotten, as every rollout id before it meant.
    box4: bool = False
    #: Race the worlds (RolloutBatch.race): every candidate on the first `race`, the rest of `worlds` only
    #: for the network's move and the challengers not trailing it; 0 = every candidate on every world, as
    #: every rollout id before it meant.
    race: int = 0


@dataclass(frozen=True)
class Search:
    """Batched PUCT search (ai/search/batched_mcts.py). `node_filter` is "all" or "card"
    (SELECT_CARD / SELECT_PLAY_MODE only); `subsample` is the fraction of those searched."""

    sims: int = 64
    determinize: bool = False
    node_filter: str = "all"
    subsample: float = 1.0
    backend: str = "cpp"
    fpu: float = 0.0


@dataclass(frozen=True)
class Bot:
    """A rule-based bot with no network: `heuristic` or `random`."""

    name: str = "heuristic"


Inference = Union[Policy, Gumbel, Rollout, Search, Bot]

KINDS: Final[Dict[str, type]] = {"policy": Policy, "gumbel": Gumbel, "rollout": Rollout,
                                 "search": Search, "bot": Bot}
_KIND_OF: Final[Dict[type, str]] = {cls: kind for kind, cls in KINDS.items()}
#: A bot's player id, and the load_agent spec it plays as.
BOTS: Final[Dict[str, Tuple[str, str]]] = {"heuristic": ("HeuristicBot", "heuristic"),
                                           "random": ("RandomBot", "random")}
_NODE_FILTERS: Final = ("all", "card")
_BACKENDS: Final = ("cpp", "python")
#: ai.search.rollout_root's STOPS and LEAF_SIDES, without importing it (torch-free here).
_STOPS: Final = ("boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal")
_LEAVES: Final = ("mover", "decider", "own")
#: Fields added to a kind after ids using it were published, with the value those ids meant. They are
#: written into an id or a stored spec only when they differ from it.
LATE_FIELDS: Final[Dict[str, Dict[str, Union[str, int]]]] = {
    "rollout": {"stop": "boundaries", "leaf": "mover", "leaf_net": "", "chance": "shared", "belief": 0,
                "aligned": False, "box4": False, "race": 0}}
_CHANCES: Final = ("shared", "keyed", "keyed2")
#: Characters a leaf_net name may not hold: the id grammar's own separators.
_LEAF_NET_BAD: Final = frozenset("(),~:= ")
#: ai.search.rollout_root.parse_rule's grammar, without importing it (torch-free here).
_RULE_RE = re.compile(r"^(argmax|z(\d+(\.\d+)?)?|kl(\d+(\.\d+)?)?)$")
_ID_RE = re.compile(r"^(?P<network>[^~]+?)(?:~(?P<kind>[a-z]+)(?:\((?P<args>[^()]*)\))?)?$")


class PlayerSpecError(ValueError):
    pass


def kind_of(spec: Inference) -> str:
    return _KIND_OF[type(spec)]


def _fmt(value: Union[int, float, str, bool]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _check(spec: Inference) -> None:
    """Refuse a spec the agent loader would not play as written."""
    if isinstance(spec, Policy) and spec.temperature < 0:
        raise PlayerSpecError(f"temperature must be >= 0, got {spec.temperature}")
    if isinstance(spec, (Gumbel, Search)) and spec.sims < 1:
        raise PlayerSpecError(f"sims must be >= 1, got {spec.sims}")
    if isinstance(spec, (Gumbel, Rollout)) and spec.k < 1:
        raise PlayerSpecError(f"k must be >= 1, got {spec.k}")
    if isinstance(spec, Rollout):
        if spec.worlds < 1:
            raise PlayerSpecError(f"worlds must be >= 1, got {spec.worlds}")
        if spec.horizon < 0:
            raise PlayerSpecError(f"horizon must be >= 0, got {spec.horizon}")
        if not _RULE_RE.match(spec.rule):
            raise PlayerSpecError(f"rule must be argmax, z<x> or kl<t>, got {spec.rule!r}")
        if spec.stop not in _STOPS:
            raise PlayerSpecError(f"stop must be one of {_STOPS}, got {spec.stop!r}")
        if spec.leaf not in _LEAVES:
            raise PlayerSpecError(f"leaf must be one of {_LEAVES}, got {spec.leaf!r}")
        if spec.chance not in _CHANCES:
            raise PlayerSpecError(f"chance must be one of {_CHANCES}, got {spec.chance!r}")
        if spec.belief < 0:
            raise PlayerSpecError(f"belief must be >= 0 hands, got {spec.belief}")
        if spec.race and not 0 < spec.race < spec.worlds:
            raise PlayerSpecError(f"race must be between 0 and worlds ({spec.worlds}), got {spec.race}")
        if set(spec.leaf_net) & _LEAF_NET_BAD:
            raise PlayerSpecError(f"leaf_net may not contain any of {''.join(sorted(_LEAF_NET_BAD))!r} "
                                  f"(register the evaluator under a plain name): {spec.leaf_net!r}")
    if isinstance(spec, Search):
        if spec.node_filter not in _NODE_FILTERS:
            raise PlayerSpecError(f"node_filter must be one of {_NODE_FILTERS}, got {spec.node_filter!r}")
        if spec.backend not in _BACKENDS:
            raise PlayerSpecError(f"backend must be one of {_BACKENDS}, got {spec.backend!r}")
        if not 0 < spec.subsample <= 1:
            raise PlayerSpecError(f"subsample must be in (0, 1], got {spec.subsample}")
    if isinstance(spec, Bot) and spec.name not in BOTS:
        raise PlayerSpecError(f"bot must be one of {sorted(BOTS)}, got {spec.name!r}")


def player_id(network: Optional[str], spec: Inference) -> str:
    """The canonical id of `spec` run on `network` (None for a bot)."""
    _check(spec)
    if isinstance(spec, Bot):
        if network is not None:
            raise PlayerSpecError("a bot has no network")
        return BOTS[spec.name][0]
    if not network:
        raise PlayerSpecError(f"a {kind_of(spec)} player needs a network")
    if "~" in network or ":" in network:
        raise PlayerSpecError(f"a network name may not contain '~' or ':': {network!r}")
    if spec == Policy():
        return network
    late = LATE_FIELDS.get(kind_of(spec), {})
    fields = ",".join(f"{f.name}={_fmt(getattr(spec, f.name))}" for f in dataclasses.fields(spec)
                      if f.name not in late or getattr(spec, f.name) != late[f.name])
    return f"{network}~{kind_of(spec)}({fields})"


def _coerce(cls: type, name: str, raw: str) -> Union[int, float, str, bool]:
    default = getattr(cls(), name)
    if isinstance(default, bool):
        if raw not in ("true", "false"):
            raise PlayerSpecError(f"{name} must be true or false, got {raw!r}")
        return raw == "true"
    try:
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
    except ValueError:
        raise PlayerSpecError(f"{name}: not a number: {raw!r}") from None
    return raw


def spec_from_dict(d: Mapping[str, object]) -> Inference:
    """The spec stored in leaderboard/players.json: {"kind": ..., <field>: <value>, ...}. Every
    field is required, so a stored spec never silently takes up a changed default -- except a
    `LATE_FIELDS` one, which a spec stored before it existed lacks and which means its old value."""
    kind = d.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise PlayerSpecError(f"unknown inference kind {kind!r}; one of {sorted(KINDS)}")
    cls = KINDS[kind]
    names = [f.name for f in dataclasses.fields(cls)]
    late = LATE_FIELDS.get(kind, {})
    d = {**late, **d}
    extra = sorted(set(d) - set(names) - {"kind"})
    missing = [n for n in names if n not in d]
    if extra or missing:
        raise PlayerSpecError(f"{kind} spec: unknown fields {extra}, missing fields {missing}")
    values: Dict[str, Union[int, float, str, bool]] = {}
    for n in names:
        v = d[n]
        default = getattr(cls(), n)
        if isinstance(default, bool) != isinstance(v, bool) or (
                isinstance(default, (int, float)) and not isinstance(v, (int, float))) or (
                isinstance(default, str) and not isinstance(v, str)) or (
                # An integer field takes an integer only: 4.0 or 4.5 for `k` would be stored, run
                # (or truncated) and named differently from the id that registered it.
                isinstance(default, int) and not isinstance(default, bool) and not isinstance(v, int)):
            raise PlayerSpecError(f"{kind}.{n}: expected {type(default).__name__}, got {v!r}")
        if isinstance(v, bool) or isinstance(v, str):
            values[n] = v
        elif isinstance(v, (int, float)):
            values[n] = float(v) if isinstance(default, float) else v
    spec = cls(**values)
    _check(spec)
    return spec


def spec_to_dict(spec: Inference) -> Dict[str, object]:
    late = LATE_FIELDS.get(kind_of(spec), {})
    return {"kind": kind_of(spec), **{k: v for k, v in dataclasses.asdict(spec).items()
                                      if k not in late or v != late[k]}}


#: The policy's argmax: the temperature at or under which a Policy is greedy (the tournament's cut-off).
GREEDY_TEMPERATURE: Final = 0.05


def workbench_search(spec: Inference) -> Optional[str]:
    """How the web workbench plays this player, as its `search=` value
    (web/ui/src/analysis/rollout_spec.ts): "policy" for the greedy policy, the rollout root with every
    field spelled out -- the workbench's defaults for a field left out are not this module's -- and
    None for what it cannot run: a sampled policy, a Gumbel or PUCT search, a root whose leaves another
    network reads."""
    if isinstance(spec, Policy):
        return "policy" if spec.temperature <= GREEDY_TEMPERATURE else None
    if not isinstance(spec, Rollout) or spec.leaf_net:
        return None
    return (f"rollout:{spec.k}:{spec.worlds}:{spec.horizon}:{spec.rule}:{spec.stop}:{spec.leaf}"
            + (f":{spec.chance}" if spec.chance != "shared" else "") + (":aligned" if spec.aligned else "")
            + (":box4" if spec.box4 else "") + (f":belief{spec.belief}" if spec.belief else "")
            + (f":race{spec.race}" if spec.race else ""))


def parse_player_id(pid: str) -> Tuple[Optional[str], Inference]:
    """(network, spec) of a player id; the inverse of `player_id`. A search spec's fields may be
    given in any order or left out (taking today's defaults) -- the canonical id spells them all,
    which is what `player_id` of the result returns."""
    for name, (bot_id, _) in BOTS.items():
        if pid == bot_id:
            return None, Bot(name)
    m = _ID_RE.match(pid.strip())
    if not m:
        raise PlayerSpecError(f"not a player id: {pid!r}")
    network, kind, args = m.group("network"), m.group("kind"), m.group("args")
    if kind is None:
        return network, Policy()
    if kind not in KINDS or kind == "bot":
        raise PlayerSpecError(f"unknown inference kind {kind!r} in {pid!r}")
    cls = KINDS[kind]
    names = {f.name for f in dataclasses.fields(cls)}
    values: Dict[str, Union[int, float, str, bool]] = {}
    for part in (args or "").split(","):
        if not part.strip():
            continue
        key, sep, raw = part.partition("=")
        key = key.strip()
        if not sep or key not in names:
            raise PlayerSpecError(f"{kind}: unknown field {key!r} in {pid!r}; fields {sorted(names)}")
        values[key] = _coerce(cls, key, raw.strip())
    spec = cls(**values)
    _check(spec)
    return network, spec


def load_spec(spec: Inference, checkpoint: Optional[str], leaf_checkpoint: Optional[str] = None) -> str:
    """The `tools.lib.player_agent.load_agent` string that plays `spec` on `checkpoint`. A policy
    pins its own temperature, so the tournament's shared --temperature cannot change it. A rollout
    root with a `leaf_net` needs that network's file as `leaf_checkpoint`."""
    _check(spec)
    if isinstance(spec, Bot):
        return BOTS[spec.name][1]
    if not checkpoint:
        raise PlayerSpecError(f"a {kind_of(spec)} player needs a checkpoint file")
    if ":" in checkpoint:
        raise PlayerSpecError(f"agent specs are ':'-separated, so a checkpoint path may not contain one: {checkpoint!r}")
    if isinstance(spec, Policy):
        return f"temp:{_fmt(spec.temperature)}:{checkpoint}"
    if isinstance(spec, Gumbel):
        return f"gumbel:{checkpoint}:{spec.sims}:{spec.k}:{_fmt(spec.fpu)}"
    if isinstance(spec, Rollout):
        tail = (f":{spec.stop}" if spec.stop != "boundaries" else "") + (f":{spec.leaf}" if spec.leaf != "mover" else "")
        tail += f":{spec.chance}" if spec.chance != "shared" else ""
        tail += ":aligned" if spec.aligned else ""
        tail += ":box4" if spec.box4 else ""
        tail += f":belief{spec.belief}" if spec.belief else ""
        tail += f":race{spec.race}" if spec.race else ""
        if spec.leaf_net:
            if not leaf_checkpoint:
                raise PlayerSpecError(f"leaf_net={spec.leaf_net} needs its checkpoint file")
            if ":" in leaf_checkpoint:
                raise PlayerSpecError(f"a leaf checkpoint path may not contain ':': {leaf_checkpoint!r}")
            tail += f":{leaf_checkpoint}"
        elif leaf_checkpoint:
            raise PlayerSpecError("a leaf checkpoint was given for a spec without leaf_net")
        return f"rollout:{checkpoint}:{spec.k}:{spec.worlds}:{spec.horizon}:{spec.rule}{tail}"
    return (f"search:{checkpoint}:{spec.sims}:{'determinize' if spec.determinize else 'none'}:"
            f"{spec.node_filter}:{_fmt(spec.subsample)}:{spec.backend}:{_fmt(spec.fpu)}")
