"""Does the critic price battlegrounds -- and access to them -- at all?

`research/experiments.md` §4 records that battlegrounds sit empty in late positions and that the
count plateaus from turn 8: the agent stops contesting the map once its early cards are spent. The
loop that would explain it is that the agent never holds battlegrounds, so it never experiences a
scoring card paying out on them, so the critic never learns they are worth anything, so the policy
has no gradient toward taking them.

That story has two halves and they need different fixes -- a critic that cannot price the board, or
a policy that can but never explores there. This measures the first half, the same way §4.8
measured whether the policy conditions on the trap bit: perturb the board in one controlled way and
see how far `v_win` moves.

**What counts as valuable is not only control.** Access matters and is cheaper to buy: influence in
a battleground the opponent also stands in, or in a country adjacent to one, is what makes a later
contest possible at all. Early USSR influence in the South America battlegrounds is worth far more
than turn-2 VP arithmetic suggests, and holding Thailand when the opponent can reach it is a
different proposition from holding it when they cannot. So the perturbations here separate:

* **control** -- enough influence to hold the battleground outright;
* **presence** -- influence in it without reaching control, which is access to a future fight;
* **adjacent access** -- influence in a neighbour of a battleground, the battleground untouched;
* **plain influence** -- the same amount in a non-battleground with no battleground neighbour.
  That last one is the control condition: any critic prefers more influence to less, so the
  question is only whether it prefers it *where the game pays for it*.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

EUROPE, ASIA, MIDDLE_EAST, AFRICA, CENTRAL_AMERICA, SOUTH_AMERICA = range(6)

REGION_NAMES = {EUROPE: "Europe", ASIA: "Asia", MIDDLE_EAST: "Middle East",
                AFRICA: "Africa", CENTRAL_AMERICA: "Central America",
                SOUTH_AMERICA: "South America"}


def battlegrounds() -> List[int]:
    return [c for c in range(84) if bool(ts.MapData.get_country_info(c)["battleground"])]


def neighbours(cid: int) -> List[int]:
    return [int(n) for n in ts.MapData.get_country_info(cid)["neighbors"] if int(n) < 84]


def adjacent_to_battleground() -> List[int]:
    bgs = set(battlegrounds())
    return [c for c in range(84)
            if c not in bgs and any(n in bgs for n in neighbours(c))]


def isolated_non_battlegrounds() -> List[int]:
    bgs = set(battlegrounds())
    return [c for c in range(84)
            if c not in bgs and not any(n in bgs for n in neighbours(c))]


def region_of(cid: int) -> int:
    return int(ts.MapData.get_country_info(cid)["region"])


@dataclass
class Perturbation:
    """A named edit to the board, and the value change it produced."""

    name: str
    deltas: List[float] = field(default_factory=list)

    def mean(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else float("nan")

    def stderr(self) -> float:
        if len(self.deltas) < 2:
            return float("nan")
        return float(np.std(self.deltas, ddof=1) / np.sqrt(len(self.deltas)))


def value_of(model: Any, state: ts.GameState, mover: ts.Player, device: Any) -> float:
    """`v_win` from the mover's point of view."""
    import torch

    obs = np.asarray(ts.extract_observation(state, mover), dtype=np.float32)[None, :]
    mask = np.asarray(ts.get_flat_action_mask(state), dtype=np.uint8)[None, :]
    with torch.no_grad():
        _logits, v_win, _v_vp = model(torch.from_numpy(obs).to(device),
                                      torch.from_numpy(mask).to(device))
    return float(v_win.squeeze().item())


def _add_influence(state: ts.GameState, cid: int, player: ts.Player, amount: int) -> None:
    country = state.get_country(cid)
    us, ussr = int(country.us_influence), int(country.ussr_influence)
    if player == ts.Player.US:
        us = min(20, us + amount)
    else:
        ussr = min(20, ussr + amount)
    state.set_country(cid, us, ussr)


def _clear(state: ts.GameState, cid: int, player: ts.Player) -> None:
    country = state.get_country(cid)
    us, ussr = int(country.us_influence), int(country.ussr_influence)
    state.set_country(cid, 0 if player == ts.Player.US else us,
                      0 if player == ts.Player.USSR else ussr)


def _controls(state: ts.GameState, cid: int, player: ts.Player) -> bool:
    return bool(ts.Scoring.is_controlled_by(state, cid, player))


def _control_gap(state: ts.GameState, cid: int, player: ts.Player) -> int:
    """Influence the player still needs in `cid` to control it."""
    country = state.get_country(cid)
    mine = int(country.us_influence if player == ts.Player.US else country.ussr_influence)
    theirs = int(country.ussr_influence if player == ts.Player.US else country.us_influence)
    return theirs + int(ts.MapData.get_country_info(cid)["stability"]) - mine


def _rotate(pool: Sequence[int], k: int) -> List[int]:
    """The pool starting from a different place each time.

    Every arm probes one country per position. Without the rotation that would be the same
    country in every position of a turn bucket, and the arm would measure Angola rather than
    battlegrounds.
    """
    if not pool:
        return []
    i = k % len(pool)
    return list(pool[i:]) + list(pool[:i])


def collect_positions(model: Any, device: Any, num_envs: int = 128, base_seed: int = 424242,
                      max_iters: int = 3000, per_turn: int = 40,
                      turns: Sequence[int] = (2, 3, 5, 8),
                      ) -> Dict[int, List[Tuple[ts.GameState, ts.Player]]]:
    """Self-play positions, bucketed by turn, so early and late can be told apart."""
    import torch

    runner = ts.VectorizedBatchRunner(num_envs, base_seed)
    out: Dict[int, List[Tuple[ts.GameState, ts.Player]]] = defaultdict(list)
    wanted = set(int(t) for t in turns)
    model.eval()

    for _ in range(max_iters):
        if all(len(out[t]) >= per_turn for t in wanted):
            break
        terminals = runner.get_terminals()
        if all(terminals):
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            choice = model(torch.from_numpy(obs).to(device),
                           torch.from_numpy(masks).to(device))[0].argmax(dim=-1).cpu().numpy()

        for i in range(num_envs):
            if terminals[i]:
                continue
            state = runner.get_state(i)
            turn = int(state.turn)
            if turn not in wanted or len(out[turn]) >= per_turn:
                continue
            mover = state.ctx().decision_player
            if mover == ts.Player.NONE:
                mover = state.phasing_player
            if mover == ts.Player.NONE:
                continue
            out[turn].append((state.clone(), mover))

        runner.step_flat_all([int(c) for c in choice], auto_advance=True)

    return dict(out)


CONTROL = "control a battleground"
PRESENCE = "presence in a battleground"
ADJACENT = "access: adjacent to a battleground"
PLAIN = "plain influence, no battleground near"
LOST = "opponent controls a battleground"


def measure(model: Any, device: Any,
            positions: Dict[int, List[Tuple[ts.GameState, ts.Player]]],
            amount: int = 2) -> Dict[int, Dict[str, Perturbation]]:
    """Value change from each kind of board edit, per turn bucket.

    Every edit adds Influence for the mover; only *where* differs, and the plain-influence arm
    adds exactly the same amount somewhere the game does not pay for it. A critic that has merely
    learned "more Influence is better" moves the same for all of them.
    """
    bgs = battlegrounds()
    adjacent = adjacent_to_battleground()
    isolated = isolated_non_battlegrounds()
    out: Dict[int, Dict[str, Perturbation]] = {}

    for turn, entries in sorted(positions.items()):
        named = {n: Perturbation(n) for n in (CONTROL, PRESENCE, ADJACENT, PLAIN, LOST)}
        for k, (state, mover) in enumerate(entries):
            opponent = ts.Player.USSR if mover == ts.Player.US else ts.Player.US
            base = value_of(model, state, mover, device)
            rotated = _rotate(bgs, k)

            for cid in rotated:
                gap = _control_gap(state, cid, mover)
                if gap <= 0:
                    continue
                probe = state.clone()
                _add_influence(probe, cid, mover, gap)
                if _controls(probe, cid, mover):
                    named[CONTROL].deltas.append(value_of(model, probe, mover, device) - base)
                break

            for cid in rotated:
                # Skip any battleground where `amount` would tip into control -- that is the
                # other arm, and mixing them would let control's value leak into presence's.
                if _control_gap(state, cid, mover) <= amount:
                    continue
                probe = state.clone()
                _add_influence(probe, cid, mover, amount)
                named[PRESENCE].deltas.append(value_of(model, probe, mover, device) - base)
                break

            for name, pool in ((ADJACENT, adjacent), (PLAIN, isolated)):
                if not pool:
                    continue
                probe = state.clone()
                _add_influence(probe, _rotate(pool, k)[0], mover, amount)
                named[name].deltas.append(value_of(model, probe, mover, device) - base)

            for cid in rotated:
                gap = _control_gap(state, cid, opponent)
                if gap <= 0:
                    continue
                probe = state.clone()
                _add_influence(probe, cid, opponent, gap)
                if _controls(probe, cid, opponent):
                    named[LOST].deltas.append(value_of(model, probe, mover, device) - base)
                break

        out[turn] = named
    return out


def measure_region(model: Any, device: Any,
                   positions: Dict[int, List[Tuple[ts.GameState, ts.Player]]],
                   region: int, side: Optional[ts.Player] = None,
                   amount: int = 2) -> Dict[int, Perturbation]:
    """Value of a foothold in one region's battlegrounds, per turn and per Influence point.

    Early USSR presence in the South America battlegrounds is the case that motivates this: worth
    far more than the turn-2 VP arithmetic says, because it decides who can contest the region for
    the rest of the game. If the critic prices it flat, the policy has no reason to go there.

    `side` restricts to positions where that player is the mover, since the claim is specifically
    about the USSR.
    """
    targets = [c for c in battlegrounds() if region_of(c) == region]
    label = f"{REGION_NAMES.get(region, str(region))} battlegrounds"
    out: Dict[int, Perturbation] = {}
    for turn, entries in sorted(positions.items()):
        pert = Perturbation(label)
        for state, mover in entries:
            if side is not None and mover != side:
                continue
            base = value_of(model, state, mover, device)
            probe = state.clone()
            for cid in targets:
                _add_influence(probe, cid, mover, amount)
            # Per Influence point, so it is comparable with the single-country arms above.
            pert.deltas.append(
                (value_of(model, probe, mover, device) - base) / (amount * len(targets)))
        out[turn] = pert
    return out


def _access_board(state: ts.GameState, cid: int, stability: int, nbrs: Sequence[int],
                  owner: ts.Player, other_has_access: bool) -> ts.GameState:
    """`cid` held by `owner`, the other side either standing next to it or shut out entirely."""
    probe = state.clone()
    probe.set_country(cid, 0, 0)
    _add_influence(probe, cid, owner, stability)
    other = ts.Player.USSR if owner == ts.Player.US else ts.Player.US
    for n in nbrs:
        _clear(probe, n, other)
    if other_has_access and nbrs:
        _add_influence(probe, nbrs[0], other, 1)
    return probe


def measure_access_asymmetry(model: Any, device: Any,
                             positions: Dict[int, List[Tuple[ts.GameState, ts.Player]]],
                             country: str = "Thailand",
                             ) -> Dict[int, Dict[str, Perturbation]]:
    """Is holding a battleground worth more when the opponent cannot reach it?

    Control of Thailand against an opponent with a foothold next door is a different asset from
    control of Thailand the opponent cannot touch, and the same holds the other way round. Both
    boards carry identical Influence in the battleground itself; only the opponent's *access*
    differs. A critic that reads control off the influence counts alone cannot tell them apart,
    and the difference measured here comes out at zero.

    Two contrasts, both from the mover's point of view:

    * **mine, opponent shut out** minus **mine, opponent adjacent** -- what uncontested control is
      worth over contested control;
    * **theirs, I have access** minus **theirs, I am shut out** -- what a way back in is worth once
      the battleground is already lost.
    """
    cid = int(ts.MapData.get_country_by_name(country))
    stability = int(ts.MapData.get_country_info(cid)["stability"])
    nbrs = neighbours(cid)
    mine_shut = f"{country} mine, opponent shut out"
    mine_open = f"{country} mine, opponent adjacent"
    theirs_open = f"{country} theirs, I have access"
    theirs_shut = f"{country} theirs, I am shut out"
    out: Dict[int, Dict[str, Perturbation]] = {}

    for turn, entries in sorted(positions.items()):
        named = {k: Perturbation(k) for k in (mine_shut, mine_open, theirs_open, theirs_shut)}
        for state, mover in entries:
            opponent = ts.Player.USSR if mover == ts.Player.US else ts.Player.US
            base = value_of(model, state, mover, device)
            for key, owner, access in ((mine_shut, mover, False), (mine_open, mover, True),
                                       (theirs_open, opponent, True),
                                       (theirs_shut, opponent, False)):
                probe = _access_board(state, cid, stability, nbrs, owner, access)
                named[key].deltas.append(value_of(model, probe, mover, device) - base)
        out[turn] = named
    return out


ACCESS_CONTRASTS = (("uncontested control is worth this much more",
                     "{c} mine, opponent shut out", "{c} mine, opponent adjacent"),
                    ("a way back in is worth this much",
                     "{c} theirs, I have access", "{c} theirs, I am shut out"))


def _far_country(cid: int) -> int:
    """A country in the same region as `cid`, not adjacent to it and not a battleground.

    Somewhere an Influence point is worth roughly what it is worth next to `cid`, minus the
    adjacency. Non-battleground so the relocated point does not buy a foothold that is valuable in
    its own right.
    """
    bgs = set(battlegrounds())
    near = set(neighbours(cid)) | {cid}
    same = [c for c in range(84)
            if region_of(c) == region_of(cid) and c not in near and c not in bgs]
    return same[0] if same else next(c for c in range(84) if c not in near)


def measure_access_matched(model: Any, device: Any,
                           positions: Dict[int, List[Tuple[ts.GameState, ts.Player]]],
                           country: str = "Thailand",
                           ) -> Dict[int, Dict[str, Perturbation]]:
    """The same access question, with the Influence held constant instead of deleted.

    `measure_access_asymmetry` builds its "shut out" board by removing the other side's Influence
    from every neighbour, so its contrast conflates *access* with the other side simply having
    less on the board -- and the critic certainly prices the latter. Here the point is moved
    rather than removed: one Influence either in a neighbour of the battleground or in a
    non-adjacent, non-battleground country of the same region. Both boards carry identical totals
    for both players, and the only difference is who can reach the battleground.

    That is the Thailand question in its strict form. Anything the contrast shows here is access.
    """
    cid = int(ts.MapData.get_country_by_name(country))
    stability = int(ts.MapData.get_country_info(cid)["stability"])
    nbrs = neighbours(cid)
    far = _far_country(cid)
    keys = [f"{country} mine, opponent 1 next door", f"{country} mine, opponent 1 elsewhere",
            f"{country} theirs, my 1 next door", f"{country} theirs, my 1 elsewhere"]
    out: Dict[int, Dict[str, Perturbation]] = {}

    for turn, entries in sorted(positions.items()):
        named = {k: Perturbation(k) for k in keys}
        for state, mover in entries:
            opponent = ts.Player.USSR if mover == ts.Player.US else ts.Player.US
            base = value_of(model, state, mover, device)
            for key, owner, adjacent in ((keys[0], mover, True), (keys[1], mover, False),
                                         (keys[2], opponent, True), (keys[3], opponent, False)):
                other = ts.Player.USSR if owner == ts.Player.US else ts.Player.US
                probe = state.clone()
                probe.set_country(cid, 0, 0)
                _add_influence(probe, cid, owner, stability)
                for n in nbrs:
                    _clear(probe, n, other)
                _clear(probe, far, other)
                _add_influence(probe, nbrs[0] if adjacent else far, other, 1)
                named[key].deltas.append(value_of(model, probe, mover, device) - base)
        out[turn] = named
    return out


MATCHED_CONTRASTS = (("the opponent's access to it costs me this much",
                      "{c} mine, opponent 1 elsewhere", "{c} mine, opponent 1 next door"),
                     ("my access to theirs is worth this much",
                      "{c} theirs, my 1 next door", "{c} theirs, my 1 elsewhere"))
