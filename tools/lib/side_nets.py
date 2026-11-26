"""One side's network of a --ladder-side-nets (A39) state, taken out alone, and put back (owner, 2026-11-26).

An A39 LadderNet is two complete networks: its own modules play the US, `ussr_net.*` the USSR. A seat-locked run
(an exploiter of one seat) trains one of them only, so it is trained as that network alone -- a plain LadderNet with
the same structural flags (A8 for T1's), nothing of the other side built, run or kept in the optimizer -- and put back
beside the base's other side for anything that plays both seats (reads, the leaderboard).

`extract` takes a plain checkpoint (a state dict: a snapshot or an SWA) or a resume state. From a resume state it
keeps everything a branch needs: the side's weights, its pi_ref, its Adam moments mapped by parameter name, the step
counters and every RNG stream; it records `side_only`, which makes every launch resuming it seat-locked
(generic_trainer.check_side_only). `combine` is the inverse for plain checkpoints: the other side's keys are the
base's, bit for bit."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch

#: The key prefix of the USSR side's network in an A39 state dict; the US side's keys have none.
USSR_PREFIX = "ussr_net."
SIDES = ("us", "ussr")

#: Resume-state entries that hold no weights and are carried as they are.
_CARRIED = ("total_env_steps", "total_iterations", "iteration", "elapsed_seconds", "torch_rng_state",
            "torch_cuda_rng_state", "numpy_rng_state", "python_rng_state", "steps_since_ref_update",
            "env_game_rng_state", "seed", "seeds", "wolf_sp_ussr", "adv_std_ema", "adv_std_ema_steps", "ent_coef_seat",
            "setup_ent_coef")
#: The entries split by side.
_WEIGHTS = ("model_state_dict", "reference_state_dict", "optimizer_state_dict", "optimizer_param_names", "side_only",
            "extracted_from")
#: Entries a single-side branch cannot carry: refused rather than dropped when present.
_REFUSED = ("opponent_pool", "ema_state_dict", "active_search", "searched_opponent", "head_init_state")


def is_side_nets(sd: Mapping[str, Any]) -> bool:
    return any(k.startswith(USSR_PREFIX) for k in sd)


def side_state(sd: Mapping[str, Any], side: str) -> Dict[str, Any]:
    """The `side`'s network out of an A39 state dict, under a plain LadderNet's keys, in the state's order."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if not is_side_nets(sd):
        raise ValueError("not a --ladder-side-nets state: it has no ussr_net.* keys")
    if side == "us":
        return {k: v for k, v in sd.items() if not k.startswith(USSR_PREFIX)}
    return {k[len(USSR_PREFIX):]: v for k, v in sd.items() if k.startswith(USSR_PREFIX)}


def _side_name(name: str, side: str) -> str | None:
    """A parameter name of the A39 network as the side's network names it, or None for the other side's."""
    if side == "us":
        return None if name.startswith(USSR_PREFIX) else name
    return name[len(USSR_PREFIX):] if name.startswith(USSR_PREFIX) else None


def side_optimizer_state(opt: Mapping[str, Any], names: List[str], model_sd: Mapping[str, Any],
                         side: str) -> tuple[Dict[str, Any], List[str]]:
    """The side's Adam moments, one group indexed in the side network's parameter order, and those parameters' names.

    `names` is the saved optimizer's parameter behind each index (`optimizer_param_names`); the groups' hyperparameters
    are the launch's on load (generic_trainer._load_optimizer_state maps one group onto its groups)."""
    flat = [int(i) for g in opt["param_groups"] for i in g["params"]]
    if len(flat) != len(names):
        raise ValueError("the optimizer's parameter names do not cover its groups")
    by_name = {n: flat[k] for k, n in enumerate(names)}
    order = [k for k in model_sd if k in by_name and _side_name(k, side) is not None]
    state = opt["state"]
    new_state = {i: state[by_name[k]] for i, k in enumerate(order) if by_name[k] in state}
    group = {key: v for key, v in opt["param_groups"][0].items() if key not in ("params", "lr_mult")}
    group["params"] = list(range(len(order)))
    return {"state": new_state, "param_groups": [group]}, [str(_side_name(k, side)) for k in order]


def extract(blob: Mapping[str, Any], side: str, source: str = "") -> Dict[str, Any]:
    """`side`'s network out of an A39 checkpoint (a state dict) or resume state (see the module docstring)."""
    if "model_state_dict" not in blob:
        return side_state(blob, side)
    unknown = sorted(set(blob) - set(_CARRIED) - set(_REFUSED) - set(_WEIGHTS))
    if unknown:
        raise ValueError(f"the resume state has entries extract does not know how to split: {unknown}")
    bad = [k for k in _REFUSED if blob.get(k) is not None]
    if bad:
        raise ValueError(f"the resume state carries {', '.join(bad)}, which a single-side branch cannot take")
    if blob.get("side_only") is not None:
        raise ValueError(f"the resume state is already the {blob['side_only']} network alone")
    names = blob.get("optimizer_param_names")
    if names is None:
        raise ValueError("the resume state has no optimizer_param_names; its moments cannot be mapped by name")
    sd = blob["model_state_dict"]
    opt, opt_names = side_optimizer_state(blob["optimizer_state_dict"], list(names), sd, side)
    out: Dict[str, Any] = {k: blob[k] for k in _CARRIED if k in blob}
    out.update({
        "model_state_dict": side_state(sd, side),
        "reference_state_dict": side_state(blob["reference_state_dict"], side),
        "optimizer_state_dict": opt,
        "optimizer_param_names": opt_names,
        "opponent_pool": None,
        "ema_state_dict": None,
        "side_only": side,
        "extracted_from": source,
    })
    return out


def combine(side_sd: Mapping[str, Any], base_sd: Mapping[str, Any], side: str) -> Dict[str, Any]:
    """An A39 state dict: `side`'s network from `side_sd` (a plain LadderNet state), the other side's from `base_sd`
    (an A39 state) exactly as it is there, in the base's key order."""
    if is_side_nets(side_sd):
        raise ValueError("the side network is itself a --ladder-side-nets state")
    if "model_state_dict" in side_sd or "model_state_dict" in base_sd:
        raise ValueError("combine takes plain checkpoints (state dicts), not resume states")
    own = side_state(base_sd, side)
    if set(own) != set(side_sd):
        raise ValueError(f"the side network's keys are not the base's {side} network's: "
                         f"{sorted(set(own) ^ set(side_sd))[:6]}")
    for k, v in own.items():
        w = side_sd[k]
        if isinstance(v, torch.Tensor) and (not isinstance(w, torch.Tensor) or tuple(v.shape) != tuple(w.shape)):
            raise ValueError(f"{k}: the side network's shape {getattr(w, 'shape', None)} is not the base's "
                             f"{tuple(v.shape)}")
    if "obs_features" in own and int(own["obs_features"]) != int(side_sd["obs_features"]):
        raise ValueError(f"the side network plays feature bits {int(side_sd['obs_features'])}, the base "
                         f"{int(own['obs_features'])}")
    out: Dict[str, Any] = {}
    for k, v in base_sd.items():
        name = _side_name(k, side)
        out[k] = side_sd[name] if name is not None else v
    return out
