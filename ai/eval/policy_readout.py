"""What the policy believed at one decision node, recorded next to the move it made.

A replay says what was played. It does not say whether the policy nearly avoided the blunder the
counter charged it with: `p_chosen = 0.02` on a mistake the mode would have dodged is a sampling
problem, `p_chosen = 0.81` is a policy problem, and those have different fixes. Both numbers are
already inside the forward pass the generator makes -- `sample_action` computes the masked logits
and both value heads and throws all but the sampled index away -- so the readout costs nothing
beyond formatting.

**The probabilities are the model's own distribution, at temperature 1.** Self-play samples at
0.3, so the distribution that was sampled from is not the one the policy believes; both are
recorded (`p_chosen` and `p_chosen_sampled`) because they answer different questions.

This does NOT replace `sample_action`, which is on the training hot path. It repeats its three
lines of sampling, and `tests/training/test_policy_readout.py` pins the two together under a
fixed torch seed: if they ever diverge, a traced replay would be a *different game* from the
untraced one with the same seed, and every replay-vs-training comparison would quietly change
meaning.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ts_engine as ts
from ai.eval.replay_critic import evaluate
from bindings.action_encoder import ActionEncoder
from web.server.replay_types import PolicyChoiceDict, ReplayCriticDict, ReplayPolicyDict

#: How many legal actions to list in `top`, by descending probability. **0 means all of them**,
#: which is the default: the workbench puts each probability on the card, mode button or country
#: it belongs to, so there is no list to keep short, and a distribution missing its tail cannot
#: be read off the board. The widest node is an 84-way influence placement, which costs about
#: 4.5 KB against a step that already carries a ~33 KB state snapshot.
TRACE_TOP_K: int = 0

#: Probabilities below this are not listed individually, when a cap is in force at all. The
#: chosen action is exempt: a 0.001 choice is exactly the case worth looking at.
TRACE_P_FLOOR: float = 0.0


def read_policy(
    model: Any,
    obs: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    deterministic: bool = False,
    state: Optional[ts.GameState] = None,
    top_k: int = TRACE_TOP_K,
    p_floor: float = TRACE_P_FLOOR,
    source: str = "policy",
    include: Optional[int] = None,
) -> Tuple[int, ReplayPolicyDict]:
    """Sample an action and describe the distribution it came from.

    `obs` and `mask` are single-row batches, as the callers already build them. `state` is the
    pre-action position, used only to name actions; without it the names are omitted.
    `top_k=0` (the default) lists every legal action; a non-zero cap keeps that many by
    descending probability, and `include` then forces an action into the listing whatever its
    probability -- the annotator needs the probability of the move the replay recorded, which
    under a cap is often the one that would fall off the end.

    Returns the flat action index and the trace block for the replay step.
    """
    if obs.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError(
            f"read_policy reads one node at a time; got batch {obs.shape[0]}/{mask.shape[0]}")

    with torch.no_grad():
        masked_logits, v_win, v_vp = model(obs, mask)

        # Identical to sample_action, and in the same order, so the torch RNG is consumed the
        # same way and a traced game is the same game.
        if deterministic:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            scaled_logits = masked_logits / max(temperature, 1e-4)
            dist = torch.distributions.Categorical(logits=scaled_logits)
            actions = dist.sample()
        chosen = int(actions.item())

        probs = F.softmax(masked_logits, dim=-1)[0]
        log_probs = F.log_softmax(masked_logits, dim=-1)[0]
        entropy = float(-(probs * log_probs).sum().item())
        argmax_idx = int(torch.argmax(masked_logits, dim=-1).item())
        if deterministic:
            p_chosen_sampled = 1.0
        else:
            p_chosen_sampled = float(
                F.softmax(masked_logits / max(temperature, 1e-4), dim=-1)[0, chosen].item())

    legal = torch.nonzero(mask[0] > 0, as_tuple=False).flatten().tolist()
    # An empty mask cannot happen -- the engine guarantees CONFIRM_DONE -- but the chosen action
    # must be describable either way, so fall back to it rather than producing an empty trace.
    if not legal:
        legal = [chosen]

    ranked = sorted(legal, key=lambda i: float(probs[i].item()), reverse=True)
    if top_k:
        kept = [i for i in ranked if float(probs[i].item()) >= p_floor][:top_k]
        for must in (chosen, include):
            if must is not None and must not in kept:
                kept.append(must)
    else:
        kept = ranked

    top: List[PolicyChoiceDict] = []
    for i in kept:
        entry: PolicyChoiceDict = {"idx": int(i), "p": round(float(probs[i].item()), 6)}
        if state is not None:
            entry["name"] = ActionEncoder.get_action_name(state, int(i))
        top.append(entry)
    top.sort(key=lambda e: e["p"], reverse=True)

    listed = sum(float(probs[i].item()) for i in kept)
    readout: ReplayPolicyDict = {
        "source": source,
        "temperature": float(temperature),
        "n_legal": len(legal),
        "chosen_idx": chosen,
        "p_chosen": round(float(probs[chosen].item()), 6),
        "p_chosen_sampled": round(p_chosen_sampled, 6),
        "argmax_idx": argmax_idx,
        "p_max": round(float(probs[argmax_idx].item()), 6),
        "entropy": round(entropy, 6),
        "top": top,
        # Clamped at zero: the listed probabilities are rounded, so a distribution fully covered
        # by `top` can otherwise land on -1e-7 and read as a missing mass that does not exist.
        "p_tail": round(max(0.0, 1.0 - listed), 6),
        "v_win": round(float(v_win.flatten()[0].item()), 6),
        "v_vp": round(float(v_vp.flatten()[0].item()), 6),
    }
    return chosen, readout


def name_actions(readout: ReplayPolicyDict, state: ts.GameState) -> None:
    """Fill in the human-readable name of every listed action, in place.

    A bot driven by the JSON view has no `GameState` and cannot name a flat index; its caller
    does. Separating the two keeps `read_policy` usable from either side.
    """
    for entry in readout.get("top", []):
        if "name" not in entry:
            entry["name"] = ActionEncoder.get_action_name(state, int(entry["idx"]))


def read_critic(model: Any, state: ts.GameState, at: str = "after") -> ReplayCriticDict:
    """Both value heads from both perspectives, on one position.

    Reads through `replay_critic.evaluate` rather than repeating it, so the inline trace and the
    post-hoc annotator cannot disagree about what "the critic said" means. Both perspectives,
    because `v_win_us + v_win_ussr` is pure model error for a zero-sum critic and is the only
    asymmetry instrument that costs a single extra forward.
    """
    v = evaluate(model, state)
    return {
        "v_win_us": round(v["v_win_us"], 6),
        "v_win_ussr": round(v["v_win_ussr"], 6),
        "v_vp_us": round(v["v_vp_us"], 6),
        "v_vp_ussr": round(v["v_vp_ussr"], 6),
        "win_residual": round(v["win_residual"], 6),
        "vp_residual": round(v["vp_residual"], 6),
        "at": at,
    }


def unasked_readout(chosen: int, state: Optional[ts.GameState] = None,
                    source: str = "forced", n_legal: int = 1) -> ReplayPolicyDict:
    """The trace for a step the policy did not choose.

    Two cases, and they are not the same: a *forced* step has one legal action and the settle
    policy played it, so certainty is a fact about the node and the probabilities are exact
    without a forward pass; a *scripted* step (a forced opening) overrode a real choice, so
    there is no distribution to report and none is invented. Either way the step is marked, so
    a reader can tell "the policy was certain" from "the policy was never asked".
    """
    entry: PolicyChoiceDict = {"idx": int(chosen), "p": 1.0}
    if state is not None:
        entry["name"] = ActionEncoder.get_action_name(state, int(chosen))
    out: ReplayPolicyDict = {
        "source": source,
        "n_legal": int(n_legal),
        "chosen_idx": int(chosen),
    }
    if n_legal <= 1:
        out.update({
            "p_chosen": 1.0,
            "p_chosen_sampled": 1.0,
            "argmax_idx": int(chosen),
            "p_max": 1.0,
            "entropy": 0.0,
            "top": [entry],
            "p_tail": 0.0,
        })
    return out
