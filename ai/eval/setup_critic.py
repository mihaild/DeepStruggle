"""Which of two scripted setups does a checkpoint's critic prefer, holding the deal fixed?

`forced_setup.py` plays games out from a forced opening; this asks the cheaper question of what
the critic thinks of the board the moment setup ends. Each seed fixes the deal, both openings are
scripted from `tools.lib.openings` (so a replay made with `play_match.py --opening` is the same
board), and the state after the fifteenth placement is read through `replay_critic.evaluate` from
both perspectives. No game is played, so a sweep over every snapshot of a lineage costs seconds.

Read the values as **policy-conditional**: V^pi is the value of the position *under this
checkpoint's own continuation*. "The critic prefers setup A" means "this policy does better from
A", which is the question when asking why a policy chose a setup, and not the same as "A is the
better setup".

    PYTHONPATH=.:build/release python -m ai.eval.setup_critic \\
        --checkpoints <snap.pt> ... --openings ph_west_germany ph_no_west_germany \\
        --seeds 101 102 103 104 105 106 107 108 --output-json <out.json>
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

import numpy as np
import ts_engine as ts

from ai.eval.replay_critic import _resolve_chance, evaluate
from tools.lib.checkpoint_id import checkpoint_label
from tools.lib.openings import OPENINGS, SETUP_DECISIONS, acting_side, scripted_setup_index

#: Every card id the engine knows, used to check that two openings saw the same deal.
_CARD_IDS = range(1, 111)


def after_setup(seed: int, opening: str) -> ts.GameState:
    """The state after all fifteen placements of `opening`, dealt from `seed`.

    Raises rather than returning a partly scripted board: a placement the engine refuses, or a
    setup that is still running once the script is spent, would otherwise be measured under the
    opening's name.
    """
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    cursor = {"US": 0, "USSR": 0}
    for _ in range(SETUP_DECISIONS):
        idx = scripted_setup_index(state, acting_side(state), opening, cursor)
        if idx is None:
            raise RuntimeError(f"'{opening}' ran out of placements before setup ended (seed {seed})")
        ts.Engine.step_flat(state, idx)
        _resolve_chance(state)
    if state.current_phase == ts.Phase.SETUP:
        raise RuntimeError(f"'{opening}' spent its script but setup is still running (seed {seed})")
    return state


def deal(state: ts.GameState) -> List[int]:
    """Where every card is, as a comparable fingerprint of the deal."""
    return [int(state.get_card_location(c)) for c in _CARD_IDS]


def measure(model: Any, seeds: Sequence[int], openings: Sequence[str]) -> Dict[str, Any]:
    """Per-seed and mean critic values after each opening, for one model."""
    from bindings.ts_env import check_obs_width

    check_obs_width(model)
    model.eval()
    per: Dict[str, List[Dict[str, float]]] = {o: [] for o in openings}
    for seed in seeds:
        states = {o: after_setup(seed, o) for o in openings}
        deals = {o: deal(s) for o, s in states.items()}
        if len({tuple(d) for d in deals.values()}) != 1:
            raise RuntimeError(f"seed {seed}: the openings saw different deals, so the comparison "
                               "would not hold the hand fixed")
        for o, s in states.items():
            per[o].append({"seed": float(seed), **evaluate(model, s)})
    mean = {o: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
            for o, rows in per.items()}
    return {"per_seed": per, "mean": mean}


def main() -> None:
    from tools.lib.player_agent import NeuralAgent, resolve_device

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--openings", nargs=2, default=["ph_west_germany", "ph_no_west_germany"],
                    choices=sorted(OPENINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 109)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    a, b = args.openings
    report: Dict[str, Any] = {"openings": args.openings, "seeds": args.seeds, "checkpoints": {}}
    print(f"{'checkpoint':<24} {'v_win_us ' + a:>30} {'v_win_us ' + b:>30} {'diff':>8}"
          f" {'US-view seeds won':>18} {'USSR-view diff':>15}")
    for path in args.checkpoints:
        label = checkpoint_label(path)
        model = NeuralAgent.from_checkpoint(path, device=device).model
        r = measure(model, args.seeds, args.openings)
        report["checkpoints"][label] = {"path": path, **r}
        ma, mb = r["mean"][a], r["mean"][b]
        wins = sum(x["v_win_us"] > y["v_win_us"]
                   for x, y in zip(r["per_seed"][a], r["per_seed"][b]))
        # The USSR's view, flipped to the US side, so both columns read "good for the US".
        ussr_diff = -(ma["v_win_ussr"] - mb["v_win_ussr"])
        print(f"{label:<24} {ma['v_win_us']:>30.4f} {mb['v_win_us']:>30.4f}"
              f" {ma['v_win_us'] - mb['v_win_us']:>+8.4f} {wins:>11}/{len(args.seeds)}"
              f" {ussr_diff:>+15.4f}")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
