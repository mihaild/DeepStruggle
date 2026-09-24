"""ai/eval/decision_entropy: per-seat, per-decision-type option counts and policy spread (P23)."""

from __future__ import annotations

import math

import torch

from ai.eval.decision_entropy import Stats, probe
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


def test_stats_skip_forced_decisions_and_normalise_by_log_legal() -> None:
    s = Stats()
    s.add(1, 0.0, 1.0)
    s.add(4, math.log(4), 0.25)
    r = s.row()
    assert (r["n"], r["forced"], r["legal"]) == (2.0, 1.0, 4.0)
    assert abs(r["entropy_frac"] - 1.0) < 1e-9 and abs(r["top_p"] - 0.25) < 1e-9


def test_probe_covers_both_seats_and_the_opening_in_both_views() -> None:
    torch.manual_seed(0)
    net = create_coldwar_net_v2()
    for merged in (False, True):
        stats = probe(net, merged, envs=4, steps=40, seed=1, device="cpu")
        seats = {k[0] for k in stats}
        assert seats == {"US", "USSR"}
        assert ("US", "opening", "POINT_NODE") in stats and ("USSR", "opening", "POINT_NODE") in stats
        for st in stats.values():
            r = st.row()
            assert 0.0 <= r["entropy_frac"] <= 1.0 + 1e-6 and 0.0 < r["top_p"] <= 1.0 + 1e-6
