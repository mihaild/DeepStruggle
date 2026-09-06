"""Reordering an influence placement must not count as disagreement."""

from __future__ import annotations

from ai.eval.agreement import score_groups


def test_a_permuted_placement_agrees_completely() -> None:
    """Same countries, different order: the play is identical and should score identical."""
    groups = [[0, 1, 2]]
    human = [10, 20, 30]
    model = [30, 10, 20]
    r = score_groups(groups, human, model)
    assert r.decisions == 3
    assert r.ordered_hits == 0          # every position differs
    assert r.unordered_hits == 3        # the multiset is the same
    assert r.unordered == 100.0


def test_a_genuinely_different_country_still_costs() -> None:
    groups = [[0, 1, 2]]
    human = [10, 20, 30]
    model = [10, 20, 99]
    r = score_groups(groups, human, model)
    assert r.unordered_hits == 2
    assert round(r.unordered, 1) == round(200.0 / 3, 1)


def test_repeats_are_a_multiset_not_a_set() -> None:
    """Two points into one country is two entries; guessing it three times is not free."""
    groups = [[0, 1, 2]]
    human = [10, 10, 20]
    model = [10, 10, 10]
    r = score_groups(groups, human, model)
    assert r.unordered_hits == 2        # the third 10 has nothing left to match
    assert r.ordered_hits == 2


def test_ungrouped_decisions_are_scored_exactly_as_before() -> None:
    """A single-point decision has no order to be insensitive to; both figures must agree."""
    groups = [[0], [1], [2]]
    human = [10, 20, 30]
    model = [10, 99, 30]
    r = score_groups(groups, human, model)
    assert r.ordered_hits == r.unordered_hits == 2
    assert r.groups == 0 and r.grouped_decisions == 0


def test_the_two_measures_span_the_same_decisions() -> None:
    """Both are over every decision, so they stay directly comparable."""
    groups = [[0, 1], [2], [3, 4, 5]]
    human = [1, 2, 3, 4, 5, 6]
    model = [2, 1, 3, 6, 5, 4]
    r = score_groups(groups, human, model)
    assert r.decisions == 6
    assert r.ordered_hits == 2          # positions 2 and 4
    assert r.unordered_hits == 6        # both runs are permutations
    assert r.grouped_decisions == 5 and r.groups == 2


def test_unordered_is_never_worse_than_ordered() -> None:
    """It relaxes the ordered measure, so it can only ever credit more."""
    groups = [[0, 1, 2, 3]]
    for model in ([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1], [9, 9, 9, 9]):
        r = score_groups(groups, [1, 2, 3, 4], model)
        assert r.unordered_hits >= r.ordered_hits


def test_a_coup_or_realignment_run_is_never_reordered() -> None:
    """The board changes between those points, so the sequence is itself the decision.

    Blacklisted rather than whitelisting the order-free cases: spreading Influence is the ordinary
    case, and a card that spreads it in some new way should not have to be remembered.
    """
    import ts_engine as ts

    from ai.eval.agreement import group_point_runs, order_matters

    state = ts.GameState()
    ts.Engine.init_game(state, 3)
    ctx = state.ctx()

    ctx.op_mode = ts.OpMode.INFLUENCE
    ctx.resolving_card = 0
    ctx.pending_op_card = 0
    assert not order_matters(state)

    for mode in (ts.OpMode.COUP, ts.OpMode.REALIGN):
        ctx.op_mode = mode
        assert order_matters(state), f"{mode} changes the board between points"

    # Che's second coup is offered only if the first removed Influence, so the pair is a sequence.
    ctx.op_mode = ts.OpMode.INFLUENCE
    ctx.resolving_card = 107
    assert order_matters(state)

    # Blacklisted decisions are never merged into a run, so each is scored strictly.
    ctx.op_mode = ts.OpMode.REALIGN
    ctx.decision_type = ts.DecisionType.POINT_NODE
    groups = group_point_runs([state, state, state], [ts.Player.US] * 3)
    assert [len(g) for g in groups] == [1, 1, 1]
