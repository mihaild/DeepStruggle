"""P15-X4b: the CE term toward the searcher, and the guarantees around it.

The arm's whole claim to being one factor is that search produces *targets* and never *acts*, so
the state distribution is the baseline's. Two things therefore have to hold, and both are easy to
break silently:

* with `search_ce_coef = 0` nothing changes at all — not the buffer, not the loss, not the
  sampled actions;
* a searched decision is exempt from advantage filtering, because putting gradient on exactly
  those states is the point and the filter drops the ones whose outcome-advantage is small,
  which late in a run is most of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ai.training.rollout_buffer import RolloutBuffer

OBS, ACT, N, T = 8, 12, 4, 6


def _buf() -> RolloutBuffer:
    return RolloutBuffer(buffer_size=T, num_envs=N, obs_dim=OBS, action_dim=ACT, device="cpu")


def _add(buf: RolloutBuffer, search_pi=None, has_search=None) -> None:
    buf.add(obs=torch.zeros(N, OBS), masks=torch.ones(N, ACT, dtype=torch.uint8),
            actions=torch.zeros(N, dtype=torch.long), log_probs=torch.zeros(N),
            rewards=np.zeros(N, dtype=np.float32), dones=torch.zeros(N),
            values_win=torch.zeros(N), values_vp=torch.zeros(N),
            players=torch.ones(N, dtype=torch.int8),
            search_pi=search_pi, has_search=has_search)


def test_the_buffer_defaults_to_no_targets() -> None:
    """An unsearched run must carry an all-zero flag, or the CE term fires on nothing."""
    b = _buf()
    for _ in range(T):
        _add(b)
    assert float(b.has_search.sum()) == 0.0
    assert float(b.search_pi.abs().sum()) == 0.0


def test_a_target_lands_on_the_step_and_env_it_belongs_to() -> None:
    b = _buf()
    pi = torch.zeros(N, ACT)
    pi[2, 5] = 1.0
    flag = torch.zeros(N)
    flag[2] = 1.0
    _add(b)                      # step 0: nothing
    _add(b, pi, flag)            # step 1: env 2 searched
    _add(b)                      # step 2: nothing
    assert float(b.has_search[0].sum()) == 0.0
    assert float(b.has_search[1, 2]) == 1.0
    assert float(b.has_search[1].sum()) == 1.0
    assert float(b.search_pi[1, 2, 5]) == 1.0
    assert float(b.has_search[2].sum()) == 0.0


def test_batches_carry_the_target_after_the_learner_mask() -> None:
    """Appended, not inserted: every existing caller unpacks the first nine by position."""
    b = _buf()
    pi = torch.zeros(N, ACT)
    pi[:, 3] = 1.0
    for _ in range(T):
        _add(b, pi, torch.ones(N))
    b.compute_gae(last_v_win=torch.zeros(N), last_v_vp=torch.zeros(N),
                  last_dones=torch.zeros(N, dtype=torch.bool),
                  last_players=torch.ones(N, dtype=torch.int8))
    batch = next(b.get_batches(batch_size=8))
    assert len(batch) == 11, "search target and flag are elements 10 and 11"
    b_search_pi, b_has_search = batch[9], batch[10]
    assert b_search_pi.shape == (8, ACT)
    assert b_has_search.shape == (8,)
    assert float(b_has_search.sum()) == 8.0
    assert torch.allclose(b_search_pi.sum(dim=-1), torch.ones(8))


def test_the_ce_term_is_zero_where_nothing_was_searched() -> None:
    """The loss must not average a zero row in: that would pull toward a uniform target."""
    logits = torch.randn(6, ACT, requires_grad=True)
    log_p = torch.log_softmax(logits, dim=-1)
    pi = torch.zeros(6, ACT)
    has = torch.zeros(6)
    pi[1, 4] = 1.0
    has[1] = 1.0
    pi[4, 0] = 1.0
    has[4] = 1.0

    sel = has > 0.5
    ce = -(pi[sel] * log_p[sel]).sum(dim=-1).mean()
    # by hand, over the two searched rows only
    manual = -(log_p[1, 4] + log_p[4, 0]) / 2
    assert torch.allclose(ce, manual, atol=1e-6)

    # and the unsearched rows contribute no gradient
    ce.backward()
    grad = logits.grad
    assert grad is not None
    for i in (0, 2, 3, 5):
        assert float(grad[i].abs().sum()) == pytest.approx(0.0, abs=1e-9)


def test_a_searched_sample_survives_advantage_filtering() -> None:
    """The filter keeps large |A|; a searched decision is kept regardless. X4b exists for them."""
    adv = torch.tensor([0.001, 5.0, 0.002, 4.0])
    learner = torch.ones(4)
    has_search = torch.tensor([1.0, 0.0, 0.0, 0.0])

    keep = learner > 0.5
    thresh = torch.quantile(adv.abs()[keep].float(), 0.5)
    keep_filtered = keep & (adv.abs() >= thresh)
    assert not bool(keep_filtered[0]), "the fixture must have the searched sample below threshold"

    keep_x4b = keep & ((adv.abs() >= thresh) | (has_search > 0.5))
    assert bool(keep_x4b[0]), "a searched decision was dropped by the advantage filter"
    assert bool(keep_x4b[1]) and bool(keep_x4b[3])


def test_the_flag_is_needed_because_an_unsearched_row_is_all_zeros() -> None:
    """Why `has_search` exists rather than inferring from the target being non-zero."""
    b = _buf()
    _add(b)
    row = b.search_pi[0, 0]
    assert float(row.sum()) == 0.0
    assert float(b.has_search[0, 0]) == 0.0, (
        "an all-zero row is indistinguishable from a degenerate target without the flag")


def test_the_ce_metrics_are_registered_for_logging() -> None:
    """Computing a metric is not logging it, and this repo has been bitten by the gap twice.

    `train_step` returns `search_ce` and `search_ce_grad_frac`, but `generic_trainer` writes
    `training_metrics.jsonl` through an allow-list, so a key that nothing registers is dropped
    without a word -- which is exactly what happened to the PFSP per-opponent win rates, as the
    comment beside the forwarding loop records. The first version of this instrumentation was
    dropped the same way, and the arm it was built to diagnose ran without it.

    Asserting on the source of the registration block rather than on a live run: constructing a
    trainer needs a GPU and an env, and the thing that broke was the registration, not the
    computation.
    """
    import inspect

    from ai.training import generic_trainer

    src = inspect.getsource(generic_trainer.train_pipeline)
    block = src[src.index("active_aux_losses: List[str] = []"):]
    block = block[:block.index("prev_steps")]

    assert '"search_ce"' in block, (
        "search_ce is computed but never registered in active_aux_losses, so it will not reach "
        "training_metrics.jsonl")
    assert '"search_ce_grad_frac"' in block, (
        "search_ce_grad_frac is computed but never registered, so the CE term's share of the "
        "update stays invisible -- which is what made the 20M collapse silent")
    # The guard must be able to fail. A block that does not mention the key at all is the
    # regression being guarded against, so check the assertion would catch it.
    assert '"search_ce"' not in block.replace('"search_ce"', "", 1), (
        "the registration appears more than once; this test's teeth check is unreliable")

    assert "if search_ce_coef > 0.0" in block, (
        "the CE metrics must be registered only when the term is on; logged unconditionally they "
        "are a flat zero line that reads as 'present and converged'")


def test_the_ppo_ratio_cannot_overflow_to_nan() -> None:
    """A forced low-probability action must not blow up the update.

    --setup-explore-frac replaces the opening with a uniform legal choice and stores the policy's
    own log-prob of it. At the opening that probability is around 6e-8 -- the measured logit gap is
    16.56 -- so the stored log-prob is near -16.6, and exp(cur_lp - old_lp) overflows to inf on a
    modest shift. Clipping the ratio afterwards does not save it: inf survives clamp and
    inf * advantage is NaN.

    E3-32-30 died exactly this way at 53.7M steps, with every instrument healthy in the iteration
    before -- AUC 0.781, entropy 1.04, KL 0.126 -- so nothing in the metrics would have warned.
    """
    import inspect

    from ai.training.nash_pg import NashPGTrainer

    src = inspect.getsource(NashPGTrainer.train_step)
    line = next((l for l in src.splitlines() if "ratio = torch.exp(" in l), None)
    assert line is not None, "the PPO ratio is no longer computed with torch.exp"
    assert "clamp" in line, (
        "torch.exp() on the raw log-ratio can overflow to inf and poison the batch with NaN; "
        "bound the exponent before the exponential")


def test_forced_setup_actions_are_stored_with_their_own_log_prob() -> None:
    """The override must not be recorded as if the policy had chosen it at temperature.

    The ratio starts at 1.0 only because the stored log-prob is the policy's canonical log-prob of
    the action actually taken. If the override were applied AFTER the gather, the buffer would hold
    the log-prob of a different action and every ratio involving a forced opening would be wrong.
    """
    import inspect

    from ai.training.nash_pg import BaseNashPGTrainer

    src = inspect.getsource(BaseNashPGTrainer.collect_rollouts)
    force = src.index("_force_setup_exploration")
    gather = src.index("log_probs_t = unscaled_log_probs.gather")
    assert force < gather, (
        "the setup override runs after the log-prob gather, so the buffer would record the "
        "log-prob of an action that was not taken")


def test_mass_outside_the_legal_mask_is_measured_not_inferred() -> None:
    """The alignment check must count misplaced target mass directly.

    `search_ce` reports this only by accident. The mask fill is -1e9, so target mass e on a
    MASKED action contributes e * 1e9 to the cross-entropy, which is why search_ce is observed
    above 1e5 -- impossible for a 212-way softmax, whose maximum is log(212) = 5.36 -- in about a
    third of iterations on every search arm measured.

    Inferring the mass by dividing the loss by 1e9 works only while the misplaced action happens
    to be illegal. The same misalignment landing on a LEGAL wrong action produces an ordinary CE
    and is invisible, which is precisely how the earlier search-target off-by-one survived: the
    targets stayed legal, so no mask caught them.

    This pins the arithmetic the diagnostic rests on, and that the metric is registered.
    """
    import inspect

    import torch

    from ai.training import generic_trainer, nash_pg

    # 1. the arithmetic: mass e on a masked action costs e * 1e9 of CE
    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, True, False]])
    masked = torch.where(mask, logits, torch.tensor(-1e9))
    log_p = torch.log_softmax(masked, dim=-1)

    eps = 1e-4
    target = torch.tensor([[1.0 - eps, 0.0, 0.0, eps]])
    ce = -(target * log_p).sum(dim=-1).mean()
    assert ce > 1e4, f"masked mass should blow the CE up, got {ce}"
    assert abs(float(ce) / 1e9 - eps) < 0.2 * eps, (
        "CE/1e9 should recover the misplaced mass, which is the inference the metric replaces")

    # the measured quantity is exact where the inferred one is approximate
    measured = (target * (~mask).to(target.dtype)).sum(dim=-1)
    # exact up to float32 epsilon at this magnitude (~1e-11), against the inferred value's ~20%
    assert abs(float(measured[0]) - eps) < 1e-9

    # 2. a LEGAL wrong action is invisible to the loss but not to a target check
    legal_wrong = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    ce_legal = -(legal_wrong * log_p).sum(dim=-1).mean()
    assert ce_legal < 5.36 + 1e-6, (
        "a legal target must produce an ordinary CE -- this is the silent case")
    assert float((legal_wrong * (~mask).to(legal_wrong.dtype)).sum()) == 0.0

    # 3. the metrics exist and are registered, or none of the above reaches the log
    src = inspect.getsource(nash_pg.NashPGTrainer.train_step)
    assert "search_target_illegal_mass_max" in src
    assert "search_target_illegal_row_frac" in src

    block = inspect.getsource(generic_trainer.train_pipeline)
    block = block[block.index("active_aux_losses: List[str] = []"):]
    block = block[:block.index("prev_steps")]
    for key in ("search_target_illegal_mass_max", "search_target_illegal_row_frac"):
        assert f'"{key}"' in block, f"{key} is computed but never registered for logging"


def test_search_targets_are_filtered_by_the_real_mask() -> None:
    """A determinized search may propose actions that are illegal in the true state.

    batched_mcts.py says so where it handles this for an ACTING agent: "a determinized search can
    legitimately return an action that is illegal in the real state, because in this game the
    legal SET itself can depend on hidden information", and there the search proposes and the true
    mask disposes. `_search_targets` calls `run()` directly, which is not that path, and used to
    write the visit counts straight into the target with only a bounds check -- so visits that
    were legal only under the sampled determinization became target mass on masked actions.

    The symptom was search_ce above 1e5 in about a third of iterations on every search arm, which
    is impossible for a 212-way softmax (max log(212) = 5.36) and arises because the mask fill is
    -1e9.

    This pins the normalisation the filter must preserve, and that the source still applies it.
    """
    import inspect

    import numpy as np

    from ai.training import nash_pg

    # the renormalisation: drop illegal visits, and what survives must still sum to 1
    acts = [3, 7, 11]
    visits = np.array([10.0, 1.0, 5.0], dtype=np.float32)
    legal = {3: True, 7: False, 11: True}          # 7 legal only under determinization

    keep = np.array([legal[a] for a in acts], dtype=bool)
    kept = np.where(keep, visits, 0.0)
    total = float(kept.sum())
    assert total == 15.0
    probs = kept / total
    assert abs(float(probs.sum()) - 1.0) < 1e-6, "a filtered target must still be a distribution"
    assert float(probs[1]) == 0.0, "the illegal action must carry no mass"
    # and the surviving two keep their relative weights
    assert abs(float(probs[0]) - 10.0 / 15.0) < 1e-6
    assert abs(float(probs[2]) - 5.0 / 15.0) < 1e-6

    # a row where EVERY visit is illegal yields no target rather than a guessed one
    all_illegal = np.where(np.array([False, False, False]), visits, 0.0)
    assert float(all_illegal.sum()) == 0.0

    src = inspect.getsource(nash_pg.NashPGTrainer._search_targets)
    assert "get_legal_mask" in src, (
        "_search_targets must consult the real state's mask; without it a determinized search's "
        "answer becomes target mass on actions that cannot be played")
    assert "search_dropped_visit_frac" in src, (
        "how much the mask rejected must be logged, or a filter that stops working is invisible")


def test_filter_search_visits_drops_illegal_and_renormalises() -> None:
    """Exercise the dropping path directly; a real run is far too coarse to hit it.

    The rate is roughly one row in 160, so the 4-iteration CPU smoke that verified the metrics
    reach the log reported search_dropped_visit_frac = 0.0 -- correct, and no evidence the filter
    works. These cases drive it on purpose.
    """
    import numpy as np

    from ai.training.nash_pg import filter_search_visits

    legal = np.array([1, 1, 1, 0, 1, 0], dtype=np.uint8)   # 3 and 5 illegal in the real state
    width = 6

    # an illegal recommendation is dropped and the rest renormalise
    pairs, dropped = filter_search_visits([0, 3, 4], np.array([10.0, 2.0, 5.0]), legal, width)
    assert dropped == 2.0
    assert pairs is not None
    got = dict(pairs)
    assert 3 not in got, "the illegal action kept target mass"
    assert abs(got[0] - 10.0 / 15.0) < 1e-9
    assert abs(got[4] - 5.0 / 15.0) < 1e-9
    assert abs(sum(p for _, p in pairs) - 1.0) < 1e-9, "target must remain a distribution"

    # nothing illegal: untouched, and nothing reported as dropped
    pairs, dropped = filter_search_visits([0, 1, 2], np.array([1.0, 1.0, 2.0]), legal, width)
    assert dropped == 0.0
    assert pairs is not None and abs(sum(p for _, p in pairs) - 1.0) < 1e-9

    # every visit illegal: no target at all rather than a guessed one
    pairs, dropped = filter_search_visits([3, 5], np.array([4.0, 6.0]), legal, width)
    assert pairs is None, "a row with no legal visit must produce no target"
    assert dropped == 10.0

    # out-of-range action ids are dropped like illegal ones, not indexed with
    pairs, dropped = filter_search_visits([0, 99, -1], np.array([3.0, 1.0, 1.0]), legal, width)
    assert dropped == 2.0
    assert pairs is not None and dict(pairs) == {0: 1.0}

    # a mask narrower than the target width must not be indexed past its end
    narrow = np.array([1, 1], dtype=np.uint8)
    pairs, dropped = filter_search_visits([0, 4], np.array([1.0, 1.0]), narrow, width)
    assert pairs is not None and dict(pairs) == {0: 1.0}
    assert dropped == 1.0


def test_search_target_entropy_is_the_targets_own_entropy_and_is_logged() -> None:
    """Separates "the CE term taught the policy to flatten" from "the policy flattened anyway".

    The decline signature on every X4b arm is policy entropy RISING while strength falls, which is
    backwards for a policy that is merely over-sharpening. If the searcher's visit distribution
    is itself flattening, the CE term is actively teaching that, and the loop is the explanation.
    If target entropy stays put while the policy's climbs, the loop is dead. Nothing logged that
    number, so neither reading could be tested.
    """
    import inspect

    import torch

    from ai.training import generic_trainer, nash_pg

    # the quantity: entropy of the TARGET, not of the policy
    one_hot = torch.zeros(1, 8)
    one_hot[0, 3] = 1.0
    ent = -(one_hot * one_hot.clamp_min(1e-12).log()).sum(dim=-1)
    assert abs(float(ent[0])) < 1e-6, "a decisive target must have ~zero entropy"

    # a uniform target over 4 legal actions is log(4)
    uni = torch.zeros(1, 8)
    uni[0, :4] = 0.25
    ent = -(uni * uni.clamp_min(1e-12).log()).sum(dim=-1)
    assert abs(float(ent[0]) - math.log(4)) < 1e-6

    # zeros outside the legal set must not contribute: clamp_min guards log(0) -> -inf, and
    # 0 * -inf would be NaN, which would silently poison the mean for a whole iteration
    assert not torch.isnan(ent).any()

    src = inspect.getsource(nash_pg.NashPGTrainer.train_step)
    assert "search_target_entropy" in src

    block = inspect.getsource(generic_trainer.train_pipeline)
    block = block[block.index("active_aux_losses: List[str] = []"):]
    block = block[:block.index("prev_steps")]
    assert '"search_target_entropy"' in block, (
        "search_target_entropy is computed but never registered, so it will not reach "
        "training_metrics.jsonl -- the same allow-list gap that dropped search_ce")
