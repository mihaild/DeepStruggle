"""Behavioural test suite: does the policy make specific, known-good choices?

Win rate against a baseline says whether an agent is better; it does not say whether it
understands anything in particular. This suite asks named questions -- "does the US waste
NATO's 4 Ops on its weak event?" -- and reports a pass rate that moves for a legible
reason.

Two assertion shapes cover most of what is worth testing:

* :class:`NeverArgmax` -- a blunder. The action must not be the policy's top choice, and
  must stay under a probability ceiling. Used where a move is close to always wrong.
* :class:`Prefer` -- a comparative. One action must outrank another in the same position.
  This needs no absolute threshold, which makes it the most robust form available: it asks
  only for an ordering, not for calibration.

Both read the masked policy distribution, so illegal actions carry no probability and a
malformed position raises rather than quietly passing.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import ts_engine as ts

from ai.eval.positions import PLAY_MODE_ACTION, PositionBuilder, card_action, legal_mask, step_to_play_mode


@dataclass
class Assertion:
    """Base assertion over a masked policy distribution."""

    def check(self, probs: np.ndarray) -> "AssertionOutcome":
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


@dataclass
class AssertionOutcome:
    passed: bool
    detail: str
    # An assertion whose question the policy never answered -- a comparative where the
    # policy put no mass on either option. Counting that as a failure would penalise a
    # deterministic bot for choosing a third action, which the claim says nothing about.
    inconclusive: bool = False


@dataclass
class NeverArgmax(Assertion):
    """`action` must not be the top choice, and must stay below `max_prob`."""

    action: int
    label: str
    max_prob: float = 0.20

    def check(self, probs: np.ndarray) -> AssertionOutcome:
        p = float(probs[self.action])
        top = int(np.argmax(probs))
        is_top = top == self.action
        passed = (not is_top) and p <= self.max_prob
        return AssertionOutcome(
            passed,
            f"P({self.label})={p:.3f} (ceiling {self.max_prob:.2f}), "
            f"{'IS the top choice' if is_top else 'not top choice'}",
        )

    def describe(self) -> str:
        return f"never choose {self.label} (P <= {self.max_prob:.2f}, not argmax)"


@dataclass
class Prefer(Assertion):
    """`preferred` must carry more probability mass than `over`."""

    preferred: int
    over: int
    preferred_label: str
    over_label: str
    margin: float = 0.0
    epsilon: float = 1e-9

    def check(self, probs: np.ndarray) -> AssertionOutcome:
        a, b = float(probs[self.preferred]), float(probs[self.over])
        if a <= self.epsilon and b <= self.epsilon:
            return AssertionOutcome(
                False,
                f"policy put no mass on either option "
                f"({self.preferred_label}, {self.over_label}) - ordering undefined",
                inconclusive=True,
            )
        passed = a > b + self.margin
        return AssertionOutcome(
            passed, f"P({self.preferred_label})={a:.3f} vs P({self.over_label})={b:.3f}"
        )

    def describe(self) -> str:
        return f"prefer {self.preferred_label} over {self.over_label}"


@dataclass
class BehavioralTest:
    """One named question, its position, and the assertion that answers it."""

    claim_id: str
    description: str
    builder: PositionBuilder
    assertion: Assertion
    # When set, select this card first so the assertion is evaluated at the
    # SELECT_PLAY_MODE node rather than at SELECT_CARD.
    play_card_first: Optional[int] = None
    tier: str = "A"

    def position(self) -> ts.GameState:
        state = self.builder.build()
        if self.play_card_first is not None:
            state = step_to_play_mode(state, self.play_card_first)
        return state


@dataclass
class TestResult:
    claim_id: str
    description: str
    tier: str
    passed: bool
    detail: str
    error: Optional[str] = None
    inconclusive: bool = False


@dataclass
class SuiteResult:
    results: List[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        runnable = [r for r in self.results if r.error is None and not r.inconclusive]
        if not runnable:
            return 0.0
        return sum(r.passed for r in runnable) / len(runnable)

    def by_tier(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tier in sorted({r.tier for r in self.results}):
            rs = [r for r in self.results if r.tier == tier and r.error is None
                  and not r.inconclusive]
            if rs:
                out[tier] = sum(r.passed for r in rs) / len(rs)
        return out

    def format_report(self) -> str:
        lines = ["", f"{'claim':44s} {'tier':5s} {'result':7s} detail", "-" * 118]
        for r in sorted(self.results, key=lambda x: (x.error is not None, x.passed, x.claim_id)):
            if r.error is not None:
                lines.append(f"{r.claim_id:44s} {r.tier:5s} {'ERROR':7s} {r.error}")
            elif r.inconclusive:
                lines.append(f"{r.claim_id:44s} {r.tier:5s} {'N/A':7s} {r.detail}")
            else:
                lines.append(
                    f"{r.claim_id:44s} {r.tier:5s} {'PASS' if r.passed else 'FAIL':7s} {r.detail}"
                )
        errored = sum(1 for r in self.results if r.error is not None)
        na = sum(1 for r in self.results if r.inconclusive)
        scored = [r for r in self.results if r.error is None and not r.inconclusive]
        lines.append("-" * 118)
        lines.append(
            f"pass rate: {self.pass_rate * 100:.1f}%  "
            f"({sum(r.passed for r in scored)}/{len(scored)} scored"
            + (f", {na} inconclusive" if na else "")
            + (f", {errored} errored)" if errored else ")")
        )
        for tier, rate in self.by_tier().items():
            lines.append(f"   tier {tier}: {rate * 100:.1f}%")
        return "\n".join(lines)


# (state, observation, mask) -> probability vector. The state is passed so that
# state-based baselines (HeuristicBot) can be scored on the same footing as networks.
PolicyFn = Callable[[ts.GameState, np.ndarray, np.ndarray], np.ndarray]


def torch_policy(net: torch.nn.Module, device: str = "cpu") -> PolicyFn:
    """Wraps a ColdWarNet so the suite only depends on obs+mask -> probability vector."""

    def fn(state: ts.GameState, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del state  # a network conditions only on the observation
        o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = net(o, m)
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return fn


def run_suite(policy: PolicyFn, tests: Sequence[BehavioralTest]) -> SuiteResult:
    out = SuiteResult()
    for t in tests:
        try:
            state = t.position()
            mask = legal_mask(state)
            if mask.sum() == 0:
                raise ValueError("position has no legal actions")
            obs = np.array(ts.extract_observation(state, state.ctx().decision_player), copy=True)
            probs = policy(state, obs, mask)
            outcome = t.assertion.check(probs)
            out.results.append(
                TestResult(t.claim_id, t.description, t.tier, outcome.passed, outcome.detail,
                           inconclusive=outcome.inconclusive)
            )
        except Exception as exc:  # a malformed position must be loud, not silently passing
            out.results.append(
                TestResult(t.claim_id, t.description, t.tier, False, "", error=str(exc))
            )
    return out


__all__ = [
    "Assertion",
    "AssertionOutcome",
    "BehavioralTest",
    "NeverArgmax",
    "PLAY_MODE_ACTION",
    "Prefer",
    "SuiteResult",
    "TestResult",
    "card_action",
    "run_suite",
    "torch_policy",
]
