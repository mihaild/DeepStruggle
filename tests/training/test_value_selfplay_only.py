"""--value-selfplay-only (P44 diagnostic, 2026-11-14): the value loss counts only the rows its mask marks."""
from types import SimpleNamespace

import pytest
import torch

from ai.training.nash_pg import NashPGTrainer


def _loss(rows: torch.Tensor) -> torch.Tensor:
    owner = SimpleNamespace(vp_coef=0.05)
    v_win = torch.tensor([0.5, -0.5, 0.0, 1.0])
    v_vp = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ret_win = torch.tensor([1.0, -1.0, 1.0, -1.0])
    ret_vp = torch.tensor([0.0, 0.0, 0.0, 0.0])
    return NashPGTrainer._value_loss_rows(owner, v_win, v_vp, ret_win, ret_vp, None, rows)  # type: ignore[arg-type]


def test_all_rows_marked_is_the_plain_mse() -> None:
    expected = torch.tensor([0.25, 0.25, 1.0, 4.0]).mean() + 0.05 * torch.tensor([0.01, 0.04, 0.09, 0.16]).mean()
    assert float(_loss(torch.ones(4))) == pytest.approx(float(expected))


def test_unmarked_rows_do_not_count() -> None:
    expected = (0.25 + 0.25) / 2 + 0.05 * (0.01 + 0.04) / 2
    assert float(_loss(torch.tensor([1.0, 1.0, 0.0, 0.0]))) == pytest.approx(expected)


def test_categorical_value_is_refused() -> None:
    owner = SimpleNamespace(vp_coef=0.05)
    with pytest.raises(ValueError, match="scalar value heads"):
        NashPGTrainer._value_loss_rows(owner, torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2),  # type: ignore[arg-type]
                                       torch.zeros(2, 3), torch.ones(2))
