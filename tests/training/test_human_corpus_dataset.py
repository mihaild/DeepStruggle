"""The human-corpus dataset must not repeat the archived warmup set's failure.

That set stored `(seed, actions)` and re-derived observations, so an engine change silently
truncated it -- 27% of its decisions survived and nothing said so. This one stores observations
materialised, which removes that failure mode but introduces two of its own: a mask that does not
permit the action stored beside it, and a value target on a game that has no outcome. Both are
pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.training.human_corpus_dataset import (MASK_BITS, HumanCorpusDataset,
                                              HumanCorpusWriter)

OBS_DIM = 12


def _sample(action: int, legal: list[int], side: int):
    mask = np.zeros(MASK_BITS, dtype=np.uint8)
    for a in legal:
        mask[a] = 1
    return (np.arange(OBS_DIM, dtype=np.float32) + action, mask, action, side)


@pytest.fixture()
def built(tmp_path):
    writer = HumanCorpusWriter(str(tmp_path / "ds"))
    # A finished game: the US won, so a US decision is labelled +1 and a USSR decision -1.
    writer.add_game([_sample(3, [3, 7], 1), _sample(7, [3, 7], -1)],
                    us_utility=1.0, final_vp=20)
    # A game whose recording stopped: no outcome, so no value target.
    writer.add_game([_sample(11, [11, 12], 1)], us_utility=None, final_vp=None)
    meta = writer.write()
    return str(tmp_path / "ds"), meta


def test_the_written_meta_counts_what_was_added(built) -> None:
    _, meta = built
    assert meta["samples"] == 3
    assert meta["games"] == 2
    assert meta["games_with_outcome"] == 1
    assert meta["samples_with_outcome"] == 2


def test_every_stored_mask_permits_its_stored_action(built) -> None:
    """The archived set's collapse was demonstrated actions going illegal. Assert it directly."""
    directory, _ = built
    rows = list(HumanCorpusDataset(directory).stream_transitions())
    assert len(rows) == 3
    for obs, mask, action, _win, _vp, _has in rows:
        assert mask.shape == (MASK_BITS,)
        assert mask[action] == 1, f"action {action} is not legal in its own mask"
        assert obs.shape == (OBS_DIM,)


def test_a_game_without_an_outcome_carries_no_value_target(built) -> None:
    """Half the corpus stops mid-game. Labelling those teaches the critic results that never were."""
    directory, _ = built
    rows = list(HumanCorpusDataset(directory).stream_transitions())
    settled = [r for r in rows if r[5] == 1]
    unsettled = [r for r in rows if r[5] == 0]
    assert len(settled) == 2 and len(unsettled) == 1
    for _obs, _mask, _action, win, vp, _has in unsettled:
        assert win == 0.0 and vp == 0.0


def test_the_label_is_from_the_mover_s_point_of_view(built) -> None:
    """A US win is +1 for a US decision and -1 for a USSR one in the same game."""
    directory, _ = built
    rows = list(HumanCorpusDataset(directory).stream_transitions())
    by_action = {r[2]: r for r in rows}
    assert by_action[3][3] == pytest.approx(1.0)    # US moved
    assert by_action[7][3] == pytest.approx(-1.0)   # USSR moved, same won game


def test_batches_round_trip_and_carry_the_outcome_flag(built) -> None:
    directory, _ = built
    ds = HumanCorpusDataset(directory)
    batches = list(ds.stream_batches(batch_size=2, shuffle=False))
    assert batches, "expected at least one batch"
    obs, mask, action, win, vp, has_outcome = batches[0]
    assert obs.shape == (2, OBS_DIM)
    assert mask.shape == (2, MASK_BITS)
    for row, act in zip(mask.tolist(), action.tolist()):
        assert row[act] == 1
    assert has_outcome.shape == (2,)


def test_a_directory_without_meta_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        HumanCorpusDataset(str(tmp_path))
