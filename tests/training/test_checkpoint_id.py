"""Checkpoint names must identify a run, not just a step count.

Snapshot filenames collide by construction: the steps a run snapshots at are a deterministic
function of its configuration, so two runs with the same settings produce byte-identically named
files. Every tool that reduced a checkpoint to its basename lost the only thing that told them
apart, and the reports that resulted could not be cited.
"""

from __future__ import annotations

import pytest

from tools.lib.checkpoint_id import checkpoint_label, unique_labels


class TestLabel:
    def test_run_and_steps(self) -> None:
        assert checkpoint_label(
            "/data/checkpoints/E4-02-01_20260919_040456/snapshot_150011904steps.pt"
        ) == "E4-02-01@150M"

    def test_the_collision_that_motivated_this(self) -> None:
        """The same filename in two runs must produce two different labels."""
        a = checkpoint_label("/d/E4-01-01_20260919_003959/snapshot_150011904steps.pt")
        b = checkpoint_label("/d/E4-02-01_20260919_040456/snapshot_150011904steps.pt")
        assert a != b
        assert a == "E4-01-01@150M" and b == "E4-02-01@150M"

    def test_sub_million_steps_keep_resolution(self) -> None:
        assert checkpoint_label(
            "/d/E4-02-01_20260919_040456/snapshot_65536steps.pt") == "E4-02-01@66k"

    def test_named_snapshots_keep_their_name(self) -> None:
        assert checkpoint_label(
            "/d/E4-02-01_20260919_040456/snapshot_final.pt") == "E4-02-01@final"

    def test_loose_checkpoint_falls_back_to_its_stem(self) -> None:
        """A file in the shared root has no run to name it by, and inventing one would be worse."""
        assert checkpoint_label("/workspace/data/checkpoints/E4_1_warmup.pt") == "E4_1_warmup"

    def test_non_conforming_run_dir_still_beats_nothing(self) -> None:
        assert checkpoint_label("/d/some_old_run/snapshot_80019456steps.pt") == "some_old_run@80M"


class TestUniqueness:
    def test_distinct_runs_are_accepted(self) -> None:
        assert unique_labels([
            "/d/E4-01-01_20260919_003959/snapshot_150011904steps.pt",
            "/d/E4-02-01_20260919_040456/snapshot_150011904steps.pt",
        ]) == ["E4-01-01@150M", "E4-02-01@150M"]

    def test_a_genuine_clash_raises_rather_than_suffixing(self) -> None:
        """#1/#2 suffixing is what made the old reports unciteable; a clash is a mistake."""
        with pytest.raises(ValueError, match="share the label"):
            unique_labels([
                "/d/E4-02-01_20260919_040456/snapshot_150011904steps.pt",
                "/other/E4-02-01_20260919_040456/snapshot_150011904steps.pt",
            ])


class TestLineagesAndBranches:
    def test_a_final_is_labelled_by_its_budget(self, tmp_path) -> None:
        """One short name spans several budgets once a lineage is continued, so `@final` is
        ambiguous; the run's recorded budget disambiguates it."""
        import json
        d = tmp_path / "E4-08-03_20260921_023139"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"train_steps": 160_000_000}))
        assert checkpoint_label(str(d / "snapshot_final.pt")) == "E4-08-03@160M"

    def test_replicate_and_branch_names_are_kept_whole(self) -> None:
        assert checkpoint_label(
            "/d/E4-08-01-2_20260920_004011/snapshot_80019456steps.pt") == "E4-08-01-2@80M"
        assert checkpoint_label(
            "/d/E4-17-06-50M.11_20260922_094429/snapshot_110034944steps.pt") \
            == "E4-17-06-50M.11@110M"

    def test_one_lineage_across_directories_does_not_clash(self) -> None:
        assert unique_labels([
            "/d/E4-08-03_20260919_223012/snapshot_80019456steps.pt",
            "/d/E4-08-03_20260921_023139/snapshot_160038912steps.pt",
            "/d/E4-08-03_20260922_202204/snapshot_240058368steps.pt",
        ]) == ["E4-08-03@80M", "E4-08-03@160M", "E4-08-03@240M"]
