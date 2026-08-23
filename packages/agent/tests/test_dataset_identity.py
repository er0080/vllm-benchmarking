"""What a run says about the data it measured.

Invariant 6 has required a dataset identity since the first schema, and the column was NULL
on every run this project produced until protocol 7 (issue #82). The gap was not cosmetic:
a workload is content-addressed on its *arguments*, and for every dataset except the
generated ones the decisive argument is a path on the GPU host. Overwrite the file and both
runs still claim the identical workload, group into one series, and hand the difference to
whichever config changed. Nothing errors.

So the test that matters most is the first one here: same path, different bytes, different
identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllmbench_agent import dataset
from vllmbench_agent.dataset import identify_dataset
from vllmbench_protocol.wire import BenchRequest


def request(**overrides: object) -> BenchRequest:
    fields: dict[str, object] = {"model": "m", "dataset_name": "sharegpt"}
    fields.update(overrides)
    return BenchRequest.model_validate(fields)


class TestALocalFile:
    def test_editing_the_file_changes_the_identity(self, tmp_path: Path) -> None:
        """The whole point. One path, two different corpora, two different runs."""
        path = tmp_path / "prompts.json"
        path.write_text('[{"prompt": "first"}]')
        before = identify_dataset(request(dataset_path=str(path)))
        path.write_text('[{"prompt": "second"}]')
        after = identify_dataset(request(dataset_path=str(path)))
        assert before is not None and after is not None
        assert before != after

    def test_the_same_bytes_at_a_different_path_are_the_same_data(self, tmp_path: Path) -> None:
        """Content, not location: a dataset copied to a second host is the same dataset.

        Identity is about what was measured. Two runs on two hosts reading byte-identical
        corpora measured the same thing, and a path-based identity would deny it.
        """
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        first.write_text("same")
        second.write_text("same")
        assert identify_dataset(request(dataset_path=str(first))) == identify_dataset(
            request(dataset_path=str(second))
        )

    def test_the_form_is_self_describing(self, tmp_path: Path) -> None:
        path = tmp_path / "d.json"
        path.write_text("abc")
        found = identify_dataset(request(dataset_path=str(path)))
        assert found is not None
        scheme, digest, size = found.split(":")
        assert scheme == "sha256"
        assert len(digest) == 64
        assert size == "3"

    def test_a_local_file_wins_over_hf_name(self, tmp_path: Path) -> None:
        """Upstream's help: set `--hf-name` *when* `--dataset-path` is a local path.

        The two together mean a local file carrying an HF dataset's schema, so the bytes
        are what was read and the repo id would be describing something else.
        """
        path = tmp_path / "local.json"
        path.write_text("bytes")
        found = identify_dataset(request(dataset_path=str(path), hf_name="org/dataset"))
        assert found is not None and found.startswith("sha256:")

    def test_a_missing_file_is_unknown_not_a_lie(self, tmp_path: Path) -> None:
        """`vllm bench serve` is about to fail on the same path with a better message."""
        assert identify_dataset(request(dataset_path=str(tmp_path / "gone.json"))) is None

    def test_an_unreadable_file_does_not_fail_the_run(self, tmp_path: Path) -> None:
        """A measurement is the valuable thing. Awkward provenance is recorded as absent,
        never raised into the benchmark path."""
        path = tmp_path / "locked.json"
        path.write_text("secret")
        path.chmod(0o000)
        try:
            assert identify_dataset(request(dataset_path=str(path))) is None
        finally:
            path.chmod(0o600)

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        assert identify_dataset(request(dataset_path=str(tmp_path))) is None


class TestAFileTooLargeToHashWhole:
    @pytest.fixture(autouse=True)
    def _small_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shrunk rather than writing gigabytes: the branch is what is under test."""
        monkeypatch.setattr(dataset, "FULL_HASH_MAX_BYTES", 64)
        monkeypatch.setattr(dataset, "SAMPLE_BYTES", 16)

    def test_it_says_it_is_not_a_full_hash(self, tmp_path: Path) -> None:
        """Labelled weaker because it is weaker. A prefix claiming `sha256:` would make
        an edit in the middle of a large file look like proof the data was unchanged."""
        path = tmp_path / "big.json"
        path.write_bytes(b"x" * 200)
        found = identify_dataset(request(dataset_path=str(path)))
        assert found is not None and found.startswith("sha256-head-tail:")

    def test_a_changed_head_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "big.json"
        path.write_bytes(b"a" + b"x" * 199)
        before = identify_dataset(request(dataset_path=str(path)))
        path.write_bytes(b"b" + b"x" * 199)
        assert identify_dataset(request(dataset_path=str(path))) != before

    def test_a_changed_tail_is_caught(self, tmp_path: Path) -> None:
        path = tmp_path / "big.json"
        path.write_bytes(b"x" * 199 + b"a")
        before = identify_dataset(request(dataset_path=str(path)))
        path.write_bytes(b"x" * 199 + b"b")
        assert identify_dataset(request(dataset_path=str(path))) != before

    def test_length_alone_separates_two_files_sharing_both_ends(self, tmp_path: Path) -> None:
        """The easy way to defeat a head-and-tail digest, which is why size is hashed in."""
        short, long = tmp_path / "s.json", tmp_path / "l.json"
        short.write_bytes(b"h" * 16 + b"m" * 68 + b"t" * 16)
        long.write_bytes(b"h" * 16 + b"m" * 168 + b"t" * 16)
        assert identify_dataset(request(dataset_path=str(short))) != identify_dataset(
            request(dataset_path=str(long))
        )

    def test_a_middle_edit_is_admittedly_missed(self, tmp_path: Path) -> None:
        """Documenting the limit rather than pretending it is not there.

        This is the trade the `-head-tail` label exists to disclose: reading forty
        gigabytes before every benchmark would be a real cost on the machine under test.
        If this ever starts passing — a full hash, or a sampled middle — the label is what
        must change with it.
        """
        path = tmp_path / "big.json"
        path.write_bytes(b"h" * 16 + b"a" * 100 + b"t" * 16)
        before = identify_dataset(request(dataset_path=str(path)))
        path.write_bytes(b"h" * 16 + b"b" * 100 + b"t" * 16)
        assert identify_dataset(request(dataset_path=str(path))) == before


class TestAHuggingFaceDataset:
    def test_the_resolved_revision_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The repo id alone is a moving target; the commit is what was actually read."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        ref = tmp_path / "hub" / "datasets--org--corpus" / "refs"
        ref.mkdir(parents=True)
        (ref / "main").write_text("abc123def456\n")
        found = identify_dataset(request(dataset_name="hf", dataset_path="org/corpus"))
        assert found == "hf:org/corpus@abc123def456"

    def test_an_unresolvable_repo_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Still better than NULL: the repo id is a fact even when the commit is not."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        found = identify_dataset(request(dataset_name="hf", dataset_path="org/corpus"))
        assert found == "hf:org/corpus@unresolved"

    def test_the_cache_is_read_not_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An offline GPU host must still produce a run. Nothing here may block on
        huggingface.co while a benchmark waits."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setattr(
            dataset, "_hf_revision", lambda repo: pytest.fail("should not have been re-fetched")
        )
        path = tmp_path / "local.json"
        path.write_text("bytes")
        assert identify_dataset(request(dataset_path=str(path))) is not None


class TestAGeneratedDataset:
    def test_it_says_generated_rather_than_leaving_a_null(self) -> None:
        """So "no dataset identity" stops being ambiguous between "generated" and
        "nobody looked" — the third done-when bullet of issue #82."""
        found = identify_dataset(
            request(dataset_name="random", random_input_len=512, random_output_len=128)
        )
        assert found is not None and found.startswith("generated:random:")

    def test_the_knobs_are_the_identity(self) -> None:
        """A generator is deterministic given its inputs, so its inputs identify its output."""
        short = identify_dataset(request(dataset_name="random", random_input_len=512))
        long = identify_dataset(request(dataset_name="random", random_input_len=4096))
        assert short != long

    def test_it_does_not_say_synthetic(self) -> None:
        """Invariant 7 owns that word for runs that did not measure real hardware. A
        generated prompt set on a real GPU is a real measurement, and borrowing the
        quarantine vocabulary would eventually put it in quarantine."""
        found = identify_dataset(request(dataset_name="prefix_repetition"))
        assert found is not None and "synthetic" not in found

    def test_a_named_dataset_with_no_path_is_unknown(self) -> None:
        """`sonnet` without its file: we know the name, and the name is already the
        workload hash. Repeating it here would look like evidence about contents."""
        assert identify_dataset(request(dataset_name="sonnet")) is None
