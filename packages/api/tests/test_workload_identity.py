"""What makes two workloads the same workload.

Workloads are content-addressed on what they send, so the identity field list is the
definition of "same traffic". A field left out of it is not a cosmetic omission: the second
create returns the first row, and a sweep authored across "both" runs one of them twice
while labelling the results as two.

`extra_args` was outside that list and unreachable from every interface, so the collision
was never reachable either. These tests exist so it does not become reachable again.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllmbench_api.hashing import workload_hash
from vllmbench_api.routers.runs import _WORKLOAD_IDENTITY_FIELDS
from vllmbench_api.schemas import MAX_EXTRA_ARG_LENGTH, MAX_EXTRA_ARGS, WorkloadCreate


def identity(payload: WorkloadCreate) -> dict[str, object]:
    """The digest input, built exactly as the router builds it."""
    return {field: getattr(payload, field) for field in _WORKLOAD_IDENTITY_FIELDS}


def digest(**overrides: object) -> str:
    fields: dict[str, object] = {"name": "w", "dataset_name": "blazedit"}
    fields.update(overrides)
    return workload_hash(identity(WorkloadCreate.model_validate(fields)))


class TestExtraArgsIsIdentity:
    def test_two_edit_distance_bands_are_two_workloads(self) -> None:
        """The case that motivated this: same dataset, different slice of it.

        Before `extra_args` joined the identity these hashed the same, so the second
        create returned the first and an A/B across them compared a workload with itself.
        """
        small = digest(
            extra_args=["--blazedit-min-distance", "0.0", "--blazedit-max-distance", "0.2"]
        )
        large = digest(
            extra_args=["--blazedit-min-distance", "0.6", "--blazedit-max-distance", "1.0"]
        )
        assert small != large

    def test_identical_flags_are_one_workload(self) -> None:
        """Content addressing still holds: same traffic, same row."""
        args = ["--blazedit-max-distance", "0.2"]
        assert digest(extra_args=args) == digest(extra_args=list(args))

    def test_order_matters(self) -> None:
        """argv is ordered and later flags win, so a reordering can send different traffic."""
        assert digest(extra_args=["--a", "1", "--b", "2"]) != digest(
            extra_args=["--b", "2", "--a", "1"]
        )

    def test_no_flags_differs_from_some_flags(self) -> None:
        assert digest() != digest(extra_args=["--seed", "7"])

    def test_the_field_is_actually_in_the_list(self) -> None:
        """Belt and braces: the tests above would all pass if the digest ignored nothing.

        This asserts the mechanism rather than the symptom, so removing the field from
        `_WORKLOAD_IDENTITY_FIELDS` fails here with an obvious reason rather than as four
        confusing equality failures.
        """
        assert "extra_args" in _WORKLOAD_IDENTITY_FIELDS


class TestFrameworkFlagsAreRefused:
    def test_result_filename_is_refused(self) -> None:
        """The worst one. Later argv wins, so this would redirect the benchmark's output.

        Nothing would error: the run finishes, the file we read is absent or stale, and the
        summary is a row of NULLs indistinguishable from a run that measured nothing.
        """
        with pytest.raises(ValidationError, match="set by this framework"):
            WorkloadCreate(name="w", extra_args=["--result-filename", "/tmp/elsewhere.json"])

    def test_the_equals_form_is_refused_too(self) -> None:
        """`--flag=value` is accepted by argparse and would otherwise slip past a bare match."""
        with pytest.raises(ValidationError, match="set by this framework"):
            WorkloadCreate(name="w", extra_args=["--num-prompts=1"])

    @pytest.mark.parametrize(
        "flag", ["--model", "--dataset-name", "--max-concurrency", "--base-url"]
    )
    def test_other_owned_flags(self, flag: str) -> None:
        with pytest.raises(ValidationError, match="set by this framework"):
            WorkloadCreate(name="w", extra_args=[flag, "x"])

    def test_a_dataset_option_passes(self) -> None:
        """The whole point: flags this build has no field for still get through."""
        payload = WorkloadCreate(name="w", extra_args=["--blazedit-min-distance", "0.0"])
        assert payload.extra_args == ["--blazedit-min-distance", "0.0"]


class TestBounds:
    def test_too_many_items(self) -> None:
        with pytest.raises(ValidationError, match="at most"):
            WorkloadCreate(name="w", extra_args=["--x"] * (MAX_EXTRA_ARGS + 1))

    def test_an_overlong_item(self) -> None:
        with pytest.raises(ValidationError, match="longer than"):
            WorkloadCreate(name="w", extra_args=["x" * (MAX_EXTRA_ARG_LENGTH + 1)])


class TestOlderRows:
    def test_an_empty_object_reads_as_no_flags(self) -> None:
        """Rows predating the list type hold `{}`, including the CI seed.

        WorkloadOut inherits this model, so without the coercion reading one of those rows
        raises on a value that means exactly what an empty list means.
        """
        assert WorkloadCreate.model_validate({"name": "w", "extra_args": {}}).extra_args == []

    def test_null_reads_as_no_flags(self) -> None:
        assert WorkloadCreate.model_validate({"name": "w", "extra_args": None}).extra_args == []

    def test_an_old_row_hashes_like_an_empty_list(self) -> None:
        """So a workload re-submitted after the type change is still the same workload."""
        assert digest(extra_args={}) == digest(extra_args=[])
