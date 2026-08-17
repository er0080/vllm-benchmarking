"""Sweep expansion and tensor-parallel config variants.

The variant generator gets the most attention here because it is the one place in the
system that *writes* vLLM YAML. Invariant 5's promise is that what you read is what runs,
so a generator that reorders keys, drops a comment, or edits the wrong line breaks the
promise quietly — the config still looks fine, and only the engine knows the difference.

The fixture below is a real production config from the first GPU host, comments and inline
JSON included, because that is the input that actually has to survive.
"""

from __future__ import annotations

import pytest

from vllmbench_api.sweep_plan import (
    PlannedRun,
    SweepPlanError,
    config_family_text,
    engine_starts,
    expand,
    read_tensor_parallel_size,
    tensor_parallel_variant,
    validate_tensor_parallel,
)
from vllmbench_db.enums import ReplicateOrder

# Verbatim from ~/vllm-env/qwen3.8-27b-fp8.yaml on the real host.
REAL_CONFIG = """\
model: Qwen/Qwen3.8-27B-FP8
host: 0.0.0.0
port: 8000
served-model-name: Qwen3.8-27B

tensor-parallel-size: 2
max-model-len: 262144          # halve it unless agents truly need 256K
gpu-memory-utilization: 0.90   # 0.95 is risky with NCCL buffers on 24GB cards

max-num-seqs: 2                # was 1 — this is the fix
enable-chunked-prefill: true

kv-cache-dtype: fp8            # test this on Ampere, may not be supported

speculative-config: {"method":"mtp","num_speculative_tokens":3}
default-chat-template-kwargs: {"preserve_thinking": true}
"""


class TestTensorParallelVariant:
    def test_only_the_value_changes(self) -> None:
        """Everything except the one value must survive byte for byte."""
        out = tensor_parallel_variant(REAL_CONFIG, 4)

        assert "tensor-parallel-size: 4" in out
        assert "tensor-parallel-size: 2" not in out

        # Diff is exactly one line.
        before = REAL_CONFIG.splitlines()
        after = out.splitlines()
        assert len(before) == len(after)
        differing = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
        assert len(differing) == 1

    def test_comments_and_inline_json_survive(self) -> None:
        out = tensor_parallel_variant(REAL_CONFIG, 8)
        assert "# halve it unless agents truly need 256K" in out
        assert "# was 1 — this is the fix" in out
        assert 'speculative-config: {"method":"mtp","num_speculative_tokens":3}' in out
        # Blank lines are structure too; a YAML round-trip would eat them.
        assert "\n\ntensor-parallel-size" in out

    def test_trailing_comment_on_the_edited_line_is_kept(self) -> None:
        config = "model: m\ntensor-parallel-size: 2   # two cards\n"
        out = tensor_parallel_variant(config, 4)
        assert out == "model: m\ntensor-parallel-size: 4   # two cards\n"

    def test_underscore_spelling_is_edited_in_place(self) -> None:
        # vLLM accepts both spellings; the variant must not switch the author's.
        out = tensor_parallel_variant("model: m\ntensor_parallel_size: 1\n", 2)
        assert out == "model: m\ntensor_parallel_size: 2\n"

    def test_appended_when_absent(self) -> None:
        out = tensor_parallel_variant("model: m\nmax-num-seqs: 8\n", 2)
        assert out == "model: m\nmax-num-seqs: 8\ntensor-parallel-size: 2\n"

    def test_appended_when_the_file_has_no_trailing_newline(self) -> None:
        out = tensor_parallel_variant("model: m", 2)
        assert out == "model: m\ntensor-parallel-size: 2\n"

    def test_indented_occurrence_is_not_touched(self) -> None:
        """An indented key belongs to some nested structure, not to the engine.

        Editing it would change a setting the author never meant to expose as an axis,
        and the resulting config would not mean what the variant's name claims.
        """
        config = "model: m\nsomething:\n  tensor-parallel-size: 9\n"
        out = tensor_parallel_variant(config, 4)
        assert "  tensor-parallel-size: 9" in out
        assert out.endswith("tensor-parallel-size: 4\n")

    def test_commented_out_occurrence_is_not_touched(self) -> None:
        config = "model: m\n#tensor-parallel-size: 9\n"
        out = tensor_parallel_variant(config, 4)
        assert "#tensor-parallel-size: 9" in out
        assert "\ntensor-parallel-size: 4\n" in out

    def test_duplicate_keys_are_refused(self) -> None:
        # Which one vLLM honours is already ambiguous; picking one to edit would produce
        # a variant whose name lies about what it runs.
        config = "tensor-parallel-size: 1\nmodel: m\ntensor-parallel-size: 2\n"
        with pytest.raises(SweepPlanError, match="2 lines"):
            tensor_parallel_variant(config, 4)

    def test_crlf_line_endings_are_preserved(self) -> None:
        out = tensor_parallel_variant("model: m\r\ntensor-parallel-size: 2\r\n", 4)
        assert out == "model: m\r\ntensor-parallel-size: 4\r\n"
        assert "\n" not in out.replace("\r\n", "")  # no stray bare newlines introduced

    def test_zero_and_negative_are_refused(self) -> None:
        for bad in (0, -1):
            with pytest.raises(SweepPlanError, match="at least 1"):
                tensor_parallel_variant(REAL_CONFIG, bad)

    def test_variants_are_distinct_texts(self) -> None:
        # They are content-addressed downstream, so two TP values must not collide.
        assert tensor_parallel_variant(REAL_CONFIG, 1) != tensor_parallel_variant(REAL_CONFIG, 2)

    def test_round_trip_back_to_the_original(self) -> None:
        there = tensor_parallel_variant(REAL_CONFIG, 4)
        back = tensor_parallel_variant(there, 2)
        assert back == REAL_CONFIG


class TestReadTensorParallelSize:
    def test_reads_the_real_config(self) -> None:
        assert read_tensor_parallel_size(REAL_CONFIG) == 2

    def test_absent_is_none_not_one(self) -> None:
        # "not stated" and "stated as 1" are different facts about a config.
        assert read_tensor_parallel_size("model: m\n") is None

    def test_quoted_value(self) -> None:
        assert read_tensor_parallel_size('tensor-parallel-size: "4"\n') == 4

    def test_unparseable_value_is_none(self) -> None:
        assert read_tensor_parallel_size("tensor-parallel-size: auto\n") is None


class TestValidateTensorParallel:
    def test_accepts_what_the_host_has(self) -> None:
        validate_tensor_parallel(2, host_gpu_count=2, host_name="ubuntu-llm")

    def test_refuses_more_than_the_host_has(self) -> None:
        """Caught at authoring because the runtime failure is not clean.

        vLLM can come up on fewer devices than asked for, and the run would then be
        normalized per-GPU against a device count that never existed — a wrong number
        rather than a missing one.
        """
        with pytest.raises(SweepPlanError, match="exceeds the 2 GPU"):
            validate_tensor_parallel(4, host_gpu_count=2, host_name="ubuntu-llm")

    def test_unknown_device_count_does_not_block(self) -> None:
        # A host that has not been probed yet reports 0. Refusing then would block work
        # on a fact we simply do not have.
        validate_tensor_parallel(8, host_gpu_count=0, host_name="unprobed")


class TestExpand:
    def test_grouped_keeps_a_config_contiguous(self) -> None:
        plan = expand(config_count=2, workload_count=2, replicates=2)
        assert len(plan) == 8
        assert [p.config_index for p in plan] == [0, 0, 0, 0, 1, 1, 1, 1]
        # ...and a point's replicates are adjacent.
        assert [p.replicate_idx for p in plan[:4]] == [0, 1, 0, 1]

    def test_interleaved_repeats_the_whole_matrix(self) -> None:
        plan = expand(
            config_count=2, workload_count=2, replicates=2, order=ReplicateOrder.INTERLEAVED
        )
        assert len(plan) == 8
        assert [p.replicate_idx for p in plan] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_both_orderings_cover_the_same_matrix(self) -> None:
        # Same work, different sequence — an ordering that dropped or duplicated a point
        # would silently change what the sweep measured.
        grouped = expand(config_count=3, workload_count=2, replicates=3)
        interleaved = expand(
            config_count=3, workload_count=2, replicates=3, order=ReplicateOrder.INTERLEAVED
        )
        key = lambda p: (p.config_index, p.workload_index, p.replicate_idx)  # noqa: E731
        assert sorted(map(key, grouped)) == sorted(map(key, interleaved))
        assert len(set(map(key, grouped))) == len(grouped)

    def test_seq_is_dense_and_ordered(self) -> None:
        plan = expand(config_count=2, workload_count=3, replicates=2)
        assert [p.seq for p in plan] == list(range(len(plan)))

    def test_engine_starts_is_the_cost_difference(self) -> None:
        """The reason GROUPED is the default, made countable.

        Same matrix, same measurements; interleaving multiplies engine restarts by the
        replicate count, and a restart is minutes for a large model.
        """
        assert engine_starts(expand(config_count=4, workload_count=3, replicates=3)) == 4
        assert (
            engine_starts(
                expand(
                    config_count=4,
                    workload_count=3,
                    replicates=3,
                    order=ReplicateOrder.INTERLEAVED,
                )
            )
            == 12
        )

    def test_single_point_sweep(self) -> None:
        assert expand(config_count=1, workload_count=1, replicates=1) == [
            PlannedRun(seq=0, config_index=0, workload_index=0, replicate_idx=0)
        ]

    @pytest.mark.parametrize(
        ("configs", "workloads", "replicates", "message"),
        [
            (0, 1, 1, "server config"),
            (1, 0, 1, "workload"),
            (1, 1, 0, "replicates"),
        ],
    )
    def test_empty_axes_are_refused(
        self, configs: int, workloads: int, replicates: int, message: str
    ) -> None:
        with pytest.raises(SweepPlanError, match=message):
            expand(config_count=configs, workload_count=workloads, replicates=replicates)


class TestConfigFamily:
    """Grouping configs that differ only in tensor-parallel width.

    The scaling view is built on this: a curve assembled from configs that differ in
    anything else is a comparison of configurations wearing a scaling chart's clothes.
    """

    def test_widths_of_one_config_share_a_family(self) -> None:
        assert config_family_text(tensor_parallel_variant(REAL_CONFIG, 1)) == config_family_text(
            tensor_parallel_variant(REAL_CONFIG, 8)
        )

    def test_absent_and_explicit_one_are_the_same_config(self) -> None:
        # "no tensor-parallel line" and "tensor-parallel-size: 1" describe the same
        # engine, so they must land in the same family.
        assert config_family_text("model: m\nmax-num-seqs: 8\n") == config_family_text(
            "model: m\nmax-num-seqs: 8\ntensor-parallel-size: 1\n"
        )

    def test_position_of_the_line_does_not_change_the_family(self) -> None:
        """Why the line is deleted rather than rewritten to 1.

        Rewriting leaves the key where the author put it and appends it when absent, so
        two identical engines would land in different families purely because of where
        the line sat in the file.
        """
        first = config_family_text("tensor-parallel-size: 2\nmodel: m\nmax-num-seqs: 8\n")
        last = config_family_text("model: m\nmax-num-seqs: 8\ntensor-parallel-size: 4\n")
        assert first == last == "model: m\nmax-num-seqs: 8\n"

    def test_any_other_difference_splits_the_family(self) -> None:
        assert config_family_text("model: m\nmax-num-seqs: 8\n") != config_family_text(
            "model: m\nmax-num-seqs: 64\n"
        )

    def test_an_ambiguous_config_is_its_own_family(self) -> None:
        # Two top-level declarations: which one vLLM honours is already unknowable, so
        # this config stands alone rather than being folded in with a guess.
        ambiguous = "tensor-parallel-size: 1\nmodel: m\ntensor-parallel-size: 2\n"
        assert config_family_text(ambiguous) == ambiguous

    def test_comments_and_indented_occurrences_survive(self) -> None:
        config = "model: m\n#tensor-parallel-size: 9\nnested:\n  tensor-parallel-size: 4\n"
        assert config_family_text(config) == config
