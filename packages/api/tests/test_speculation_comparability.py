"""Charting speculating and non-speculating runs together.

Inter-token latency is the wait between *emissions*. Without speculation each emission
carries one token, so it tracks TPOT. Under speculation an emission carries however many
drafted tokens the target accepted, so the gap grows while the tokens arrive faster.

Measured on `ubuntu-llm` across four MTP depths, median ITL went 25.0 → 33.9 → 42.6 →
44.1 ms while per-user throughput went 39.6 → 54.0 → 59.1 → 69.4 tok/s. 44.1 ms across 3.16
accepted tokens is 14.0 ms per token, which is that run's measured TPOT of 14.4. Neither
number is wrong. A chart with both on one ITL axis shows the fastest configuration as 76%
worse, with nothing saying the bars measure different things (issue #86).

The fix is *not* to drop ITL. Three tokens every 44 ms reads differently in a streaming UI
from one every 25 ms, and for an interactive coding assistant that is worth measuring. The
fix is to say what is being compared.
"""

from __future__ import annotations

import datetime as dt
import uuid

from vllmbench_api.analysis import (
    EMISSION_BASED_METRICS,
    METRICS_BY_KEY,
    ComparabilityKey,
    RunRecord,
    comparability_key,
    group_warnings,
    speculation_warning,
)
from vllmbench_protocol import NO_SPECULATION

HOST = uuid.uuid4()


def run(method: str | None = NO_SPECULATION, tokens: int | None = 0, **overrides) -> RunRecord:
    fields = {
        "run_id": uuid.uuid4(),
        "finished_at": dt.datetime.now(dt.UTC),
        "gpu_host_id": HOST,
        "gpu_host_name": "ubuntu-llm",
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "vllm_version": "0.25.1",
        "bench_client_location": "loopback",
        "speculative_method": method,
        "speculative_tokens": tokens,
    }
    fields.update(overrides)
    return RunRecord(**fields)


class TestTheEmissionAxisIsIdentifiedFromTheMetricTable:
    def test_itl_is_marked_and_tpot_is_not(self) -> None:
        """TPOT counts tokens and stays comparable; ITL counts emissions and does not.
        Getting this backwards would warn on the one axis that is actually safe."""
        assert METRICS_BY_KEY["itl_ms_median"].emission_based
        assert METRICS_BY_KEY["itl_ms_p99"].emission_based
        assert not METRICS_BY_KEY["tpot_ms_mean"].emission_based
        assert not METRICS_BY_KEY["per_user_output_tok_s"].emission_based

    def test_the_warning_names_metrics_from_that_table(self) -> None:
        """One list, not two. A second hardcoded list drifts away from the first, and the
        drift is invisible: the warning simply stops mentioning a metric it should."""
        warning = speculation_warning([run(), run("mtp", 3)])
        assert warning is not None
        for key in EMISSION_BASED_METRICS:
            assert METRICS_BY_KEY[key].label in warning


class TestWhenItFires:
    def test_speculating_and_not_is_flagged(self) -> None:
        assert speculation_warning([run(), run("mtp", 3)]) is not None

    def test_two_depths_are_flagged_too(self) -> None:
        """Both arms speculate, and their emissions still carry different token counts.
        Measured: 33.9 ms at depth 1 against 44.1 ms at depth 3, both faster than off."""
        assert speculation_warning([run("mtp", 1), run("mtp", 3)]) is not None

    def test_two_methods_at_one_depth_are_flagged(self) -> None:
        """Acceptance differs by method, so tokens per emission does too."""
        assert speculation_warning([run("mtp", 3), run("ngram", 3)]) is not None

    def test_the_message_says_what_to_use_instead(self) -> None:
        """A warning that only says "careful" makes the reader do the work again."""
        warning = speculation_warning([run(), run("mtp", 3)])
        assert warning is not None and "TPOT" in warning

    def test_it_reaches_the_group(self) -> None:
        assert any("emission" in w for w in group_warnings([run(), run("mtp", 3)]))


class TestWhenItStaysQuiet:
    def test_one_arm_is_not_a_comparison(self) -> None:
        assert speculation_warning([run("mtp", 3), run("mtp", 3)]) is None

    def test_all_non_speculating_is_not_a_comparison(self) -> None:
        """The overwhelmingly common case. A warning on every chart is not a warning."""
        assert speculation_warning([run(), run(), run()]) is None

    def test_an_empty_group_says_nothing(self) -> None:
        assert speculation_warning([]) is None

    def test_a_whole_group_of_unknowns_says_nothing(self) -> None:
        """Every run recorded before protocol 7. They are all equally unknown, so nothing
        is being compared across a boundary — and warning on the entire existing history
        would train the reader to ignore the warning by the time it matters."""
        assert speculation_warning([run(None, None), run(None, None)]) is None


class TestUnknownIsItsOwnAnswer:
    def test_an_unknown_beside_a_known_is_flagged(self) -> None:
        """Not because the two definitely differ, but because nothing here can say they
        do not — which is the honest statement and the one the reader needs."""
        warning = speculation_warning([run(None, None), run("mtp", 3)])
        assert warning is not None and "unknown" in warning

    def test_an_unknown_is_not_read_as_not_speculating(self) -> None:
        """The tempting shortcut, and the one that destroys the column's value: a run
        nobody asked would be filed in the non-speculative arm and compared as evidence."""
        assert speculation_warning([run(None, None), run()]) is not None

    def test_the_message_counts_them(self) -> None:
        warning = speculation_warning([run(None, None), run(None, None), run("mtp", 3)])
        assert warning is not None and "2 run(s)" in warning


class TestSpeculationIsNotAPartition:
    def test_it_is_absent_from_the_comparability_key(self) -> None:
        """Deliberate. Comparing speculation settings is *why* someone runs this sweep —
        partitioning them into separate charts would break the tool's headline use, the
        way partitioning on config hash would. Almost every metric compares fine across
        the boundary; the emission-based ones get a warning, not a wall."""
        assert not any("spec" in f for f in ComparabilityKey.__slots__)

    def test_two_speculation_settings_share_a_series(self) -> None:
        assert comparability_key(run()) == comparability_key(run("mtp", 3))
