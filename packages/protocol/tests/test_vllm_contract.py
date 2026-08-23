"""CI tier 2: re-derive the vLLM contract from a live server.

The unit tests validate our assumptions against a captured payload. These validate the
capture itself is still true. Without them, a vLLM upgrade that renames a field would
leave every unit test green while production wrote NULLs — which is the failure mode this
tier exists to prevent, and the one that already happened once during development.

Requires a running ``vllm/vllm-openai-cpu`` container. Skipped otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from vllmbench_protocol.bench_result import SUMMARY_FIELD_MAP, flatten_bench_result
from vllmbench_protocol.metrics import ALL_METRICS
from vllmbench_protocol.server_info import NO_SPECULATION, parse_speculation

#: Emitted only when the engine is drafting, so absent from the plain server's payload and
#: required in the speculating one's. Both halves are asserted — see TestSpeculativeContract.
SPECULATIVE_SOURCES = frozenset(k for k in SUMMARY_FIELD_MAP if k.startswith("spec_decode_"))

pytestmark = pytest.mark.vllm_cpu

BASE_URL = os.environ.get("VLLM_CPU_BASE_URL", "http://localhost:8100")
BENCH_RESULT_PATH = os.environ.get("VLLM_BENCH_RESULT")

#: A second engine, started with ngram speculation. Separate because speculation is fixed
#: at engine start and both payload shapes are part of the contract.
SPECULATIVE_BASE_URL = os.environ.get("VLLM_CPU_SPECULATIVE_BASE_URL", "http://localhost:8101")
SPECULATIVE_BENCH_RESULT_PATH = os.environ.get("VLLM_BENCH_RESULT_SPECULATIVE")


@pytest.fixture
def metrics_text() -> str:
    try:
        response = httpx.get(f"{BASE_URL}/metrics", timeout=30.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"no vLLM CPU backend at {BASE_URL}: {exc}")
    response.raise_for_status()
    return response.text


class TestServerContract:
    def test_health_endpoint_exists(self) -> None:
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=30.0)
        except httpx.HTTPError as exc:
            pytest.skip(f"no vLLM CPU backend at {BASE_URL}: {exc}")
        # The agent polls this to decide a server is ready. If it moves, every run
        # would start benchmarking against a model that has not finished loading.
        assert response.status_code == 200

    def test_models_endpoint_reports_the_served_model(self) -> None:
        try:
            response = httpx.get(f"{BASE_URL}/v1/models", timeout=30.0)
        except httpx.HTTPError as exc:
            pytest.skip(f"no vLLM CPU backend at {BASE_URL}: {exc}")
        ids = [m["id"] for m in response.json()["data"]]
        assert ids


class TestMetricsContract:
    def test_every_metric_we_scrape_is_exposed(self, metrics_text: str) -> None:
        exposed = {
            line.split()[2] for line in metrics_text.splitlines() if line.startswith("# TYPE vllm:")
        }
        missing = set(ALL_METRICS) - exposed
        assert not missing, (
            f"vLLM no longer exposes: {sorted(missing)}. "
            "Update vllmbench_protocol.metrics rather than working around it."
        )

    def test_there_is_still_no_prefix_cache_hit_rate_gauge(self, metrics_text: str) -> None:
        """Guards the reasoning, not just the names.

        Storing the two counters instead of a rate is a deliberate choice. If upstream
        ever adds a rate gauge this fails, and the choice gets revisited on purpose
        rather than by accident.
        """
        assert "vllm:prefix_cache_hit_rate" not in metrics_text
        assert "vllm:prefix_cache_queries_total" in metrics_text
        assert "vllm:prefix_cache_hits_total" in metrics_text


class TestBenchResultContract:
    """Validates a result produced by the live server in this CI run."""

    @pytest.fixture
    def live_payload(self) -> dict[str, object]:
        if not BENCH_RESULT_PATH:
            pytest.skip("VLLM_BENCH_RESULT not set; no live benchmark result to check")
        path = Path(BENCH_RESULT_PATH)
        if not path.is_file():
            pytest.skip(f"benchmark result not found at {path}")
        return json.loads(path.read_text())

    def test_all_mapped_fields_are_present_in_live_output(
        self, live_payload: dict[str, object]
    ) -> None:
        """Every non-conditional mapped field must still come out of a live server.

        Asserted as an equality against the speculative set rather than as a subset. The
        speculative fields are legitimately absent — this backend is not drafting — but a
        subset check would also swallow `p99_ttft_ms` disappearing, which is the rename
        this tier exists to catch.

        The speculative names are covered by :class:`TestSpeculativeContract`, against a
        second engine that is actually drafting.
        """
        missing = set(SUMMARY_FIELD_MAP) - live_payload.keys()
        assert missing == set(SPECULATIVE_SOURCES), (
            f"this vLLM version no longer emits: {sorted(missing - SPECULATIVE_SOURCES)}. "
            "SUMMARY_FIELD_MAP is now wrong and would write NULLs."
        )

    def test_the_live_payload_is_not_secretly_speculating(
        self, live_payload: dict[str, object]
    ) -> None:
        """If this backend ever starts drafting, the test above becomes wrong silently.

        It would then be asserting that fields which *are* present are absent, and would
        fail confusingly. This says why first.
        """
        present = set(SPECULATIVE_SOURCES) & live_payload.keys()
        assert not present, (
            f"the plain backend is now speculating and emits {sorted(present)}, so the "
            "equality above is asserting that present fields are absent. Something is "
            "starting this engine with a drafter that nothing asked for."
        )

    def test_live_output_flattens_without_error(self, live_payload: dict[str, object]) -> None:
        flat = flatten_bench_result(live_payload, gpu_count=1)
        assert flat["successful_requests"] is not None
        assert flat["benchmark_duration_sec"] is not None

    def test_checked_in_fixture_still_matches_live_shape(
        self, live_payload: dict[str, object]
    ) -> None:
        """Catches fields *added* upstream, which the other tests would not.

        A new field is not a failure, but it is a prompt: it may be something worth
        promoting to a column instead of leaving in `extra`.
        """
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "bench_serve_v0.25.1.json").read_text()
        )
        added = live_payload.keys() - fixture.keys()
        assert not added, (
            f"vLLM now emits fields the fixture does not have: {sorted(added)}. "
            "Refresh the fixture and consider promoting them to columns."
        )


class TestSpeculativeContract:
    """The other half of the payload contract, against an engine that is drafting.

    `--save-result` emits the six `spec_decode_*` fields **only when drafts happened**, and
    omits them entirely otherwise — so the plain server above can never exercise those names,
    and until this existed they were pinned by a captured fixture alone. A fixture does not
    move when upstream does, which is the whole failure mode this tier exists to prevent
    (issue #85).

    Two things had to be true for this to be reachable at all, and both were measured rather
    than assumed. ngram needs no draft model, so it runs on a CPU backend with dummy weights.
    And drafting needs repetition *within* a sequence: with sampled decoding over random
    prompts the engine speculates and never once drafts — 8 prompts, zero drafts, no fields
    in the payload. Greedy decoding against dummy weights loops, which supplies it.
    """

    @pytest.fixture
    def speculative_payload(self) -> dict[str, object]:
        if not SPECULATIVE_BENCH_RESULT_PATH:
            pytest.skip("VLLM_BENCH_RESULT_SPECULATIVE not set")
        path = Path(SPECULATIVE_BENCH_RESULT_PATH)
        if not path.is_file():
            pytest.skip(f"speculative benchmark result not found at {path}")
        return json.loads(path.read_text())

    def test_every_speculative_field_name_still_exists(
        self, speculative_payload: dict[str, object]
    ) -> None:
        """The names in SUMMARY_FIELD_MAP, re-derived from a server rather than a fixture.

        A rename here writes NULL into every speculative column of every run, and nothing
        errors while it happens: a summary row full of NULLs is indistinguishable from a
        benchmark that legitimately measured nothing.
        """
        missing = set(SPECULATIVE_SOURCES) - speculative_payload.keys()
        assert not missing, (
            f"this vLLM version no longer emits: {sorted(missing)}. SUMMARY_FIELD_MAP is "
            "now wrong for every speculative run and would write NULLs."
        )

    def test_it_actually_drafted(self, speculative_payload: dict[str, object]) -> None:
        """Otherwise the test above is vacuous in the way that matters.

        An engine configured to speculate that never finds a draft emits none of the fields,
        so a payload with zero drafts would fail the assertion above for a reason that has
        nothing to do with upstream renaming anything. This says which it is.
        """
        assert speculative_payload.get("spec_decode_num_drafts", 0) > 0, (
            "the speculating engine produced no drafts, so this payload cannot pin the "
            "field names. The benchmark needs prompts ngram can match — greedy decoding "
            "is what supplies the repetition here."
        )

    def test_the_engine_reports_what_it_is_doing(self) -> None:
        """`/server_info` is where a run's speculation provenance comes from, so a change
        in its shape must fail here rather than turning every run's method into NULL."""
        try:
            response = httpx.get(
                f"{SPECULATIVE_BASE_URL}/server_info",
                params={"config_format": "json"},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            pytest.skip(f"no speculating backend at {SPECULATIVE_BASE_URL}: {exc}")
        response.raise_for_status()
        found = parse_speculation(response.json())
        assert found is not None, "/server_info no longer describes speculation as we parse it"
        assert (found.method, found.tokens) == ("ngram", 3), (
            f"the engine reports {found}, which is not what this job asked it to run"
        )

    def test_a_plain_engine_says_it_is_not_speculating(self) -> None:
        """The three-state distinction, live: "no" must stay different from "no answer".

        If vLLM ever omits `speculative_config` instead of nulling it, the parser starts
        returning None here — correctly, since it can no longer tell — and this fails while
        the difference still matters, rather than after a chart has been drawn on it.
        """
        try:
            response = httpx.get(
                f"{BASE_URL}/server_info", params={"config_format": "json"}, timeout=60.0
            )
        except httpx.HTTPError as exc:
            pytest.skip(f"no vLLM CPU backend at {BASE_URL}: {exc}")
        response.raise_for_status()
        found = parse_speculation(response.json())
        assert found is not None
        assert found.method == NO_SPECULATION
        assert not found.is_speculating


class TestTheEndpointsWeDependOn:
    """The gated endpoints, held to still being there.

    They moved behind `VLLM_SERVER_DEV_MODE` once and it went unnoticed for months, because
    a 404 to a cache reset looks exactly like a version that never had the endpoint (issue
    #87). Both containers in this job set the variable, as the agent does.
    """

    @pytest.mark.parametrize(
        "endpoint", ["/reset_prefix_cache", "/reset_mm_cache", "/reset_encoder_cache"]
    )
    def test_each_reset_endpoint_answers(self, endpoint: str) -> None:
        try:
            response = httpx.post(f"{BASE_URL}{endpoint}", timeout=30.0)
        except httpx.HTTPError as exc:
            pytest.skip(f"no vLLM CPU backend at {BASE_URL}: {exc}")
        assert response.status_code < 400, (
            f"{endpoint} answered {response.status_code}. The agent depends on this "
            "between sweep points; without it a warm prefix cache carries from one "
            "configuration to the next and the ordering of a matrix decides its winner."
        )
