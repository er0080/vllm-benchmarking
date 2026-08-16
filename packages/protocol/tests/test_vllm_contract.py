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

pytestmark = pytest.mark.vllm_cpu

BASE_URL = os.environ.get("VLLM_CPU_BASE_URL", "http://localhost:8100")
BENCH_RESULT_PATH = os.environ.get("VLLM_BENCH_RESULT")


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
        missing = set(SUMMARY_FIELD_MAP) - live_payload.keys()
        assert not missing, (
            f"this vLLM version no longer emits: {sorted(missing)}. "
            "SUMMARY_FIELD_MAP is now wrong and would write NULLs."
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
