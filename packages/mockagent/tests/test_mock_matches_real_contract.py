"""The mock must be indistinguishable from the real agent except in what it reports.

CLAUDE.md: "when the real agent's contract changes, the mock changes in the same PR."
That is a rule someone has to remember, so these tests make forgetting it fail loudly.
A mock that has drifted is worse than no mock: it makes tests pass while the real path
is broken.
"""

from __future__ import annotations

import httpx
import pytest

from vllmbench_agent.main import create_app as create_real_app
from vllmbench_agent.settings import AgentSettings
from vllmbench_mockagent.main import SYNTHETIC_SOURCE
from vllmbench_mockagent.main import create_app as create_mock_app
from vllmbench_protocol import PROTOCOL_VERSION, AgentClient
from vllmbench_protocol.wire import HealthResponse, HostInfo

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture
def real_app():
    return create_real_app(AgentSettings(token=TOKEN))


@pytest.fixture
def mock_app():
    return create_mock_app(token=TOKEN)


@pytest.fixture
def mock_app_factory():
    """Builds a *fresh* mock, for tests about state that accumulates across requests."""
    return lambda: create_mock_app(token=TOKEN)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://agent",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


# FastAPI's own plumbing, not part of the agent contract.
FRAMEWORK_ROUTES = {"/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}


def _contract_routes(app) -> set[tuple[str, tuple[str, ...]]]:
    return {
        (r.path, tuple(sorted(r.methods)))
        for r in app.routes
        if hasattr(r, "methods") and r.path not in FRAMEWORK_ROUTES
    }


class TestContractParity:
    async def test_both_expose_the_same_routes(self, real_app, mock_app) -> None:
        """Compares the *whole* contract, not a hardcoded subset.

        An earlier version listed the two endpoints that existed at the time, so when
        the real agent gained server lifecycle and benchmarking the mock silently fell
        behind and this still passed. A parity test with an allowlist is a parity test
        that stops working the moment it would first be useful.
        """
        real = _contract_routes(real_app)
        mock = _contract_routes(mock_app)
        assert real == mock, (
            f"only in real agent: {sorted(real - mock)}; only in mock: {sorted(mock - real)}"
        )

    async def test_both_validate_against_the_same_models(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with _client(app) as http:
                HealthResponse.model_validate((await http.get("/health")).json())
                HostInfo.model_validate((await http.get("/host-info")).json())

    async def test_both_enforce_authentication_identically(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://agent"
            ) as http:
                assert (await http.get("/host-info")).status_code == 401
                bad = await http.get("/host-info", headers={"Authorization": "Bearer nope"})
                assert bad.status_code == 403

    async def test_both_report_the_same_protocol_version(self, real_app, mock_app) -> None:
        for app in (real_app, mock_app):
            async with _client(app) as http:
                info = HostInfo.model_validate((await http.get("/host-info")).json())
                assert info.protocol_version == PROTOCOL_VERSION

    async def test_the_shared_client_works_against_the_mock(self, mock_app) -> None:
        async with _client(mock_app) as http:
            info = await AgentClient("http://agent", TOKEN, client=http).host_info()
            assert info.gpu_count == 2


class TestMockDeclaresItselfSynthetic:
    async def test_mock_sets_synthetic_source(self, mock_app) -> None:
        # The load-bearing difference. Everything the control plane does to quarantine
        # these runs keys off this one field being set by the producer.
        async with _client(mock_app) as http:
            info = HostInfo.model_validate((await http.get("/host-info")).json())
            assert info.synthetic_source == SYNTHETIC_SOURCE

    async def test_real_and_mock_differ_only_in_that_field(self, real_app, mock_app) -> None:
        async with _client(real_app) as http:
            real = HostInfo.model_validate((await http.get("/host-info")).json())
        async with _client(mock_app) as http:
            mock = HostInfo.model_validate((await http.get("/host-info")).json())

        assert real.synthetic_source is None
        assert mock.synthetic_source is not None

        # Both answer the environment question. It is optional on the wire so that an
        # agent which could not run the check still produces a valid payload, and that
        # optionality is exactly what would let the mock quietly stop reporting — leaving
        # the control plane's handling of a conflicting environment unreachable in every
        # test, on the one path that has no other way to be exercised.
        assert real.environment is not None
        assert mock.environment is not None


class TestMockIsMultiGpu:
    async def test_mock_reports_more_than_one_device(self, mock_app) -> None:
        """Two devices, not one.

        A single-device mock would let per-device handling — the TP-imbalance charts,
        the per-GPU normalization — break without any test noticing, because everything
        degenerates correctly at gpu_count=1.
        """
        async with _client(mock_app) as http:
            info = HostInfo.model_validate((await http.get("/host-info")).json())
            assert info.gpu_count >= 2
            assert [g.index for g in info.gpus] == list(range(info.gpu_count))


class TestSyntheticResultsAreUsable:
    """The mock has to be good enough to build charts against, not merely well-formed."""

    async def test_bench_requires_a_ready_server(self, mock_app) -> None:
        # Mirrors the real agent's refusal. Control-plane code that handles this must be
        # exercised somewhere that does not need a GPU.
        async with _client(mock_app) as http:
            response = await http.post("/bench", json={"model": "m"})
            assert response.status_code == 409

    async def test_full_lifecycle_produces_a_flattenable_result(self, mock_app) -> None:
        from vllmbench_protocol.bench_result import flatten_bench_result

        async with _client(mock_app) as http:
            start = await http.post(
                "/server/start",
                json={"config_yaml": "model: m\n", "config_hash": "c" * 16, "port": 8000},
            )
            assert start.status_code == 200

            bench = await http.post(
                "/bench", json={"model": "m", "num_prompts": 64, "max_concurrency": 16}
            )
            assert bench.status_code == 200
            raw = bench.json()["raw_result"]

            # The load-bearing property: synthetic results must survive the same
            # flattening as real ones. A mock emitting different field names would let a
            # flattening bug reach production untested.
            flat = flatten_bench_result(raw, gpu_count=1)
            assert flat["successful_requests"] == 64
            assert flat["ttft_ms_p99"] > flat["ttft_ms_mean"]

    async def test_throughput_saturates_rather_than_growing_without_bound(self, mock_app) -> None:
        """Charts need a knee, or the Pareto view has nothing to show.

        Doubling concurrency well past saturation must not double throughput. If it did,
        every configuration would look better than the last and the frontier would be a
        straight line.
        """
        async with _client(mock_app) as http:
            await http.post(
                "/server/start",
                json={"config_yaml": "model: m\n", "config_hash": "c" * 16, "port": 8000},
            )
            low = (await http.post("/bench", json={"model": "m", "max_concurrency": 64})).json()[
                "raw_result"
            ]["output_throughput"]
            high = (await http.post("/bench", json={"model": "m", "max_concurrency": 128})).json()[
                "raw_result"
            ]["output_throughput"]

        assert high > low * 0.95
        assert high < low * 1.3

    async def test_latency_degrades_under_load(self, mock_app) -> None:
        async with _client(mock_app) as http:
            await http.post(
                "/server/start",
                json={"config_yaml": "model: m\n", "config_hash": "c" * 16, "port": 8000},
            )
            light = (await http.post("/bench", json={"model": "m", "max_concurrency": 8})).json()[
                "raw_result"
            ]["mean_ttft_ms"]
            heavy = (await http.post("/bench", json={"model": "m", "max_concurrency": 256})).json()[
                "raw_result"
            ]["mean_ttft_ms"]
        assert heavy > light * 2

    async def test_tensor_parallel_size_is_echoed_from_the_config(self, mock_app) -> None:
        # Lets TP sweeps and per-GPU normalization be exercised with no hardware, which
        # is the only way invariant 8's charting can be built before a GPU host exists.
        async with _client(mock_app) as http:
            await http.post(
                "/server/start",
                json={
                    "config_yaml": "model: m\ntensor_parallel_size: 4\n",
                    "config_hash": "c" * 16,
                    "port": 8000,
                },
            )
            bench = (await http.post("/bench", json={"model": "m"})).json()
        assert bench["tensor_parallel_size"] == 4
        assert bench["device_indices"] == [0, 1, 2, 3]


class TestReplicatesDiffer:
    """Repeating a point must move the numbers, or the spread rendering is untestable.

    Every analysis view draws a band across replicates and states what it measures. With
    a mock that returns byte-identical results for repeated requests, that band is always
    zero-width in development — so the one piece of the chart that exists to convey
    uncertainty is the piece nobody can see working before it meets real data.
    """

    @staticmethod
    async def _bench(http, *, concurrency: int = 16) -> dict:
        response = await http.post(
            "/bench", json={"model": "m", "num_prompts": 64, "max_concurrency": concurrency}
        )
        assert response.status_code == 200
        return response.json()["raw_result"]

    async def _ready(self, http) -> None:
        start = await http.post(
            "/server/start",
            json={"config_yaml": "model: m\n", "config_hash": "c" * 16, "port": 8000},
        )
        assert start.status_code == 200

    async def test_repeated_runs_of_one_point_vary(self, mock_app) -> None:
        async with _client(mock_app) as http:
            await self._ready(http)
            runs = [await self._bench(http) for _ in range(3)]

        throughputs = [r["output_throughput"] for r in runs]
        assert len(set(throughputs)) == 3, "replicates must not be identical"
        # Varied, but still recognisably the same measurement — a spread this wide would
        # mean the mock was generating noise rather than plausible repeat measurements.
        spread = (max(throughputs) - min(throughputs)) / min(throughputs)
        assert 0 < spread < 0.25, spread

    async def test_variation_is_reproducible_across_processes(self, mock_app_factory) -> None:
        """Deterministic, so a test that asserts on mock output stays stable.

        The variation comes from a per-process counter, not a random source: the same
        sequence of requests to a fresh mock yields the same sequence of numbers.
        """
        first = []
        second = []
        for sink in (first, second):
            async with _client(mock_app_factory()) as http:
                await self._ready(http)
                for _ in range(2):
                    sink.append((await self._bench(http))["output_throughput"])

        assert first == second

    async def test_different_points_are_independent(self, mock_app) -> None:
        # Replicate 0 of one workload must not be affected by how many times a different
        # workload has been run, or the sweep order would change the numbers.
        async with _client(mock_app) as http:
            await self._ready(http)
            a_first = await self._bench(http, concurrency=8)
            await self._bench(http, concurrency=32)
            await self._bench(http, concurrency=32)

        async with _client(create_mock_app(token=TOKEN)) as http:
            await self._ready(http)
            a_alone = await self._bench(http, concurrency=8)

        assert a_first["output_throughput"] == a_alone["output_throughput"]
