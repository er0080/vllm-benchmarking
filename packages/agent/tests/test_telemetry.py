"""The telemetry sampler.

Two properties are being protected, and they pull against each other.

*The series must be honest.* A failed scrape is an absence, never a zero — a zero reads
as an idle engine and averages into a sweep as though it had been measured. And the
series must span the whole run: a timeline that stops early looks like an engine that
went quiet rather than a sampler that gave up.

*The sampler must not perturb what it measures.* It runs on the machine under test,
during the measurement. Bounded memory, no catch-up bursts, short timeouts.
"""

from __future__ import annotations

import asyncio
from collections import deque

import httpx
import pytest

from vllmbench_agent.telemetry import TelemetrySampler
from vllmbench_protocol.wire import EngineSampleWire, GpuSampleWire

LOADED_METRICS = """\
vllm:num_requests_running{engine="0",model_name="m"} 64.0
vllm:num_requests_waiting{engine="0",model_name="m"} 12.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 250.0
vllm:num_preemptions_total{engine="0",model_name="m"} 3.0
"""


def _transport(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestScraping:
    async def test_records_the_engine_state(self) -> None:
        sampler = TelemetrySampler(base_url="http://engine")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=LOADED_METRICS)

        async with _transport(handler) as client:
            sample = await sampler._sample_engine(client, offset=1.5)

        assert sample is not None
        assert sample.offset_seconds == 1.5
        assert sample.num_requests_running == 64
        # Not 0.0 from the _by_reason line that follows it.
        assert sample.num_requests_waiting == 12
        assert sample.kv_cache_usage_fraction == pytest.approx(0.42)
        assert sample.prefix_cache_queries_total == 1000

    async def test_a_failed_scrape_records_nothing(self) -> None:
        """Absence, not zero.

        Under saturation an engine can be too busy to answer /metrics — which is
        precisely when the timeline matters. Recording zeros there would draw a picture
        of an idle engine at the moment of peak load.
        """
        sampler = TelemetrySampler(base_url="http://engine")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("engine busy")

        async with _transport(handler) as client:
            assert await sampler._sample_engine(client, offset=0.0) is None

    async def test_a_non_200_records_nothing(self) -> None:
        sampler = TelemetrySampler(base_url="http://engine")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")

        async with _transport(handler) as client:
            assert await sampler._sample_engine(client, offset=0.0) is None

    async def test_an_empty_payload_records_nothing(self) -> None:
        # A 200 with no recognisable metrics means the contract moved. A sample of all
        # NULLs would be indistinguishable from an engine that measured nothing.
        sampler = TelemetrySampler(base_url="http://engine")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="# HELP only\n")

        async with _transport(handler) as client:
            assert await sampler._sample_engine(client, offset=0.0) is None

    async def test_scrapes_the_engine_not_the_agent(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, text=LOADED_METRICS)

        sampler = TelemetrySampler(base_url="http://127.0.0.1:8000/")
        async with _transport(handler) as client:
            await sampler._sample_engine(client, offset=0.0)

        # Trailing slash normalized, and no double slash — a 404 here would silently
        # produce a run with no telemetry at all.
        assert seen == ["http://127.0.0.1:8000/metrics"]


class TestDecimation:
    """Staying bounded without lying about the shape of the run."""

    def _fill(self, sampler: TelemetrySampler, ticks: int, devices: list[int]) -> None:
        for tick in range(ticks):
            sampler._engine.append((tick, EngineSampleWire(offset_seconds=float(tick))))
            for device in devices:
                sampler._gpu.append(
                    (tick, GpuSampleWire(offset_seconds=float(tick), gpu_index=device))
                )
            sampler._tick_index = tick + 1

    def test_drops_whole_ticks_never_whole_devices(self) -> None:
        """The bug this exists to prevent.

        Each tick appends one sample per device, so thinning the flat list by position
        keeps device 0 and deletes device 1 entirely — presenting as a resolution
        reduction while actually erasing half the topology. On a TP=2 run that is the
        whole point of per-device sampling, gone.
        """
        sampler = TelemetrySampler(base_url="http://engine", device_indices=[0, 1])
        self._fill(sampler, ticks=100, devices=[0, 1])

        sampler._decimate()

        devices = {sample.gpu_index for sample in sampler.gpu_samples}
        assert devices == {0, 1}, "a device was dropped entirely"
        # Both devices keep the same number of samples as each other.
        counts = [sum(1 for s in sampler.gpu_samples if s.gpu_index == d) for d in (0, 1)]
        assert counts[0] == counts[1]

    def test_repeated_decimation_keeps_shrinking(self) -> None:
        """The second bug: a `tick % 2` filter works once and then never again.

        After one pass every surviving tick is even, so the buffer stops shrinking while
        the cap keeps firing — unbounded memory on the machine under test, which is the
        exact failure the cap exists to prevent.
        """
        sampler = TelemetrySampler(base_url="http://engine", device_indices=[0])
        self._fill(sampler, ticks=64, devices=[0])

        sizes = []
        for _ in range(3):
            sampler._decimate()
            sizes.append(len(sampler.engine_samples))

        assert sizes[0] > sizes[1] > sizes[2], f"stopped shrinking: {sizes}"

    def test_span_is_preserved_not_truncated(self) -> None:
        # Keeping the first N samples would end the timeline partway through the run,
        # which reads as "the engine went quiet".
        sampler = TelemetrySampler(base_url="http://engine", device_indices=[0])
        self._fill(sampler, ticks=100, devices=[0])
        last_before = sampler.engine_samples[-1].offset_seconds

        sampler._decimate()

        assert sampler.engine_samples[0].offset_seconds == 0.0
        # Still reaches the end of the run, within one retained interval.
        assert sampler.engine_samples[-1].offset_seconds >= last_before - 2

    def test_reported_interval_tracks_the_real_spacing(self) -> None:
        # A consumer differencing counters between adjacent samples divides by this. If
        # it kept saying 1s after decimation, every derived rate would be double.
        sampler = TelemetrySampler(base_url="http://engine", interval_seconds=1.0)
        self._fill(sampler, ticks=32, devices=[])

        sampler._decimate()
        assert sampler.effective_interval_seconds == 2.0
        sampler._decimate()
        assert sampler.effective_interval_seconds == 4.0
        assert sampler.decimated is True

    def test_not_flagged_when_untouched(self) -> None:
        sampler = TelemetrySampler(base_url="http://engine")
        assert sampler.decimated is False
        assert sampler.effective_interval_seconds == 1.0


class TestLifecycle:
    async def test_stop_is_idempotent_and_safe_before_start(self) -> None:
        # The bench endpoint stops the sampler in a `finally`, which runs even when the
        # benchmark failed before anything was sampled.
        sampler = TelemetrySampler(base_url="http://engine")
        await sampler.stop()
        await sampler.stop()

    async def test_samples_accumulate_while_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=LOADED_METRICS)

        # A real client over a mock transport, so the response carries its request and
        # raise_for_status behaves exactly as it does in production.
        #
        # `telemetry.httpx` is the httpx module itself, so the replacement must close
        # over the original class — referring to httpx.AsyncClient inside it would call
        # the patched version and recurse.
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler)),
        )

        sampler = TelemetrySampler(base_url="http://engine", interval_seconds=0.05)
        sampler.start()
        await asyncio.sleep(0.3)
        await sampler.stop()

        assert len(sampler.engine_samples) >= 3
        assert sampler.engine_samples[0].num_requests_running == 64
        # Offsets increase, and start near zero rather than at a wall clock.
        offsets = [s.offset_seconds for s in sampler.engine_samples]
        assert offsets == sorted(offsets)
        assert offsets[0] < 0.1

    def test_interval_has_a_floor(self) -> None:
        # A zero or negative interval would spin the event loop on the machine under
        # test, which is a far worse perturbation than any sampling.
        assert TelemetrySampler(base_url="http://e", interval_seconds=0)._interval >= 0.05
        assert TelemetrySampler(base_url="http://e", interval_seconds=-5)._interval >= 0.05


class TestGpuSampling:
    def test_no_devices_means_no_samples(self) -> None:
        # A host with no GPUs is a supported development configuration, not an error.
        sampler = TelemetrySampler(base_url="http://engine", device_indices=[])
        assert sampler._sample_gpus(offset=0.0) == []

    def test_missing_nvml_degrades_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def no_pynvml(name: str, *args: object, **kwargs: object):
            if name == "pynvml":
                raise ImportError("no nvml here")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_pynvml)
        sampler = TelemetrySampler(base_url="http://engine", device_indices=[0, 1])
        assert sampler._sample_gpus(offset=0.0) == []


def test_buffers_start_empty() -> None:
    sampler = TelemetrySampler(base_url="http://engine")
    assert isinstance(sampler._engine, deque)
    assert sampler.engine_samples == []
    assert sampler.gpu_samples == []
