"""Sampling the engine and the GPUs while a benchmark runs.

This code runs on the machine under test, during the measurement, which makes its own
cost part of the contract (CLAUDE.md, "Working on the agent"): telemetry sampling must
not be a measurable perturbation of the thing it measures. Everything here is shaped by
that.

- One HTTP GET to loopback and one NVML read per device, per tick. At the 1s default on
  a two-GPU host that is three cheap operations a second against an engine doing
  thousands of token generations in the same window.
- Bounded memory. A sweep point can legitimately run for hours, and unbounded buffering
  on the box whose behaviour we are protecting is a leak in the worst possible place.
- A tick that overruns is skipped, not queued. Falling behind must not turn into a burst
  of catch-up scrapes at exactly the moment the engine is already struggling.
- Failures are absences, never zeros. A scrape that fails records nothing for that field;
  writing 0 would be indistinguishable from an idle engine, and would average into a
  sweep as though it had been measured.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from typing import Any

import httpx

from vllmbench_protocol.metrics import parse_metrics
from vllmbench_protocol.wire import EngineSampleWire, GpuSampleWire

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 1.0

# Roughly two hours at the default interval. Past this the series is decimated rather
# than truncated — see _decimate.
MAX_ENGINE_SAMPLES = 7200

# A short timeout on purpose. If /metrics cannot answer in a second the engine is busy,
# and waiting longer to find that out both delays the next tick and adds load.
SCRAPE_TIMEOUT_SECONDS = 1.0


def _as_int(value: float | None) -> int | None:
    """Counters and queue depths arrive as floats from Prometheus."""
    return None if value is None else int(value)


class TelemetrySampler:
    """Collects engine and per-device samples for the duration of one benchmark.

    Offsets are recorded relative to the sampler's own start rather than as wall-clock
    timestamps. The control plane anchors them to the run, so the two hosts' clocks
    disagreeing cannot slide a timeline out of alignment with the window it describes.
    """

    def __init__(
        self,
        *,
        base_url: str,
        device_indices: list[int] | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._device_indices = list(device_indices or [])
        self._interval = max(0.05, interval_seconds)

        # Tagged with the tick that produced them. Decimation has to drop whole ticks:
        # a two-GPU host appends two GPU samples per tick, so thinning the flat list by
        # position would keep device 0 and delete device 1 entirely — turning a
        # resolution reduction into the silent loss of half the topology.
        self._engine: deque[tuple[int, EngineSampleWire]] = deque()
        self._gpu: deque[tuple[int, GpuSampleWire]] = deque()
        self._tick_index = 0
        self._task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._decimated = False
        # Doubles on each decimation; only ticks divisible by it are retained.
        self._keep_stride = 1
        # Grows with the stride, so the reported interval always describes the spacing of
        # the samples actually returned.
        self._effective_interval = self._interval

    # -- lifecycle -----------------------------------------------------------------

    async def __aenter__(self) -> TelemetrySampler:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def start(self) -> None:
        if self._task is not None:
            return
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    # -- results -------------------------------------------------------------------

    @property
    def engine_samples(self) -> list[EngineSampleWire]:
        return [sample for _, sample in self._engine]

    @property
    def gpu_samples(self) -> list[GpuSampleWire]:
        return [sample for _, sample in self._gpu]

    @property
    def decimated(self) -> bool:
        return self._decimated

    @property
    def effective_interval_seconds(self) -> float:
        return self._effective_interval

    # -- internals -----------------------------------------------------------------

    async def _run(self) -> None:
        # One client for the whole run: reconnecting every second would add TCP setup to
        # the engine's load for no benefit.
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_SECONDS) as client:
            next_tick = time.monotonic()
            while True:
                await self._tick(client)

                # Drift-free scheduling: advance by whole intervals from the origin so a
                # slow tick does not push every later sample later. If we have fallen
                # behind by more than one interval, skip ahead rather than firing a
                # burst of catch-up scrapes at an already-struggling engine.
                now = time.monotonic()
                next_tick += self._interval
                if next_tick <= now:
                    missed = int((now - next_tick) // self._interval) + 1
                    next_tick += missed * self._interval
                await asyncio.sleep(max(0.0, next_tick - now))

    async def _tick(self, client: httpx.AsyncClient) -> None:
        offset = time.monotonic() - (self._started_at or time.monotonic())
        tick = self._tick_index
        self._tick_index += 1

        engine = await self._sample_engine(client, offset)
        if engine is not None:
            self._engine.append((tick, engine))

        for sample in self._sample_gpus(offset):
            self._gpu.append((tick, sample))

        if len(self._engine) > MAX_ENGINE_SAMPLES:
            self._decimate()

    async def _sample_engine(
        self, client: httpx.AsyncClient, offset: float
    ) -> EngineSampleWire | None:
        try:
            response = await client.get(f"{self._base_url}/metrics")
            response.raise_for_status()
        except (TimeoutError, httpx.HTTPError) as exc:
            # Debug, not warning: an engine that is briefly too busy to answer /metrics
            # is normal under saturation, and a warning per second would bury the log.
            log.debug("metrics scrape failed at +%.1fs: %s", offset, exc)
            return None

        values = parse_metrics(response.text)
        if not values:
            return None

        return EngineSampleWire(
            offset_seconds=offset,
            num_requests_running=_as_int(values.get("num_requests_running")),
            num_requests_waiting=_as_int(values.get("num_requests_waiting")),
            kv_cache_usage_fraction=values.get("kv_cache_usage_fraction"),
            num_preemptions_total=_as_int(values.get("num_preemptions_total")),
            prefix_cache_queries_total=_as_int(values.get("prefix_cache_queries_total")),
            prefix_cache_hits_total=_as_int(values.get("prefix_cache_hits_total")),
        )

    def _sample_gpus(self, offset: float) -> list[GpuSampleWire]:
        if not self._device_indices:
            return []
        try:
            import pynvml
        except ImportError:
            return []

        samples: list[GpuSampleWire] = []
        try:
            pynvml.nvmlInit()
        except Exception as exc:
            log.debug("NVML unavailable for sampling: %s", exc)
            return []

        try:
            for index in self._device_indices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                except Exception as exc:
                    log.debug("no NVML handle for device %d: %s", index, exc)
                    continue
                samples.append(
                    GpuSampleWire(
                        offset_seconds=offset,
                        gpu_index=index,
                        # Each reading is independent: a driver that refuses one counter
                        # should not cost us the other five.
                        sm_utilization_pct=_nvml(
                            lambda h=handle: float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                        ),
                        memory_used_bytes=_nvml(
                            lambda h=handle: int(pynvml.nvmlDeviceGetMemoryInfo(h).used)
                        ),
                        power_watts=_nvml(
                            lambda h=handle: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        ),
                        temperature_c=_nvml(
                            lambda h=handle: float(
                                pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                            )
                        ),
                        sm_clock_mhz=_nvml(
                            lambda h=handle: int(
                                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                            )
                        ),
                        memory_clock_mhz=_nvml(
                            lambda h=handle: int(
                                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                            )
                        ),
                    )
                )
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()

        return samples

    def _decimate(self) -> None:
        """Halve the resolution instead of dropping the tail.

        A cap that simply stops recording produces a timeline that ends partway through
        the run, which reads as "the engine went quiet" rather than "we stopped looking".
        Keeping alternate *ticks* preserves the shape and the full span, and doubling the
        reported interval keeps any rate derived from adjacent samples correct.

        Whole ticks, not alternate list entries: each tick contributes one sample per
        device, so thinning positionally would delete entire devices rather than halve
        the resolution.
        """
        # Stride doubles each time. Filtering on `tick % 2` would work once and then
        # never again — after the first pass every surviving tick is already even, so
        # the buffer would stop shrinking while the cap kept firing every tick.
        self._keep_stride *= 2
        self._engine = deque((t, s) for t, s in self._engine if t % self._keep_stride == 0)
        self._gpu = deque((t, s) for t, s in self._gpu if t % self._keep_stride == 0)
        self._effective_interval = self._interval * self._keep_stride
        if not self._decimated:
            log.info(
                "telemetry exceeded %d samples; halving resolution to %.1fs",
                MAX_ENGINE_SAMPLES,
                self._effective_interval,
            )
        self._decimated = True


def _nvml(read: Any) -> Any:
    """Run one NVML read, returning None if the driver will not answer it.

    Per reading rather than per device: not every counter is supported on every card
    (consumer boards commonly refuse power or clock reads), and one unsupported field
    must not blank out the five that work.
    """
    try:
        return read()
    except Exception:
        return None
