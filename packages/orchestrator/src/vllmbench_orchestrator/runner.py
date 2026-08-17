"""Executing a queued run against a GPU host.

The sequence is: claim the run, start a server from its config, benchmark it, persist,
tear down. Every step can fail, and the ordering is chosen so that failure never leaves
the host worse than it found it — the server is stopped in a ``finally``, because a run
that fails after starting an engine must not leave VRAM held for the next one.

Claiming uses ``SELECT ... FOR UPDATE SKIP LOCKED``. Only one orchestrator runs today,
but the sweep engine in 0.4.0 will want to claim work without two workers racing for the
same run, and retrofitting that later means retrofitting it into code that has already
recorded results.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import RunStatus, SweepStatus
from vllmbench_db.models import (
    EngineSample,
    GpuHost,
    GpuSample,
    Run,
    RunSummary,
    ServerConfig,
    Sweep,
    Workload,
)
from vllmbench_protocol import AgentClient, AgentError
from vllmbench_protocol.bench_result import BenchResultError, flatten_bench_result
from vllmbench_protocol.wire import BenchRequest, BenchResponse, HostInfo, StartServerRequest

log = logging.getLogger("vllmbench.orchestrator.runner")

# The port the agent is told to serve on. Fixed for now: one engine per host, so there is
# nothing to deconflict with.
SERVER_PORT = 8000


class RunFailed(RuntimeError):
    """The run could not be completed. The message is recorded on the run."""


async def claim_next_run(session: AsyncSession) -> Run | None:
    """Take ownership of one queued run, or return None.

    SKIP LOCKED rather than a plain lock: a second worker should move on to other work
    rather than block behind the first.

    Ordered by ``queued_at`` then ``sweep_seq``. A sweep materializes all of its runs in
    one transaction, so Postgres stamps them with an identical ``now()`` and FIFO alone
    leaves the order to chance — which matters because the plan deliberately keeps runs
    sharing a server config adjacent. Executing a sweep in arbitrary order would still
    produce correct measurements, but would restart the engine on nearly every run, and a
    restart is minutes for a large model.
    """
    run = await session.scalar(
        select(Run)
        .where(Run.status == RunStatus.QUEUED)
        .order_by(Run.queued_at, Run.sweep_seq.nulls_first())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if run is None:
        return None

    run.status = RunStatus.STARTING
    run.started_at = dt.datetime.now(dt.UTC)
    if run.sweep_id is not None:
        await _mark_sweep_running(session, run.sweep_id)
    await session.commit()
    log.info("claimed run %s", run.id)
    return run


async def _mark_sweep_running(session: AsyncSession, sweep_id: uuid.UUID) -> None:
    """Move a sweep from queued to running when its first run is claimed."""
    sweep = await session.get(Sweep, sweep_id)
    if sweep is None or sweep.status is not SweepStatus.QUEUED:
        return
    sweep.status = SweepStatus.RUNNING
    sweep.started_at = dt.datetime.now(dt.UTC)


async def settle_sweep(session: AsyncSession, sweep_id: uuid.UUID) -> None:
    """Close a sweep out once none of its runs can still change.

    Derived from the runs rather than counted as they finish, so a crash between a run
    finishing and the sweep being updated cannot leave the sweep permanently mid-flight.
    The runs are the record; the sweep's status is a summary of them.

    A sweep with any failed run is failed. Partial success is still a failure to deliver
    the matrix that was asked for, and a sweep that reports success while missing points
    would have to be checked run-by-run to be trusted — which is what the status exists to
    avoid.
    """
    sweep = await session.get(Sweep, sweep_id)
    if sweep is None or sweep.status in (
        SweepStatus.SUCCEEDED,
        SweepStatus.FAILED,
        SweepStatus.CANCELLED,
    ):
        return

    counts: dict[RunStatus, int] = {}
    for run_status, count in (
        await session.execute(
            select(Run.status, func.count()).where(Run.sweep_id == sweep_id).group_by(Run.status)
        )
    ).all():
        counts[run_status] = count

    outstanding = sum(
        counts.get(state, 0)
        for state in (RunStatus.QUEUED, RunStatus.STARTING, RunStatus.BENCHMARKING)
    )
    if outstanding:
        return

    failed = counts.get(RunStatus.FAILED, 0)
    total = sum(counts.values())
    if failed:
        sweep.status = SweepStatus.FAILED
        sweep.error = f"{failed} of {total} runs failed"
    elif counts.get(RunStatus.CANCELLED, 0) and not counts.get(RunStatus.SUCCEEDED, 0):
        sweep.status = SweepStatus.CANCELLED
    else:
        sweep.status = SweepStatus.SUCCEEDED

    sweep.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()
    log.info("sweep %s finished: %s", sweep_id, sweep.status)


async def execute_run(session: AsyncSession, run: Run, token: str) -> None:
    """Drive one run to a terminal state.

    Never raises: a run that fails must be recorded as failed with a reason, because an
    exception escaping here would leave it stuck in a non-terminal state forever and the
    UI would show it as perpetually running.
    """
    host = await session.get(GpuHost, run.gpu_host_id)
    config = await session.get(ServerConfig, run.server_config_id)
    workload = await session.get(Workload, run.workload_id)
    if host is None or config is None or workload is None:
        await _fail(session, run, "run references a host, config or workload that no longer exists")
        return

    client = AgentClient(host.agent_url, token, timeout=None)
    server_started = False

    try:
        info = await client.host_info()
        _record_provenance(run, info)

        log.info("run %s: starting server (config %s)", run.id, run.config_hash[:12])
        server_started = True
        status = await client_start_server(client, config)

        # The engine's own version wins over the agent's environment probe: the agent
        # may live in a separate venv, and it is the engine that produced the numbers.
        if status.vllm_version:
            run.vllm_version = status.vllm_version

        # Declared topology is what the config asked for...
        if status.tensor_parallel_size:
            run.tensor_parallel_size = status.tensor_parallel_size
        if status.pipeline_parallel_size:
            run.pipeline_parallel_size = status.pipeline_parallel_size

        # ...and gpu_count is what NVML observed. These are deliberately different
        # sources. Per-GPU normalization divides by the observed count, so if a config
        # requested more devices than the host could give, the figures still describe
        # the hardware that actually ran.
        if status.device_indices:
            run.device_indices = status.device_indices
            run.gpu_count = max(1, len(status.device_indices))

        run.status = RunStatus.BENCHMARKING
        await session.commit()

        # Weights identifier from the config, alias from the engine.
        #
        # The config is the only place the weights id is written, and it is what the
        # tokenizer must be loaded from. The alias is whatever the engine is actually
        # answering to, which is ground truth and can differ from the config — so the
        # engine's own /v1/models wins, with the config's `served-model-name` as the
        # fallback for an engine that did not report one.
        model, config_alias = _model_names_from_config(config)
        served_model_name = status.served_model_name or config_alias

        log.info(
            "run %s: benchmarking weights %s served as %s",
            run.id,
            model,
            served_model_name or model,
        )
        response = await client_bench(
            client, workload, model=model, served_model_name=served_model_name
        )

        if response.device_indices:
            run.device_indices = response.device_indices
            run.gpu_count = max(1, len(response.device_indices))
        if response.tensor_parallel_size:
            run.tensor_parallel_size = response.tensor_parallel_size

        run.raw_result = response.raw_result
        run.log_excerpt = "\n".join(response.stdout_tail[-40:])

        # Flattening happens here rather than in the agent so that the raw payload
        # crosses the host boundary intact. If this mapping is ever wrong, the stored
        # raw_result is what allows recomputation instead of re-measuring.
        flat = flatten_bench_result(response.raw_result, gpu_count=run.gpu_count)
        summary = RunSummary(run_id=run.id, **flat)
        session.add(summary)

        _record_telemetry(session, run, response)

        run.status = RunStatus.SUCCEEDED
        run.finished_at = dt.datetime.now(dt.UTC)
        await session.commit()
        log.info("run %s: succeeded", run.id)

    except BenchResultError as exc:
        # The benchmark ran but its output did not match the contract. Distinguished
        # from a failed benchmark because the fix is different: this means vLLM changed
        # its schema, not that the configuration was bad.
        await _fail(session, run, f"benchmark result did not match the expected schema: {exc}")
    except AgentError as exc:
        await _fail(session, run, str(exc))
    except Exception as exc:
        log.exception("run %s: unexpected failure", run.id)
        await _fail(session, run, f"unexpected failure: {exc}")
    finally:
        if server_started:
            # Unconditional. A run that failed after starting an engine must not leave
            # VRAM held for whatever runs next.
            try:
                await client.stop_server()
            except AgentError as exc:
                log.error("run %s: could not stop the server: %s", run.id, exc)
        await client.aclose()

        # After teardown, and outside the success path, so a sweep is closed out whether
        # its last run succeeded, failed, or died in a way that only `_fail` recorded.
        if run.sweep_id is not None:
            try:
                await settle_sweep(session, run.sweep_id)
            except Exception:
                # The runs are the record; a sweep whose summary status lagged is a
                # cosmetic problem, and it must not turn into a failed run.
                log.exception("could not settle sweep %s", run.sweep_id)


async def client_start_server(client: AgentClient, config: ServerConfig):
    return await client.start_server(
        StartServerRequest(
            config_yaml=config.yaml,
            config_hash=config.config_hash,
            port=SERVER_PORT,
        )
    )


async def client_bench(
    client: AgentClient, workload: Workload, *, model: str, served_model_name: str | None
):
    return await client.bench(
        BenchRequest(
            model=model,
            served_model_name=served_model_name,
            dataset_name=workload.dataset_name,
            dataset_path=workload.dataset_path,
            hf_name=workload.hf_name,
            num_prompts=workload.num_prompts,
            request_rate=workload.request_rate,
            max_concurrency=workload.max_concurrency,
            burstiness=workload.burstiness,
            random_input_len=workload.input_len,
            random_output_len=workload.output_len,
        )
    )


def _record_telemetry(session: AsyncSession, run: Run, response: BenchResponse) -> None:
    """Turn the agent's offset-based samples into absolute rows.

    The agent reports offsets from the moment sampling began rather than wall-clock
    timestamps, and they are anchored here. Two hosts are involved and their clocks are
    not synchronised — the GPU host has no reason to run NTP against ours — so trusting
    its wall clock would let a few seconds of skew slide the whole timeline out of the
    window it is meant to describe. Offsets from a known anchor cannot skew.

    The anchor is the moment the benchmark ended minus its measured duration, because
    that is the pair of facts we hold most precisely: the response arrived just now, and
    it says how long it took.
    """
    if not response.engine_samples and not response.gpu_samples:
        return

    anchor = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=response.duration_seconds)

    for sample in response.engine_samples:
        session.add(
            EngineSample(
                run_id=run.id,
                sampled_at=anchor + dt.timedelta(seconds=sample.offset_seconds),
                num_requests_running=sample.num_requests_running,
                num_requests_waiting=sample.num_requests_waiting,
                kv_cache_usage_fraction=sample.kv_cache_usage_fraction,
                num_preemptions_total=sample.num_preemptions_total,
                prefix_cache_queries_total=sample.prefix_cache_queries_total,
                prefix_cache_hits_total=sample.prefix_cache_hits_total,
            )
        )

    for gpu in response.gpu_samples:
        session.add(
            GpuSample(
                run_id=run.id,
                gpu_index=gpu.gpu_index,
                sampled_at=anchor + dt.timedelta(seconds=gpu.offset_seconds),
                sm_utilization_pct=gpu.sm_utilization_pct,
                memory_used_bytes=gpu.memory_used_bytes,
                power_watts=gpu.power_watts,
                temperature_c=gpu.temperature_c,
                sm_clock_mhz=gpu.sm_clock_mhz,
                memory_clock_mhz=gpu.memory_clock_mhz,
            )
        )

    log.info(
        "run %s: %d engine samples, %d gpu samples at %.1fs resolution%s",
        run.id,
        len(response.engine_samples),
        len(response.gpu_samples),
        response.telemetry_interval_seconds or 0.0,
        " (decimated)" if response.telemetry_decimated else "",
    )


def _model_names_from_config(config: ServerConfig) -> tuple[str, str | None]:
    """Read ``(weights_id, served_alias)`` out of the config YAML.

    Two names, not one. ``vllm bench serve --model`` is the *weights* identifier and vLLM
    loads the tokenizer from it; ``--served-model-name`` is the alias the API answers to.

    An earlier version of this returned a single name and preferred the alias, to stop
    real configs 404-ing. That fixed the request and broke tokenization, which is the
    worse trade: a bad alias makes the benchmark die, but a *plausible* alias makes it
    tokenize against the wrong tokenizer and report input-token counts that are simply
    wrong, with every appearance of success.

    A narrow line read rather than a YAML parse, because invariant 5 keeps the config
    opaque. Hyphens and underscores are both accepted, since vLLM accepts both.
    """
    found: dict[str, str] = {}
    for line in config.yaml.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        name = key.strip().replace("-", "_")
        if name in ("model", "served_model_name"):
            cleaned = value.strip().strip("\"'")
            if cleaned:
                found[name] = cleaned

    weights = found.get("model")
    if not weights:
        raise RunFailed(
            "could not find a `model:` entry in the server config; `vllm bench serve` "
            "needs one to load the tokenizer"
        )
    return weights, found.get("served_model_name")


def _record_provenance(run: Run, info: HostInfo) -> None:
    run.vllm_version = info.vllm_version
    run.agent_version = info.agent_version
    run.protocol_version = info.protocol_version
    run.driver_version = info.driver_version
    run.cuda_version = info.cuda_version
    run.gpu_model = info.gpus[0].name if info.gpus else None

    # Re-checked at execution time rather than trusted from creation: a host could have
    # been swapped between queueing and running, and the run must describe what actually
    # produced it.
    run.is_synthetic = info.synthetic_source is not None
    run.synthetic_source = info.synthetic_source


async def _fail(session: AsyncSession, run: Run, message: str) -> None:
    log.error("run %s failed: %s", run.id, message)
    run.status = RunStatus.FAILED
    run.error = message[:4000]
    run.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()
