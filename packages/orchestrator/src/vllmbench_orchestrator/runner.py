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

import asyncio
import contextlib
import datetime as dt
import logging
import uuid
from collections.abc import Iterator
from typing import NamedTuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import FailureKind, RunStatus, SweepStatus
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
from vllmbench_protocol import (
    AgentClient,
    AgentError,
    AgentUnreachable,
    EnvironmentStatus,
    classify_agent_error,
)
from vllmbench_protocol import FailureKind as WireFailureKind
from vllmbench_protocol.bench_result import (
    BenchResultError,
    EmptyBenchResult,
    flatten_bench_result,
)
from vllmbench_protocol.wire import (
    BenchRequest,
    BenchResponse,
    HostInfo,
    ServerState,
    StartServerRequest,
)

log = logging.getLogger("vllmbench.orchestrator.runner")

# The port the agent is told to serve on. Fixed for now: one engine per host, so there is
# nothing to deconflict with.
SERVER_PORT = 8000

# How long to keep trying to reach an agent that is not answering, before giving up on
# the run. Backoff between attempts, in seconds; the sum is the window.
#
# This is reconnection, not retry. Nothing is re-measured here — the probe runs before
# the engine starts, so there is no result a second attempt could paper over. It exists
# because an agent restart is seconds (a `uv`-installed systemd unit coming back), and
# losing a point of a six-hour sweep to a restart that finished before anyone noticed is
# not a defensible failure mode. The window closes once the benchmark is running: a lost
# connection then means the measurement is lost, and the honest answer is a failed run.
RECONNECT_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)


class RunFailed(RuntimeError):
    """The run could not be completed. The message is recorded on the run.

    Carries a :class:`FailureKind` so the failure is countable as well as readable. The
    kind is decided where the failure happens, because the phase a run was in is
    evidence no later reader of the message can recover.
    """

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class RunCancelled(RuntimeError):
    """The run was stopped on request.

    Separate from RunFailed because they mean opposite things about the sweep. A failure
    is a result to investigate; a cancellation is the system doing exactly what it was
    told. Recording one as the other would fill a cancelled sweep with red rows and make
    a real failure inside it impossible to spot.
    """


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


async def _ensure_server(
    client: AgentClient,
    config: ServerConfig,
    *,
    run_id: uuid.UUID,
    readiness_timeout_seconds: float,
):
    """Get a ready engine for this config, reusing the running one when it already is.

    A sweep varies workloads and replicates underneath a single server config far more
    often than it changes the config, and a restart costs minutes for a large model —
    72s for a 27B on the reference host, and that is a fast case. Reusing the engine
    across workload-only changes is most of the difference between a sweep that takes an
    afternoon and one that takes a day.

    Reuse is keyed on the config hash the *agent* reports, not on what this process
    believes it started. The orchestrator can restart mid-sweep, and after that its only
    honest source for what is loaded is the host itself.

    Caches are still reset before every benchmark (``reset_caches_first``), so a reused
    engine does not carry a warm prefix cache into the next point — which would inflate
    it and make the order of a matrix silently affect its results.
    """
    current = await client.server_status()
    if current.state is ServerState.READY and current.config_hash == config.config_hash:
        log.info("run %s: reusing the running engine (config %s)", run_id, config.config_hash[:12])
        return current

    if current.state in (ServerState.READY, ServerState.STARTING):
        log.info(
            "run %s: config changed (%s -> %s); restarting the engine",
            run_id,
            (current.config_hash or "none")[:12],
            config.config_hash[:12],
        )
        await client.stop_server()

    log.info("run %s: starting server (config %s)", run_id, config.config_hash[:12])
    return await client_start_server(
        client, config, readiness_timeout_seconds=readiness_timeout_seconds
    )


async def _next_run_wants_same_config(session: AsyncSession, run: Run) -> bool:
    """Whether the next queued run on this host uses the same server config.

    Decides whether to leave the engine loaded. Looked up rather than assumed, because
    "the next point in the sweep" is only knowable from the plan, and the plan is in the
    database precisely so no process has to hold it.
    """
    next_hash = await session.scalar(
        select(Run.config_hash)
        .where(Run.status == RunStatus.QUEUED, Run.gpu_host_id == run.gpu_host_id)
        .order_by(Run.queued_at, Run.sweep_seq.nulls_first())
        .limit(1)
    )
    return next_hash == run.config_hash


class Budgets(NamedTuple):
    """How long this run is allowed to take, per phase.

    Resolved once, at the top of the run, from the host's defaults and the sweep's
    overrides. Passed to the agent rather than enforced here: the agent owns the
    processes, so it is the only side that can kill one, and a control-plane-side
    deadline would leave a benchmark running on the host with nobody watching it.
    """

    model_load_seconds: float
    benchmark_seconds: float


async def _budgets(session: AsyncSession, run: Run, host: GpuHost) -> Budgets:
    """Host defaults, overridden by the sweep where it said something.

    Null on the sweep means "no opinion", so raising a host's default also raises every
    sweep that never had one. A sweep that copied the host's numbers at authoring time
    would instead be frozen at whatever they were that day.
    """
    load = float(host.model_load_timeout_seconds)
    bench = float(host.benchmark_timeout_seconds)

    if run.sweep_id is not None:
        sweep = await session.get(Sweep, run.sweep_id)
        if sweep is not None:
            if sweep.model_load_timeout_seconds is not None:
                load = float(sweep.model_load_timeout_seconds)
            if sweep.benchmark_timeout_seconds is not None:
                bench = float(sweep.benchmark_timeout_seconds)

    return Budgets(model_load_seconds=load, benchmark_seconds=bench)


async def _reach_host(client: AgentClient, *, run_id: uuid.UUID) -> HostInfo:
    """Get the host's facts, tolerating an agent that is briefly away.

    Only :class:`AgentUnreachable` is retried, and only here. A refused token or a
    protocol mismatch will not fix itself, so retrying them just delays the report; and
    every call after this one either starts an engine or runs a benchmark, neither of
    which is safe to repeat.

    This is also the reachability probe for the run — it is the first authenticated call
    made, so an agent that is down fails here, before an engine is started, rather than
    somewhere less recoverable.
    """
    last: AgentUnreachable | None = None
    for attempt, pause in enumerate((*RECONNECT_BACKOFF_SECONDS, None), start=1):
        try:
            return await client.host_info()
        except AgentUnreachable as exc:
            last = exc
            if pause is None:
                break
            log.warning(
                "run %s: agent unreachable (attempt %d/%d), retrying in %.0fs: %s",
                run_id,
                attempt,
                len(RECONNECT_BACKOFF_SECONDS) + 1,
                pause,
                exc.detail,
            )
            await asyncio.sleep(pause)

    assert last is not None
    window = sum(RECONNECT_BACKOFF_SECONDS)
    raise RunFailed(
        FailureKind.AGENT_UNREACHABLE,
        f"{last} (still unreachable after {len(RECONNECT_BACKOFF_SECONDS) + 1} attempts "
        f"over {window:.0f}s)",
    )


async def execute_run(
    session: AsyncSession,
    run: Run,
    token: str,
    cancel: asyncio.Event | None = None,
) -> None:
    """Drive one run to a terminal state.

    Never raises: a run that fails must be recorded as failed with a reason, because an
    exception escaping here would leave it stuck in a non-terminal state forever and the
    UI would show it as perpetually running.
    """
    host = await session.get(GpuHost, run.gpu_host_id)
    config = await session.get(ServerConfig, run.server_config_id)
    workload = await session.get(Workload, run.workload_id)
    if host is None or config is None or workload is None:
        await _fail(
            session,
            run,
            FailureKind.INTERNAL,
            "run references a host, config or workload that no longer exists",
        )
        return

    client = AgentClient(host.agent_url, token, timeout=None)
    server_started = False

    try:
        if cancel is not None and cancel.is_set():
            raise RunCancelled("cancelled before the engine was started")

        budgets = await _budgets(session, run, host)
        info = await _reach_host(client, run_id=run.id)
        _record_provenance(run, info)

        server_started = True
        # Every failure from here to readiness is the engine refusing to come up, so the
        # phase default is engine_load_failed and vLLM's own output narrows it: out of
        # memory, or a configuration it would not accept. Distinguishing those matters
        # because they have opposite responses — one is a sweep point that asked for
        # more than the card has, the other is a config that is simply wrong.
        with _failing_as(FailureKind.ENGINE_LOAD_FAILED):
            status = await _ensure_server(
                client,
                config,
                run_id=run.id,
                readiness_timeout_seconds=budgets.model_load_seconds,
            )

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

        # What the engine resolved for speculation, from its own /server_info. Recorded
        # here as well as after the benchmark so that a run which fails mid-flight still
        # says what it was measuring. `None` stays NULL: an engine that would not answer
        # must not be recorded as one that denied speculating.
        if status.speculative_method is not None:
            run.speculative_method = status.speculative_method
            run.speculative_tokens = status.speculative_tokens

        # Same reasoning as speculation above: recorded at engine start as well as after
        # the benchmark, so a run that dies mid-flight still says what it was launched
        # under. An empty mapping is a real answer -- none of these were set -- so the
        # guard is against the agent being too old to have sent the field at all, which
        # protocol negotiation should already have refused.
        if status.engine_env:
            run.engine_env = dict(status.engine_env)

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
        with _failing_as(FailureKind.BENCHMARK_FAILED):
            response = await _bench_or_cancel(
                client,
                workload,
                model=model,
                served_model_name=served_model_name,
                cancel=cancel,
                run_id=run.id,
                timeout_seconds=budgets.benchmark_seconds,
            )

        if response.device_indices:
            run.device_indices = response.device_indices
            run.gpu_count = max(1, len(response.device_indices))
        if response.tensor_parallel_size:
            run.tensor_parallel_size = response.tensor_parallel_size
        if response.peer_access is not None:
            run.peer_access = response.peer_access.value
        if response.speculative_method is not None:
            run.speculative_method = response.speculative_method
            run.speculative_tokens = response.speculative_tokens
        if response.engine_env:
            run.engine_env = dict(response.engine_env)

        # Invariant 6 has required this since the first schema and nothing wrote it until
        # protocol 7. Only the agent can compute it: `--dataset-path` names a file on the
        # GPU host, which the control plane cannot see (invariant 1).
        run.dataset_identity = response.dataset_identity

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

    except RunCancelled as exc:
        await _cancel_run(session, run, str(exc))
    except RunFailed as exc:
        await _fail(session, run, exc.kind, str(exc))
    except EmptyBenchResult as exc:
        # The benchmark ran and completed nothing. Not a schema problem — every field the
        # contract requires is present — so it is filed as a failed benchmark, which is
        # what it is, and the work to do is on the host rather than here.
        await _fail(session, run, FailureKind.BENCHMARK_FAILED, str(exc))
    except BenchResultError as exc:
        # The benchmark ran but its output did not match the contract. Distinguished
        # from a failed benchmark because the fix is different: this means vLLM changed
        # its schema, not that the configuration was bad — so the work to do is in this
        # repository, not on the host.
        await _fail(
            session,
            run,
            FailureKind.RESULT_SCHEMA_MISMATCH,
            f"benchmark result did not match the expected schema: {exc}",
        )
    except AgentError as exc:
        # Reached only outside the phase blocks above — teardown, or a call added later
        # without one. `internal` rather than a plausible guess, so an unclassified path
        # shows up as the defect it is instead of quietly inflating a real kind.
        await _fail(session, run, _kind_of(exc, FailureKind.INTERNAL), str(exc))
    except Exception as exc:
        log.exception("run %s: unexpected failure", run.id)
        await _fail(session, run, FailureKind.INTERNAL, f"unexpected failure: {exc}")
    finally:
        if server_started:
            # Left running only when the next queued run wants the same config, and only
            # when this run ended cleanly. Anything else — a failure, a cancellation, an
            # empty queue — tears the engine down, because the reason to keep VRAM held
            # is that something is about to use it, and in those cases nothing is.
            keep = False
            if run.status is RunStatus.SUCCEEDED:
                try:
                    keep = await _next_run_wants_same_config(session, run)
                except Exception:
                    log.exception("run %s: could not look ahead; tearing down", run.id)

            if keep:
                log.info("run %s: leaving the engine loaded for the next point", run.id)
            else:
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


async def client_start_server(
    client: AgentClient, config: ServerConfig, *, readiness_timeout_seconds: float
):
    return await client.start_server(
        StartServerRequest(
            config_yaml=config.yaml,
            config_hash=config.config_hash,
            port=SERVER_PORT,
            readiness_timeout_seconds=readiness_timeout_seconds,
        )
    )


async def client_bench(
    client: AgentClient,
    workload: Workload,
    *,
    model: str,
    served_model_name: str | None,
    timeout_seconds: float,
):
    return await client.bench(
        BenchRequest(
            timeout_seconds=timeout_seconds,
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
            # Verbatim passthrough for flags this build predates. Coerced because rows
            # written before this column held a list carry `{}`, which means the same
            # thing and must not raise on the way to the agent.
            extra_args=list(workload.extra_args) if isinstance(workload.extra_args, list) else [],
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
            FailureKind.ENGINE_CONFIG_REJECTED,
            "could not find a `model:` entry in the server config; `vllm bench serve` "
            "needs one to load the tokenizer",
        )
    return weights, found.get("served_model_name")


def _record_provenance(run: Run, info: HostInfo) -> None:
    run.vllm_version = info.vllm_version
    run.agent_version = info.agent_version
    run.protocol_version = info.protocol_version
    run.driver_version = info.driver_version
    run.cuda_version = info.cuda_version
    # Whether the machine this was measured on satisfied its own dependency constraints.
    # Copied onto the run like every other fact here rather than joined from the host,
    # because the host record moves and this run's claim about it must not.
    run.environment_status = (
        info.environment.status.value if info.environment else EnvironmentStatus.NOT_REPORTED.value
    )
    run.gpu_model = info.gpus[0].name if info.gpus else None

    # Re-checked at execution time rather than trusted from creation: a host could have
    # been swapped between queueing and running, and the run must describe what actually
    # produced it.
    run.is_synthetic = info.synthetic_source is not None
    run.synthetic_source = info.synthetic_source


async def _bench_or_cancel(
    client: AgentClient,
    workload: Workload,
    *,
    model: str,
    served_model_name: str | None,
    cancel: asyncio.Event | None,
    run_id: uuid.UUID,
    timeout_seconds: float,
):
    """Run the benchmark, or stop it if cancellation arrives while it is running.

    The benchmark is the long part — minutes to hours — so waiting for it to finish
    before honouring a cancellation would make cancelling a sweep almost meaningless.

    Order matters on the way out: the *client* is stopped first, through the agent, and
    the engine afterwards by the caller's teardown. Killing the engine first would leave
    `vllm bench serve` firing requests at a dead socket until its own timeout, which is
    both slow and a good way to end up with an orphan.
    """
    bench = asyncio.ensure_future(
        client_bench(
            client,
            workload,
            model=model,
            served_model_name=served_model_name,
            timeout_seconds=timeout_seconds,
        )
    )
    if cancel is None:
        return await bench

    waiter = asyncio.ensure_future(cancel.wait())
    try:
        done, _ = await asyncio.wait({bench, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if bench in done:
            return bench.result()

        log.info("run %s: cancellation requested; stopping the benchmark", run_id)
        with contextlib.suppress(AgentError):
            await client.cancel_bench()

        # Wait for the agent to report back rather than abandoning the request. The
        # benchmark call is what holds the connection, and dropping it would leave the
        # host finishing work whose result nobody is listening for.
        with contextlib.suppress(AgentError, asyncio.CancelledError):
            await bench
        raise RunCancelled("cancelled while the benchmark was running")
    finally:
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter


@contextlib.contextmanager
def _failing_as(default: FailureKind) -> Iterator[None]:
    """Turn agent errors raised inside this block into a classified failure.

    The block is the evidence. Which phase a run was in is knowable here and nowhere
    later, so the kind is fixed at the point the call is made rather than reconstructed
    afterwards from whatever text came back.

    ``RunFailed`` passes through untouched: something closer to the failure already
    named it, and a wrapper has no business overruling that.
    """
    try:
        yield
    except RunFailed:
        raise
    except AgentError as exc:
        raise RunFailed(_kind_of(exc, default), str(exc)) from exc


def _kind_of(exc: AgentError, default: FailureKind) -> FailureKind:
    """Bridge the protocol's classification onto the column's vocabulary.

    Two enums with the same members, on purpose: the protocol package owns the
    classification and must stay installable on a GPU host, while the database package
    owns what can be stored. A test asserts they agree, so this lookup cannot fail
    silently — but it falls back to the phase default rather than raising, because a
    failure being mislabelled is not a reason to lose it.
    """
    wire = classify_agent_error(exc, default=WireFailureKind(default.value))
    try:
        return FailureKind(wire.value)
    except ValueError:  # pragma: no cover - the pinning test makes this unreachable
        log.error("no stored FailureKind for %r; recording %s", wire, default)
        return default


async def _cancel_run(session: AsyncSession, run: Run, message: str) -> None:
    log.info("run %s cancelled: %s", run.id, message)
    run.status = RunStatus.CANCELLED
    run.error = message[:4000]
    run.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()


async def _fail(session: AsyncSession, run: Run, kind: FailureKind, message: str) -> None:
    """Record a failure with both its kind and its full text.

    Both, always. The kind is what makes eleven failed points in a sweep answerable as
    one question; the text is the only thing that says which card ran out of how much
    memory. Neither substitutes for the other, and a database constraint requires the
    kind so that no future path can fail without one.
    """
    log.error("run %s failed (%s): %s", run.id, kind, message)
    run.status = RunStatus.FAILED
    run.failure_kind = kind.value
    run.error = message[:4000]
    run.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()
