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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import RunStatus
from vllmbench_db.models import GpuHost, Run, RunSummary, ServerConfig, Workload
from vllmbench_protocol import AgentClient, AgentError
from vllmbench_protocol.bench_result import BenchResultError, flatten_bench_result
from vllmbench_protocol.wire import BenchRequest, HostInfo, StartServerRequest

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
    """
    run = await session.scalar(
        select(Run)
        .where(Run.status == RunStatus.QUEUED)
        .order_by(Run.queued_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if run is None:
        return None

    run.status = RunStatus.STARTING
    run.started_at = dt.datetime.now(dt.UTC)
    await session.commit()
    log.info("claimed run %s", run.id)
    return run


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

        # The engine's own served name wins over anything read from the config. A
        # config with `served-model-name` answers under that alias, so benchmarking the
        # `model:` line would 404 — a failure real configs produce and the mock never
        # would.
        model = status.served_model_name or _model_from_config(config)

        log.info("run %s: benchmarking model %s", run.id, model)
        response = await client_bench(client, workload, model=model)

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


async def client_start_server(client: AgentClient, config: ServerConfig):
    return await client.start_server(
        StartServerRequest(
            config_yaml=config.yaml,
            config_hash=config.config_hash,
            port=SERVER_PORT,
        )
    )


async def client_bench(client: AgentClient, workload: Workload, *, model: str):
    return await client.bench(
        BenchRequest(
            model=model,
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


def _model_from_config(config: ServerConfig) -> str:
    """Read the model name out of the config YAML.

    The one thing we must extract, because ``vllm bench serve`` needs ``--model`` and the
    config is the only place it is written. Deliberately a narrow read rather than a full
    parse: invariant 5 keeps the config opaque otherwise.
    """
    # served-model-name first: when present it is the alias the API answers to, and the
    # `model:` line is only the weights to load. Hyphens and underscores are both
    # accepted because vLLM accepts both spellings.
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

    for preferred in ("served_model_name", "model"):
        if preferred in found:
            return found[preferred]
    raise RunFailed(
        "could not find a `model:` entry in the server config; "
        "`vllm bench serve` needs one to know what to request"
    )


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
