"""MCP server, mounted on this service at ``/mcp``.

Implements ADR 0001, whose binding decisions are: mounted on the existing ``api`` service
rather than a separate one, Streamable HTTP, bearer token, read and write tools shipped
together, behind ``VLLMBENCH_MCP_ENABLED``.

The decisive reason for mounting rather than splitting is that **there must be no way for
the MCP surface to drift from the REST surface**. That is a property of code, not of
intent, so the tools here call the same router functions the HTTP routes call. They are
plain async functions taking a session; nothing is reimplemented, so a fix to a query is a
fix to both interfaces at once and a divergence is not expressible.

Two things this module owns that the REST surface does not need:

*Context economy.* An agent pays for every token it reads. List tools cap their page size
at a hard maximum regardless of what is asked for, and return summary fields rather than
whole objects — a config's YAML is fetched deliberately with ``get_config``, not carried
in every list result.

*A stated population.* Every tool that returns measurements takes ``source`` and defaults
to real. Invariant 7 is not something an agent should have to remember: as with the HTTP
analysis endpoints, there is no value meaning both.
"""

from __future__ import annotations

import functools
import inspect
import logging
import uuid
from collections.abc import Callable
from typing import Any

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from vllmbench_api.analysis import PARETO_X, PARETO_Y, RunSource
from vllmbench_api.routers import analysis as analysis_routes
from vllmbench_api.routers import hosts as host_routes
from vllmbench_api.routers import runs as run_routes
from vllmbench_api.routers import sweeps as sweep_routes
from vllmbench_api.schemas import (
    AnalysisOut,
    ConfigCreate,
    DurationEstimateOut,
    PointOut,
    RunOut,
    SweepCreate,
    SweepOut,
    WorkloadCreate,
)
from vllmbench_api.settings import ApiSettings
from vllmbench_db.enums import InitiatedBy
from vllmbench_db.models import McpWriteAudit
from vllmbench_protocol import PROTOCOL_VERSION, __version__

#: Hard ceilings, applied whatever a caller asks for. An agent that requests everything is
#: usually not choosing to — it is defaulting — and a list that quietly grows without bound
#: is how a long session runs out of context on the least interesting data it holds.
MAX_PAGE = 100
DEFAULT_PAGE = 25

#: Recorded as the client identity on everything this surface creates, so a run can name
#: not just the interface but the thing on the other end of it.
MCP_CLIENT_NAME = "mcp"


log = logging.getLogger(__name__)

#: Longest argument value kept verbatim in an audit row. A config's YAML is worth keeping
#: — it is what the caller actually asked to store — but there is no reason for this table
#: to be able to grow without bound on one call.
MAX_LOGGED_VALUE = 8192


def _loggable(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as they arrived, with only outsized strings trimmed.

    Raw before derived, as with every other record here: if the surface turns out to have
    mishandled an argument, this is the thing that says what it was handed. Nothing is
    redacted because nothing that reaches these tools is a secret — the bearer token is
    consumed by the transport and never appears in a tool argument.
    """
    trimmed: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > MAX_LOGGED_VALUE:
            trimmed[key] = value[:MAX_LOGGED_VALUE] + f"… [{len(value)} chars]"
        else:
            trimmed[key] = value
    return trimmed


def _is_refusal(exc: BaseException) -> bool:
    """Whether the surface declined a call it understood, rather than breaking on it.

    Everything this module raises deliberately is a ValueError, and everything the
    routers raise deliberately is an HTTPException with a 4xx status — a busy host, an
    oversized matrix, a hash nobody stored. Anything else is a bug, and the log should
    not describe a bug as a policy decision.
    """
    if isinstance(exc, ValueError):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and 400 <= status_code < 500


class StaticTokenVerifier(TokenVerifier):
    """A single shared bearer token.

    ADR 0001 names OAuth 2.1 as the standard for remote servers and this is not that. It
    is deliberate for a LAN-only surface behind a feature flag: the deployment story is
    one operator, one control plane, one token in an environment variable, and an OAuth
    server would be more moving parts than the thing it protects.

    An empty configured token disables the surface rather than accepting everything, which
    is the failure mode a misconfigured deployment would otherwise have.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not self._token or token != self._token:
            return None
        return AccessToken(token=token, client_id="vllmbench-mcp", scopes=["vllmbench"])


def _page(limit: int | None) -> int:
    """Clamp a requested page size.

    ``None`` means "unasked" and takes the default; an explicit 0 or negative means the
    caller asked for something impossible and gets one row rather than silently getting
    twenty-five, which would look like the request had been honoured.
    """
    return max(1, min(DEFAULT_PAGE if limit is None else limit, MAX_PAGE))


def _source(value: str) -> RunSource:
    """Parse the population, refusing anything that is not one of the two.

    An unrecognised value becomes an error rather than silently defaulting to real: an
    agent asking for something this service does not have should be told so, not handed
    numbers from a population it did not ask about.
    """
    try:
        return RunSource(value)
    except ValueError as exc:
        raise ValueError(
            f"source must be 'real' or 'synthetic', not {value!r}; there is no value meaning both"
        ) from exc


#: What the report tabulates. Deliberately per-GPU first: an agent skimming a table reads
#: the leftmost number, and the aggregate is the one that is not comparable across
#: tensor-parallel widths (invariant 8).
REPORT_METRICS = (
    ("total_token_throughput_per_gpu", "tok/s per GPU"),
    (PARETO_Y, "per-user tok/s"),
    ("ttft_ms_p99", "TTFT p99 ms"),
    ("total_token_throughput_tok_sec", "tok/s aggregate"),
)


def _cell(point: PointOut, key: str) -> str:
    """One metric, with its spread, or an em dash.

    The spread travels with the number rather than sitting in a footnote, because the
    number alone invites a comparison the spread would have forbidden.
    """
    spread = point.metrics.get(key)
    if spread is None:
        return "—"
    if spread.n < 2:
        return f"{spread.median:,.1f}"
    return f"{spread.median:,.1f} ±{(spread.max - spread.min) / 2:,.1f}"


def _duration(seconds: float) -> str:
    """Seconds as something a reader can act on.

    Rounded to the minute above an hour, because an estimate extrapolated from a handful
    of runs does not have second-level precision and printing it as though it does invites
    more trust than it has earned.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def _render_estimate(estimate: DurationEstimateOut) -> list[str]:
    """How much longer, with the arithmetic visible.

    The two components are shown separately because they behave differently: benchmark
    time scales with the runs left, engine-load time with the *config changes* left. A
    reader deciding whether to wait needs to know which of those is the bulk of it.
    """
    remaining = (
        f"{estimate.runs_remaining} run(s) left, {estimate.engine_loads_remaining} of them "
        "restarting the engine."
    )
    if estimate.seconds_remaining is None:
        lines = [f"**Remaining:** {remaining} No time estimate yet.", ""]
    else:
        lines = [
            f"**Remaining:** ~{_duration(estimate.seconds_remaining)} — {remaining}",
            "",
        ]
        if estimate.median_run_seconds is not None:
            detail = f"Based on {estimate.sample_size} completed run(s): "
            detail += f"{_duration(estimate.median_run_seconds)} per benchmark"
            if estimate.median_engine_load_seconds is not None:
                detail += f", {_duration(estimate.median_engine_load_seconds)} per engine load"
            lines += [detail + ".", ""]
    for caveat in estimate.caveats:
        lines += [f"> {caveat}", ""]
    return lines


def _render_sweep_report(sweep: SweepOut, analysis: AnalysisOut) -> str:
    """A sweep as prose and tables.

    Written for a reader that will act on it. That means the caveats are inline rather
    than appended: the synthetic banner is the first thing on the page, each comparability
    group is its own section with its own heading, and a group's warnings sit above its
    table rather than below it.
    """
    out: list[str] = [f"# {sweep.name}", ""]

    if sweep.is_synthetic:
        out += [
            "> **Synthetic. These are not measurements of any real hardware.** They come "
            "from the mock agent or the CPU backend and exist to exercise the framework. "
            "Do not compare them with real results or draw a tuning conclusion from them.",
            "",
        ]

    if sweep.description:
        out += [sweep.description, ""]

    progress = sweep.progress
    out += [
        f"**Status:** {sweep.status} — {progress.succeeded} succeeded, "
        f"{progress.failed} failed, {progress.cancelled} cancelled, of {progress.total}.",
        "",
        f"**Plan:** {sweep.replicates} replicate(s), {sweep.replicate_order} order, "
        f"{sweep.engine_starts} engine load(s). Started by `{sweep.initiated_by}`.",
        "",
    ]
    if sweep.error:
        out += [f"**Error:** {sweep.error}", ""]

    if sweep.estimated_remaining is not None:
        out += _render_estimate(sweep.estimated_remaining)

    if not analysis.groups:
        out += [
            "No charted results. " + _no_results_reason(sweep, analysis),
            "",
        ]
        return "\n".join(out)

    excluded = analysis.excluded
    if any((excluded.failed, excluded.cancelled, excluded.unfinished)):
        out += [
            f"{excluded.failed} failed, {excluded.cancelled} cancelled and "
            f"{excluded.unfinished} unfinished run(s) are counted here but not charted.",
            "",
        ]

    if len(analysis.groups) > 1:
        out += [
            f"Results fall into {len(analysis.groups)} sets that cannot be compared with "
            "each other — they differ in host, GPU, vLLM version or where the benchmark "
            "client ran. Each has its own table below. Do not read across them.",
            "",
        ]

    for group in analysis.groups:
        # The label already names host, GPU and vLLM version — it is what partitioned
        # these points in the first place. Only what it leaves out goes underneath it.
        out += [
            f"## {group.label}",
            "",
            f"Benchmark client: {group.bench_client_location}. {group.run_count} run(s).",
            "",
        ]
        for warning in group.warnings:
            out += [f"> {warning}", ""]

        headers = (
            ["", "Config", "Workload", "TP"] + [label for _, label in REPORT_METRICS] + ["reps"]
        )
        out += ["| " + " | ".join(headers) + " |"]
        out += ["| " + " | ".join("---" for _ in headers) + " |"]
        for point in group.points:
            # The frontier marker is the whole reason to read the table in order: these
            # are the configurations nothing else beats on both axes at once.
            row = [
                "★" if point.on_pareto_frontier else "",
                point.config_name,
                point.workload_name,
                str(point.tensor_parallel_size),
                *(_cell(point, key) for key, _ in REPORT_METRICS),
                str(point.replicates),
            ]
            out += ["| " + " | ".join(row) + " |"]
        out += [""]

        frontier = [p for p in group.points if p.on_pareto_frontier]
        if frontier:
            out += [
                f"★ marks the {len(frontier)} point(s) on the Pareto frontier of per-GPU "
                "throughput against per-user output rate — nothing measured here beats "
                "them on both at once. Everything else is dominated: some starred point "
                "is better on both axes, so there is no reason to run it.",
                "",
            ]
        spread_notes = {p.spread_note for p in group.points if p.replicates > 1}
        for note in sorted(spread_notes):
            out += [note, ""]
        out += [
            "± is half the observed range across replicates. A difference smaller than "
            "that is not a result.",
            "",
        ]

    return "\n".join(out)


def _no_results_reason(sweep: SweepOut, analysis: AnalysisOut) -> str:
    """Why an empty report is empty.

    An agent that gets a blank report and no reason will retry, or worse conclude the
    configuration performed at zero.
    """
    if sweep.progress.total == 0:
        return "This sweep has no runs."
    if sweep.progress.succeeded == 0:
        return "No run in it succeeded."
    if analysis.excluded.succeeded_without_summary:
        return (
            f"{analysis.excluded.succeeded_without_summary} run(s) finished but recorded no "
            "summary metrics, which means the benchmark output did not parse."
        )
    return "Its runs carry no charted metrics."


def build_mcp_server(sessions: async_sessionmaker[Any], settings: ApiSettings) -> MCPServer:
    """Assemble the MCP server. Called only when the feature flag is on."""
    server = MCPServer(
        name="vllmbench",
        title="vLLM Benchmarking",
        version=__version__,
        instructions=(
            "Measure and tune vLLM serving configurations. Runs are immutable once "
            "finished and no tool here mutates or deletes one. Results are partitioned by "
            "provenance: figures from different GPUs, hosts or vLLM versions are never "
            "returned as one comparable series. Throughput is reported per GPU as well as "
            "aggregate, because a tensor-parallel run out-throughputs a single-GPU one "
            "trivially while possibly being worse per device."
        ),
        token_verifier=StaticTokenVerifier(settings.mcp_token),
        # The SDK requires auth settings alongside a verifier. This service is a resource
        # server only — it verifies a token it was configured with and issues none — so
        # the issuer URL is nominal. ADR 0001 names OAuth 2.1 as the standard for remote
        # servers; that is post-1.0 work, and until then the surface is LAN-only behind a
        # feature flag, which is what makes a shared secret proportionate.
        auth=AuthSettings(
            issuer_url=AnyHttpUrl("http://localhost:8000"),
            resource_server_url=AnyHttpUrl("http://localhost:8000/mcp"),
            required_scopes=["vllmbench"],
        ),
    )

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        return server.tool()(fn)

    # -- Inventory ---------------------------------------------------------------

    @tool
    async def list_hosts() -> list[dict[str, Any]]:
        """GPU hosts this control plane knows about, and what each last reported."""
        async with sessions() as session:
            found = await host_routes.list_hosts(session)
            return [
                {
                    "id": str(host.id),
                    "name": host.name,
                    "gpu_count": host.gpu_count,
                    "vllm_version": host.vllm_version,
                    "gpu_models": sorted({d.name for d in host.devices}),
                    # Stated on every host, so an agent never has to infer it. Anything
                    # this host produces is quarantined from real measurements.
                    "synthetic_source": host.synthetic_source,
                    "last_seen_at": host.last_seen_at.isoformat() if host.last_seen_at else None,
                }
                for host in found
            ]

    @tool
    async def list_configs(limit: int | None = None) -> list[dict[str, Any]]:
        """Server configurations, newest first. YAML is omitted — use get_config."""
        async with sessions() as session:
            found = await run_routes.list_configs(session)
            return [
                {
                    "config_hash": config.config_hash,
                    "name": config.name,
                    "notes": config.notes,
                    "created_at": config.created_at.isoformat(),
                }
                for config in found[: _page(limit)]
            ]

    @tool
    async def get_config(config_hash: str) -> dict[str, Any]:
        """One configuration's exact YAML.

        The text is the configuration: it is what gets written to disk and passed to
        `vllm serve --config`, byte for byte. Nothing is normalized on the way in or out,
        so what this returns is what ran.
        """
        async with sessions() as session:
            for config in await run_routes.list_configs(session):
                if config.config_hash == config_hash:
                    return {
                        "config_hash": config.config_hash,
                        "name": config.name,
                        "yaml": config.yaml,
                        "notes": config.notes,
                    }
        raise ValueError(f"no config with hash {config_hash!r}")

    @tool
    async def list_workloads(limit: int | None = None) -> list[dict[str, Any]]:
        """Benchmark workloads — the traffic each run was measured under."""
        async with sessions() as session:
            found = await run_routes.list_workloads(session)
            return [
                {
                    "workload_hash": workload.workload_hash,
                    "name": workload.name,
                    "dataset_name": workload.dataset_name,
                    "num_prompts": workload.num_prompts,
                    # Null means unbounded, genuinely: no --max-concurrency, or
                    # --request-rate inf. Not zero, which would mean the opposite.
                    "max_concurrency": workload.max_concurrency,
                    "request_rate": workload.request_rate,
                    "input_len": workload.input_len,
                    "output_len": workload.output_len,
                }
                for workload in found[: _page(limit)]
            ]

    # -- Sweeps ------------------------------------------------------------------

    @tool
    async def list_sweeps(limit: int | None = None) -> list[dict[str, Any]]:
        """Sweeps, newest first, with run counts by status."""
        async with sessions() as session:
            found = await sweep_routes.list_sweeps(session)
            return [
                {
                    "id": str(sweep.id),
                    "name": sweep.name,
                    "status": sweep.status,
                    "replicates": sweep.replicates,
                    "replicate_order": sweep.replicate_order,
                    "is_synthetic": sweep.is_synthetic,
                    "progress": sweep.progress.model_dump(),
                    "engine_starts": sweep.engine_starts,
                }
                for sweep in found[: _page(limit)]
            ]

    @tool
    async def get_sweep(sweep_id: str) -> dict[str, Any]:
        """One sweep, including how far through it is."""
        async with sessions() as session:
            sweep = await sweep_routes.get_sweep(uuid.UUID(sweep_id), session)
            return sweep.model_dump(mode="json")

    # -- Runs --------------------------------------------------------------------

    @tool
    async def query_runs(limit: int | None = None) -> list[dict[str, Any]]:
        """Recent runs with their headline metrics and the provenance behind them."""
        async with sessions() as session:
            found = await run_routes.list_runs(session, limit=_page(limit))
            return [
                {
                    "id": str(run.id),
                    "status": run.status,
                    "config_hash": run.config_hash,
                    "workload_hash": run.workload_hash,
                    "vllm_version": run.vllm_version,
                    "gpu_model": run.gpu_model,
                    "gpu_count": run.gpu_count,
                    "tensor_parallel_size": run.tensor_parallel_size,
                    "is_synthetic": run.is_synthetic,
                    "synthetic_source": run.synthetic_source,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                    # The per-GPU figures, not the aggregate: the aggregate is not
                    # comparable across tensor-parallel sizes and an agent reading a list
                    # is exactly who would compare them.
                    "total_token_throughput_per_gpu": (
                        run.summary.total_token_throughput_per_gpu if run.summary else None
                    ),
                    "ttft_ms_p99": run.summary.ttft_ms_p99 if run.summary else None,
                    "error": run.error,
                }
                for run in found
            ]

    @tool
    async def get_run(run_id: str) -> dict[str, Any]:
        """One run in full, including every flattened metric."""
        async with sessions() as session:
            # Through the same response model the HTTP route serializes with, rather than
            # reading the ORM object directly: one definition of what a run looks like on
            # the wire, so the two interfaces cannot describe it differently.
            run = await run_routes.get_run(uuid.UUID(run_id), session)
            return RunOut.model_validate(run).model_dump(mode="json")

    # -- Analysis ----------------------------------------------------------------

    @tool
    async def get_pareto(
        source: str = "real",
        host_id: str | None = None,
        sweep_id: str | None = None,
        pareto_x: str = PARETO_X,
        pareto_y: str = PARETO_Y,
    ) -> dict[str, Any]:
        """Measurement points, partitioned into sets that may honestly be compared.

        Each group is one host, GPU model, vLLM version and benchmark-client location.
        Points from different groups are never comparable as a series — that is why they
        arrive separated rather than as one list to be sorted.

        Each point is a median across its replicates with the spread beside it, and states
        what that spread measures. A difference smaller than a point's own spread is not a
        result.
        """
        async with sessions() as session:
            result = await analysis_routes.analysis_points(
                session,
                source=_source(source),
                host_id=uuid.UUID(host_id) if host_id else None,
                sweep_id=[uuid.UUID(sweep_id)] if sweep_id else None,
                pareto_x=pareto_x,
                pareto_y=pareto_y,
            )
            return result.model_dump(mode="json")

    @tool
    async def compare_runs(left: str, right: str, source: str = "real") -> dict[str, Any]:
        """Two measurement points side by side, with a diff of their configurations.

        Takes ``point_id`` values from get_pareto. This is the one comparison that may
        cross a provenance boundary — comparing two vLLM versions is a supported use — and
        every difference between the two sides is listed in the result, flagged as
        invalidating or merely notable.
        """
        async with sessions() as session:
            result = await analysis_routes.analysis_compare(
                session, left=left, right=right, source=_source(source)
            )
            return result.model_dump(mode="json")

    @tool
    async def get_run_telemetry(run_id: str, max_samples: int = 200) -> dict[str, Any]:
        """Engine and per-device series for one run, thinned to at most `max_samples`.

        Thinned by stride rather than averaged, and per device rather than pooled. A
        host-level average of two GPUs destroys the imbalance that makes a tensor-parallel
        run diagnosable, and it is the summary most likely to look reassuring.

        `sample_count` is what was recorded and `stride` is what you got — a stride above
        1 means this is every nth reading, not a run that sampled slowly.
        """
        async with sessions() as session:
            # Thinning happens in the query, not here. Doing it after the fact would load
            # every sample of a multi-hour run in order to discard 98% of them, and the
            # obvious way to do it — striding the flat list — silently drops whole
            # devices, because gpu_sample is stored interleaved per device.
            result = await run_routes.get_run_telemetry(
                uuid.UUID(run_id), session, max_samples=max(1, max_samples)
            )
            return result.model_dump(mode="json")

    # -- Write -------------------------------------------------------------------
    #
    # Enabled by default, and separately switchable. Nothing here mutates or deletes a
    # run: runs are immutable once terminal, and the only write that touches one at all
    # is cancelling a sweep, which moves queued runs to cancelled — a transition, not an
    # edit to a measurement.
    #
    # Every one of these records `initiated_by="mcp"` and the client's name. Invariant 6
    # says a run must be able to state what produced it, and "which interface asked" only
    # became a real question when this surface was added.

    async def _record(
        tool_name: str,
        arguments: dict[str, Any],
        outcome: str,
        *,
        error: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Append one row to the audit log.

        In its own session, deliberately. The call being recorded may have died with its
        own transaction in an unusable state, and the refused and failed calls are the
        ones this table exists for — writing the record through the same session would
        lose exactly the rows worth keeping.

        A failure to audit never fails the call. An operator losing one log line is a
        smaller problem than a control plane that stops accepting work because its
        logging table is unhappy; it is logged loudly instead.
        """
        try:
            async with sessions() as session:
                session.add(
                    McpWriteAudit(
                        tool=tool_name,
                        client=MCP_CLIENT_NAME,
                        arguments=_loggable(arguments),
                        outcome=outcome,
                        error=error,
                        subject=subject,
                    )
                )
                await session.commit()
        except Exception:  # pragma: no cover - the database being down fails the tool first
            log.exception("could not write the MCP audit record for %s", tool_name)

    def write_tool(subject_key: str | None = None) -> Callable[..., Any]:
        """Register a write tool, audited, with the read-only switch enforced here.

        The switch lives in this decorator rather than in each tool body so that a tool
        added later cannot forget it — and so that being switched off is *recorded* as a
        refusal rather than vanishing. An agent repeatedly bouncing off a read-only
        control plane is something an operator should be able to see.

        ``__signature__`` is copied across explicitly: the SDK builds each tool's schema
        by inspecting the callable, and a wrapper advertising ``**kwargs`` would publish
        a tool that takes anything and documents nothing.

        ``eval_str=True`` on that copy is not optional. This module uses postponed
        annotation evaluation, so a signature taken without it carries the *string*
        ``"dict[str, Any]"`` as the return type; setting ``__signature__`` then bypasses
        the SDK's own resolution, and every write tool publishes a schema wrapping its
        result in a ``result`` field that the read tools do not have. The surface stays
        up and the shapes silently disagree.
        """

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            async def wrapper(**kwargs: Any) -> Any:
                if not settings.mcp_write_enabled:
                    message = (
                        "write tools are disabled on this control plane "
                        "(VLLMBENCH_MCP_WRITE_ENABLED=false)"
                    )
                    await _record(fn.__name__, kwargs, "refused", error=message)
                    raise ValueError(message)
                try:
                    result = await fn(**kwargs)
                except Exception as exc:
                    # Refused is the surface declining something it understood; failed is
                    # everything else. The distinction matters when reading the log back:
                    # a hundred refusals is an agent that needs better instructions, and
                    # one failure is a bug.
                    outcome = "refused" if _is_refusal(exc) else "failed"
                    await _record(fn.__name__, kwargs, outcome, error=str(exc))
                    raise
                subject = (
                    str(result.get(subject_key))
                    if subject_key and isinstance(result, dict) and result.get(subject_key)
                    else None
                )
                await _record(fn.__name__, kwargs, "succeeded", subject=subject)
                return result

            wrapper.__signature__ = inspect.signature(fn, eval_str=True)  # type: ignore[attr-defined]
            return server.tool()(wrapper)

        return decorate

    @write_tool(subject_key="config_hash")
    async def create_config(name: str, yaml: str, notes: str | None = None) -> dict[str, Any]:
        """Store a vLLM server configuration, exactly as given.

        The YAML is what will be written to disk and passed to `vllm serve --config`, byte
        for byte — nothing is reordered, normalized or re-emitted. Configurations are
        content-addressed, so submitting text that already exists returns the existing one
        rather than a duplicate.
        """
        async with sessions() as session:
            config = await run_routes.create_config(
                ConfigCreate(name=name, yaml=yaml, notes=notes), session
            )
            return {"config_hash": config.config_hash, "name": config.name}

    @write_tool(subject_key="workload_hash")
    async def create_workload(
        name: str,
        dataset_name: str = "random",
        num_prompts: int = 200,
        max_concurrency: int | None = None,
        request_rate: float | None = None,
        input_len: int | None = None,
        output_len: int | None = None,
    ) -> dict[str, Any]:
        """Define the traffic a run is measured under.

        Leave `max_concurrency` or `request_rate` null for unbounded — that is genuinely
        the absence of a limit, and is not the same as zero. Workloads are
        content-addressed on what they send, not on their name.
        """
        async with sessions() as session:
            workload = await run_routes.create_workload(
                WorkloadCreate(
                    name=name,
                    dataset_name=dataset_name,
                    num_prompts=num_prompts,
                    max_concurrency=max_concurrency,
                    request_rate=request_rate,
                    input_len=input_len,
                    output_len=output_len,
                ),
                session,
            )
            return {"workload_hash": workload.workload_hash, "name": workload.name}

    @write_tool(subject_key="id")
    async def create_sweep(
        name: str,
        gpu_host_id: str,
        config_hashes: list[str],
        workload_hashes: list[str],
        replicates: int = 3,
        tensor_parallel_sizes: list[int] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Author a sweep. Every run is created immediately, in the order it will execute.

        Starts queued: materializing the runs is committing to them. A host holds one
        active sweep at a time, so this fails if the host is busy rather than interleaving
        two sweeps' engine restarts. The matrix is bounded, so a wrong product is refused
        here rather than filling the queue with days of work.

        Returns the exact run count and how many engine loads the plan implies — most of a
        sweep's wall clock is model loading, so that second number is the one that
        predicts how long it will take.
        """
        async with sessions() as session:
            configs = {c.config_hash: c.id for c in await run_routes.list_configs(session)}
            workloads = {w.workload_hash: w.id for w in await run_routes.list_workloads(session)}
            missing = [h for h in config_hashes if h not in configs] + [
                h for h in workload_hashes if h not in workloads
            ]
            if missing:
                raise ValueError(f"unknown config or workload hashes: {', '.join(missing)}")

            sweep = await sweep_routes.create_sweep(
                SweepCreate(
                    name=name,
                    description=description,
                    gpu_host_id=uuid.UUID(gpu_host_id),
                    server_config_ids=[configs[h] for h in config_hashes],
                    workload_ids=[workloads[h] for h in workload_hashes],
                    replicates=replicates,
                    tensor_parallel_sizes=tensor_parallel_sizes,
                    initiated_by=InitiatedBy.MCP,
                    initiated_by_client=MCP_CLIENT_NAME,
                ),
                session,
            )
            return {
                "id": str(sweep.id),
                "status": sweep.status,
                "total_runs": sweep.progress.total,
                "engine_starts": sweep.engine_starts,
                "is_synthetic": sweep.is_synthetic,
            }

    @write_tool(subject_key="id")
    async def cancel_sweep(sweep_id: str) -> dict[str, Any]:
        """Stop a sweep. Queued runs are cancelled and one in flight is interrupted.

        Runs that already finished keep their results — cancelling a sweep is a decision
        about the work remaining, never about the measurements already taken.
        """
        async with sessions() as session:
            sweep = await sweep_routes.cancel_sweep(uuid.UUID(sweep_id), session)
            return {
                "id": str(sweep.id),
                "status": sweep.status,
                "progress": sweep.progress.model_dump(),
            }

    # -- Resources ---------------------------------------------------------------
    #
    # Resources rather than more tools because the client decides when to read them and
    # can cache them by URI. Both of these are stable content addressed by an identifier
    # the agent already holds: a config's YAML never changes (the hash *is* the text),
    # and a finished sweep's report never changes because its runs are immutable.
    #
    # They are prose, not JSON. A tool result is parsed; a resource is read, and the
    # things worth saying about a sweep — that these two groups may not be compared, that
    # a difference is inside the spread — are sentences rather than fields. The JSON is
    # still there in get_pareto for an agent that wants to compute on it.

    @server.resource(
        "vllmbench://config/{config_hash}",
        name="Server configuration",
        description="The exact vLLM YAML behind a config hash, byte for byte.",
        mime_type="text/yaml",
    )
    async def config_resource(config_hash: str) -> str:
        """The configuration text itself, with nothing wrapped around it.

        Invariant 5: what is stored is what is written to disk and passed to
        `vllm serve --config`. Returning it as YAML rather than a JSON envelope means an
        agent can hand the bytes straight to a host, which is the point.
        """
        async with sessions() as session:
            for config in await run_routes.list_configs(session):
                if config.config_hash == config_hash:
                    return config.yaml
        raise ValueError(f"no config with hash {config_hash!r}")

    @server.resource(
        "vllmbench://sweep/{sweep_id}/report",
        name="Sweep report",
        description="What one sweep measured, partitioned into comparable sets.",
        mime_type="text/markdown",
    )
    async def sweep_report(sweep_id: str) -> str:
        """One sweep, written out.

        Partitioned the same way every chart is: a section per host / GPU / vLLM version
        / bench-client location, because those cannot be read as one series. An agent
        that reads only the numbers and not the headings will still not have two GPU
        models in one table, since they are not in one table.
        """
        async with sessions() as session:
            identifier = uuid.UUID(sweep_id)
            sweep = await sweep_routes.get_sweep(identifier, session)
            analysis = await analysis_routes.analysis_points(
                session,
                source=RunSource.SYNTHETIC if sweep.is_synthetic else RunSource.REAL,
                sweep_id=[identifier],
            )
        return _render_sweep_report(sweep, analysis)

    @tool
    async def server_info() -> dict[str, Any]:
        """What this control plane is, and what it will and will not do."""
        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "write_tools_enabled": settings.mcp_write_enabled,
            "populations": [s.value for s in RunSource],
            "notes": [
                "Runs are immutable once terminal; no tool mutates or deletes one.",
                "Real and synthetic runs are never returned together.",
                "Throughput is reported per GPU as well as aggregate.",
            ],
        }

    return server


class _AtRoot:
    """Serves a mounted ASGI app at the mount path itself, with no redirect.

    Starlette's ``Mount("/mcp")`` matches ``/mcp/...`` but not bare ``/mcp``, which the
    router then answers with a 307. An MCP client does not follow it, and the failure
    surfaces as "Unexpected content type" rather than as a redirect — so the URL an
    operator would naturally write appears broken for a reason nothing in the message
    hints at.

    This forwards to the *same* Starlette app the mount serves, with the path rewritten to
    its root. Forwarding to that app rather than to the session manager underneath it is
    the load-bearing part: the bearer-token check is middleware on that app, and calling
    the manager directly would serve every request unauthenticated. Verified by a test
    that a wrong token is refused — which is how the shortcut was caught.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._inner({**scope, "path": "/", "raw_path": b"/"}, receive, send)


def mount_mcp(app: Any, server: MCPServer, settings: ApiSettings) -> None:
    """Attach the MCP transport at ``/mcp``, answering with and without a trailing slash."""
    inner = server.streamable_http_app(
        streamable_http_path="/",
        # ADR 0001 targets a stateless core, so the service can sit behind a proxy with no
        # session affinity.
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=bool(settings.mcp_allowed_hosts),
            allowed_hosts=list(settings.mcp_allowed_hosts),
        ),
    )
    app.router.routes.append(
        Route("/mcp", endpoint=_AtRoot(inner), methods=["GET", "POST", "DELETE"])
    )
    app.mount("/mcp", inner)
