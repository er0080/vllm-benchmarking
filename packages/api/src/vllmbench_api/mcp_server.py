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

*A schema that is the documentation.* No agent reads a README before calling a tool, so
whatever ``tools/list`` returns is the whole contract. Every parameter carries a
description, every closed set of values is published as an ``enum``, and every tool
carries behavioural hints — a harness deciding what to auto-approve should not have to
parse English to learn that ``cancel_sweep`` is not a read. What cannot be expressed in
the schema is refused at the boundary instead: an unknown value is an error here, never a
quiet substitution, because a tool reporting success on a question it did not answer is
the one failure an agent cannot detect.
"""

from __future__ import annotations

import functools
import inspect
import logging
import uuid
from collections.abc import Callable
from typing import Annotated, Any, Literal

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl, Field
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from vllmbench_api.analysis import METRICS, METRICS_BY_KEY, PARETO_X, PARETO_Y, RunSource
from vllmbench_api.reports import render_sweep_report
from vllmbench_api.routers import analysis as analysis_routes
from vllmbench_api.routers import hosts as host_routes
from vllmbench_api.routers import runs as run_routes
from vllmbench_api.routers import sweeps as sweep_routes
from vllmbench_api.schemas import (
    ConfigAnnotate,
    ConfigCreate,
    ConfigValidationRequest,
    RunOut,
    SweepCreate,
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

#: Valid Pareto axes, derived from the metric catalogue rather than restated beside it.
#: The published ``enum`` and what the analysis layer can actually plot are then the same
#: list, which is the only arrangement in which they cannot drift apart.
METRIC_KEYS: tuple[str, ...] = tuple(metric.key for metric in METRICS)

#: Parameters shared across tools, described once. A description written nineteen times
#: is a description that will be right in eighteen places.
SourceArg = Annotated[
    Literal["real", "synthetic"],
    Field(
        description=(
            "Which population to read. 'real' is measurements from real hardware; "
            "'synthetic' is the mock agent or the CPU backend. There is no value meaning "
            "both — synthetic runs are quarantined and never appear beside real ones."
        )
    ),
]

LimitArg = Annotated[
    int | None,
    Field(
        description=(
            f"How many rows to return. Defaults to {DEFAULT_PAGE}; capped at {MAX_PAGE} "
            "however large a value is passed."
        )
    ),
]

HostIdArg = Annotated[
    str | None,
    Field(
        description="Restrict to one GPU host, by the id from list_hosts. Null means every host."
    ),
]

SweepIdFilterArg = Annotated[
    str | None,
    Field(description="Restrict to one sweep, by the id from list_sweeps. Null means every run."),
]

ParetoAxisArg = Annotated[
    str,
    Field(
        description=(
            "Metric key for this axis. The response's own `metrics` catalogue says what "
            "each one measures and its unit; lower-is-better metrics are allowed and the "
            "frontier orients them correctly. An unrecognised key is refused, not "
            "substituted."
        ),
        # Derived from the catalogue, so the advertised values are the accepted values.
        json_schema_extra={"enum": list(METRIC_KEYS)},
    ),
]


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


def _metric_key(value: str, parameter: str) -> str:
    """Parse a Pareto axis, refusing anything the analysis layer cannot plot.

    The HTTP endpoint underneath substitutes the default axis for an unrecognised key,
    which is right for a browser: a stale bookmark should still draw a chart rather than
    an error page. It is wrong here, and the difference is who is reading. A person sees
    the axis label and knows immediately that they are looking at something else; an agent
    is handed a number that answers a question it did not ask, from a call that reported
    success. So the substitution is refused before it can happen, and the fallback stays
    where it belongs — see ``vllmbench_api.routers.analysis.analysis_points``.
    """
    if value in METRICS_BY_KEY:
        return value
    raise ValueError(
        f"{parameter} must be one of: {', '.join(METRIC_KEYS)}. "
        f"Got {value!r}, which is not a metric this control plane records."
    )


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
            "trivially while possibly being worse per device. Two resource templates are "
            "also served — vllmbench://config/{config_hash} for a configuration's exact "
            "YAML and vllmbench://sweep/{sweep_id}/report for what a sweep measured. "
            "Being templated, they appear under resources/templates/list rather than "
            "resources/list, which is empty."
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

    def tool(*, open_world: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a read tool, with the hints that say it is one.

        ``read_only_hint`` is the field a harness consults when deciding what it may call
        without asking, so it is set here rather than per tool: a read added later is
        annotated by construction, and cannot arrive looking like a write.

        ``inspect.cleandoc`` is not cosmetic. The SDK publishes ``__doc__`` verbatim, and
        a docstring indented to sit inside a nested function arrives with eight leading
        spaces on every line after the first — which is a code block in any client that
        renders the description as markdown.
        """

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            return server.tool(
                description=inspect.cleandoc(fn.__doc__ or ""),
                annotations=ToolAnnotations(
                    read_only_hint=True,
                    destructive_hint=False,
                    idempotent_hint=True,
                    # False means the tool answers from this control plane's own database.
                    # True is reserved for the ones that reach across the host boundary.
                    open_world_hint=open_world,
                ),
            )(fn)

        return decorate

    # -- Inventory ---------------------------------------------------------------

    @tool()
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

    @tool()
    async def list_configs(limit: LimitArg = None) -> list[dict[str, Any]]:
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

    @tool()
    async def get_config(
        config_hash: Annotated[
            str,
            Field(
                description=(
                    "The config's content hash, from list_configs or from a run's "
                    "`config_hash`. Configurations are addressed by content, so this "
                    "identifies exact bytes rather than a name that could be reused."
                )
            ),
        ],
    ) -> dict[str, Any]:
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

    @tool()
    async def get_config_lineage(
        config_hash: Annotated[
            str, Field(description="The config's content hash, from list_configs.")
        ],
    ) -> dict[str, Any]:
        """Where a configuration came from, and what came from it.

        The record of a tuning session: which config this was edited from, back to the
        original, and which configs were edited from it. A configuration on its own says
        what was set; this says what it was set *instead of*.
        """
        async with sessions() as session:
            result = await run_routes.config_lineage(config_hash, session)
            return result.model_dump(mode="json")

    @tool()
    async def list_workloads(limit: LimitArg = None) -> list[dict[str, Any]]:
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

    @tool()
    async def list_sweeps(limit: LimitArg = None) -> list[dict[str, Any]]:
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

    @tool()
    async def get_sweep(
        sweep_id: Annotated[str, Field(description="The sweep's id, from list_sweeps.")],
    ) -> dict[str, Any]:
        """One sweep, including how far through it is."""
        async with sessions() as session:
            sweep = await sweep_routes.get_sweep(uuid.UUID(sweep_id), session)
            return sweep.model_dump(mode="json")

    # -- Runs --------------------------------------------------------------------

    @tool()
    async def query_runs(limit: LimitArg = None) -> list[dict[str, Any]]:
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
                    # Both, for the same reason a person gets both: the kind is what
                    # makes a page of failed runs answerable at a glance, and the text is
                    # the only part that says which card wanted how much memory.
                    "failure_kind": run.failure_kind,
                    "error": run.error,
                }
                for run in found
            ]

    @tool()
    async def get_run(
        run_id: Annotated[
            str, Field(description="The run's id, from query_runs or a sweep's run list.")
        ],
    ) -> dict[str, Any]:
        """One run in full, including every flattened metric."""
        async with sessions() as session:
            # Through the same response model the HTTP route serializes with, rather than
            # reading the ORM object directly: one definition of what a run looks like on
            # the wire, so the two interfaces cannot describe it differently.
            run = await run_routes.get_run(uuid.UUID(run_id), session)
            return RunOut.model_validate(run).model_dump(mode="json")

    # -- Analysis ----------------------------------------------------------------

    @tool()
    async def get_pareto(
        source: SourceArg = "real",
        host_id: HostIdArg = None,
        sweep_id: SweepIdFilterArg = None,
        pareto_x: ParetoAxisArg = PARETO_X,
        pareto_y: ParetoAxisArg = PARETO_Y,
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
                # Parsed rather than passed through: the endpoint would substitute the
                # default for an unrecognised key, and this caller cannot see that happen.
                pareto_x=_metric_key(pareto_x, "pareto_x"),
                pareto_y=_metric_key(pareto_y, "pareto_y"),
            )
            return result.model_dump(mode="json")

    @tool()
    async def compare_runs(
        left: Annotated[
            str,
            Field(description="A `point_id` from get_pareto — one config, workload and TP size."),
        ],
        right: Annotated[str, Field(description="The `point_id` to compare it against.")],
        source: SourceArg = "real",
    ) -> dict[str, Any]:
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

    # Reaches across the host boundary when gpu_host_id is given: the target host's own
    # vLLM version and device count are what it checks against.
    @tool(open_world=True)
    async def validate_config(
        yaml: Annotated[
            str,
            Field(
                description=(
                    "The candidate vLLM YAML, exactly as it would be passed to "
                    "`vllm serve --config`. Nothing is stored and nothing is rewritten."
                )
            ),
        ],
        gpu_host_id: Annotated[
            str | None,
            Field(
                description=(
                    "Check against this host's own vLLM version and device count, by the "
                    "id from list_hosts. Null checks against the reference version in the "
                    "abstract and skips the topology checks rather than guessing at them."
                )
            ),
        ] = None,
        tensor_parallel_is_swept: Annotated[
            bool,
            Field(
                description=(
                    "True when this config is destined for a sweep that varies "
                    "tensor-parallel-size. The sweep overrides that setting per run, so a "
                    "value in the file is not the value that will run — set this and the "
                    "topology checks stop objecting to a number that is about to change."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Check a configuration before committing GPU time to it.

        Reads only — nothing is stored and nothing is rewritten. Every finding names the
        setting at fault and says what is wrong with it; the fix stays with you.

        Pass `gpu_host_id` to check against that host's own vLLM version and device
        count. Without it the config is checked in the abstract against the reference
        version, and the topology checks are skipped rather than guessed at.

        `severity` matters: "error" means `vllm serve` will refuse to start, "warning"
        means it will start and may not do what you meant. `checked_against` names the
        vLLM version whose arguments were used, and `exact_version_match` is false when
        the target host runs something this control plane has no capture for — the result
        is still useful, but an unknown setting may be one that version added.
        """
        async with sessions() as session:
            result = await run_routes.validate_config_endpoint(
                ConfigValidationRequest(
                    yaml=yaml,
                    gpu_host_id=uuid.UUID(gpu_host_id) if gpu_host_id else None,
                    tensor_parallel_is_swept=tensor_parallel_is_swept,
                ),
                session,
            )
            return result.model_dump(mode="json")

    @tool()
    async def get_run_telemetry(
        run_id: Annotated[str, Field(description="The run's id, from query_runs.")],
        max_samples: Annotated[
            int,
            Field(
                description=(
                    "Ceiling on samples returned per series. The response's `stride` says "
                    "what thinning this forced; `sample_count` says what was recorded."
                )
            ),
        ] = 200,
    ) -> dict[str, Any]:
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

    def write_tool(
        subject_key: str | None = None,
        *,
        destructive: bool = False,
        idempotent: bool = True,
        open_world: bool = False,
    ) -> Callable[..., Any]:
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
            # From ``fn``, not from ``wrapper``: functools.wraps copies the docstring, but
            # taking it from the original is what makes that a convenience rather than a
            # dependency.
            return server.tool(
                description=inspect.cleandoc(fn.__doc__ or ""),
                annotations=ToolAnnotations(
                    read_only_hint=False,
                    destructive_hint=destructive,
                    idempotent_hint=idempotent,
                    open_world_hint=open_world,
                ),
            )(wrapper)

        return decorate

    # Idempotent because configurations are content-addressed: the same YAML submitted
    # twice returns the same row, so a retry after a dropped response is safe.
    @write_tool(subject_key="config_hash")
    async def create_config(
        name: Annotated[
            str,
            Field(
                description=(
                    "A human label for this configuration. Not its identity — two configs "
                    "with the same name and different YAML are two configurations."
                )
            ),
        ],
        yaml: Annotated[
            str,
            Field(
                description=(
                    "The vLLM YAML, stored byte for byte and passed to "
                    "`vllm serve --config` unmodified. Nothing is reordered or re-emitted, "
                    "so comments survive and the hash is of exactly what will run."
                )
            ),
        ],
        notes: Annotated[
            str | None, Field(description="Free text about why this configuration exists.")
        ] = None,
        derived_from: Annotated[
            str | None,
            Field(
                description=(
                    "The config hash this was edited from, when it is an edit. Builds the "
                    "lineage get_config_lineage reads back."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Store a vLLM server configuration, exactly as given.

        The YAML is what will be written to disk and passed to `vllm serve --config`, byte
        for byte — nothing is reordered, normalized or re-emitted. Configurations are
        content-addressed, so submitting text that already exists returns the existing one
        rather than a duplicate.

        Pass `derived_from` (a config hash) when this is an edit of an earlier config. That
        is what makes a tuning session readable afterwards: without it, twenty
        configurations are twenty unrelated files rather than a record of what was tried in
        what order. It is recorded only when this call actually creates a row — a config
        that already exists has a history that already happened.
        """
        async with sessions() as session:
            parent_id = None
            if derived_from:
                for candidate in await run_routes.list_configs(session):
                    if candidate.config_hash == derived_from:
                        parent_id = candidate.id
                        break
                if parent_id is None:
                    raise ValueError(f"no config with hash {derived_from!r} to derive from")
            config = await run_routes.create_config(
                ConfigCreate(name=name, yaml=yaml, notes=notes, parent_id=parent_id), session
            )
            return {"config_hash": config.config_hash, "name": config.name}

    @write_tool(subject_key="config_hash")
    async def annotate_config(
        config_hash: Annotated[
            str, Field(description="The configuration to annotate, by content hash.")
        ],
        justified_by_run_id: Annotated[
            str | None,
            Field(
                description=(
                    "A run that used this configuration, as the evidence for keeping it. "
                    "A run of a different config is not evidence and is refused."
                )
            ),
        ] = None,
        justification_note: Annotated[
            str | None,
            Field(description="What that run showed — the reading, not just the pointer."),
        ] = None,
        notes: Annotated[
            str | None, Field(description="Replacement free-text notes. Null leaves them alone.")
        ] = None,
        name: Annotated[
            str | None, Field(description="Replacement label. Null leaves it alone.")
        ] = None,
    ) -> dict[str, Any]:
        """Record why a configuration is worth keeping.

        Changes what is *said about* a config, never the config. The YAML is not editable —
        editing it is a different content hash and therefore a different configuration, so
        an edit is a `create_config` with `derived_from` set.

        `justified_by_run_id` must be a run that actually used this configuration. A run of
        something else is not evidence for it, and is refused rather than stored.
        """
        async with sessions() as session:
            result = await run_routes.annotate_config(
                config_hash,
                ConfigAnnotate(
                    name=name,
                    notes=notes,
                    justified_by_run_id=(
                        uuid.UUID(justified_by_run_id) if justified_by_run_id else None
                    ),
                    justification_note=justification_note,
                ),
                session,
            )
            return result.model_dump(mode="json")

    # Content-addressed on what it sends, so the same traffic definition submitted twice
    # is one workload.
    @write_tool(subject_key="workload_hash")
    async def create_workload(
        name: Annotated[
            str, Field(description="A human label. Not part of the workload's identity.")
        ],
        dataset_name: Annotated[
            str,
            Field(
                description=(
                    "The upstream dataset `vllm bench serve` will send — 'random' "
                    "synthesises prompts to input_len/output_len."
                )
            ),
        ] = "random",
        num_prompts: Annotated[
            int, Field(description="How many requests the benchmark sends in total.")
        ] = 200,
        max_concurrency: Annotated[
            int | None,
            Field(
                description=(
                    "Requests in flight at once. Null is unbounded — genuinely no limit, "
                    "which is not the same as zero. This is the axis a saturation curve "
                    "sweeps."
                )
            ),
        ] = None,
        request_rate: Annotated[
            float | None,
            Field(description="Requests per second offered. Null is unbounded (upstream's inf)."),
        ] = None,
        input_len: Annotated[
            int | None, Field(description="Prompt tokens per request, for synthetic datasets.")
        ] = None,
        output_len: Annotated[
            int | None, Field(description="Tokens to generate per request, for synthetic datasets.")
        ] = None,
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

    # Not idempotent, and the only tool here that commits another machine to hours of
    # work: calling it twice authors two sweeps, and the second is refused only because
    # the host is already busy with the first.
    @write_tool(subject_key="id", idempotent=False, open_world=True)
    async def create_sweep(
        name: Annotated[str, Field(description="A human label for the sweep.")],
        gpu_host_id: Annotated[
            str, Field(description="The host that will run it, by the id from list_hosts.")
        ],
        config_hashes: Annotated[
            list[str],
            Field(
                description=(
                    "Configurations to sweep, by content hash. One axis of the matrix; "
                    "runs are grouped by config so the engine reloads once per config "
                    "rather than once per run."
                )
            ),
        ],
        workload_hashes: Annotated[
            list[str],
            Field(description="Workloads to sweep, by content hash. The second axis."),
        ],
        replicates: Annotated[
            int,
            Field(
                description=(
                    "Runs per matrix point. More than one because a single measurement "
                    "has no spread, and a difference smaller than the spread is not a "
                    "result."
                )
            ),
        ] = 3,
        tensor_parallel_sizes: Annotated[
            list[int] | None,
            Field(
                description=(
                    "Tensor-parallel sizes to sweep, overriding what each config's YAML "
                    "says. Null runs each config at the size it declares. Throughput is "
                    "reported per GPU as well as aggregate, because a wider run "
                    "out-throughputs a narrower one trivially."
                )
            ),
        ] = None,
        description: Annotated[
            str | None, Field(description="What this sweep is trying to find out.")
        ] = None,
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

    # Destructive in the MCP sense — it takes work away that cannot be given back, and a
    # harness should ask before calling it. Idempotent all the same: cancelling an already
    # cancelled sweep changes nothing further.
    @write_tool(subject_key="id", destructive=True)
    async def cancel_sweep(
        sweep_id: Annotated[
            str, Field(description="The sweep to stop, by the id from list_sweeps.")
        ],
    ) -> dict[str, Any]:
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
        return render_sweep_report(sweep, analysis)

    @tool()
    async def server_info() -> dict[str, Any]:
        """What this control plane is, and what it will and will not do.

        Names the resource templates as well as the tools. Both resources here are
        templated, so they are advertised under `resources/templates/list` and *not* under
        `resources/list`, which is empty and correct — a caller that checks only the latter
        concludes this server has no resources at all. That has happened. Stating them
        here is cheaper than expecting everyone to know the distinction.
        """
        return {
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "write_tools_enabled": settings.mcp_write_enabled,
            "populations": [s.value for s in RunSource],
            "pareto_axes": list(METRIC_KEYS),
            "resource_templates": [
                "vllmbench://config/{config_hash}",
                "vllmbench://sweep/{sweep_id}/report",
            ],
            "notes": [
                "Runs are immutable once terminal; no tool mutates or deletes one.",
                "Real and synthetic runs are never returned together.",
                "Throughput is reported per GPU as well as aggregate.",
                "Resources are templated: list them with resources/templates/list.",
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
