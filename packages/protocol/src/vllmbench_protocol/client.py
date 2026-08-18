"""HTTP client for the GPU-host agent.

Lives in ``protocol`` rather than in the API or the orchestrator because both of them
need it and neither should depend on the other. It costs the GPU host nothing: the agent
already depends on httpx to scrape vLLM's ``/metrics``.
"""

from __future__ import annotations

from types import TracebackType

import httpx

from vllmbench_protocol.errors import (
    AgentAuthError,
    AgentError,
    AgentUnreachable,
    ProtocolMismatch,
)
from vllmbench_protocol.failures import FAILURE_KIND_HEADER
from vllmbench_protocol.version import PROTOCOL_VERSION
from vllmbench_protocol.wire import (
    AUTH_SCHEME,
    BenchRequest,
    BenchResponse,
    CancelResponse,
    HealthResponse,
    HostInfo,
    ServerStatus,
    StartServerRequest,
)

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Sentinel so `timeout=None` can mean "no timeout" rather than "use the default".
_UNSET = httpx.Timeout(-1.0)


class AgentClient:
    """Talks to one agent.

    Every call that reaches an authenticated endpoint goes through the protocol-version
    check first, so a mismatched agent fails at connect time rather than at the moment a
    sweep has results to write.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: httpx.Timeout | None = _UNSET,
        client: httpx.AsyncClient | None = None,
        expected_protocol_version: int = PROTOCOL_VERSION,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        # Injectable so the mismatch path is testable without patching module globals,
        # and so a future caller can inspect a host it cannot fully talk to.
        self._expected_protocol_version = expected_protocol_version
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            # `timeout=None` is an explicit request for no timeout, used when starting a
            # server: model load legitimately takes minutes, and cutting it off would
            # abandon a healthy engine with nobody tracking it.
            timeout=DEFAULT_TIMEOUT if timeout is _UNSET else timeout,
            headers={"Authorization": f"{AUTH_SCHEME} {token}"},
        )

    async def __aenter__(self) -> AgentClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self._client.get(path)
        except httpx.HTTPError as exc:
            raise AgentUnreachable(self.base_url, str(exc)) from exc
        return self._checked(response)

    async def _post(self, path: str, json: object | None = None) -> httpx.Response:
        try:
            response = await self._client.post(path, json=json)
        except httpx.HTTPError as exc:
            raise AgentUnreachable(self.base_url, str(exc)) from exc
        return self._checked(response)

    def _checked(self, response: httpx.Response) -> httpx.Response:
        if response.status_code in (401, 403):
            raise AgentAuthError(self.base_url)
        if response.status_code >= 400:
            # Surface the agent's own explanation rather than a bare status. For a
            # failed model load that detail is the vLLM log tail, and it is the only
            # thing that distinguishes an OOM from a bad configuration.
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            error = AgentError(
                f"agent at {self.base_url} returned {response.status_code}: {detail}"
            )
            # The agent's own name for what went wrong, when it sent one. It saw the
            # whole vLLM log rather than the tail that fits in a response, and it knows
            # which of its own deadlines expired — neither of which survives the trip.
            # Absent from an older agent, which is why nothing downstream requires it.
            error.reported_kind = response.headers.get(FAILURE_KIND_HEADER)
            raise error
        return response

    async def health(self) -> HealthResponse:
        """Liveness only. Does not require a valid token, and does not check protocol."""
        return HealthResponse.model_validate((await self._get("/health")).json())

    async def host_info(self, *, check_protocol: bool = True) -> HostInfo:
        payload = (await self._get("/host-info")).json()

        # Check the protocol version against the *raw* payload, before model validation.
        #
        # Ordering matters more than it looks. The wire models use extra="forbid", so a
        # newer agent sending a field this build has never heard of fails validation —
        # and validating first turns the one situation the version check exists to
        # explain into an opaque "extra inputs are not permitted". That is exactly what
        # happened against a real host: a stale control plane returned a 500 mentioning
        # a field name, when it should have said "this agent speaks protocol 2 and I
        # speak 1".
        if check_protocol and isinstance(payload, dict):
            reported = payload.get("protocol_version")
            if reported != self._expected_protocol_version:
                raise ProtocolMismatch(
                    self.base_url,
                    reported if isinstance(reported, int) else -1,
                    self._expected_protocol_version,
                )

        return HostInfo.model_validate(payload)

    # -- server lifecycle ----------------------------------------------------------

    async def server_status(self) -> ServerStatus:
        return ServerStatus.model_validate((await self._get("/server")).json())

    async def start_server(self, request: StartServerRequest) -> ServerStatus:
        """Start a vLLM server and wait for it to be ready.

        This blocks for as long as the model takes to load, which is why callers
        construct the client with ``timeout=None``. A read timeout here would abandon a
        server that is loading perfectly well, leaving it running with nobody tracking
        it — the orphan case, created by the client rather than a crash.
        """
        response = await self._post("/server/start", json=request.model_dump(mode="json"))
        return ServerStatus.model_validate(response.json())

    async def stop_server(self) -> ServerStatus:
        return ServerStatus.model_validate((await self._post("/server/stop")).json())

    # -- benchmarking --------------------------------------------------------------

    async def bench(self, request: BenchRequest) -> BenchResponse:
        response = await self._post("/bench", json=request.model_dump(mode="json"))
        return BenchResponse.model_validate(response.json())

    async def cancel_bench(self) -> CancelResponse:
        """Stop the benchmark in flight, if there is one.

        Uses its own short timeout rather than the client's: this is called to reclaim a
        host, often while the normal client is configured with no timeout at all so that
        a model load can take as long as it needs. Waiting indefinitely to cancel would
        defeat the point.
        """
        response = await self._client.post("/bench/cancel", timeout=30.0)
        return CancelResponse.model_validate(self._checked(response).json())
