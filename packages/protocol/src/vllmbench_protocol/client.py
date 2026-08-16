"""HTTP client for the GPU-host agent.

Lives in ``protocol`` rather than in the API or the orchestrator because both of them
need it and neither should depend on the other. It costs the GPU host nothing: the agent
already depends on httpx to scrape vLLM's ``/metrics``.
"""

from __future__ import annotations

from types import TracebackType

import httpx

from vllmbench_protocol.errors import AgentAuthError, AgentUnreachable, ProtocolMismatch
from vllmbench_protocol.version import PROTOCOL_VERSION
from vllmbench_protocol.wire import AUTH_SCHEME, HealthResponse, HostInfo

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


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
        timeout: httpx.Timeout | None = None,
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
            timeout=timeout or DEFAULT_TIMEOUT,
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

        if response.status_code in (401, 403):
            raise AgentAuthError(self.base_url)
        response.raise_for_status()
        return response

    async def health(self) -> HealthResponse:
        """Liveness only. Does not require a valid token, and does not check protocol."""
        return HealthResponse.model_validate((await self._get("/health")).json())

    async def host_info(self, *, check_protocol: bool = True) -> HostInfo:
        info = HostInfo.model_validate((await self._get("/host-info")).json())
        if check_protocol and info.protocol_version != self._expected_protocol_version:
            raise ProtocolMismatch(
                self.base_url, info.protocol_version, self._expected_protocol_version
            )
        return info
