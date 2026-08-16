"""Bearer-token authentication for the agent's authenticated endpoints."""

from __future__ import annotations

import hmac
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Header, HTTPException, status


def token_dependency(expected: str) -> Callable[..., Coroutine[Any, Any, None]]:
    """Build a FastAPI dependency that checks the bearer token.

    A closure over the expected token rather than a module-level import, so tests can
    build an app with a known token without mutating global state.
    """

    async def require_token(authorization: str | None = Header(default=None)) -> None:
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() != "bearer" or not credential:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="expected 'Bearer <token>'",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Constant-time. The comparison is cheap, and the alternative leaks the token one
        # byte at a time to anyone who can measure response latency.
        if not hmac.compare_digest(credential, expected):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid token")

    return require_token
