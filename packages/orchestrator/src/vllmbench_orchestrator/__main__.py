"""Orchestrator entry point.

A separate service from the API on purpose: sweeps run for hours, and restarting the API
must not kill one in flight (ROADMAP 0.4.0, "sweep survives an API restart").

Milestone 0.1.0 scope: start, confirm the database is reachable, and idle. The sweep state
machine lands in 0.4.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_protocol import PROTOCOL_VERSION, __version__

log = logging.getLogger("vllmbench.orchestrator")

IDLE_INTERVAL_SECONDS = 30


async def _wait_for_database(
    factory: async_sessionmaker[AsyncSession], *, attempts: int = 30
) -> None:
    """Poll until Postgres answers.

    Compose orders startup, but a healthy container is not the same as an accepting
    connection pool, and the orchestrator must not crash-loop through the gap.
    """
    for attempt in range(1, attempts + 1):
        try:
            async with factory() as session:
                await session.execute(text("SELECT 1"))
            log.info("database reachable")
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            log.warning("database not ready (%s/%s): %s", attempt, attempts, exc)
            await asyncio.sleep(2)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    )
    log.info("orchestrator %s (protocol %d) starting", __version__, PROTOCOL_VERSION)

    engine = create_engine()
    sessions = create_session_factory(engine)

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stopping.set)

    try:
        await _wait_for_database(sessions)
        log.info("idle: no sweep engine until milestone 0.4.0")
        while not stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stopping.wait(), timeout=IDLE_INTERVAL_SECONDS)
    finally:
        log.info("orchestrator stopping")
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
