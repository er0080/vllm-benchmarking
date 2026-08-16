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
from vllmbench_orchestrator.runner import claim_next_run, execute_run
from vllmbench_orchestrator.settings import OrchestratorSettings
from vllmbench_protocol import PROTOCOL_VERSION, __version__

log = logging.getLogger("vllmbench.orchestrator")

# How often to look for queued work when there is none. Short enough that a run
# triggered from the UI starts promptly, long enough not to hammer the database while
# idle overnight.
POLL_INTERVAL_SECONDS = 2.0


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


async def _poll_once(factory: async_sessionmaker[AsyncSession], token: str) -> bool:
    """Claim and execute one run. Returns whether there was work to do.

    One at a time and deliberately: a GPU host runs a single engine, so a second
    concurrent run would contend for the same VRAM and measure the contention.
    """
    async with factory() as session:
        run = await claim_next_run(session)
        if run is None:
            return False
        await execute_run(session, run, token)
        return True


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

        settings = OrchestratorSettings()
        if not settings.token:
            log.warning(
                "VLLMBENCH_TOKEN is not set; runs will fail to authenticate against the agent"
            )

        log.info("polling for queued runs every %.1fs", POLL_INTERVAL_SECONDS)
        while not stopping.is_set():
            try:
                worked = await _poll_once(sessions, settings.token)
            except Exception:
                # A crash here would stop every future run, not just this one. Log and
                # continue; the run itself has already been marked failed with a reason.
                log.exception("poll iteration failed")
                worked = False

            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopping.wait(), timeout=POLL_INTERVAL_SECONDS)
    finally:
        log.info("orchestrator stopping")
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
