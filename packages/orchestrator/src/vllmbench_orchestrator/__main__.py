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
import uuid
from typing import NamedTuple

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vllmbench_db.enums import RunStatus, SweepStatus
from vllmbench_db.models import Run, Sweep
from vllmbench_db.retention import prune_telemetry
from vllmbench_db.session import create_engine, create_session_factory, database_password
from vllmbench_orchestrator.runner import claim_next_run, execute_run
from vllmbench_orchestrator.settings import OrchestratorSettings
from vllmbench_protocol import PROTOCOL_VERSION, TRANSIENT_KINDS, __version__
from vllmbench_protocol.logging import bound, configure_logging

log = logging.getLogger("vllmbench.orchestrator")


class PollResult(NamedTuple):
    """What one poll iteration did, and what the loop should do next.

    ``host_unreachable`` is separate from ``worked`` because they call for opposite
    responses: work found means come straight back for more, a host that is not
    answering means stand off for a while.
    """

    worked: bool
    host_unreachable: bool


# How often to look for queued work when there is none. Short enough that a run
# triggered from the UI starts promptly, long enough not to hammer the database while
# idle overnight.
POLL_INTERVAL_SECONDS = 2.0

# How long to stand down after a run failed because its host could not be reached.
#
# Without this, a host that reboots mid-sweep loses the entire remaining queue in a few
# seconds: each run is claimed, fails its reconnect window, and the next is claimed
# immediately. Every one of those failures is real and correctly recorded — nothing is
# hidden — but they are all the same failure, and burning fifty points on one restart is
# not a defensible response to it.
#
# Deliberately not a retry, and deliberately not per-host. It only slows how fast work is
# pulled while the last thing we heard from a host was silence; a run that has already
# failed stays failed, and any queued run is still executed exactly once.
UNREACHABLE_BACKOFF_SECONDS = 30.0

# How often to apply the telemetry retention horizon, when one is set.
#
# Once a day, and from the idle path rather than a timer: the orchestrator's whole
# purpose is executing runs, and a deletion competing with a benchmark for the same
# database is exactly the kind of interference that shows up as an unexplained latency
# outlier in somebody's results.
RETENTION_INTERVAL_SECONDS = 86400.0


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


# How often the cancellation watcher re-reads the database while a run is in flight.
# The cost is one small query per interval against a run that may last an hour; the
# benefit is that cancelling a sweep takes effect in seconds rather than at the end of
# the current point.
CANCEL_POLL_SECONDS = 3.0


async def _watch_for_cancellation(
    factory: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    sweep_id: uuid.UUID | None,
    cancel: asyncio.Event,
) -> None:
    """Set ``cancel`` when this run, or the sweep it belongs to, is cancelled.

    Runs on its own session. Sharing the executing run's session would mean two tasks
    issuing statements on one connection, which SQLAlchemy's async session does not
    support and which fails in ways that look like unrelated corruption.
    """
    while not cancel.is_set():
        await asyncio.sleep(CANCEL_POLL_SECONDS)
        try:
            async with factory() as session:
                if sweep_id is not None:
                    sweep_status = await session.scalar(
                        select(Sweep.status).where(Sweep.id == sweep_id)
                    )
                    if sweep_status is SweepStatus.CANCELLED:
                        log.info("sweep %s cancelled; stopping run %s", sweep_id, run_id)
                        cancel.set()
                        return

                run_status = await session.scalar(select(Run.status).where(Run.id == run_id))
                if run_status is RunStatus.CANCELLED:
                    log.info("run %s cancelled directly", run_id)
                    cancel.set()
                    return
        except Exception:
            # Never let the watcher kill the run it is watching. A database blip should
            # cost a missed cancellation check, not a lost measurement.
            log.exception("cancellation watch failed for run %s", run_id)


async def _poll_once(factory: async_sessionmaker[AsyncSession], token: str) -> PollResult:
    """Claim and execute one run.

    One at a time and deliberately: a GPU host runs a single engine, so a second
    concurrent run would contend for the same VRAM and measure the contention.
    """
    async with factory() as session:
        run = await claim_next_run(session)
        if run is None:
            return PollResult(worked=False, host_unreachable=False)

        cancel = asyncio.Event()
        watcher = asyncio.create_task(
            _watch_for_cancellation(factory, run.id, run.sweep_id, cancel)
        )
        # Bound here rather than passed down. Everything logged for the rest of this
        # run — by the runner, by SQLAlchemy, by httpx — carries these, so "what did
        # this run do" is a filter rather than a grep across three services' formats.
        try:
            with bound(run_id=str(run.id), sweep_id=str(run.sweep_id) if run.sweep_id else None):
                await execute_run(session, run, token, cancel=cancel)
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        return PollResult(
            worked=True,
            host_unreachable=run.failure_kind in TRANSIENT_KINDS,
        )


async def _apply_retention(factory: async_sessionmaker[AsyncSession], *, days: int) -> None:
    """Run one retention pass. Never raises.

    A failure here must not stop the orchestrator executing runs — the queue is the job,
    and reclaiming disk is housekeeping. Loud in the log, invisible to the sweep.
    """
    try:
        async with factory() as session:
            result = await prune_telemetry(session, older_than_days=days)
        if result.runs:
            log.info(
                "retention: pruned %d run(s) older than %d days, ~%.1f MB",
                result.runs,
                days,
                result.bytes_reclaimed / 1e6,
            )
    except Exception:
        log.exception("retention pass failed; runs are unaffected")


async def main() -> None:
    settings = OrchestratorSettings()
    configure_logging(
        "orchestrator",
        secrets=[settings.token, database_password()],
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

        if not settings.token:
            log.warning(
                "VLLMBENCH_TOKEN is not set; runs will fail to authenticate against the agent"
            )

        if settings.telemetry_retention_days:
            log.info(
                "telemetry retention: deleting samples for runs finished more than %d days ago",
                settings.telemetry_retention_days,
            )
        else:
            log.info("telemetry retention is off; samples are kept indefinitely")

        log.info("polling for queued runs every %.1fs", POLL_INTERVAL_SECONDS)
        next_retention = 0.0
        while not stopping.is_set():
            try:
                result = await _poll_once(sessions, settings.token)
            except Exception:
                # A crash here would stop every future run, not just this one. Log and
                # continue; the run itself has already been marked failed with a reason.
                log.exception("poll iteration failed")
                result = PollResult(worked=False, host_unreachable=False)

            pause = 0.0
            if result.host_unreachable:
                pause = UNREACHABLE_BACKOFF_SECONDS
                log.warning(
                    "last run failed with an unreachable host; pausing %.0fs before "
                    "claiming more work",
                    pause,
                )
            elif not result.worked:
                pause = POLL_INTERVAL_SECONDS

            # Only while idle. A deletion competing with a benchmark for the same
            # database is the kind of interference that surfaces as an unexplained
            # latency outlier in somebody's results.
            now = asyncio.get_running_loop().time()
            if settings.telemetry_retention_days and not result.worked and now >= next_retention:
                await _apply_retention(sessions, days=settings.telemetry_retention_days)
                next_retention = now + RETENTION_INTERVAL_SECONDS

            if pause:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stopping.wait(), timeout=pause)
    finally:
        log.info("orchestrator stopping")
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
