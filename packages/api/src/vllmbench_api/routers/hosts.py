"""GPU host registry.

Registering a host performs a real handshake rather than just storing a URL. A row that
says "host registered" while the agent is unreachable, mis-tokened, or speaking a
different protocol is worse than an error — it defers the failure to the moment a sweep
starts, which is the most expensive point to discover it.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vllmbench_api.deps import SessionDep, SettingsDep
from vllmbench_api.reference import reference_vllm_version
from vllmbench_api.schemas import HostCreate, HostFacts, HostOut
from vllmbench_db.models import GpuDevice, GpuHost, Run
from vllmbench_protocol import (
    AgentAuthError,
    AgentClient,
    AgentUnreachable,
    ProtocolMismatch,
)
from vllmbench_protocol.wire import HostInfo

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hosts", tags=["hosts"])


async def _handshake(agent_url: str, token: str) -> HostInfo:
    """Contact the agent, or translate the failure into something actionable.

    Each failure gets a distinct status because the operator's next step differs:
    unreachable means check the network, 502-with-auth means check the token, and a
    protocol mismatch means upgrade one side.
    """
    try:
        async with AgentClient(agent_url, token) as client:
            return await client.host_info()
    except ProtocolMismatch as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except AgentAuthError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except AgentUnreachable as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


async def _apply_facts(session: AsyncSession, host: GpuHost, info: HostInfo) -> None:
    host.agent_version = info.agent_version
    host.protocol_version = info.protocol_version
    host.vllm_version = info.vllm_version
    host.driver_version = info.driver_version
    host.cuda_version = info.cuda_version
    host.gpu_count = info.gpu_count
    # Trusted from the producer, never inferred (invariant 7).
    host.synthetic_source = info.synthetic_source
    host.last_seen_at = dt.datetime.now(dt.UTC)

    # Replace the device inventory rather than merging: a host can have cards added,
    # removed, or reordered, and a stale row would silently misattribute telemetry.
    #
    # The flush is load-bearing. Without it the new INSERTs are emitted in the same
    # unit of work as the cascade DELETEs and race them, violating the
    # (gpu_host_id, device_index) unique constraint on every refresh of an existing
    # host — which is to say, always, in production.
    if host.devices:
        host.devices.clear()
        await session.flush()

    for gpu in info.gpus:
        host.devices.append(
            GpuDevice(
                device_index=gpu.index,
                name=gpu.name,
                uuid_str=gpu.uuid,
                vram_bytes=gpu.vram_bytes,
            )
        )


def _to_facts(host: GpuHost) -> HostFacts:
    reference = reference_vllm_version()
    matches: bool | None = None
    if reference is not None and host.vllm_version is not None:
        matches = host.vllm_version == reference

    return HostFacts(
        **HostOut.model_validate(host).model_dump(),
        reference_vllm_version=reference,
        vllm_version_matches_reference=matches,
    )


@router.get("", response_model=list[HostFacts])
async def list_hosts(session: SessionDep) -> list[HostFacts]:
    result = await session.execute(
        select(GpuHost).options(selectinload(GpuHost.devices)).order_by(GpuHost.created_at)
    )
    return [_to_facts(host) for host in result.scalars()]


@router.post("", response_model=HostFacts, status_code=status.HTTP_201_CREATED)
async def register_host(
    payload: HostCreate,
    session: SessionDep,
    settings: SettingsDep,
) -> HostFacts:
    info = await _handshake(payload.agent_url, settings.token)

    host = GpuHost(name=payload.name, agent_url=payload.agent_url)
    session.add(host)
    await _apply_facts(session, host, info)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"a host named {payload.name!r} already exists",
        ) from exc

    await session.refresh(host, attribute_names=["devices"])
    log.info("registered host %s (%d GPUs)", host.name, host.gpu_count)
    return _to_facts(host)


@router.post("/{host_id}/refresh", response_model=HostFacts)
async def refresh_host(
    host_id: uuid.UUID,
    session: SessionDep,
    settings: SettingsDep,
) -> HostFacts:
    host = await session.get(GpuHost, host_id, options=[selectinload(GpuHost.devices)])
    if host is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="host not found")

    info = await _handshake(host.agent_url, settings.token)
    await _apply_facts(session, host, info)
    await session.commit()
    await session.refresh(host, attribute_names=["devices"])
    return _to_facts(host)


@router.delete("/{host_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_host(host_id: uuid.UUID, session: SessionDep) -> None:
    host = await session.get(GpuHost, host_id)
    if host is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="host not found")

    # Runs are measurements. Deleting a host must never delete them, and it must not
    # orphan their provenance either — a run has to keep being able to say what produced
    # it (invariant 6). So this refuses rather than cascading.
    run_count = await session.scalar(
        select(func.count()).select_from(Run).where(Run.gpu_host_id == host_id)
    )
    if run_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"host has {run_count} recorded run(s) and cannot be deleted. "
                "Runs are measurements; removing the host would strip their provenance."
            ),
        )

    await session.delete(host)
    await session.commit()


@router.get("/{host_id}", response_model=HostFacts)
async def get_host(host_id: uuid.UUID, session: SessionDep) -> HostFacts:
    host = await session.get(GpuHost, host_id, options=[selectinload(GpuHost.devices)])
    if host is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="host not found")
    return _to_facts(host)
