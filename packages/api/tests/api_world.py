"""The builder for a populated database, shared by the tests that need one.

Its own module, uniquely named, rather than living in ``conftest.py``. Fixtures are found
by pytest per directory, but ``from conftest import World`` is a plain import of a
top-level module called ``conftest`` — and this repository has two, so it resolves to
whichever landed in ``sys.modules`` first. That is the same trap ``vllmbench_db.testing``
was created to avoid, and it fails as a confusing ImportError from an unrelated package.

Runs are inserted already terminal rather than transitioned into it: a run in
``succeeded`` is immutable by database trigger, so building one any other way would be
fighting the schema for no benefit.
"""

from __future__ import annotations

import datetime as dt
import os

from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_db.enums import (
    FailureKind,
    InitiatedBy,
    ReplicateOrder,
    RunStatus,
    SweepStatus,
)
from vllmbench_db.models import GpuHost, Run, RunSummary, ServerConfig, Sweep, Workload


class World:
    """Builder for the rows an analysis query reads.

    Runs are inserted already terminal rather than transitioned into it: a run in
    ``succeeded`` is immutable by database trigger, so building one any other way would
    be fighting the schema for no benefit here.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.workload: Workload | None = None

    async def host(self, name: str, *, synthetic: str | None = None, gpus: int = 2) -> GpuHost:
        host = GpuHost(
            name=name, agent_url="http://agent", gpu_count=gpus, synthetic_source=synthetic
        )
        self.session.add(host)
        await self.session.flush()
        return host

    async def config(self, name: str, yaml: str = "model: m\n") -> ServerConfig:
        config = ServerConfig(config_hash=os.urandom(32).hex(), name=name, yaml=yaml)
        self.session.add(config)
        await self.session.flush()
        return config

    async def a_workload(self, *, max_concurrency: int | None = 16) -> Workload:
        workload = Workload(
            workload_hash=os.urandom(32).hex(),
            name=f"c{max_concurrency}",
            dataset_name="random",
            num_prompts=64,
            max_concurrency=max_concurrency,
            input_len=512,
            output_len=128,
        )
        self.session.add(workload)
        await self.session.flush()
        return workload

    async def sweep(
        self,
        host: GpuHost,
        *,
        order: ReplicateOrder = ReplicateOrder.GROUPED,
        name: str = "s",
        synthetic: bool = False,
    ) -> Sweep:
        sweep = Sweep(
            name=name,
            is_synthetic=synthetic,
            gpu_host_id=host.id,
            status=SweepStatus.SUCCEEDED,
            replicates=3,
            replicate_order=order,
            initiated_by=InitiatedBy.UI,
        )
        self.session.add(sweep)
        await self.session.flush()
        return sweep

    async def run(
        self,
        host: GpuHost,
        config: ServerConfig,
        workload: Workload,
        *,
        sweep: Sweep | None = None,
        status: RunStatus = RunStatus.SUCCEEDED,
        # Settable because a terminal run cannot be aged afterwards — the immutability
        # trigger refuses even a raw UPDATE, which is the point of it. A test that needs
        # a run to look two hundred days old has to build it that way.
        finished_at: dt.datetime | None = None,
        failure_kind: FailureKind | str | None = None,
        summary: bool = True,
        tpot: float = 25.0,
        per_gpu: float = 1000.0,
        vllm_version: str = "0.25.1",
        gpu_model: str = "NVIDIA GeForce RTX 3090",
        driver_version: str = "580.95.05",
        tp: int = 2,
        replicate_idx: int = 0,
    ) -> Run:
        run = Run(
            sweep_id=sweep.id if sweep else None,
            replicate_idx=replicate_idx,
            server_config_id=config.id,
            workload_id=workload.id,
            gpu_host_id=host.id,
            status=status,
            # A failed run names its failure — the check constraint requires it, so a
            # builder that omitted it would produce rows the schema rejects. Tests that
            # care which kind pass their own.
            failure_kind=(
                failure_kind
                if failure_kind is not None
                else (FailureKind.INTERNAL if status is RunStatus.FAILED else None)
            ),
            finished_at=finished_at or dt.datetime.now(dt.UTC),
            config_hash=config.config_hash,
            workload_hash=workload.workload_hash,
            vllm_version=vllm_version,
            gpu_model=gpu_model,
            driver_version=driver_version,
            gpu_count=tp,
            tensor_parallel_size=tp,
            is_synthetic=host.synthetic_source is not None,
            synthetic_source=host.synthetic_source,
            initiated_by=InitiatedBy.UI,
        )
        self.session.add(run)
        await self.session.flush()
        if summary:
            self.session.add(
                RunSummary(
                    run_id=run.id,
                    tpot_ms_mean=tpot,
                    tpot_ms_p99=tpot * 2,
                    total_token_throughput_per_gpu=per_gpu,
                    output_token_throughput_per_gpu=per_gpu / 2,
                    total_token_throughput_tok_sec=per_gpu * tp,
                    ttft_ms_p99=120.0,
                )
            )
        await self.session.commit()
        return run
