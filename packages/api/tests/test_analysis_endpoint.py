"""The analysis query, end to end against a real database.

:mod:`test_analysis` covers the rules; this covers the wiring — that the joins produce
the records those rules expect, that the filters and the exclusion tally see the same
population, and above all that a synthetic run cannot reach a real chart through this
endpoint (invariant 7).
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.enums import InitiatedBy, ReplicateOrder, RunStatus, SweepStatus
from vllmbench_db.models import (
    GpuHost,
    Run,
    RunSummary,
    ServerConfig,
    Sweep,
    Workload,
)
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration

POINTS = "/api/analysis/points"


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_engine(test_database_url())
    factory = create_session_factory(engine)
    api_app.state.engine = engine
    api_app.state.sessions = factory
    api_app.state.settings = ApiSettings(token="test-token-not-a-real-secret")

    await reset_database(engine)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def client(session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as c:
        yield c


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

    async def a_workload(self, *, max_concurrency: int = 16) -> Workload:
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
        self, host: GpuHost, *, order: ReplicateOrder = ReplicateOrder.GROUPED
    ) -> Sweep:
        sweep = Sweep(
            name="s",
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
            finished_at=dt.datetime.now(dt.UTC),
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


@pytest.fixture
def world(session: AsyncSession) -> World:
    return World(session)


async def test_replicates_collapse_to_one_point_with_a_band(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("tp2")
    workload = await world.a_workload()
    sweep = await world.sweep(host)
    for i, throughput in enumerate((900.0, 1000.0, 1100.0)):
        await world.run(host, config, workload, sweep=sweep, per_gpu=throughput, replicate_idx=i)

    body = (await client.get(POINTS)).json()

    assert body["run_count"] == 3
    (group,) = body["groups"]
    (point,) = group["points"]
    assert point["replicates"] == 3
    spread = point["metrics"]["total_token_throughput_per_gpu"]
    assert (spread["median"], spread["min"], spread["max"]) == (1000.0, 900.0, 1100.0)
    assert spread["values"] == [900.0, 1000.0, 1100.0]
    # The band has to say what it measures, not merely exist.
    assert point["spread_basis"] == "grouped"
    assert "back-to-back" in point["spread_note"]


async def test_synthetic_runs_never_appear_in_a_real_query(
    client: httpx.AsyncClient, world: World
) -> None:
    """Invariant 7, at the layer that feeds every chart.

    The mock agent's runs are correctly flagged at creation; this is the other half —
    that a flagged run cannot be drawn beside a real one, no matter what the caller asks
    for. There is no request that returns both.
    """
    real_host = await world.host("ubuntu-llm")
    mock_host = await world.host("mock", synthetic="mock_agent")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(real_host, config, workload, per_gpu=1000.0)
    await world.run(mock_host, config, workload, per_gpu=99999.0)

    real = (await client.get(POINTS)).json()
    assert real["run_count"] == 1
    assert all(g["gpu_host_name"] == "ubuntu-llm" for g in real["groups"])
    # ...and the caller is told the other population exists rather than left guessing.
    assert real["excluded"]["other_source"] == 1
    assert real["excluded"]["other_source_name"] == "synthetic"

    synthetic = (await client.get(POINTS, params={"source": "synthetic"})).json()
    assert synthetic["run_count"] == 1
    assert synthetic["groups"][0]["gpu_host_name"] == "mock"

    # No third value exists, so "both" is not a request that can be made.
    assert (await client.get(POINTS, params={"source": "all"})).status_code == 422


async def test_two_vllm_versions_are_separate_groups(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(host, config, workload, vllm_version="0.25.1")
    await world.run(host, config, workload, vllm_version="0.26.0")

    body = (await client.get(POINTS)).json()

    assert len(body["groups"]) == 2
    assert {g["vllm_version"] for g in body["groups"]} == {"0.25.1", "0.26.0"}
    # Same config and workload, so without partitioning these would have merged into one
    # point and averaged two versions together.
    assert all(len(g["points"]) == 1 for g in body["groups"])


async def test_mixed_driver_versions_warn_within_one_group(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(host, config, workload, driver_version="580.95.05")
    await world.run(host, config, workload, driver_version="581.00.00")

    (group,) = (await client.get(POINTS)).json()["groups"]
    assert len(group["points"]) == 1
    assert any("driver version" in w for w in group["warnings"])


async def test_pareto_frontier_marks_the_configs_worth_considering(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    # Two trade-offs and one config that loses on both axes.
    dense = await world.config("dense")
    quick = await world.config("quick")
    loser = await world.config("loser")
    await world.run(host, dense, workload, per_gpu=2000.0, tpot=50.0)
    await world.run(host, quick, workload, per_gpu=800.0, tpot=10.0)
    await world.run(host, loser, workload, per_gpu=700.0, tpot=60.0)

    (group,) = (await client.get(POINTS)).json()["groups"]

    on_frontier = {p["config_name"] for p in group["points"] if p["on_pareto_frontier"]}
    assert on_frontier == {"dense", "quick"}
    assert len(group["pareto_point_ids"]) == 2


async def test_per_user_rate_is_derived_from_tpot(client: httpx.AsyncClient, world: World) -> None:
    host = await world.host("ubuntu-llm")
    await world.run(host, await world.config("c"), await world.a_workload(), tpot=25.0)

    (group,) = (await client.get(POINTS)).json()["groups"]
    (point,) = group["points"]
    assert point["metrics"]["per_user_output_tok_s"]["median"] == pytest.approx(40.0)
    assert point["metrics"]["per_user_output_tok_s_p99"]["median"] == pytest.approx(20.0)


async def test_excluded_counts_explain_the_gaps(client: httpx.AsyncClient, world: World) -> None:
    """A chart cannot show why a point is missing, so the counts travel beside it."""
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(host, config, workload)
    await world.run(host, config, workload, status=RunStatus.FAILED, summary=False)
    await world.run(host, config, workload, status=RunStatus.CANCELLED, summary=False)
    await world.run(host, config, workload, status=RunStatus.QUEUED, summary=False)
    await world.run(host, config, workload, summary=False)

    body = (await client.get(POINTS)).json()

    assert body["run_count"] == 1
    assert body["excluded"] == {
        "failed": 1,
        "cancelled": 1,
        "unfinished": 1,
        "succeeded_without_summary": 1,
        "other_source": 0,
        "other_source_name": "synthetic",
    }


async def test_filters_narrow_both_the_points_and_the_tally(
    client: httpx.AsyncClient, world: World
) -> None:
    """The exclusion counts must describe the filtered population, not the whole table.

    Counts from a wider scope would explain away points that were never asked for, which
    is more misleading than reporting nothing.
    """
    host = await world.host("ubuntu-llm")
    other = await world.host("second-host")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(host, config, workload)
    await world.run(other, config, workload, status=RunStatus.FAILED, summary=False)

    unfiltered = (await client.get(POINTS)).json()
    assert unfiltered["excluded"]["failed"] == 1

    filtered = (await client.get(POINTS, params={"host_id": str(host.id)})).json()
    assert filtered["run_count"] == 1
    assert filtered["excluded"]["failed"] == 0


async def test_tensor_parallel_sizes_share_a_group(client: httpx.AsyncClient, world: World) -> None:
    """The TP scaling view needs TP=1 and TP=2 on one chart.

    Safe only because the plotted figures are per-device; the aggregate throughput of the
    same runs is not comparable, and the metric registry says so.
    """
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    await world.run(host, await world.config("tp1"), workload, tp=1)
    await world.run(host, await world.config("tp2"), workload, tp=2)

    (group,) = (await client.get(POINTS)).json()["groups"]
    assert {p["tensor_parallel_size"] for p in group["points"]} == {1, 2}


async def test_ad_hoc_replicates_degrade_the_spread_claim(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    await world.run(host, config, workload, per_gpu=900.0)
    await world.run(host, config, workload, per_gpu=1100.0)

    (group,) = (await client.get(POINTS)).json()["groups"]
    (point,) = group["points"]
    assert point["spread_basis"] == "mixed"
    assert "drift" in point["spread_note"]


async def test_metric_registry_travels_with_the_data(
    client: httpx.AsyncClient, world: World
) -> None:
    body = (await client.get(POINTS)).json()
    by_key = {m["key"]: m for m in body["metrics"]}
    assert by_key["total_token_throughput_per_gpu"]["per_gpu"] is True
    assert by_key["ttft_ms_p99"]["better"] == "lower"
    assert body["pareto_x"] == "total_token_throughput_per_gpu"
    assert body["pareto_y"] == "per_user_output_tok_s"


async def test_an_unknown_axis_falls_back_rather_than_erroring(
    client: httpx.AsyncClient, world: World
) -> None:
    # The frontier is a default view, not a user-specified query; a stale bookmark should
    # show the standard chart rather than a 500.
    body = (await client.get(POINTS, params={"pareto_x": "not_a_metric"})).json()
    assert body["pareto_x"] == "total_token_throughput_per_gpu"


async def test_empty_database_is_an_empty_answer_not_an_error(
    client: httpx.AsyncClient, world: World
) -> None:
    body = (await client.get(POINTS)).json()
    assert body["groups"] == []
    assert body["run_count"] == 0
    assert body["truncated"] is False


async def test_truncation_is_reported(client: httpx.AsyncClient, world: World) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    for _ in range(3):
        await world.run(host, config, workload)

    body = (await client.get(POINTS, params={"limit": 2})).json()
    assert body["run_count"] == 2
    assert body["truncated"] is True, "a chart missing its tail must not look complete"


async def test_run_ids_link_back_to_the_runs_behind_a_point(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("c")
    workload = await world.a_workload()
    sweep = await world.sweep(host)
    runs = [await world.run(host, config, workload, sweep=sweep, replicate_idx=i) for i in range(2)]

    (group,) = (await client.get(POINTS)).json()["groups"]
    (point,) = group["points"]
    assert {uuid.UUID(r) for r in point["run_ids"]} == {r.id for r in runs}
    assert point["sweep_ids"] == [str(sweep.id)]


SCALING = "/api/analysis/scaling"

TP_CONFIG = "model: m\nmax-num-seqs: 8\ntensor-parallel-size: {tp}\n"


async def _tp_family(world: World, host: GpuHost, workload: Workload, *, at: dict[int, float]):
    """A config measured at several tensor-parallel widths, one config per width.

    Written the way a sweep writes it — a derived variant per width, differing only in
    the tensor-parallel line — because that is what the family grouping has to recognise.
    """
    for tp, per_gpu in at.items():
        config = await world.config(f"base TP{tp}", yaml=TP_CONFIG.format(tp=tp))
        await world.run(host, config, workload, tp=tp, per_gpu=per_gpu)


async def test_scaling_groups_configs_that_differ_only_in_width(
    client: httpx.AsyncClient, world: World
) -> None:
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    await _tp_family(world, host, workload, at={1: 1000.0, 2: 800.0})

    body = (await client.get(SCALING, params={"source": "real"})).json()

    (group,) = body["groups"]
    (curve,) = group["curves"]
    assert [s["tensor_parallel_size"] for s in curve["steps"]] == [1, 2]
    assert curve["baseline_tp"] == 1
    assert curve["baseline_is_single_gpu"] is True
    # 800 per GPU on two devices against 1000 on one: 80% efficient, 1.6x aggregate.
    assert [s["efficiency"] for s in curve["steps"]] == pytest.approx([1.0, 0.8])
    assert [s["speedup"] for s in curve["steps"]] == pytest.approx([1.0, 1.6])


async def test_unrelated_configs_are_not_a_scaling_curve(
    client: httpx.AsyncClient, world: World
) -> None:
    """The rule the whole view rests on.

    Two different configurations at two widths would otherwise be drawn as one config
    scaling badly, when nothing was scaled at all.
    """
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    a = await world.config("a", yaml="model: m\nmax-num-seqs: 8\ntensor-parallel-size: 1\n")
    b = await world.config("b", yaml="model: m\nmax-num-seqs: 64\ntensor-parallel-size: 2\n")
    await world.run(host, a, workload, tp=1, per_gpu=1000.0)
    await world.run(host, b, workload, tp=2, per_gpu=400.0)

    body = (await client.get(SCALING)).json()

    assert body["groups"] == []
    assert body["single_width_families"] == 2, "and the reader is told they were dropped"


async def test_a_family_is_recognised_across_separate_sweeps(
    client: httpx.AsyncClient, world: World
) -> None:
    """Derived from the config text, not from a lineage column.

    A TP=1 run authored by hand and a TP=2 variant generated by a sweep are the same
    engine configuration at two widths, and the curve has to join them.
    """
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    sweep = await world.sweep(host)
    one = await world.config("hand-written", yaml="model: m\nmax-num-seqs: 8\n")
    two = await world.config(
        "swept TP2", yaml="model: m\nmax-num-seqs: 8\ntensor-parallel-size: 2\n"
    )
    await world.run(host, one, workload, tp=1, per_gpu=1000.0)
    await world.run(host, two, workload, sweep=sweep, tp=2, per_gpu=900.0)

    (group,) = (await client.get(SCALING)).json()["groups"]
    (curve,) = group["curves"]
    # "no tensor-parallel line" and "tensor-parallel-size: 1" are the same engine config.
    assert len(curve["steps"]) == 2


async def test_workloads_get_their_own_curves(client: httpx.AsyncClient, world: World) -> None:
    host = await world.host("ubuntu-llm")
    light = await world.a_workload(max_concurrency=8)
    heavy = await world.a_workload(max_concurrency=64)
    await _tp_family(world, host, light, at={1: 1000.0, 2: 900.0})
    await _tp_family(world, host, heavy, at={1: 2000.0, 2: 1400.0})

    (group,) = (await client.get(SCALING)).json()["groups"]
    assert {c["workload_name"] for c in group["curves"]} == {"c8", "c64"}
    assert all(len(c["steps"]) == 2 for c in group["curves"])


async def test_an_aggregate_metric_is_refused_rather_than_mislabelled(
    client: httpx.AsyncClient, world: World
) -> None:
    """Efficiency divided from aggregates is speedup wearing the wrong name.

    A config that merely kept up would report 2.0 "efficiency" at twice the width, which
    reads as superlinear scaling. The endpoint falls back to the per-GPU default instead.
    """
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    await _tp_family(world, host, workload, at={1: 1000.0, 2: 800.0})

    body = (await client.get(SCALING, params={"metric": "total_token_throughput_tok_sec"})).json()
    assert body["metric"] == "total_token_throughput_per_gpu"


async def test_incomparable_runs_still_do_not_share_a_curve(
    client: httpx.AsyncClient, world: World
) -> None:
    # The same guard as the Pareto view: partitioning happens before any curve is built,
    # so a vLLM upgrade halfway through a TP sweep cannot look like a scaling result.
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    one = await world.config("base TP1", yaml=TP_CONFIG.format(tp=1))
    two = await world.config("base TP2", yaml=TP_CONFIG.format(tp=2))
    await world.run(host, one, workload, tp=1, per_gpu=1000.0, vllm_version="0.25.1")
    await world.run(host, two, workload, tp=2, per_gpu=900.0, vllm_version="0.26.0")

    body = (await client.get(SCALING)).json()
    assert body["groups"] == []


async def test_synthetic_scaling_is_a_separate_population(
    client: httpx.AsyncClient, world: World
) -> None:
    mock = await world.host("mock", synthetic="mock_agent")
    workload = await world.a_workload()
    await _tp_family(world, mock, workload, at={1: 9999.0, 2: 9999.0})

    assert (await client.get(SCALING)).json()["groups"] == []
    assert (await client.get(SCALING, params={"source": "synthetic"})).json()["groups"] != []


async def test_a_family_survives_the_tp_line_moving(
    client: httpx.AsyncClient, world: World
) -> None:
    """Two engines identical except for where the author put the tensor-parallel line.

    Deriving the family by rewriting the width to 1 would split these, because rewriting
    preserves the line's position and appending puts it last.
    """
    host = await world.host("ubuntu-llm")
    workload = await world.a_workload()
    top = await world.config(
        "tp first", yaml="tensor-parallel-size: 1\nmodel: m\nmax-num-seqs: 8\n"
    )
    bottom = await world.config(
        "tp last", yaml="model: m\nmax-num-seqs: 8\ntensor-parallel-size: 2\n"
    )
    await world.run(host, top, workload, tp=1, per_gpu=1000.0)
    await world.run(host, bottom, workload, tp=2, per_gpu=900.0)

    (group,) = (await client.get(SCALING)).json()["groups"]
    (curve,) = group["curves"]
    assert len(curve["steps"]) == 2
