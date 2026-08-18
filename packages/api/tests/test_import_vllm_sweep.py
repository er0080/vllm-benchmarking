"""Importing a `vllm bench sweep serve` output directory.

Against a **captured** directory — `fixtures/vllm_sweep_v0.25.1/` was produced by running
the real tool, not written by hand. A hand-made fixture would encode this author's belief
about the layout, which is the belief under test.

The theme running through these tests is ADR 0003: the upstream output carries no vLLM
version, GPU model, driver, host or device count. Not one field of provenance. So the
question every test here asks is what happens to the facts the files cannot supply.
"""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
from api_world import World
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.importers import (
    SweepDirectoryError,
    build_points,
    parse_directory_name,
    parse_params,
    reconstructed_yaml,
)
from vllmbench_db.models import Run

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "vllm_sweep_v0.25.1"
IMPORT = "/api/import/vllm-sweep"


def captured_files() -> dict[str, dict]:
    return {
        str(path.relative_to(FIXTURE)): json.loads(path.read_text())
        for path in FIXTURE.rglob("*.json")
    }


# ---------------------------------------------------------------------------
# Parsing — no database, no services
# ---------------------------------------------------------------------------


class TestDirectoryNames:
    def test_the_captured_layout_parses(self) -> None:
        serve, bench = parse_directory_name(
            "SERVE--max_num_seqs=4-BENCH--max_concurrency=2-num_prompts=8"
        )
        assert serve == {"max_num_seqs": 4}
        assert bench == {"max_concurrency": 2, "num_prompts": 8}

    def test_values_come_back_as_the_type_they_went_in_as(self) -> None:
        """The directory name is the only record of these, and it is a string.

        `max_num_seqs=4` returning "4" would put a string in the reconstructed config and
        claim the engine was handed one.
        """
        assert parse_params("a=4-b=0.9-c=true-d=none-e=text") == {
            "a": 4,
            "b": 0.9,
            "c": True,
            "d": None,
            "e": "text",
        }

    def test_a_value_containing_a_dash_survives(self) -> None:
        """Model ids and paths contain dashes, and so does the separator.

        Splitting naively would turn one parameter into two and silently make two
        different sweep points look like the same one.
        """
        assert parse_params("model=facebook/opt-125m-max_num_seqs=4") == {
            "model": "facebook/opt-125m",
            "max_num_seqs": 4,
        }

    def test_a_name_that_is_not_a_sweep_point_is_refused(self) -> None:
        with pytest.raises(SweepDirectoryError, match="not a sweep point directory"):
            parse_directory_name("results")
        with pytest.raises(SweepDirectoryError, match=r"no .*BENCH"):
            parse_directory_name("SERVE--max_num_seqs=4")


class TestReadingTheCapturedDirectory:
    def test_it_finds_both_points_and_their_replicates(self) -> None:
        points = build_points(captured_files())

        assert [p.replicates for p in points] == [2, 2]
        assert [p.serve_params for p in points] == [{"max_num_seqs": 4}, {"max_num_seqs": 8}]

    def test_summary_json_is_skipped(self) -> None:
        """It is exactly the list of the run files — same keys, no aggregation.

        Reading it too would double every replicate. The run files are preferred because
        they carry `run_number`, which is what makes replicates distinguishable.
        """
        points = build_points(captured_files())

        assert sum(p.replicates for p in points) == 4  # not 8

    def test_pointing_at_the_wrong_directory_says_so(self) -> None:
        with pytest.raises(SweepDirectoryError, match="experiment directory"):
            build_points({"summary.json": {}})

    def test_the_reconstructed_config_admits_what_it_is(self) -> None:
        """Invariant 5 is about what gets passed to `vllm serve --config`, and this was
        not. Anything fixed in --serve-cmd was never recorded by the tool."""
        yaml = reconstructed_yaml({"max_num_seqs": 4}, "facebook/opt-125m")

        assert "Not a runnable configuration" in yaml
        assert "model: facebook/opt-125m" in yaml
        assert "max-num-seqs: 4" in yaml


# ---------------------------------------------------------------------------
# Importing — end to end
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration


def declared(host_id: str, **overrides: object) -> dict[str, object]:
    return {
        "gpu_host_id": host_id,
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "vllm_version": "0.25.1",
        "gpu_count": 1,
        "tensor_parallel_size": 1,
        "bench_client_location": "loopback",
        **overrides,
    }


@pytest.mark.integration
class TestImport:
    async def test_a_captured_sweep_round_trips_into_charts(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        host = await world.host("imported-host")
        await world.session.commit()

        response = await client.post(
            IMPORT,
            json={
                "experiment_name": "demo",
                "declared": declared(str(host.id)),
                "files": captured_files(),
            },
        )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["points_imported"] == 2
        assert body["runs_imported"] == 4

        # And the imported runs reach the analysis surface the charts read.
        analysis = (await client.get("/api/analysis/points")).json()
        assert analysis["run_count"] == 4
        (group,) = analysis["groups"]
        assert len(group["points"]) == 2
        # Two replicates per point, so a spread exists — which is only true because the
        # per-run files were read rather than summary.json.
        assert all(p["replicates"] == 2 for p in group["points"])

    async def test_provenance_the_files_cannot_supply_is_required(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """Refused, not defaulted.

        A default here is a fabricated provenance column. `gpu_count` is the sharpest
        case: per-GPU throughput is the axis every comparison view defaults to, and it
        cannot be derived without one.
        """
        host = await world.host("h")
        await world.session.commit()
        fields = declared(str(host.id))
        del fields["gpu_count"]

        response = await client.post(
            IMPORT,
            json={"experiment_name": "demo", "declared": fields, "files": captured_files()},
        )

        assert response.status_code == 422
        assert "gpu_count" in response.text

    async def test_every_imported_run_is_marked_forever(
        self, client: httpx.AsyncClient, world: World, session: AsyncSession
    ) -> None:
        """A GPU model NVML reported and one a person typed are different kinds of fact."""
        host = await world.host("h")
        await world.session.commit()
        await client.post(
            IMPORT,
            json={
                "experiment_name": "demo",
                "declared": declared(str(host.id)),
                "files": captured_files(),
            },
        )

        runs = list((await session.execute(select(Run))).scalars())

        assert len(runs) == 4
        assert all(r.imported_from == "vllm bench sweep serve · demo" for r in runs)
        # And a run this framework measured stays null, so the two are distinguishable.
        measured = await world.run(host, await world.config("c"), await world.a_workload())
        assert measured.imported_from is None

    async def test_the_declared_values_land_on_the_runs(
        self, client: httpx.AsyncClient, world: World, session: AsyncSession
    ) -> None:
        host = await world.host("h")
        await world.session.commit()
        await client.post(
            IMPORT,
            json={
                "experiment_name": "demo",
                "declared": declared(str(host.id), gpu_count=2, tensor_parallel_size=2),
                "files": captured_files(),
            },
        )

        run = (await session.execute(select(Run))).scalars().first()

        assert run is not None
        assert run.gpu_count == 2
        assert run.tensor_parallel_size == 2
        assert run.vllm_version == "0.25.1"

    async def test_per_gpu_throughput_uses_the_declared_device_count(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """Invariant 8, and the reason gpu_count is mandatory.

        The same files declared at one device and at two must not produce the same
        per-GPU figure — that number is aggregate divided by devices, and getting it
        wrong is the silent halving this project has already been bitten by once.
        """
        one = await world.host("one-gpu")
        await world.session.commit()
        two = await world.host("two-gpu")
        await world.session.commit()
        files = captured_files()

        await client.post(
            IMPORT,
            json={
                "experiment_name": "a",
                "declared": declared(str(one.id), gpu_count=1),
                "files": files,
            },
        )
        await client.post(
            IMPORT,
            json={
                "experiment_name": "b",
                "declared": declared(str(two.id), gpu_count=2, tensor_parallel_size=2),
                "files": files,
            },
        )

        groups = (await client.get("/api/analysis/points")).json()["groups"]
        by_host = {g["gpu_host_name"]: g for g in groups}
        single = by_host["one-gpu"]["points"][0]["metrics"]["total_token_throughput_per_gpu"]
        double = by_host["two-gpu"]["points"][0]["metrics"]["total_token_throughput_per_gpu"]

        assert single["median"] == pytest.approx(double["median"] * 2)

    async def test_a_payload_that_is_not_a_bench_result_refuses_the_whole_import(
        self, client: httpx.AsyncClient, world: World, session: AsyncSession
    ) -> None:
        """A half-imported sweep is worse than none, and a summary row full of NULLs is
        indistinguishable from a benchmark that legitimately measured nothing."""
        host = await world.host("h")
        await world.session.commit()
        files = captured_files()
        key = next(k for k in files if k.endswith("run=1.json"))
        files[key] = {"not": "a bench result"}

        response = await client.post(
            IMPORT,
            json={"experiment_name": "demo", "declared": declared(str(host.id)), "files": files},
        )

        assert response.status_code == 400
        assert "did not match the expected" in response.text
        assert list((await session.execute(select(Run))).scalars()) == []

    async def test_an_unknown_host_is_refused(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        import uuid as _uuid

        response = await client.post(
            IMPORT,
            json={
                "experiment_name": "demo",
                "declared": declared(str(_uuid.uuid4())),
                "files": captured_files(),
            },
        )
        assert response.status_code == 404

    async def test_a_group_containing_imported_runs_says_so(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """ "Group or warn, never silently overlay" — applied to a difference no chart
        can see.

        These runs sit together because their *declared* provenance matched, not because
        anything observed it. That is a weaker claim than the rest of the group rests on
        (ADR 0003), and a reader comparing them has to be told.
        """
        host = await world.host("mixed-host")
        await world.session.commit()
        await client.post(
            IMPORT,
            json={
                "experiment_name": "demo",
                "declared": declared(str(host.id)),
                "files": captured_files(),
            },
        )
        # A run this framework actually measured, on the same host and version.
        await world.run(host, await world.config("measured"), await world.a_workload(), tp=1)

        (group,) = (await client.get("/api/analysis/points")).json()["groups"]

        assert any("declared by a person, not observed" in w for w in group["warnings"])
        assert any("measured run" in w for w in group["warnings"])
