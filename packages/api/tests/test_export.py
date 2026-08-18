"""Exporting a result set, and what a row has to carry to survive leaving.

A file is where a result goes to be read by someone who cannot see the filters that
produced it. It gets forwarded, pasted into a spreadsheet beside somebody else's numbers,
and charted by a tool that has never heard of vLLM. Invariants 6 and 7 are promises about
what a number may be compared with, and neither survives the export unless every row says
so itself.
"""

from __future__ import annotations

import csv
import io
import json

import httpx
import pytest
from api_world import World

pytestmark = pytest.mark.integration

EXPORT = "/api/analysis/export"


def rows_of(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


async def a_populated_host(world: World) -> None:
    host = await world.host("ubuntu-llm")
    config = await world.config("tp1")
    workload = await world.a_workload(max_concurrency=16)
    sweep = await world.sweep(host)
    for i, throughput in enumerate((900.0, 1000.0)):
        await world.run(host, config, workload, sweep=sweep, per_gpu=throughput, replicate_idx=i)


class TestProvenanceTravelsWithTheRow:
    async def test_every_row_states_what_produced_it(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """Invariant 6 does not stop at the database.

        The redundancy down these columns is the point: it is what makes a row that has
        been sorted, filtered or pasted somewhere else still self-describing.
        """
        await a_populated_host(world)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv")).text)

        assert row["gpu_host"] == "ubuntu-llm"
        assert row["gpu_model"] == "NVIDIA GeForce RTX 3090"
        assert row["vllm_version"] == "0.25.1"
        assert row["bench_client_location"] == "loopback"
        assert row["tensor_parallel_size"] == "2"

    async def test_the_population_is_a_column_not_just_a_filter(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """A synthetic row loose in a spreadsheet has no flag to consult.

        Invariant 7 is a promise about where these numbers may appear, and a file is
        exactly where it would otherwise be lost.
        """
        host = await world.host("mock", synthetic="mock_agent")
        config = await world.config("c")
        workload = await world.a_workload()
        await world.run(host, config, workload)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv&source=synthetic")).text)

        assert row["source"] == "synthetic"

    async def test_the_filename_says_which_population(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """The filename is the part that survives being forwarded."""
        await a_populated_host(world)

        real = await client.get(f"{EXPORT}?format=csv")
        synthetic = await client.get(f"{EXPORT}?format=csv&source=synthetic")

        assert "vllmbench-analysis-real.csv" in real.headers["content-disposition"]
        assert "vllmbench-analysis-synthetic.csv" in synthetic.headers["content-disposition"]

    async def test_the_comparability_group_rides_along(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """CSV cannot refuse an invalid overlay the way a chart can.

        The only honest thing a flat file can do is say which rows belong together, so
        the group label is on every row rather than implied by ordering.
        """
        await a_populated_host(world)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv")).text)

        assert "ubuntu-llm" in row["comparability_group"]
        assert "0.25.1" in row["comparability_group"]


class TestNumbersCarryTheirSpread:
    async def test_a_median_never_travels_alone(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """A difference smaller than a point's own spread is not a result.

        A reader holding only the median has no way to know that, so the observed range
        goes in adjacent columns rather than a footnote a spreadsheet will drop.
        """
        await a_populated_host(world)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv")).text)

        assert float(row["total_token_throughput_per_gpu"]) == 950.0
        assert float(row["total_token_throughput_per_gpu__min"]) == 900.0
        assert float(row["total_token_throughput_per_gpu__max"]) == 1000.0
        assert row["replicates"] == "2"
        assert row["spread_basis"] == "grouped"

    async def test_an_unbounded_workload_is_blank_not_zero(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """Null max_concurrency means no limit, which is the opposite of zero."""
        host = await world.host("h")
        config = await world.config("c")
        workload = await world.a_workload(max_concurrency=None)
        await world.run(host, config, workload)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv")).text)

        assert row["max_concurrency"] == ""

    async def test_run_ids_make_a_figure_checkable(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """The way back to the raw records is what makes an exported number checkable
        rather than merely quotable."""
        await a_populated_host(world)

        (row,) = rows_of((await client.get(f"{EXPORT}?format=csv")).text)

        assert len(row["run_ids"].split()) == 2


class TestTheExportMatchesTheScreen:
    async def test_it_honours_the_same_filters_as_the_charts(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """An export that disagrees with the chart it was taken from is worse than none."""
        await a_populated_host(world)
        other = await world.host("other-host")
        await world.run(other, await world.config("z"), await world.a_workload(max_concurrency=8))

        everything = rows_of((await client.get(f"{EXPORT}?format=csv")).text)
        narrowed = rows_of((await client.get(f"{EXPORT}?format=csv&host_id={other.id}")).text)

        assert len(everything) == 2
        assert [r["gpu_host"] for r in narrowed] == ["other-host"]

    async def test_json_keeps_the_grouping_csv_has_to_flatten(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        await a_populated_host(world)

        body = json.loads((await client.get(f"{EXPORT}?format=json")).text)

        assert body["source"] == "real"
        (group,) = body["groups"]
        assert group["points"][0]["replicates"] == 2

    async def test_an_unknown_format_is_refused(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        assert (await client.get(f"{EXPORT}?format=xlsx")).status_code == 422


class TestSweepReport:
    async def test_it_is_the_same_report_the_agent_surface_serves(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """One implementation, two interfaces — so the two cannot come to disagree about
        what a sweep measured."""
        host = await world.host("ubuntu-llm")
        config = await world.config("tp1")
        workload = await world.a_workload()
        sweep = await world.sweep(host)
        await world.run(host, config, workload, sweep=sweep)

        response = await client.get(f"/api/sweeps/{sweep.id}/report")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "ubuntu-llm" in response.text
        assert "not a result" in response.text

    async def test_a_synthetic_sweeps_report_leads_with_the_quarantine(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        """A shared file is exactly where a synthetic number would otherwise pass as a
        measurement."""
        host = await world.host("mock", synthetic="mock_agent")
        sweep = await world.sweep(host, synthetic=True)
        await world.run(host, await world.config("c"), await world.a_workload(), sweep=sweep)

        text = (await client.get(f"/api/sweeps/{sweep.id}/report")).text

        assert "not measurements of any real hardware" in text
        assert text.index("not measurements of any real hardware") < text.index("tok/s per GPU")

    async def test_download_names_the_file_after_the_sweep(
        self, client: httpx.AsyncClient, world: World
    ) -> None:
        host = await world.host("h")
        sweep = await world.sweep(host, name="TP x concurrency")
        await world.run(host, await world.config("c"), await world.a_workload(), sweep=sweep)

        disposition = (await client.get(f"/api/sweeps/{sweep.id}/report?download=true")).headers[
            "content-disposition"
        ]

        assert "tp-x-concurrency" in disposition
