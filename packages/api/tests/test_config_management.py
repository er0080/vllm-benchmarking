"""Lineage, annotation, and export.

Milestone 0.7.0's bar is one sentence: *the winning configuration from a sweep can be
exported and run in production unchanged*. "Unchanged" is the whole test — an export that
adds a provenance header would not change what vLLM does, but it would change the hash,
and then the file in production is no longer the file that was measured.

The other half is what makes a configuration defensible six months later: which config it
was edited from, and which measurement justifies keeping it. A YAML on its own is a list
of numbers with no argument attached.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.enums import InitiatedBy, RunStatus
from vllmbench_db.models import GpuHost, Run, ServerConfig, Workload
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url

pytestmark = pytest.mark.integration

BASE = "model: facebook/opt-125m\ntensor-parallel-size: 1\n"


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


async def create(client: httpx.AsyncClient, name: str, yaml: str, **extra: object) -> dict:
    response = await client.post("/api/configs", json={"name": name, "yaml": yaml, **extra})
    assert response.status_code == 201, response.text
    return response.json()


class TestExport:
    async def test_the_exported_bytes_are_the_stored_bytes(self, client: httpx.AsyncClient) -> None:
        """0.7.0's acceptance criterion, stated as an assertion.

        No header, no comment, no reordering. Anything added would change the hash, and
        then the file running in production is not the file that was measured.
        """
        config = await create(client, "winner", BASE)

        response = await client.get(f"/api/configs/{config['config_hash']}/export")

        assert response.status_code == 200
        assert response.text == config["yaml"]
        # And re-importing what came out yields the same configuration, not a new one.
        again = await client.post(
            "/api/configs", json={"name": "round-trip", "yaml": response.text}
        )
        assert again.json()["config_hash"] == config["config_hash"]

    async def test_it_downloads_under_a_name_that_identifies_it(
        self, client: httpx.AsyncClient
    ) -> None:
        """A directory of `config.yaml` files tells nobody anything."""
        config = await create(client, "Qwen 9B / TP2 baseline", BASE)

        response = await client.get(f"/api/configs/{config['config_hash']}/export")

        disposition = response.headers["content-disposition"]
        assert "qwen-9b-tp2-baseline" in disposition
        assert config["config_hash"][:12] in disposition

    async def test_an_unknown_hash_is_a_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/configs/" + "0" * 64 + "/export")).status_code == 404


class TestLineage:
    async def test_a_chain_of_edits_is_walkable(self, client: httpx.AsyncClient) -> None:
        first = await create(client, "baseline", BASE)
        second = await create(
            client, "more cache", BASE + "gpu-memory-utilization: 0.95\n", parent_id=first["id"]
        )
        third = await create(
            client,
            "and chunked prefill",
            BASE + "gpu-memory-utilization: 0.95\nenable-chunked-prefill: true\n",
            parent_id=second["id"],
        )

        lineage = (await client.get(f"/api/configs/{third['config_hash']}/lineage")).json()

        # Nearest parent first, back to the original.
        assert [a["name"] for a in lineage["ancestors"]] == ["more cache", "baseline"]
        assert lineage["children"] == []
        assert lineage["truncated"] is False

    async def test_children_are_listed(self, client: httpx.AsyncClient) -> None:
        """What did I try next — the question a reader actually has."""
        base = await create(client, "baseline", BASE)
        await create(
            client, "branch a", BASE + "gpu-memory-utilization: 0.95\n", parent_id=base["id"]
        )
        await create(client, "branch b", BASE + "max-num-seqs: 512\n", parent_id=base["id"])

        lineage = (await client.get(f"/api/configs/{base['config_hash']}/lineage")).json()

        assert sorted(c["name"] for c in lineage["children"]) == ["branch a", "branch b"]
        assert lineage["ancestors"] == []

    async def test_resubmitting_existing_text_does_not_rewrite_its_history(
        self, client: httpx.AsyncClient
    ) -> None:
        """The row's history already happened.

        Content addressing means the same YAML is the same config. Letting a later
        submission reparent it would make the derivation chain depend on who submitted
        last rather than on what was edited.
        """
        original = await create(client, "original", BASE)
        other = await create(client, "unrelated", BASE + "max-num-seqs: 8\n")

        response = await client.post(
            "/api/configs",
            json={"name": "re-submitted", "yaml": BASE, "parent_id": other["id"]},
        )

        assert response.json()["id"] == original["id"]
        assert response.json()["parent_id"] is None

    async def test_an_unknown_parent_is_refused(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/configs",
            json={"name": "orphan", "yaml": BASE, "parent_id": str(uuid.uuid4())},
        )
        assert response.status_code == 400


class TestAnnotation:
    async def _a_run_of(self, session: AsyncSession, config_hash: str) -> Run:
        host = GpuHost(name=f"h-{os.urandom(3).hex()}", agent_url="http://a", gpu_count=1)
        workload = Workload(
            workload_hash=os.urandom(32).hex(), name="w", dataset_name="random", num_prompts=8
        )
        config = await session.scalar(
            select(ServerConfig).where(ServerConfig.config_hash == config_hash)
        )
        assert config is not None
        session.add_all([host, workload])
        await session.flush()
        run = Run(
            server_config_id=config.id,
            workload_id=workload.id,
            gpu_host_id=host.id,
            status=RunStatus.SUCCEEDED,
            finished_at=dt.datetime.now(dt.UTC),
            config_hash=config_hash,
            workload_hash=workload.workload_hash,
            gpu_count=1,
            tensor_parallel_size=1,
            initiated_by=InitiatedBy.UI,
        )
        session.add(run)
        await session.commit()
        return run

    async def test_a_config_can_point_at_the_run_that_justifies_it(
        self, client: httpx.AsyncClient, session: AsyncSession
    ) -> None:
        config = await create(client, "the winner", BASE)
        run = await self._a_run_of(session, config["config_hash"])

        response = await client.patch(
            f"/api/configs/{config['config_hash']}",
            json={
                "justified_by_run_id": str(run.id),
                "notes": "best per-GPU at c16",
                "justification_note": "beat TP2 on both axes",
            },
        )

        assert response.status_code == 200
        assert response.json()["justified_by_run_id"] == str(run.id)
        assert response.json()["notes"] == "best per-GPU at c16"
        assert response.json()["justification_note"] == "beat TP2 on both axes"

        # And it survives into the list, which is where somebody would actually see it.
        listed = {c["config_hash"]: c for c in (await client.get("/api/configs")).json()}
        assert listed[config["config_hash"]]["justified_by_run_id"] == str(run.id)

    async def test_a_run_of_a_different_config_cannot_justify_this_one(
        self, client: httpx.AsyncClient, session: AsyncSession
    ) -> None:
        """The entire value of the link is that it points at evidence.

        A run of some other configuration is not evidence for this one, and a link that
        can be wrong is worse than no link.
        """
        config = await create(client, "candidate", BASE)
        other = await create(client, "other", BASE + "max-num-seqs: 8\n")
        run = await self._a_run_of(session, other["config_hash"])

        response = await client.patch(
            f"/api/configs/{config['config_hash']}", json={"justified_by_run_id": str(run.id)}
        )

        assert response.status_code == 400
        assert "cannot justify" in response.text

    async def test_the_justification_can_be_withdrawn(
        self, client: httpx.AsyncClient, session: AsyncSession
    ) -> None:
        """Omitting the field leaves it alone; clearing it takes a deliberate flag."""
        config = await create(client, "the winner", BASE)
        run = await self._a_run_of(session, config["config_hash"])
        await client.patch(
            f"/api/configs/{config['config_hash']}", json={"justified_by_run_id": str(run.id)}
        )

        untouched = await client.patch(
            f"/api/configs/{config['config_hash']}", json={"notes": "still thinking"}
        )
        assert untouched.json()["justified_by_run_id"] == str(run.id)

        cleared = await client.patch(
            f"/api/configs/{config['config_hash']}", json={"clear_justification": True}
        )
        assert cleared.json()["justified_by_run_id"] is None

    async def test_the_yaml_cannot_be_edited(self, client: httpx.AsyncClient) -> None:
        """There is no endpoint where it can.

        Editing the text is different bytes, a different hash, a different configuration —
        so an edit is a creation. A config whose text could change would silently
        invalidate every run that claims it.
        """
        config = await create(client, "fixed", BASE)

        response = await client.patch(
            f"/api/configs/{config['config_hash']}",
            json={"name": "renamed", "yaml": "model: something-else\n"},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "renamed"
        assert response.json()["yaml"] == config["yaml"]
        assert response.json()["config_hash"] == config["config_hash"]
