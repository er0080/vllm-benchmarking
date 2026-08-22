"""The published MCP contract: what ``tools/list`` says this server is.

Separate from ``test_mcp_server.py``, which exercises the mounted transport and needs a
database. Nothing here calls a tool, so it runs in tier 1 — which is the point. The
failure this guards against is a tool added without a description or without hints, and
that should be caught on the push that adds it rather than on the PR.

**For an MCP server the schema is the documentation.** No agent reads a guide before
calling something; whatever ``tools/list`` returns is the entire contract. A missing
parameter description is therefore not an untidiness, it is a gap in the interface, and
the only way it stays fixed is if something asserts it.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, cast

import pytest
from mcp.types import Tool, ToolAnnotations

from vllmbench_api.analysis import METRICS_BY_KEY
from vllmbench_api.mcp_server import METRIC_KEYS, _metric_key, build_mcp_server
from vllmbench_api.settings import ApiSettings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GUIDE = REPO_ROOT / "docs" / "mcp.md"

#: Named rather than derived, so that adding a write tool and forgetting to annotate it
#: fails here. A test that asked the server which tools are writes would agree with
#: whatever the server said, including when the server is wrong.
WRITE_TOOLS = frozenset(
    {"create_config", "annotate_config", "create_workload", "create_sweep", "cancel_sweep"}
)

#: Taking work away that cannot be given back. Everything else either adds a row or
#: returns one.
DESTRUCTIVE_TOOLS = frozenset({"cancel_sweep"})


@pytest.fixture(scope="module")
def tools() -> dict[str, Tool]:
    """The surface as a client receives it.

    No sessionmaker: listing tools never opens one, and a fixture that needed a database
    would push this suite into tier 2 and out of the push that breaks it.
    """
    server = build_mcp_server(cast(Any, None), ApiSettings(mcp_token="not-a-real-secret"))

    import anyio

    return {tool.name: tool for tool in anyio.run(server.list_tools)}


def _properties(tool: Tool) -> dict[str, Any]:
    return (tool.input_schema or {}).get("properties") or {}


def _hints(tool: Tool) -> ToolAnnotations:
    """The tool's annotations, insisted upon.

    ``annotations`` is optional in the protocol, and a tool without them is the exact
    regression this file exists to catch — so an absent block fails loudly here rather than
    quietly making every hint assertion below unfalsifiable.
    """
    assert tool.annotations is not None, f"{tool.name} publishes no annotations"
    return tool.annotations


class TestEveryToolDescribesItself:
    def test_the_expected_tools_are_published(self, tools: dict[str, Tool]) -> None:
        assert WRITE_TOOLS <= set(tools), "a write tool disappeared from the surface"

    def test_every_tool_has_a_description(self, tools: dict[str, Tool]) -> None:
        assert [name for name, tool in tools.items() if not (tool.description or "").strip()] == []

    def test_every_parameter_has_a_description(self, tools: dict[str, Tool]) -> None:
        """The gap that made this suite exist: 45 parameters, none documented.

        Names carry the easy ones — ``run_id`` needs no gloss. They do not carry
        ``tensor_parallel_is_swept``, and an agent cannot tell the two cases apart before
        guessing.
        """
        missing = [
            f"{name}.{parameter}"
            for name, tool in tools.items()
            for parameter, schema in _properties(tool).items()
            if not schema.get("description")
        ]
        assert missing == []

    def test_descriptions_are_dedented(self, tools: dict[str, Tool]) -> None:
        """A docstring inside a nested function is indented; ``__doc__`` keeps that.

        Four or more leading spaces is a code block in every markdown renderer, so the
        symptom is a tool whose reasoning arrives as a monospace slab.
        """
        indented = [
            name
            for name, tool in tools.items()
            for line in (tool.description or "").splitlines()
            if line.startswith("    ")
        ]
        assert indented == []


class TestBehaviouralHints:
    """What a harness reads when deciding whether to ask permission first."""

    def test_every_tool_is_annotated(self, tools: dict[str, Tool]) -> None:
        assert [name for name, tool in tools.items() if tool.annotations is None] == []

    def test_reads_and_writes_are_distinguishable(self, tools: dict[str, Tool]) -> None:
        writes = {name for name, tool in tools.items() if _hints(tool).read_only_hint is False}
        assert writes == set(WRITE_TOOLS)

    def test_only_cancelling_is_destructive(self, tools: dict[str, Tool]) -> None:
        """`create_sweep` commits a host to hours of work but takes nothing away.

        Cancelling does, and it is the one call here that a harness should stop and ask
        about.
        """
        destructive = {
            name for name, tool in tools.items() if _hints(tool).destructive_hint is True
        }
        assert destructive == set(DESTRUCTIVE_TOOLS)

    def test_authoring_a_sweep_is_not_idempotent(self, tools: dict[str, Tool]) -> None:
        """Calling it twice authors two sweeps; content-addressed creates do not.

        A harness retrying a dropped response is the case this describes.
        """
        assert _hints(tools["create_sweep"]).idempotent_hint is False
        assert _hints(tools["create_config"]).idempotent_hint is True
        assert _hints(tools["create_workload"]).idempotent_hint is True

    def test_nothing_claims_to_reach_outside_this_control_plane(
        self, tools: dict[str, Tool]
    ) -> None:
        """No tool contacts the GPU host, so none of them is open-world.

        This asserted the opposite once. `validate_config` was marked open-world on the
        reasoning that checking against a host's vLLM version must mean asking the host;
        it reads the row that host last wrote. The agent is reached by host registration
        and by the orchestrator, and neither is a tool.

        A tool that live-probes the agent would be the first true case — and would have to
        change this test deliberately, which is the point of asserting the whole set rather
        than each tool.
        """
        open_world = {name for name, tool in tools.items() if _hints(tool).open_world_hint}
        assert open_world == set()


class TestDescriptionsMatchBehaviour:
    """Prose, asserted where getting it wrong sends a caller down the wrong path."""

    def test_cancelling_describes_the_handover(self, tools: dict[str, Tool]) -> None:
        """It used to say the in-flight run "is interrupted", which this call does not do.

        The orchestrator stops it, within about three seconds. An agent that cancels and
        immediately polls sees that run still active, and a description promising an
        interruption makes the handover look like a failure — inviting a retry that is not
        needed.
        """
        description = tools["cancel_sweep"].description or ""
        assert "orchestrator" in description
        assert "is interrupted" not in description


class TestClosedSetsArePublishedAsEnums:
    """A valid value an agent has to discover by failing is an undocumented one."""

    def test_source_publishes_both_populations(self, tools: dict[str, Tool]) -> None:
        for name in ("get_pareto", "compare_runs"):
            assert _properties(tools[name])["source"]["enum"] == ["real", "synthetic"]

    def test_pareto_axes_publish_the_metric_catalogue(self, tools: dict[str, Tool]) -> None:
        """Derived from ``METRICS``, so the advertised keys are the accepted keys.

        Restating them in the signature is the drift this exists to prevent: the list
        would be correct on the day it was written and wrong on the day a metric is added.
        """
        properties = _properties(tools["get_pareto"])
        for axis in ("pareto_x", "pareto_y"):
            assert properties[axis]["enum"] == list(METRIC_KEYS)
        assert set(METRIC_KEYS) == set(METRICS_BY_KEY)


class TestAxisParsing:
    """The bug this issue started from: a call that succeeded on a different axis."""

    def test_every_catalogue_key_is_accepted(self) -> None:
        for key in METRIC_KEYS:
            assert _metric_key(key, "pareto_x") == key

    def test_an_unknown_axis_is_refused_not_substituted(self) -> None:
        with pytest.raises(ValueError, match="pareto_x must be one of"):
            _metric_key("tokens_per_watt", "pareto_x")

    def test_a_real_metric_that_is_not_an_axis_is_still_refused(self) -> None:
        """`mean_ttft_ms` is the dangerous input: not a typo, just not a key here.

        It is the shape of a metric name an agent would reasonably try, and the old
        behaviour answered it on total throughput without saying so.
        """
        with pytest.raises(ValueError, match="not a metric this control plane records"):
            _metric_key("mean_ttft_ms", "pareto_x")

    def test_the_refusal_names_the_valid_keys(self) -> None:
        """So recovery takes one turn rather than a search."""
        with pytest.raises(ValueError) as caught:
            _metric_key("nonsense", "pareto_y")
        assert "total_token_throughput_per_gpu" in str(caught.value)


class TestTheGuideMatchesTheSurface:
    """`docs/mcp.md` is a hand-written list, and hand-written lists drift.

    ADR 0001 is the precedent: it named ``start_sweep`` for a tool that shipped as
    ``create_sweep`` and never mentioned three others. Nothing checked it, so nothing
    stopped it.
    """

    def test_the_guide_exists(self) -> None:
        assert GUIDE.is_file()

    def test_it_lists_exactly_the_tools_that_ship(self, tools: dict[str, Tool]) -> None:
        # Table rows opening with a lowercase backticked identifier. Environment variables
        # are uppercase and resource URIs contain punctuation, so neither is caught here.
        documented = set(re.findall(r"^\| `([a-z][a-z0-9_]*)`", GUIDE.read_text(), re.MULTILINE))
        assert documented == set(tools)
