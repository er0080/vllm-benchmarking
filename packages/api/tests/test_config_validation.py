"""The configuration validation engine.

Unit tests against the *captured* catalogue, not a hand-made one. A fixture invented here
would encode the same belief about vLLM's option set that the engine is supposed to be
checking, and both would agree while being wrong together.

The arrangement of these tests follows the cost of the mistake they catch. A config that
cannot start wastes a model load and a host claim; a config that starts and does something
other than what was written wastes a sweep and produces numbers that look fine.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from vllmbench_api.config_validation import Severity, validate_config
from vllmbench_api.main import app as api_app
from vllmbench_api.settings import ApiSettings
from vllmbench_db.models import GpuHost
from vllmbench_db.session import create_engine, create_session_factory
from vllmbench_db.testing import reset_database, test_database_url
from vllmbench_protocol.serve_args import ServeArguments, reference_serve_arguments


@pytest.fixture(scope="module")
def arguments() -> ServeArguments:
    return reference_serve_arguments()


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


@pytest.fixture
async def host_id(session: AsyncSession) -> str:
    """A host on the reference version, so its catalogue is an exact match."""
    host = GpuHost(
        name="ubuntu-llm",
        agent_url="http://agent",
        gpu_count=2,
        vllm_version=reference_serve_arguments().vllm_version,
    )
    session.add(host)
    await session.commit()
    return str(host.id)


@pytest.fixture
async def future_host_id(session: AsyncSession) -> str:
    """A host running a vLLM this control plane has never captured."""
    host = GpuHost(name="ahead", agent_url="http://agent", gpu_count=8, vllm_version="99.0.0")
    session.add(host)
    await session.commit()
    return str(host.id)


def errors(result: object) -> list[str]:
    return [f.message for f in result.findings if f.severity is Severity.ERROR]  # type: ignore[attr-defined]


class TestTheConfigThatActuallyRan:
    def test_it_passes(self, arguments: ServeArguments) -> None:
        """The config behind the first real sweep, byte for byte.

        A validator that rejects a configuration known to have run is worse than no
        validator: it teaches the author to ignore it.
        """
        result = validate_config(
            "model: Qwen/Qwen3.5-9B\n"
            "tensor-parallel-size: 1\n"
            "max-model-len: 8192\n"
            "gpu-memory-utilization: 0.90\n",
            arguments,
            gpu_count=2,
        )
        assert result.valid, result.findings
        assert result.findings == []

    def test_the_tp2_arm_passes_too(self, arguments: ServeArguments) -> None:
        result = validate_config(
            "model: Qwen/Qwen3.5-9B\ntensor-parallel-size: 2\nmax-model-len: 8192\n",
            arguments,
            gpu_count=2,
        )
        assert result.valid, result.findings


class TestArgumentsThatDoNotExist:
    def test_a_misspelling_is_caught_and_named(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\ngpu_memory_utilisation: 0.9\n", arguments)

        assert not result.valid
        (finding,) = [f for f in result.findings if f.key == "gpu_memory_utilisation"]
        assert "no `gpu_memory_utilisation` setting" in finding.message
        # The suggestion is offered, never applied: invariant 5 says validate, do not
        # transform, and a validator that rewrote the author's file would be doing both.
        assert finding.suggestion == "gpu-memory-utilization: 0.9"

    def test_underscores_and_dashes_are_the_same_argument(self, arguments: ServeArguments) -> None:
        """vLLM's parser normalizes them, so a validator that did not would reject
        configurations the engine accepts."""
        dashed = validate_config("model: m\ntensor-parallel-size: 1\n", arguments)
        scored = validate_config("model: m\ntensor_parallel_size: 1\n", arguments)
        assert dashed.valid and scored.valid

    def test_a_flags_negative_form_is_recognised(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\nno-enable-prefix-caching: true\n", arguments)
        assert result.valid, result.findings


class TestValuesTheParserWillRefuse:
    def test_a_choice_that_sounds_right_but_is_not(self, arguments: ServeArguments) -> None:
        """`fp16` is what everyone calls it and not what vLLM accepts."""
        result = validate_config("model: m\ndtype: fp16\n", arguments)

        assert not result.valid
        (finding,) = [f for f in result.findings if f.key == "dtype"]
        # The accepted list comes from the parser, so it cannot drift from what runs.
        assert "`float16`" in finding.message and "`half`" in finding.message
        assert finding.suggestion == "dtype: float16"

    def test_a_number_that_is_not_one(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\ntensor-parallel-size: two\n", arguments)
        assert "must be a whole number" in errors(result)[0]

    def test_a_switch_given_prose(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\nenable-prefix-caching: yes please\n", arguments)
        assert "on/off switch" in errors(result)[0]

    def test_true_is_not_a_number(self, arguments: ServeArguments) -> None:
        """bool is an int in Python and emphatically not one here."""
        result = validate_config("model: m\ntensor-parallel-size: true\n", arguments)
        assert not result.valid

    def test_a_human_readable_size_is_fine(self, arguments: ServeArguments) -> None:
        """`8k` is valid, and rejecting it would fail a config vLLM runs happily."""
        for value in ("8192", "8k", "8K", "auto"):
            result = validate_config(f"model: m\nmax-model-len: {value}\n", arguments)
            assert result.valid, (value, result.findings)

    def test_but_nonsense_is_not(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\nmax-model-len: enormous\n", arguments)
        assert not result.valid

    def test_an_uncheckable_type_is_left_alone(self, arguments: ServeArguments) -> None:
        """A value we cannot verify is not evidence of a mistake.

        Inventing a rule for `parse_dataclass` and friends would produce false errors on
        configurations that run, which costs more trust than the check would buy.
        """
        result = validate_config(
            'model: m\nkv-transfer-config: \'{"kv_connector": "x"}\'\n', arguments
        )
        assert result.valid, result.findings


class TestTheDocumentItself:
    def test_a_key_set_twice_is_an_error(self, arguments: ServeArguments) -> None:
        """Plain YAML takes the last one silently, which is the whole problem.

        The file hashes as a distinct config from the same file with one line removed,
        runs at whichever value came second, and nothing in any result says so.
        """
        result = validate_config(
            "model: m\ngpu-memory-utilization: 0.90\nmax-model-len: 8192\n"
            "gpu-memory-utilization: 0.98\n",
            arguments,
        )

        assert not result.valid
        (finding,) = [f for f in result.findings if "twice" in f.message]
        assert "lines 2 and 4" in finding.message

    def test_broken_yaml_reports_where(self, arguments: ServeArguments) -> None:
        result = validate_config("model: [unclosed\n", arguments)
        assert not result.valid
        assert "not valid YAML" in result.findings[0].message

    def test_a_list_is_not_a_config(self, arguments: ServeArguments) -> None:
        result = validate_config("- model: m\n- dtype: auto\n", arguments)
        assert not result.valid
        assert "mapping of settings" in result.findings[0].message

    def test_an_empty_config_is_refused_rather_than_passed(self, arguments: ServeArguments) -> None:
        """An empty document validating cleanly would be the most misleading pass here."""
        assert not validate_config("", arguments).valid
        assert not validate_config("# just a comment\n", arguments).valid

    def test_a_missing_model_warns_but_does_not_block(self, arguments: ServeArguments) -> None:
        """It can still start — the model may come from the command line. It just cannot
        be handed to another host and expected to do the same thing."""
        result = validate_config("tensor-parallel-size: 1\n", arguments)

        assert result.valid
        (finding,) = result.findings
        assert finding.severity is Severity.WARNING


class TestTopology:
    def test_more_gpus_than_the_host_has(self, arguments: ServeArguments) -> None:
        """The cheapest possible thing to catch, and one of the most expensive to
        discover from a failed sweep several minutes into a model load."""
        result = validate_config("model: m\ntensor-parallel-size: 4\n", arguments, gpu_count=2)
        assert "asks for 4 GPU(s)" in errors(result)[0]

    def test_tensor_and_pipeline_multiply(self, arguments: ServeArguments) -> None:
        result = validate_config(
            "model: m\ntensor-parallel-size: 2\npipeline-parallel-size: 2\n",
            arguments,
            gpu_count=2,
        )
        assert "asks for 4 GPU(s)" in errors(result)[0]

    def test_without_a_host_the_topology_is_not_second_guessed(
        self, arguments: ServeArguments
    ) -> None:
        """Validating a config in the abstract must not invent a machine to reject it
        against."""
        result = validate_config("model: m\ntensor-parallel-size: 8\n", arguments)
        assert result.valid, result.findings

    def test_zero_is_refused(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\ntensor-parallel-size: 0\n", arguments)
        assert not result.valid


class TestSweepInteraction:
    def test_a_swept_tensor_parallel_size_is_flagged_as_ignored(
        self, arguments: ServeArguments
    ) -> None:
        """Not an error — writing a baseline and then sweeping over it is reasonable.

        But the number in the file will not reach the engine, and an author reading the
        config back later has no way to know that from the text.
        """
        result = validate_config(
            "model: m\ntensor-parallel-size: 1\n",
            arguments,
            gpu_count=8,
            tensor_parallel_is_swept=True,
        )

        assert result.valid
        (finding,) = result.findings
        assert finding.severity is Severity.WARNING
        assert "replaced and never reaches the engine" in finding.message


class TestWhatItCheckedAgainst:
    def test_the_version_is_stated(self, arguments: ServeArguments) -> None:
        result = validate_config("model: m\n", arguments)
        assert result.checked_against == arguments.vllm_version
        assert result.exact_version_match

    def test_an_inexact_match_is_carried_on_the_result(self, arguments: ServeArguments) -> None:
        """Benchmarking one vLLM version against another is a headline use of this tool,
        so validating against a different catalogue is normal — and it changes what a
        clean result means, so it is stated rather than assumed."""
        result = validate_config("model: m\n", arguments, exact_version_match=False)
        assert not result.exact_version_match


def test_errors_sort_above_warnings(arguments: ServeArguments) -> None:
    """What stops it running goes at the top."""
    result = validate_config("dtype: fp16\n", arguments)
    severities = [f.severity for f in result.findings]
    assert severities == sorted(severities, key=lambda s: s is not Severity.ERROR)


# ---------------------------------------------------------------------------
# Through the endpoint, where the host supplies the version and the device count
# ---------------------------------------------------------------------------


class TestAgainstAHost:
    """The checks that need to know which machine this is for.

    Split from the pure ones because they are the only place the engine consults the
    database, and because the fallback behaviour — a host running a vLLM we have no
    capture for — is a normal situation rather than an error path.

    Marked integration for that reason: everything above this point runs in tier 1 with
    no services, and the engine is deliberately built so that most of it can.
    """

    pytestmark = pytest.mark.integration

    async def test_the_hosts_device_count_bounds_the_topology(
        self, client: httpx.AsyncClient, host_id: str
    ) -> None:
        body = (
            await client.post(
                "/api/configs/validate",
                json={"yaml": "model: m\ntensor-parallel-size: 4\n", "gpu_host_id": host_id},
            )
        ).json()

        assert body["valid"] is False
        assert "asks for 4 GPU(s)" in body["findings"][0]["message"]
        assert body["exact_version_match"] is True

    async def test_a_version_we_have_no_capture_for_falls_back_and_says_so(
        self, client: httpx.AsyncClient, future_host_id: str
    ) -> None:
        """Not an error. Benchmarking one vLLM version against another is a headline use
        of this tool, so a host ahead of our captures still gets checked — but a clean
        result means something weaker, and the caller is told which."""
        body = (
            await client.post(
                "/api/configs/validate",
                json={"yaml": "model: m\n", "gpu_host_id": future_host_id},
            )
        ).json()

        assert body["valid"] is True
        assert body["exact_version_match"] is False
        assert body["checked_against"] == reference_serve_arguments().vllm_version

    async def test_an_unknown_host_is_a_404_rather_than_a_silent_abstract_check(
        self, client: httpx.AsyncClient
    ) -> None:
        """Quietly dropping the topology checks would return `valid: true` for a config
        that cannot run on the machine the caller named."""
        response = await client.post(
            "/api/configs/validate",
            json={"yaml": "model: m\n", "gpu_host_id": str(uuid.uuid4())},
        )
        assert response.status_code == 404

    async def test_validating_stores_nothing(
        self, client: httpx.AsyncClient, session: AsyncSession
    ) -> None:
        """The engine describes a file; it does not create one."""
        before = len((await client.get("/api/configs")).json())
        await client.post("/api/configs/validate", json={"yaml": "model: brand-new\n"})
        assert len((await client.get("/api/configs")).json()) == before

    async def test_a_failing_config_can_still_be_created(self, client: httpx.AsyncClient) -> None:
        """The catalogue is a capture of one version, and a host running something newer
        may legitimately accept an argument this service has never heard of. Validation
        advises; it does not police."""
        yaml_text = "model: m\nsetting-from-the-future: 1\n"
        assert (await client.post("/api/configs/validate", json={"yaml": yaml_text})).json()[
            "valid"
        ] is False

        created = await client.post("/api/configs", json={"name": "future", "yaml": yaml_text})
        assert created.status_code == 201
