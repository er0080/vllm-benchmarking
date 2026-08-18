"""Structured logging, and the promise that a secret never reaches a log line.

The redaction tests carry the weight here. A structured field that goes missing is an
inconvenience; a token in a log file is a credential in whatever ships those logs, and it
stays there after it is rotated. So the cases below are deliberately the ones nobody
writes on purpose — an exception message, a traceback, a library's own output — because
those are where a leak actually comes from.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from vllmbench_protocol.logging import (
    MIN_SECRET_LENGTH,
    REDACTED,
    JsonFormatter,
    TextFormatter,
    bound,
    configure_logging,
    context,
    redact,
    scrub,
)

TOKEN = "s3cret-agent-token-not-a-real-one"


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Undo the global state these tests deliberately touch.

    Both halves are process-wide by design: one service, one set of credentials, one
    root handler. That is right in production and a hazard in a test session — and
    `configure_logging` clears root handlers, which is where pytest's own `caplog`
    lives. Without this, running these tests would silently break capture for every
    test that ran afterwards, in files with nothing to do with logging.
    """
    from vllmbench_protocol import logging as module

    secrets = set(module._secrets)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    uvicorn = {
        name: (list(logging.getLogger(name).handlers), logging.getLogger(name).propagate)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    try:
        yield
    finally:
        module._secrets.clear()
        module._secrets.update(secrets)
        root.handlers[:] = handlers
        root.setLevel(level)
        for name, (saved_handlers, propagate) in uvicorn.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = saved_handlers
            logger.propagate = propagate


def _record(message: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, args, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestRedaction:
    def test_a_registered_secret_never_reaches_the_output(self) -> None:
        redact(TOKEN)
        rendered = JsonFormatter("api").format(_record("connecting with %s", TOKEN))

        assert TOKEN not in rendered
        assert REDACTED in rendered

    def test_it_survives_lazy_formatting(self) -> None:
        """`log.info("%s", token)` does not build a string until the formatter runs.

        A filter would have to reimplement %-formatting to see this. The formatter sees
        what actually gets written, which is why the scrubbing lives there.
        """
        redact(TOKEN)
        rendered = TextFormatter("agent").format(_record("token=%s", TOKEN))
        assert TOKEN not in rendered

    def test_it_reaches_inside_a_traceback(self) -> None:
        """The case that matters, and the one a hand-placed mask always misses.

        Nobody writes a log line containing a database password. Postgres quotes the
        whole DSN back on a connection failure, from inside the driver, and it arrives
        here as an exception nobody inspected.
        """
        password = "hunter2-but-long-enough"
        redact(password)

        try:
            raise ConnectionError(
                f"connection to server failed: postgresql://vllmbench:{password}@db:5432/vllmbench"
            )
        except ConnectionError:
            import sys

            record = _record("database not ready")
            record.exc_info = sys.exc_info()
            rendered = JsonFormatter("orchestrator").format(record)

        assert password not in rendered
        assert REDACTED in json.loads(rendered)["exception"]

    def test_it_reaches_structured_fields_too(self) -> None:
        """A secret passed as context is still a secret.

        The JSON body is scrubbed after serialization rather than field by field, so
        there is no path into the output that skips it.
        """
        redact(TOKEN)
        rendered = JsonFormatter("api").format(_record("handshake", agent_token=TOKEN))
        assert TOKEN not in rendered

    def test_a_short_value_is_refused_loudly(self, caplog: pytest.LogCaptureFixture) -> None:
        """Replacing a three-character value everywhere would corrupt the log itself.

        Refusing is right; refusing *silently* would be the worst option available,
        because the caller then believes something is masked and it is not.
        """
        # A value with no accidental overlap with the warning's own wording, so the
        # last assertion below is testing what it claims to.
        with caplog.at_level(logging.WARNING):
            redact("pw1")

        assert scrub("pw1") == "pw1"
        assert "will NOT be masked" in caplog.text
        # And the warning must not name the thing it declined to protect, which would
        # put the unmasked value in the log by way of complaining that it is unmasked.
        assert "pw1" not in caplog.text

    def test_the_floor_matches_what_a_real_token_must_be(self) -> None:
        """`AgentSettings.token` requires 8 characters, so a real token always qualifies.

        Pinned because the two numbers drifting apart would mean a valid token that the
        redactor quietly declines to mask.
        """
        from vllmbench_agent.settings import AgentSettings

        field = AgentSettings.model_fields["token"]
        minimums = [m.min_length for m in field.metadata if hasattr(m, "min_length")]
        assert MIN_SECRET_LENGTH <= minimums[0]

    def test_a_secret_containing_another_is_fully_replaced(self) -> None:
        """Longest-first, so the shorter one's replacement is not left embedded."""
        redact("abcdefgh", "abcdefgh-and-more")
        assert scrub("value=abcdefgh-and-more") == f"value={REDACTED}"

    def test_empty_and_none_are_ignored(self) -> None:
        """Unset settings are normal — an unconfigured MCP token is empty, not absent."""
        redact("", None)
        assert scrub("anything at all") == "anything at all"


class TestBoundContext:
    def test_fields_reach_records_from_anywhere(self) -> None:
        """Including records this project did not create.

        That is the reason for a filter rather than a LoggerAdapter: SQLAlchemy's warning
        during a run should carry the run id, and SQLAlchemy has never heard of us.
        """
        from vllmbench_protocol.logging import ContextFilter

        with bound(run_id="abc"):
            record = _record("something a library logged")
            ContextFilter().filter(record)

        assert record.run_id == "abc"  # type: ignore[attr-defined]

    def test_nesting_adds_rather_than_replaces(self) -> None:
        with bound(sweep_id="s1"):
            with bound(run_id="r1"):
                assert context() == {"sweep_id": "s1", "run_id": "r1"}
            assert context() == {"sweep_id": "s1"}
        assert context() == {}

    def test_it_unwinds_even_when_the_block_raises(self) -> None:
        with pytest.raises(ValueError), bound(run_id="r1"):
            raise ValueError("boom")
        assert context() == {}

    def test_bound_fields_appear_in_the_json(self) -> None:
        record = _record("starting", run_id="r1", sweep_id="s1")
        payload = json.loads(JsonFormatter("orchestrator").format(record))

        assert payload["run_id"] == "r1"
        assert payload["sweep_id"] == "s1"
        assert payload["service"] == "orchestrator"
        assert payload["message"] == "starting"


class TestConfiguration:
    def test_the_root_handler_is_replaced_not_appended(self) -> None:
        """A second handler would be a path around the redaction.

        The whole guarantee rests on there being exactly one way out, so configuring
        twice must not leave the first handler in place.
        """
        configure_logging("api", fmt="json")
        configure_logging("api", fmt="json")

        assert len(logging.getLogger().handlers) == 1

    def test_uvicorn_is_unhooked_so_its_access_log_is_scrubbed_too(self) -> None:
        """uvicorn installs handlers with propagate=False, putting itself outside ours.

        Started from the command line there is no option to prevent that, so the loggers
        are unhooked after the fact instead.
        """
        access = logging.getLogger("uvicorn.access")
        access.addHandler(logging.NullHandler())
        access.propagate = False

        configure_logging("api", fmt="json")

        assert access.handlers == []
        assert access.propagate is True

    def test_format_is_selectable(self) -> None:
        configure_logging("api", fmt="text")
        assert isinstance(logging.getLogger().handlers[0].formatter, TextFormatter)

        configure_logging("api", fmt="json")
        assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)

    def test_json_output_is_one_object_per_line(self) -> None:
        """A log collector splits on newlines. A multi-line record is a corrupt record."""
        rendered = JsonFormatter("api").format(_record("line one\nline two"))

        assert "\n" not in rendered
        assert json.loads(rendered)["message"] == "line one\nline two"


class TestServicesRegisterWhatTheyHold:
    """Each service's startup registers its own secrets before anything else runs.

    Asserted by inspection rather than by starting the services, because the property
    that matters is *which values are registered*, and a running service would only
    demonstrate that on a log line that happened to contain one. A secret added to
    settings and not to this list is the failure mode — it produces no error, no test
    failure anywhere else, and a credential in a log file the first time something goes
    wrong near it.
    """

    def test_the_api_registers_both_tokens_and_the_database_password(self) -> None:
        import inspect

        import vllmbench_api.main as api_main

        source = inspect.getsource(api_main.lifespan)
        assert "configure_logging" in source
        for held in ("settings.token", "settings.mcp_token", "database_password()"):
            assert held in source, f"the API holds {held} and does not redact it"

    def test_the_orchestrator_registers_its_token_and_the_database_password(self) -> None:
        import inspect

        import vllmbench_orchestrator.__main__ as orchestrator_main

        source = inspect.getsource(orchestrator_main.main)
        assert "configure_logging" in source
        for held in ("settings.token", "database_password()"):
            assert held in source, f"the orchestrator holds {held} and does not redact it"

    def test_the_agent_registers_its_token(self) -> None:
        import inspect

        import vllmbench_agent.main as agent_main

        source = inspect.getsource(agent_main.main)
        assert "configure_logging" in source
        assert "settings.token" in source

    def test_the_agent_stops_uvicorn_installing_its_own_handlers(self) -> None:
        """It runs uvicorn programmatically, where `log_config=None` is available.

        The compose services start uvicorn from the command line and rely on the
        unhooking in `configure_logging` instead; here the cleaner option exists and
        should be used.
        """
        import inspect

        import vllmbench_agent.main as agent_main

        assert "log_config=None" in inspect.getsource(agent_main.main)
