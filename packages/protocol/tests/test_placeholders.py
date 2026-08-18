"""Credentials nobody got round to changing.

The failure this guards is not a crash. `.env.example` ships `change-me`, which is nine
characters and so clears every length check in the system; a stack holding it starts
cleanly, reports itself healthy, and runs for months on a shared secret that anyone with
the repository already has.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from vllmbench_protocol import is_placeholder, warn_about_placeholders


class TestRecognisingThem:
    @pytest.mark.parametrize("value", ["change-me", "CHANGE-ME", "  change-me  ", "changeme"])
    def test_the_example_value_in_its_usual_disguises(self, value: str) -> None:
        assert is_placeholder(value)

    @pytest.mark.parametrize("value", ["password", "secret", "vllmbench", "todo"])
    def test_the_words_people_reach_for_instead(self, value: str) -> None:
        assert is_placeholder(value)

    def test_a_generated_token_is_not_one(self) -> None:
        assert not is_placeholder("Zx4kQ9vLmP2rT8wYb6NcH3jF5sD7gA1e")

    def test_absence_is_not_a_placeholder(self) -> None:
        """Empty is a different problem with a different message, handled by the caller."""
        assert not is_placeholder("")
        assert not is_placeholder(None)

    def test_a_real_secret_that_merely_contains_one_is_left_alone(self) -> None:
        assert not is_placeholder("change-me-not-really-a1b2c3d4e5f6")


class TestWarning:
    def test_it_names_the_variable_and_returns_it(self, caplog: pytest.LogCaptureFixture) -> None:
        log = logging.getLogger("test.placeholders")
        with caplog.at_level(logging.WARNING):
            found = warn_about_placeholders(log, VLLMBENCH_TOKEN="change-me")

        assert found == ["VLLMBENCH_TOKEN"]
        assert "VLLMBENCH_TOKEN" in caplog.text

    def test_it_never_prints_the_value(self, caplog: pytest.LogCaptureFixture) -> None:
        """Complaining about a credential is a poor reason to put one in a log file.

        `change-me` is public, but the check must not learn the habit: the same code path
        sees whatever an operator actually set, including a half-changed real secret.
        """
        log = logging.getLogger("test.placeholders")
        with caplog.at_level(logging.WARNING):
            warn_about_placeholders(log, POSTGRES_PASSWORD="change-me")

        assert "change-me" not in caplog.text

    def test_it_says_how_to_generate_one(self, caplog: pytest.LogCaptureFixture) -> None:
        log = logging.getLogger("test.placeholders")
        with caplog.at_level(logging.WARNING):
            warn_about_placeholders(log, VLLMBENCH_TOKEN="change-me")

        assert "token_urlsafe" in caplog.text

    def test_a_real_secret_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        log = logging.getLogger("test.placeholders")
        with caplog.at_level(logging.WARNING):
            assert warn_about_placeholders(log, VLLMBENCH_TOKEN="Zx4kQ9vLmP2rT8wYb6Nc") == []

        assert caplog.text == ""


class TestServicesCheckWhatTheyHold:
    """Same shape as the redaction registry test, and for the same reason.

    A secret added to settings and not to this call is silent: nothing errors, nothing
    fails, and the deployment runs on an example value with no line anywhere saying so.
    """

    def test_the_api_checks_its_token_and_the_database_password(self) -> None:
        import vllmbench_api.main as api_main

        source = inspect.getsource(api_main.lifespan)
        assert "warn_about_placeholders" in source
        for held in ("VLLMBENCH_TOKEN=", "POSTGRES_PASSWORD="):
            assert held in source, f"the API holds {held} and does not check it"

    def test_the_orchestrator_checks_its_token_and_the_database_password(self) -> None:
        import vllmbench_orchestrator.__main__ as orchestrator_main

        source = inspect.getsource(orchestrator_main.main)
        assert "warn_about_placeholders" in source
        for held in ("VLLMBENCH_TOKEN=", "POSTGRES_PASSWORD="):
            assert held in source, f"the orchestrator holds {held} and does not check it"
