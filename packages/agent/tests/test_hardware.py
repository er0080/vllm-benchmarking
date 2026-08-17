"""Locating the `vllm` executable.

This is a small function guarding a large failure. The recommended deployment installs
the agent *into* the vLLM environment, and for a while the lookup consulted only PATH —
so that arrangement produced an agent that imported vLLM, reported the right version on
``/host-info``, answered every health check, and then failed every run with "no `vllm`
executable found". Healthy-looking and completely useless.

The tests below fix the resolution order in place, because the natural "fix" for that
symptom is to put the venv on PATH, which README explicitly argues against.
"""

from __future__ import annotations

import os
import stat
import sysconfig
from pathlib import Path

import pytest

from vllmbench_agent.hardware import (
    child_environment,
    resolve_vllm_binary,
    vllm_binary_search_detail,
)


def _make_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def isolated_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A PATH with no `vllm` on it, so PATH cannot mask the case under test."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


class TestResolution:
    def test_finds_vllm_beside_the_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_path: None
    ) -> None:
        """The normal case: agent installed into the vLLM venv, nothing on PATH.

        This is the regression. It failed on a real host with vLLM plainly installed.
        """
        scripts = tmp_path / "venv" / "bin"
        expected = _make_executable(scripts / "vllm")
        monkeypatch.setattr(
            sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
        )

        assert resolve_vllm_binary() == str(expected)

    def test_explicit_setting_wins_over_the_interpreter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VLLMBENCH_VLLM_BIN exists for the isolated-install case, so it must win."""
        scripts = tmp_path / "venv" / "bin"
        _make_executable(scripts / "vllm")
        monkeypatch.setattr(
            sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
        )

        elsewhere = _make_executable(tmp_path / "other" / "vllm")
        assert resolve_vllm_binary(str(elsewhere)) == str(elsewhere)

    def test_falls_back_to_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Agent installed beside rather than inside the vLLM environment."""
        monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path / "no-scripts"))
        on_path = _make_executable(tmp_path / "bin" / "vllm")
        monkeypatch.setenv("PATH", str(on_path.parent))

        assert resolve_vllm_binary() == str(on_path)

    def test_none_when_nowhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_path: None
    ) -> None:
        monkeypatch.setattr(sysconfig, "get_path", lambda name: str(tmp_path / "no-scripts"))
        assert resolve_vllm_binary() is None

    def test_non_executable_file_is_not_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_path: None
    ) -> None:
        """A `vllm` that cannot be executed is not a `vllm`.

        Worth pinning: a non-executable file here would be resolved, then fail at
        spawn with an OSError attributed to the run rather than to the install.
        """
        scripts = tmp_path / "venv" / "bin"
        scripts.mkdir(parents=True)
        (scripts / "vllm").write_text("not executable")
        monkeypatch.setattr(
            sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
        )

        assert resolve_vllm_binary() is None

    def test_explicit_setting_that_is_missing_does_not_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An explicit path that is wrong must fail, not silently find another vLLM.

        Falling back would mean benchmarking a different build than the operator named,
        and recording it under provenance that says otherwise.
        """
        scripts = tmp_path / "venv" / "bin"
        _make_executable(scripts / "vllm")
        monkeypatch.setattr(
            sysconfig, "get_path", lambda name: str(scripts) if name == "scripts" else ""
        )

        assert resolve_vllm_binary(str(tmp_path / "nonexistent" / "vllm")) is None


class TestSearchDetail:
    def test_names_the_script_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sysconfig, "get_path", lambda name: "/opt/venv/bin")
        detail = vllm_binary_search_detail()
        assert "/opt/venv/bin" in detail
        assert "PATH" in detail

    def test_names_the_explicit_setting(self) -> None:
        detail = vllm_binary_search_detail("/nope/vllm")
        assert "/nope/vllm" in detail
        assert "VLLMBENCH_VLLM_BIN" in detail


def test_real_interpreter_scripts_dir_is_discoverable() -> None:
    """sysconfig must actually answer for the interpreter running the tests.

    Guards the assumption the fix rests on: if `get_path("scripts")` returned nothing
    useful in a venv, the lookup would silently degrade to PATH-only again.
    """
    scripts = sysconfig.get_path("scripts")
    assert scripts
    assert os.path.isdir(scripts)


class TestChildEnvironment:
    """The environment vLLM is launched in.

    vLLM shells out to its own tooling — `ninja` for the inductor compile — and finds it
    the way an activated venv would. The agent execs by absolute path and activates
    nothing, so without this the child gets a bare PATH and dies with
    ``FileNotFoundError: 'ninja'``.

    The reason it needs a test rather than a comment: the failure is intermittent. With a
    warm compile cache everything passes, which is how the first real runs on the GPU
    host succeeded while being broken.
    """

    def test_puts_the_executable_directory_on_path(self, tmp_path: Path) -> None:
        executable = _make_executable(tmp_path / "venv" / "bin" / "vllm")
        env = child_environment(str(executable))
        assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "venv" / "bin")

    def test_keeps_the_rest_of_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        executable = _make_executable(tmp_path / "venv" / "bin" / "vllm")
        env = child_environment(str(executable))
        assert env["PATH"].endswith("/usr/bin:/bin")

    def test_does_not_duplicate_an_entry_already_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bin_dir = tmp_path / "venv" / "bin"
        executable = _make_executable(bin_dir / "vllm")
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin")
        env = child_environment(str(executable))
        assert env["PATH"].split(os.pathsep).count(str(bin_dir)) == 1

    def test_inherits_the_agent_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # HF_HOME decides which weights the engine resolves, so losing it would mean
        # re-downloading models and, worse, benchmarking a different snapshot than the
        # operator's own runs use.
        monkeypatch.setenv("HF_HOME", "/data/hf")
        executable = _make_executable(tmp_path / "venv" / "bin" / "vllm")
        assert child_environment(str(executable))["HF_HOME"] == "/data/hf"
