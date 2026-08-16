"""Benchmark invocation and result handling.

Two things are being protected. The argument list, because a flag that silently stops
applying turns into a benchmark measuring something other than what was asked for. And
the result handling, because the ways this fails quietly — exit zero with no file, a
truncated JSON — would otherwise record a run with no metrics and no sign of trouble.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from vllmbench_agent.bench import BenchError, build_argv, run_benchmark
from vllmbench_protocol.wire import BenchRequest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "vllm"
    target.write_text((FIXTURES / "fake_vllm").read_text())
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return target


class TestArgumentConstruction:
    def test_includes_save_result(self, fake_vllm: Path, tmp_path: Path) -> None:
        # Without --save-result there is no JSON, and the only alternative would be
        # scraping the human-readable table — a second parser that drifts on its own.
        argv = build_argv(
            BenchRequest(model="m"), base_url="http://127.0.0.1:8000", result_path=tmp_path / "r"
        )
        assert "--save-result" in argv
        assert "--result-filename" in argv

    def test_targets_loopback(self, fake_vllm: Path, tmp_path: Path) -> None:
        # Invariant 2. If this ever pointed at a remote host, network RTT would enter
        # every TTFT and ITL figure and nothing downstream could tell.
        argv = build_argv(
            BenchRequest(model="m"), base_url="http://127.0.0.1:8000", result_path=tmp_path / "r"
        )
        assert argv[argv.index("--base-url") + 1] == "http://127.0.0.1:8000"

    def test_unbounded_rate_becomes_inf(self, fake_vllm: Path, tmp_path: Path) -> None:
        argv = build_argv(
            BenchRequest(model="m", request_rate=None),
            base_url="http://x",
            result_path=tmp_path / "r",
        )
        assert argv[argv.index("--request-rate") + 1] == "inf"

    def test_unbounded_concurrency_omits_the_flag(self, fake_vllm: Path, tmp_path: Path) -> None:
        # Passing a large number instead of omitting it would quietly become a cap, and
        # a capped run mislabelled as uncapped is an invalid saturation measurement.
        argv = build_argv(
            BenchRequest(model="m", max_concurrency=None),
            base_url="http://x",
            result_path=tmp_path / "r",
        )
        assert "--max-concurrency" not in argv

    def test_explicit_limits_are_passed_through(self, fake_vllm: Path, tmp_path: Path) -> None:
        argv = build_argv(
            BenchRequest(model="m", request_rate=12.5, max_concurrency=64, burstiness=0.5),
            base_url="http://x",
            result_path=tmp_path / "r",
        )
        assert argv[argv.index("--request-rate") + 1] == "12.5"
        assert argv[argv.index("--max-concurrency") + 1] == "64"
        assert argv[argv.index("--burstiness") + 1] == "0.5"

    def test_extra_args_are_appended_verbatim(self, fake_vllm: Path, tmp_path: Path) -> None:
        # An escape hatch for flags this build predates, so a new vLLM option does not
        # require a release here.
        argv = build_argv(
            BenchRequest(model="m", extra_args=["--seed", "7"]),
            base_url="http://x",
            result_path=tmp_path / "r",
        )
        assert argv[-2:] == ["--seed", "7"]

    def test_missing_binary_is_explained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(BenchError, match="VLLMBENCH_VLLM_BIN"):
            build_argv(BenchRequest(model="m"), base_url="http://x", result_path=tmp_path / "r")


class TestExecution:
    async def test_returns_the_verbatim_payload(self, fake_vllm: Path) -> None:
        response = await run_benchmark(
            BenchRequest(model="facebook/opt-125m"), base_url="http://127.0.0.1:8000"
        )
        # Field names as vLLM actually emits them, not as the docs describe them.
        assert response.raw_result["completed"] == 8
        assert response.raw_result["p99_ttft_ms"] == 734.4
        assert response.duration_seconds > 0

    async def test_non_zero_exit_raises_with_output(
        self, fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_VLLM_BENCH_MODE", "fail")
        with pytest.raises(BenchError) as exc:
            await run_benchmark(BenchRequest(model="m"), base_url="http://x")
        assert "exited with code 2" in str(exc.value)
        assert "exploded" in str(exc.value)

    async def test_success_without_a_result_file_raises(
        self, fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The quietest possible failure, and the one that must be loudest.

        Exit zero with no result file means --save-result changed behaviour. Treating it
        as success would record a run with no metrics, indistinguishable from a
        benchmark that legitimately measured nothing.
        """
        monkeypatch.setenv("FAKE_VLLM_BENCH_MODE", "no_result")
        with pytest.raises(BenchError, match="wrote no result"):
            await run_benchmark(BenchRequest(model="m"), base_url="http://x")

    async def test_malformed_json_raises(
        self, fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_VLLM_BENCH_MODE", "bad_json")
        with pytest.raises(BenchError, match="not valid JSON"):
            await run_benchmark(BenchRequest(model="m"), base_url="http://x")

    async def test_timeout_is_enforced(
        self, fake_vllm: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A hung benchmark must not pin a GPU host indefinitely — the whole sweep behind
        # it would stall with no indication of why.
        monkeypatch.setenv("FAKE_VLLM_BENCH_MODE", "hang")
        monkeypatch.setenv("FAKE_VLLM_MODE", "hang")
        with pytest.raises(BenchError, match="timeout"):
            await run_benchmark(BenchRequest(model="m", timeout_seconds=2.0), base_url="http://x")

    async def test_result_flattens_with_the_shared_mapping(self, fake_vllm: Path) -> None:
        from vllmbench_protocol.bench_result import flatten_bench_result

        response = await run_benchmark(BenchRequest(model="m"), base_url="http://x")
        flat = flatten_bench_result(response.raw_result, gpu_count=2)
        assert flat["successful_requests"] == 8
        assert flat["output_token_throughput_per_gpu"] == pytest.approx(82.89 / 2)
