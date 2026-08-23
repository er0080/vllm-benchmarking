"""Reading peer-access state out of NVML.

Peer-to-peer DMA between consumer GPUs is enabled by rebuilding the *same driver version*
from patched sources, so nothing else a run records changes when it is turned on. That
makes this probe the only thing standing between two very different populations of
measurement and a chart that draws them as one series — which is worth more tests than a
function this size would normally earn.

The fake NVML below is built from a payload captured on a real dual-3090 host running
610.43.02, not from NVML's documentation. That distinction is not academic here: see
``test_a_one_tuple_caps_index_is_unwrapped``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from vllmbench_agent.hardware import probe_peer_access
from vllmbench_protocol import PeerAccessStatus

# Captured from `nvidia-ml-py` 13.610.43 on the reference host. NVML_P2P_CAPS_INDEX_READ
# really is a one-tuple there — a stray trailing comma upstream — and every other constant
# really is a bare int.
CAPS_READ: Any = (0,)
CAPS_WRITE = 1
STATUS_OK = 0
STATUS_GPU_NOT_SUPPORTED = 2


def _fake_pynvml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_count: int = 2,
    status: dict[tuple[int, int, int], int] | None = None,
    default_status: int = STATUS_OK,
    caps_read: Any = CAPS_READ,
    init_error: Exception | None = None,
    query_error: Exception | None = None,
) -> types.ModuleType:
    """Install a stand-in ``pynvml``, shaped like the real one on the reference host."""
    module = types.ModuleType("pynvml")

    module.NVML_P2P_CAPS_INDEX_READ = caps_read  # type: ignore[attr-defined]
    module.NVML_P2P_CAPS_INDEX_WRITE = CAPS_WRITE  # type: ignore[attr-defined]
    module.NVML_P2P_STATUS_OK = STATUS_OK  # type: ignore[attr-defined]
    module.NVML_P2P_STATUS_GPU_NOT_SUPPORTED = STATUS_GPU_NOT_SUPPORTED  # type: ignore[attr-defined]
    module.NVML_P2P_STATUS_UNKNOWN = 6  # type: ignore[attr-defined]

    def nvml_init() -> None:
        if init_error is not None:
            raise init_error

    module.nvmlInit = nvml_init  # type: ignore[attr-defined]
    module.nvmlShutdown = lambda: None  # type: ignore[attr-defined]
    module.nvmlDeviceGetCount = lambda: device_count  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = lambda i: f"handle-{i}"  # type: ignore[attr-defined]

    def get_p2p_status(handle: str, peer: str, index: Any) -> int:
        if query_error is not None:
            raise query_error
        # ctypes raises on a tuple; the real binding does exactly this.
        if not isinstance(index, int):
            raise TypeError(f"Don't know how to convert parameter 3: {index!r}")
        a = int(handle.removeprefix("handle-"))
        b = int(peer.removeprefix("handle-"))
        return (status or {}).get((a, b, index), default_status)

    module.nvmlDeviceGetP2PStatus = get_p2p_status  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module


class TestObservedStates:
    def test_all_pairs_ok_reads_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_pynvml(monkeypatch)
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.OK
        assert detail == []

    def test_a_refusing_driver_reads_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What the reference host reports before the module is patched."""
        _fake_pynvml(monkeypatch, default_status=STATUS_GPU_NOT_SUPPORTED)
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNSUPPORTED
        assert any("gpu_not_supported" in line for line in detail)

    def test_one_broken_pair_is_not_a_weaker_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Partial peer access is UNSUPPORTED, because the engine falls back for the group.

        Reporting OK because most pairs work would put a run whose all-reduce staged
        through host memory in a series with runs whose did not.
        """
        _fake_pynvml(
            monkeypatch,
            device_count=4,
            status={(1, 2, CAPS_WRITE): STATUS_GPU_NOT_SUPPORTED},
        )
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNSUPPORTED
        assert detail == ["1->2 write: gpu_not_supported"]

    def test_both_directions_are_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_pynvml(monkeypatch, status={(1, 0, CAPS_WRITE): STATUS_GPU_NOT_SUPPORTED})
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNSUPPORTED
        assert detail == ["1->0 write: gpu_not_supported"]

    def test_read_and_write_are_both_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_pynvml(monkeypatch, status={(0, 1, 0): STATUS_GPU_NOT_SUPPORTED})
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNSUPPORTED
        assert detail == ["0->1 read: gpu_not_supported"]


class TestScoping:
    def test_one_device_is_single_device_not_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The distinction the TP=1 control depends on.

        A run on one GPU has no peer access to report. Calling that "unsupported" would
        put it on one side of a boundary it cannot be on either side of, and would split a
        drift control away from itself the moment the interconnect changed underneath it.
        """
        _fake_pynvml(monkeypatch, device_count=2)
        status, detail = probe_peer_access([0])
        assert status is PeerAccessStatus.SINGLE_DEVICE
        assert detail == []

    def test_a_host_with_one_device_is_single_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_pynvml(monkeypatch, device_count=1)
        assert probe_peer_access()[0] is PeerAccessStatus.SINGLE_DEVICE

    def test_scoping_ignores_pairs_the_run_did_not_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run on 0 and 1 is not made unsupported by a broken pair on 2 and 3."""
        _fake_pynvml(
            monkeypatch,
            device_count=4,
            status={(2, 3, CAPS_WRITE): STATUS_GPU_NOT_SUPPORTED},
        )
        assert probe_peer_access([0, 1])[0] is PeerAccessStatus.OK
        assert probe_peer_access([2, 3])[0] is PeerAccessStatus.UNSUPPORTED

    def test_duplicate_indices_do_not_fabricate_a_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_pynvml(monkeypatch)
        assert probe_peer_access([1, 1])[0] is PeerAccessStatus.SINGLE_DEVICE


class TestUpstreamQuirks:
    def test_a_one_tuple_caps_index_is_unwrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`nvidia-ml-py` 13.610.43 defines NVML_P2P_CAPS_INDEX_READ as ``(0,)``.

        Passing that through to ctypes raises ``ArgumentError`` on every host, so the
        obvious implementation — read the documented constant, hand it to NVML — reports
        UNAVAILABLE everywhere and every run records "nobody could tell". This is the
        payload-not-documentation rule (CLAUDE.md) catching a live upstream typo, so it is
        pinned here rather than trusted to stay fixed.
        """
        _fake_pynvml(monkeypatch, caps_read=(0,))
        assert probe_peer_access()[0] is PeerAccessStatus.OK

    def test_an_int_caps_index_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """And keeps working when upstream fixes the comma."""
        _fake_pynvml(monkeypatch, caps_read=0)
        assert probe_peer_access()[0] is PeerAccessStatus.OK

    def test_a_missing_caps_index_is_unavailable_not_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_pynvml(monkeypatch, caps_read=None)
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNAVAILABLE
        assert any("READ" in line for line in detail)


class TestDegradation:
    def test_no_pynvml_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The laptop case. The agent must start and answer, reporting absence as data."""
        monkeypatch.setitem(sys.modules, "pynvml", None)
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNAVAILABLE
        assert detail

    def test_nvml_that_will_not_initialise_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_pynvml(monkeypatch, init_error=RuntimeError("driver not loaded"))
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNAVAILABLE
        assert any("driver not loaded" in line for line in detail)

    def test_a_refused_query_is_unavailable_not_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NVML declining to answer is not NVML answering "no".

        Recording it as UNSUPPORTED would assert an observation nobody made, and the
        assertion would be wrong in the direction that silently merges two populations.
        """
        _fake_pynvml(monkeypatch, query_error=RuntimeError("NVML_ERROR_NOT_SUPPORTED"))
        status, detail = probe_peer_access()
        assert status is PeerAccessStatus.UNAVAILABLE
        assert any("NVML_ERROR_NOT_SUPPORTED" in line for line in detail)


def test_probing_never_imports_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent may not initialise a CUDA context to answer this.

    ``cudaDeviceCanAccessPeer`` is the more direct question, but reaching it means
    importing torch and creating a context on every device — inside the agent, on the
    machine under test, potentially while a benchmark is running. Telemetry that perturbs
    the thing it measures is the one thing the agent may not do.
    """
    _fake_pynvml(monkeypatch)
    before = set(sys.modules)
    probe_peer_access()
    assert "torch" not in (set(sys.modules) - before)
