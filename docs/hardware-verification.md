# Awaiting hardware verification

Everything in this repository is built and tested without a GPU. That is deliberate — CI
has no GPU and the GPU host is the system under test, not a CI runner — but it means a
specific set of code paths have been written against a contract and exercised only through
their degradation paths.

This file lists them. It is the handover checklist for the first real GPU host, and it
should be worked through **before** trusting any measurement the system produces.

Each entry says what was verified, what was not, and how the untested path could fail.

---

## 0.1.0 — Foundations

### NVML device enumeration
**File:** `packages/agent/src/vllmbench_agent/hardware.py` → `probe_gpus`

- **Verified:** the no-driver path. On a machine without NVIDIA drivers the agent starts,
  reports zero GPUs, and the control plane registers the host without error.
- **Not verified:** that a real device enumerates correctly — index, name, UUID, VRAM.
- **How it could fail:** `nvmlDeviceGetName` returns `bytes` on some driver versions and
  `str` on others; `_decode` handles both, but the branch has never run against a driver.
  A wrong device name would land in run provenance and mislabel every result from that
  host.

### Driver and CUDA version probing
**File:** `hardware.py` → `probe_driver_version`, `probe_cuda_version`

- **Verified:** both return `None` cleanly with no NVML.
- **Not verified:** the CUDA version arithmetic. NVML packs the version as
  `major * 1000 + minor * 10`; the decode is written from documentation, not observation.
  This is exactly the class of assumption that was wrong for the benchmark JSON.
- **How it could fail:** a plausible-but-wrong version string recorded as provenance on
  every run.

### vLLM version detection
**File:** `hardware.py` → `probe_vllm_version`

- **Verified:** returns `None` when vLLM is absent, and the subprocess fallback is
  structurally correct.
- **Not verified:** the import path, which is the one that matters. The agent is installed
  *into* the vLLM venv, so `import vllm; vllm.__version__` should be the answer used in
  practice, and it has never run.
- **How it could fail:** silently falling through to the subprocess path, or to `None`,
  making every run's `vllm_version` provenance empty.

### Multi-GPU registration
**File:** `packages/api/src/vllmbench_api/routers/hosts.py`

- **Verified:** two-device registration and refresh, against the mock agent.
- **Not verified:** against real devices, including the device-inventory replacement path
  when cards change.

---

## How to work through this

Once the agent is installed on the GPU host:

```bash
# On the GPU host, inside the vLLM environment
uv pip install <this repo>
VLLMBENCH_TOKEN=... vllmbench-agent

# Confirm the facts are real, not defaults
curl -s -H "Authorization: Bearer $VLLMBENCH_TOKEN" http://localhost:9110/host-info | jq
```

Check each field against `nvidia-smi` by hand. Specifically:

- `gpus[].name` matches `nvidia-smi --query-gpu=name --format=csv`
- `gpus[].vram_bytes` matches `--query-gpu=memory.total`
- `driver_version` matches `--query-gpu=driver_version`
- `cuda_version` matches `nvidia-smi | head -3`
- `vllm_version` matches `python -c 'import vllm; print(vllm.__version__)'` **in the same
  environment the agent runs in**

A mismatch in any of these is a provenance bug, which under invariant 6 makes every run
from that host invalid rather than merely mislabelled.
