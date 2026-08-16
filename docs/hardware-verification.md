# Awaiting hardware verification

Everything in this repository is built and tested without a GPU. That is deliberate — CI
has no GPU and the GPU host is the system under test, not a CI runner — but it means a
specific set of code paths have been written against a contract and exercised only through
their degradation paths.

This file lists them. It is the handover checklist for the first real GPU host, and it
should be worked through **before** trusting any measurement the system produces.

Each entry says what was verified, what was not, and how the untested path could fail.

---

## Verified on real hardware — 2026-08-16

Host `ubuntu-llm` at 192.168.10.102: **2× NVIDIA GeForce RTX 3090** (24 GiB each),
driver 610.43.02, CUDA 13.3, vLLM 0.25.1, agent in an isolated venv.

Checked read-only while a 27B model was mid-generation, so none of this disturbed a
running job.

| Item | Result |
| --- | --- |
| NVML device enumeration | ✅ Both devices, correct names, UUIDs, 24 GiB VRAM each |
| CUDA version arithmetic | ✅ `13.3` — the `major*1000 + minor*10` decode was right |
| Driver version | ✅ `610.43.02` |
| `synthetic_source` on a real agent | ✅ null, as invariant 7 requires |
| Every `/metrics` name we scrape | ✅ All six present on real GPU vLLM |
| Absence of a prefix-cache hit-rate gauge | ✅ Confirmed — the counters decision holds |
| Protocol mismatch refusal | ✅ Refused with both versions named; inspection escape hatch still worked |
| vLLM version probe | ❌ Returned null on the first attempt. Fixed — see below. |

Re-verified after reinstalling the agent **into the vLLM venv** (`~/vllm-env/.venv`):

| Item | Result |
| --- | --- |
| `vllm_version` | ✅ `0.25.1`, `"imported from the agent's own environment"` |
| Protocol 2 handshake | ✅ Control plane registered the host end to end |
| Reference version match | ✅ `0.25.1` matches `VLLM_REFERENCE_VERSION` |
| Install footprint | ✅ 2 packages added, nothing upgraded or downgraded |

### What the null version taught us

The agent runs in its own venv, so `import vllm` correctly fails and the subprocess
fallback runs. That fallback had a 15-second timeout, and `vllm --version` imports torch
and initializes CUDA — easily longer than that on a loaded host. The payload gave no clue
which of three problems it was.

Three changes came out of it:

1. The probe timeout is now 120s.
2. `HostInfo.vllm_probe_detail` explains a null version rather than leaving it bare.
3. **Run provenance now reads the engine's own `/version` endpoint** rather than the
   agent's environment probe. That is strictly better: the agent may live in a different
   venv, and it is the engine that produced the numbers.

---

## 0.1.0 — Foundations

All items in this milestone are now **verified** — see the table above. The remaining
work is in 0.2.0 and needs the GPU to be idle.

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

---

## 0.2.0 — Single run, end to end

### vLLM server launch and readiness
**File:** `packages/agent/src/vllmbench_agent/vllm_server.py`

- **Verified:** the whole state machine against a fake `vllm` binary — start, readiness
  polling, crash during load, never-becoming-ready, refusal of a second server, and
  teardown including SIGKILL escalation. Also verified end to end against the mock agent.
- **Not verified:** that a real `vllm serve --config` accepts the YAML we write, that a
  real engine's `/v1/models` appears when we expect, and how long a genuine model load
  takes relative to the 900s default timeout.
- **How it could fail:** a config vLLM rejects would surface as "exited with code N"
  with the reason only in the log tail. Worth reading that tail on the first real run.

### Teardown actually frees VRAM
**File:** `vllm_server.py`, `reaper.py`

- **Verified:** the process tree is signalled and reaped; no process survives.
- **Not verified:** that VRAM is actually released. Process exit and memory release are
  not the same event, and a driver that holds allocations after exit would look identical
  from the agent's side.
- **How to check:** watch `nvidia-smi --query-gpu=memory.used --format=csv -l 1` across a
  full run and confirm it returns to baseline after teardown.

### Orphan reaping on a real host
**File:** `reaper.py`

- **Verified:** reaping, the stale-record path, and the refusal to kill an unverifiable
  or recycled pid — all with fake processes.
- **Not verified:** against a real multi-process vLLM. The worker processes are children
  of the server, so signalling the group should cover them, but that has only been tested
  against a single-process fake.
- **How to check:** start a run, `kill -9` the agent, restart it, and confirm both the
  server *and* its per-device workers are gone and VRAM is released.

### Topology reporting
**File:** `runner.py`, `vllm_server.py`, `hardware.py` → `devices_for_process`

- **Closed in design, unverified in practice.** The gap is now filled by two separate
  sources rather than one: the config's *declared* `tensor_parallel_size`, and the
  devices NVML *observes* the server's process tree occupying. Per-GPU normalization
  divides by the observed count, so a config asking for more devices than the host can
  give no longer produces a run claiming a topology that never existed. A disagreement
  between the two is logged as a warning.
- **Not verified:** that `nvmlDeviceGetComputeRunningProcesses` actually reports vLLM's
  worker processes on this driver. It should — the workers are children of the server and
  the family set includes them — but it has only run against a single-process fake.
- **How to check:** start a TP=2 run and confirm `device_indices` comes back `[0, 1]`
  rather than `[0]` or empty.
