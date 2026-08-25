# vLLM custom all-reduce silently returns NaN over patched-driver P2P

**Status:** root cause isolated, fix not attempted
**Reported by:** vllm-benchmarking project, 2026-08-24
**Severity:** silent data corruption — no error, no warning, plausible throughput
**Audience:** a team picking this up to attempt a fix in vLLM, in the driver patch, or both
**Companion finding:** [`nccl-p2p-cuda-graph-replay.md`](nccl-p2p-cuda-graph-replay.md) — the
same driver patch also hangs NCCL's peer-to-peer transport, but only under CUDA graph replay.
Peer *writes* work in both cases; peer reads and graph-captured mappings are what fail.

---

## 1. Summary

On consumer GPUs with a community P2P driver patch applied, vLLM's custom all-reduce
kernel returns **NaN for 100% of elements** while reporting no error. The engine serves
normally, `vllm bench serve` records plausible tokens per second, and the model emits token
0 (`!!!!!!`) for every prompt.

The driver patch is not broken in the way you would expect. Peer-to-peer *data movement*
works correctly and roughly doubles in bandwidth. What does not work is **cross-process
CUDA IPC peer access**, which is the specific mechanism custom all-reduce is built on.

vLLM already has a check for exactly this failure — `gpu_p2p_access_check` — and it
correctly detects it. **The check is disabled by default.** `VLLM_SKIP_P2P_CHECK` defaults
to `"1"`, on the documented assumption that "drivers can report p2p status correctly". A
P2P patch is precisely a driver that reports P2P status *incorrectly*.

There are two separable pieces of work here:

- **A safety fix** in vLLM, so this class of failure cannot be silent. Low risk, no
  performance recovery.
- **A capability fix**, so custom all-reduce actually works on patched consumer hardware.
  Higher risk, recovers a real speedup.

They are independent. The safety fix is worth doing regardless of whether the capability
fix is feasible.

---

## 2. Why this matters more than a normal bug

vLLM's own comment anticipates this failure mode as a **hang**:

```python
# We assume drivers can report p2p status correctly.
# If the program hangs when using custom allreduce,
# potantially caused by a bug in the driver (535 series),
# if might be helpful to set VLLM_SKIP_P2P_CHECK=0
"VLLM_SKIP_P2P_CHECK": lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "1") == "1",
```

A hang is loud. What we observed is silent: the server is healthy, latency and throughput
are measurable and *better* than the working configuration, and the only symptom is that
the tokens are meaningless. A benchmark harness cannot tell the difference. Anyone
following one of the many "patch your driver for P2P on 3090s" write-ups can land in this
state and record a 6–15% improvement that is measuring a model computing nothing.

Neither vLLM's guidance nor any P2P patch README we found warns that the failure can be
silent corruption rather than a crash.

---

## 3. Environment

Everything below was held constant unless stated.

| | |
| --- | --- |
| GPUs | 2 × NVIDIA GeForce RTX 3090 (GA102) |
| Topology | `PHB` — both on CPU root ports `00:01.0` / `00:01.1`, PCIe 3.0 x8 each |
| ACS | **not implemented** on either root port (checked; nothing to disable) |
| BAR1 | 32 GB per card (Resizable BAR already enabled) |
| CPU / board | Intel i9-9900K / Z390 |
| OS | Ubuntu 24.04, kernel 7.0.0-30-generic |
| Stock driver | NVIDIA 610.43.02, **open** kernel module via DKMS |
| Patch | [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules) branch `610.43.02-p2p`, built at commit **`3590ded`** |
| vLLM | 0.25.1 |
| PyTorch | 2.11.0+cu128 |
| Model (serving repro) | `Qwen/Qwen3.8-27B-FP8`, TP=2 |

**The patch commit matters.** `3590ded` is the pure P2P mod. The branch tip additionally
carries `565e1b5` ("Experimental: 5000x faster cudaHostRegister"), which the repository's
own README says "may misbehave in edge cases the stock driver handles correctly". We
deliberately excluded it so the treatment is one thing. **The NaN is present without it.**

### Installation gotcha

On Ubuntu, `/etc/depmod.d/ubuntu.conf` sets `search updates ubuntu built-in`. DKMS installs
to `updates/dkms/`; `make modules_install` (which the patch's own `install.sh` runs) writes
to `kernel/`. On a machine with the distribution's DKMS driver installed, the patch install
**reports success and changes nothing**. Install the built `.ko` files into
`/lib/modules/$(uname -r)/updates/dkms/` and run `depmod -a`. Verify with
`modinfo -n nvidia`.

Also note `srcversion` is **byte-identical** between stock and patched builds of both
`nvidia.ko` and `nvidia-uvm.ko`, because the patch's substance lives in `src/nvidia/`,
which is linked in as a prebuilt object and never reaches modpost. It is not usable as a
"which build is loaded" fingerprint.

---

## 4. Evidence

Each row is a measurement, not an inference. Scripts are in
[§8](#8-reproduction-artifacts).

### 4.1 What works

| Check | Result |
| --- | --- |
| `nvidia-smi topo -p2p rw` | `OK` both directions (was `GNS`) |
| `torch.cuda.can_device_access_peer(0,1)` | `True` |
| Raw device-to-device copy, **data correctness** | **0 mismatches** at 4 K, 1 M, 16 M elements |
| Raw device-to-device copy, bandwidth | 3.11 → **6.69 GB/s** (85% of PCIe 3.0 x8 line rate) |
| NCCL all-reduce, default transport (SHM) | **exact** 2.0 at every size |
| NCCL all-reduce, `NCCL_P2P_LEVEL=SYS` | **exact** 2.0 at every size, ~30% faster ≥ 128 KB |

The data path is sound. This is the part that rules out "the patch corrupts DMA".

### 4.2 What fails

`CustomAllreduce.all_reduce()` on a tensor of ones, world size 2. Correct answer is 2.0
everywhere.

| Message | NCCL (same process) | vLLM custom all-reduce |
| --- | --- | --- |
| 8 KB | exact 2.0 | **4,096 NaN of 4,096** |
| 32 KB | exact 2.0 | **16,384 of 16,384** |
| 128 KB | exact 2.0 | **65,536 of 65,536** |
| 512 KB | exact 2.0 | **262,144 of 262,144** |
| 2 MB | exact 2.0 | **1,048,576 of 1,048,576** |

### 4.3 Characterising the NaN

| Question | Finding |
| --- | --- |
| Is the kernel skipping the write? | **No.** Output pre-filled with sentinel `-7.0`; **zero** sentinel values survive. The kernel actively writes every element. |
| Does it corrupt its input? | **No.** Source tensor still all 1.0 afterwards. |
| Is it dtype-specific? | **No.** bf16, fp16 and fp32 all 100% NaN. |
| What is the bit pattern? | Uniformly `0x7FFF` in bf16 — canonical positive quiet NaN. **Not** `0xFFFF`, so this is *not* the all-ones pattern a failed PCIe read returns. |
| Is it `expandable_segments`? | **No.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was present on the host and is a known custom-allreduce IPC hazard. Tested with and without: **identical NaN**. |
| Is it the IOMMU mode? | **No.** Identical under `intel_iommu=off` and under the patch README's own `intel_iommu=on iommu=pt` (`DMAR: IOMMU enabled`, `Default domain type: Passthrough`). |

### 4.4 vLLM detects it when asked

With `VLLM_SKIP_P2P_CHECK=0`, in the real engine:

```
INFO  all_reduce_utils.py:359] generating GPU P2P access cache in ~/.cache/vllm/gpu_p2p_access_cache_for_0,1.json
WARN  custom_all_reduce.py:162] Custom allreduce is disabled because your platform lacks
                                GPU P2P capability or P2P test failed.
```

```json
{ "0->0": true, "0->1": false, "1->0": false, "1->1": true }
```

Custom all-reduce is disabled automatically and **generated output is correct**.

---

## 5. Root cause

`vllm/distributed/device_communicators/all_reduce_utils.py` documents the exact mechanism
in its own docstring:

> Usually, checking if P2P access is enabled can be done by
> `torch.cuda.can_device_access_peer(src, tgt)`. However, sometimes the driver might be
> broken, and it returns `True` even if P2P access is not actually possible. […]
> We need to combine p2p and cuda IPC, so that: process src creates a tensor in GPU src,
> passes IPC handle to process tgt, and process tgt accesses the tensor in GPU tgt.

That combination — **peer access + cross-process CUDA IPC** — is what fails here.

**Stated precisely:**

> The patch grants intra-process peer access (`cudaDeviceCanAccessPeer` → `True`, and
> DMA-engine peer copies that move correct bytes). It does **not** grant working
> cross-process CUDA IPC peer access. vLLM's custom all-reduce depends on the latter, and
> vLLM's default configuration infers the latter from the former without testing it.

### What is *not* established

The document stops here deliberately. The following are open and are the actual work:

- **Where** in the IPC path it breaks: does `cudaIpcOpenMemHandle` return a pointer that
  does not map peer BAR1, or does the mapping succeed and the kernel's accesses fail?
- Whether the **data** region or the **synchronisation flag** region (or both) is affected.
  `0x7FFF` uniformly is suspiciously deterministic for "reads garbage" — it may indicate the
  kernel taking a failure path rather than reading corrupt data.
- Whether this is specific to the **610.43.02 port** of the patch or affects the
  longer-lived 550/580/590 branches too.
- Whether other IPC-based collectives (TensorRT-LLM, SGLang) fail the same way.

---

## 6. Workstreams

### WS-1 — Safety: make this class of failure non-silent

**Independent of everything else. Do this first; it is small.**

| Option | Change | Risk | Recovers perf |
| --- | --- | --- | --- |
| **1a** | Flip `VLLM_SKIP_P2P_CHECK` default to `0` | Adds ~6 s to engine start; two extra processes | No |
| **1b** | Run the real check only when the driver looks non-stock | Needs a reliable "is this driver patched" signal — note `srcversion` does **not** work | No |
| **1c** | **Post-init numerical self-test**: after `CustomAllreduce` initialises, all-reduce a known vector and assert the result. Disable and warn on mismatch. | Milliseconds. Catches this *and any future variant*, including ones the IPC probe would pass. | No |

**Recommendation: 1c, optionally with 1a.** 1c is the only option that is robust to
mechanisms nobody has thought of yet, and it is cheap — one small all-reduce at startup.
The current check validates the *transport*; 1c validates the *result*, which is what
actually matters and is what caught this bug.

Suggested assertion: all-reduce a vector of ones, require every element `== world_size`.
That is exactly the test in §4.2 and it is unambiguous.

### WS-2 — Capability: make custom all-reduce work over BAR1 P2P

Ranked by expected information per unit effort.

**H1 — IPC mapping does not reach peer BAR1.**
Write a minimal two-process test: process A allocates on GPU 0 and exports an IPC handle;
process B opens it, sets device to GPU 1, and reads/writes. Verify bytes both ways. This is
~60 lines and definitively separates "IPC is broken" from "the kernel is broken". *Do this
first — everything else depends on the answer.*

**H2 — Data lands but synchronisation flags do not.**
vLLM's kernel spin-waits on flags in a shared metadata region. If flag writes are not
visible across the peer mapping, ranks proceed on unwritten data. Test by instrumenting the
flag region directly, or by inserting a heavy barrier before the reduction and seeing
whether the NaN disappears. A positive result here points at memory-ordering/coherence
rather than at mapping.

**H3 — Version-specific regression in the 610 port.**
Build the `580.95.05-p2p` or `590.48.01-p2p` branch (the more widely used ones) and re-run
§4.2. Note this changes the userspace driver too, so it is a bigger change than it looks.
A pass here localises the bug to the 610 port and makes it an upstream patch issue rather
than a vLLM issue.

**H4 — vLLM could avoid IPC.**
If H1 confirms IPC is unavailable while peer access works, a same-process multi-device path
or a symmetric-memory path could get the speedup without IPC. This is a design change, not
a bug fix, and should only be considered after H1–H3.

### WS-3 — Upstream reporting

Regardless of whether a fix lands, both projects should know:

- **vLLM**: the failure mode is silent NaN, not the hang the code comment anticipates. Even
  if the default does not change, the comment and the docs should say so. Reference
  [#2728](https://github.com/vllm-project/vllm/issues/2728), which is the same class.
- **The patch maintainers**: the README should state that peer access being enabled does
  not imply working cross-process IPC, and that frameworks relying on IPC may silently
  produce garbage. Also worth noting the `updates/dkms` install trap from §3.

---

## 7. Scope

**In scope**

- Determining where in the peer-access/IPC path the failure occurs
- A vLLM-side change that makes the failure loud
- A vLLM-side or patch-side change that makes custom all-reduce correct on this hardware

**Out of scope**

- Making NVIDIA's stock driver support P2P on GeForce cards
- Performance work on the NCCL path (`NCCL_P2P_LEVEL` tuning is a separate, working win —
  see §9)
- Anything requiring NVLink hardware

**Non-goals**

- Reverse-engineering closed driver internals. If H1–H3 do not localise it, the honest
  outcome is a documented incompatibility plus WS-1, which is still a real improvement.

---

## 8. Reproduction artifacts

All scripts are standalone and need only the vLLM virtualenv.

| Script | What it establishes |
| --- | --- |
| `p2p_probe.py` | Peer access state, copy bandwidth, PCIe link state under load |
| `allreduce_probe.py` | NCCL all-reduce latency across message sizes (`torchrun --nproc_per_node=2`) |
| `customar_probe.py` | vLLM custom all-reduce vs NCCL, timing |
| `customar_correctness.py` | **The core repro** — NCCL exact vs custom all-reduce NaN |
| `customar_sentinel.py` | Distinguishes "wrote NaN" from "wrote nothing" |
| `greedy_capture.py` | Captures greedy completions for token-level diffing |
| `gate2.sh` | End-to-end: same prompts against both all-reduce paths |

### Minimal repro

```bash
# On a host with the P2P patch installed and peer access reading OK:
torchrun --nproc_per_node=2 customar_correctness.py
# Expect: nccl exact 2.0 at every size; custom_ar 100% NaN at every size.
```

### Confirming the workaround

```bash
VLLM_SKIP_P2P_CHECK=0 vllm serve --config <tp2-config.yaml>
# Expect: "Custom allreduce is disabled because your platform lacks GPU P2P
#          capability or P2P test failed" — and correct output.
```

---

## 9. What to tell users today

Until a fix exists, on any host running a P2P-patched driver:

1. **Set `VLLM_SKIP_P2P_CHECK=0`.** This makes vLLM verify rather than assume, and it
   correctly disables custom all-reduce here. Costs ~6 s at engine start.
2. **Or set `disable-custom-all-reduce: true` explicitly.** Equivalent outcome, no startup
   cost, but it will not protect against a future path that also depends on IPC.
3. **`NCCL_P2P_LEVEL=SYS`** (or `PHB` for this topology) is the configuration that actually
   delivers a correct speedup: verified numerically exact, and ~30% faster for messages
   ≥ 128 KB. NCCL declines the P2P path by default at `PHB` topology even when the driver
   allows it, so without this the patch buys nothing at all for vLLM.
4. **Verify output, not just throughput.** Any benchmark on a patched driver should diff
   generated tokens against a known-good configuration before its numbers are trusted.

---

## 10. Success criteria

**WS-1 is done when** a vLLM engine on a driver that misreports P2P capability either
refuses to use custom all-reduce or fails loudly, and a regression test covers it.

**WS-2 is done when** either custom all-reduce produces exact results over BAR1 P2P on
consumer hardware, or the failure is localised precisely enough to be a filed, actionable
bug against a named component.

**Either outcome is a result.** A documented, reproducible "this cannot work and here is
why" is worth substantially more than the current state, where a widely-shared tuning
recipe silently corrupts output on hardware people are actively buying for this purpose.
