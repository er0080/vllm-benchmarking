# NCCL peer-to-peer does not survive CUDA graph replay on the community P2P driver

**Status:** root cause localised, mechanism hypothesised, not fixed
**Measured:** 2026-08-24, `ubuntu-llm`, 2× RTX 3090
**Companion finding:** [`vllm-custom-allreduce-cuda-ipc-p2p.md`](vllm-custom-allreduce-cuda-ipc-p2p.md)

---

## 1. Summary

On the community BAR1 P2P driver patch, enabling NCCL's peer-to-peer transport hangs vLLM's
tensor-parallel workers — but **only when CUDA graphs are captured**. Disabling graph capture
(`enforce-eager`) makes the same configuration run clean. Nothing else we varied does.

The failure is a hang, not a crash and not corruption. Workers stop answering vLLM's
`shm_broadcast` channel, `EngineCore` warns "No available shared memory broadcast block found
in 60 seconds" once a minute for five minutes, then raises `TimeoutError` and declares
`EngineDeadError`. Requests in flight return 500.

Two consequences matter operationally:

- **For speculative decoding, NCCL P2P is not merely risky, it is strictly dominated.** The
  only configuration that works — eager mode — runs at 50.77 tok/s, less than half the
  121.51 tok/s obtained by simply not using P2P.
- **The experimental push kernel is, on this hardware, the more reliable component.** It works
  with CUDA graphs, works with MTP, and produced greedy token hashes identical to NCCL. It did
  not fail in any configuration tested.

---

## 2. Why this is worth a driver team's time

The same driver patch has now produced two distinct failures against two different consumers
of peer memory:

| consumer | access pattern | failure |
| --- | --- | --- |
| vLLM custom all-reduce (upstream, pull-based) | SM-issued peer **reads** | returns NaN for 100% of elements, silently |
| NCCL P2P transport | peer transfers inside a **replayed CUDA graph** | hangs indefinitely |

A third consumer — a push-based kernel that writes to peers and never reads from them, and
does not use in-band completion flags — works correctly in both graph and eager mode.

That pattern suggests the patch delivers peer **writes** dependably and something weaker for
everything else. A driver-side fix that made peer reads and graph-captured mappings behave
like a real peer aperture would repair both failures at once, and would let this hardware use
upstream vLLM unmodified.

---

## 3. Environment

| | |
| --- | --- |
| GPUs | 2× NVIDIA GeForce RTX 3090 |
| Topology | `PHB` (across the CPU host bridge), **PCIe 3.0 x8** — root ports are `Width x8`, cards report `LnkSta: Width x8 (downgraded)` from `LnkCap: Width x16` |
| Driver | 610.43.02, community BAR1 P2P patch (`aikitoria/open-gpu-kernel-modules`) |
| Kernel | 7.0.0-30-generic, `intel_iommu=on iommu=pt pcie_aspm=off` |
| vLLM | 0.25.1 |
| torch / NCCL | 2.11.0+cu128 / **2.28.9** |
| Model | `Qwen/Qwen3.8-27B-FP8`, TP=2, fp8 KV cache |
| Speculation | MTP, `num_speculative_tokens: 3` |
| Workload | ShareGPT-format, 8 prompts, `--max-concurrency 2`, `--sharegpt-output-len 512` |

IOMMU is in passthrough, which is what the patch requires — not a deviation. ACS is not
implicated: the GPUs' root ports (`00:01.0`, `00:01.1`) advertise no ACS capability at all,
and the two chipset bridges that do have every control bit clear.

---

## 4. The controlled series

One variable at a time, everything else identical. `ok`/`failed` are requests out of 8.

| # | CUDA graphs | NCCL P2P | custom all-reduce | ok | failed | tokens | duration | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | FULL_AND_PIECEWISE | `SYS` | off | 2 | 6 | 7 | 300.8 s | 0.02 |
| 2 | FULL_AND_PIECEWISE | `PHB` | off | 2 | 6 | 7 | 300.8 s | 0.02 |
| 3 | FULL_AND_PIECEWISE | `SYS` | off, push env unset | 2 | 6 | 7 | 300.7 s | 0.02 |
| 4 | FULL_AND_PIECEWISE | `SYS` + `NCCL_PROTO=Simple` | off | **0** | 8 | 0 | 300.1 s | 0.00 |
| 5 | **PIECEWISE** | `SYS` | off | 2 | 6 | 4 | 303.5 s | 0.01 |
| 6 | **off (`enforce-eager`)** | `SYS` | off | **8** | **0** | **3,926** | 77.3 s | 50.77 |
| 7 | FULL_AND_PIECEWISE | off | off | **8** | **0** | 3,771 | 31.0 s | 121.51 |
| 8 | FULL_AND_PIECEWISE | off | **push** | **8** | **0** | 3,771 | 27.2 s | **138.67** |
| 9 | FULL_AND_PIECEWISE | `SYS` | **push** | **0** | 8 | 0 | — | 0.00 |

Rows 1–3 establish that the *level* is irrelevant — `SYS` and `PHB` both enable P2P at this
topology and fail identically — and that the push environment variables are not involved
(row 3 has them unset and fails the same way; all of rows 1–5 set
`disable-custom-all-reduce: true`, so the custom kernel is not on the path at all).

Row 6 is the finding. Row 5 sharpens it: the trigger is not capturing the *whole* decode step,
because PIECEWISE capture is enough.

Rows 7 and 8 are the deployable configurations. Row 9 shows the push kernel does not rescue
NCCL — it replaces NCCL's P2P, it does not coexist with it.

### The failure signature

```
08:18:55 (EngineCore) No available shared memory broadcast block found in 60 seconds.
08:19:55 (EngineCore) ... repeated
08:20:55 (EngineCore) ... repeated
08:21:55 (EngineCore) ... repeated
08:22:53 (EngineCore) EngineCore encountered a fatal error.
                      mq.dequeue → shm_broadcast.acquire_read → TimeoutError
08:22:53 (APIServer)  vllm.v1.engine.exceptions.EngineDeadError
```

The scheduler dump at the moment of death shows both requests mid-decode with
`scheduled_spec_decode_tokens: [-1, -1, -1]` and `num_spec_tokens_to_schedule: 3` — the
drafter was asked for tokens and the step never returned. Last worker output before silence
is Triton JIT compiling `eagle_prepare_next_token_padded_kernel` and
`eagle_step_slot_mapping_metadata_kernel`.

**The timeout surfaces in vLLM's own shared-memory IPC, not in NCCL.** A worker blocked inside
a collective cannot service `shm_broadcast`, so `EngineCore` times out waiting on it. The
NCCL stall is upstream of the error that gets reported, which is why the message names the
wrong subsystem.

### Non-speculative behaviour

MTP is an amplifier, not the cause. Across protocol-9 runs with speculation off:

| `NCCL_P2P_LEVEL` | successful runs | engine crashes | degenerate runs |
| --- | --- | --- | --- |
| `SYS` | 59 | 2 | 1 |
| unset / disabled | 59 | 0 | 0 |

Both crashes hit at concurrency 8. The degenerate run was recorded `succeeded` at 0.015 tok/s
— a benchmark whose requests mostly failed. So the same fault occurs without speculation, just
rarely. MTP issues several extra small collectives per decode step, all inside the captured
graph, which is a plausible reason it moves the failure from occasional to near-certain.

### Non-speculative throughput, for the cost side of the trade

Per-GPU output tok/s, 3 replicates, SDs 0.002–0.37:

| concurrency | neither | push only | NCCL P2P only | both |
| --- | --- | --- | --- | --- |
| 1 | 19.491 | 19.927 | 20.201 | 20.403 |
| 2 | 31.463 | 32.392 | 33.057 | 33.490 |
| 4 | 49.796 | 52.091 | 53.676 | 54.789 |
| 8 | 68.718 | 74.314 | 75.940 | 80.099 |
| 16 | 89.805 | 95.721 | 101.770 | 105.406 |

The two accelerations are **additive but sub-additive**, capturing 80–90% of the sum of their
individual gains. Interaction term at concurrency 16: observed 105.406 against an additive
prediction of 107.686, i.e. −2.28 tok/s. That is the signature of two mechanisms that mostly,
but not cleanly, partition message sizes — custom all-reduce takes messages below its size
threshold, NCCL takes the rest.

The patched driver with P2P declined is indistinguishable from the stock driver (19.49 vs
19.48, 68.72 vs 68.65, 89.78 vs 89.68), so the patch is genuinely inert until something asks
for P2P.

---

## 5. What is established, and what is not

**Established by measurement:**

- CUDA graph capture is necessary for the hang. Removing it, and nothing else, fixes it.
- The NCCL P2P *level* is irrelevant; enabling P2P at all is what matters.
- The push kernel is not involved: it is disabled by config in every hanging arm.
- Bare NCCL all-reduce over this same P2P path **works** — `allreduce_probe.py`, 8 KB to
  8 MB, correct results, up to 5.88 GB/s. It never captures a graph.
- The push kernel works *with* CUDA graphs and MTP, producing greedy token hashes identical
  to the NCCL path (`402c8b268867fb72`).

**Not established:**

- That the driver patch is the cause rather than a contributing condition. We have not run
  this series on the stock driver, because the stock driver cannot enable P2P at all — the
  comparison does not exist. It remains possible that NCCL P2P plus CUDA graphs is fragile on
  *any* PCIe-P2P consumer setup.
- Which specific NCCL structure goes stale across replay. We have not instrumented NCCL.
- Whether the failure is in the mapping, the completion signalling, or the ordering.
- Whether `NCCL_DEBUG=INFO` shows the transport selection changing between capture and replay.
  This was not captured and is the cheapest next diagnostic.

---

## 6. Mechanism hypothesis

The patch does not create a peer aperture. Per the
[tinygrad patchset discussion](https://github.com/tinygrad/tinygrad/discussions/4486), it
rewrites GMMU page-table entries so peer addresses become **bus addresses under
`GMMU_APERTURE_SYS_NONCOH`** — a non-coherent *system* aperture — because
`GMMU_APERTURE_PEER` is unsupported on these parts. The stated intent is to "use bus addr
(mmio addr for another gpu) to simulate pcie dma transaction between gpu and cpu."

Peer memory is therefore an MMIO window that behaves like peer memory for straightforward
DMA, which is consistent with everything that works here: bulk copies are correct to 16M
elements, and bare NCCL all-reduce is correct and fast.

CUDA graph capture records buffer addresses and mappings into a structure replayed many times
without re-validation. A synthesised mapping is exactly the kind that can lose validity or
ordering guarantees across replay, where a native peer aperture would not. That would leave
NCCL spinning on a completion that never arrives — which is what we observe.

This is a hypothesis consistent with the evidence, not a diagnosis. It predicts that
instrumenting NCCL's P2P setup across a capture/replay boundary would show either a stale
mapping or a lost completion signal.

---

## 7. Hypotheses tested and falsified

Recorded so nobody spends a cycle re-running them.

| hypothesis | test | result |
| --- | --- | --- |
| LL/LL128's in-band 8-byte flag is not atomic over a non-coherent aperture | `NCCL_PROTO=Simple` | **Worse** — 0/8 vs 2/8. Falsified. |
| ACS on the root ports redirects P2P to the root complex | `lspci -vvv` on `00:01.0`/`00:01.1` | No ACS capability advertised. Ruled out. |
| IOMMU is misconfigured for the patch | Patch README requires `iommu=pt`; host has it | Compliant. Ruled out. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks peer IPC | Set in every arm, including all passing ones | Not sufficient to cause it. |
| The push kernel or its env vars are implicated | Arm 3, push env unset | Fails identically. Ruled out. |
| It is specific to `NCCL_P2P_LEVEL=SYS` | `PHB` arm | Identical failure. It is P2P itself. |
| A narrower graph mode avoids it | `cudagraph_mode: PIECEWISE` | Hangs. Falsified. |

---

## 8. Corroboration

[**tinygrad/open-gpu-kernel-modules #45**](https://github.com/tinygrad/open-gpu-kernel-modules/issues/45)
— 2× RTX 4090, same patch family, NCCL 2.26.5. `all_reduce_perf` hangs at collective setup
across all message sizes; `NCCL_P2P_DISABLE=1` resolves it completely. Open, no maintainer
explanation. Our result is the same failure with the trigger identified.

[**tinygrad/open-gpu-kernel-modules #17**](https://github.com/tinygrad/open-gpu-kernel-modules/issues/17)
— P2P improves 2-GPU alltoall (6.75 vs 4.86 GB/s) but destroys 8-GPU (0.49 vs 4.76). Scale
dependent, consistent with a marginal rather than broken transport.

[**vllm-project/vllm #41530**](https://github.com/vllm-project/vllm/issues/41530) — the
identical symptom chain (repeated `shm_broadcast` stalls → `sample_tokens` RPC timeout →
`EngineDeadError`) on 8× datacenter GPUs running DeepSeek-V4-Pro with MTP, NCCL 2.28.9, and
**no P2P patch at all**. Root cause unidentified upstream, no workaround. Their engine ran
90 minutes before hanging; ours hangs in under five.

That last one is important for scoping: there may be an upstream vLLM fragility in the
MTP/`shm_broadcast` path that a marginal P2P transport makes trivially reachable. The driver
team should not assume the whole fault is theirs.

---

## 9. Operational guidance

For anyone running this hardware today.

**With speculative decoding (MTP):**

```yaml
disable-custom-all-reduce: false   # push kernel; 138.67 tok/s
```
```bash
export VLLM_CUSTOM_ALLREDUCE_PUSH=1
export VLLM_PUSHAR_LIB=/path/to/pushar.so
# NCCL_P2P_LEVEL deliberately unset
```

Falling back to `disable-custom-all-reduce: true` with `NCCL_P2P_LEVEL` unset gives 121.51
tok/s and uses no experimental code. Both are safe. **Do not set `NCCL_P2P_LEVEL` at all** —
`enforce-eager` is the only way to combine it with MTP and it costs more than P2P gains.

**Without speculative decoding**, `NCCL_P2P_LEVEL=SYS` is worth +13.4% at concurrency 16 and
stacks with the push kernel to +17.4%, at a measured cost of 3 bad runs in 59.

**`NCCL_P2P_LEVEL` is host-wide.** The agent passes its whole environment to every engine it
launches, so a host cannot have it for one workload and not another. It is also invisible to
the config hash, which is why `run.engine_env` exists (protocol 9, issue #124) — without it,
runs differing in this setting are byte-identical in every other recorded field.

---

## 10. Reproduction artifacts

In [`repro/`](repro/):

| file | what it does |
| --- | --- |
| `mtp_arm.sh` | One MTP benchmark under a named environment variation. `NCCL_P2P_LEVEL_SET`, `NCCL_PROTO_SET`, `CFG_SET` select the arm; remaining arguments name variables to unset. Emits a single `RESULT` line. |
| `mtp_matrix.sh` | The four-arm matrix, capturing greedy token hashes *before* benchmarking so correctness and speed are separate claims. |
| `allreduce_probe.py` | Bare NCCL all-reduce timing. Passes over P2P — the control that showed the transport is not broken outright. |
| `greedy_capture.py` | Greedy completions at temperature 0, hashed, for comparing all-reduce paths. |
| `mtp-repro.yaml` | The MTP config under test. |
| `mtp-eager.yaml` | Same, plus `enforce-eager: true` — the arm that passes. |
| `mtp-piecewise.yaml` | Same, plus `cudagraph_mode: PIECEWISE` — the arm that does not. |

Minimum reproduction: start `vllm serve` with `mtp-repro.yaml` and `NCCL_P2P_LEVEL=SYS`, send
8 ShareGPT prompts at concurrency 2, wait five minutes. Then repeat with `mtp-eager.yaml`.

---

## 11. Questions this leaves for the kernel team

1. Does a peer mapping created under `GMMU_APERTURE_SYS_NONCOH` remain valid, and correctly
   ordered, across CUDA graph replay? If not, is that fixable in the patch or inherent to
   simulating a peer aperture with system-noncoherent bus addresses?
2. Why do peer **writes** appear dependable while peer **reads** return NaN
   ([companion finding](vllm-custom-allreduce-cuda-ipc-p2p.md)) and graph-replayed transfers
   stall? A single answer covering all three would be the useful one.
3. Would `NCCL_DEBUG=INFO` across a capture/replay boundary show the transport being
   re-selected or a mapping being reused? Cheapest next diagnostic and we have not run it.
4. Is the PCIe 3.0 **x8** link relevant, or incidental? All measurements here are at x8.
