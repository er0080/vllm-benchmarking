# NCCL P2P hangs vLLM when CUDA graphs and expandable segments are both on

**Status:** cause identified and matched to a known upstream issue; not a driver-patch defect
**Measured:** 2026-08-24 / 2026-08-25, `ubuntu-llm`, 2× RTX 3090
**Companion finding:** [`vllm-custom-allreduce-cuda-ipc-p2p.md`](vllm-custom-allreduce-cuda-ipc-p2p.md)
**Supersedes:** PR #127, closed unmerged — see §2

---

## 1. Summary

Enabling NCCL's peer-to-peer transport hangs vLLM's tensor-parallel workers — but only when
**all three** of the following hold at once:

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. CUDA graphs are captured (any mode, including `PIECEWISE`)
3. NCCL P2P is enabled (`NCCL_P2P_LEVEL=SYS` or `PHB`)

Remove any one and the run is clean. This is a known upstream interaction, not a property of the
community P2P driver patch: NCCL auto-registers collective buffers under graph capture and assumes
each is one contiguous allocation, which expandable segments make false
([pytorch/pytorch#158029](https://github.com/pytorch/pytorch/issues/158029)).

With expandable segments off, `NCCL_P2P_LEVEL=SYS` runs clean under MTP and combines with the push
all-reduce kernel for the fastest configuration measured on this host — 139.94 tok/s against
121.12 with neither.

The failure is a hang, not a crash and not corruption. Workers stop answering vLLM's
`shm_broadcast` channel, `EngineCore` warns "No available shared memory broadcast block found in
60 seconds" once a minute for five minutes, then raises `TimeoutError: RPC call to sample_tokens
timed out` and declares `EngineDeadError`. Requests in flight return 500.

**The timeout surfaces in vLLM's own shared-memory IPC, not in NCCL.** A worker blocked inside a
collective cannot service `shm_broadcast`, so `EngineCore` times out waiting on it. The NCCL stall
is upstream of the error that gets reported, which is why the message names the wrong subsystem.

---

## 2. What this replaces, and the mistake that made it necessary

An earlier version of this document concluded that **NCCL P2P does not survive CUDA graph replay**
on the patched driver, and proposed a mechanism in which the patch's synthesised peer mapping
loses validity across replay. It was retracted before merge.

Every arm behind that conclusion had `expandable_segments:True` set. Because it was set in *all*
of them — including the passing `enforce-eager` arm — it was entered in the falsified-hypotheses
table as "not sufficient to cause it" and dismissed.

That inference is invalid. A variable held constant across every arm in a series is **untestable
within that series**, not ruled out. Holding it constant is what makes the other comparisons
clean; it says nothing about the held variable's own role. It was in fact a necessary condition,
and `enforce-eager` passed because it removes condition 2, not because graph replay was the fault.

A second, narrower error followed. The retraction itself proposed memory pressure as the
replacement cause, on the grounds that `enforce-eager` also frees CUDA graph memory pools. The
attribution arms falsify that too: the failing arm had **more** free VRAM than a passing one —
2,329 MiB against 1,291 MiB.

| `gpu-memory-utilization` | expandable segments | free after load | result |
| --- | --- | --- | --- |
| 0.95 | on | not recorded | hang, 5× |
| 0.95 | **off** | 1,291 / 1,357 MiB | **8/8**, 126.97 tok/s |
| 0.90 | **on** | 2,329 / 2,357 MiB | **0/8**, hang |
| 0.90 | off | 2,563 / 2,629 MiB | 8/8, 127.12 tok/s |

Both wrong answers came from the same habit: taking the first mechanism that fit the arms already
run, instead of asking what the series could not distinguish.

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

IOMMU is in passthrough, which is what the patch requires. ACS is not implicated: the GPUs' root
ports (`00:01.0`, `00:01.1`) advertise no ACS capability at all.

---

## 4. The controlled cross

`gpu-memory-utilization: 0.90` throughout, MTP on, `ok`/`failed` out of 8. Greedy hashes are
captured before benchmarking, so correctness and speed are separate claims.

| # | expandable | CUDA graphs | NCCL P2P | custom all-reduce | ok | failed | tok/s | greedy hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | off | on | off | off | **8** | 0 | 121.12 | `402c8b268867fb72` |
| 2 | off | on | `SYS` | off | **8** | 0 | 127.12 | `402c8b268867fb72` |
| 3 | off | on | off | push | **8** | 0 | 138.37 | `402c8b268867fb72` |
| 4 | off | on | `SYS` | push | **8** | 0 | **139.94** | `402c8b268867fb72` |
| 5 | **on** | on | off | off | **8** | 0 | 115.18 | `402c8b268867fb72` |
| 6 | **on** | on | off | push | **8** | 0 | 138.01 | `402c8b268867fb72` |
| 7 | **on** | **off** (eager) | `SYS` | off | **8** | 0 | 55.45 | `6582e99dac76f022` |
| 8 | **on** | on | `SYS` | off | **0** | 8 | 0.00 | — |
| 9 | **on** | on | `SYS` | push | **0** | 8 | 0.00 | — |

Rows 8 and 9 are the only failures, and they are the only rows with all three conditions present.
Every passing row removes one of them: rows 1–4 remove expandable segments, rows 5–6 remove NCCL
P2P, row 7 removes CUDA graphs.

Three secondary results fall out of this:

- **The push kernel is immune.** Row 6 runs clean with expandable segments on: it does not use
  NCCL's registration path. It also does not rescue NCCL — row 9 still fails, because enabling the
  push kernel does not disable NCCL P2P.
- **Expandable segments have no consistent throughput cost.** Row 5 is 5% below row 1 (115.18
  against 121.12), but it also generated 4% more tokens (3,938 against 3,771), so the two are not
  measuring the same work — batch composition shifts with allocator timing, and MTP acceptance
  shifts with it. Rows 3 and 6 differ by 0.3%, and the §5 intervention arms match their
  expandable-off equivalents to within the replicate spread. The reason not to set it is the hang
  and the VRAM, not throughput.
- **Eager mode is not a numerically identical fallback.** Row 7's greedy hash differs from every
  other passing row. Disabling graph capture changes kernel selection, so greedy output changes
  with it. That is expected, but it means "just run eager" is not a free substitution if outputs
  are being compared across configurations.

### The failure signature

```
03:26:58 (EngineCore) No available shared memory broadcast block found in 60 seconds.
03:27:58 (EngineCore) ... repeated
03:28:58 (EngineCore) ... repeated
03:29:58 (EngineCore) ... repeated
03:31:58 (EngineCore) TimeoutError: RPC call to sample_tokens timed out.
03:31:58 (APIServer)  vllm.v1.engine.exceptions.EngineDeadError
```

---

## 5. The upstream cause

[**pytorch/pytorch#158029**](https://github.com/pytorch/pytorch/issues/158029) — "NCCL + cudagraphs
+ expandable segments result in IMA". Reproduced on 8 GPUs with `all_gather_into_tensor` inside a
`torch.cuda.graph()` context. The stated cause: NCCL performs automatic buffer registration under
CUDA graphs and assumes input/output memory is one contiguous block. Expandable segments reserve a
virtual address range with `cuMemAddressReserve` and commit physical memory lazily with
`cuMemMap`, so that assumption does not hold. Open, classified as requiring an upstream NCCL fix.
The documented workaround is `NCCL_GRAPH_REGISTER=0`.

[**vllm-project/vllm#42609**](https://github.com/vllm-project/vllm/issues/42609) — the same
allocator behaviour breaking a different consumer: vLLM's *upstream* custom all-reduce fails at
`cudaIpcGetMemHandle` with "invalid argument" under expandable segments, because an IPC handle
requires memory from a single `cudaMalloc`/`cuMemCreate` allocation and the base pointer of an
expandable segment is not that. Scoped to DP>1 **and** TP>1, so it does not apply to this host's
TP=2 / DP=1 topology, but it is the same failure class.

vLLM 0.25.1 already knows about this class of bug and guards one instance of it. `VllmConfig.
_verify_kv_transfer_compat` **raises** if `expandable_segments:True` is set alongside any KV
connector, with the comment that CUDA VMM "can remap a virtual address range to different physical
pages over the engine's lifetime", leaving registrations "pointing at stale physical pages after
any remap". That is the same mechanism as the NCCL graph-registration failure. The guard covers
RDMA-pinned KV cache and does not cover NCCL P2P under graphs.

### The intervention

Correlation with an upstream issue is not a diagnosis, so the workaround was run as an
intervention. `NCCL_GRAPH_REGISTER=0`, **expandable segments left on**, everything else as in the
failing arms:

| arm | `gpu-mem-util` | custom all-reduce | without the flag | with `NCCL_GRAPH_REGISTER=0` |
| --- | --- | --- | --- | --- |
| `reg0-090-sys` | 0.90 | off | 0/8, hang | **8/8**, 127.29 tok/s |
| `reg0-095-sys` | 0.95 | off | hung 5× | **8/8**, 127.40 tok/s |
| `reg0-090-both` | 0.90 | push | 0/8, hang | **8/8**, 140.00 tok/s |

All three produced greedy hash `402c8b268867fb72`, matching every other passing arm. The middle
row is the configuration that hung five separate times during the retracted investigation.

This confirms the mechanism by intervention rather than by resemblance: disabling exactly the
registration path that pytorch#158029 names, and changing nothing else, converts every failing arm
into a passing one.

Upstream warns that the flag costs performance. On this host that cost is not measurable —
127.29 against 127.12 tok/s for the same arm with expandable segments off instead, and 140.00
against 139.94 on the push path. Both are within the replicate spread of §8. The registration
NCCL is being denied is a zero-copy optimisation whose benefit scales with message size, and TP=2
all-reduce on a 27B model does not appear to reach the size where it pays.

That makes `NCCL_GRAPH_REGISTER=0` a viable second line of defence for anyone who needs expandable
segments for other reasons. It is not the primary recommendation, because not setting
`PYTORCH_CUDA_ALLOC_CONF` at all avoids the VRAM cost as well as the hang.

---

## 6. What remains for the kernel team

Much less than the retracted document claimed. The hang is an upstream PyTorch/NCCL interaction
reproducible on stock hardware with no patch involved, and the correct fix is a configuration
change on our side. Two patch-specific questions survive, and the first is the smaller one:

**Upstream reports this as an invalid memory access; we observe a hang.** On stock hardware a
stale registration resolves to an address that faults. Under the patch, peer addresses are bus
addresses in a non-coherent *system* aperture (`GMMU_APERTURE_SYS_NONCOH`), per the
[tinygrad patchset discussion](https://github.com/tinygrad/tinygrad/discussions/4486) — because
`GMMU_APERTURE_PEER` is unsupported on these parts. A stale mapping into an MMIO window may be
valid-but-wrong rather than invalid, which would turn a fault into a silent spin on a completion
flag that never arrives.

If that is right it is a *diagnosability* concern rather than a correctness one — the patch turns
a crash into a hang — and it is worth knowing whether other classes of bad access degrade the same
way. It is not, on this evidence, a reason to hold the patch.

**Does bare NCCL P2P hold up under sustained load with no CUDA graphs and no PyTorch allocator?**
This is the open one, and §8 is why. `all_reduce_perf` from `nccl-tests` is reported hanging on
2× RTX 4090 under this patch family with none of the machinery in §5 present, and our sweep
database carries 2 unexplained engine crashes in 66 P2P runs that the allocator fix does not
account for. Running `all_reduce_perf` on this host would settle whether those belong to the patch
or upstream, and it needs no vLLM at all.

The companion finding — [peer **reads** returning NaN](vllm-custom-allreduce-cuda-ipc-p2p.md) — is
unaffected by any of this. It was measured on isolated kernel tests with no engine, no CUDA graphs
and no allocator involvement, and it stands on its own.

---

## 7. Hypotheses tested and falsified

Recorded so nobody spends a cycle re-running them.

| hypothesis | test | result |
| --- | --- | --- |
| ~~`expandable_segments:True` breaks peer IPC~~ | ~~Set in every arm, including passing ones~~ | **This was the cause.** The reasoning that dismissed it is corrected in §2. |
| NCCL P2P does not survive CUDA graph replay | §4 rows 2 and 4 — graphs on, P2P on, expandable off | **Falsified.** Clean at 127.12 and 139.94 tok/s. |
| Memory pressure from CUDA graph pools | §2 attribution table | **Falsified.** The failing arm had 1 GiB *more* free VRAM than a passing one. |
| LL/LL128's in-band 8-byte flag is not atomic over a non-coherent aperture | `NCCL_PROTO=Simple` | Worse — 0/8 vs 2/8. Falsified. |
| ACS on the root ports redirects P2P to the root complex | `lspci -vvv` on `00:01.0`/`00:01.1` | No ACS capability advertised. Ruled out. |
| IOMMU is misconfigured for the patch | Patch README requires `iommu=pt`; host has it | Compliant. Ruled out. |
| The push kernel or its env vars are implicated | Arm with push env unset | Fails identically. Ruled out. |
| It is specific to `NCCL_P2P_LEVEL=SYS` | `PHB` arm | Identical failure. It is P2P itself. |
| A narrower graph mode avoids it | `cudagraph_mode: PIECEWISE` | Hangs. Falsified — any capture is enough. |

The `NCCL_PROTO`, ACS, IOMMU, `PHB` and `PIECEWISE` rows were all measured against the
expandable-segments-on baseline. They remain valid as relative results — the held variable was
held for all of them — but they describe behaviour in a configuration nobody should now run.

---

## 8. The residual, and why it is not closed

Fixing the allocator interaction does not account for everything observed at `NCCL_P2P_LEVEL=SYS`.

**Under MTP, the configuration is clean at n=7.** Five dedicated replicates plus the two
attribution arms, all 8/8, mean 127.47 tok/s with an SD of 0.14 and the same greedy hash every
time. Whatever residual exists is not reachable this way.

**Without speculation, the sweep database shows a small excess of engine crashes at `SYS`.**
Every protocol-9 run has `PYTORCH_CUDA_ALLOC_CONF` absent — the agent's environment comes from its
own env file, not the interactive shell — so these are all expandable-segments-off runs:

| `NCCL_P2P_LEVEL` | runs | engine crashes |
| --- | --- | --- |
| `SYS` | 66 | 2 |
| absent | 66 | 0 |

Two events is not enough to distinguish from chance: Fisher's exact two-sided **p = 0.50**. It is
recorded because it is the only signal left, not because it is established.

Two external reports bear on it, and they point in different directions:

[**tinygrad/open-gpu-kernel-modules #45**](https://github.com/tinygrad/open-gpu-kernel-modules/issues/45)
— 2× RTX 4090, same patch family, NCCL 2.26.5. `all_reduce_perf` hangs at collective setup across
all message sizes; `NCCL_P2P_DISABLE=1` resolves it. **This cannot be the interaction documented
here**: `nccl-tests` uses neither CUDA graphs nor PyTorch's caching allocator. It is evidence that
a patch-level P2P fragility exists independently of anything in §5, and it is the better candidate
for the residual above.

[**vllm-project/vllm #41530**](https://github.com/vllm-project/vllm/issues/41530) — the identical
symptom chain (repeated `shm_broadcast` stalls → `sample_tokens` RPC timeout → `EngineDeadError`)
on 8× datacenter GPUs running MTP with NCCL 2.28.9 and **no P2P patch at all**. Root cause
unidentified upstream. Evidence in the other direction: some of what looks patch-specific is an
upstream MTP fragility that any marginal transport makes reachable.

The cheapest next diagnostic is `all_reduce_perf` from `nccl-tests` on this host — no vLLM, no
graphs, no PyTorch allocator. If it hangs, the residual is tinygrad #45 and belongs to the kernel
team. If it passes, the residual is upstream and does not.

---

## 9. Operational guidance

**Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on this host.** It breaks NCCL P2P
under CUDA graphs, and it costs roughly 250 MiB of VRAM — which on a 24 GiB card at
`gpu-memory-utilization: 0.95` with MTP halves the longest prompt that will fit, from 4,411 tokens
to 2,199. If it is needed for some other reason, set `NCCL_GRAPH_REGISTER=0` alongside it; that
removes the hang at no measurable throughput cost, but not the VRAM cost.

**Best measured configuration under MTP**, 139.94 tok/s:

```yaml
disable-custom-all-reduce: false   # push kernel
gpu-memory-utilization: 0.90
```
```bash
export NCCL_P2P_LEVEL=SYS
export VLLM_CUSTOM_ALLREDUCE_PUSH=1
export VLLM_PUSHAR_LIB=/path/to/pushar.so
# PYTORCH_CUDA_ALLOC_CONF deliberately unset
```

For a configuration using no experimental code, drop the push kernel and keep
`NCCL_P2P_LEVEL=SYS`: 127.12 tok/s, against 121.12 with no P2P at all.

**`gpu-memory-utilization` is a separate concern from any of the above.** At 0.95 with MTP, a
prompt over ~4,400 tokens exhausts VRAM and kills the engine mid-request regardless of allocator
or transport settings. 0.90 fits prompts past 70,000 tokens. 0.85 will not load the model at
`max-model-len: 262144`.

**`NCCL_P2P_LEVEL` is host-wide.** The agent passes its whole environment to every engine it
launches, so a host cannot have it for one workload and not another. It is also invisible to the
config hash, which is why `run.engine_env` exists (protocol 9, issue #124) — without it, runs
differing in this setting are byte-identical in every other recorded field. That column is what
established that no run in the sweep database had expandable segments set, which is how the
residual in §8 was scoped.

---

## 10. Reproduction artifacts

In [`repro/`](repro/):

| file | what it does |
| --- | --- |
| `arm2.sh` | One MTP arm under a named environment. Takes its base environment from `~/.bashrc` via `bash -ic` rather than carrying a copy, so a changed shell profile cannot silently test yesterday's configuration. Emits one `RESULT2` line. |
| `expandable_cross.sh` | The §4 cross: expandable segments against transport and graph mode. |
| `graph_register_arms.sh` | The `NCCL_GRAPH_REGISTER=0` intervention. |
| `noexp_replicates.sh` | Five replicates of MTP + `SYS` + graphs with expandable segments off, sizing the §8 residual. |
| `expandable-segments-results.txt` | Every `RESULT2` line behind §2, §4, §5 and §8, with the environment each arm ran under. |
| `mtp090-base.yaml` | MTP config, custom all-reduce off. |
| `mtp090-push.yaml` | Same, custom all-reduce on for the push kernel. |
| `greedy_capture.py` | Greedy completions at temperature 0, hashed, for comparing all-reduce paths. |
| `allreduce_probe.py` | Bare NCCL all-reduce timing. Passes over P2P — the control showing the transport is not broken outright. |

Minimum reproduction: start `vllm serve --config mtp090-base.yaml` with `NCCL_P2P_LEVEL=SYS` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, send 8 ShareGPT prompts at concurrency 2, wait
five minutes. Then unset `PYTORCH_CUDA_ALLOC_CONF` and repeat.
