# Hardware verification

Everything in this repository is built and tested without a GPU. That is deliberate — CI
has no GPU and the GPU host is the system under test, not a CI runner — but it means code
paths get written against a contract and exercised only through their degradation paths.

This file records what has since been checked against real hardware, and what each check
found. It is kept because the findings are the useful part: every one of them was a bug
that looked like working software right up until a real GPU was involved.

Host `ubuntu-llm`: **2× NVIDIA GeForce RTX 3090** (24 GiB each), driver 610.43.02,
CUDA 13.3, vLLM 0.25.1.

---

## Verified — 2026-08-16, read-only

Checked while a 27B model was mid-generation, so none of it disturbed a running job.

| Item | Result |
| --- | --- |
| NVML device enumeration | ✅ Both devices, correct names, UUIDs, 24 GiB each |
| CUDA version arithmetic | ✅ `13.3` — the `major*1000 + minor*10` decode was right |
| Driver version | ✅ `610.43.02` |
| `synthetic_source` on a real agent | ✅ null, as invariant 7 requires |
| Every `/metrics` name we scrape | ✅ All six present on real GPU vLLM |
| Absence of a prefix-cache hit-rate gauge | ✅ Confirmed — the counters decision holds |
| Protocol mismatch refusal | ✅ Refused with both versions named |
| vLLM version probe | ❌ Returned null. Fixed — 15s timeout could not cover a torch import. |

---

## Verified — 2026-08-17, GPU idle

The full run path, with both cards free.

| Item | Result |
| --- | --- |
| Real `vllm serve --config` accepts our YAML | ✅ Including a production config with inline-JSON `speculative-config`, `reasoning-parser` and hyphenated keys, byte-for-byte |
| Model load time vs the 900s timeout | ✅ 27B FP8 at TP=2 ready in ~72s; opt-125m in ~25s |
| `--save-result` → column flattening | ✅ Every summary field matched the printed table |
| Benchmark client on loopback | ✅ `--base-url http://127.0.0.1:8000` (invariant 2) |
| Device attribution via NVML | ✅ See below — this one is worth reading |
| Per-GPU normalization | ✅ TP=1 4427 tok/s per GPU vs TP=2 1938 — see below |
| VRAM returns to baseline | ✅ 36 MiB → 7646 MiB → 36 MiB across a full cycle |
| Orphan reaping, real multi-process vLLM | ✅ See below |
| `gpu-memory-utilization` honoured | ✅ 0.30 of 24 GiB → 7.6 GiB resident |

### Device attribution is real, not inferred

`device_indices: [0, 1]` on a TP=2 run proves nothing on its own — `list(range(tp))`
produces the same answer. So the engine was pinned to physical GPU 1 with
`CUDA_VISIBLE_DEVICES=1` and run at TP=1:

```
tensor_parallel_size  1
gpu_count             1
device_indices        [1]
```

A config-derived answer would have said `[0]`. NVML process attribution is genuinely
being used, which is what invariant 8 needs: per-GPU normalization divides by the count
of devices actually occupied, not the count requested.

### Per-GPU normalization, and why it is not cosmetic

Same model, same workload, real measurements:

| TP | devices | tok/s total | tok/s per GPU |
| --- | --- | --- | --- |
| 1 | `[0]` | 4427 | **4427** |
| 2 | `[0, 1]` | 3875 | **1938** |

TP=2 is worse in aggregate *and* less than half as good per device, because
communication overhead dominates for a 125M model. Charting aggregate throughput would
have shown TP=2 as merely slightly behind rather than badly wasteful.

### Orphan reaping against a real engine

A TP=2 engine leaves a four-process tree: the server, `VLLM::EngineCore`, and one
`VLLM::Worker_TP` per device, holding 15.7 GiB across both cards.

`kill -9` on the agent, leaving the engine orphaned, then restart:

```
reaping orphaned vLLM server from a previous agent lifetime: pid=10984 pgid=10984
pid 10984 exited on SIGTERM
```

Parent and both workers gone, VRAM back to 36/4 MiB, no SIGKILL escalation needed.
Signalling the process group is what covers the workers — this is what
`start_new_session=True` at spawn buys.

---

## What the real hardware found

Five bugs, none of which any test caught, and all of which presented as healthy software.

**1 — The documented install left the host depending on a source checkout.**
`uv pip install ./packages/agent` from inside the workspace resolves `vllmbench-protocol`
through `tool.uv.sources` and installs it *editable*, even when the path is passed
explicitly. Deleting the clone killed the agent with `ModuleNotFoundError`. Now installed
from git, and `agent-install` in CI installs from a throwaway clone, deletes it, and
fails if anything still points at a directory.

**2 — The recommended deployment could not run anything.**
`resolve_vllm_binary` consulted only PATH, while the version probe answered by importing
vLLM. So an agent installed into the vLLM venv — the arrangement README recommends —
reported `vllm_version: 0.25.1`, passed every health check, and failed every run with
"no `vllm` executable found". Now resolved from the interpreter's own script directory.

**3 — The served-model alias was being used as the weights identifier.**
`vllm bench serve` loads the *tokenizer* from `--model`. Sending the alias there died on
`opt-125m is not a valid model identifier` — the lucky outcome. Had the alias been a real
repo id, it would have tokenized against an unrelated tokenizer and recorded confident,
wrong input-token counts. `--model` and `--served-model-name` are now separate all the
way through the wire (protocol 3).

**4 — vLLM could not find its own build tooling.**
The agent execs `vllm` by absolute path and activates nothing, so the child inherited a
bare PATH and vLLM's inductor step died with `FileNotFoundError: 'ninja'` — with ninja
sitting in the venv the whole time. Intermittent, because a warm compile cache hides it:
the first real runs passed for that reason alone, and a sweep, which varies settings and
so invalidates cache keys, would have hit it constantly.

**5 — The log tail was the wrong window.**
vLLM prints the worker's exception, unwinds through several hundred lines, and signs off
with `RuntimeError: Engine core initialization failed. See root cause above.` A 200-line
tail reliably keeps that sentence and discards what it points at. Terminal exception
lines are now collected as output streams past, and lead the failure message.

The common thread: each was a difference between *the environment vLLM is installed in*
and *the environment the agent gave it*, and none of them is visible without a real vLLM
on a real host.

---

## Re-checking after a change

```bash
# On the GPU host, inside the vLLM environment
uv pip install "git+https://github.com/er0080/vllm-benchmarking@v1.0.0rc4#subdirectory=packages/agent"
VLLMBENCH_TOKEN=... vllmbench-agent

curl -s -H "Authorization: Bearer $VLLMBENCH_TOKEN" http://localhost:9110/host-info | jq
```

Check each field against `nvidia-smi` by hand: `gpus[].name`, `gpus[].vram_bytes`,
`driver_version`, `cuda_version`, and `vllm_version` against
`python -c 'import vllm; print(vllm.__version__)'` **in the environment the agent runs
in**. A mismatch in any of these is a provenance bug, which under invariant 6 makes every
run from that host invalid rather than merely mislabelled.

### The agent's environment is the engine's environment

The agent passes its own environment to every `vllm` process it starts, so anything that
changes how vLLM resolves models changes the results. `HF_HOME` is the one that bites:
it is commonly set in `~/.bashrc`, which a service-launched agent never sources, and
without it the engine resolves against `~/.cache/huggingface` and re-downloads weights
the host already has. Set it where the agent will actually see it.

---

## First real sweep, 2026-08-17

Twelve runs on `ubuntu-llm` (2× RTX 3090, vLLM 0.25.1, driver 610.43.02): Qwen3.5-9B at
tensor-parallel 1 and 2, three concurrencies, two replicates, grouped. All twelve
succeeded. This is the first time the 0.5.0 analysis views were fed anything but the mock
agent, and it is recorded here because several of the findings are about the framework
rather than about the model.

### Per-device attribution is real, and it matters more than expected

The agent reported `devices=[0]` for the TP=1 engine and `devices=[0, 1]` for TP=2, both
observed through NVML rather than read back from the config. Runs are *created* carrying
the host's device count — 2 — and corrected to what actually ran, so a TP=1 run ends up
`gpu_count=1, device_indices=[0]`.

The arithmetic check that this landed: for the TP=1 runs, aggregate throughput and
per-GPU throughput are the same number (702.2), and for TP=2 the per-GPU figure is exactly
half the aggregate (1013.1 → 506.6). Had the correction not happened, every per-GPU figure
in every view would have read half its true value, and would have looked entirely
plausible doing so.

### Aggregate throughput and per-GPU throughput disagree about the answer

At concurrency 4, moving from TP=1 to TP=2 is:

| | aggregate | per GPU | per user |
| --- | --- | --- | --- |
| TP=1 | 702 tok/s | 702 | 40.8 tok/s |
| TP=2 | 1017 tok/s | 508 | 63.6 tok/s |

**+44.8% aggregate, −27.6% per GPU.** Reported on aggregate alone this is an unambiguous
win; normalized per device it is a trade — you bought latency with a card. Invariant 8
exists for exactly this, and the first real sweep produced a case of it.

### Tensor-parallel efficiency is not monotonic in concurrency

| concurrency | TP=2 speed-up | TP=2 efficiency |
| --- | --- | --- |
| 4 | 1.45× | 72.4% |
| 16 | 1.11× | 55.5% |
| 64 | 1.73× | 86.4% |

The dip at 16 and the recovery at 64 are explained by the other half of the data: TP=1
peaks at concurrency 16 (1796 tok/s/GPU) and *falls* at 64 (1681). With ~4 GB of KV cache
left after weights on a single 24 GB card, concurrency 64 thrashes. TP=2 has twice the
cache headroom and keeps scaling, so the second card earns its place precisely where the
first one runs out — and is worst value at 16, where one card is still comfortable.

A scaling view that reported a single efficiency number per configuration would have
averaged this away.

### Concurrency 64 is off the frontier at both widths

Four of six points are Pareto-optimal. Both concurrency-64 points are dominated by TP=1 at
concurrency 16 — lower per-GPU throughput *and* lower per-user rate. On this model and
these cards there is no operating point where 64 is the right answer.

### Replicate spread is far tighter than the mock suggests

Back-to-back replicates differed by **0.0–1.0%** (0.1% at concurrency 4). The mock agent
injects ±4%, which is now known to be pessimistic for grouped replicates on real hardware.
Useful calibration: at this repeatability a 2% difference between configurations is a
result, not noise. Note the caveat the spread carries — `grouped` measures repeatability
under near-identical conditions and says nothing about drift between sittings.

### Housekeeping behaved

Two engine loads for twelve runs, exactly as the plan predicted. Loads took 172 s (TP=1)
and 186 s (TP=2) — long enough that grouped ordering saved roughly twelve minutes over
interleaved. VRAM returned to 36 MiB between the two engines, and no `vllm serve` process
survived the sweep. Per-device utilization within the TP=2 group was balanced to 1–3%,
with memory within 0.1 GB across the pair.

`reset caches: none available` is expected here rather than a fault: prefix caching is off
in this configuration, so there is no prefix cache to carry between points.
