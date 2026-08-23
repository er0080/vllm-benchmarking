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
uv pip install "git+https://github.com/er0080/vllm-benchmarking@v1.0.0rc5#subdirectory=packages/agent"
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

`reset caches: none available` was read at the time as expected rather than a fault, on the
grounds that prefix caching was off in this configuration so there was nothing to carry
between points. That explanation was true and the conclusion was wrong: the endpoint was
returning 404 on every configuration, including ones with prefix caching on. See
[Speculative decoding](#verified--2026-08-23-speculative-decoding-and-two-bugs-under-it)
below, and issue #87.

---

## Verified — 2026-08-23, speculative decoding, and two bugs under it

The first sweep run for a reason other than testing the framework: choosing a serving
configuration for a Jupyter notebook coding assistant on `Qwen3.8-27B-FP8`. Twenty-four
runs, four multi-token-prediction depths against two workloads, three replicates each,
authored entirely through the MCP surface so every run carries `initiated_by: mcp`.

### The question

Qwen3.8-27B-FP8 ships MTP weights — `text_config.mtp_num_hidden_layers = 1` — so
speculative decoding is available without a separate draft model. Whether to use it, and
how deep to draft, is a measurement rather than a matter of opinion: acceptance depends on
how predictable the output is, which depends on the traffic.

Two workloads were built from `blazedit`, split by edit distance, because that is the axis
that decides how much a coding assistant is *copying* rather than composing:

- **notebook edit — high copy** (distance 0.0–0.2): small edits to existing code, where
  most of the output already appears in the prompt
- **notebook author — low copy** (distance 0.6–1.0): substantially new code

Both at `max-num-seqs: 2`, TP=2, 8 prompts, 512 output tokens.

### Results

Medians over replicates. `per user` is 1000 / mean TPOT; `per GPU` is the aggregate
throughput divided by the two devices the runs held.

**notebook edit — high copy**

| depth | accept % | tokens/step | TPOT ms | per user | per GPU | TTFT ms | emission gap ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off | — | — | 25.24 | 39.6 | 60.8 | 341 | 25.0 |
| 1 | 88.6 | 1.89 | 18.53 | 54.0 | 81.2 | 364 | 33.9 |
| 2 | 81.7 | 2.63 | 16.92 | 59.1 | 85.8 | 382 | 42.6 |
| 3 | 72.0 | 3.16 | 14.43 | 69.3 | 101.0 | 404 | 44.1 |

**notebook author — low copy**

| depth | accept % | tokens/step | TPOT ms | per user | per GPU | TTFT ms | emission gap ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off | — | — | 24.91 | 40.2 | 39.0 | 127 | 25.0 |
| 1 | 82.1 | 1.82 | 18.59 | 53.8 | 51.4 | 168 | 33.9 |
| 2 | 71.9 | 2.44 | 17.73 | 56.4 | 57.4 | 189 | 42.6 |
| 3 | 61.6 | 2.85 | 15.58 | 64.2 | 63.8 | 198 | 44.1 |

**+75% per-user throughput on editing, +60% on authoring**, at depth 3, against no
speculation. Every adjacent pair's replicate ranges are disjoint — at depth 2 the edit
workload spans 57.0–61.3 tok/s and at depth 3 it spans 68.3–74.4 — so the ordering is a
result rather than a spread. Depth 3 is the only point on the Pareto frontier.

(The depth-3 edit point carries four replicates rather than three: an earlier single-run
sweep measured the same configuration against the same workload, and content addressing
correctly pools it with the other three rather than treating it as a separate point.)

### Acceptance rate falls with depth, and it does not matter

The tempting summary is "depth 1 has the best acceptance rate, 88.6%". Depth 1 is the
worst-performing speculative arm in every column that describes speed.

Acceptance *rate* is per drafted token and necessarily decays with depth — each additional
position is conditioned on the last one having been right. Acceptance *length*, the tokens
gained per drafting step, rises anyway: 1.89 → 2.63 → 3.16. That is the figure the speed-up
is proportional to, and the two move in opposite directions.

This is why both are stored as columns. A framework recording only the rate would have
pointed at the worst arm and looked well-calibrated doing it.

### The payoff had not turned over at depth 3

Marginal gain per additional draft token, on editing: +14.4 tok/s, then +5.1, then +10.2.
Decelerating but not reversing, and the third step is larger than the second. Depths 4–6
are not measured, so the point at which drafting stops paying for itself on this model and
these cards is unknown — an open question rather than a conclusion.

### The high-copy workload benefits more, as predicted, and TTFT does not follow

Editing accepts more than authoring at every depth (88.6% vs 82.1% at depth 1; 72.0% vs
61.6% at depth 3), which is what "the output is already in the prompt" should produce.

TTFT moves the other way and by more than speculation explains: 341 ms editing against
127 ms authoring at baseline. That is the prompt length, not the drafter — the high-copy
prompts are longer.

Speculation adds TTFT, and the cost **grows with depth** rather than being a fixed
overhead paid once:

| depth | edit | author |
| --- | --- | --- |
| off | 341 ms | 127 ms |
| 1 | 364 (+23) | 168 (+41) |
| 2 | 382 (+41) | 189 (+62) |
| 3 | 404 (+63) | 198 (+71) |

Roughly 20 ms per additional speculative token on editing. Small against the throughput
it buys, and worth knowing for an interactive assistant, where the first token is the part
a person is waiting on: depth 3 is 18% slower to start and 75% faster thereafter.

### Two bugs, both found by using the framework rather than reading it

**Inter-token latency measures a different quantity under speculation** (issue #86). The
emission-gap column above rises 25.0 → 44.1 ms across the same arms whose per-user
throughput rises 39.6 → 69.3 tok/s. Both are correct: 44.1 ms divided by 3.16 accepted
tokens is 14.0 ms per token, which is that arm's measured TPOT of 14.4. A chart with both
on one ITL axis shows the fastest configuration as 76% worse and says nothing about why.

Not fixed by dropping the metric — chunked delivery of three tokens every 44 ms is a real
property of a streaming UI, and for an interactive assistant it is worth measuring. Fixed by
naming it *emission gap*, and by warning when a comparison group mixes speculation settings.

**Cache resets between sweep points had never happened** (issue #87). `/reset_prefix_cache`
is gated behind `VLLM_SERVER_DEV_MODE`, which the agent did not set, so every reset returned
404 — and the supervisor treated a 404 as "this vLLM version does not have that endpoint".
Measured on this host: 404 without the variable, 200 with it.

It cost this sweep nothing, and the database is what says so rather than hope.
`prefix_cache_hits_total` is 0 across all 8091 queries in it: the prompts are long and
distinct and `max-num-seqs` is 2, so nothing was ever reusable. A sweep over a dataset with
shared prefixes — a system prompt, a repeated notebook preamble — would have had its later
points inflated by its earlier ones, and the ordering of the matrix would have chosen the
winner. Storing counters rather than a sampled hit rate is what made that answerable at all.

### Speculation was not recorded as provenance, and now is

Grouping these results by depth required regexing the *name* of each configuration, which
is precisely what invariant 8 forbids for parallelism topology and forbids here for the
same reason: a config saying `num_speculative_tokens: 3` is not proof the engine drafted
three tokens. If the MTP head had failed to load and vLLM had carried on without it, the
configuration would still say 3.

From protocol 7 the agent reads `speculative_method` and `speculative_tokens` from the
engine's own `/server_info` — an endpoint behind the same `VLLM_SERVER_DEV_MODE` gate as the
cache resets, so one fix opened both. Verified on this host across both states: speculating
reports `{"method": "ngram", "num_speculative_tokens": 3}`, and not speculating reports the
key present and `null`, which is what keeps "the engine says no" distinct from "nobody
asked".

### What this says about the framework

Three of the four defects this sweep produced sat at a seam between layers that each worked
correctly and were each tested. The reset call was correct and the endpoint was absent; ITL
was flattened correctly and meant something else; the config recorded speculation and the
run did not. None would have been found by more unit tests of the parts.

The argument for tier 2 is that a fake can only confirm our own assumptions. This is the
argument for using the thing: an integration test can only confirm the assumptions somebody
thought to write down.

---

## Verified — 2026-08-23, 1.0.0rc5 on real hardware

The release that carries the three fixes above, checked on the host that motivated them.
Agent upgraded rc4 → rc5 with the scoped reinstall (exactly two packages moved; the vLLM
environment did not), then a two-run sweep — one speculative arm, one baseline, same
workload — authored through MCP.

| Item | Result |
| --- | --- |
| Protocol handshake after the bump | ✅ Agent 1.0.0rc5 / protocol 7, control plane the same, host reports healthy |
| Cache resets actually happen | ✅ `reset caches: /reset_prefix_cache, /reset_mm_cache, /reset_encoder_cache` on both engine loads, where every previous sweep logged `none available` |
| Speculation read from the engine | ✅ `mtp` at depth 3 on the speculative arm, `none` at 0 on the baseline |
| Dataset identity | ✅ `sha256:00f3871d786b…:13350` |
| Environment status recorded on the run | ✅ `conflicts`, the fastapi ceiling divergence that has been there since the first install |
| Mixed-speculation warning | ✅ Fired on a comparison spanning the two arms |

### The dataset hash checks itself

The workload's file is `notebook-edit-00f3871d786b.json`, named by its own content hash
before any of this existed — a workaround for the gap issue #82 was filed about. The agent
computed `sha256:00f3871d786bd7487ff5e1cb3b9c6b11152f31427c4f2080f1cadda30be8953f:13350`.
The filename prefix and the recorded identity agree, from two independent sources.

### The cache fix did not move the numbers, exactly as predicted

This is the check worth having done. Fixing a reset that never happened *could* have changed
every measurement taken before it, which would have invalidated the MTP sweep. Against the
same configuration and workload as yesterday:

| | MTP sweep, 01:20 UTC | rc5 check, 15:40 UTC |
| --- | --- | --- |
| | median of 4 replicates | 1 run |
| per-user output | 69.3 tok/s | 69.3 |
| emission gap (median) | 44.1 ms | 44.1 |
| acceptance length | 3.16 | 3.16 |
| acceptance rate | 72.0% | 72.0% |
| baseline emission gap | 25.0 ms | 25.0 |

A single run against a median of four is not a spread comparison, and the agreement to three
significant figures is partly luck. What it does rule out is a shift large enough to change
any conclusion drawn from that sweep.

Which is what the analysis in issue #87 said would happen: `prefix_cache_hits_total` was 0
across all 8091 queries in that sweep, so there was no carryover to remove. Predicted from
stored counters, then confirmed by measurement rather than assumed.

### The warning earns its place immediately

Asked for a Pareto view with the emission gap on one axis, the two arms come back with
**both on the frontier** — the baseline "wins" on emission gap, 25.0 ms against 44.1, while
being 43% slower per user. That is the exact reading the warning exists to prevent, and it
appeared on the first comparison anyone made:

> mixes mtp depth 3, no speculation; Emission gap median, Emission gap p99 count the wait
> between emissions, and a speculative emission carries several tokens — so those figures
> rise with depth even as generation gets faster. Compare speed with TPOT or per-user
> output rate.
