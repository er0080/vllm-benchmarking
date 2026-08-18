# Standard work: running a benchmark you can defend

The repeatable procedure for producing a measurement somebody will act on. It exists
because the expensive mistakes in this work are not wrong code — they are a sweep that ran
correctly and answered a different question than the one asked.

Written from the first real sweep this framework ran; the numbers quoted are from
[hardware-verification.md](hardware-verification.md).

---

## Before you start: what makes a result comparable

Four facts partition every result set, and results from different partitions cannot be
read as one series no matter how similar the configs look:

| Fact | Why it partitions |
| --- | --- |
| GPU host and model | Different silicon, different everything |
| vLLM version | A release can move throughput more than a config change |
| Benchmark client location | A remote client puts network RTT inside TTFT and ITL |
| Tensor-parallel width | Aggregate throughput is not comparable across widths |

The framework enforces the first three by refusing to chart across them. The fourth it
handles by reporting **per-GPU throughput alongside aggregate**, and defaulting every
comparison to per-GPU.

**Know which question you are asking before you author the matrix.** "Which config is
fastest" and "how many cards should this model get" are different sweeps.

---

## 1. Quiet the host

Anything running on the GPU host during a sweep is inside the measurement.

- No other process holding VRAM. Check `nvidia-smi` reads near-idle before starting; a
  clean host on 2× RTX 3090 sits at ~36 MiB.
- No CI runner, agent, or scheduled job on that machine. This is deliberate and
  documented in CLAUDE.md: the GPU host is the system under test.
- The benchmark client runs on the GPU host over loopback. This is not a convenience —
  it keeps network RTT out of TTFT and ITL, and a run measured any other way is recorded
  as such and never charted beside a loopback one.

## 2. Author the config, and check it before it costs anything

Use the Configs tab or `POST /api/configs/validate`. Validation is free; a bad config is
discovered when `vllm serve` exits, which on a large model is several minutes into a sweep
that has already claimed the host.

It catches the mistakes that actually happen: `dtype: fp16` (vLLM wants `float16`), a
tensor-parallel size larger than the host has cards, a setting spelled with the wrong
separator, and a key set twice — which YAML resolves silently to the last one.

**Leave the comments in.** The stored YAML is what runs, byte for byte, and six months
later `# 0.95 is risky with NCCL buffers on 24GB cards` is the most valuable line in the
file.

## 3. Size the matrix by engine loads, not by run count

Most of a sweep's wall clock is model loading, not benchmarking. On the first real sweep,
two engine loads accounted for 358 of 1180 seconds while the ten remaining runs shared the
other 655.

The authoring response returns **`engine_starts`** alongside the run count. That is the
number that predicts the duration.

- Order replicates **grouped** unless you have a reason not to. Grouped runs every
  replicate of a point before moving on, so a config is loaded once. Interleaved reloads
  the engine between replicates — on the first sweep that would have cost roughly twelve
  extra minutes for twelve runs.
- Grouped replicates measure *repeatability* under near-identical conditions. They
  understate the variance between separate sittings, and the spread is labelled with which
  it is.

## 4. Include at least two replicates

Without a spread there is no way to tell a result from noise.

Back-to-back replicates on real hardware differed by **0.0–1.0%**. The mock agent injects
±4%, so the dev environment is pessimistic — do not calibrate your expectations on it.

At that repeatability a 2% difference between configs is a result. A difference smaller
than a point's own spread is not, and every view and export carries the spread beside the
median so that judgement can be made.

## 5. Sweep the axis you are asking about

- **Concurrency** answers "where does this configuration saturate".
- **Tensor-parallel size** answers "how many cards should this model get". It is a
  first-class axis: the planner derives one config variant per width, and each is stored
  as an ordinary content-addressed config.

Do not sweep both against a wide config matrix on the first attempt. The product is a
Cartesian one, and the framework refuses a matrix above 500 runs precisely because it is
easy to author days of GPU time by accident.

## 6. Read the frontier, not the maximum

The primary view is per-user output tok/s against per-GPU total tok/s. Points on the
frontier are the ones nothing else beats on both axes; everything else is dominated and
there is no reason to run it.

Two traps this framework exists to surface:

- **Aggregate throughput flatters wide topologies.** Going TP=1 → TP=2 at concurrency 4
  measured **+44.8% aggregate and −27.6% per GPU**. On aggregate alone it looks like a
  clear win; per device it is a trade — you bought latency with a card.
- **Scaling efficiency is not monotonic.** The same sweep measured 72.4% at concurrency 4,
  55.5% at 16, and 86.4% at 64 — because TP=1 peaks at 16 and *degrades* at 64 once KV
  cache runs out. The second card earns its place exactly where the first runs out.

Check the device-balance view before believing a tensor-parallel result. A 1–3% gap
between devices is healthy; a large one means the run was not doing what it looked like.

## 7. Record why, then export

A configuration on its own says what was set, never why.

1. **Annotate the winner** with the run that justifies it. The framework refuses a run of
   any other configuration — a citation that can point at the wrong evidence is worse than
   no citation.
2. **Export the config** (Configs → Export). The bytes are identical to what ran, so it
   can go straight into production.
3. **Export the result set** (CSV or JSON) or **share the sweep report** (markdown). Every
   exported row carries its own provenance and population, because a file gets read by
   people who cannot see the filters that produced it.

## 8. Confirm the host is clean

After any sweep, normal or cancelled: no surviving `vllm serve` process, and VRAM back to
idle. Orphaned processes holding VRAM are a serious failure mode, and the next sweep on
that host will measure the consequences rather than the configuration.

---

## Importing results measured elsewhere

`vllm bench sweep serve` output can be imported, but its files carry **no provenance at
all** — no vLLM version, GPU model, driver, host, or device count. You will be asked to
declare those, and the import is refused without them rather than defaulted, because a
default there is a fabricated provenance column. Per-GPU throughput cannot be derived
without a device count at all.

Imported runs are marked permanently, and any comparison that groups them with measured
runs says so. See [ADR 0003](adr/0003-importing-upstream-sweeps.md).
