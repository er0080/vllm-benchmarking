# Tuning playbook

The framework measures and displays; you decide. This is how to read what it displays and
what each reading implies about the next configuration to try.

For how to produce a measurement worth reading in the first place, see
[benchmark-procedure.md](benchmark-procedure.md). This document assumes you have one.

Every number here comes from a real sweep on this framework: Qwen3.5-9B at `max-model-len`
8192, `gpu-memory-utilization` 0.90, `max-num-seqs` 64, on 2× RTX 3090 with vLLM 0.25.1.
Tensor-parallel size 1 and 2 crossed with concurrency 4, 16 and 64, two replicates each.

---

## First: is this a result, or is it noise?

A point's replicate spread is the smallest difference you can claim to have measured.
Anything narrower than it is a story about noise.

Back-to-back replicates on that sweep agreed to within **0.0–1.0% on throughput** — at that
repeatability a 2% throughput difference between configurations is real. **Tail latency is
not that stable**: the same two replicates of TP=2 at concurrency 4 reported a p99 TTFT of
634 ms and 1246 ms. Nothing changed between them. A p99 over 128 prompts is an estimate
from roughly one request, and it moves accordingly.

So: **read medians for throughput, read p99 for shape, and never let a p99 decide between
two configurations on its own.**

The views chart the spread beside the median for this reason. If the spreads overlap, you
have not distinguished the points — you have run out of measurement, and the next step is
more replicates, not a different config.

---

## Pareto frontier — which configurations are worth keeping

*Analysis tab.*

**Axes:** per-GPU total throughput (tok/s/GPU) against per-user output rate
(1000 / mean TPOT — the generation speed one request actually experiences).

Those two axes are the tuning trade in its entirety. Throughput is what the operator buys;
per-user rate is what the user feels. Everything else is a way of explaining a movement
along them.

A point is **on the frontier** when nothing else beats it on both axes. Points off the
frontier are dominated: something is faster for the user *and* cheaper per card, so there
is no traffic pattern for which the dominated point is the right answer. That is the
strongest statement this framework makes, and it is the one to act on first — **delete
dominated configurations from the candidate set and stop re-running them.**

From the sweep:

| Point | tok/s/GPU | tok/s/user | |
| --- | --- | --- | --- |
| TP=2, concurrency 4 | 509 | 63.6 | frontier — fastest for one user |
| TP=1, concurrency 4 | 703 | 40.8 | frontier |
| TP=2, concurrency 16 | 997 | 32.3 | frontier |
| TP=1, concurrency 16 | 1796 | 29.0 | frontier — most throughput per card |
| TP=1, concurrency 64 | 1681 | 22.9 | dominated |
| TP=2, concurrency 64 | 1452 | 12.0 | dominated |

**Concurrency 64 is off the frontier at both widths.** Pushing load past the knee bought
nothing on either axis — it cost throughput *and* per-user speed. That is the single most
common tuning mistake this view catches.

**What to change next:** pick the frontier point whose per-user rate meets your latency
budget, and take the throughput that comes with it. If no frontier point meets the budget,
the frontier itself has to move — that is a configuration change (KV cache headroom,
`max-num-seqs`, tensor-parallel width, quantization), not a concurrency change.

---

## Response to load — where this configuration saturates

*Load tab.*

**Axes:** throughput and latency against offered load (concurrency or request rate).

Every serving configuration has a knee. Below it, added load buys throughput. Above it,
added load buys queueing and nothing else.

| Concurrency | TP=1 output tok/s | TP=1 p99 TTFT |
| --- | --- | --- |
| 4 | 140 | 0.68 s |
| 16 | 359 | 2.36 s |
| 64 | 336 | **20.8 s** |

The knee is at 16. Between 16 and 64 throughput *fell* 6% while p99 TTFT rose by a factor
of nine. Those 64 requests were not being served concurrently; they were standing in a
queue, and the queue is where the 20 seconds went.

**Read the curve, not the endpoint.** A throughput number quoted without the load it was
measured at is unusable — 336 tok/s sounds like a result and is, in fact, a configuration
being run 4× past its useful limit.

**What to change next:** hold load at the knee and change the configuration to move the
knee. If throughput is still climbing at the highest load you tried, you have not found the
knee yet — extend the sweep rather than concluding.

---

## The telemetry timeline — why it saturated

*Run detail page.*

A run detail page carries KV cache utilization, running and waiting queue depth, and
preemption counts, sampled every second alongside the benchmark. This is the view that
turns a shape into a cause, and it is why the framework samples at all.

The same three TP=1 runs:

| Concurrency | Peak KV cache | Mean running | Peak waiting | Preemptions |
| --- | --- | --- | --- | --- |
| 4 | 25% | 3.6 | 0 | 0 |
| 16 | **100%** | 11.5 | 7 | 0 |
| 64 | **100%** | 13.1 | 55 | **22** |

At concurrency 64 the engine held an average of 13 requests in flight out of 64 offered.
The other 51 were waiting, because the KV cache was full — and 22 requests that had already
started generating were *preempted* and had to be redone. The config carries the reason in
a comment: `max-model-len: 8192 # short on purpose: at TP=1 the weights leave ~4GB for KV`.

Read it against the same runs at TP=2, where the second card's memory goes almost entirely
to KV cache:

| Concurrency | Peak KV cache | Mean running | Preemptions |
| --- | --- | --- | --- |
| 64 (TP=1) | 100% | 13.1 | 22 |
| 64 (TP=2) | **25%** | **39.2** | 0 |

Three times as many requests actually in flight, from the same offered load.

**What to change next, by what the timeline shows:**

| Reading | Bottleneck | Next change |
| --- | --- | --- |
| KV cache at 100%, requests waiting, preemptions > 0 | KV cache capacity | Raise `gpu-memory-utilization`, lower `max-model-len`, quantize the KV cache, or add a card |
| KV cache low, requests waiting, GPU SM utilization high | Compute | The engine is working as hard as it can; only a wider topology or a smaller model helps |
| KV cache low, few requests running, SM utilization low | Not the server | Offered load, client, or `max-num-seqs` ceiling — check the workload before touching the config |
| Queue depth spiky rather than flat | Arrival pattern | A burstiness or request-rate effect, not a capacity one |

Preemptions above zero are worth treating as a hard signal. Work that was already done gets
thrown away and redone, so the throughput cost is larger than the count suggests.

---

## Tensor-parallel scaling — should this model get more cards

*Scaling tab.*

**Curve:** per-GPU throughput against tensor-parallel width, with efficiency relative to
the narrowest width measured.

Aggregate throughput is not comparable across widths — a TP=4 run trivially out-throughputs
a TP=1 run while potentially being far worse per device. Every throughput figure in this
framework carries both, and this view is built on the per-GPU one for that reason. Fed an
aggregate figure it would compute speedup and call it efficiency, reporting 2.0 for a
topology that merely kept up.

Scaling efficiency of TP=2 against TP=1, from the same sweep:

| Concurrency | Efficiency |
| --- | --- |
| 4 | 72.4% |
| 16 | **55.5%** |
| 64 | **86.4%** |

**Efficiency is not monotonic in load, and the shape is the answer.** At low load the
second card is mostly idle. At concurrency 16 it is at its worst — TP=1 is at its knee and
has not run out of KV cache yet, so the second card adds latency without adding capacity.
At 64 it is at its best, because that is exactly where TP=1 falls over.

**The second card earns its place where the first one runs out.** Which means the question
"should this model get more cards" cannot be answered at a single load, and a scaling
number quoted without one should be distrusted.

**What to change next:** if efficiency at your real load is below roughly 70%, the extra
cards are being spent on the wrong problem — look at the telemetry timeline for what the
narrow topology was actually short of. Two independent TP=1 servers may serve your traffic
better than one TP=2 server, and that is a supported comparison rather than a heresy.

---

## Per-device balance — is this run what it looks like

*Balance tab.*

**Chart:** SM utilization, memory, power and clocks for every device in the run,
attributed per device rather than averaged.

A host-level average destroys the signal that makes a tensor-parallel run diagnosable. This
view exists because a TP=4 run where one device sits at 60% while three sit at 95% is
telling you something specific, and the average is 87% either way.

Healthy, from the sweep at TP=2:

| Concurrency | GPU 0 SM | GPU 1 SM | Gap |
| --- | --- | --- | --- |
| 4 | 88.8% | 89.5% | 0.8% |
| 64 | 78.4% | 75.7% | 3.4% |

A few percent is normal. **A large gap means the run was not doing what it looked like** —
a device throttling, a device shared with something else, an interconnect bottleneck, or an
uneven split. Check this before believing any tensor-parallel result, and certainly before
recording one as a decision. Memory should be near-identical across devices; power and
clocks reveal a thermally limited card, which is a host problem rather than a config one.

---

## Compare two points — what actually differed

*Compare tab.*

Pick two points and read the config diff beside the metric diff. Use it when a result is
surprising: the usual explanation is that the two configurations differ in something other
than the axis under study, and this is the view that shows it in one place.

It is also where a decision gets recorded. When a configuration wins, annotate it with the
run that justifies it — the framework refuses a run of any other configuration, because a
citation that can point at the wrong evidence is worse than no citation. Then export the
config; the bytes are identical to what ran and go straight into production.

---

## The traps, condensed

- **Aggregate throughput flatters wide topologies.** TP=1 → TP=2 at concurrency 4 measured
  +44.8% aggregate and −27.6% per GPU. On aggregate alone it is a clear win; per device you
  bought latency with a card.
- **A better TPOT is not a better configuration.** At concurrency 64, TP=1's mean TPOT
  (43.6 ms) beat TP=2's (83.1 ms) — while serving half the throughput and a p99 TTFT of
  20.8 s against 9.5 s. TP=1 looked fast per token because it was only decoding 13 requests
  at a time; the other 51 were queued and are not in that number.
- **Peak throughput is usually past the frontier.** The highest aggregate number in a sweep
  is frequently a dominated point.
- **A p99 from one replicate is not a measurement.** See the top of this document.
- **Never compare across provenance.** Different GPU, different vLLM version, or a
  benchmark client that was not on loopback are different experiments. The framework groups
  and warns rather than silently overlaying; do not overrule it by exporting and merging.
