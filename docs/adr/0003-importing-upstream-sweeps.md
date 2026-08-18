# ADR 0003 — Imported results declare the provenance their files cannot carry

- **Status:** accepted
- **Issue:** none — see CLAUDE.md, issues are optional through 0.8.0
- **Milestone:** 0.8.0
- **Supersedes:** nothing

> Accepted. The decisions below are binding; changing one requires a superseding ADR, not
> an edit to this file. Rejected alternatives are retained deliberately — they are the
> record of why the project is not doing the obvious other thing.

---

## Summary

`vllm bench sweep serve` output can be imported, but **only alongside provenance the
operator states explicitly**, and every imported run is marked as imported for the rest of
its life.

Decided, in one line: **the file says what was measured; the operator says what measured
it; the database records which of the two each fact came from.**

---

## What the files actually contain

Captured from a real run of `vllm bench sweep serve` against vLLM 0.25.1, not read from
documentation:

```
<output_dir>/<experiment>/SERVE--max_num_seqs=4-BENCH--max_concurrency=2-num_prompts=8/
    run=0.json
    run=1.json
    summary.json
```

Two findings, neither of which the `--dry-run` output or the docs mention:

1. **`summary.json` is exactly the list of the `run=N.json` payloads** — same keys, no
   aggregation, no extra fields. It is a convenience file, not a different contract.
2. **There is no provenance anywhere in any of it.** No vLLM version, no GPU model, no
   driver, no host, no device count, no tensor-parallel size. The full key set is the
   `--save-result` payload plus `run_number` and the `SERVE--` overrides.

The serve overrides *are* recoverable — they appear both in the directory name and as
top-level keys in each run JSON — but only the ones passed through `--serve-params`.
Anything in the fixed `--serve-cmd` is unrecoverable, which in practice includes the model
and very often `--tensor-parallel-size`.

## Why this is a problem and not an inconvenience

Invariant 6: *a run that cannot state what produced it is not a valid result*. An imported
run cannot state any of it.

Invariant 8 makes it sharper. Every throughput figure carries a per-GPU value, comparison
views default to per-GPU, and **per-GPU cannot be computed without a device count.** A
sweep imported without one is not merely under-documented; its headline number cannot be
derived at all.

So an importer that accepted a directory and nothing else would have to invent a GPU
count, or store aggregate-only runs that no comparison view can honestly show. Both are
worse than refusing.

## The decision

**Required from the operator**, because the files cannot supply them: the GPU host, GPU
model, vLLM version, device count and tensor-parallel size, and where the benchmark client
ran. The import is refused without them rather than defaulted — a default here is a
fabricated provenance column, which is the specific failure this project exists to avoid.

**Recorded as declared, not observed.** `Run.imported_from` names the source, and it is
never null for an imported run. A GPU model that NVML reported and one that a person typed
are different epistemic objects, and a chart that cannot tell them apart will eventually
be asked to explain a discrepancy nobody can resolve.

**Charted, but never silently.** Imported runs are real measurements and belong in the
analysis — that is the entire point of an importer. But any comparability group containing
them carries a warning, in keeping with the existing rule: group or warn, never silently
overlay.

**Reconstructed configs are labelled as reconstructed.** Invariant 5 says what is stored is
what was passed to `vllm serve --config`. For an imported run that file never existed — the
server was configured by CLI flags, and we can see only the swept subset. The config we
store is therefore a faithful transcription of the overrides we can observe and *not* the
complete server configuration. Its name and notes say so. Pretending otherwise would put a
config in the database that claims to be runnable and is not.

## Rejected alternatives

**Infer the GPU count from throughput.** Requires knowing per-GPU throughput, which is the
thing being inferred.

**Default the device count to 1.** Silently halves or doubles every per-GPU figure
depending on the truth, and produces numbers that look plausible — the exact shape of the
per-device attribution bug this project already found once.

**Import as synthetic.** Wrong in the other direction. These are real measurements of real
hardware; invariant 7 is about runs produced by the mock agent or CPU backend, and
overloading it would make "synthetic" mean two unrelated things.

**Refuse to import at all.** Defensible, and rejected because the framework should be able
to ingest a result set produced before it existed — which is most of them.

**Store only `summary.json`.** It is the same data, but the per-run files carry
`run_number`, which is what makes replicates distinguishable and the spread computable.
