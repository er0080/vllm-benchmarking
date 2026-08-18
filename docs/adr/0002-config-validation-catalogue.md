# ADR 0002 — The vLLM argument catalogue is a captured artifact

- **Status:** accepted
- **Issue:** none — see CLAUDE.md, issues are optional through 0.8.0
- **Milestone:** 0.6.0
- **Supersedes:** nothing

> Accepted. The decisions below are binding; changing one requires a superseding ADR, not
> an edit to this file. Rejected alternatives are retained deliberately — they are the
> record of why the project is not doing the obvious other thing.

---

## Summary

The config validation engine checks candidate YAML against a **catalogue of arguments
dumped from a real vLLM's `argparse`**, shipped as data in `vllmbench_protocol`, and
re-derived from the live container by tier 2 on every PR.

Decided, in one line: **capture the argument set from the parser, ship it as versioned
data, fall back to the reference version when a host runs something we have not captured,
and say which was used.**

---

## Why not write the rules by hand

This is the same reasoning that governs `bench_result` and `metrics`, applied to a third
upstream surface, and it has already been paid for once: vLLM's published benchmarking
docs describe `--save-result` fields that vLLM 0.25.1 does not emit.

A hand-written catalogue encodes the author's belief about vLLM's option set, which is
exactly the belief under test. The failure mode is worse here than a wrong field name. A
validator that believes in an argument vLLM has dropped will pass a config that cannot
start — at the one moment somebody is trusting it not to. A validator that has never heard
of an argument vLLM added will reject a working config, which teaches the author to ignore
it, which is worse than having no validator at all.

The capture is not hypothetical protection. Of the arguments most commonly named in vLLM
tuning material, `swap-space` and `cuda-graph-sizes` **do not exist in 0.25.1**. Both would
have been in a hand-written list.

## Why the parser and not `--help`

The help output is prose, formatted for a terminal, and it reflows between releases. The
parser is the thing that actually accepts or rejects a config. `scripts/capture_serve_args.py`
reads `make_arg_parser()` directly and records each argument's type, choices, `nargs` and
aliases.

This has a cost: it reaches into vLLM internals, and `FlexibleArgumentParser` had already
moved from `vllm.utils` to `vllm.utils.argparse_utils` by 0.25.1. That breakage is loud —
the script fails to import — which is the right failure. Parsing help text would have kept
working while silently producing a worse catalogue.

## Why no agent endpoint, and no protocol bump

The roadmap assumed `validate_config` needed the agent to report what its vLLM accepts,
and therefore a new endpoint and protocol 6. It does not.

Every host already reports `vllm_version` as part of invariant 6 provenance. That is enough
to select a catalogue:

- A host whose version we have captured is checked exactly.
- A host running anything else is checked against the reference, and the result carries
  `exact_version_match: false`.

The fallback is not a degraded mode to be apologised for. The version policy in CLAUDE.md
makes benchmarking one vLLM version against another a first-class use, so a host ahead of
our captures is an expected situation rather than a misconfiguration. What matters is that
a clean result means something weaker in that case, and the caller is told so rather than
left to assume.

An agent endpoint that dumps the catalogue from the host's own vLLM remains worth building
— it would make every host exact. It is now an *enhancement*, not a prerequisite, and it no
longer blocks 0.6.0 behind redeploying the agent on a machine that is otherwise working.

## Why validation advises rather than blocks

`POST /api/configs/validate` stores nothing, and a config that fails it can still be
created.

The catalogue is a capture of one version. A host running something newer may legitimately
accept an argument this control plane has never heard of, and refusing to store that config
would make the framework unable to benchmark the release it is most interesting to
benchmark. Invariant 5 points the same way: validate, do not transform. Nothing here emits
YAML, reorders keys, fills defaults, or hands back a corrected config. Suggestions are
offered to the author, never applied behind them.

## Errors and warnings are different things

An **error** means `vllm serve` will refuse to start. A **warning** means it will start and
may not do what was meant — `tensor-parallel-size` written into a config the sweep will
overwrite, or a config with no `model`.

Collapsing them into "problems" would hide the distinction that matters, and the warning
kind is the more dangerous one: it produces a run that succeeds, charts, and answers a
different question than the one asked.

---

## Rejected alternatives

**Vendor a JSON Schema of vLLM's config.** Upstream does not publish one. Deriving and
maintaining it is the hand-written catalogue with extra steps.

**Import vLLM in the control plane and ask it directly.** Violates invariant 1: the control
plane has no GPU access and must not depend on a vLLM installation. It would also pin the
control plane to one vLLM version, which is precisely the thing the version policy exists
to avoid.

**Ask the agent at validation time.** Correct, and deferred rather than rejected — see
above. It requires a protocol bump and an agent redeploy to deliver a feature that works
without either.

**Validate only what the sweep planner touches.** Cheap, and misses the entire class this
exists for: `dtype: fp16`, a key set twice, a topology larger than the host.

**Reject unknown arguments as warnings rather than errors.** Tempting, since our catalogue
can lag upstream. Rejected because the overwhelmingly common cause of an unknown argument
is a typo, and a warning for a typo is a finding people scroll past. The escape hatch is
that validation does not block creation.
