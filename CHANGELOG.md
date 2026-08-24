# Changelog

Generated from the Conventional Commit history by
`scripts/generate_changelog.py`. Do not edit by hand — run `make changelog`.

Squash merge makes each pull request title the permanent commit subject, so this
file is that history grouped by release. The reasoning behind a change lives in
its pull request body; [ROADMAP.md](https://github.com/er0080/vllm-benchmarking/blob/main/ROADMAP.md) narrates what
each release candidate was for and what it caught.

## 1.1.0 — 2026-08-23

### Features

- record the interconnect a run was measured over ([#118](https://github.com/er0080/vllm-benchmarking/pull/118))

### Fixes

- **ci:** a release sorts after its own release candidates ([#115](https://github.com/er0080/vllm-benchmarking/pull/115))

## 1.0.0 — 2026-08-23

### Features

- **ci:** generate CHANGELOG.md from the commit history ([#107](https://github.com/er0080/vllm-benchmarking/pull/107))
- **compose:** pull pinned published images instead of building ([#106](https://github.com/er0080/vllm-benchmarking/pull/106))

### Documentation

- the README describes standing up the stack from images ([#111](https://github.com/er0080/vllm-benchmarking/pull/111))
- every CI job gates the merge, and adding one is two changes ([#109](https://github.com/er0080/vllm-benchmarking/pull/109))
- record what the rc6 publish did ([#105](https://github.com/er0080/vllm-benchmarking/pull/105))

## 1.0.0rc6 — 2026-08-23

### Features

- **ci:** prove the published images pull without credentials ([#103](https://github.com/er0080/vllm-benchmarking/pull/103))
- **ci:** publish multi-arch images and the agent wheels from a v* tag ([#102](https://github.com/er0080/vllm-benchmarking/pull/102))

### Fixes

- **api:** the partial-run warning claimed a direction the data contradicts ([#98](https://github.com/er0080/vllm-benchmarking/pull/98))
- **agent:** a crashed engine is not a configuration problem ([#97](https://github.com/er0080/vllm-benchmarking/pull/97))

### Documentation

- record the rc5 hardware verification ([#93](https://github.com/er0080/vllm-benchmarking/pull/93))

## 1.0.0rc5 — 2026-08-23

### Features

- **protocol:** record speculation and dataset identity as provenance ([#89](https://github.com/er0080/vllm-benchmarking/pull/89))
- **protocol:** flatten speculative decoding metrics onto columns ([#84](https://github.com/er0080/vllm-benchmarking/pull/84))

### Fixes

- **agent:** cache resets between sweep points have never happened ([#88](https://github.com/er0080/vllm-benchmarking/pull/88))
- **api:** a workload's extra_args now reaches the agent, and is part of its identity ([#81](https://github.com/er0080/vllm-benchmarking/pull/81))

### Tests

- **ci:** re-derive the speculative field names from a live server ([#90](https://github.com/er0080/vllm-benchmarking/pull/90))

## 1.0.0rc4 — 2026-08-22

### Fixes

- **agent:** a virtualenv that no longer adds up now says so ([#78](https://github.com/er0080/vllm-benchmarking/pull/78))
- **mcp:** two tools described themselves inaccurately ([#76](https://github.com/er0080/vllm-benchmarking/pull/76))

### Documentation

- the repository states its terms ([#77](https://github.com/er0080/vllm-benchmarking/pull/77))

## 1.0.0rc3 — 2026-08-22

### Fixes

- **mcp:** an unknown axis is refused, and the surface documents itself ([#73](https://github.com/er0080/vllm-benchmarking/pull/73))

## 1.0.0rc2 — 2026-08-18

### Fixes

- the install stops asking for things it ignores, and says when a secret is fake ([#68](https://github.com/er0080/vllm-benchmarking/pull/68))
- **web:** the UI stopped working whenever the api container moved ([#67](https://github.com/er0080/vllm-benchmarking/pull/67))

### Documentation

- a quick start that tells you what is about to happen ([#70](https://github.com/er0080/vllm-benchmarking/pull/70))

## 1.0.0rc1 — 2026-08-18

### Features

- the flattening layer was hiding a run that measured nothing ([#42](https://github.com/er0080/vllm-benchmarking/pull/42))
- bound what grows, and never the measurements ([#41](https://github.com/er0080/vllm-benchmarking/pull/41))
- logs you can query, and that cannot carry a secret out ([#40](https://github.com/er0080/vllm-benchmarking/pull/40))
- every failure names itself, and timeouts stop being hard-coded ([#39](https://github.com/er0080/vllm-benchmarking/pull/39))
- import upstream sweeps, CSV/JSON export, shareable reports ([#37](https://github.com/er0080/vllm-benchmarking/pull/37))
- configuration management — editor, lineage, export, justification ([#35](https://github.com/er0080/vllm-benchmarking/pull/35))
- **api:** config validation against a captured vLLM argument catalogue ([#34](https://github.com/er0080/vllm-benchmarking/pull/34))
- **api:** MCP resources, context economy, and a duration estimate that decomposes ([#33](https://github.com/er0080/vllm-benchmarking/pull/33))
- **api:** MCP write tools, and record which interface asked ([#31](https://github.com/er0080/vllm-benchmarking/pull/31))
- **api:** MCP read surface mounted at /mcp ([#30](https://github.com/er0080/vllm-benchmarking/pull/30))
- saved analysis views, completing milestone 0.5.0 ([#29](https://github.com/er0080/vllm-benchmarking/pull/29))
- **web:** one filter selection shared by every analysis view ([#28](https://github.com/er0080/vllm-benchmarking/pull/28))
- **analysis:** compare two points side by side, with a diff of the config text ([#27](https://github.com/er0080/vllm-benchmarking/pull/27))
- **analysis:** compare per-device utilization to expose tensor-parallel imbalance ([#26](https://github.com/er0080/vllm-benchmarking/pull/26))
- **web:** chart response to load — latency percentiles and throughput saturation ([#25](https://github.com/er0080/vllm-benchmarking/pull/25))
- **analysis:** chart tensor-parallel scaling, and what efficiency is measured against ([#24](https://github.com/er0080/vllm-benchmarking/pull/24))
- chart the Pareto frontier, and the comparable-run-set query beneath it ([#23](https://github.com/er0080/vllm-benchmarking/pull/23))
- **web:** author, watch and cancel sweeps from the browser ([#22](https://github.com/er0080/vllm-benchmarking/pull/22))
- reuse the engine across a sweep, and cancel a run in flight ([#21](https://github.com/er0080/vllm-benchmarking/pull/21))
- **api:** author sweeps as a materialized plan ([#20](https://github.com/er0080/vllm-benchmarking/pull/20))
- telemetry — sample the engine and every GPU during a run, and chart it ([#19](https://github.com/er0080/vllm-benchmarking/pull/19))
- provenance from the engine, device attribution from NVML ([#15](https://github.com/er0080/vllm-benchmarking/pull/15))
- single run end to end ([#14](https://github.com/er0080/vllm-benchmarking/pull/14))
- **agent:** vLLM server lifecycle, benchmarking, and orphan reaping ([#13](https://github.com/er0080/vllm-benchmarking/pull/13))
- agent, mock agent and GPU host registry ([#10](https://github.com/er0080/vllm-benchmarking/pull/10))
- compose stack and results schema ([#9](https://github.com/er0080/vllm-benchmarking/pull/9))

### Fixes

- **docs:** the agent upgrade must not reinstall the vLLM environment ([#61](https://github.com/er0080/vllm-benchmarking/pull/61))
- the integration suite no longer defaults to the results database ([#58](https://github.com/er0080/vllm-benchmarking/pull/58))
- **web:** form controls stretched to 14rem tall by a row rule in a column ([#36](https://github.com/er0080/vllm-benchmarking/pull/36))
- make the recommended agent deployment actually work, and verify it on real hardware ([#18](https://github.com/er0080/vllm-benchmarking/pull/18))
- check the protocol version before validating the payload ([#17](https://github.com/er0080/vllm-benchmarking/pull/17))
- install into the vLLM venv, and benchmark the served model name ([#16](https://github.com/er0080/vllm-benchmarking/pull/16))

### Documentation

- read every document as a set, and fix what disagreed ([#57](https://github.com/er0080/vllm-benchmarking/pull/57))
- a quick start that was actually run from nothing ([#53](https://github.com/er0080/vllm-benchmarking/pull/53))
- the upgrade path, and what this release cannot do ([#52](https://github.com/er0080/vllm-benchmarking/pull/52))
- a tuning playbook, read off the sweep we already ran ([#51](https://github.com/er0080/vllm-benchmarking/pull/51))
- an agent installation guide written from a clean install ([#50](https://github.com/er0080/vllm-benchmarking/pull/50))
- issue-driven change management starts at 0.10.0, not 0.9.0 ([#38](https://github.com/er0080/vllm-benchmarking/pull/38))
- record the first real sweep, and what it said about the framework ([#32](https://github.com/er0080/vllm-benchmarking/pull/32))
- track code paths awaiting GPU verification ([#12](https://github.com/er0080/vllm-benchmarking/pull/12))
- defer the issue-per-PR requirement to 0.9.0 ([#7](https://github.com/er0080/vllm-benchmarking/pull/7))
- graduate MCP proposal to ADR and amend roadmap ([#6](https://github.com/er0080/vllm-benchmarking/pull/6))
- propose MCP server for agent-driven benchmarking ([#4](https://github.com/er0080/vllm-benchmarking/pull/4))
- multi-GPU scope, mermaid architecture, issue-driven workflow ([#3](https://github.com/er0080/vllm-benchmarking/pull/3))
- establish project README, CLAUDE.md and roadmap to 1.0.0

### CI

- tiers 1 and 2, and align to the real vLLM contract ([#11](https://github.com/er0080/vllm-benchmarking/pull/11))
