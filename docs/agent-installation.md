# Installing the agent on a GPU host

The agent is the only part of this system that touches a GPU. It is a `uv`-installable
Python package that starts `vllm serve`, runs `vllm bench serve` against it over loopback,
samples NVML and `/metrics` while that happens, and reports back to the control plane over
HTTP. There is no container runtime on this host and nothing is scheduled here — the GPU
host is the system under test.

**Requirements:** one or more NVIDIA GPUs with a working driver, a Python environment with
vLLM installed, `uv`, and a port the control host can reach (9110 by default).

---

## 1. Install

Install into the **vLLM environment**, from git:

```bash
source /path/to/vllm-env/.venv/bin/activate
uv pip install "git+https://github.com/er0080/vllm-benchmarking@v1.0.0rc5#subdirectory=packages/agent"
```

Pinned to a tag, so the host is on a known commit — `pip show` and `direct_url.json`
record which one, which is better provenance than a directory path. `@main` tracks the
tip instead, which is what a development host wants and a measurement host does not.

Two things about that command are deliberate.

**Into the vLLM environment**, because the agent invokes that environment's own `vllm`
binary. It adds nothing: vLLM's server is itself a FastAPI and uvicorn application using
pydantic, psutil and NVML, so every dependency is already present and only two small
pure-Python packages are installed.

**From git, not from a clone.** `vllmbench-protocol` is a workspace member, so inside a
checkout `uv pip install ./packages/agent` resolves it through `tool.uv.sources` and
installs it *editable* — a `.pth` file pointing back at the checkout. Nothing complains.
The agent then works until the clone is moved or deleted, at which point it dies with
`ModuleNotFoundError: vllmbench_protocol` on a machine with no source tree and no
explanation. Installing from git resolves the same workspace inside a throwaway clone and
installs both packages normally. CI proves this rather than trusting it: `agent-install`
installs from a throwaway clone, deletes it, and fails if anything in the resulting
environment still points at a directory.

If isolation is genuinely required — a shared host, an immutable environment — install the
agent elsewhere and set `VLLMBENCH_VLLM_BIN` to the absolute path of the `vllm`
executable. Do **not** solve it by putting the vLLM venv on `PATH`: that is invisible in
`ps`, silently lost across systemd units and reboots, and when wrong it yields a confusing
null version rather than an error.

## 2. Verify the install

```bash
python -c "import vllmbench_agent, vllmbench_protocol; print(vllmbench_protocol.__version__)"
test -z "$(ls "$VIRTUAL_ENV"/lib/python*/site-packages/_editable_impl_vllmbench_*.pth 2>/dev/null)" \
  && echo "self-contained" || echo "EDITABLE — reinstall from git"
```

Both lines should print. The second is the check that catches the trap above.

### Check what the install did to vLLM's environment

You installed the agent into the virtualenv vLLM lives in. That is the documented
deployment and it is what lets the agent invoke vLLM's own binaries — but it means the
agent's dependency resolution and vLLM's ceilings met in one place with nothing arbitrating
between them:

```bash
uv pip check
```

On the first real GPU host this said:

```
The package `vllm` requires `fastapi[standard]>=0.133.0,<0.137.0`,
but `0.141.1` is installed
```

vLLM worked anyway, which is luck rather than design. **This is the moment to act on it** —
the alternative is finding out during a forty-minute sweep, on the machine whose behaviour
you are trying to hold still.

The agent runs the same check itself, at startup and on every handshake with the control
plane, so you will also see it in the log:

```
WARNING this environment does not satisfy its own declared constraints (1 conflict)
WARNING   vllm 0.25.1 requires fastapi[standard]<0.137.0,>=0.133.0, but fastapi 0.141.1 is installed
```

It never refuses to start. A conflict is not proof that anything is broken, and an agent
that exited would take a working GPU host offline over a warning. What it does instead is
*record* it: every run measured here carries the status, and the host page shows the
conflict lines, so a number produced on an inconsistent environment can be recognised as
one later rather than blending in.

If it matters on your host, the way out is not to share the environment: point
`VLLMBENCH_VLLM_BIN` at vLLM's `vllm` binary and install the agent into a virtualenv of its
own. Section 3 covers that setting.

## 3. Configure

The agent reads its settings from the environment, prefixed `VLLMBENCH_`.

| Variable | Default | What it does |
| --- | --- | --- |
| `VLLMBENCH_TOKEN` | *required* | Shared secret with the control plane, minimum 8 characters. There is no default: an agent that starts with a guessable token is worse than one that refuses to start, because the failure is invisible until somebody else finds it. |
| `VLLMBENCH_PORT` | `9110` | Listening port. |
| `VLLMBENCH_HOST` | `0.0.0.0` | Listening address. The control plane is on another machine by design. |
| `VLLMBENCH_LOG_LEVEL` | `INFO` | |
| `VLLMBENCH_VLLM_BIN` | *(unset)* | Absolute path to `vllm`, when the agent is not installed in the vLLM environment. |
| `VLLMBENCH_MIN_FREE_DISK_BYTES` | `1073741824` | Refuse to start an engine or a benchmark below this. A disk with no room turns a forty-minute benchmark into a write error at the end of it. `0` disables the check. |
| `VLLMBENCH_TELEMETRY_INTERVAL_SECONDS` | `1.0` | Sampling period for `/metrics` and NVML. This is a knob on how much the measurement is disturbed; a request can override it per run. |

Generate the token on the control host and copy it to both sides:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Keep it in a file the agent's service reads, mode `600`, rather than on a command line
where `ps` can see it:

```bash
umask 077
cat > ~/.vllmbench-agent.env <<'EOF'
VLLMBENCH_TOKEN=paste-the-token-here
EOF
```

### The agent gets the environment of whatever starts it

This is the failure that costs the most time, and it is not specific to this agent.
Variables set in `~/.bashrc` reach interactive shells and nothing else. A service does not
get them. The one that matters here is `HF_HOME`: without it vLLM resolves models against
`~/.cache/huggingface` and re-downloads weights the host already has, which presents as a
sweep that hangs on its first model load.

Put anything the engine needs in the same env file as the token:

```bash
HF_HOME=/home/you/hf
```

## 4. Run it

As a systemd user service — survives logout, restarts on failure, and journals:

```ini
# ~/.config/systemd/user/vllmbench-agent.service
[Unit]
Description=vLLM benchmarking agent
After=network-online.target

[Service]
Type=exec
EnvironmentFile=%h/.vllmbench-agent.env
ExecStart=%h/vllm-env/.venv/bin/vllmbench-agent
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now vllmbench-agent
loginctl enable-linger "$USER"     # so it starts at boot, not at login
journalctl --user -u vllmbench-agent -f
```

`ExecStart` is the absolute path into the venv rather than a bare name, for the same
reason `VLLMBENCH_VLLM_BIN` exists: a service has no `PATH` worth relying on.

For a quick session, tmux does the same job without a unit file:

```bash
tmux new-session -d -s vllmbench-agent \
  'set -a; . ~/.vllmbench-agent.env; set +a; exec ~/vllm-env/.venv/bin/vllmbench-agent'
```

Either way, check it:

```bash
curl -s localhost:9110/health
```

## 5. Register it with the control plane

From the UI, **Hosts → Register**, or:

```bash
curl -X POST http://control-host:8000/api/hosts \
  -H 'Content-Type: application/json' \
  -d '{"name":"gpu-1","agent_url":"http://192.168.1.50:9110"}'
```

Use a LAN address. Registration performs the protocol handshake and reads the host's
facts — per-device GPU model and VRAM, driver, CUDA and vLLM versions — which become
provenance on every run that host produces. If the response is a connection error, the
port is not reachable from the control host; if it names an authentication failure, the
two sides hold different tokens — the agent answers 403 to a token it does not recognise
and 401 to a request carrying none.

## 6. Upgrade

Upgrade the agent and the control plane together. They version in lockstep and the
handshake refuses a mismatch, naming both versions — a stale agent fails at connect time
rather than producing subtly wrong data mid-sweep.

```bash
systemctl --user stop vllmbench-agent
source /path/to/vllm-env/.venv/bin/activate
uv pip install \
  --reinstall-package vllmbench-agent --reinstall-package vllmbench-protocol \
  "git+https://github.com/er0080/vllm-benchmarking@v1.0.0rc5#subdirectory=packages/agent"
systemctl --user start vllmbench-agent
```

Some form of reinstall flag is required: without one, `uv` sees a package version it
already has and does nothing, even though the tag moved — which leaves the old agent in
place and is exactly the stale agent the handshake exists to catch.

Use `--reinstall-package`, not `--reinstall`. The bare flag applies to the whole
resolution, so in a shared vLLM environment it reinstalls that environment's packages too
and can move their versions. This is the machine whose behavior a measurement depends on;
nothing the agent does should perturb it.

See [upgrading.md](upgrading.md) for the whole-deployment order of operations.

Upgrading vLLM itself is a separate decision and the agent does not manage it. A GPU host
running a different vLLM version from `VLLM_REFERENCE_VERSION` is recorded and warned
about, never blocked — comparing vLLM versions is a legitimate use of this tool.

## 7. Uninstall

```bash
systemctl --user disable --now vllmbench-agent
source /path/to/vllm-env/.venv/bin/activate
uv pip uninstall vllmbench-agent vllmbench-protocol
```

Nothing else on the host is touched. The agent's working directories live under the system
temp directory and are swept at startup; results live on the control host.

---

## When something is wrong

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: vllmbench_protocol` | Installed editable from a clone that has since moved. Reinstall from git. |
| Registration is refused as unauthorized | The two sides hold different tokens. The agent answers 403 to a wrong token, 401 to a missing one. |
| Registration times out | Port not reachable from the control host, or the agent bound to `127.0.0.1`. |
| Handshake refused, naming two versions | Agent and control plane are on different releases. Upgrade both. |
| `vllm_version: null` on the host facts | The agent cannot find `vllm`. It is not in this environment and `VLLMBENCH_VLLM_BIN` is unset. |
| Runs fail with `host_disk_full` | Below `VLLMBENCH_MIN_FREE_DISK_BYTES`. This is a refusal before work starts, not a failure at the end of one. |
| A run hangs on model load | Usually `HF_HOME` missing from the service environment, so vLLM is re-downloading weights. |
| VRAM still held after a sweep | Report it. Every `vllm serve` the agent starts, it kills, including on agent crash and restart; orphans are reaped at startup. |

The agent's logs are JSON lines carrying the run and sweep in flight, so a single run's
trail is a filter rather than a grep:

```bash
journalctl --user -u vllmbench-agent -o cat | jq 'select(.run_id == "...")'
```
