import { useCallback, useEffect, useRef, useState } from "react";

import { api, configsApi, runsApi, workloadsApi } from "./api";
import { TelemetryTimeline } from "./TelemetryTimeline";
import { failureLabel } from "./types";
import type { Config, Host, Run, RunStatus, Workload } from "./types";

const TERMINAL: RunStatus[] = ["succeeded", "failed", "cancelled"];
const POLL_MS = 2000;

const DEFAULT_YAML = `model: facebook/opt-125m
max_num_seqs: 64
gpu_memory_utilization: 0.9
`;

function fmt(value: number | null | undefined, digits = 1, suffix = ""): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`badge status-${status}`}>{status}</span>;
}

function RunRow({ run, onSelect }: { run: Run; onSelect: (id: string) => void }) {
  const s = run.summary;
  return (
    <tr onClick={() => onSelect(run.id)} className="clickable">
      <td>
        <StatusBadge status={run.status} />
      </td>
      <td>{new Date(run.queued_at).toLocaleTimeString()}</td>
      <td>{run.config_hash.slice(0, 8)}</td>
      <td>TP{run.tensor_parallel_size}</td>
      {/* Per-GPU, not aggregate: aggregate throughput is not comparable across
          different parallelism topologies. */}
      <td>{fmt(s?.output_token_throughput_per_gpu, 0)}</td>
      <td>{fmt(s?.ttft_ms_p99, 0)}</td>
      <td>{fmt(s?.itl_ms_median, 1)}</td>
      <td>{run.is_synthetic ? <span className="badge badge-synthetic">synthetic</span> : ""}</td>
    </tr>
  );
}

function RunDetail({ run, onClose }: { run: Run; onClose: () => void }) {
  const s = run.summary;
  return (
    <div className="card">
      <div className="host-head">
        <strong>Run {run.id.slice(0, 8)}</strong>
        <StatusBadge status={run.status} />
        {run.is_synthetic && (
          <span className="badge badge-synthetic">synthetic · {run.synthetic_source}</span>
        )}
        <span className="spacer" />
        <button onClick={onClose}>Close</button>
      </div>

      {run.is_synthetic && (
        <p className="notice">
          Produced by a stand-in, not real hardware. These numbers are not measurements.
        </p>
      )}

      {run.error && (
        <p className="error" style={{ whiteSpace: "pre-wrap" }}>
          {run.failure_kind && (
            <>
              <strong>{failureLabel(run.failure_kind)}</strong>
              {"\n\n"}
            </>
          )}
          {run.error}
        </p>
      )}

      {s && (
        <>
          <h3>Throughput</h3>
          <dl className="facts">
            <div className="fact">
              <dt>Output tok/s per GPU</dt>
              <dd>{fmt(s.output_token_throughput_per_gpu, 1)}</dd>
            </div>
            <div className="fact">
              <dt>Output tok/s total</dt>
              <dd>{fmt(s.output_token_throughput_tok_sec, 1)}</dd>
            </div>
            <div className="fact">
              <dt>Requests/s</dt>
              <dd>{fmt(s.request_throughput_req_sec, 2)}</dd>
            </div>
            <div className="fact">
              <dt>Requests</dt>
              <dd>
                {s.successful_requests ?? "—"}
                {s.failed_requests ? ` (${s.failed_requests} failed)` : ""}
              </dd>
            </div>
          </dl>

          <h3>Latency</h3>
          <div className="devices">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Mean</th>
                  <th>Median</th>
                  <th>p99</th>
                  <th>Std</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>TTFT (ms)</td>
                  <td>{fmt(s.ttft_ms_mean)}</td>
                  <td>{fmt(s.ttft_ms_median)}</td>
                  <td>{fmt(s.ttft_ms_p99)}</td>
                  <td>{fmt(s.ttft_ms_std)}</td>
                </tr>
                <tr>
                  <td>TPOT (ms)</td>
                  <td>{fmt(s.tpot_ms_mean, 2)}</td>
                  <td>{fmt(s.tpot_ms_median, 2)}</td>
                  <td>{fmt(s.tpot_ms_p99, 2)}</td>
                  <td>{fmt(s.tpot_ms_std, 2)}</td>
                </tr>
                <tr>
                  <td>ITL (ms)</td>
                  <td>{fmt(s.itl_ms_mean, 2)}</td>
                  <td>{fmt(s.itl_ms_median, 2)}</td>
                  <td>{fmt(s.itl_ms_p99, 2)}</td>
                  <td>{fmt(s.itl_ms_std, 2)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Only for terminal runs: telemetry arrives with the benchmark result, so a
          run still in flight has nothing to show and would just render an empty axis. */}
      {(run.status === "succeeded" || run.status === "failed") && (
        <>
          <h3>Timeline</h3>
          <TelemetryTimeline runId={run.id} />
        </>
      )}

      <h3>Provenance</h3>
      <dl className="facts">
        <div className="fact">
          <dt>vLLM</dt>
          <dd>{run.vllm_version ?? "—"}</dd>
        </div>
        <div className="fact">
          <dt>GPU</dt>
          <dd>{run.gpu_model ?? "—"}</dd>
        </div>
        <div className="fact">
          <dt>Driver</dt>
          <dd>{run.driver_version ?? "—"}</dd>
        </div>
        <div className="fact">
          <dt>Topology</dt>
          <dd>
            TP{run.tensor_parallel_size} · PP{run.pipeline_parallel_size} ·{" "}
            {run.gpu_count} GPU{run.gpu_count === 1 ? "" : "s"}
          </dd>
        </div>
        <div className="fact">
          <dt>Devices</dt>
          <dd>{run.device_indices?.join(", ") ?? "—"}</dd>
        </div>
        <div className="fact">
          <dt>Bench client</dt>
          <dd>{run.bench_client_location}</dd>
        </div>
        <div className="fact">
          <dt>Config</dt>
          <dd>{run.config_hash.slice(0, 16)}</dd>
        </div>
        <div className="fact">
          <dt>Initiated by</dt>
          <dd>{run.initiated_by}</dd>
        </div>
      </dl>
    </div>
  );
}

export function RunsView() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [configs, setConfigs] = useState<Config[]>([]);
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [hostId, setHostId] = useState("");
  const [configId, setConfigId] = useState("");
  const [workloadId, setWorkloadId] = useState("");
  const [configName, setConfigName] = useState("baseline");
  const [configYaml, setConfigYaml] = useState(DEFAULT_YAML);
  const [workloadName, setWorkloadName] = useState("default");
  const [numPrompts, setNumPrompts] = useState(200);
  const [concurrency, setConcurrency] = useState(32);

  const timer = useRef<number | null>(null);

  const loadAll = useCallback(async () => {
    try {
      const [h, c, w, r] = await Promise.all([
        api.listHosts(),
        configsApi.list(),
        workloadsApi.list(),
        runsApi.list(),
      ]);
      setHosts(h);
      setConfigs(c);
      setWorkloads(w);
      setRuns(r);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void loadAll();
  }, [loadAll]);

  // Poll only while something is in flight. A run takes minutes, and polling an idle
  // system forever is pure noise against the database.
  useEffect(() => {
    const active = runs.some((r) => !TERMINAL.includes(r.status));
    if (!active) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    if (timer.current) return;
    timer.current = window.setInterval(() => void loadAll(), POLL_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
    };
  }, [runs, loadAll]);

  const trigger = async () => {
    setBusy(true);
    setError(null);
    try {
      const config = configId
        ? configs.find((c) => c.id === configId)!
        : await configsApi.create(configName, configYaml);
      const workload = workloadId
        ? workloads.find((w) => w.id === workloadId)!
        : await workloadsApi.create({
            name: workloadName,
            num_prompts: numPrompts,
            max_concurrency: concurrency,
            input_len: 512,
            output_len: 128,
          });
      await runsApi.create(hostId, config.id, workload.id);
      await loadAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const selectedRun = runs.find((r) => r.id === selected) ?? null;

  return (
    <>
      <section>
        <h2>Trigger a run</h2>
        <div className="card">
          {hosts.length === 0 ? (
            <p className="muted">
              Register a GPU host first, or run <code>make dev</code> for the mock agent.
            </p>
          ) : (
            <>
              <div className="field-grid">
                <label>
                  <span>Host</span>
                  <select value={hostId} onChange={(e) => setHostId(e.target.value)}>
                    <option value="">Select…</option>
                    {hosts.map((h) => (
                      <option key={h.id} value={h.id}>
                        {h.name}
                        {h.synthetic_source ? " (synthetic)" : ""}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  <span>Config</span>
                  <select value={configId} onChange={(e) => setConfigId(e.target.value)}>
                    <option value="">New from YAML below…</option>
                    {configs.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} · {c.config_hash.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  <span>Workload</span>
                  <select value={workloadId} onChange={(e) => setWorkloadId(e.target.value)}>
                    <option value="">New from fields below…</option>
                    {workloads.map((w) => (
                      <option key={w.id} value={w.id}>
                        {w.name} · {w.num_prompts}p · c{w.max_concurrency ?? "∞"}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {!configId && (
                <>
                  <label className="stacked">
                    <span>Config name</span>
                    <input value={configName} onChange={(e) => setConfigName(e.target.value)} />
                  </label>
                  <label className="stacked">
                    <span>vLLM YAML — stored and executed verbatim</span>
                    <textarea
                      rows={6}
                      value={configYaml}
                      onChange={(e) => setConfigYaml(e.target.value)}
                      spellCheck={false}
                    />
                  </label>
                </>
              )}

              {!workloadId && (
                <div className="field-grid">
                  <label>
                    <span>Workload name</span>
                    <input
                      value={workloadName}
                      onChange={(e) => setWorkloadName(e.target.value)}
                    />
                  </label>
                  <label>
                    <span>Prompts</span>
                    <input
                      type="number"
                      value={numPrompts}
                      onChange={(e) => setNumPrompts(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    <span>Max concurrency</span>
                    <input
                      type="number"
                      value={concurrency}
                      onChange={(e) => setConcurrency(Number(e.target.value))}
                    />
                  </label>
                </div>
              )}

              <button className="primary" disabled={!hostId || busy} onClick={() => void trigger()}>
                {busy ? "Queueing…" : "Queue run"}
              </button>
              {error && <p className="error">{error}</p>}
            </>
          )}
        </div>
      </section>

      <section>
        <h2>Runs</h2>
        {selectedRun && <RunDetail run={selectedRun} onClose={() => setSelected(null)} />}
        {runs.length === 0 ? (
          <p className="muted">No runs yet.</p>
        ) : (
          <div className="card devices">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Queued</th>
                  <th>Config</th>
                  <th>Topology</th>
                  <th>Tok/s per GPU</th>
                  <th>p99 TTFT (ms)</th>
                  <th>Median ITL (ms)</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <RunRow key={r.id} run={r} onSelect={setSelected} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
