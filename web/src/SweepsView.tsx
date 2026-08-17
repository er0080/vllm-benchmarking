/**
 * Authoring and watching sweeps.
 *
 * The authoring form's job is to make the cost of a sweep visible *before* it is started.
 * A matrix that looks small on screen can be a day of GPU time, and the number that
 * predicts it is not the run count but the number of model loads — so the estimate is
 * shown live as the axes change, and it is the reason the replicate-order choice is here
 * rather than buried in defaults.
 *
 * Progress is rendered from counts the API already computes, so watching a sweep costs
 * one small request rather than fetching every run on the page.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, configsApi, sweepsApi, workloadsApi } from "./api";
import type { Config, Host, Sweep, SweepStatus, Workload } from "./types";

const ACTIVE: SweepStatus[] = ["queued", "running"];
const POLL_MS = 3000;

/** Segments in the order a run moves through them, so the bar fills left to right. */
const SEGMENTS = [
  { key: "succeeded", label: "succeeded", color: "var(--series-3)" },
  { key: "failed", label: "failed", color: "var(--danger)" },
  { key: "cancelled", label: "cancelled", color: "var(--text-muted)" },
  { key: "benchmarking", label: "benchmarking", color: "var(--series-1)" },
  { key: "starting", label: "starting", color: "var(--series-4)" },
  { key: "queued", label: "queued", color: "var(--border)" },
] as const;

function ProgressBar({ sweep }: { sweep: Sweep }) {
  const total = Math.max(1, sweep.progress.total);
  return (
    <>
      <div
        className="progress"
        role="img"
        aria-label={SEGMENTS.map((s) => `${sweep.progress[s.key]} ${s.label}`).join(", ")}
      >
        {SEGMENTS.map((segment) => {
          const count = sweep.progress[segment.key];
          if (!count) return null;
          return (
            <span
              key={segment.key}
              style={{ width: `${(count / total) * 100}%`, background: segment.color }}
              title={`${count} ${segment.label}`}
            />
          );
        })}
      </div>
      {/* The counts in text as well as colour: the bar is a summary, and identity in a
          stacked bar should never rest on colour alone. */}
      <p className="muted" style={{ margin: "0.35rem 0 0" }}>
        {SEGMENTS.filter((s) => sweep.progress[s.key] > 0)
          .map((s) => `${sweep.progress[s.key]} ${s.label}`)
          .join(" · ")}
        {" · "}
        {sweep.progress.total} total
      </p>
    </>
  );
}

function SweepCard({ sweep, onChanged }: { sweep: Sweep; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const active = ACTIVE.includes(sweep.status);

  const cancel = async () => {
    setBusy(true);
    setError(null);
    try {
      await sweepsApi.cancel(sweep.id);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="card host">
      <div className="host-head">
        <strong>{sweep.name}</strong>
        <span className={`badge status-${sweep.status}`}>{sweep.status}</span>
        {sweep.is_synthetic && <span className="badge badge-synthetic">synthetic</span>}
        <span className="spacer" />
        {active && (
          <button disabled={busy} onClick={() => void cancel()}>
            {busy ? "Cancelling…" : "Cancel"}
          </button>
        )}
      </div>

      {sweep.description && <p className="muted">{sweep.description}</p>}

      <ProgressBar sweep={sweep} />

      <dl className="facts" style={{ marginTop: "0.75rem" }}>
        <div className="fact">
          <dt>Replicates</dt>
          <dd>
            {sweep.replicates} · {sweep.replicate_order}
          </dd>
        </div>
        <div className="fact">
          <dt>Model loads</dt>
          <dd>{sweep.engine_starts}</dd>
        </div>
        <div className="fact">
          <dt>Started</dt>
          <dd>{sweep.started_at ? new Date(sweep.started_at).toLocaleString() : "—"}</dd>
        </div>
        <div className="fact">
          <dt>Finished</dt>
          <dd>{sweep.finished_at ? new Date(sweep.finished_at).toLocaleString() : "—"}</dd>
        </div>
      </dl>

      {sweep.error && <p className="error">{sweep.error}</p>}
      {error && <p className="error">{error}</p>}
    </article>
  );
}

/** Parses "1, 2, 4" into [1, 2, 4]. Returns null for an empty field — meaning no axis. */
function parseTensorParallel(raw: string): number[] | null {
  const values = raw
    .split(/[,\s]+/)
    .map((token) => token.trim())
    .filter(Boolean)
    .map(Number);
  if (!values.length) return null;
  if (values.some((v) => !Number.isInteger(v) || v < 1)) {
    throw new Error("tensor-parallel sizes must be whole numbers of 1 or more");
  }
  return values;
}

export function SweepsView() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [configs, setConfigs] = useState<Config[]>([]);
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [name, setName] = useState("");
  const [hostId, setHostId] = useState("");
  const [configIds, setConfigIds] = useState<string[]>([]);
  const [workloadIds, setWorkloadIds] = useState<string[]>([]);
  const [tpRaw, setTpRaw] = useState("");
  const [replicates, setReplicates] = useState(3);
  const [order, setOrder] = useState<"grouped" | "interleaved">("grouped");

  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const [h, c, w, s] = await Promise.all([
        api.listHosts(),
        configsApi.list(),
        workloadsApi.list(),
        sweepsApi.list(),
      ]);
      setHosts(h);
      setConfigs(c);
      setWorkloads(w);
      setSweeps(s);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll only while something is actually moving. A sweep runs for hours, and polling an
  // idle system forever is noise against the database.
  useEffect(() => {
    const active = sweeps.some((s) => ACTIVE.includes(s.status));
    if (!active) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    if (timer.current) return;
    timer.current = window.setInterval(() => void load(), POLL_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
    };
  }, [sweeps, load]);

  const host = hosts.find((h) => h.id === hostId) ?? null;

  /**
   * What the matrix will cost, recomputed as it is edited.
   *
   * Model loads matter more than run count here: a restart is minutes for a large model,
   * so interleaving replicates can multiply the wall clock several times over while the
   * run count stays identical. Showing it before the sweep starts is the whole point.
   */
  const estimate = useMemo(() => {
    let tp: number[] | null = null;
    try {
      tp = parseTensorParallel(tpRaw);
    } catch {
      return null;
    }
    const configCount = configIds.length * (tp ? tp.length : 1);
    const runs = configCount * workloadIds.length * replicates;
    if (!runs) return null;
    return {
      runs,
      configs: configCount,
      loads: order === "interleaved" ? configCount * replicates : configCount,
    };
  }, [configIds, workloadIds, tpRaw, replicates, order]);

  const tooWide = useMemo(() => {
    if (!host) return null;
    let tp: number[] | null = null;
    try {
      tp = parseTensorParallel(tpRaw);
    } catch {
      return null;
    }
    const over = (tp ?? []).filter((v) => host.gpu_count > 0 && v > host.gpu_count);
    return over.length ? over : null;
  }, [host, tpRaw]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await sweepsApi.create({
        name: name.trim(),
        gpu_host_id: hostId,
        server_config_ids: configIds,
        workload_ids: workloadIds,
        tensor_parallel_sizes: parseTensorParallel(tpRaw),
        replicates,
        replicate_order: order,
      });
      setName("");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const canSubmit =
    !busy && name.trim() !== "" && hostId !== "" && configIds.length > 0 && workloadIds.length > 0;

  return (
    <>
      <section>
        <h2>Author a sweep</h2>
        <div className="card">
          {hosts.length === 0 ? (
            <p className="muted">Register a GPU host first.</p>
          ) : (
            <>
              <div className="field-grid">
                <label>
                  <span>Name</span>
                  <input value={name} onChange={(e) => setName(e.target.value)} />
                </label>
                <label>
                  <span>Host</span>
                  <select value={hostId} onChange={(e) => setHostId(e.target.value)}>
                    <option value="">Select…</option>
                    {hosts.map((h) => (
                      <option key={h.id} value={h.id}>
                        {h.name} ({h.gpu_count} GPU{h.gpu_count === 1 ? "" : "s"})
                        {h.synthetic_source ? " · synthetic" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Tensor-parallel sizes</span>
                  <input
                    placeholder="e.g. 1, 2 — blank leaves each config alone"
                    value={tpRaw}
                    onChange={(e) => setTpRaw(e.target.value)}
                  />
                </label>
              </div>

              <div className="field-grid">
                <label>
                  <span>Server configs ({configIds.length} selected)</span>
                  <select
                    multiple
                    size={5}
                    value={configIds}
                    onChange={(e) =>
                      setConfigIds(Array.from(e.target.selectedOptions, (o) => o.value))
                    }
                  >
                    {configs.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} · {c.config_hash.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Workloads ({workloadIds.length} selected)</span>
                  <select
                    multiple
                    size={5}
                    value={workloadIds}
                    onChange={(e) =>
                      setWorkloadIds(Array.from(e.target.selectedOptions, (o) => o.value))
                    }
                  >
                    {workloads.map((w) => (
                      <option key={w.id} value={w.id}>
                        {w.name} · {w.num_prompts}p · c{w.max_concurrency ?? "∞"}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="field-grid">
                <label>
                  <span>Replicates</span>
                  <input
                    type="number"
                    min={1}
                    max={25}
                    value={replicates}
                    onChange={(e) => setReplicates(Number(e.target.value))}
                  />
                </label>
                <label>
                  <span>Replicate order</span>
                  <select
                    value={order}
                    onChange={(e) => setOrder(e.target.value as "grouped" | "interleaved")}
                  >
                    <option value="grouped">Grouped — replicates back to back</option>
                    <option value="interleaved">Interleaved — repeat the whole matrix</option>
                  </select>
                </label>
              </div>

              {/* What the two orderings actually claim about the spread. Without this the
                  choice reads as a scheduling detail rather than a measurement one. */}
              <p className="muted">
                {order === "grouped"
                  ? "Replicates run consecutively, so the spread measures repeatability under near-identical conditions. One model load per config."
                  : "The whole matrix repeats, so each replicate meets different thermal and clock state and the spread reflects run-to-run variance. Costs a model load per config per replicate."}
              </p>

              {tooWide && (
                <p className="notice">
                  {host?.name} reports {host?.gpu_count} GPU
                  {host?.gpu_count === 1 ? "" : "s"}, so TP {tooWide.join(", ")} will be
                  refused — per-GPU figures would be normalized against devices the run
                  never had.
                </p>
              )}

              {estimate && (
                <p className="muted">
                  <strong>{estimate.runs}</strong> runs across {estimate.configs} config
                  {estimate.configs === 1 ? "" : "s"}, with{" "}
                  <strong>{estimate.loads}</strong> model load
                  {estimate.loads === 1 ? "" : "s"} — usually most of the wall clock.
                </p>
              )}

              <button className="primary" disabled={!canSubmit} onClick={() => void submit()}>
                {busy ? "Creating…" : "Create and queue"}
              </button>
              {error && <p className="error">{error}</p>}
            </>
          )}
        </div>
      </section>

      <section>
        <h2>Sweeps</h2>
        {sweeps.length === 0 ? (
          <p className="muted">No sweeps yet.</p>
        ) : (
          sweeps.map((s) => <SweepCard key={s.id} sweep={s} onChanged={load} />)
        )}
      </section>
    </>
  );
}
