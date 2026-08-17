import { useCallback, useEffect, useState } from "react";

import { api } from "./api";
import { RunsView } from "./RunsView";
import { SweepsView } from "./SweepsView";
import type { Host } from "./types";

function formatBytes(bytes: number | null): string {
  if (bytes === null) return "—";
  return `${(bytes / 1024 ** 3).toFixed(0)} GiB`;
}

function Fact({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div className="fact">
      <dt>{label}</dt>
      <dd>{value === null || value === "" ? "—" : value}</dd>
    </div>
  );
}

function HostCard({ host, onChanged }: { host: Host; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // A version mismatch is deliberately not an error state. Benchmarking one vLLM
  // version against another is a supported use of this tool, so this informs rather
  // than blocks.
  const versionDrift =
    host.vllm_version_matches_reference === false && host.reference_vllm_version !== null;

  return (
    <article className="card host">
      <div className="host-head">
        <strong>{host.name}</strong>
        {host.synthetic_source && (
          <span className="badge badge-synthetic">synthetic · {host.synthetic_source}</span>
        )}
        <span className="host-url">{host.agent_url}</span>
        <span className="spacer" />
        <button disabled={busy} onClick={() => act(() => api.refreshHost(host.id))}>
          {busy ? "…" : "Refresh"}
        </button>
        <button disabled={busy} onClick={() => act(() => api.deleteHost(host.id))}>
          Remove
        </button>
      </div>

      {host.synthetic_source && (
        <p className="notice">
          This host is backed by a stand-in, not real hardware. Anything it produces is
          quarantined from real measurements and must never be read as one.
        </p>
      )}

      {versionDrift && (
        <p className="notice">
          Running vLLM {host.vllm_version}; this build is tested against{" "}
          {host.reference_vllm_version}. Not a problem — results record the version they
          were produced with — but comparisons across versions are not like-for-like.
        </p>
      )}

      <dl className="facts">
        <Fact label="GPUs" value={host.gpu_count} />
        <Fact label="vLLM" value={host.vllm_version} />
        <Fact label="Driver" value={host.driver_version} />
        <Fact label="CUDA" value={host.cuda_version} />
        <Fact label="Agent" value={host.agent_version} />
        <Fact label="Protocol" value={host.protocol_version} />
        <Fact
          label="Last seen"
          value={host.last_seen_at ? new Date(host.last_seen_at).toLocaleString() : null}
        />
      </dl>

      {host.devices.length > 0 ? (
        <div className="devices">
          <table>
            <thead>
              <tr>
                <th>Index</th>
                <th>Device</th>
                <th>VRAM</th>
              </tr>
            </thead>
            <tbody>
              {host.devices.map((d) => (
                <tr key={d.device_index}>
                  <td>{d.device_index}</td>
                  <td>{d.name}</td>
                  <td>{formatBytes(d.vram_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No GPUs reported by this host.</p>
      )}

      {error && <p className="error">{error}</p>}
    </article>
  );
}

function RegisterForm({ onRegistered }: { onRegistered: () => void }) {
  const [name, setName] = useState("");
  const [agentUrl, setAgentUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.registerHost(name.trim(), agentUrl.trim());
      setName("");
      setAgentUrl("");
      onRegistered();
    } catch (e) {
      // Carries the API's actionable detail: unreachable, bad token, or protocol
      // mismatch naming both versions.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <input
        placeholder="Host name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        required
      />
      <input
        placeholder="http://192.168.1.100:9110"
        value={agentUrl}
        onChange={(e) => setAgentUrl(e.target.value)}
        required
      />
      <button className="primary" type="submit" disabled={busy}>
        {busy ? "Contacting agent…" : "Register"}
      </button>
      {error && <p className="error">{error}</p>}
    </form>
  );
}

type Tab = "hosts" | "runs" | "sweeps";

export function App() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [tab, setTab] = useState<Tab>("hosts");

  const load = useCallback(async () => {
    try {
      setHosts(await api.listHosts());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="page">
      <header>
        <h1>vLLM Benchmarking</h1>
        <p>Milestone 0.4.0 — sweeps.</p>
      </header>

      <nav className="tabs">
        {(["hosts", "runs", "sweeps"] as Tab[]).map((name) => (
          <button key={name} aria-current={tab === name} onClick={() => setTab(name)}>
            {name.charAt(0).toUpperCase() + name.slice(1)}
          </button>
        ))}
      </nav>

      {tab === "hosts" && (
        <>
          <section>
            <h2>Register a GPU host</h2>
            <div className="card">
              <RegisterForm onRegistered={load} />
              <p className="muted" style={{ marginBottom: 0, marginTop: "0.6rem" }}>
                Registering performs a live handshake with the agent. It fails rather than
                storing a host it cannot reach.
              </p>
            </div>
          </section>

          <section>
            <h2>Hosts</h2>
            {error && <p className="error">{error}</p>}
            {!loaded && <p className="muted">Loading…</p>}
            {loaded && hosts.length === 0 && !error && (
              <p className="muted">
                No hosts yet. Start the agent on your GPU host, or run{" "}
                <code>make dev</code> to bring up the mock agent.
              </p>
            )}
            {hosts.map((host) => (
              <HostCard key={host.id} host={host} onChanged={load} />
            ))}
          </section>
        </>
      )}

      {tab === "runs" && <RunsView />}
      {tab === "sweeps" && <SweepsView />}
    </div>
  );
}
