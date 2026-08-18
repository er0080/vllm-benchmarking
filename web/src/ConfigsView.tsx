/**
 * Authoring, checking and exporting server configurations.
 *
 * The editor is a plain textarea over the exact YAML, on purpose. Invariant 5 says what
 * is stored is what runs, so a form of typed fields would mean this view holding opinions
 * about vLLM's option set — opinions that rot every release, and that would silently drop
 * any setting the form did not know about. The validation engine supplies the
 * intelligence instead, and it gets its knowledge from a captured parser rather than from
 * anything written here.
 *
 * **Nothing in this view rewrites the author's text.** Suggestions are shown next to the
 * finding that produced them and applied by the author or not at all. A validator that
 * silently corrected a config would make the bytes that ran different from the bytes
 * somebody wrote, which is the one property this whole screen exists to preserve.
 *
 * Saving never overwrites. Configurations are content-addressed, so editing produces a
 * new one and the old text is still there — which is why lineage is worth recording and
 * why there is no delete button.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, configsApi } from "./api";
import type { Config, ConfigValidation, Finding, Host, Lineage } from "./types";

const STARTER = `model: Qwen/Qwen3.5-9B
tensor-parallel-size: 1
max-model-len: 8192
gpu-memory-utilization: 0.90
`;

/** How long to sit still before validating. Long enough not to fire mid-word, short
 *  enough that the answer feels attached to the edit. */
const VALIDATE_AFTER_MS = 400;

function Findings({ result }: { result: ConfigValidation }) {
  const errors = result.findings.filter((f) => f.severity === "error");
  const warnings = result.findings.filter((f) => f.severity === "warning");

  if (result.findings.length === 0) {
    return (
      <p className="ok">
        No problems found, checked against vLLM {result.checked_against}.
        {!result.exact_version_match && (
          <>
            {" "}
            That is not the version your target host reports, so a setting it added more
            recently would show here as unknown.
          </>
        )}
      </p>
    );
  }

  return (
    <div className="findings">
      {/* Errors and warnings are kept visually apart because they are different claims:
          one says the engine will refuse this file, the other says it will start and may
          not do what you meant. A single list of "problems" hides that. */}
      {errors.length > 0 && (
        <>
          <h4 className="finding-heading">
            {errors.length} {errors.length === 1 ? "error" : "errors"} — vLLM will not start
          </h4>
          <ul>
            {errors.map((finding, index) => (
              <FindingRow key={index} finding={finding} />
            ))}
          </ul>
        </>
      )}
      {warnings.length > 0 && (
        <>
          <h4 className="finding-heading">
            {warnings.length} {warnings.length === 1 ? "warning" : "warnings"} — it will
            start; check this is what you meant
          </h4>
          <ul>
            {warnings.map((finding, index) => (
              <FindingRow key={index} finding={finding} />
            ))}
          </ul>
        </>
      )}
      <p className="muted">
        Checked against vLLM {result.checked_against}
        {result.exact_version_match ? "" : " (your host runs a different version)"}.
      </p>
    </div>
  );
}

function FindingRow({ finding }: { finding: Finding }) {
  return (
    <li className={`finding finding-${finding.severity}`}>
      {finding.line !== null && <span className="finding-line">line {finding.line}</span>}
      <span>{finding.message}</span>
      {finding.suggestion && (
        <>
          {" "}
          <code className="finding-suggestion">{finding.suggestion}</code>
        </>
      )}
    </li>
  );
}

function LineagePanel({ lineage, onOpen }: { lineage: Lineage; onOpen: (hash: string) => void }) {
  if (lineage.ancestors.length === 0 && lineage.children.length === 0) {
    return <p className="muted">Written from scratch, and nothing has been derived from it yet.</p>;
  }
  return (
    <div className="lineage">
      {lineage.ancestors.length > 0 && (
        <p className="muted">
          Edited from{" "}
          {lineage.ancestors.map((node, index) => (
            <span key={node.id}>
              {index > 0 && " ← "}
              <button className="linky" onClick={() => onOpen(node.config_hash)}>
                {node.name}
              </button>
            </span>
          ))}
          {lineage.truncated && " … (chain truncated)"}
        </p>
      )}
      {lineage.children.length > 0 && (
        <p className="muted">
          Tried next:{" "}
          {lineage.children.map((node, index) => (
            <span key={node.id}>
              {index > 0 && ", "}
              <button className="linky" onClick={() => onOpen(node.config_hash)}>
                {node.name}
              </button>
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

export function ConfigsView() {
  const [configs, setConfigs] = useState<Config[]>([]);
  const [hosts, setHosts] = useState<Host[]>([]);
  const [hostId, setHostId] = useState<string | null>(null);

  const [name, setName] = useState("new config");
  const [yaml, setYaml] = useState(STARTER);
  // Which stored config the editor's text started from. Cleared as soon as the text
  // stops matching it, because "derived from X" has to mean the author actually edited
  // X rather than happened to have it open.
  const [parent, setParent] = useState<Config | null>(null);

  const [validation, setValidation] = useState<ConfigValidation | null>(null);
  const [checking, setChecking] = useState(false);
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [saved, setSaved] = useState<Config | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const [c, h] = await Promise.allSettled([configsApi.list(), api.listHosts()]);
    if (c.status === "fulfilled") setConfigs(c.value);
    if (h.status === "fulfilled") setHosts(h.value);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Validation is debounced rather than tied to a button: the whole value of checking a
  // config is that it happens before you have committed to anything, and a check you
  // have to ask for is one you ask for after you have decided.
  useEffect(() => {
    if (!yaml.trim()) {
      setValidation(null);
      return;
    }
    let cancelled = false;
    setChecking(true);
    const timer = setTimeout(() => {
      configsApi
        .validate(yaml, hostId)
        .then((result) => !cancelled && setValidation(result))
        .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
        .finally(() => !cancelled && setChecking(false));
    }, VALIDATE_AFTER_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [yaml, hostId]);

  const openConfig = useCallback(
    async (hash: string) => {
      const found = configs.find((c) => c.config_hash === hash);
      if (!found) return;
      setName(found.name);
      setYaml(found.yaml);
      setParent(found);
      setSaved(found);
      setError(null);
      try {
        setLineage(await configsApi.lineage(found.config_hash));
      } catch {
        setLineage(null);
      }
    },
    [configs],
  );

  // The editor's text has moved away from the config it was opened from, so this is now
  // an edit of that config rather than that config.
  const edited = parent !== null && parent.yaml.trim() !== yaml.trim();
  const storedMatch = useMemo(
    () => configs.find((c) => c.yaml.trim() === yaml.trim()) ?? null,
    [configs, yaml],
  );

  const save = async () => {
    setError(null);
    try {
      const created = await configsApi.create(name, yaml, edited ? parent.id : null);
      setSaved(created);
      setParent(created);
      await load();
      setLineage(await configsApi.lineage(created.config_hash));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const importFile = async (file: File) => {
    const text = await file.text();
    setYaml(text);
    // A file has no history in this database, so it starts a new lineage rather than
    // claiming to be an edit of whatever happened to be open.
    setParent(null);
    setSaved(null);
    setLineage(null);
    setName(file.name.replace(/\.ya?ml$/i, "") || "imported config");
  };

  return (
    <section>
      <h2>Configurations</h2>

      <div className="card">
        <div className="toolbar">
          <label>
            Open
            <select
              value={saved?.config_hash ?? ""}
              onChange={(e) => e.target.value && void openConfig(e.target.value)}
            >
              <option value="">New configuration…</option>
              {configs.map((config) => (
                <option key={config.id} value={config.config_hash}>
                  {config.name} ({config.config_hash.slice(0, 8)})
                  {config.justified_by_run_id ? " ★" : ""}
                </option>
              ))}
            </select>
          </label>

          <label>
            Check against
            <select
              value={hostId ?? ""}
              onChange={(e) => setHostId(e.target.value || null)}
            >
              {/* Without a host the topology checks are skipped rather than guessed at:
                  validating a config in the abstract must not invent a machine to reject
                  it against. */}
              <option value="">No host — reference vLLM, no GPU count</option>
              {hosts.map((host) => (
                <option key={host.id} value={host.id}>
                  {host.name} ({host.gpu_count} GPU, vLLM {host.vllm_version ?? "unknown"})
                </option>
              ))}
            </select>
          </label>

          <span className="spacer" />

          <input
            ref={fileInput}
            type="file"
            accept=".yaml,.yml,text/yaml"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void importFile(file);
              e.target.value = "";
            }}
          />
          <button onClick={() => fileInput.current?.click()}>Import YAML…</button>
          {saved && (
            // A plain link, so the browser downloads the exact bytes the API sends. The
            // whole point of the export is that the file in production is the file that
            // was measured.
            <a className="button" href={configsApi.exportUrl(saved.config_hash)} download>
              Export
            </a>
          )}
        </div>

        <label className="stacked">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>

        <label className="stacked">
          Configuration (native vLLM YAML — stored and executed verbatim)
          <textarea
            className="yaml-editor"
            spellCheck={false}
            rows={16}
            value={yaml}
            onChange={(e) => setYaml(e.target.value)}
          />
        </label>

        <div className="toolbar">
          <button className="primary" onClick={() => void save()} disabled={!yaml.trim()}>
            {edited ? "Save as a new configuration" : "Save"}
          </button>
          {checking && <span className="muted">checking…</span>}
          {/* Never a block on saving. The catalogue is a capture of one vLLM version, and
              a host running something newer may legitimately accept a setting this
              control plane has never heard of. */}
          {validation && !validation.valid && (
            <span className="muted">You can still save this — validation advises, it does not block.</span>
          )}
        </div>

        {error && <p className="error">{error}</p>}
        {validation && <Findings result={validation} />}

        {storedMatch && !saved && (
          <p className="notice">
            This is byte-identical to <strong>{storedMatch.name}</strong>, which already
            exists. Saving will return that configuration rather than creating a second
            copy — configurations are identified by their content.
          </p>
        )}

        {saved && (
          <>
            <h3>
              {saved.name} <code className="muted">{saved.config_hash.slice(0, 12)}</code>
            </h3>
            {lineage && <LineagePanel lineage={lineage} onOpen={(h) => void openConfig(h)} />}
            <Justification config={saved} onChanged={async () => {
              await load();
              const fresh = (await configsApi.list()).find((c) => c.id === saved.id);
              if (fresh) setSaved(fresh);
            }} />
          </>
        )}
      </div>
    </section>
  );
}

/**
 * The measurement that defends this configuration.
 *
 * A YAML file six months old is a list of numbers with no argument attached. This is the
 * link back to the run that settles it — and the API refuses a run of any other config,
 * because a citation that can point at the wrong evidence is worse than no citation.
 */
function Justification({ config, onChanged }: { config: Config; onChanged: () => void }) {
  const [runId, setRunId] = useState(config.justified_by_run_id ?? "");
  const [note, setNote] = useState(config.justification_note ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setRunId(config.justified_by_run_id ?? "");
    setNote(config.justification_note ?? "");
  }, [config]);

  const submit = async (clear: boolean) => {
    setBusy(true);
    setError(null);
    try {
      await configsApi.annotate(config.config_hash,
        clear
          ? { clear_justification: true }
          : { justified_by_run_id: runId || null, justification_note: note },
      );
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="justification">
      <h4>Why this configuration</h4>
      <div className="toolbar">
        <label>
          Run that justifies it
          <input
            placeholder="run id from the Runs tab"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            size={38}
          />
        </label>
        <label>
          Because
          <input
            placeholder="beat TP2 on both axes at concurrency 16"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            size={40}
          />
        </label>
        <button disabled={busy || !runId} onClick={() => void submit(false)}>
          Record
        </button>
        {config.justified_by_run_id && (
          <button disabled={busy} onClick={() => void submit(true)}>
            Withdraw
          </button>
        )}
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
