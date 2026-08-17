/**
 * One filter selection, shared by every analysis view.
 *
 * Lifted out of the individual views because the alternative — each tab owning its own
 * filters — makes the tabs lie to each other. A reader who narrows the Pareto view to one
 * sweep and then switches to scaling is asking about *that sweep*, and being silently
 * shown the whole history instead is worse than not having the filter: the numbers look
 * like an answer to the question they just asked.
 *
 * `source` lives here too, and remains a single choice with no value meaning both. It is
 * the same invariant-7 guarantee the API makes, held in the one place the UI could
 * otherwise break it.
 */
import { useCallback, useEffect, useState } from "react";

import { api, sweepsApi } from "./api";
import type { AnalysisQuery, Host, RunSource, Sweep } from "./types";

export type Filters = {
  source: RunSource;
  hostId: string | null;
  sweepId: string | null;
  tensorParallelSizes: number[];
};

export const NO_FILTERS: Filters = {
  source: "real",
  hostId: null,
  sweepId: null,
  tensorParallelSizes: [],
};

/** Filters as the API takes them. One conversion, so no view invents its own. */
export function toQuery(filters: Filters): AnalysisQuery {
  return {
    source: filters.source,
    ...(filters.hostId ? { hostId: filters.hostId } : {}),
    ...(filters.sweepId ? { sweepIds: [filters.sweepId] } : {}),
    ...(filters.tensorParallelSizes.length
      ? { tensorParallelSizes: filters.tensorParallelSizes }
      : {}),
  };
}

/** Whether anything is narrowing the population, for a view that wants to say so. */
export function isNarrowed(filters: Filters): boolean {
  return Boolean(filters.hostId || filters.sweepId || filters.tensorParallelSizes.length);
}

const TP_CHOICES = [1, 2, 4, 8];

export function FilterBar({
  filters,
  onChange,
  showTensorParallel = true,
}: {
  filters: Filters;
  onChange: (next: Filters) => void;
  /** The scaling view hides this: filtering the axis under study to one value is the
      only query that cannot produce a curve. */
  showTensorParallel?: boolean;
}) {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [sweeps, setSweeps] = useState<Sweep[]>([]);

  const load = useCallback(async () => {
    // Failures here are not surfaced: an empty option list degrades to "no filter
    // available", which is the correct behaviour, and an error banner over a filter bar
    // would sit above charts that are working fine.
    const [h, s] = await Promise.allSettled([api.listHosts(), sweepsApi.list()]);
    if (h.status === "fulfilled") setHosts(h.value);
    if (s.status === "fulfilled") setSweeps(s.value);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Sweeps are offered from the population being viewed. A synthetic sweep in a real-run
  // filter list would produce an empty chart with no explanation of why.
  const relevantSweeps = sweeps.filter(
    (sweep) => sweep.is_synthetic === (filters.source === "synthetic"),
  );

  const toggleTp = (size: number) => {
    const current = filters.tensorParallelSizes;
    onChange({
      ...filters,
      tensorParallelSizes: current.includes(size)
        ? current.filter((s) => s !== size)
        : [...current, size].sort((a, b) => a - b),
    });
  };

  return (
    <div className="card filters">
      <div className="toolbar">
        <label>
          Runs
          <select
            value={filters.source}
            onChange={(e) =>
              // Changing population invalidates a sweep chosen from the other one.
              onChange({ ...filters, source: e.target.value as RunSource, sweepId: null })
            }
          >
            <option value="real">Real measurements</option>
            <option value="synthetic">Synthetic (mock / CPU backend)</option>
          </select>
        </label>

        <label>
          Host
          <select
            value={filters.hostId ?? ""}
            onChange={(e) => onChange({ ...filters, hostId: e.target.value || null })}
          >
            <option value="">All hosts</option>
            {hosts.map((host) => (
              <option key={host.id} value={host.id}>
                {host.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Sweep
          <select
            value={filters.sweepId ?? ""}
            onChange={(e) => onChange({ ...filters, sweepId: e.target.value || null })}
          >
            <option value="">All runs</option>
            {relevantSweeps.map((sweep) => (
              <option key={sweep.id} value={sweep.id}>
                {sweep.name} ({sweep.progress.succeeded} succeeded)
              </option>
            ))}
          </select>
        </label>

        {showTensorParallel && (
          <fieldset className="chips">
            <legend>TP</legend>
            {TP_CHOICES.map((size) => (
              <label key={size} className="chip">
                <input
                  type="checkbox"
                  checked={filters.tensorParallelSizes.includes(size)}
                  onChange={() => toggleTp(size)}
                />
                {size}
              </label>
            ))}
          </fieldset>
        )}

        <span className="spacer" />
        {isNarrowed(filters) && (
          <button onClick={() => onChange({ ...NO_FILTERS, source: filters.source })}>
            Clear filters
          </button>
        )}
      </div>

      {filters.source === "synthetic" && (
        <p className="notice" style={{ marginBottom: 0 }}>
          Synthetic runs. These come from the mock agent or the CPU backend and are not
          measurements of any real hardware. They are quarantined from real results and can
          never appear on the same chart as one.
        </p>
      )}
    </div>
  );
}
