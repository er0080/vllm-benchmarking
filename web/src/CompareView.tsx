/**
 * Two measurement points, side by side, with a diff of the configs behind them.
 *
 * Deliberately not a chart. The question here is "what is different about these two and
 * what did it buy", which is a table of numbers and a diff of text — a chart of two bars
 * would be decoration around four values.
 *
 * **This is the one view that may cross a comparability boundary.** Every chart refuses
 * to overlay two vLLM versions, because a reader cannot see from a scatter that the blue
 * dots came from a different build. Here the reader has named both sides and the
 * differences are listed back at them, which makes it the comparison the version policy
 * exists to enable rather than the accident the charts guard against. Invalidating
 * differences are stated loudly; they are not refused.
 *
 * **The diff is over the exact stored config text.** Invariant 5 makes that text the
 * config — what is stored is what runs — so the honest answer to "what changed" is which
 * bytes changed. Comparing parsed settings instead would need opinions about vLLM's
 * option set that rot with every release, would normalize away the comment an author
 * wrote to explain a value, and would call two configs identical when one declares a key
 * twice in a way that changes what the engine does.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import { analysisApi } from "./api";
import type { Filters } from "./AnalysisFilters";
import { toQuery } from "./AnalysisFilters";
import type { Analysis, AnalysisPoint, Comparison, MetricComparison } from "./types";

function format(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(2);
}

function changeLabel(metric: MetricComparison): string {
  if (metric.change === null) return "—";
  if (metric.change === 0) return "no change";
  const pct = `${metric.change > 0 ? "+" : ""}${(metric.change * 100).toFixed(1)}%`;
  // Direction and judgement are separate: a latency metric falling is a fall *and* an
  // improvement, and collapsing them would hide which way the number actually moved.
  return `${pct} ${metric.is_improvement ? "better" : "worse"}`;
}

function pointLabel(point: AnalysisPoint): string {
  return `${point.config_name} · ${point.workload_name} (TP ${point.tensor_parallel_size})`;
}

function DiffPanel({ comparison }: { comparison: Comparison }) {
  if (comparison.configs_identical) {
    return (
      <p className="muted">
        Both sides ran the same configuration — identical by content hash, so identical
        byte for byte. Any difference in the numbers came from the workload or from
        run-to-run variance.
      </p>
    );
  }
  return (
    <pre className="diff">
      {comparison.config_diff.map((line, index) => (
        <div key={index} className={`diff-${line.kind}`}>
          <span className="diff-gutter">
            {line.kind === "added" ? "+" : line.kind === "removed" ? "−" : " "}
          </span>
          {line.text || " "}
        </div>
      ))}
    </pre>
  );
}

function PointPicker({
  label,
  points,
  value,
  onChange,
}: {
  label: string;
  points: AnalysisPoint[];
  value: string;
  onChange: (id: string) => void;
}) {
  return (
    <label>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {points.map((point) => (
          <option key={point.point_id} value={point.point_id}>
            {pointLabel(point)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function CompareView({ filters }: { filters: Filters }) {
  const [available, setAvailable] = useState<AnalysisPoint[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [left, setLeft] = useState<string | null>(null);
  const [right, setRight] = useState<string | null>(null);

  const loadPoints = useCallback(async () => {
    try {
      const data: Analysis = await analysisApi.points(toQuery(filters));
      // Flattened across comparability groups on purpose: this view is allowed to
      // compare across them, and hiding the other groups behind a selector would make
      // the one comparison it uniquely permits the hardest one to ask for.
      const points = data.groups.flatMap((g) => g.points);
      setAvailable(points);
      setLeft(points[0]?.point_id ?? null);
      setRight(points[1]?.point_id ?? points[0]?.point_id ?? null);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoaded(true);
    }
  }, [filters]);

  useEffect(() => {
    void loadPoints();
  }, [loadPoints]);

  useEffect(() => {
    if (!left || !right) {
      setComparison(null);
      return;
    }
    let cancelled = false;
    analysisApi
      .compare(left, right, filters.source)
      .then((c) => !cancelled && setComparison(c))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [left, right, filters.source]);

  const invalidating = useMemo(
    () => (comparison?.provenance_differences ?? []).filter((d) => d.invalidating),
    [comparison],
  );
  const notable = useMemo(
    () => (comparison?.provenance_differences ?? []).filter((d) => !d.invalidating),
    [comparison],
  );

  return (
    <section>
      <h2>Compare two points</h2>

      <div className="card">
        <div className="toolbar">
          {available.length > 0 && left && right && (
            <>
              <PointPicker label="Left" points={available} value={left} onChange={setLeft} />
              <PointPicker label="Right" points={available} value={right} onChange={setRight} />
            </>
          )}
          <span className="spacer" />
          <button onClick={() => void loadPoints()}>Refresh</button>
        </div>

        {error && <p className="error">{error}</p>}
        {!loaded && <p className="muted">Loading…</p>}
        {loaded && !error && available.length < 2 && (
          <p className="muted">
            Two measurement points are needed to compare. There{" "}
            {available.length === 1 ? "is one" : "are none"} in this population.
          </p>
        )}

        {comparison && (
          <>
            {invalidating.length > 0 && (
              <p className="notice">
                These two are <strong>not like-for-like</strong>:{" "}
                {invalidating
                  .map((d) => `${d.label} ${d.left ?? "unknown"} against ${d.right ?? "unknown"}`)
                  .join("; ")}
                . The comparison still runs — deliberately, because comparing vLLM versions
                or hardware is a supported use — but the difference in the numbers below is
                not attributable to the configuration alone. No chart will draw these
                together.
              </p>
            )}

            {notable.length > 0 && (
              <p className="muted">
                Also differing:{" "}
                {notable
                  .map((d) => `${d.label} (${d.left ?? "—"} → ${d.right ?? "—"})`)
                  .join(", ")}
                .
              </p>
            )}

            <div className="devices">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>{comparison.left.config_name}</th>
                    <th>{comparison.right.config_name}</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.metrics.map((metric) => (
                    <tr key={metric.key}>
                      <td>
                        {metric.label} <span className="muted">{metric.unit}</span>
                      </td>
                      <td>{format(metric.left)}</td>
                      <td>{format(metric.right)}</td>
                      <td
                        className={
                          metric.is_improvement === null
                            ? undefined
                            : metric.is_improvement
                              ? "delta-better"
                              : "delta-worse"
                        }
                      >
                        {changeLabel(metric)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="muted">
              {comparison.left.replicates} replicate
              {comparison.left.replicates === 1 ? "" : "s"} against{" "}
              {comparison.right.replicates}. {comparison.left.spread_note} A change smaller
              than either side's own spread is not a result.
            </p>

            <h3>Configuration</h3>
            <DiffPanel comparison={comparison} />
          </>
        )}
      </div>
    </section>
  );
}
