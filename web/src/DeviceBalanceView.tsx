/**
 * Per-device balance within a tensor-parallel run.
 *
 * This is the view that justifies keying `gpu_sample` per device rather than storing a
 * host-level average. One GPU at 95% while its peer sits at 60% means a third of a device
 * was idle for the entire run — and the average of those two, 77.5%, looks like a
 * perfectly healthy machine. No amount of later querying recovers the difference once it
 * has been averaged at write time, which is why the schema refuses to.
 *
 * **Grouped bars, one bar per device, one group per run.** Imbalance is then literally
 * the difference in bar heights inside a group; no derived statistic has to be trusted
 * for the reader to see it. Colour follows the device index, fixed, because a device that
 * lags across several runs is a hardware finding rather than a configuration one, and
 * that pattern is only visible if device 1 is the same colour in every group.
 *
 * **Per run, not per measurement point.** Every other analysis view aggregates replicates,
 * because the question there is what a configuration does. Here the question is what one
 * execution did, and averaging three replicates would hide the single run that went
 * wrong — which is the only run anyone opens this view to find. So runs are listed
 * individually, worst imbalance first.
 *
 * **A single-device run is flagged, not scored.** "No imbalance measurable" and "perfectly
 * balanced" are different facts; reporting 0.0 for a TP=1 run would state the second when
 * only the first is true.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type * as echarts from "echarts";

import { analysisApi } from "./api";
import type { Filters } from "./AnalysisFilters";
import { toQuery } from "./AnalysisFilters";
import { Chart, cssVar, seriesColors, useTheme } from "./chartkit";
import type { DeviceBalance, DeviceBalanceGroup, RunBalance } from "./types";

/** Runs charted at once, worst first. The table below carries the rest. */
const MAX_RUNS = 8;

/** Imbalance past which a run is called out in words as well as drawn. */
const NOTABLE_IMBALANCE = 0.1;

type Ink = { text: string; muted: string; grid: string; surface: string };

type BarMetric = {
  key: "sm_utilization_pct" | "memory_used_bytes" | "power_watts";
  title: string;
  unit: string;
  scale?: (v: number) => number;
  max?: number;
};

const BAR_METRICS = [
  { key: "sm_utilization_pct", title: "SM utilization", unit: "%", max: 100 },
  {
    key: "memory_used_bytes",
    title: "Peak memory per device",
    unit: " GiB",
    // Peak rather than mean, which is what the API aggregates: VRAM is claimed and held,
    // so the high-water mark is what the device actually needed.
    scale: (v) => v / 1024 ** 3,
  },
  { key: "power_watts", title: "Power draw", unit: " W" },
] as const satisfies readonly BarMetric[];

function runLabel(run: RunBalance, replicated: boolean): string {
  // The replicate number only appears when there is more than one, because this view
  // lists executions rather than points: three replicates were otherwise three
  // identically labelled bar groups with no way to tell which one went wrong.
  const base = `${run.config_name} · ${run.workload_name}`;
  return replicated ? `${base} #${run.replicate_idx + 1}` : base;
}

/** Points measured more than once in this set — the ones whose labels need a number. */
function replicatedPoints(runs: RunBalance[]): Set<string> {
  const counts = new Map<string, number>();
  for (const run of runs) {
    const key = `${run.config_name}\u0000${run.workload_name}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return new Set([...counts].filter(([, n]) => n > 1).map(([key]) => key));
}

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(0)}%`;
}

export function balanceOption(
  runs: RunBalance[],
  metric: BarMetric,
  ink: Ink,
  colors: string[],
): echarts.EChartsOption | null {
  const deviceIndices = [...new Set(runs.flatMap((r) => r.devices.map((d) => d.gpu_index)))].sort(
    (a, b) => a - b,
  );
  if (deviceIndices.length === 0) return null;
  const replicated = replicatedPoints(runs);
  const label = (run: RunBalance) =>
    runLabel(run, replicated.has(`${run.config_name}\u0000${run.workload_name}`));

  const series: echarts.SeriesOption[] = deviceIndices.map((index) => ({
    type: "bar",
    name: `GPU ${index}`,
    // Keyed to the device index, not to its position in this chart's list. A run that
    // held only devices 2 and 3 must not paint them with device 0 and 1's colours.
    itemStyle: {
      color: colors[index % colors.length] ?? colors[0] ?? "#2a78d6",
      // 4px rounded data-ends, anchored to the baseline.
      borderRadius: [4, 4, 0, 0],
    },
    // A surface-coloured gap between adjacent bars, so a group reads as separate devices
    // rather than one wide block.
    barGap: "12%",
    barCategoryGap: "36%",
    data: runs.map((run) => {
      const device = run.devices.find((d) => d.gpu_index === index);
      const raw = device?.[metric.key] ?? null;
      return raw === null ? null : (metric.scale ? metric.scale(raw) : raw);
    }),
  }));

  return {
    // Right margin holds the last rotated tick label, which extends past its own
    // category and is otherwise clipped at the canvas edge.
    grid: { left: 70, right: 44, top: 62, bottom: 78 },
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif" },
    title: {
      text: metric.title,
      left: 0,
      top: 0,
      textStyle: { color: ink.text, fontSize: 13, fontWeight: 600 },
    },
    legend: {
      top: 22,
      left: 0,
      itemWidth: 14,
      textStyle: { color: ink.muted, fontSize: 11 },
    },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, confine: true },
    xAxis: {
      type: "category",
      data: runs.map(label),
      axisLine: { lineStyle: { color: ink.grid } },
      axisTick: { show: false },
      // Rotated because config names are long and a horizontal label set would either
      // overlap or be silently dropped by the layout.
      axisLabel: { color: ink.muted, fontSize: 10, rotate: 30, hideOverlap: false },
    },
    yAxis: {
      type: "value",
      name: metric.unit.trim(),
      nameLocation: "middle",
      nameGap: 50,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      ...(metric.max !== undefined ? { max: metric.max } : {}),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: ink.grid } },
    },
    series,
  };
}

function BalanceTable({ group }: { group: DeviceBalanceGroup }) {
  return (
    <div className="devices">
      <table>
        <thead>
          <tr>
            <th>Config</th>
            <th>Workload</th>
            <th>TP</th>
            <th>Per-device SM %</th>
            <th>Utilization gap</th>
            <th>Memory gap</th>
          </tr>
        </thead>
        <tbody>
          {group.runs.map((run) => (
            <tr key={run.run_id}>
              <td>{run.config_name}</td>
              <td>{run.workload_name}</td>
              <td>{run.tensor_parallel_size}</td>
              <td>
                {run.devices
                  .map((d) =>
                    d.sm_utilization_pct === null ? "—" : d.sm_utilization_pct.toFixed(0),
                  )
                  .join(" / ")}
              </td>
              <td>
                {run.is_single_device ? "one device" : percent(run.imbalances["sm_utilization_pct"])}
              </td>
              <td>
                {run.is_single_device ? "" : percent(run.imbalances["memory_used_bytes"])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DeviceBalanceView({ filters }: { filters: Filters }) {
  const [data, setData] = useState<DeviceBalance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [groupId, setGroupId] = useState<string | null>(null);
  const [metricKey, setMetricKey] = useState<BarMetric["key"]>("sm_utilization_pct");
  const [showTable, setShowTable] = useState(false);
  const theme = useTheme();

  const load = useCallback(async () => {
    try {
      setData(await analysisApi.deviceBalance(toQuery(filters)));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoaded(true);
    }
  }, [filters]);

  useEffect(() => {
    void load();
  }, [load]);

  const group: DeviceBalanceGroup | null = useMemo(() => {
    if (!data || data.groups.length === 0) return null;
    return data.groups.find((g) => g.group_id === groupId) ?? data.groups[0] ?? null;
  }, [data, groupId]);

  // Single-device runs are excluded from the chart: a group of one bar shows nothing
  // about balance, and including them would dilute the runs that have something to say.
  const charted = useMemo(
    () => (group ? group.runs.filter((r) => !r.is_single_device).slice(0, MAX_RUNS) : []),
    [group],
  );
  const metric = BAR_METRICS.find((m) => m.key === metricKey) ?? BAR_METRICS[0];
  const worst = charted[0];

  const option = useMemo(() => {
    if (charted.length === 0) return null;
    const ink: Ink = {
      text: cssVar("--text", "#16181d"),
      muted: cssVar("--text-muted", "#6b7280"),
      grid: cssVar("--grid-line", "#e8eaee"),
      surface: cssVar("--surface", "#ffffff"),
    };
    return balanceOption(charted, metric, ink, seriesColors());
    // theme: ECharts bakes colours in at option time.
  }, [charted, metric, theme]);

  const singleDevice = group ? group.runs.filter((r) => r.is_single_device).length : 0;

  return (
    <section>
      <h2>Per-device balance</h2>

      <div className="card">
        <div className="toolbar">

          {data && data.groups.length > 1 && (
            <label>
              Comparison set
              <select value={group?.group_id ?? ""} onChange={(e) => setGroupId(e.target.value)}>
                {data.groups.map((g) => (
                  <option key={g.group_id} value={g.group_id}>
                    {g.label} ({g.run_count} runs)
                  </option>
                ))}
              </select>
            </label>
          )}

          <label>
            Metric
            <select
              value={metricKey}
              onChange={(e) => setMetricKey(e.target.value as BarMetric["key"])}
            >
              {BAR_METRICS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.title}
                </option>
              ))}
            </select>
          </label>

          <span className="spacer" />
          <button onClick={() => setShowTable((s) => !s)}>
            {showTable ? "Hide table" : "Show table"}
          </button>
          <button onClick={() => void load()}>Refresh</button>
        </div>


        {error && <p className="error">{error}</p>}
        {!loaded && <p className="muted">Loading…</p>}

        {loaded && !error && charted.length === 0 && (
          <p className="muted">
            Nothing to compare yet. This view needs a run that held more than one GPU and
            was sampled while it ran.
            {singleDevice > 0 && (
              <>
                {" "}
                {singleDevice} single-device run{singleDevice === 1 ? "" : "s"} matched;
                one device has no balance to report.
              </>
            )}
            {data && data.runs_without_telemetry > 0 && (
              <>
                {" "}
                {data.runs_without_telemetry} run
                {data.runs_without_telemetry === 1 ? "" : "s"} produced no per-device
                samples at all.
              </>
            )}
          </p>
        )}

        {group && option && (
          <>
            <p className="muted" style={{ marginTop: 0 }}>
              {group.label} — {charted.length} multi-GPU run
              {charted.length === 1 ? "" : "s"}, worst imbalance first.
              {group.runs.length - singleDevice > charted.length && (
                <>
                  {" "}
                  {group.runs.length - singleDevice - charted.length} more in the table.
                </>
              )}
            </p>

            {group.warnings.map((w) => (
              <p className="notice" key={w}>
                {w}.
              </p>
            ))}

            {worst && (worst.worst_imbalance ?? 0) >= NOTABLE_IMBALANCE && (
              <p className="notice">
                {runLabel(worst, false)} left{" "}
                {percent(worst.imbalances["sm_utilization_pct"])} of a device on the
                table: its quietest GPU did that much less work than its busiest. A
                host-level average would have reported the mean of the two and looked
                healthy.
              </p>
            )}

            <Chart option={option} height={330} />

            <p className="muted">
              One bar per GPU, one group per run. Uneven bars inside a group are work that
              did not split evenly — the signal a host-level average destroys, and the
              reason these samples are stored per device.
              {data && data.runs_without_telemetry > 0 && (
                <>
                  {" "}
                  {data.runs_without_telemetry} matching run
                  {data.runs_without_telemetry === 1 ? " was" : "s were"} never sampled and
                  cannot be shown; that is not a clean bill of health.
                </>
              )}
            </p>

            {showTable && <BalanceTable group={group} />}
          </>
        )}
      </div>
    </section>
  );
}
