/**
 * Tensor-parallel scaling: does TP=N earn its extra devices?
 *
 * The Pareto view answers "which configuration should I run"; this one answers the
 * question that decides how much hardware to buy. They need different charts because
 * adding devices moves two numbers in opposite directions and only one of them is the
 * cost: aggregate throughput goes up, which is what an operator feels, while per-device
 * throughput usually goes down, which is what the bill reflects.
 *
 * **Two stacked charts, never one with two y-axes.** Aggregate tokens per second and a
 * dimensionless efficiency ratio are different units, and a dual-axis chart lets any two
 * series be made to look correlated by choosing the scales. They share the x axis
 * instead.
 *
 * **The 1.0 line on the efficiency chart is the whole point.** It is where every added
 * device pulled its weight. Drawing an "ideal linear" reference on the throughput chart
 * would say the same thing but needs one dashed line per series; expressed as efficiency
 * it is a single horizontal rule, and the gap below it is the waste.
 *
 * **A curve is one config family running one workload.** Both halves matter. Points from
 * different configurations are a comparison of configurations, not a scaling
 * measurement; points from different workloads measure the traffic rather than the
 * topology. Neither is scaling and both would look exactly like it. The API enforces
 * this; the view chooses a family and draws one series per workload, because scaling
 * frequently differs by load and that difference is usually the finding.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type * as echarts from "echarts";

import { analysisApi } from "./api";
import { Chart, DASH, SYMBOLS, cssVar, seriesColors, useTheme } from "./chartkit";
import type { Metric, RunSource, Scaling, ScalingCurve, ScalingGroup } from "./types";

/** The palette's validated slot count. Beyond it, series are listed rather than drawn. */
const MAX_SERIES = 4;

function format(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function percent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(0)}%`;
}

/** Every tensor-parallel width on any of these curves, ascending. */
export function widths(curves: ScalingCurve[]): number[] {
  return [...new Set(curves.flatMap((c) => c.steps.map((s) => s.tensor_parallel_size)))].sort(
    (a, b) => a - b,
  );
}

type Ink = { text: string; muted: string; grid: string; surface: string };

function baseOption(
  title: string,
  ink: Ink,
  axis: string[],
  showAxisName: boolean,
  yName: string,
) {
  return {
    grid: { left: 76, right: 24, top: 40, bottom: showAxisName ? 48 : 28 },
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif" },
    title: {
      text: title,
      left: 0,
      top: 0,
      textStyle: { color: ink.text, fontSize: 13, fontWeight: 600 },
    },
    xAxis: {
      // Category, not value: tensor-parallel sizes double, so 1-2-4-8 on a linear axis
      // would crush the interesting low end into the left margin.
      type: "category" as const,
      data: axis,
      name: showAxisName ? "tensor-parallel size" : "",
      nameLocation: "middle" as const,
      nameGap: 30,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      axisLine: { lineStyle: { color: ink.grid } },
      axisTick: { show: false },
      axisLabel: { color: ink.muted, fontSize: 11 },
      boundaryGap: true,
    },
    yAxis: {
      type: "value" as const,
      // Named on both charts. One is tokens per second and the other is a percentage of
      // a baseline; an unlabelled axis leaves the reader to infer which from the tick
      // values, and 100 is a plausible tick on either.
      name: yName,
      nameLocation: "middle" as const,
      nameGap: 58,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: ink.grid } },
    },
    tooltip: { trigger: "axis" as const, confine: true },
  };
}

function seriesFor(
  curves: ScalingCurve[],
  axis: number[],
  ink: Ink,
  colors: string[],
  pick: (step: ScalingCurve["steps"][number]) => number | null,
): echarts.SeriesOption[] {
  return curves.map((curve, index) => {
    const color = colors[index] ?? colors[0] ?? "#2a78d6";
    const byWidth = new Map(curve.steps.map((s) => [s.tensor_parallel_size, s]));
    return {
      type: "line",
      name: curve.workload_name,
      // Aligned to the shared category axis, with a null where a width was not measured
      // — connectNulls stays off so a gap reads as "not measured" rather than as a
      // straight line through a point nobody ran.
      data: axis.map((tp) => {
        const step = byWidth.get(tp);
        return step ? pick(step) : null;
      }),
      connectNulls: false,
      symbol: SYMBOLS[index] ?? "circle",
      symbolSize: 9,
      lineStyle: { width: 2, color, type: DASH[index % DASH.length] ?? "solid" },
      itemStyle: { color, borderColor: ink.surface, borderWidth: 2 },
    };
  });
}

/**
 * Aggregate throughput spans an order of magnitude across workloads — a low-concurrency
 * curve and a high-concurrency one differ by 8x here — and on a shared linear axis the
 * quiet one flattens against the floor and stops being readable at all.
 *
 * So the scale follows the data and the axis says which it used. Switching silently
 * would be the trap; a chart that is unreadable for half its inputs is the alternative.
 */
const LOG_SCALE_THRESHOLD = 10;

export function throughputOption(
  curves: ScalingCurve[],
  axis: number[],
  ink: Ink,
  colors: string[],
): echarts.EChartsOption {
  const series = seriesFor(curves, axis, ink, colors, (s) => s.aggregate_median);
  const values = curves
    .flatMap((c) => c.steps.map((s) => s.aggregate_median))
    .filter((v): v is number => v !== null && v > 0);
  const useLog =
    values.length > 1 && Math.max(...values) / Math.min(...values) > LOG_SCALE_THRESHOLD;
  const base = baseOption(
    "Aggregate throughput",
    ink,
    axis.map(String),
    false,
    useLog ? "tok/s (log scale)" : "tok/s",
  );

  return {
    ...base,
    legend: {
      top: 0,
      right: 0,
      itemWidth: 18,
      textStyle: { color: ink.muted, fontSize: 11 },
    },
    yAxis: { ...base.yAxis, ...(useLog ? { type: "log" as const } : {}) },
    series,
  };
}

export function efficiencyOption(
  curves: ScalingCurve[],
  axis: number[],
  ink: Ink,
  colors: string[],
): echarts.EChartsOption {
  const series = seriesFor(curves, axis, ink, colors, (s) =>
    s.efficiency === null ? null : s.efficiency * 100,
  );
  const first = series[0];
  if (first && first.type === "line") {
    // The reference every point is read against: 100% is every added device pulling its
    // weight. Attached to the first series rather than drawn as its own so it carries no
    // legend entry — it is the axis's meaning, not another measurement.
    first.markLine = {
      silent: true,
      symbol: "none",
      label: {
        formatter: "every device pulling its weight",
        color: ink.muted,
        fontSize: 10,
        position: "insideEndTop",
      },
      lineStyle: { color: ink.muted, type: "dashed", width: 1 },
      data: [{ yAxis: 100 }],
    };
  }
  const base = baseOption(
    "Per-GPU efficiency, against the narrowest width measured",
    ink,
    axis.map(String),
    true,
    "% of baseline per-GPU rate",
  );
  return {
    ...base,
    // No second legend: the series are the same as the chart above and it sits directly
    // alongside. Repeating it would spend ink restating what the reader just read.
    legend: { show: false },
    yAxis: {
      ...base.yAxis,
      // Zero-based on purpose. Efficiency is read against 100%, and cropping the axis to
      // the data would turn a 7-point spread into a dramatic-looking collapse — the
      // classic way a ratio chart lies.
      min: 0,
      axisLabel: { color: ink.muted, fontSize: 11, formatter: "{value}%" },
    },
    series,
  };
}

function CurveTable({ curves }: { curves: ScalingCurve[] }) {
  return (
    <div className="devices">
      <table>
        <thead>
          <tr>
            <th>Workload</th>
            <th>TP</th>
            <th>GPUs</th>
            <th>Per GPU</th>
            <th>Aggregate</th>
            <th>Speed-up</th>
            <th>Efficiency</th>
          </tr>
        </thead>
        <tbody>
          {curves.flatMap((curve) =>
            curve.steps.map((step) => (
              <tr key={`${curve.family}:${curve.workload_hash}:${step.tensor_parallel_size}`}>
                <td>{curve.workload_name}</td>
                <td>{step.tensor_parallel_size}</td>
                <td>{step.gpu_count}</td>
                <td>
                  {step.per_gpu
                    ? step.per_gpu.n > 1
                      ? `${format(step.per_gpu.median)} (${format(step.per_gpu.min)}–${format(step.per_gpu.max)})`
                      : format(step.per_gpu.median)
                    : "—"}
                </td>
                <td>{format(step.aggregate_median)}</td>
                <td>{step.speedup === null ? "—" : `${format(step.speedup, 2)}×`}</td>
                <td>{percent(step.efficiency)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

export function ScalingView() {
  const [data, setData] = useState<Scaling | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [source, setSource] = useState<RunSource>("real");
  const [groupId, setGroupId] = useState<string | null>(null);
  const [family, setFamily] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  const theme = useTheme();

  const load = useCallback(async () => {
    try {
      setData(await analysisApi.scaling({ source }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoaded(true);
    }
  }, [source]);

  useEffect(() => {
    void load();
  }, [load]);

  const group: ScalingGroup | null = useMemo(() => {
    if (!data || data.groups.length === 0) return null;
    return data.groups.find((g) => g.group_id === groupId) ?? data.groups[0] ?? null;
  }, [data, groupId]);

  // Families in this group, most-measured first — the default selection is the one with
  // the most to say rather than whichever sorted first.
  const families = useMemo(() => {
    if (!group) return [];
    const seen = new Map<string, { family: string; name: string; curves: number }>();
    for (const curve of group.curves) {
      const entry = seen.get(curve.family) ?? {
        family: curve.family,
        name: curve.config_name,
        curves: 0,
      };
      entry.curves += 1;
      seen.set(curve.family, entry);
    }
    return [...seen.values()].sort((a, b) => b.curves - a.curves);
  }, [group]);

  const chosenFamily = family ?? families[0]?.family ?? null;
  const curves = useMemo(
    () => (group ? group.curves.filter((c) => c.family === chosenFamily) : []),
    [group, chosenFamily],
  );
  const drawn = curves.slice(0, MAX_SERIES);
  const axis = useMemo(() => widths(drawn), [drawn]);

  const metric: Metric | undefined = data?.metrics.find((m) => m.key === data.metric);
  const relativeBaseline = drawn.some((c) => !c.baseline_is_single_gpu);

  const options = useMemo(() => {
    if (drawn.length === 0 || axis.length === 0) return null;
    const colors = seriesColors();
    const ink: Ink = {
      text: cssVar("--text", "#16181d"),
      muted: cssVar("--text-muted", "#6b7280"),
      grid: cssVar("--grid-line", "#e8eaee"),
      surface: cssVar("--surface", "#ffffff"),
    };
    return {
      throughput: throughputOption(drawn, axis, ink, colors),
      efficiency: efficiencyOption(drawn, axis, ink, colors),
    };
    // theme: ECharts bakes colours in at option time.
  }, [drawn, axis, theme]);

  return (
    <section>
      <h2>Tensor-parallel scaling</h2>

      <div className="card">
        <div className="toolbar">
          <label>
            Runs
            <select value={source} onChange={(e) => setSource(e.target.value as RunSource)}>
              <option value="real">Real measurements</option>
              <option value="synthetic">Synthetic (mock / CPU backend)</option>
            </select>
          </label>

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

          {families.length > 1 && (
            <label>
              Configuration
              <select value={chosenFamily ?? ""} onChange={(e) => setFamily(e.target.value)}>
                {families.map((f) => (
                  <option key={f.family} value={f.family}>
                    {f.name} ({f.curves} workload{f.curves === 1 ? "" : "s"})
                  </option>
                ))}
              </select>
            </label>
          )}

          <span className="spacer" />
          <button onClick={() => setShowTable((s) => !s)}>
            {showTable ? "Hide table" : "Show table"}
          </button>
          <button onClick={() => void load()}>Refresh</button>
        </div>

        {source === "synthetic" && (
          <p className="notice">
            Synthetic runs. These come from the mock agent or the CPU backend and are not
            measurements of any real hardware.
          </p>
        )}

        {error && <p className="error">{error}</p>}
        {!loaded && <p className="muted">Loading…</p>}

        {loaded && !error && data && data.groups.length === 0 && (
          <p className="muted">
            No scaling curves yet. A curve needs one configuration measured at two or more
            tensor-parallel sizes against the same workload — run a sweep with a
            tensor-parallel axis to produce one.
            {data.single_width_families > 0 && (
              <>
                {" "}
                {data.single_width_families} configuration
                {data.single_width_families === 1 ? " was" : "s were"} measured at only one
                width.
              </>
            )}
          </p>
        )}

        {group && options && (
          <>
            <p className="muted" style={{ marginTop: 0 }}>
              {group.label} — {metric?.label ?? data?.metric}, normalized per device.
              {curves.length > drawn.length && (
                <>
                  {" "}
                  Showing {drawn.length} of {curves.length} workloads; the palette has{" "}
                  {MAX_SERIES} colour-blind-validated slots. The table below has them all.
                </>
              )}
            </p>

            {group.warnings.map((w) => (
              <p className="notice" key={w}>
                {w}.
              </p>
            ))}

            {relativeBaseline && (
              <p className="notice">
                Efficiency here is measured against the narrowest width that was actually
                run, which was not a single GPU. That makes it a ratio between two
                already-parallel configurations, not the parallel efficiency the number
                normally means — benchmark the same config at TP=1 to get that.
              </p>
            )}

            <Chart option={options.throughput} height={230} />
            <Chart option={options.efficiency} height={250} />

            <p className="muted">
              Above the line means added devices more than paid for themselves — real, and
              usually a sign the model only fits in KV cache once it is split. Below it,
              the gap is the share of each device that the extra width wasted.
            </p>

            {showTable && <CurveTable curves={curves} />}
          </>
        )}
      </div>
    </section>
  );
}
