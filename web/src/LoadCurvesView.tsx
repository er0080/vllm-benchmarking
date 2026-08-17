/**
 * Response to load: latency and throughput against concurrency or request rate.
 *
 * The Pareto view collapses each configuration to one operating point, which is what
 * makes it comparable — and also what it hides. A config that looks like a single dot
 * there is really a curve, and the *shape* of that curve is what tells you where it
 * breaks: the concurrency at which throughput stops rising and the queue starts
 * absorbing everything, and the point where p99 latency separates from the median.
 *
 * Two modes over the same axis, because they answer the same question from opposite
 * sides:
 *
 * **Latency** draws the median as a solid line and p99 as a dashed line of the same
 * colour. The gap between them is the finding — a config whose median holds steady while
 * its p99 triples under load is queueing, and a summary that reported only the median
 * would have called it fine. Same hue for both because they are one configuration seen at
 * two percentiles, not two configurations.
 *
 * **Throughput** draws the saturation curve. The knee is where more offered load stops
 * buying tokens per second, and beyond it every extra request is pure latency.
 *
 * The x axis is whichever workload dimension actually varies — max concurrency or request
 * rate. Only one of them usually does, and forcing the reader to choose the one their
 * sweep swept is a question the data can answer.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type * as echarts from "echarts";

import { analysisApi } from "./api";
import type { Filters } from "./AnalysisFilters";
import { toQuery } from "./AnalysisFilters";
import { Chart, SYMBOLS, cssVar, seriesColors, useTheme } from "./chartkit";
import type { Analysis, AnalysisGroup, AnalysisPoint, Metric } from "./types";

/** The palette's validated slot count. Beyond it, configs are listed rather than drawn. */
const MAX_SERIES = 4;

/** Which workload dimension the curve runs along. */
type LoadAxis = "max_concurrency" | "request_rate";

const LOAD_AXES: { key: LoadAxis; label: string; unit: string }[] = [
  { key: "max_concurrency", label: "Max concurrency", unit: "concurrent requests" },
  { key: "request_rate", label: "Request rate", unit: "req/s" },
];

/** Latency pairs drawn together: the median and the tail it hides. */
const LATENCY_PAIRS: { label: string; median: string; p99: string }[] = [
  { label: "Time to first token", median: "ttft_ms_median", p99: "ttft_ms_p99" },
  { label: "Inter-token latency", median: "itl_ms_median", p99: "itl_ms_p99" },
  { label: "Time per output token", median: "tpot_ms_mean", p99: "tpot_ms_p99" },
];

const THROUGHPUT_METRICS = [
  "total_token_throughput_per_gpu",
  "output_token_throughput_per_gpu",
  "request_throughput_req_sec",
];

type Ink = { text: string; muted: string; grid: string; surface: string };

function loadValue(point: AnalysisPoint, axis: LoadAxis): number | null {
  const raw = axis === "max_concurrency" ? point.max_concurrency : point.request_rate;
  // Null means unbounded — `--request-rate inf`, no `--max-concurrency`. Genuinely the
  // absence of a limit, so it has no position on an axis of limits and is dropped rather
  // than plotted at zero, which would be the opposite of what it means.
  return raw === null || !Number.isFinite(raw) ? null : raw;
}

/** Distinct load values present, ascending. */
function loadValues(points: AnalysisPoint[], axis: LoadAxis): number[] {
  const seen = new Set<number>();
  for (const point of points) {
    const value = loadValue(point, axis);
    if (value !== null) seen.add(value);
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * The axis a sweep actually varied.
 *
 * A workload fixes one and leaves the other unbounded, so in practice exactly one has
 * more than a single value. Picking it beats asking the reader which dimension their own
 * sweep swept.
 */
export function defaultLoadAxis(points: AnalysisPoint[]): LoadAxis {
  const concurrency = loadValues(points, "max_concurrency").length;
  const rate = loadValues(points, "request_rate").length;
  return rate > concurrency ? "request_rate" : "max_concurrency";
}

/** One config's points along the load axis, ordered. */
type Series = {
  configHash: string;
  configName: string;
  tensorParallelSize: number;
  points: { load: number; point: AnalysisPoint }[];
};

export function seriesByConfig(points: AnalysisPoint[], axis: LoadAxis): Series[] {
  const byConfig = new Map<string, Series>();
  for (const point of points) {
    const load = loadValue(point, axis);
    if (load === null) continue;
    const existing = byConfig.get(point.config_hash) ?? {
      configHash: point.config_hash,
      configName: point.config_name,
      tensorParallelSize: point.tensor_parallel_size,
      points: [],
    };
    existing.points.push({ load, point });
    byConfig.set(point.config_hash, existing);
  }
  return [...byConfig.values()]
    .map((s) => ({ ...s, points: s.points.sort((a, b) => a.load - b.load) }))
    // A single measurement is a point, not a curve. Ordering by length puts the
    // configurations with something to say first; ones measured once are dropped from
    // the chart and named in the caption instead.
    .filter((s) => s.points.length > 1)
    .sort((a, b) => b.points.length - a.points.length || a.configName.localeCompare(b.configName));
}

function baseOption(title: string, ink: Ink, axisName: string, ticks: number[]) {
  return {
    // Room at the top for a title *and* a legend on its own row. They shared a line
    // first, and with one entry per configuration the legend printed straight through
    // the title — unreadable, and only visible by rendering it.
    grid: { left: 78, right: 24, top: 64, bottom: 48 },
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif" },
    title: {
      text: title,
      left: 0,
      top: 0,
      textStyle: { color: ink.text, fontSize: 13, fontWeight: 600 },
    },
    xAxis: {
      // Category rather than a value axis. Load values are a handful of discrete
      // settings, usually doubling; on a log value axis the ticks land on decades — 1,
      // 10, 100 — while every measurement sits between them, so the reader cannot tell
      // which concurrency a point is without counting. Category labels each measured
      // value exactly, and even spacing of doubling values is what a log axis was for.
      type: "category" as const,
      data: ticks.map(String),
      boundaryGap: false,
      name: axisName,
      nameLocation: "middle" as const,
      nameGap: 30,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      axisLine: { lineStyle: { color: ink.grid } },
      axisTick: { show: false },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: ink.grid } },
    },
    yAxis: {
      type: "value" as const,
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

function line(
  name: string,
  data: (number | null)[],
  color: string,
  ink: Ink,
  symbol: string,
  dashed: boolean,
): echarts.SeriesOption {
  return {
    type: "line",
    name,
    data,
    // On, and this is the opposite of the usual rule — it is right *because* the axis is
    // shared. A null here means the tick belongs to some other configuration's sweep, not
    // that this one failed at that load; the segment either side of it joins two of this
    // config's own real measurements, exactly as it would on an axis holding only its own
    // values. What a reader must not lose is which points were measured, and symbols
    // carry that: a long segment with no marker at the tick it crosses is visibly an
    // interpolation, not a measurement.
    connectNulls: true,
    showSymbol: true,
    symbol,
    symbolSize: dashed ? 7 : 9,
    lineStyle: { width: 2, color, type: dashed ? "dashed" : "solid" },
    itemStyle: { color, borderColor: ink.surface, borderWidth: dashed ? 1 : 2 },
  };
}

/** Values aligned to the shared category axis, with a gap where a load was not run. */
function alignTo(ticks: number[], s: Series, key: string): (number | null)[] {
  const byLoad = new Map(s.points.map((p) => [p.load, p.point]));
  return ticks.map((load) => byLoad.get(load)?.metrics[key]?.median ?? null);
}

export function latencyOption(
  series: Series[],
  pair: (typeof LATENCY_PAIRS)[number],
  axisLabel: string,
  ink: Ink,
  colors: string[],
): echarts.EChartsOption | null {
  const ticks = [...new Set(series.flatMap((s) => s.points.map((p) => p.load)))].sort(
    (a, b) => a - b,
  );
  if (ticks.length === 0) return null;

  const out: echarts.SeriesOption[] = [];
  series.forEach((s, index) => {
    const color = colors[index] ?? colors[0] ?? "#2a78d6";
    const symbol = SYMBOLS[index] ?? "circle";
    for (const [key, dashed, name] of [
      [pair.median, false, s.configName],
      [pair.p99, true, `${s.configName} · p99`],
    ] as const) {
      const data = alignTo(ticks, s, key);
      if (data.some((y) => y !== null)) {
        out.push(line(name, data, color, ink, symbol, dashed));
      }
    }
  });
  if (out.length === 0) return null;

  const base = baseOption(pair.label, ink, axisLabel, ticks);
  return {
    ...base,
    legend: {
      // One entry per configuration, not per line. The p99 companions share the colour
      // and are distinguished by dash, which the caption explains — listing all eight
      // doubled the legend and pushed it through the title.
      data: series.map((s) => s.configName),
      top: 22,
      left: 0,
      itemWidth: 18,
      textStyle: { color: ink.muted, fontSize: 11 },
    },
    yAxis: { ...base.yAxis, name: "ms — median solid, p99 dashed" },
    series: out,
  };
}

export function throughputOption(
  series: Series[],
  metric: Metric,
  axisLabel: string,
  ink: Ink,
  colors: string[],
): echarts.EChartsOption | null {
  const ticks = [...new Set(series.flatMap((s) => s.points.map((p) => p.load)))].sort(
    (a, b) => a - b,
  );
  if (ticks.length === 0) return null;

  const out = series.flatMap((s, index) => {
    const data = alignTo(ticks, s, metric.key);
    if (!data.some((y) => y !== null)) return [];
    return [
      line(
        s.configName,
        data,
        colors[index] ?? colors[0] ?? "#2a78d6",
        ink,
        SYMBOLS[index] ?? "circle",
        false,
      ),
    ];
  });
  if (out.length === 0) return null;

  const base = baseOption(metric.label, ink, axisLabel, ticks);
  return {
    ...base,
    legend: {
      top: 22,
      left: 0,
      itemWidth: 18,
      textStyle: { color: ink.muted, fontSize: 11 },
    },
    yAxis: { ...base.yAxis, name: metric.unit },
    series: out,
  };
}

function LoadTable({
  series,
  axisLabel,
  pair,
  metricKey,
}: {
  series: Series[];
  axisLabel: string;
  pair: (typeof LATENCY_PAIRS)[number];
  metricKey: string;
}) {
  const cell = (point: AnalysisPoint, key: string) => {
    const spread = point.metrics[key];
    if (!spread) return "—";
    return spread.n > 1
      ? `${spread.median.toFixed(1)} (${spread.min.toFixed(1)}–${spread.max.toFixed(1)})`
      : spread.median.toFixed(1);
  };
  return (
    <div className="devices">
      <table>
        <thead>
          <tr>
            <th>Config</th>
            <th>{axisLabel}</th>
            <th>{pair.label} median</th>
            <th>{pair.label} p99</th>
            <th>Throughput</th>
            <th>Replicates</th>
          </tr>
        </thead>
        <tbody>
          {series.flatMap((s) =>
            s.points.map(({ load, point }) => (
              <tr key={`${s.configHash}:${load}`}>
                <td>{s.configName}</td>
                <td>{load}</td>
                <td>{cell(point, pair.median)}</td>
                <td>{cell(point, pair.p99)}</td>
                <td>{cell(point, metricKey)}</td>
                <td title={point.spread_note}>{point.replicates}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

export function LoadCurvesView({ filters }: { filters: Filters }) {
  const [data, setData] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [groupId, setGroupId] = useState<string | null>(null);
  const [axis, setAxis] = useState<LoadAxis | null>(null);
  const [pairLabel, setPairLabel] = useState(LATENCY_PAIRS[0]!.label);
  const [metricKey, setMetricKey] = useState(THROUGHPUT_METRICS[0]!);
  const [showTable, setShowTable] = useState(false);
  const theme = useTheme();

  const load = useCallback(async () => {
    try {
      setData(await analysisApi.points(toQuery(filters)));
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

  const group: AnalysisGroup | null = useMemo(() => {
    if (!data || data.groups.length === 0) return null;
    return data.groups.find((g) => g.group_id === groupId) ?? data.groups[0] ?? null;
  }, [data, groupId]);

  const chosenAxis = axis ?? (group ? defaultLoadAxis(group.points) : "max_concurrency");
  const allSeries = useMemo(
    () => (group ? seriesByConfig(group.points, chosenAxis) : []),
    [group, chosenAxis],
  );
  const drawn = allSeries.slice(0, MAX_SERIES);

  const pair = LATENCY_PAIRS.find((p) => p.label === pairLabel) ?? LATENCY_PAIRS[0]!;
  const metric = data?.metrics.find((m) => m.key === metricKey);
  const axisLabel = LOAD_AXES.find((a) => a.key === chosenAxis)?.label ?? "Load";

  const options = useMemo(() => {
    if (drawn.length === 0) return null;
    const colors = seriesColors();
    const ink: Ink = {
      text: cssVar("--text", "#16181d"),
      muted: cssVar("--text-muted", "#6b7280"),
      grid: cssVar("--grid-line", "#e8eaee"),
      surface: cssVar("--surface", "#ffffff"),
    };
    return {
      latency: latencyOption(drawn, pair, axisLabel, ink, colors),
      throughput: metric ? throughputOption(drawn, metric, axisLabel, ink, colors) : null,
    };
    // theme: ECharts bakes colours in at option time.
  }, [drawn, pair, metric, axisLabel, theme]);

  const singlePoint = group ? new Set(group.points.map((p) => p.config_hash)).size - allSeries.length : 0;

  return (
    <section>
      <h2>Response to load</h2>

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
            Load axis
            <select
              value={chosenAxis}
              onChange={(e) => setAxis(e.target.value as LoadAxis)}
            >
              {LOAD_AXES.map((a) => (
                <option key={a.key} value={a.key}>
                  {a.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Latency
            <select value={pair.label} onChange={(e) => setPairLabel(e.target.value)}>
              {LATENCY_PAIRS.map((p) => (
                <option key={p.label} value={p.label}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Throughput
            <select value={metricKey} onChange={(e) => setMetricKey(e.target.value)}>
              {THROUGHPUT_METRICS.map((key) => (
                <option key={key} value={key}>
                  {data?.metrics.find((m) => m.key === key)?.label ?? key}
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

        {loaded && !error && allSeries.length === 0 && (
          <p className="muted">
            No load curves yet. A curve needs one configuration measured against two or
            more workloads that differ in {axisLabel.toLowerCase()} — a sweep with several
            workloads on its axis produces one.
          </p>
        )}

        {group && options && (
          <>
            <p className="muted" style={{ marginTop: 0 }}>
              {group.label}.
              {allSeries.length > drawn.length && (
                <>
                  {" "}
                  Showing {drawn.length} of {allSeries.length} configurations; the palette
                  has {MAX_SERIES} colour-blind-validated slots. The table below has them
                  all.
                </>
              )}
              {singlePoint > 0 && (
                <>
                  {" "}
                  {singlePoint} configuration{singlePoint === 1 ? " was" : "s were"}{" "}
                  measured at only one load and cannot be drawn as a curve.
                </>
              )}
            </p>

            {group.warnings.map((w) => (
              <p className="notice" key={w}>
                {w}.
              </p>
            ))}

            {options.latency ? (
              <Chart option={options.latency} height={260} />
            ) : (
              <p className="muted">No {pair.label.toLowerCase()} recorded for these runs.</p>
            )}
            {options.throughput && <Chart option={options.throughput} height={260} />}

            <p className="muted">
              A median that holds while its p99 pulls away is queueing: requests are
              waiting rather than generating, and a summary reporting only the median
              would call that configuration healthy. On the throughput chart the knee is
              where extra load stops buying tokens per second and starts buying only
              latency.
            </p>

            {showTable && (
              <LoadTable
                series={allSeries}
                axisLabel={axisLabel}
                pair={pair}
                metricKey={metricKey}
              />
            )}
          </>
        )}
      </div>
    </section>
  );
}
