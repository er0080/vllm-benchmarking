/**
 * The run detail timeline: what the engine and each GPU were doing, second by second.
 *
 * This is the view that turns "which config won" into "why". A p99 TTFT of 25s says a
 * configuration was slow; a waiting queue climbing while KV cache sits at 11% says the
 * engine was admission-limited, not memory-limited — and those have opposite fixes.
 *
 * Three stacked charts rather than one with several axes. A dual-axis chart lets any two
 * series be made to look correlated by choosing the scales, and every reader has to
 * check which axis a line belongs to before believing it. Requests, percentages and
 * watts are different units, so they get different plots over a shared time axis.
 *
 * Per device, never averaged. One GPU at 60% while its peer sits at 95% is the finding
 * that makes a tensor-parallel run diagnosable; the mean of the two is the one summary
 * guaranteed to hide it (invariant 8).
 */
import { useEffect, useMemo, useState } from "react";
import type * as echarts from "echarts";

import { Chart, DASH, cssVar, seriesColors, useTheme } from "./chartkit";

import { runsApi } from "./api";
import type { RunTelemetry } from "./types";

/** Which per-device metric the GPU chart is showing. */
type GpuMetric = {
  key: "sm_utilization_pct" | "memory_used_bytes" | "power_watts" | "temperature_c" | "sm_clock_mhz";
  label: string;
  unit: string;
  /** Converts the stored value into the unit shown. Storage stays raw; display converts. */
  scale?: (v: number) => number;
  max?: number;
};

const GPU_METRICS = [
  { key: "sm_utilization_pct", label: "SM utilization", unit: "%", max: 100 },
  {
    key: "memory_used_bytes",
    label: "Memory used",
    unit: " GiB",
    scale: (v) => v / 1024 ** 3,
  },
  { key: "power_watts", label: "Power", unit: " W" },
  { key: "temperature_c", label: "Temperature", unit: " °C" },
  { key: "sm_clock_mhz", label: "SM clock", unit: " MHz" },
] as const satisfies readonly GpuMetric[];

const DEFAULT_METRIC: GpuMetric = GPU_METRICS[0];

/**
 * Line style per device, on top of color.
 *
 * Not decoration. Two GPUs in a tensor-parallel run frequently sit at *exactly* the same
 * utilization — that is what balanced work looks like — and two solid lines at the same
 * value render as one, so the later series silently erases the earlier one. A chart whose
 * whole purpose is showing each device separately must not disappear a device when the
 * devices agree. Keyed to position in the device list, so a device keeps its style.
 *
 * It doubles as the secondary encoding that makes the series distinguishable without
 * color, for colorblind readers and for print.
 */
/** Seconds from the first sample, so the axis reads as "into the run". */
function relativeSeconds(iso: string, origin: number): number {
  return (new Date(iso).getTime() - origin) / 1000;
}

/** Shared axis, grid and tooltip styling — recessive, so the data carries the ink. */
function baseOption(
  title: string,
  text: string,
  muted: string,
  grid: string,
  /** Only the bottom chart names the axis — all three share it, and repeating the
      label three times spends vertical space on information already given. */
  showAxisName = false,
): echarts.EChartsOption {
  return {
    // Right margin holds the end-of-line labels; without it they render past the
    // canvas and are simply invisible, which is how a chart loses its direct labels
    // without anything looking broken.
    grid: { left: 62, right: 96, top: 36, bottom: showAxisName ? 42 : 26 },
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif" },
    tooltip: {
      trigger: "axis",
      // Crosshair: on a time series the question is always "what was everything doing
      // at this instant", which a per-mark tooltip cannot answer.
      axisPointer: { type: "cross", label: { backgroundColor: muted } },
      confine: true,
    },
    xAxis: {
      type: "value",
      name: showAxisName ? "seconds into run" : "",
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { lineStyle: { color: grid } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: grid } },
    },
    title: {
      text: title,
      left: 0,
      top: 0,
      textStyle: { color: text, fontSize: 13, fontWeight: 600 },
    },
  };
}

export function TelemetryTimeline({ runId }: { runId: string }) {
  const [data, setData] = useState<RunTelemetry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<GpuMetric>(DEFAULT_METRIC);
  const [showTable, setShowTable] = useState(false);
  const theme = useTheme();

  useEffect(() => {
    let cancelled = false;
    runsApi
      .telemetry(runId)
      .then((t) => !cancelled && setData(t))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const options = useMemo(() => {
    if (!data || data.sample_count === 0) return null;

    const colors = seriesColors();
    const text = cssVar("--text", "#16181d");
    const muted = cssVar("--text-muted", "#6b7280");
    const grid = cssVar("--grid-line", "#e8eaee");
    const surface = cssVar("--surface", "#ffffff");

    const stamps = [...data.engine, ...data.gpu].map((s) => new Date(s.sampled_at).getTime());
    const origin = Math.min(...stamps);

    // Direct labels on the last point only. A number on every point is noise; one at the
    // end of the line means identity never depends on color alone.
    const endLabel = (formatter: (v: number) => string) => ({
      show: true,
      position: "right" as const,
      formatter: (p: { data: [number, number] }) => formatter(p.data[1]),
      color: text,
      fontSize: 11,
      backgroundColor: surface,
      padding: [2, 4] as [number, number],
      borderRadius: 3,
    });

    const line = (
      name: string,
      color: string,
      points: [number, number][],
      label: object,
      /** Secondary encoding, keyed to the device — see DASH. */
      dash: "solid" | "dashed" | "dotted" = "solid",
    ) => ({
      name,
      type: "line" as const,
      showSymbol: false,
      symbolSize: 8,
      lineStyle: { width: 2, color, type: dash },
      itemStyle: { color },
      data: points,
      endLabel: label,
      // Direct labels are selective by definition: where two series converge their end
      // labels overlap and become unreadable, so the collided one is dropped. Identity
      // survives because the legend is always present.
      labelLayout: { hideOverlap: true },
      emphasis: { focus: "series" as const },
    });

    const queue: echarts.EChartsOption = {
      ...baseOption("Request queue", text, muted, grid),
      legend: {
        top: 2,
        right: 8,
        textStyle: { color: muted, fontSize: 11 },
        icon: "roundRect",
        itemWidth: 12,
        itemHeight: 8,
      },
      series: [
        line(
          "Running",
          colors[0],
          data.engine.map((s) => [relativeSeconds(s.sampled_at, origin), s.num_requests_running ?? 0]),
          endLabel((v) => `${Math.round(v)}`),
        ),
        line(
          "Waiting",
          colors[1],
          data.engine.map((s) => [relativeSeconds(s.sampled_at, origin), s.num_requests_waiting ?? 0]),
          endLabel((v) => `${Math.round(v)}`),
        ),
      ],
    };

    // One series, so no legend box — the title already names it.
    const kv: echarts.EChartsOption = {
      ...baseOption("KV cache utilization", text, muted, grid),
      yAxis: {
        ...(baseOption("", text, muted, grid).yAxis as object),
        max: 100,
        axisLabel: { color: muted, fontSize: 11, formatter: "{value}%" },
      },
      series: [
        line(
          "KV cache",
          colors[0],
          // Stored as a 0..1 fraction because that is what vLLM emits; scaled to percent
          // here, at the only place that shows it to a person.
          data.engine.map((s) => [
            relativeSeconds(s.sampled_at, origin),
            (s.kv_cache_usage_fraction ?? 0) * 100,
          ]),
          endLabel((v) => `${v.toFixed(1)}%`),
        ),
      ],
    };

    const perDevice = data.gpu_indices.map((index, position) =>
      line(
        `GPU ${index}`,
        colors[position % colors.length] ?? colors[0],
        data.gpu
          .filter((s) => s.gpu_index === index)
          .map((s) => {
            const raw = s[metric.key] ?? 0;
            return [relativeSeconds(s.sampled_at, origin), metric.scale ? metric.scale(raw) : raw] as [
              number,
              number,
            ];
          }),
        endLabel((v) => `${v.toFixed(metric.scale ? 1 : 0)}${metric.unit}`),
        DASH[position % DASH.length],
      ),
    );

    const gpu: echarts.EChartsOption = {
      ...baseOption(`${metric.label} per GPU`, text, muted, grid, true),
      // A legend is required for two or more series and pointless for one — the title
      // already names it. Spread so the key is absent rather than explicitly undefined.
      ...(data.gpu_indices.length > 1
        ? {
            // No icon override: the legend swatch draws the actual line, dash and all,
            // which is what makes the secondary encoding legible.
            legend: {
              top: 2,
              right: 8,
              textStyle: { color: muted, fontSize: 11 },
              itemWidth: 18,
              itemHeight: 8,
            },
          }
        : {}),
      yAxis: {
        ...(baseOption("", text, muted, grid).yAxis as object),
        // Spread rather than assigned: a metric with no natural ceiling must leave the
        // axis to autoscale, and an explicit `undefined` is not the same as absent.
        ...(metric.max === undefined ? {} : { max: metric.max }),
        axisLabel: {
          color: muted,
          fontSize: 11,
          formatter: (v: number) => `${v}${metric.unit}`,
        },
      },
      series: perDevice,
    };

    return { queue, kv, gpu };
  }, [data, metric, theme]);

  if (error) return <p className="error">Telemetry unavailable: {error}</p>;
  if (!data) return <p className="muted">Loading telemetry…</p>;

  if (data.sample_count === 0) {
    return (
      <p className="muted">
        No telemetry for this run. Runs recorded before 0.3.0 have none, and an agent
        older than protocol 4 does not sample.
      </p>
    );
  }

  return (
    <>
      <div className="chart-controls">
        <label>
          <span>Per-GPU metric</span>
          <select
            value={metric.key}
            onChange={(e) =>
              setMetric(GPU_METRICS.find((m) => m.key === e.target.value) ?? DEFAULT_METRIC)
            }
          >
            {GPU_METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <span className="spacer" />
        <button onClick={() => setShowTable((v) => !v)}>
          {showTable ? "Hide values" : "Show values"}
        </button>
      </div>

      {options && (
        <>
          <Chart option={options.queue} height={200} />
          <Chart option={options.kv} height={180} />
          <Chart option={options.gpu} height={200} />
        </>
      )}

      {/* The table is the accessibility floor: every number the charts draw, readable
          without relying on color or on reading a line's position. */}
      {showTable && <TelemetryTable data={data} metric={metric} />}
    </>
  );
}

function TelemetryTable({ data, metric }: { data: RunTelemetry; metric: GpuMetric }) {
  const origin = Math.min(...data.engine.map((s) => new Date(s.sampled_at).getTime()));
  return (
    <div className="devices" style={{ maxHeight: 320, overflowY: "auto" }}>
      <table>
        <caption className="muted">
          Engine samples, and {metric.label.toLowerCase()} per GPU
        </caption>
        <thead>
          <tr>
            <th>t (s)</th>
            <th>Running</th>
            <th>Waiting</th>
            <th>KV cache</th>
            {data.gpu_indices.map((i) => (
              <th key={i}>GPU {i}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.engine.map((s) => {
            const t = relativeSeconds(s.sampled_at, origin);
            return (
              <tr key={s.sampled_at}>
                <td>{t.toFixed(0)}</td>
                <td>{s.num_requests_running ?? "—"}</td>
                <td>{s.num_requests_waiting ?? "—"}</td>
                <td>
                  {s.kv_cache_usage_fraction === null || s.kv_cache_usage_fraction === undefined
                    ? "—"
                    : `${(s.kv_cache_usage_fraction * 100).toFixed(1)}%`}
                </td>
                {data.gpu_indices.map((index) => {
                  // Nearest GPU sample to this engine sample. The two series are stored
                  // separately because a scrape can fail while NVML answers, so they are
                  // not guaranteed to share timestamps.
                  const match = data.gpu.find(
                    (g) => g.gpu_index === index && g.sampled_at === s.sampled_at,
                  );
                  const raw = match?.[metric.key];
                  if (raw === null || raw === undefined) return <td key={index}>—</td>;
                  const shown = metric.scale ? metric.scale(raw) : raw;
                  return (
                    <td key={index}>
                      {shown.toFixed(metric.scale ? 1 : 0)}
                      {metric.unit}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
