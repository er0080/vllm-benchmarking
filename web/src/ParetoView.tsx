/**
 * The Pareto frontier: per-user output tok/s against per-GPU total tok/s.
 *
 * This is the view the project exists to produce. Serving has one fundamental trade-off
 * — batch harder and the machine produces more tokens per second while each individual
 * user waits longer between them — and every knob in a vLLM config is a position on that
 * curve. A table of thirty configurations does not tell you which ones are worth
 * considering; the frontier does, by removing the ones that are worse on *both* axes at
 * once and therefore never the right answer whatever you care about.
 *
 * Four decisions carry most of the weight here:
 *
 * **Both axes are normalized or per-request.** A TP=4 run trivially out-throughputs TP=1
 * on aggregate tokens per second while potentially being far worse per device, so the x
 * axis is per-GPU (invariant 8). Plotting aggregate throughput would make "add GPUs" look
 * like a free win on every chart.
 *
 * **Uncertainty is drawn, not averaged away.** Each point is a median with a cross
 * showing min-to-max across its replicates, in both dimensions, because both axes are
 * measured quantities. A point measured once gets a hollow marker instead of a small
 * cross — "no spread was measured" and "the spread was narrow" are different claims and
 * must not look the same.
 *
 * **Incomparable runs are never overlaid.** The API partitions by host, GPU, vLLM version
 * and bench-client location; this view charts one partition at a time and names the
 * others rather than merging them. Two vLLM versions on one scatter is not a comparison,
 * it is a mistake that looks like one.
 *
 * **Colour carries topology, labels carry identity.** Tensor-parallel size is the
 * categorical encoding because it is bounded and it is the question the chart is usually
 * being asked. Config names are direct-labelled on frontier points only — a name on every
 * point would be unreadable, and the frontier is the part a reader acts on.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type * as echarts from "echarts";

import { analysisApi } from "./api";
import { Chart, SYMBOLS, cssVar, seriesColors, useTheme } from "./chartkit";
import type {
  Analysis,
  AnalysisGroup,
  AnalysisPoint,
  Metric,
  RunSource,
  Spread,
} from "./types";

/**
 * Palette slot per tensor-parallel size, fixed rather than assigned in order of
 * appearance.
 *
 * Colour follows the entity, never its rank. If slots were handed out by position in
 * whatever sizes happen to be on screen, filtering TP=1 away would repaint TP=2 blue —
 * silently changing the meaning a reader has already learned. These four cover every
 * topology a single host can run, since TP must divide the device count.
 */
const TP_SLOT: Record<number, number> = { 1: 0, 2: 1, 4: 2, 8: 3 };
const MAX_TP_SERIES = 4;

function format(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(2);
}

function metricOf(metrics: Metric[], key: string): Metric | undefined {
  return metrics.find((m) => m.key === key);
}

/** The distinct tensor-parallel sizes on a chart, in ascending order. */
function tpSeries(points: AnalysisPoint[]): number[] {
  return [...new Set(points.map((p) => p.tensor_parallel_size))].sort((a, b) => a - b);
}

type Plotted = {
  point: AnalysisPoint;
  x: Spread;
  y: Spread;
};

function plottable(points: AnalysisPoint[], xKey: string, yKey: string): Plotted[] {
  const out: Plotted[] = [];
  for (const point of points) {
    const x = point.metrics[xKey];
    const y = point.metrics[yKey];
    // A point missing either axis is dropped rather than pinned to zero, which would
    // put it on the frontier by accident.
    if (x && y) out.push({ point, x, y });
  }
  return out;
}

export function paretoOption(
  group: AnalysisGroup,
  xMetric: Metric,
  yMetric: Metric,
  colors: string[],
  ink: { text: string; muted: string; grid: string; surface: string },
): echarts.EChartsOption | null {
  const plotted = plottable(group.points, xMetric.key, yMetric.key);
  if (plotted.length === 0) return null;

  // One series per tensor-parallel size, in its fixed palette slot. A size outside
  // {1,2,4,8} cannot be given a validated colour of its own, so it joins a single
  // honestly-named "other" series rather than quietly borrowing another size's colour —
  // which would put two topologies behind one legend entry.
  const tps = tpSeries(plotted.map((p) => p.point));
  const slotOf = (tp: number) => TP_SLOT[tp] ?? MAX_TP_SERIES;
  const slots = [...new Set(tps.map(slotOf))].sort((a, b) => a - b);
  const nameOfSlot = (slot: number) =>
    slot === MAX_TP_SERIES
      ? `TP ${tps.filter((tp) => slotOf(tp) === slot).join(", ")}`
      : `TP ${tps.find((tp) => slotOf(tp) === slot)}`;
  const seriesNames = slots.map(nameOfSlot);

  // A point is a config *running a workload*, so the config name alone does not name it:
  // one config swept across three concurrencies produced three points all labelled the
  // same thing, which was worse than no label. The workload is appended only when the
  // group actually holds more than one, so the common case stays short.
  const manyWorkloads = new Set(plotted.map(({ point }) => point.workload_hash)).size > 1;
  const labelFor = ({ point }: Plotted) =>
    manyWorkloads ? `${point.config_name} · ${point.workload_name}` : point.config_name;

  // The frontier staircase, drawn first so markers sit on top of it.
  const frontier = plotted
    .filter(({ point }) => point.on_pareto_frontier)
    .sort((a, b) => a.x.median - b.x.median);

  const series: echarts.SeriesOption[] = [
    {
      type: "line",
      name: "Frontier",
      data: frontier.map(({ x, y }) => [x.median, y.median]),
      // A staircase, not a smooth curve: nothing was measured between two configs, and
      // an interpolating line would invent operating points that do not exist.
      step: "end",
      showSymbol: false,
      lineStyle: { width: 2, color: ink.muted, type: "dashed" },
      silent: true,
      z: 1,
      tooltip: { show: false },
    },
  ];

  // Error crosses, one series so they share a legend entry with nothing — they are not
  // an identity, they are the uncertainty of every mark.
  const bars: [number, number][][] = [];
  for (const { x, y } of plotted) {
    if (x.n > 1) bars.push([[x.min, y.median], [x.max, y.median]]);
    if (y.n > 1) bars.push([[x.median, y.min], [x.median, y.max]]);
  }
  for (const [from, to] of bars.map((b) => [b[0]!, b[1]!])) {
    series.push({
      type: "line",
      data: [from, to],
      showSymbol: false,
      lineStyle: { width: 1, color: ink.muted, opacity: 0.55 },
      silent: true,
      z: 2,
      tooltip: { show: false },
    });
  }

  slots.forEach((slot, index) => {
    const name = seriesNames[index] ?? `TP ${slot}`;
    const mine = plotted.filter(({ point }) => slotOf(point.tensor_parallel_size) === slot);
    const color = colors[Math.min(slot, colors.length - 1)] ?? colors[0] ?? "#2a78d6";
    const symbol = SYMBOLS[Math.min(slot, SYMBOLS.length - 1)] ?? "circle";
    series.push({
      type: "scatter",
      name,
      symbol,
      symbolSize: 13,
      z: 3,
      itemStyle: {
        color,
        opacity: 1,
        // A ring in the surface colour, so overlapping marks stay countable.
        borderColor: ink.surface,
        borderWidth: 2,
      },
      data: mine.map(({ point, x, y }) => ({
        value: [x.median, y.median],
        name: point.config_name,
        // Hollow for a single run: no spread was measured, which is not the same as a
        // spread that came out small, and the two must not render identically. The ring
        // is thick and fully opaque because at 1px against a dark surface the mark all
        // but disappears — verified by rendering both themes, not by reasoning about it.
        // Spread conditionally rather than set to undefined — exactOptionalPropertyTypes.
        ...(point.spread_basis === "single"
          ? {
              itemStyle: {
                color: ink.surface,
                borderColor: color,
                borderWidth: 2.5,
                opacity: 1,
              },
            }
          : {}),
        label: {
          show: point.on_pareto_frontier,
          position: "top",
          distance: 8,
          formatter: labelFor({ point, x, y }),
          // Text wears text ink, never the series colour; the mark beside it carries
          // the identity.
          color: ink.text,
          fontSize: 11,
          // A surface-coloured plate behind the text. hideOverlap keeps labels off each
          // other but not off other series' markers, and a frontier label landing on a
          // neighbouring point rendered as "seqs-82-tp1" — a real misreading, not a
          // cosmetic one.
          backgroundColor: ink.surface,
          padding: [2, 3],
          borderRadius: 3,
        },
        point,
        xSpread: x,
        ySpread: y,
      })),
      labelLayout: { hideOverlap: true },
    });
  });

  return {
    grid: { left: 70, right: 28, top: 44, bottom: 56 },
    textStyle: { fontFamily: "system-ui, -apple-system, sans-serif" },
    legend: {
      top: 0,
      right: 0,
      itemWidth: 14,
      textStyle: { color: ink.muted, fontSize: 11 },
      // The frontier line is explained by the caption, not by a legend swatch that
      // competes with the topology series for attention.
      data: seriesNames,
    },
    tooltip: {
      trigger: "item",
      confine: true,
      formatter: (params: unknown) => {
        const data = (params as { data?: Record<string, unknown> }).data;
        const point = data?.["point"] as AnalysisPoint | undefined;
        const x = data?.["xSpread"] as Spread | undefined;
        const y = data?.["ySpread"] as Spread | undefined;
        if (!point || !x || !y) return "";
        const band = (s: Spread, unit: string) =>
          s.n > 1
            ? `${format(s.median)} ${unit} <span style="opacity:.7">(${format(s.min)}–${format(s.max)}, n=${s.n})</span>`
            : `${format(s.median)} ${unit} <span style="opacity:.7">(1 run)</span>`;
        return [
          `<strong>${point.config_name}</strong>`,
          `<div style="opacity:.75">${point.workload_name} · TP ${point.tensor_parallel_size} · ${point.gpu_count} GPU</div>`,
          `<div style="margin-top:6px">${xMetric.label}: ${band(x, xMetric.unit)}</div>`,
          `<div>${yMetric.label}: ${band(y, yMetric.unit)}</div>`,
          `<div style="margin-top:6px;max-width:22rem;opacity:.75">${point.spread_note}</div>`,
          point.on_pareto_frontier
            ? `<div style="margin-top:6px">On the frontier.</div>`
            : "",
        ].join("");
      },
    },
    xAxis: {
      type: "value",
      name: `${xMetric.label} (${xMetric.unit})`,
      nameLocation: "middle",
      nameGap: 32,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      axisLine: { lineStyle: { color: ink.grid } },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: ink.grid } },
      scale: true,
    },
    yAxis: {
      type: "value",
      name: `${yMetric.label} (${yMetric.unit})`,
      nameLocation: "middle",
      nameGap: 52,
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: ink.grid } },
      scale: true,
    },
    series,
  };
}

function PointsTable({
  group,
  xMetric,
  yMetric,
}: {
  group: AnalysisGroup;
  xMetric: Metric;
  yMetric: Metric;
}) {
  return (
    <div className="devices">
      <table>
        <thead>
          <tr>
            <th>Config</th>
            <th>Workload</th>
            <th>TP</th>
            <th>{xMetric.label}</th>
            <th>{yMetric.label}</th>
            <th>Replicates</th>
            <th>Frontier</th>
          </tr>
        </thead>
        <tbody>
          {group.points.map((point) => {
            const x = point.metrics[xMetric.key];
            const y = point.metrics[yMetric.key];
            const cell = (s: Spread | undefined) =>
              !s
                ? "—"
                : s.n > 1
                  ? `${format(s.median)} (${format(s.min)}–${format(s.max)})`
                  : format(s.median);
            return (
              <tr key={point.point_id}>
                <td>{point.config_name}</td>
                <td>{point.workload_name}</td>
                <td>{point.tensor_parallel_size}</td>
                <td>{cell(x)}</td>
                <td>{cell(y)}</td>
                <td title={point.spread_note}>
                  {point.replicates} · {point.spread_basis}
                </td>
                <td>{point.on_pareto_frontier ? "yes" : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ExcludedNote({ data }: { data: Analysis }) {
  const { excluded } = data;
  const parts: string[] = [];
  if (excluded.failed) parts.push(`${excluded.failed} failed`);
  if (excluded.cancelled) parts.push(`${excluded.cancelled} cancelled`);
  if (excluded.unfinished) parts.push(`${excluded.unfinished} still running or queued`);
  if (excluded.succeeded_without_summary) {
    parts.push(`${excluded.succeeded_without_summary} succeeded but produced no metrics`);
  }
  if (parts.length === 0 && !data.truncated) return null;

  return (
    <p className="muted" style={{ marginTop: "0.4rem" }}>
      {parts.length > 0 && (
        <>
          Not shown: {parts.join(", ")}. A missing point and a point whose every replicate
          failed look the same on a chart, so the counts are stated here.{" "}
        </>
      )}
      {data.truncated && (
        <>
          Showing the {data.limit} most recent runs only — narrow the filters to see the
          rest.
        </>
      )}
    </p>
  );
}

export function ParetoView() {
  const [data, setData] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [source, setSource] = useState<RunSource>("real");
  const [groupId, setGroupId] = useState<string | null>(null);
  const [xKey, setXKey] = useState<string | null>(null);
  const [yKey, setYKey] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);
  const theme = useTheme();

  const load = useCallback(async () => {
    try {
      setData(await analysisApi.points({ source }));
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

  const group = useMemo(() => {
    if (!data || data.groups.length === 0) return null;
    return data.groups.find((g) => g.group_id === groupId) ?? data.groups[0] ?? null;
  }, [data, groupId]);

  const xMetric = data ? metricOf(data.metrics, xKey ?? data.pareto_x) : undefined;
  const yMetric = data ? metricOf(data.metrics, yKey ?? data.pareto_y) : undefined;

  const option = useMemo(() => {
    if (!group || !xMetric || !yMetric) return null;
    const colors = seriesColors();
    const ink = {
      text: cssVar("--text", "#16181d"),
      muted: cssVar("--text-muted", "#6b7280"),
      grid: cssVar("--grid-line", "#e8eaee"),
      surface: cssVar("--surface", "#ffffff"),
    };
    return paretoOption(group, xMetric, yMetric, colors, ink);
    // theme is a dependency because ECharts bakes colours in at option time.
  }, [group, xMetric, yMetric, theme]);

  // A size outside {1,2,4,8} has no validated colour of its own; say so rather than
  // letting the reader assume the palette is telling them something it is not.
  const unslottedTp = group
    ? tpSeries(group.points).filter((tp) => TP_SLOT[tp] === undefined)
    : [];

  return (
    <section>
      <h2>Pareto frontier</h2>

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
              <select
                value={group?.group_id ?? ""}
                onChange={(e) => setGroupId(e.target.value)}
              >
                {data.groups.map((g) => (
                  <option key={g.group_id} value={g.group_id}>
                    {g.label} ({g.run_count} runs)
                  </option>
                ))}
              </select>
            </label>
          )}

          {data && (
            <>
              <label>
                X axis
                <select
                  value={xMetric?.key ?? ""}
                  onChange={(e) => setXKey(e.target.value)}
                >
                  {data.metrics.map((m) => (
                    <option key={m.key} value={m.key}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Y axis
                <select
                  value={yMetric?.key ?? ""}
                  onChange={(e) => setYKey(e.target.value)}
                >
                  {data.metrics.map((m) => (
                    <option key={m.key} value={m.key}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </label>
            </>
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
            measurements of any real hardware. They are quarantined from real results and
            can never appear on the same chart as one.
          </p>
        )}

        {error && <p className="error">{error}</p>}
        {!loaded && <p className="muted">Loading…</p>}

        {loaded && !error && data && data.groups.length === 0 && (
          <p className="muted">
            Nothing to chart yet — no succeeded runs with metrics match these filters.
            {data.excluded.other_source > 0 && (
              <>
                {" "}
                There {data.excluded.other_source === 1 ? "is" : "are"}{" "}
                {data.excluded.other_source} {data.excluded.other_source_name} run
                {data.excluded.other_source === 1 ? "" : "s"}, shown under the other Runs
                setting above.
              </>
            )}
          </p>
        )}

        {group && (
          <>
            <p className="muted" style={{ marginTop: 0 }}>
              {group.label} — {group.run_count} run{group.run_count === 1 ? "" : "s"}.
              {data && data.groups.length > 1 && (
                <>
                  {" "}
                  {data.groups.length - 1} other comparison set
                  {data.groups.length === 2 ? "" : "s"} exist
                  {data.groups.length === 2 ? "s" : ""} and cannot be charted alongside
                  this one — they differ in GPU, host, vLLM version, or where the
                  benchmark client ran.
                </>
              )}
            </p>

            {group.warnings.map((w) => (
              <p className="notice" key={w}>
                {w}. Kept on one chart because the effect is small next to the differences
                being measured — but it is not nothing.
              </p>
            ))}

            {unslottedTp.length > 0 && (
              <p className="notice">
                Tensor-parallel {unslottedTp.join(", ")} share one series: the palette has
                {" "}{MAX_TP_SERIES} colour-blind-validated slots, held by TP 1, 2, 4 and 8.
                Read the table to tell them apart.
              </p>
            )}

            {option ? (
              <Chart option={option} height={420} />
            ) : (
              <p className="muted">
                No point in this set has both {xMetric?.label} and {yMetric?.label}.
              </p>
            )}

            <p className="muted">
              Filled marks are replicated; hollow marks were measured once, so no spread
              is known. Crosses span min to max across replicates. The dashed staircase is
              the frontier: everything below and left of it is beaten on both axes at once
              and is never the right choice.
            </p>

            {data && <ExcludedNote data={data} />}

            {showTable && xMetric && yMetric && (
              <PointsTable group={group} xMetric={xMetric} yMetric={yMetric} />
            )}
          </>
        )}
      </div>
    </section>
  );
}
