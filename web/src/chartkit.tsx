/**
 * The pieces every chart in this app shares: theme awareness, the categorical palette,
 * and an ECharts container that cleans up after itself.
 *
 * Extracted when the second chart arrived. The palette in particular has to be shared:
 * it is validated as a set — lightness band, chroma floor, and colorblind separation
 * between adjacent slots — so a view that picked its own colors would be picking
 * unvalidated ones, and the check that makes the palette trustworthy is exactly the one
 * nobody re-runs.
 */
import { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";

export function cssVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * Series colors, assigned by entity in fixed order — never cycled, never derived from
 * position in a filtered list. Deriving them from position would repaint every surviving
 * series whenever one was hidden, silently changing what a reader has already learned to
 * associate with a color.
 *
 * Four slots is the whole palette. A fifth series is not a generated hue; it folds into
 * an "other" bucket or the view splits into small multiples.
 */
export function seriesColors(): [string, string, string, string] {
  return [
    cssVar("--series-1", "#2a78d6"),
    cssVar("--series-2", "#eb6834"),
    cssVar("--series-3", "#1baf7a"),
    cssVar("--series-4", "#eda100"),
  ];
}

/**
 * Line style per entity, on top of color.
 *
 * Not decoration. Two series frequently sit at *exactly* the same value — two GPUs in a
 * balanced tensor-parallel run, two configs that reach the same operating point — and
 * two solid lines at one value render as one, so the later series silently erases the
 * earlier. It doubles as the secondary encoding that keeps series distinguishable
 * without color, for colorblind readers and for print.
 */
export const DASH = ["solid", "dashed", "dotted"] as const;

/** Marker shapes, the scatter equivalent of DASH. */
export const SYMBOLS = ["circle", "triangle", "diamond", "roundRect"] as const;

export function useTheme(): string {
  // Re-render the charts when the theme changes; ECharts bakes colors in at option time.
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? "system");
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setTheme(document.documentElement.dataset.theme ?? "system"),
    );
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onMedia = () => setTheme((t) => t + " ");
    media.addEventListener("change", onMedia);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", onMedia);
    };
  }, []);
  return theme;
}

export function Chart({
  option,
  height,
  onClick,
}: {
  option: echarts.EChartsOption;
  height: number;
  onClick?: (params: echarts.ECElementEvent) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const handler = useRef(onClick);
  handler.current = onClick;

  useEffect(() => {
    if (!ref.current) return;
    const instance = echarts.init(ref.current);
    chart.current = instance;
    instance.on("click", (params) => handler.current?.(params as echarts.ECElementEvent));
    const resize = () => instance.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      instance.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    // notMerge: switching metric or group changes the series set, and merging would
    // leave the previous set on the canvas underneath the new one.
    chart.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}

/**
 * Whether an axis of these values should be logarithmic.
 *
 * Benchmark axes routinely span an order of magnitude — concurrency doubles from 4 to
 * 64, aggregate throughput differs 13x between a light and a heavy workload — and on a
 * linear axis the small end collapses into the margin and stops being readable at all.
 *
 * Shared so every chart makes the same call, and always paired with saying so on the
 * axis label. A silently switching scale is the trap; a chart that is unreadable for
 * half its inputs is the alternative.
 */
export const LOG_SCALE_THRESHOLD = 10;

export function needsLogScale(values: readonly number[]): boolean {
  const positive = values.filter((v) => Number.isFinite(v) && v > 0);
  if (positive.length < 2) return false;
  return Math.max(...positive) / Math.min(...positive) > LOG_SCALE_THRESHOLD;
}
