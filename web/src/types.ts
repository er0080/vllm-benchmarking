export interface GpuDevice {
  device_index: number;
  name: string;
  vram_bytes: number | null;
}

export interface Host {
  id: string;
  name: string;
  agent_url: string;
  agent_version: string | null;
  protocol_version: number | null;
  vllm_version: string | null;
  driver_version: string | null;
  cuda_version: string | null;
  gpu_count: number;
  /**
   * Set by the agent itself when it is a mock or CPU-backend stand-in. The UI must
   * surface it prominently: a synthetic host's numbers are not measurements, and
   * nothing downstream should be read as if they were.
   */
  synthetic_source: string | null;
  /**
   * Whether this host's Python environment satisfies the constraints everything
   * installed there declares. The agent lives in vLLM's own virtualenv, so the two can
   * diverge with nothing arbitrating between them.
   *
   * `"not_reported"` is a real answer and not a missing one — an agent older than
   * protocol 6 could not check. Rendering it as "fine" is the mistake this field exists
   * to prevent.
   */
  environment_status: "ok" | "conflicts" | "unavailable" | "not_reported";
  environment_conflicts: string[];
  last_seen_at: string | null;
  created_at: string;
  devices: GpuDevice[];
  reference_vllm_version: string | null;
  /**
   * Null when either version is unknown. A mismatch is information, never an error —
   * comparing vLLM versions is a supported use of this tool.
   */
  vllm_version_matches_reference: boolean | null;
}

export interface ApiError {
  detail: string;
}

export interface Config {
  id: string;
  config_hash: string;
  name: string;
  yaml: string;
  notes: string | null;
  /** The config this one was edited from, when it was. */
  parent_id: string | null;
  /** The run somebody would point at to defend this configuration. */
  justified_by_run_id: string | null;
  justification_note: string | null;
  created_at: string;
}

/** One problem with a candidate config, and where. */
export interface Finding {
  /** "error" — vLLM will refuse to start. "warning" — it will start and may not do what
   *  you meant. Not the same thing, and never rendered as though they were. */
  severity: "error" | "warning";
  message: string;
  key: string | null;
  line: number | null;
  /** Offered, never applied: the engine validates, it does not rewrite (invariant 5). */
  suggestion: string | null;
}

export interface ConfigValidation {
  valid: boolean;
  findings: Finding[];
  /** The vLLM version whose arguments were used. */
  checked_against: string;
  /** False when the target host runs a version with no captured catalogue, so the
   *  reference was used instead. A clean result means something weaker then. */
  exact_version_match: boolean;
}

export interface LineageNode {
  id: string;
  config_hash: string;
  name: string;
  created_at: string;
}

export interface Lineage {
  config_hash: string;
  /** Nearest parent first, back to the original. */
  ancestors: LineageNode[];
  children: LineageNode[];
  truncated: boolean;
}

export interface Workload {
  id: string;
  workload_hash: string;
  name: string;
  dataset_name: string;
  num_prompts: number;
  /** null means unbounded — `--request-rate inf` / no `--max-concurrency`. */
  request_rate: number | null;
  max_concurrency: number | null;
  input_len: number | null;
  output_len: number | null;
  created_at: string;
}

export interface RunSummary {
  successful_requests: number | null;
  failed_requests: number | null;
  benchmark_duration_sec: number | null;
  total_input_tokens: number | null;
  total_generated_tokens: number | null;
  request_throughput_req_sec: number | null;
  output_token_throughput_tok_sec: number | null;
  total_token_throughput_tok_sec: number | null;
  /**
   * The comparable figures. Aggregate throughput is not comparable across parallelism
   * topologies, so anything that ranks configurations must use these.
   */
  output_token_throughput_per_gpu: number | null;
  total_token_throughput_per_gpu: number | null;
  peak_output_token_throughput_tok_sec: number | null;
  peak_concurrent_requests: number | null;
  ttft_ms_mean: number | null;
  ttft_ms_median: number | null;
  ttft_ms_p99: number | null;
  ttft_ms_std: number | null;
  tpot_ms_mean: number | null;
  tpot_ms_median: number | null;
  tpot_ms_p99: number | null;
  tpot_ms_std: number | null;
  itl_ms_mean: number | null;
  itl_ms_median: number | null;
  itl_ms_p99: number | null;
  itl_ms_std: number | null;
}

export type RunStatus =
  | "queued"
  | "starting"
  | "benchmarking"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface Run {
  id: string;
  status: RunStatus;
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
  server_config_id: string;
  workload_id: string;
  gpu_host_id: string;
  config_hash: string;
  workload_hash: string;
  vllm_version: string | null;
  agent_version: string | null;
  gpu_model: string | null;
  driver_version: string | null;
  cuda_version: string | null;
  gpu_count: number;
  tensor_parallel_size: number;
  pipeline_parallel_size: number;
  device_indices: number[] | null;
  bench_client_location: string;
  is_synthetic: boolean;
  synthetic_source: string | null;
  initiated_by: string;
  error: string | null;
  /**
   * Which class of failure this was, for a run that has one. Never a substitute for
   * `error`, which holds the full text — this is what makes a sweep's failures
   * countable rather than eleven walls of traceback to read one at a time.
   */
  failure_kind: string | null;
  log_excerpt: string | null;
  summary: RunSummary | null;
}

export interface EngineSample {
  sampled_at: string;
  num_requests_running: number | null;
  num_requests_waiting: number | null;
  /** 0..1, exactly as vLLM emits it. Multiply for display; never store the percent. */
  kv_cache_usage_fraction: number | null;
  num_preemptions_total: number | null;
  prefix_cache_queries_total: number | null;
  prefix_cache_hits_total: number | null;
}

export interface GpuSample {
  sampled_at: string;
  gpu_index: number;
  sm_utilization_pct: number | null;
  memory_used_bytes: number | null;
  power_watts: number | null;
  temperature_c: number | null;
  sm_clock_mhz: number | null;
  memory_clock_mhz: number | null;
}

export interface RunTelemetry {
  run_id: string;
  engine: EngineSample[];
  gpu: GpuSample[];
  /** Devices that actually produced samples, in ascending order. */
  gpu_indices: number[];
  /**
   * When this run's telemetry was deleted by the retention policy, if it was. Without
   * it an empty timeline has two readings — the policy removed it, or sampling failed —
   * and those call for opposite responses.
   */
  pruned_at: string | null;
  pruned_horizon_days: number | null;
  sample_count: number;
}

export type SweepStatus = "draft" | "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface SweepProgress {
  total: number;
  queued: number;
  starting: number;
  benchmarking: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  /**
   * Failed runs by kind, highest count first. Empty when nothing failed, and for
   * sweeps whose failures predate the column.
   */
  failures: Record<string, number>;
}

export interface Sweep {
  id: string;
  name: string;
  description: string | null;
  status: SweepStatus;
  gpu_host_id: string;
  replicates: number;
  replicate_order: "grouped" | "interleaved";
  initiated_by: string;
  is_synthetic: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  progress: SweepProgress;
  /** Model loads implied by the plan — most of a sweep's wall clock. */
  engine_starts: number;
  /** Present only while a sweep still has work. Null on anything terminal. */
  estimated_remaining: DurationEstimate | null;
}

export interface DurationEstimate {
  runs_remaining: number;
  engine_loads_remaining: number;
  /** Null means not known. Never a fabricated zero — nothing has finished yet. */
  seconds_remaining: number | null;
  median_run_seconds: number | null;
  median_engine_load_seconds: number | null;
  basis: string;
  sample_size: number;
  caveats: string[];
}

export interface SweepCreate {
  name: string;
  description?: string | null;
  gpu_host_id: string;
  server_config_ids: string[];
  workload_ids: string[];
  tensor_parallel_sizes?: number[] | null;
  replicates: number;
  replicate_order: "grouped" | "interleaved";
}

/**
 * A chartable measurement and how to read it.
 *
 * Shipped by the API rather than hard-coded here so axis direction and per-GPU status
 * have one definition. A view that decided on its own which metrics are "lower is
 * better" would eventually draw a frontier upside down.
 */
export interface Metric {
  key: string;
  label: string;
  unit: string;
  better: "higher" | "lower";
  per_gpu: boolean;
  description: string;
}

/** One metric across a point's replicates. `median` is plotted; the rest is the band. */
export interface Spread {
  n: number;
  median: number;
  mean: number;
  min: number;
  max: number;
  values: number[];
  relative_range: number | null;
}

/**
 * What a point's error band actually measures. `single` means one run and no band at
 * all — which must look different from a band that happens to be narrow.
 */
export type SpreadBasis = "single" | "grouped" | "interleaved" | "mixed";

export interface AnalysisPoint {
  point_id: string;
  config_hash: string;
  config_name: string;
  workload_hash: string;
  workload_name: string;
  tensor_parallel_size: number;
  pipeline_parallel_size: number;
  gpu_count: number;
  max_concurrency: number | null;
  request_rate: number | null;
  num_prompts: number;
  replicates: number;
  run_ids: string[];
  sweep_ids: string[];
  spread_basis: SpreadBasis;
  spread_note: string;
  latest_finished_at: string | null;
  on_pareto_frontier: boolean;
  metrics: Record<string, Spread>;
}

/**
 * Points that may legitimately share a chart. The API partitions, not the client — a
 * view cannot overlay two vLLM versions by forgetting to check, because it is never
 * handed them in one series.
 */
export interface AnalysisGroup {
  group_id: string;
  label: string;
  gpu_host_id: string;
  gpu_host_name: string;
  gpu_model: string | null;
  vllm_version: string | null;
  bench_client_location: string;
  warnings: string[];
  run_count: number;
  points: AnalysisPoint[];
  pareto_point_ids: string[];
}

/** Runs the filters matched that no chart will show. */
export interface AnalysisExcluded {
  failed: number;
  cancelled: number;
  unfinished: number;
  succeeded_without_summary: number;
  other_source: number;
  other_source_name: string;
}

export type RunSource = "real" | "synthetic";

export interface Analysis {
  source: RunSource;
  run_count: number;
  truncated: boolean;
  limit: number;
  pareto_x: string;
  pareto_y: string;
  metrics: Metric[];
  excluded: AnalysisExcluded;
  groups: AnalysisGroup[];
}

export interface AnalysisQuery {
  source?: RunSource;
  hostId?: string;
  sweepIds?: string[];
  tensorParallelSizes?: number[];
  paretoX?: string;
  paretoY?: string;
}

export interface ScalingStep {
  tensor_parallel_size: number;
  gpu_count: number;
  point_id: string;
  config_name: string;
  is_baseline: boolean;
  /** Aggregate throughput relative to the baseline width — what an operator feels. */
  speedup: number | null;
  /** Per-GPU throughput relative to the baseline — parallel efficiency. */
  efficiency: number | null;
  per_gpu: Spread | null;
  aggregate_median: number | null;
}

export interface ScalingCurve {
  family: string;
  config_name: string;
  workload_hash: string;
  workload_name: string;
  max_concurrency: number | null;
  baseline_tp: number;
  /**
   * False when the narrowest width measured was itself already parallel. Efficiency
   * against a TP=2 baseline is not the parallel efficiency a reader assumes, and the
   * view must say so rather than print a number that reads as one.
   */
  baseline_is_single_gpu: boolean;
  steps: ScalingStep[];
}

export interface ScalingGroup {
  group_id: string;
  label: string;
  gpu_host_id: string;
  gpu_host_name: string;
  gpu_model: string | null;
  vllm_version: string | null;
  bench_client_location: string;
  warnings: string[];
  run_count: number;
  curves: ScalingCurve[];
}

export interface Scaling {
  source: RunSource;
  run_count: number;
  truncated: boolean;
  limit: number;
  metric: string;
  metrics: Metric[];
  excluded: AnalysisExcluded;
  groups: ScalingGroup[];
  /** Config families measured at only one width — not an error, but not a curve either. */
  single_width_families: number;
}

export interface DeviceSummary {
  gpu_index: number;
  samples: number;
  sm_utilization_pct: number | null;
  sm_utilization_max: number | null;
  memory_used_bytes: number | null;
  power_watts: number | null;
}

export interface RunBalance {
  run_id: string;
  config_name: string;
  workload_name: string;
  tensor_parallel_size: number;
  gpu_count: number;
  replicate_idx: number;
  finished_at: string | null;
  devices: DeviceSummary[];
  /** Per metric, (max - min) / max across devices. Null when it cannot be measured. */
  imbalances: Record<string, number | null>;
  worst_imbalance: number | null;
  /** A one-device run has no balance to report — not the same as balanced. */
  is_single_device: boolean;
}

export interface DeviceBalanceGroup {
  group_id: string;
  label: string;
  gpu_host_id: string;
  gpu_host_name: string;
  gpu_model: string | null;
  vllm_version: string | null;
  bench_client_location: string;
  warnings: string[];
  run_count: number;
  runs: RunBalance[];
}

export interface DeviceBalance {
  source: RunSource;
  run_count: number;
  truncated: boolean;
  limit: number;
  metrics: { key: string; label: string }[];
  excluded: AnalysisExcluded;
  groups: DeviceBalanceGroup[];
  runs_without_telemetry: number;
}

export interface DiffLine {
  kind: "context" | "added" | "removed" | "header";
  text: string;
  left_no: number | null;
  right_no: number | null;
}

export interface ProvenanceDifference {
  field: string;
  label: string;
  left: string | null;
  right: string | null;
  /** True where this difference would stop the two sharing a chart series. */
  invalidating: boolean;
}

export interface ComparisonSide {
  point_id: string;
  config_hash: string;
  config_name: string;
  config_yaml: string;
  workload_name: string;
  tensor_parallel_size: number;
  gpu_count: number;
  replicates: number;
  spread_basis: SpreadBasis;
  spread_note: string;
  gpu_host_name: string;
  gpu_model: string | null;
  vllm_version: string | null;
  bench_client_location: string;
  metrics: Record<string, Spread>;
}

export interface MetricComparison {
  key: string;
  label: string;
  unit: string;
  better: "higher" | "lower";
  left: number | null;
  right: number | null;
  change: number | null;
  /** Null for "no change" as well as "unmeasurable" — never render unchanged as a win. */
  is_improvement: boolean | null;
}

export interface Comparison {
  source: RunSource;
  left: ComparisonSide;
  right: ComparisonSide;
  config_diff: DiffLine[];
  configs_identical: boolean;
  provenance_differences: ProvenanceDifference[];
  metrics: MetricComparison[];
}

/**
 * A named analysis view: which chart, over which runs, on which axes.
 *
 * Stores a query, never a set of run ids — reopening one next month includes the runs
 * measured since, which is what makes it a view of the data rather than a snapshot that
 * silently stops tracking reality while continuing to look current.
 */
export interface SavedView {
  id: string;
  name: string;
  description: string | null;
  view: string;
  source: RunSource;
  filters: Record<string, unknown>;
  options: Record<string, unknown>;
  created_at: string;
}

export interface SavedViewCreate {
  name: string;
  description?: string | null;
  view: string;
  source: RunSource;
  filters: Record<string, unknown>;
  options?: Record<string, unknown>;
}

/**
 * Human wording for a failure kind, with the raw value as the fallback.
 *
 * The fallback matters: a newer agent or a later build can record a kind this UI has
 * never heard of, and showing the raw string is a small ugliness where showing nothing
 * would hide the failure's only heading.
 */
export const FAILURE_LABELS: Record<string, string> = {
  agent_unreachable: "Host unreachable",
  agent_auth: "Agent refused the token",
  protocol_mismatch: "Agent protocol mismatch",
  engine_out_of_memory: "Engine ran out of memory",
  engine_config_rejected: "vLLM rejected the configuration",
  engine_load_failed: "Engine failed to start",
  engine_not_ready: "Engine never became ready",
  benchmark_timeout: "Benchmark exceeded its time budget",
  benchmark_failed: "Benchmark failed",
  result_schema_mismatch: "Result did not match the expected schema",
  internal: "Internal error",
};

export function failureLabel(kind: string): string {
  return FAILURE_LABELS[kind] ?? kind;
}

/** `/api/version` — what the control plane is actually running. */
export interface ApiVersion {
  version: string;
  protocol_version: number;
}
