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
  created_at: string;
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
  sample_count: number;
}
