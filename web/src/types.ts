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
