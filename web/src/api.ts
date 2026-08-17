import type {
  Analysis,
  AnalysisQuery,
  Comparison,
  Config,
  DeviceBalance,
  Host,
  Run,
  RunSource,
  RunTelemetry,
  Scaling,
  Sweep,
  SweepCreate,
  Workload,
} from "./types";

const BASE = "/api";

class RequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "RequestError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    // The API returns actionable detail for handshake failures — unreachable agent,
    // bad token, protocol mismatch. Surfacing a generic "request failed" here would
    // throw away the only part of the message that tells the operator what to fix.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Response body was not JSON; the status line is all we have.
    }
    throw new RequestError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  listHosts: () => request<Host[]>("/hosts"),

  registerHost: (name: string, agentUrl: string) =>
    request<Host>("/hosts", {
      method: "POST",
      body: JSON.stringify({ name, agent_url: agentUrl }),
    }),

  refreshHost: (id: string) => request<Host>(`/hosts/${id}/refresh`, { method: "POST" }),

  deleteHost: (id: string) => request<void>(`/hosts/${id}`, { method: "DELETE" }),
};

export { RequestError };

export const configsApi = {
  list: () => request<Config[]>("/configs"),
  create: (name: string, yaml: string) =>
    request<Config>("/configs", { method: "POST", body: JSON.stringify({ name, yaml }) }),
};

export const workloadsApi = {
  list: () => request<Workload[]>("/workloads"),
  create: (body: Partial<Workload> & { name: string }) =>
    request<Workload>("/workloads", { method: "POST", body: JSON.stringify(body) }),
};

export const runsApi = {
  list: () => request<Run[]>("/runs"),
  get: (id: string) => request<Run>(`/runs/${id}`),
  // Separate from get(): the runs list polls every 2s while anything is in flight, and
  // a long run's telemetry is thousands of samples.
  telemetry: (id: string) => request<RunTelemetry>(`/runs/${id}/telemetry`),
  create: (gpuHostId: string, serverConfigId: string, workloadId: string) =>
    request<Run>("/runs", {
      method: "POST",
      body: JSON.stringify({
        gpu_host_id: gpuHostId,
        server_config_id: serverConfigId,
        workload_id: workloadId,
      }),
    }),
};

export const sweepsApi = {
  list: () => request<Sweep[]>("/sweeps"),
  get: (id: string) => request<Sweep>(`/sweeps/${id}`),
  create: (body: SweepCreate) =>
    request<Sweep>("/sweeps", { method: "POST", body: JSON.stringify(body) }),
  cancel: (id: string) => request<Sweep>(`/sweeps/${id}/cancel`, { method: "POST" }),
};

/**
 * The one query behind every analysis view.
 *
 * `source` takes a single population — there is no value meaning "both", so a synthetic
 * run cannot be requested alongside a real one (invariant 7).
 */
export const analysisApi = {
  points: (query: AnalysisQuery = {}) => {
    const params = new URLSearchParams();
    if (query.source) params.set("source", query.source);
    if (query.hostId) params.set("host_id", query.hostId);
    if (query.paretoX) params.set("pareto_x", query.paretoX);
    if (query.paretoY) params.set("pareto_y", query.paretoY);
    for (const id of query.sweepIds ?? []) params.append("sweep_id", id);
    for (const tp of query.tensorParallelSizes ?? []) {
      params.append("tensor_parallel_size", String(tp));
    }
    const qs = params.toString();
    return request<Analysis>(`/analysis/points${qs ? `?${qs}` : ""}`);
  },

  // Deliberately no tensor_parallel_size filter: narrowing the axis under study to one
  // value is the only query that cannot produce a curve.
  scaling: (query: AnalysisQuery = {}) => {
    const params = new URLSearchParams();
    if (query.source) params.set("source", query.source);
    if (query.hostId) params.set("host_id", query.hostId);
    for (const id of query.sweepIds ?? []) params.append("sweep_id", id);
    const qs = params.toString();
    return request<Scaling>(`/analysis/scaling${qs ? `?${qs}` : ""}`);
  },

  // Two point ids, deliberately allowed to cross a comparability boundary: a chart must
  // not silently overlay two vLLM versions, but a side-by-side the reader asked for is
  // where that comparison is the subject and every difference is listed back.
  compare: (left: string, right: string, source: RunSource = "real") =>
    request<Comparison>(
      `/analysis/compare?${new URLSearchParams({ left, right, source }).toString()}`,
    ),

  deviceBalance: (query: AnalysisQuery = {}) => {
    const params = new URLSearchParams();
    if (query.source) params.set("source", query.source);
    if (query.hostId) params.set("host_id", query.hostId);
    for (const id of query.sweepIds ?? []) params.append("sweep_id", id);
    const qs = params.toString();
    return request<DeviceBalance>(`/analysis/device-balance${qs ? `?${qs}` : ""}`);
  },
};
