import type { Host } from "./types";

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
