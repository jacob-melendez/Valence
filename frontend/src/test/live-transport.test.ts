import { afterEach, describe, expect, it, vi } from "vitest";

import { CORE_STATES, getState, setSource } from "@/lib/valence";

class FakeWebSocket {
  static last: FakeWebSocket | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) { FakeWebSocket.last = this; }
  close() { /* nothing to release */ }
}

// fetch replies are held until the test releases them, to reproduce a slow backend.
const pending: { url: string; reply: (body: unknown) => void }[] = [];
function stubNetwork() {
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal("fetch", (url: string) => new Promise((resolve) => {
    pending.push({ url, reply: (body) => resolve({ ok: true, status: 200, json: async () => body }) });
  }));
}

afterEach(() => {
  setSource("mock");
  vi.unstubAllGlobals();
  pending.length = 0;
});

describe("live transport", () => {
  it("drops REST replies that arrive after switching to mock", async () => {
    stubNetwork();
    setSource("live", "http://127.0.0.1:9");
    FakeWebSocket.last?.onopen?.();
    await vi.waitFor(() => expect(pending.length).toBe(1));
    expect(pending[0]?.url).toContain("/api/states");
    setSource("mock");
    pending[0]?.reply([...CORE_STATES, { id: "Tense", display: "Tense / Visible Frustration", tier: "P1" }]);
    await new Promise((r) => setTimeout(r, 20));
    expect(getState().source).toBe("mock");
    expect(getState().states).toHaveLength(4);
    expect(pending).toHaveLength(1); // the stale loader stopped instead of fetching the rest
  });

  it("drops replies from the previous backend after connecting to another one", async () => {
    stubNetwork();
    setSource("live", "http://127.0.0.1:9");
    FakeWebSocket.last?.onopen?.();
    await vi.waitFor(() => expect(pending.length).toBe(1));
    setSource("live", "http://127.0.0.1:10"); // live again: only the connection generation tells them apart
    pending[0]?.reply([...CORE_STATES, { id: "Tense", display: "Tense / Visible Frustration", tier: "P1" }]);
    await new Promise((r) => setTimeout(r, 20));
    expect(getState().baseUrl).toBe("http://127.0.0.1:10");
    expect(getState().states).toHaveLength(4);
  });

  it("applies replies for the current connection", async () => {
    stubNetwork();
    setSource("live", "http://127.0.0.1:9");
    FakeWebSocket.last?.onopen?.();
    await vi.waitFor(() => expect(pending.length).toBe(1));
    pending[0]?.reply([...CORE_STATES, { id: "Tense", display: "Tense / Visible Frustration", tier: "P1" }]);
    await vi.waitFor(() => expect(getState().states).toHaveLength(5));
  });
});
