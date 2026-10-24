import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createElement as h } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The console around the in-browser source: boot order and the camera, the store's toggles, and what the panels claim.
// Each test that boots imports a fresh copy of the store (boot() runs once per page load).

const gumSpy = () => {
  const calls: unknown[] = [];
  Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: {
    getUserMedia: (c: unknown) => { calls.push(c); return new Promise(() => {}); },
  } });
  return calls;
};
async function fresh() {
  vi.resetModules();
  const v = await import("@/lib/valence");
  const panels = await import("@/components/console/panels");
  return { v, panels };
}
const wait = (ms: number) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });

afterEach(() => {
  cleanup();
  delete (navigator as { mediaDevices?: unknown }).mediaDevices;
  window.history.replaceState(null, "", "/");
});

describe("boot and the camera preview", () => {
  it("the tour never asks for the camera on page load (the preview waits for boot)", async () => {
    window.history.replaceState(null, "", "/?demo=tour");
    const calls = gumSpy();
    const { v } = await fresh();
    const { Route } = await import("@/routes/index");
    const Index = (Route as unknown as { options: { component: () => React.ReactElement } }).options.component;
    await act(async () => { render(h(Index)); });
    await wait(50);
    expect([v.getState().source, v.getBrowserMode()]).toEqual(["browser", "tour"]);
    expect(calls).toEqual([]);
    v.setSource("mock");
  });

  it("?source=browser&perception=mock opens no camera or mic at all", async () => {
    window.history.replaceState(null, "", "/?source=browser&perception=mock");
    const calls = gumSpy();
    const { v } = await fresh();
    const { Route } = await import("@/routes/index");
    const Index = (Route as unknown as { options: { component: () => React.ReactElement } }).options.component;
    await act(async () => { render(h(Index)); });
    await wait(50);
    expect([v.getState().source, v.getBrowserMode(), v.isMockPerception()]).toEqual(["browser", "live", true]);
    expect(calls).toEqual([]);
    expect(screen.getByTestId("welcome-now").textContent).toMatch(/mock perception|Starting/);
    v.setSource("mock");
  });

  it("the mock source still shows the local preview once booted", async () => {
    const calls = gumSpy();
    const { v, panels } = await fresh();
    render(h(panels.CameraView));
    await wait(10);
    expect(calls).toEqual([]); // not booted yet
    await act(async () => { v.boot(); });
    await wait(10);
    expect(v.getState().source).toBe("mock");
    expect(calls).toHaveLength(1);
    v.setSource("mock");
  });
});

describe("console state around the in-browser source", () => {
  it("a new source starts with the simulation toggles off (a fresh engine has no overlay)", async () => {
    const { v } = await fresh();
    v.setSource("mock");
    await v.sendCommand({ kind: "sim", which: "face_occlusion", enabled: true });
    expect(v.getState().sim.face_occlusion).toBe(true);
    v.startBrowserSource("tour");
    expect(v.getState().sim).toEqual({ face_occlusion: false, low_light: false });
    v.setSource("mock");
  });

  it("self-report offers only the people tracked now (the room in room-only mode) and refuses taps unless running", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    const asked: number[] = [];
    v.setFeatureProvider((pid) => { asked.push(pid); return { feature_version: "emotion-v1", features: { a: 1 } }; });
    const pred = (pid: number) => ({ type: "prediction" as const, event_id: `p${pid}`, t: 1, person_id: pid, label: "Focused", label_display: "Focused",
      confidence: 0.8, probabilities: { Focused: 0.8 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up: false,
      layers: [], data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: "", source: "template" as const },
      source: "live", simulated_fields: [] });
    act(() => {
      v.handleMessage(pred(1)); // #1 left: their last card stays on screen
      v.handleMessage(pred(2));
      v.handleMessage({ type: "tracks", t: 2, capture_ts_ms: 2000, frame_w: 1280, frame_h: 720, mirrored: false,
        persons: [{ id: 2, label: "Person #2", bbox: [0.1, 0.1, 0.4, 0.9], fist_active: false, slump: false }] });
    });
    render(h(panels.SelfReport));
    expect(screen.queryByRole("button", { name: "#1" })).toBeNull();
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([2]);
    await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); });
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([2]);
    expect(screen.getByText(/paused or stopped: press Start/)).toBeInTheDocument();
    await act(async () => { await v.sendCommand({ kind: "control", action: "start" }); });
    act(() => { v.handleMessage({ type: "hello", contract_version: "1.0.0", mode: "live", room_only: true, states: [], feature_version: "", frame_w: 0, frame_h: 0, mirrored: false }); });
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([2, 0]);
    v.clearSelfReports();
    v.setSource("mock");
  });

  it("Start after a refused prompt clears the blocked-camera box once the new open begins, and the granted retry runs", async () => {
    // The engine and perception chunks stand in for the real ones (no model files in jsdom).
    const live = () => ({ kind: "", readyState: "live", stop() { this.readyState = "ended"; } });
    vi.doMock("@/engine/browser-engine", () => ({ createBrowserEngine: async () => ({
      states: [], featureVersion: "", push: () => null, tick: () => [], setRoomOnly() {}, setOverlay() {}, logBreak() {}, startBaseline() {},
      resetPeople() {}, baseline: () => ({ mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null }),
    }) }));
    vi.doMock("@/perception/index", () => ({ createPerception: () => ({ start() {}, stop() {}, onObservation() {}, counters: () => ({ frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0 }) }) }));
    const calls: unknown[] = [];
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: async (c: unknown) => {
      calls.push(c);
      if (calls.length === 1) throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
      const tracks = [{ ...live(), kind: "video" }, { ...live(), kind: "audio" }];
      return { getTracks: () => tracks, getVideoTracks: () => tracks.slice(0, 1), getAudioTracks: () => tracks.slice(1) };
    } } });
    try {
      const { v, panels } = await fresh();
      v.startBrowserSource("live");
      await wait(20);
      render(h("div", null, h(panels.SourceControls), h(panels.Welcome)));
      expect(screen.getByTestId("edge-errors").textContent).toMatch(/Camera and microphone blocked: allow them/);
      await act(async () => { await v.sendCommand({ kind: "control", action: "start" }); });
      await wait(20);
      expect(calls).toHaveLength(2);
      expect(screen.queryByTestId("edge-errors")).toBeNull();
      expect(v.getState().status).toMatchObject({ camera: true, mic: true });
      expect(screen.getByTestId("welcome-now").textContent).toMatch(/^Now running on your camera and microphone/);
      // An engine error is not about the sensors: a new open leaves it.
      act(() => { v.applyHealth({ edge_error: "engine: Error: boom" }); });
      await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); await v.sendCommand({ kind: "control", action: "start" }); });
      expect(v.getState().edgeError).toBe("engine: Error: boom");
      v.setSource("mock");
    } finally {
      vi.doUnmock("@/engine/browser-engine");
      vi.doUnmock("@/perception/index");
    }
  });

  it("the tour disables the room-baseline capture", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("tour");
    render(h(panels.RoomBaselineCard));
    expect(screen.getByRole("button", { name: /Capture 90 s/ })).toBeDisabled();
    expect(screen.getByText(/tour carries its own baseline/)).toBeInTheDocument();
    v.setSource("mock");
  });
});

describe("what the in-browser panels claim", () => {
  const status = (extra: object) => ({ type: "status" as const, mode: "live" as const, camera: true, mic: true, fps: 9.8, loop_latency_ms: 20,
    narrator: "template", offline: false, uptime_s: 30, degraded: [], purge_violation: false, ...extra });

  it("Network reads SEALED only when this host's CSP says connect-src 'self'; Welcome stays true either way", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    act(() => { v.handleMessage(status({ sealed: false })); });
    const { unmount } = render(h("div", null, h(panels.SystemStatus), h(panels.Welcome)));
    expect(screen.getByText("not sealed (no CSP on this host)")).toBeInTheDocument();
    expect(screen.queryByText(/SEALED · connect-src/)).toBeNull();
    expect(screen.getByTestId("welcome").textContent).toMatch(/deployed site's CSP .* forbids the page from connecting to any other origin\. This host serves no CSP/);
    expect(screen.getByTestId("welcome").textContent).not.toMatch(/not allowed to send data anywhere/);
    unmount();
    act(() => { v.handleMessage(status({ sealed: true })); });
    render(h("div", null, h(panels.SystemStatus), h(panels.Welcome)));
    expect(screen.getByText("SEALED · connect-src 'self'")).toBeInTheDocument();
    expect(screen.getByTestId("welcome").textContent).not.toMatch(/This host serves no CSP/);
    v.setSource("mock");
  });

  it("shows what is still loading, and a 10 Hz face rate as information, not a fault", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    act(() => { v.handleMessage(status({ camera: true, mic: true, loading: "loading perception models · 12.3 MB", face_hz: 9.8 })); });
    render(h("div", null, h(panels.SystemStatus), h(panels.Welcome)));
    expect(screen.getByTestId("status-loading").textContent).toBe("loading perception models · 12.3 MB…");
    expect(screen.getByTestId("welcome-now").textContent).toBe("Starting: loading perception models · 12.3 MB…");
    const face = screen.getByText(/blink counts undercount below ~15 Hz/).parentElement;
    expect(face?.className).not.toMatch(/text-destructive/);
    v.setSource("mock");
  });

  it("Welcome follows Stop and Pause at once, even while the last status still says loading", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    const loading = status({ camera: false, mic: false, loading: "waiting for camera and microphone" });
    act(() => { v.handleMessage(loading); });
    render(h(panels.Welcome));
    const now = () => screen.getByTestId("welcome-now").textContent;
    expect(now()).toBe("Starting: waiting for camera and microphone…");
    await act(async () => { await v.sendCommand({ kind: "control", action: "stop" }); });
    act(() => { v.handleMessage(loading); }); // a status from before the Stop
    expect(now()).toBe("Stopped: camera and microphone are released. Press Start to resume.");
    act(() => { v.startBrowserSource("tour"); });
    await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); });
    expect(now()).toBe("Paused: press Start to resume the tour.");
    v.setSource("mock");
  });

  it("a chunk of an older build reads as 'Valence Web was updated — reload the page', with a reload button", async () => {
    const { v, panels } = await fresh();
    const stale = v.loadError(new TypeError("Failed to fetch dynamically imported module: https://x.example/assets/perception-Di4KJfYG.js"));
    expect(String(stale)).toBe(`Error: ${v.STALE_BUILD}`);
    expect(String(v.loadError(new TypeError("Importing a module script failed.")))).toContain(v.STALE_BUILD);
    const other = new Error("no served models at /models/core/");
    expect(v.loadError(other)).toBe(other);
    v.startBrowserSource("tour");
    act(() => { v.applyHealth({ edge_error: v.STALE_BUILD }); });
    render(h(panels.SourceControls));
    expect(screen.getByTestId("edge-errors").textContent).toContain("Valence Web was updated — reload the page");
    expect(screen.getByRole("button", { name: /Reload/ })).toBeInTheDocument();
    v.setSource("mock");
  });
});
