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
    // No empty camera frame: the Room view says the tour has no camera.
    expect(screen.getByTestId("room-view-tour").textContent).toMatch(/^Tour: a scripted, simulated session; no camera\./);
    // Pausing the tour releases no camera or mic (none was opened): the overlay says what is paused.
    await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); });
    expect(screen.getByText("PAUSED · simulated tour")).toBeInTheDocument();
    expect(screen.queryByText(/camera and mic released/)).toBeNull();
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
    const pred = (pid: number, warming_up = false) => ({ type: "prediction" as const, event_id: `p${pid}`, t: 1, person_id: pid, label: "Focused", label_display: "Focused",
      confidence: 0.8, probabilities: { Focused: 0.8 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up,
      layers: [], data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: "", source: "template" as const },
      source: "live", simulated_fields: [] });
    act(() => {
      v.handleMessage(pred(1)); // #1 left: their last card stays on screen
      v.handleMessage(pred(2, true));
      v.handleMessage({ type: "tracks", t: 2, capture_ts_ms: 2000, frame_w: 1280, frame_h: 720, mirrored: false,
        persons: [{ id: 2, label: "Person #2", bbox: [0.1, 0.1, 0.4, 0.9], fist_active: false, slump: false }] });
    });
    render(h(panels.SelfReport));
    expect(screen.queryByRole("button", { name: "#1" })).toBeNull();
    // Warm-up: no prompt and no tap (docs/EMOTION_FEATURES.md section 6), so warm-up inputs never reach the file.
    expect(screen.getByText("available after warm-up")).toBeInTheDocument();
    expect(screen.getByTestId("self-report").className).not.toMatch(/ring-2/);
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([]);
    expect(screen.getByText(/Warming up: taps open after the first minute/)).toBeInTheDocument();
    act(() => { v.handleMessage(pred(2)); });
    expect(screen.getByText("tap now")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([2]);
    await act(async () => { await v.sendCommand({ kind: "control", action: "pause" }); });
    fireEvent.click(screen.getByRole("img", { name: /Tap your valence/ }), { clientX: 10, clientY: 10 });
    expect(asked).toEqual([2]);
    expect(screen.getByText(/paused or stopped: press Start/)).toBeInTheDocument();
    await act(async () => { await v.sendCommand({ kind: "control", action: "start" }); });
    act(() => { v.handleMessage({ type: "hello", contract_version: "1.0.0", mode: "live", room_only: true, states: [], feature_version: "", frame_w: 0, frame_h: 0, mirrored: false }); });
    act(() => { v.handleMessage(pred(0)); }); // the room's card, past its warm-up
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
      // The Welcome card carries the error itself: on phones the Source panel is at the end of the page.
      expect(screen.getByTestId("welcome-now").textContent).toMatch(/^Not running\. Camera and microphone blocked: allow them .* then press Start\.$/);
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

  it("camera busy, mic fine: Welcome runs on the microphone and says why the camera is missing", async () => {
    vi.doMock("@/engine/browser-engine", () => ({ createBrowserEngine: async () => ({
      states: [], featureVersion: "", push: () => null, tick: () => [], setRoomOnly() {}, setOverlay() {}, logBreak() {}, startBaseline() {},
      resetPeople() {}, baseline: () => ({ mode: "auto", complete: false, target_samples: 90, db_samples: 0, lux_samples: 0, db_median: null, db_iqr: null, lux_median: null }),
    }) }));
    vi.doMock("@/perception/index", () => ({ createPerception: () => ({ start() {}, stop() {}, onObservation() {}, counters: () => ({ frames_decoded: 0, frames_purged: 0, audio_blocks_purged: 0 }) }) }));
    // The combined call and the camera alone fail as busy (NotReadableError); the microphone alone opens.
    vi.stubGlobal("MediaStream", class {
      tracks: { kind: string; readyState: string; stop(): void }[] = [];
      getTracks() { return this.tracks; }
      getVideoTracks() { return this.tracks.filter((t) => t.kind === "video"); }
      getAudioTracks() { return this.tracks.filter((t) => t.kind === "audio"); }
      addTrack(t: { kind: string; readyState: string; stop(): void }) { this.tracks.push(t); }
    });
    const calls: MediaStreamConstraints[] = [];
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: async (c: MediaStreamConstraints) => {
      calls.push(c);
      if (c.video) throw Object.assign(new Error("busy"), { name: "NotReadableError" });
      const tracks = [{ kind: "audio", readyState: "live", stop() { this.readyState = "ended"; } }];
      return { getTracks: () => tracks, getVideoTracks: () => [], getAudioTracks: () => tracks };
    } } });
    try {
      const { v, panels } = await fresh();
      v.startBrowserSource("live");
      // The (mocked) engine chunk loads first; then the three opens and perception run.
      for (let i = 0; i < 100 && !(v.getState().status?.mic && !(v.getState().status as { loading?: string }).loading); i++) await wait(10);
      render(h("div", null, h(panels.SourceControls), h(panels.Welcome)));
      expect(calls.map((c) => [!!c.video, !!c.audio])).toEqual([[true, true], [true, false], [false, true]]);
      expect(v.getState().status).toMatchObject({ camera: false, mic: true });
      const CAMERA_ERR = "Camera could not start: another app may be using it, or the system blocked access.";
      expect(v.getState().edgeError).toBe(CAMERA_ERR);
      expect(screen.getByTestId("welcome-now").textContent).toMatch(/^Now running on your microphone \(no camera\)\. Camera could not start: another app may be using it/);
      expect(screen.getByTestId("welcome-now").textContent).toContain(CAMERA_ERR);
      expect(screen.getByTestId("welcome-now").textContent).not.toMatch(/Keep a relaxed, neutral face/);
      v.setSource("mock");
    } finally {
      vi.doUnmock("@/engine/browser-engine");
      vi.doUnmock("@/perception/index");
      vi.unstubAllGlobals();
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

  it("the tour's status reads TOUR with no FPS row; on a local host the header keeps the backend controls", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("tour");
    act(() => { v.handleMessage(status({ mode: "replay", camera: false, mic: false, fps: 0 })); });
    render(h("div", null, h(panels.Header), h(panels.SystemStatus)));
    expect(screen.getByText("TOUR (SIMULATED)")).toBeInTheDocument();
    expect(screen.queryByText("FPS")).toBeNull();
    expect(screen.getByTestId("source-live")).toBeInTheDocument();
    expect(screen.getByLabelText("Backend URL")).toBeInTheDocument();
    v.setSource("mock");
  });

  it("the mock stream reads MOCK (GENERATED) in Source and System status, never Live or mode: LIVE", async () => {
    const { v, panels } = await fresh();
    v.setSource("mock");
    // The mock's own status says live with a camera (what its generator stands in for).
    act(() => { v.handleMessage(status({ mode: "live", camera: true, mic: true })); });
    render(h("div", null, h(panels.SourceControls), h(panels.SystemStatus)));
    expect(screen.getByTestId("backend-mode").textContent).toBe("MOCK (GENERATED)");
    expect(screen.getAllByText("MOCK (GENERATED)")).toHaveLength(2);
    expect(screen.getByTestId("source-controls").textContent).not.toMatch(/mode: LIVE/);
    expect(screen.queryByText("● Live (camera + mic)")).toBeNull();
    expect(screen.getByRole("button", { name: "● Go live (camera + mic)" }).className).not.toMatch(/ring-2/);
    v.setSource("mock");
  });

  it("the in-browser camera reads CAMERA (ON-DEVICE) in System status", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    act(() => { v.handleMessage(status({})); });
    render(h(panels.SystemStatus));
    expect(screen.getByText("CAMERA (ON-DEVICE)")).toBeInTheDocument();
    expect(screen.queryByText("LIVE")).toBeNull();
    v.setSource("mock");
  });

  it("the purge line shows only on ticks that purged something; no transcripts tile or compliment count live", async () => {
    const { v, panels } = await fresh();
    if (!Element.prototype.scrollTo) Element.prototype.scrollTo = () => {}; // jsdom: the log scrolls to its end
    const LINE = "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.";
    const gov = (tick: number, n: number) => ({ type: "governance" as const, t: tick * 5, tick, message: LINE, frames_decoded: n,
      frames_purged: n, audio_blocks_purged: 0, transcripts_purged: 0, violations: 0, live_raw_buffers: 0, audit_hash: null, retention_s: 0, purge_gap: 0 });
    v.startBrowserSource("tour");
    act(() => { v.handleMessage(gov(1, 0)); v.handleMessage(gov(2, 0)); });
    const { unmount } = render(h(panels.Governance));
    expect(screen.queryByText(LINE)).toBeNull();
    expect(screen.getAllByText(/Simulated tour: no camera or mic, so nothing to purge/)).toHaveLength(2);
    expect(screen.queryByText("Transcripts purged")).toBeNull();
    expect(screen.getByTestId("governance-note").textContent).toMatch(/^Tour: no camera or mic/);
    // Tour ticks are 5 s of tour time, not of wall time.
    expect(screen.getByTestId("governance-note").textContent).toMatch(/engine ticks of 5 s tour time each \(about 0\.5 s apart at 10× speed\)/);
    // Three tiles in the browser: the last one spans both columns instead of sitting alone.
    expect(screen.getByText("Violations").parentElement?.className).toMatch(/col-span-2/);
    expect(screen.getByText("Frames purged").parentElement?.className).not.toMatch(/col-span-2/);
    unmount();
    v.startBrowserSource("live");
    act(() => { v.handleMessage(gov(1, 0)); v.handleMessage(gov(2, 50)); v.handleMessage(gov(3, 50)); });
    act(() => { v.handleMessage({ type: "state_vector", t: 15, person_id: 0, ambient_db: 50, light_lux: 300, hours_since_break: 0,
      time_of_day: "afternoon", clenched_fists_5m: 0, compliments_15m: 0, laughter_15m: 0, posture_slump: 0, head_in_hands: 0,
      social_withdrawal: 0, face_visible: true, facial_valence: 0, window_fill_min: 1, source: "live", simulated_fields: [] }); });
    render(h("div", null, h(panels.Governance), h(panels.StateVectors)));
    expect(screen.getAllByText(LINE)).toHaveLength(1); // tick 2 only
    expect(screen.getAllByText(/No camera frame or audio block this tick, so nothing to purge/)).toHaveLength(2);
    expect(screen.getByTestId("governance-note").textContent).toMatch(/purged follows decoded by construction; decoded may lead by the one frame being processed/);
    expect(screen.getByTestId("governance-note").textContent).toMatch(/Log lines are 5 s engine ticks;/);
    expect(screen.getByText("n/a (no speech)")).toBeInTheDocument();
    v.setSource("mock");
  });

  it("a mic-only run (no camera frame) never shows the raw-frame purge line while its audio counter climbs", async () => {
    const { v, panels } = await fresh();
    if (!Element.prototype.scrollTo) Element.prototype.scrollTo = () => {};
    const LINE = "[GOVERNANCE] Raw biometric frame purged successfully. Appended state vector to timeline.";
    const gov = (tick: number, audio: number) => ({ type: "governance" as const, t: tick * 5, tick, message: LINE, frames_decoded: 0,
      frames_purged: 0, audio_blocks_purged: audio, transcripts_purged: 0, violations: 0, live_raw_buffers: 0, audit_hash: null, retention_s: 0, purge_gap: 0 });
    v.startBrowserSource("live");
    act(() => { v.handleMessage(gov(1, 40)); v.handleMessage(gov(2, 80)); v.handleMessage(gov(3, 80)); });
    render(h(panels.Governance));
    expect(screen.queryByText(LINE)).toBeNull();
    expect(screen.queryByText(/Raw biometric frame/)).toBeNull();
    expect(screen.getAllByText("[GOVERNANCE] No camera frame this tick; audio reduced to levels and purged. Appended state vector to timeline.")).toHaveLength(2);
    expect(screen.getAllByText(/No camera frame or audio block this tick, so nothing to purge/)).toHaveLength(1); // tick 3
    v.setSource("mock");
  });

  it("the person card names its model, holds warm-up labels as provisional and points to the emotion map's own model", async () => {
    const { v, panels } = await fresh();
    v.startBrowserSource("live");
    act(() => {
      v.handleMessage(status({}));
      v.handleMessage({ type: "hello", contract_version: "1.0.0", mode: "live", room_only: false, states: v.CORE_STATES, feature_version: "x",
        frame_w: 0, frame_h: 0, mirrored: false, affect_states: [{ id: "Focused", display: "Focused", valence: 0.3, arousal: 0.2 }, { id: "Calm", display: "Calm", valence: 0.4, arousal: -0.4 }] });
    });
    const pred = (warming_up: boolean) => ({ type: "prediction" as const, event_id: "p1-000005", t: 1, person_id: 1, label: "Focused", label_display: "Focused",
      confidence: 0.96, probabilities: { Focused: 0.96 }, face_only: { label: null, confidence: null }, mask_override: false, warming_up,
      layers: [], data_quality: { face_visible_frac: 1, lux: null, ood_features: [] }, explanation: { text: "", source: "template" as const },
      source: "live", simulated_fields: [],
      affect: { valence: 0.3, arousal: 0.2, label: "Focused", label_display: "Focused", confidence: 0.84, probabilities: { Focused: 0.84 }, model: "emotion-v1@test" } });
    act(() => { v.handleMessage(pred(true)); });
    const Card = () => { const p = v.useValence().predictions[1]; return p ? h(panels.PersonCard, { p }) : null; };
    render(h("div", null, h(Card), h(panels.AffectMap), h(panels.Welcome)));
    expect(screen.getByText("4-state context model · p1-000005")).toHaveAttribute("title", "event p1-000005");
    expect(screen.getByText("Context model · 4 states")).toBeInTheDocument();
    expect(screen.getByText("2-state emotion model · emotion-v1@test")).toBeInTheDocument();
    // Alone on the emotion map, the person gets no "#1" tag over the anchor names (the card beside it names them).
    expect([...screen.getByTestId("affect-map").querySelectorAll("svg text")].map((t) => t.textContent)).not.toContain("#1");
    // Anchor names dim their fill only: a group opacity would also thin the halo that keeps them readable over a ring.
    expect(screen.getByTestId("affect-map").querySelector("svg g[opacity]")).toBeNull();
    const anchorNames = [...screen.getByTestId("affect-map").querySelectorAll("svg g")].filter((g) => g.querySelector("title")).map((g) => g.querySelector("text"));
    expect(anchorNames.length).toBeGreaterThan(0);
    expect(anchorNames.every((t) => t?.getAttribute("fill-opacity") === "0.6")).toBe(true);
    expect(screen.getByTestId("model-note").textContent).toBe(
      "This card is the 4-state context model. The emotion map below reads 2 states with its own model (now Focused 84%), so the two can differ.");
    // Warm-up: a provisional label, and the 20 s face baseline asked for on the card and in the Welcome line.
    expect(screen.getByTestId("warmup-badge").textContent).toBe("warming up · provisional for the first minute");
    expect(screen.getByText(/Keep a relaxed, neutral face: its first 20 s on camera/)).toBeInTheDocument();
    expect(screen.getByTestId("welcome-now").textContent).toMatch(/^Now running on your camera and microphone\. Keep a relaxed, neutral face for the first 20 s.*simulated sessions only/);
    act(() => { v.handleMessage(pred(false)); });
    expect(screen.queryByTestId("warmup-badge")).toBeNull();
    expect(screen.queryByText(/Keep a relaxed, neutral face: its first 20 s/)).toBeNull();
    expect(screen.getByTestId("simulated-cue").textContent).toBe("a signal · model trained on simulated data only");
    v.setSource("mock");
  });

  it("Evaluation: no chart legend over a table-only summary; rows in plain words with the model id as tooltip", async () => {
    const { v, panels } = await fresh();
    const { plainEval } = await import("@/lib/browser-source");
    const { evalSummaryOf } = await import("@/engine/browser-engine");
    v.startBrowserSource("tour");
    // The source hands this over once the engine is up (onEval); set directly here.
    Object.assign(v.getState(), { evalSummary: plainEval(evalSummaryOf({ summary: { m2: { test_macro_f1: 0.7911 }, m0: { test_macro_f1: 0.4153 } } }, null)) });
    render(h(panels.Evaluation));
    expect(screen.queryByText("Context model")).toBeNull();
    expect(screen.queryByText("Face-only baseline")).toBeNull();
    expect(screen.getByText("Face-only baseline · face snapshot (2 features), 4 states")).toHaveAttribute("title", "m0");
    expect(screen.getByText(/ECE = calibration error, lower is better · — = not recorded/)).toBeInTheDocument();
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
