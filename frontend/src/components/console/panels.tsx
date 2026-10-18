import { useEffect, useRef, useState } from "react";
import {
  LAYERS, REPLAY_SPEEDS, baselineProgress, baselineWarning, figureUrl, fmt, getState, provenanceBadges, sendCommand, setSource,
  shownExplanation, stateName, useValence,
  type EvalSummary, type PredictionMsg, type StateKey, type StateVectorMsg,
} from "@/lib/valence";

export const STATE_VAR: Record<StateKey | "Overwhelmed" | "Tense", string> = {
  Focused: "var(--state-focused)",
  Frustrated_Masked: "var(--state-frustrated)",
  Exhausted: "var(--state-exhausted)",
  Collaborative_Happy: "var(--state-happy)",
  Overwhelmed: "var(--state-overwhelmed)",
  Tense: "var(--state-tense)",
};
// Any further state the backend adds gets a colour by its position in the active state set.
const EXTRA_VARS = ["var(--state-overwhelmed)", "var(--state-tense)", "var(--primary)", "var(--warn)"];
const stateColor = (id: string | null | undefined, fallback = "var(--primary)") => {
  if (!id) return fallback;
  const known = (STATE_VAR as Record<string, string | undefined>)[id];
  if (known) return known;
  const i = getState().states.findIndex((x) => x.id === id);
  return i >= 0 ? EXTRA_VARS[i % EXTRA_VARS.length] ?? fallback : fallback;
};
const pct = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—");
/** The backend's display text for a prediction (e.g. "Frustrated", "Upbeat"); the state name if missing. */
const labelText = (p: PredictionMsg) => p.label_display || stateName(p.label);

// ---------- Header + controls ----------
export function Header() {
  const s = useValence();
  const [url, setUrl] = useState(s.baseUrl);
  useEffect(() => { setUrl(s.baseUrl); }, [s.baseUrl]); // picks up ?api= / saved URL resolved after hydration
  const connLabel = { mock: "MOCK STREAM", connecting: "CONNECTING…", open: "LIVE · CONNECTED", closed: "DISCONNECTED · RETRYING" }[s.conn];
  const connColor = s.conn === "open" || s.conn === "mock" ? "bg-ok" : s.conn === "connecting" ? "bg-warn" : "bg-destructive";
  return (
    <header className="flex flex-wrap items-center gap-4 border-b px-6 py-4">
      <div className="mr-auto">
        <h1 className="text-3xl font-bold tracking-tight">Valence <span className="text-primary">Console</span></h1>
        <p className="font-mono text-xs text-muted-foreground">context-aware affect · no faces stored · no frames transmitted</p>
      </div>
      <div className="flex items-center gap-2 rounded-md border px-3 py-2 font-mono text-sm">
        <span className={`h-3 w-3 rounded-full ${connColor} pulse-dot`} /> {connLabel}
      </div>
      {baselineWarning(s) && (
        <span data-testid="baseline-warning" title="Live sensors are read against an automatic baseline. Capture the room baseline (90 s) before judging."
          className="rounded-md bg-warn px-3 py-2 font-mono text-sm font-bold text-primary-foreground">⚠ ROOM BASELINE NOT CAPTURED</span>
      )}
      {provenanceBadges(s).map((b) => (
        <span key={b.label} title={b.title} data-testid="provenance-badge"
          className={`rounded-md px-3 py-2 font-mono text-sm font-bold ${b.tone === "warn" ? "bg-warn text-primary-foreground" : b.tone === "live" ? "bg-ok text-primary-foreground" : "border text-muted-foreground"}`}>
          {b.label}
        </span>
      ))}
      <div className="flex items-center gap-1 rounded-md border p-1 font-mono text-sm">
        {(["mock", "live"] as const).map((m) => (
          <button key={m} onClick={() => setSource(m, url)}
            className={`rounded px-3 py-1.5 uppercase ${s.source === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}>
            {m}
          </button>
        ))}
      </div>
      <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); setSource("live", url); }}>
        <input value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Backend URL"
          className="w-56 rounded-md border bg-input/30 px-3 py-2 font-mono text-sm" />
        <button className="rounded-md border px-3 py-2 font-mono text-sm hover:bg-accent">Connect</button>
      </form>
    </header>
  );
}

function Btn({ children, onClick, tone = "default", active, disabled, small }: { children: React.ReactNode; onClick: () => void; tone?: "default" | "danger" | "primary"; active?: boolean; disabled?: boolean; small?: boolean }) {
  const cls = tone === "danger" ? "border-destructive text-destructive hover:bg-destructive hover:text-destructive-foreground"
    : tone === "primary" ? "border-primary text-primary hover:bg-primary hover:text-primary-foreground" : "hover:bg-accent";
  return (
    <button onClick={onClick} disabled={disabled}
      className={`rounded-md border font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${small ? "px-3 py-1.5 text-sm" : "px-4 py-2.5 text-base"} ${cls} ${active ? "ring-2 ring-ring" : ""}`}>
      {children}
    </button>
  );
}

export function Controls() {
  const s = useValence();
  return (
    <div className="flex flex-wrap items-center gap-3 border-b px-6 py-3">
      <Btn tone="primary" active={s.running === "running"} onClick={() => sendCommand({ kind: "control", action: "start" })}>▶ Start</Btn>
      <Btn active={s.running === "paused"} onClick={() => sendCommand({ kind: "control", action: "pause" })}>❚❚ Pause</Btn>
      <Btn tone="danger" active={s.running === "stopped"} onClick={() => sendCommand({ kind: "control", action: "stop" })}>■ Stop · Kill switch</Btn>
      <Btn onClick={() => sendCommand({ kind: "break" })}>☕ Log a break</Btn>
      <Btn onClick={() => sendCommand({ kind: "demo_reset" })}>↺ Demo reset</Btn>
      <Toggle label="Room-only mode" on={s.roomOnly} onChange={(v) => sendCommand({ kind: "mode", room_only: v })} />
      <Toggle label="Simulate face hidden" on={s.sim.face_occlusion} onChange={(v) => sendCommand({ kind: "sim", which: "face_occlusion", enabled: v })} />
      <Toggle label="Simulate low light" on={s.sim.low_light} onChange={(v) => sendCommand({ kind: "sim", which: "low_light", enabled: v })} />
      {s.lastCommand && (
        <span className={`ml-auto font-mono text-xs ${s.lastCommand.ok ? "text-muted-foreground" : "text-destructive"}`}>{s.lastCommand.text}</span>
      )}
    </div>
  );
}

function Toggle({ label, on, onChange, small }: { label: string; on: boolean; onChange: (v: boolean) => void; small?: boolean }) {
  return (
    <button onClick={() => onChange(!on)} role="switch" aria-checked={on} aria-label={label} className={`flex items-center gap-2 ${small ? "text-sm" : "text-base"}`}>
      <span className={`relative h-6 w-11 rounded-full border transition-colors ${on ? "bg-warn" : "bg-muted"}`}>
        <span className={`absolute top-0.5 h-4.5 w-4.5 rounded-full bg-foreground transition-all ${on ? "left-5.5" : "left-0.5"}`} />
      </span>
      {label}
    </button>
  );
}

// ---------- Operator: source + room baseline ----------
const selectCls = "min-w-0 rounded-md border bg-card px-3 py-1.5 text-sm";

export function SourceControls() {
  const s = useValence();
  const [scenario, setScenario] = useState("full_demo");
  const [recording, setRecording] = useState("");
  const [speed, setSpeed] = useState<number>(1);
  const [loop, setLoop] = useState(false);
  const scenarios = s.scenarios.length ? s.scenarios : [{ name: "full_demo" }];
  const rec = recording || s.recordings[0]?.name || "";
  return (
    <div className="panel" data-testid="source-controls">
      <div className="panel-title">Source</div>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <Btn small tone="primary" active={s.status?.mode === "live"} onClick={() => sendCommand({ kind: "live" })}>● Go live (camera + mic)</Btn>
          <span className="font-mono text-xs text-muted-foreground">mode: <span className="text-foreground" data-testid="backend-mode">{String(s.status?.mode ?? "—").toUpperCase()}</span></span>
        </div>
        {(s.edgeError || s.micError) && (
          <div className="space-y-1 rounded-md border border-destructive bg-destructive/10 px-3 py-2 font-mono text-xs text-destructive" data-testid="edge-errors">
            {s.edgeError && <div>edge: {s.edgeError}</div>}
            {s.micError && <div>mic: {s.micError} (running video-only)</div>}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <span className="w-20 text-sm text-muted-foreground">Speed</span>
          <div className="flex items-center gap-1 rounded-md border p-0.5 font-mono text-sm" role="group" aria-label="Replay speed">
            {REPLAY_SPEEDS.map((v) => (
              <button key={v} onClick={() => setSpeed(v)} aria-pressed={speed === v}
                className={`rounded px-2.5 py-1 ${speed === v ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"}`}>{v}×</button>
            ))}
          </div>
          <Toggle small label="Loop" on={loop} onChange={setLoop} />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="w-20 text-sm text-muted-foreground">Scenario</span>
          <select aria-label="Replay scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} className={`${selectCls} flex-1`}>
            {scenarios.map((x) => <option key={x.name} value={x.name}>{x.name.replace(/_/g, " ")}{typeof x.minutes === "number" ? ` · ${fmt(x.minutes, 0)} min` : ""}</option>)}
          </select>
          <Btn small onClick={() => sendCommand({ kind: "replay", scenario, speed, loop })}>▶ Replay</Btn>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="w-20 text-sm text-muted-foreground">Recording</span>
          <select aria-label="Replay recording" value={rec} onChange={(e) => setRecording(e.target.value)} disabled={!s.recordings.length} className={`${selectCls} flex-1`}>
            {!s.recordings.length && <option value="">no recordings</option>}
            {s.recordings.map((x) => <option key={x.name} value={x.name}>{x.name}{typeof x.bytes === "number" ? ` · ${fmt(x.bytes / 1024, 0)} KB` : ""}</option>)}
          </select>
          <Btn small disabled={!rec} onClick={() => rec && sendCommand({ kind: "replay_recording", name: rec, speed, loop })}>▶ Replay</Btn>
        </div>
      </div>
    </div>
  );
}

export function RoomBaselineCard() {
  const s = useValence();
  const b = s.baseline;
  const progress = baselineProgress(b);
  const capturing = b?.mode === "capturing";
  const warn = baselineWarning(s);
  const tone = b?.mode === "captured" ? "bg-ok" : capturing ? "bg-primary" : "bg-muted-foreground";
  const stat = (k: string, v: string, id: string) => (
    <div className="rounded-md bg-muted px-3 py-2">
      <div className="text-xs text-muted-foreground">{k}</div>
      <div className="font-mono text-xl font-bold" data-testid={id}>{v}</div>
    </div>
  );
  return (
    <div className="panel" data-testid="room-baseline">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="panel-title mb-0">Room baseline</span>
        <span data-testid="baseline-mode" className={`rounded px-2 py-0.5 font-mono text-xs font-bold uppercase text-primary-foreground ${tone}`}>{b?.mode ?? "unknown"}</span>
        {warn && <span data-testid="baseline-card-warning" className="rounded bg-warn px-2 py-0.5 font-mono text-xs font-bold text-primary-foreground">⚠ live · not captured</span>}
      </div>
      {!b ? <div className="text-muted-foreground">Baseline status unavailable</div> : (
        <>
          <div className="grid grid-cols-3 gap-2">
            {stat("dB median", fmt(b.db_median), "baseline-db")}
            {stat("dB IQR", fmt(b.db_iqr), "baseline-iqr")}
            {stat("lux median", fmt(b.lux_median, 0), "baseline-lux")}
          </div>
          <div className="mt-3 flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Samples</span>
            <div className="h-3 flex-1 overflow-hidden rounded bg-muted">
              <div className={`h-full transition-all ${b.mode === "captured" ? "bg-ok" : "bg-primary"}`} style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <span className="font-mono" data-testid="baseline-progress">{b.db_samples}/{b.target_samples}{b.lux_samples !== b.db_samples ? ` · lux ${b.lux_samples}` : ""}</span>
          </div>
          {capturing && <p className="mt-2 text-sm text-warn">Sit neutral and quiet at the desk until the mode reads “captured”.</p>}
          {b.mode === "auto" && <p className="mt-2 text-xs text-muted-foreground">Automatic: the first {b.target_samples} s of the session or recording{b.complete ? "" : " (filling)"}. Scenario replays use the simulator's own baseline.</p>}
        </>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Btn small tone="primary" disabled={capturing} onClick={() => sendCommand({ kind: "baseline", seconds: 90 })}>◎ Capture 90 s</Btn>
        <Btn small onClick={() => sendCommand({ kind: "baseline_reset" })}>Reset</Btn>
      </div>
    </div>
  );
}

// ---------- Camera ----------
export function CameraView() {
  const s = useValence();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cam, setCam] = useState<"pending" | "on" | "off">("pending");
  const replay = s.status?.mode === "replay";

  useEffect(() => {
    let stream: MediaStream | null = null;
    if (!navigator.mediaDevices?.getUserMedia) { setCam("off"); return; }
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((st) => { stream = st; if (videoRef.current) videoRef.current.srcObject = st; setCam("on"); })
      .catch(() => setCam("off"));
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  const showVideo = cam === "on" && !replay;
  return (
    <div className="panel p-3">
      <div className="mb-2 flex items-center justify-between px-2">
        <span className="panel-title mb-0">Room view</span>
        <span className="font-mono text-xs text-muted-foreground">
          {showVideo ? "local preview · never recorded or uploaded" : replay ? "replay mode · tracks only" : "no camera · tracks only"}
        </span>
      </div>
      <div className="relative aspect-video overflow-hidden rounded-md grid-bg">
        <video ref={videoRef} autoPlay muted playsInline className={`h-full w-full scale-x-[-1] object-cover ${showVideo ? "" : "hidden"}`} />
        {s.persons.map((p) => {
          const pred = s.predictions[p.id];
          const color = stateColor(pred?.label);
          const [b1, y1, b2, y2] = p.bbox;
          // The preview is shown mirrored (selfie view), so unmirrored box coordinates are flipped to match (docs/API.md 4.2).
          const [x1, x2] = s.mirrored ? [b1, b2] : [1 - b2, 1 - b1];
          return (
            <div key={p.id} className="absolute border-[3px] transition-all duration-700"
              style={{ left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`, borderColor: color }}>
              <div className="absolute -top-px left-0 -translate-y-full px-2 py-1 font-mono text-sm font-bold text-primary-foreground" style={{ background: color }}>
                {p.label}{pred ? ` · ${labelText(pred)} ${pct(pred.confidence)}` : ""}
              </div>
            </div>
          );
        })}
        {s.running !== "running" && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80 text-3xl font-bold">
            {s.running === "paused" ? "PAUSED · sensing suspended" : "STOPPED · privacy kill switch"}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------- Governance ----------
export function Governance() {
  const s = useValence();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }); }, [s.govLog.length]);
  const g = s.gov;
  const counters = [
    ["Frames purged", g?.frames_purged], ["Audio blocks purged", g?.audio_blocks_purged],
    ["Transcripts purged", g?.transcripts_purged], ["Violations", g?.violations],
  ] as const;
  return (
    <div className="panel flex flex-col">
      <div className="panel-title">Governance · purge log</div>
      <div className="mb-3 grid grid-cols-2 gap-2">
        {counters.map(([k, v]) => (
          <div key={k} className="rounded-md bg-muted px-3 py-2">
            <div className="text-xs text-muted-foreground">{k}</div>
            <div className={`font-mono text-2xl font-bold ${k === "Violations" ? (v ? "text-destructive" : "text-ok") : ""}`}>{v?.toLocaleString() ?? "—"}</div>
          </div>
        ))}
      </div>
      <div className="mb-2 font-mono text-xs text-muted-foreground">
        frames decoded: <span className="text-foreground">{g?.frames_decoded?.toLocaleString() ?? "—"}</span>
        {" · "}purge gap: <span className={(g?.purge_gap ?? 0) > 2 ? "text-destructive" : "text-ok"}>{g?.purge_gap ?? "—"}</span>
        {" · "}live raw buffers: <span className="text-ok">{g?.live_raw_buffers ?? "—"}</span>
      </div>
      <div ref={ref} className="h-52 overflow-y-auto rounded-md bg-terminal p-3 font-mono text-[13px] leading-relaxed text-ok">
        {s.govLog.length === 0 && <div className="text-muted-foreground">waiting for governance events…</div>}
        {s.govLog.map((m, i) => (
          <div key={i}><span className="text-muted-foreground">#{m.tick} </span>{m.message}</div>
        ))}
      </div>
    </div>
  );
}

// ---------- Status ----------
export function SystemStatus() {
  const st = useValence().status;
  const item = (k: string, v: React.ReactNode, ok?: boolean) => (
    <div className="flex items-center justify-between border-b py-1.5 last:border-0">
      <span className="text-muted-foreground">{k}</span>
      <span className={`font-mono font-semibold ${ok === undefined ? "" : ok ? "text-ok" : "text-destructive"}`}>{v}</span>
    </div>
  );
  return (
    <div className="panel">
      <div className="panel-title">System status</div>
      {!st ? <div className="text-muted-foreground">No status yet</div> : (
        <div className="text-base">
          {item("Camera", st.camera ? "ON" : "OFF", st.camera)}
          {item("Microphone", st.mic ? "ON" : "OFF", st.mic)}
          {item("Mode", String(st.mode ?? "—").toUpperCase())}
          {item("FPS", fmt(st.fps))}
          {item("Loop latency", `${fmt(st.loop_latency_ms, 0)} ms`, (st.loop_latency_ms ?? 0) < 250)}
          {item("Narrator", st.narrator, st.narrator === "ready")}
          {item("Network", st.offline ? "OFFLINE ✓" : "ONLINE", st.offline)}
          {item("Degraded", st.degraded?.length ? st.degraded.join(", ") : "none", !st.degraded?.length)}
          {item("Purge violation", st.purge_violation ? "ALARM" : "none", !st.purge_violation)}
          {item("Uptime", `${Math.floor(st.uptime_s / 60)}m ${Math.floor(st.uptime_s % 60)}s`)}
        </div>
      )}
    </div>
  );
}

// ---------- Person card ----------
export function PersonCard({ p }: { p: PredictionMsg }) {
  const s = useValence();
  const color = stateColor(p.label);
  const probs = p.probabilities ?? {};
  // The active state set (hello.states) in order, then any extra state the backend sends.
  const active = s.states.map((x) => x.id);
  const keys: string[] = [...active, ...Object.keys(probs).filter((k) => !active.includes(k))];
  const faceLabel = p.face_only?.label ?? null;
  const ex = shownExplanation(p, s.llm[p.person_id]);
  return (
    <div className="panel border-l-8" style={{ borderLeftColor: color }}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xl text-muted-foreground">{p.person_id === 0 ? "Room (aggregate)" : `Person #${p.person_id}`}</div>
        <div className="font-mono text-xs text-muted-foreground">{p.event_id}{p.warming_up ? " · warming up" : ""}</div>
      </div>
      <div className="mt-1 flex flex-wrap items-baseline gap-3">
        <div className="text-4xl font-bold" style={{ color }}>{labelText(p)}</div>
        <div className="font-mono text-4xl font-bold">{pct(p.confidence)}</div>
      </div>

      <div className="mt-4 space-y-2">
        {keys.map((k) => (
          <div key={k} className="grid grid-cols-[14rem_1fr_3.5rem] items-center gap-3 text-base">
            <span className={k === p.label ? "font-semibold" : "text-muted-foreground"}>{stateName(k)}</span>
            <div className="h-4 overflow-hidden rounded-sm bg-muted">
              <div className="h-full transition-all duration-700" style={{ width: pct(probs[k] ?? 0), background: stateColor(k, "var(--muted-foreground)") }} />
            </div>
            <span className="text-right font-mono">{pct(probs[k] ?? 0)}</span>
          </div>
        ))}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-md bg-muted p-3">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Face-only baseline</div>
          {faceLabel ? (
            <>
              <div className="text-lg font-semibold" style={{ color: stateColor(faceLabel) }}>{stateName(faceLabel)}</div>
              <div className="font-mono">{pct(p.face_only.confidence)}</div>
            </>
          ) : (
            <>
              <div className="text-lg font-semibold text-muted-foreground">no face visible</div>
              <div className="font-mono text-muted-foreground">—</div>
            </>
          )}
        </div>
        <div className="rounded-md bg-muted p-3">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Context model</div>
          <div className="text-lg font-semibold" style={{ color }}>{labelText(p)}</div>
          <div className="font-mono">{pct(p.confidence)}</div>
        </div>
      </div>
      {p.mask_override && (
        <div className="mt-3 rounded-md border-2 border-state-frustrated bg-state-frustrated/15 px-4 py-3 text-xl font-bold text-state-frustrated">
          ⚠ MASK OVERRIDE — Context corrected the facial mask
        </div>
      )}

      <div className="mt-5">
        <div className="mb-1 flex items-center gap-2">
          <span className="panel-title mb-0">Audit explanation</span>
          <span className={`rounded px-2 py-0.5 font-mono text-xs font-bold ${ex.source === "template" ? "bg-muted text-muted-foreground" : "bg-primary text-primary-foreground"}`}>
            {ex.source}{ex.latency_ms ? ` · ${ex.latency_ms} ms` : ""}
          </span>
          {ex.asOf && <span className="font-mono text-xs text-muted-foreground">as of {ex.asOf}</span>}
        </div>
        <p className="text-lg leading-snug">{ex.text}</p>
      </div>

      <div className="mt-5">
        <div className="panel-title">Top drivers by layer</div>
        <div className="grid gap-3 md:grid-cols-2">
          {LAYERS.map((name) => {
            const l = (Array.isArray(p.layers) ? p.layers : []).find((x) => x.name === name);
            return (
              <div key={name} className="rounded-md border p-3">
                <div className="mb-1 flex justify-between">
                  <span className="font-semibold">{name}</span>
                  <span className={`font-mono font-bold ${(l?.contribution ?? 0) >= 0 ? "text-ok" : "text-destructive"}`}>
                    {l ? `${l.contribution >= 0 ? "+" : ""}${fmt(l.contribution, 2)}` : "—"}
                  </span>
                </div>
                {l?.drivers?.map((d) => (
                  <div key={d.feature} className="flex items-start justify-between gap-2 text-sm">
                    <span className="text-muted-foreground">{d.shap >= 0 ? "▲ supports" : "▼ opposes"} · {d.display}</span>
                    <span className={`font-mono ${d.shap >= 0 ? "text-ok" : "text-destructive"}`}>{d.shap >= 0 ? "+" : ""}{fmt(d.shap, 2)}</span>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------- State vectors ----------
export function StateVectors() {
  const s = useValence();
  const ids = Object.keys(s.vectors).map(Number).sort();
  const rows: [string, (v: StateVectorMsg) => string][] = [
    ["Noise", (v) => `${fmt(v.ambient_db)} dB`],
    ["Light", (v) => `${fmt(v.light_lux, 0)} lux`],
    ["Hours since break", (v) => `${fmt(v.hours_since_break)} h`],
    ["Time of day", (v) => v.time_of_day ?? "—"],
    ["Clenched fists (5 min)", (v) => String(v.clenched_fists_5m ?? "—")],
    ["Posture slump", (v) => pct(v.posture_slump)],
    ["Head in hands", (v) => pct(v.head_in_hands)],
    ["Social withdrawal", (v) => (v.social_withdrawal ? "yes" : "no")],
    ["Compliments (15 min)", (v) => String(v.compliments_15m ?? "—")],
    ["Laughter (15 min)", (v) => String(v.laughter_15m ?? "—")],
    ["Face visible", (v) => (v.face_visible ? "yes" : "NO")],
    ["Facial valence", (v) => (v.face_visible && v.facial_valence != null ? `${v.facial_valence >= 0 ? "+" : ""}${fmt(v.facial_valence, 2)}` : "n/a")],
  ];
  return (
    <div className="panel">
      <div className="panel-title">State vector</div>
      <table className="w-full text-base">
        <thead><tr className="text-left text-muted-foreground"><th className="py-1 font-normal">Signal</th>{ids.map((i) => <th key={i} className="text-right font-mono">P#{i}</th>)}</tr></thead>
        <tbody>
          {rows.map(([k, f]) => (
            <tr key={k} className="border-t">
              <td className="py-1.5 text-muted-foreground">{k}</td>
              {ids.map((i) => { const v = s.vectors[i]; return <td key={i} className="text-right font-mono font-semibold">{v ? f(v) : "—"}</td>; })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------- Timeline ----------
export function Timeline() {
  const s = useValence();
  const ids = [...new Set(s.history.map((h) => h.person_id))].sort();
  const [sel, setPid] = useState(1);
  const pid = ids.includes(sel) || !ids.length ? sel : (ids[0] ?? sel); // e.g. room-only mode streams person 0
  const pts = s.history.filter((h) => h.person_id === pid);
  const tMax = pts[pts.length - 1]?.t ?? 0;
  const tMin = tMax - 900;
  const W = 1000, H = 160;
  const x = (t: number) => ((t - tMin) / 900) * W;
  const series: { name: string; key: "ambient_db" | "light_lux" | "posture_slump" | "hours_since_break"; norm: (v: number) => number; color: string }[] = [
    { name: "Noise dB", key: "ambient_db", norm: (v) => v / 100, color: "var(--state-frustrated)" },
    { name: "Light lux", key: "light_lux", norm: (v) => Math.min(1, v / 800), color: "var(--state-focused)" },
    { name: "Slump", key: "posture_slump", norm: (v) => v, color: "var(--state-exhausted)" },
    { name: "Hours since break", key: "hours_since_break", norm: (v) => Math.min(1, v / 8), color: "var(--primary)" },
  ];
  const labels = pts.filter((p) => p.label);
  const fill = s.vectors[pid]?.window_fill_min ?? [...pts].reverse().find((p) => p.window_fill_min != null)?.window_fill_min ?? 0;
  return (
    <div className="panel">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="panel-title mb-0">15-minute timeline</span>
        {ids.map((i) => (
          <button key={i} onClick={() => setPid(i)} className={`rounded px-2 py-1 font-mono text-sm ${pid === i ? "bg-primary text-primary-foreground" : "bg-muted"}`}>Person #{i}</button>
        ))}
        <div className="ml-auto flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Window</span>
          <div className="h-3 w-40 overflow-hidden rounded bg-muted"><div className="h-full bg-primary" style={{ width: `${(fill / 15) * 100}%` }} /></div>
          <span className="font-mono">{fmt(fill)}/15 min {fill < 15 ? "· warming up" : "· full"}</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H + 40}`} className="w-full" preserveAspectRatio="none">
        {series.map((se) => {
          const d = pts.filter((p) => p[se.key] != null).map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${(H - se.norm(p[se.key] as number) * H).toFixed(1)}`).join("");
          return <path key={se.key} d={d} fill="none" stroke={se.color} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />;
        })}
        {labels.map((p, i) => {
          const next = labels[i + 1]?.t ?? tMax;
          return <rect key={i} x={x(p.t)} y={H + 12} width={Math.max(1, x(next) - x(p.t))} height={24} fill={stateColor(p.label, "var(--muted-foreground)")} />;
        })}
      </svg>
      <div className="mt-2 flex flex-wrap gap-4 text-sm">
        {series.map((se) => <span key={se.key} className="flex items-center gap-1.5"><span className="h-1 w-5" style={{ background: se.color }} />{se.name}</span>)}
        <span className="text-muted-foreground">| bar = predicted state:</span>
        {s.states.map((k) => <span key={k.id} className="flex items-center gap-1.5"><span className="h-3 w-3" style={{ background: stateColor(k.id, "var(--muted-foreground)") }} />{k.display}</span>)}
      </div>
      <div className="mt-1 flex justify-between font-mono text-xs text-muted-foreground"><span>−15 min</span><span>now</span></div>
    </div>
  );
}

// ---------- Retention ----------
const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const cell = (v: unknown) => (v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v));
const HEAD: Record<string, string> = { what: "What", where: "Where", how_long: "How long" };

/** GET /api/retention: scalars as rows, `held` (rows of objects) as a table, `never_kept` (strings) as a list. */
export function Retention() {
  const r = useValence().retention;
  const entries = r ? Object.entries(r) : [];
  const scalars = entries.filter(([, v]) => !Array.isArray(v) && !isRecord(v));
  const lists = entries.filter(([, v]) => Array.isArray(v) || isRecord(v));
  return (
    <div className="panel" data-testid="retention">
      <div className="panel-title">Data retention</div>
      {!r ? <div className="text-muted-foreground">Unavailable</div> : (
        <div className="space-y-3 text-sm">
          {scalars.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-3 border-b pb-1.5">
              <span className="text-muted-foreground">{k.replace(/_/g, " ")}</span>
              <span className={`text-right font-mono text-base font-semibold ${v === 0 ? "text-ok" : ""}`}>{cell(v)}</span>
            </div>
          ))}
          {lists.map(([k, v]) => {
            const rows = Array.isArray(v) ? v : [v];
            const objs = rows.filter(isRecord);
            if (objs.length === rows.length && objs.length > 0) {
              const cols = [...new Set(objs.flatMap((o) => Object.keys(o)))];
              return (
                <div key={k}>
                  <div className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">{k.replace(/_/g, " ")}</div>
                  <table className="w-full" data-testid={`retention-${k}`}>
                    <thead><tr className="text-left text-xs text-muted-foreground">{cols.map((c) => <th key={c} className="pb-1 pr-2 font-normal">{HEAD[c] ?? c.replace(/_/g, " ")}</th>)}</tr></thead>
                    <tbody>{objs.map((o, i) => (
                      <tr key={i} className="border-t align-top">{cols.map((c) => <td key={c} className={`py-1 pr-2 ${c === "what" ? "" : "font-mono text-xs"}`}>{cell(o[c])}</td>)}</tr>
                    ))}</tbody>
                  </table>
                </div>
              );
            }
            return (
              <div key={k}>
                <div className="mb-1 text-xs uppercase tracking-wider text-muted-foreground">{k.replace(/_/g, " ")}</div>
                <ul className="space-y-0.5" data-testid={`retention-${k}`}>
                  {rows.map((x, i) => <li key={i} className="flex gap-2"><span className="text-ok">✓</span><span>{cell(x)}</span></li>)}
                </ul>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------- Evaluation ----------
function BarPair({ data, xKey }: { data: { label: string; context: number; face_only: number }[]; xKey: string }) {
  return (
    <div className="space-y-2">
      {data.map((d) => (
        <div key={d.label}>
          <div className="text-sm text-muted-foreground">{xKey}{d.label}</div>
          {(["context", "face_only"] as const).map((k) => (
            <div key={k} className="flex items-center gap-2">
              <div className="h-3 flex-1 rounded-sm bg-muted"><div className="h-full rounded-sm" style={{ width: pct(d[k]), background: k === "context" ? "var(--primary)" : "var(--muted-foreground)" }} /></div>
              <span className="w-12 text-right font-mono text-sm">{pct(d[k])}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function AccuracyChart({ d }: { d: NonNullable<EvalSummary["accuracy_vs_history"]> }) {
  const W = 300, H = 150, maxM = Math.max(...d.map((p) => p.minutes), 1);
  const path = (k: "context" | "face_only") => d.map((p, i) => `${i ? "L" : "M"}${(p.minutes / maxM) * W},${H - p[k] * H}`).join("");
  return (
    <svg viewBox={`-30 -10 ${W + 40} ${H + 30}`} className="w-full">
      {[0, 0.5, 1].map((v) => <g key={v}><line x1={0} x2={W} y1={H - v * H} y2={H - v * H} stroke="var(--border)" /><text x={-6} y={H - v * H + 4} textAnchor="end" fontSize={10} fill="var(--muted-foreground)">{v * 100}%</text></g>)}
      <path d={path("face_only")} fill="none" stroke="var(--muted-foreground)" strokeWidth={2} strokeDasharray="4 4" />
      <path d={path("context")} fill="none" stroke="var(--primary)" strokeWidth={3} />
      {d.map((p) => <text key={p.minutes} x={(p.minutes / maxM) * W} y={H + 16} textAnchor="middle" fontSize={10} fill="var(--muted-foreground)">{p.minutes}m</text>)}
    </svg>
  );
}

function Confusion({ c }: { c: NonNullable<EvalSummary["confusion"]>[number] }) {
  const max = Math.max(...c.matrix.flat());
  const short = { Focused: "Foc", Frustrated_Masked: "Fru", Exhausted: "Exh", Collaborative_Happy: "Hap", Overwhelmed: "Ovw", Tense: "Ten" } as Record<string, string>;
  return (
    <div>
      <div className="mb-1 text-sm font-semibold">{c.name}</div>
      <div className="grid gap-0.5 font-mono text-xs" style={{ gridTemplateColumns: `2.5rem repeat(${c.labels.length}, 1fr)` }}>
        <span />{c.labels.map((l) => <span key={l} className="text-center text-muted-foreground">{short[l] ?? l}</span>)}
        {c.matrix.map((row, i) => [
          <span key={`l${i}`} className="text-muted-foreground">{short[c.labels[i] ?? ""] ?? c.labels[i]}</span>,
          ...row.map((v, j) => (
            <span key={`${i}-${j}`} className="py-1.5 text-center font-bold" style={{ background: `color-mix(in oklch, var(--primary) ${(v / max) * 85}%, var(--muted))`, color: v / max > 0.5 ? "var(--primary-foreground)" : "var(--foreground)" }}>{v}</span>
          )),
        ])}
      </div>
    </div>
  );
}

function Calibration({ d }: { d: NonNullable<EvalSummary["calibration"]> }) {
  const S = 150;
  return (
    <svg viewBox={`-10 -10 ${S + 20} ${S + 20}`} className="w-full max-w-[220px]">
      <rect x={0} y={0} width={S} height={S} fill="none" stroke="var(--border)" />
      <line x1={0} y1={S} x2={S} y2={0} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
      <path d={d.map((p, i) => `${i ? "L" : "M"}${p.predicted * S},${S - p.observed * S}`).join("")} fill="none" stroke="var(--primary)" strokeWidth={3} />
      {d.map((p, i) => <circle key={i} cx={p.predicted * S} cy={S - p.observed * S} r={4} fill="var(--primary)" />)}
    </svg>
  );
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const MODEL_NOTE: Record<string, string> = { m0: "face-only baseline", m2: "context model" };

/** Metrics table rows from either the mock rows or reports/metrics.json (`metrics.models`, served by /api/eval/summary). */
function metricRows(m: EvalSummary["metrics"]): { accLabel: string; rows: { model: string; acc: number | null; f1: number | null; ece: number | null }[] } {
  if (Array.isArray(m)) return { accLabel: "Acc", rows: m.map((r) => ({ model: String(r.model), acc: num(r.accuracy), f1: num(r.macro_f1), ece: num(r.ece) })) };
  const models = m && typeof m === "object" && m.models && typeof m.models === "object" ? m.models : {};
  const eceM2 = num(m?.calibration_m2?.ece_after);
  return {
    accLabel: "Bal acc",
    rows: Object.entries(models).map(([id, v]) => ({
      model: MODEL_NOTE[id] ? `${id} · ${MODEL_NOTE[id]}` : id,
      acc: num(v?.balanced_accuracy), f1: num(v?.macro_f1), ece: id === "m2" ? eceM2 : null,
    })),
  };
}

export function Evaluation() {
  const s = useValence();
  const e = s.evalSummary;
  const { accLabel, rows } = metricRows(e?.metrics);
  const disclosure = e && !Array.isArray(e.metrics) && typeof e.metrics?.disclosure === "string" ? e.metrics.disclosure : null;
  const figures = Array.isArray(e?.figures) ? e.figures : [];
  return (
    <div className="panel">
      <div className="mb-3 flex items-center gap-3">
        <span className="panel-title mb-0">Evaluation</span>
        <span className="flex items-center gap-1.5 text-sm"><span className="h-1 w-5 bg-primary" />Context model</span>
        <span className="flex items-center gap-1.5 text-sm"><span className="h-1 w-5 bg-muted-foreground" />Face-only baseline</span>
        {s.source === "mock" && <span className="ml-auto rounded bg-warn px-2 py-0.5 font-mono text-xs font-bold text-primary-foreground">PLACEHOLDER DATA</span>}
      </div>
      {!e ? <div className="text-muted-foreground">Evaluation summary unavailable</div>
        : e.available === false ? <div className="text-muted-foreground">No evaluation results yet</div> : (
        <div className="grid gap-6 lg:grid-cols-4">
          {Array.isArray(e.accuracy_vs_history) && <div><div className="mb-1 text-sm font-semibold">Accuracy vs history window</div><AccuracyChart d={e.accuracy_vs_history} /></div>}
          {Array.isArray(e.robustness) && <div><div className="mb-1 text-sm font-semibold">Robustness</div><BarPair xKey="" data={e.robustness.map((r) => ({ label: r.condition, context: r.context, face_only: r.face_only }))} /></div>}
          {Array.isArray(e.calibration) && <div><div className="mb-1 text-sm font-semibold">Calibration (context)</div><Calibration d={e.calibration} /></div>}
          {rows.length > 0 && (
            <div>
              <div className="mb-1 text-sm font-semibold">Metrics</div>
              <table className="w-full text-sm">
                <thead><tr className="text-left text-muted-foreground"><th className="font-normal">Model</th><th className="text-right font-normal">{accLabel}</th><th className="text-right font-normal">F1</th><th className="text-right font-normal">ECE</th></tr></thead>
                <tbody>{rows.map((m) => (
                  <tr key={m.model} className="border-t"><td className="py-1">{m.model}</td><td className="text-right font-mono">{pct(m.acc)}</td><td className="text-right font-mono">{fmt(m.f1, 2)}</td><td className="text-right font-mono">{fmt(m.ece, 2)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
          {disclosure && <p className="text-sm text-muted-foreground lg:col-span-3">{disclosure}</p>}
          {Array.isArray(e.confusion) && e.confusion.map((c) => <div key={c.name} className="lg:col-span-2"><Confusion c={c} /></div>)}
        </div>
      )}
      {s.source === "live" && figures.length > 0 && (
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {figures.map((f) => (
            <figure key={f} className="rounded-md border bg-muted p-2">
              <img src={figureUrl(f)} alt={f.replace(/_/g, " ")} className="w-full rounded" onError={(ev) => ((ev.currentTarget.style.display = "none"))} />
              <figcaption className="mt-1 font-mono text-xs text-muted-foreground">{f}.png</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
