"""FastAPI app: REST under /api, event stream on /ws/events. Contract: docs/API.md.

Run (replay demo):  VALENCE_MODE=replay uvicorn valence.api.app:app
Run (live, on Mac): VALENCE_MODE=live uvicorn valence.api.app:app   (never with --reload: it opens the camera twice)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from valence.config import Settings
from valence.contracts import CONTRACT_VERSION, STATE_DISPLAY, STATE_SETS, STATE_TIER, Heartbeat, Hello, StateInfo, Status
from valence.explain.narrator_client import NarratorClient
from valence.governance import logfilter
from valence.governance.purge import gate
from valence.models.predictor import Predictor, StubPredictor
from valence.runtime.bus import EventBus
from valence.runtime.engine import Runtime
from valence.sim import scenarios

log = logging.getLogger("valence.api")
REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
LABELS = REPO / "var" / "labels"
OVERLAYS = {"face_occlusion", "low_light"}


def load_predictor(settings: Settings) -> Predictor:
    """Trained M2 if artifacts exist, otherwise the transparent rule stub."""
    try:
        from valence.models.served import load_served  # created with the trained model

        predictor = load_served(Path(settings.artifacts_dir), settings.state_set)
        if predictor is not None:
            return predictor
    except ImportError:
        pass
    return StubPredictor(STATE_SETS[settings.state_set])


class App:
    """Process-wide state (one runtime per server)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bus = EventBus()
        self.predictor = load_predictor(settings)
        self.narrator = NarratorClient(settings.narrator_url, self.bus.publish, settings.narrator_deadline_s)
        self.runtime = Runtime(settings, self.bus, self.predictor, narrator=self.narrator)
        self.edge_stop = None
        self.edge_error: str | None = None

    def start_edge(self, obscure_face: bool = False) -> None:
        if self.edge_stop is not None:
            return
        try:
            from valence.edge.run import start as edge_start  # LOCAL session's module (Mac only)

            self.edge_stop = edge_start(self.runtime.push_obs, video=True, audio=True, obscure_face=obscure_face)
            self.edge_error = None
        except Exception as exc:  # missing module, camera denied, ...
            self.edge_error = f"{type(exc).__name__}: {exc}"
            log.error("edge failed to start: %s", self.edge_error)

    def stop_edge(self) -> None:
        if self.edge_stop is not None:
            try:
                self.edge_stop()
            finally:
                self.edge_stop = None
        self.runtime.camera = self.runtime.mic = False


STATE: App | None = None


def state() -> App:
    if STATE is None:
        raise HTTPException(503, "runtime not started")
    return STATE


def hello(a: App) -> Hello:
    agg = a.runtime.aggregator
    return Hello(
        mode=a.runtime.mode if a.runtime.mode in ("live", "replay", "hybrid") else "replay",
        room_only=a.runtime.room_only,
        states=[StateInfo(id=s, display=STATE_DISPLAY[s], tier=STATE_TIER[s]) for s in a.predictor.states],
        feature_version=getattr(a.predictor, "model_id", "unknown"),
        frame_w=agg.frame_w, frame_h=agg.frame_h, mirrored=agg.mirrored,
    )


def status(a: App) -> Status:
    rt = a.runtime
    now = time.monotonic()
    camera = rt.mode == "live" and now - rt.aggregator.last_video_t < 3.0
    mic = rt.mode == "live" and now - rt.aggregator.last_audio_t < 3.0
    degraded = []
    if rt.mode == "live" and not camera:
        degraded.append("camera")
    if rt.mode == "live" and not mic:
        degraded.append("mic")
    if not a.narrator.healthy:
        degraded.append("narrator")
    if getattr(a.predictor, "model_id", "") == "stub-rules":
        degraded.append("predictor:stub")
    if a.edge_error:
        degraded.append("edge")
    stats = gate.stats()
    return Status(
        mode=rt.mode if rt.mode in ("live", "replay", "hybrid") else "replay", camera=camera, mic=mic,
        fps=round(rt.fps, 1), loop_latency_ms=round(rt.loop_latency_ms, 1),
        narrator="ready" if a.narrator.healthy else "fallback", offline=all(os.environ.get(k) for k in ("HF_HUB_OFFLINE", "YOLO_OFFLINE")),
        uptime_s=round(time.time() - rt.started, 1), degraded=degraded, purge_violation=stats.violations > 0,
    )


async def _status_loop(a: App) -> None:
    n = 0
    while True:
        await asyncio.sleep(1.0)
        n += 1
        if n % 10 == 1:
            await asyncio.to_thread(a.narrator.check_health)
        a.bus.publish(status(a))
        if n % 5 == 0:
            a.bus.publish(Heartbeat(t=time.time()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    logfilter.install()
    settings = Settings.from_env()
    a = App(settings)
    STATE = a
    a.bus.bind(asyncio.get_running_loop())
    a.narrator.start()
    if settings.mode == "live":
        a.runtime.use_live()
        a.start_edge()
    else:
        name = os.environ.get("VALENCE_SCENARIO", "full_demo")
        speed = float(os.environ.get("VALENCE_SPEED", "1"))
        a.runtime.use_scenario(scenarios.render(name, state_set=settings.state_set), name, speed=speed, loop=True)
    a.runtime.start()
    task = asyncio.create_task(_status_loop(a))
    try:
        yield
    finally:
        task.cancel()
        a.stop_edge()
        a.runtime.close()
        a.narrator.stop()
        STATE = None


app = FastAPI(title="Valence", version=CONTRACT_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-z0-9-]+\.(lovable\.app|lovableproject\.com))$",
    allow_methods=["*"], allow_headers=["*"],
)


# --- WebSocket ---------------------------------------------------------------------------------------
@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    a = state()
    await ws.accept()
    q = a.bus.subscribe()
    try:
        await ws.send_text(hello(a).model_dump_json())
        while True:
            await ws.send_text(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        a.bus.unsubscribe(q)


# --- read endpoints -------------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    a = state()
    return {"ok": True, "contract_version": CONTRACT_VERSION, "mode": a.runtime.mode,
            "model": getattr(a.predictor, "model_id", "unknown"), "edge_error": a.edge_error}


@app.get("/api/states")
def get_states() -> list[dict[str, str]]:
    return [s.model_dump() for s in hello(state()).states]


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    a = state()
    return {"people": a.runtime.latest, "status": status(a).model_dump()}


@app.get("/api/history")
def get_history(minutes: float = 15.0) -> list[dict[str, Any]]:
    return state().runtime.history_since(min(minutes, 60.0))


@app.get("/api/governance/stats")
def governance_stats() -> dict[str, Any]:
    a = state()
    s = gate.stats()
    return {
        "frames_decoded": a.runtime.aggregator.frames_decoded, "frames_purged": s.purged.get("video", 0),
        "audio_blocks_purged": s.purged.get("audio", 0), "freed": s.freed, "unzeroable": s.unzeroable,
        "transcripts_purged": s.transcripts_purged, "violations": s.violations,
        "live_raw_buffers": s.live_raw_buffers, "retention_s": a.settings.audit_ttl_s,
        "near_field_gated_out": a.runtime.aggregator.gated_out, "audit_hash": a.runtime.audit.last_hash,
    }


@app.get("/api/governance/audit")
def governance_audit(since: int = 0) -> list[dict[str, Any]]:
    return state().runtime.audit.entries(since_seq=since)


@app.get("/api/governance/verify")
def governance_verify() -> dict[str, Any]:
    ok, n = state().runtime.audit.verify()
    return {"ok": ok, "entries": n}


@app.get("/api/retention")
def retention() -> dict[str, Any]:
    a = state()
    return {
        "raw_media_stored": 0,
        "held": [
            {"what": "rolling state vector (numeric TickRecords)", "where": "RAM", "how_long": "15 minutes"},
            {"what": "recent messages for /api/history", "where": "RAM", "how_long": "about 20 minutes"},
            {"what": "audit log: labels, confidences, SHAP layer sums, operator actions", "where":
             str(a.runtime.audit.path.relative_to(REPO)) if a.runtime.audit.path.is_relative_to(REPO) else "var/audit",
             "how_long": f"deleted at session end; TTL {a.settings.audit_ttl_s} s after a crash"},
        ],
        "never_kept": ["video frames (zeroed and released each frame)", "audio blocks (zeroed each block)",
                       "transcripts (Vosk grammar: only lexicon hits exist)", "pose keypoints (stay inside perception)",
                       "face images or embeddings"],
    }


@app.get("/api/eval/summary")
def eval_summary() -> dict[str, Any]:
    path = REPORTS / "metrics.json"
    if not path.exists():
        return {"available": False}
    figs = sorted(p.stem for p in (REPORTS / "figs").glob("*.png"))
    return {"available": True, "metrics": json.loads(path.read_text()), "figures": figs}


@app.get("/api/eval/figures/{name}.png")
def eval_figure(name: str) -> FileResponse:
    path = (REPORTS / "figs" / f"{name}.png").resolve()
    if path.parent != (REPORTS / "figs").resolve() or not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/scenarios")
def get_scenarios() -> list[dict[str, Any]]:
    return scenarios.list_scenarios()


@app.get("/api/models")
def get_models() -> dict[str, Any]:
    a = state()
    return {"served": getattr(a.predictor, "model_id", "unknown"), "states": list(a.predictor.states),
            "meta": getattr(a.predictor, "meta", {})}


@app.get("/api/narrator/health")
def narrator_health() -> dict[str, Any]:
    a = state()
    return {"healthy": a.narrator.check_health(), "url": a.settings.narrator_url, "failures": a.narrator.failures}


@app.get("/api/explain/{event_id}/counterfactual")
def counterfactual(event_id: str, feature: str, value: float) -> dict[str, Any]:
    a = state()
    fn = getattr(a.predictor, "counterfactual", None)
    if fn is None:
        raise HTTPException(501, "counterfactuals need the trained model (served predictor)")
    pid = int(event_id.split("-")[0].lstrip("p"))
    person = a.runtime.window.snapshot(pid)
    if not person:
        raise HTTPException(404, "person not in the window")
    return fn(person, a.runtime.window.room(), feature, value)


# --- control endpoints ------------------------------------------------------------------------------
@app.post("/api/control/{action}")
def control(action: str) -> dict[str, Any]:
    a = state()
    rt = a.runtime
    if action == "start":
        rt.paused = False
        if rt.mode == "live":
            a.start_edge()
        rt.start()
    elif action == "pause":  # privacy kill switch: stop capture and processing
        rt.paused = True
        a.stop_edge()
    elif action == "stop":
        rt.paused = True
        a.stop_edge()
        rt.reset_people(keep_room=False)
    else:
        raise HTTPException(404, "action must be start, pause or stop")
    rt._audit("operator", {"action": f"control_{action}"})
    return {"ok": True, "paused": rt.paused}


@app.post("/api/events/break")
def log_break(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    pid = body.get("person_id")
    a.runtime.aggregator.log_break(pid)
    a.runtime._audit("operator", {"action": "break", "person_id": pid})
    return {"ok": True}


@app.post("/api/replay/{scenario}")
def replay(scenario: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    if scenario not in {s["name"] for s in scenarios.list_scenarios()}:
        raise HTTPException(404, f"unknown scenario {scenario}")
    a.stop_edge()
    speed = float(body.get("speed", 1.0))
    a.runtime.use_scenario(scenarios.render(scenario, state_set=a.settings.state_set), scenario,
                           speed=speed, loop=bool(body.get("loop", False)))
    a.runtime.paused = False
    a.runtime.stop()
    a.runtime.start()
    return {"ok": True, "scenario": scenario, "speed": speed}


@app.get("/api/recordings")
def get_recordings() -> list[dict[str, Any]]:
    from valence.runtime.recording import list_recordings

    return list_recordings()


@app.post("/api/replay_recording/{name}")
def replay_recording(name: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Replay a features-only recording from data/real/ through the live pipeline (camera/mic fallback)."""
    from valence.runtime.recording import RECORDINGS

    a = state()
    path = (RECORDINGS / f"{name}.jsonl").resolve()
    if path.parent != RECORDINGS.resolve() or not path.exists():
        raise HTTPException(404, f"unknown recording {name}")
    a.stop_edge()
    a.runtime.use_recording(path, speed=float(body.get("speed", 1.0)), loop=bool(body.get("loop", True)))
    a.runtime.paused = False
    a.runtime.stop()
    a.runtime.start()
    return {"ok": True, "recording": name}


@app.post("/api/live")
def go_live() -> dict[str, Any]:
    a = state()
    a.runtime.use_live()
    a.runtime.reset_people(keep_room=False)
    a.start_edge()
    a.runtime.paused = False
    a.runtime.stop()
    a.runtime.start()
    return {"ok": a.edge_error is None, "edge_error": a.edge_error}


@app.post("/api/sim/{overlay}")
def sim_overlay(overlay: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    if overlay == "event":
        a.runtime.aggregator.inject(laughter=int(body.get("laughter", 0)), compliments=int(body.get("compliments", 0)),
                                    fists={int(k): int(v) for k, v in body.get("fists", {}).items()})
        a.runtime._audit("operator", {"action": "sim_event", "laughter": int(body.get("laughter", 0)),
                                      "compliments": int(body.get("compliments", 0))})
        return {"ok": True}
    if overlay == "lux":
        lux = body.get("lux")
        a.runtime.aggregator.lux_override = None if lux is None else float(lux)
        a.runtime._audit("operator", {"action": "sim_lux", "lux": lux})
        return {"ok": True, "lux_override": lux}
    if overlay not in OVERLAYS:
        raise HTTPException(404, f"overlay must be one of {sorted(OVERLAYS | {'event', 'lux'})}")
    a.runtime.set_overlay(overlay, bool(body.get("enabled", True)))
    return {"ok": True, "overlays": a.runtime.overlays}


@app.post("/api/demo/seed")
def demo_seed(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    name = body.get("scenario", "full_demo")
    minutes = float(body.get("minutes", 12))
    hours = body.get("hours_since_break")
    df = scenarios.render(name, state_set=a.settings.state_set)
    n = a.runtime.seed(df, name, minutes, None if hours is None else float(hours))
    return {"ok": True, "room_ticks_seeded": n, "simulated_fields": ["room_history"] + (["hours_since_break"] if hours is not None else [])}


@app.post("/api/demo/reset")
def demo_reset() -> dict[str, Any]:
    state().runtime.reset_people(keep_room=True)
    return {"ok": True}


@app.post("/api/calibrate/room")
def calibrate_room(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    out = a.runtime.aggregator.calibrate(body.get("spl_reference"), body.get("lux_reference"))
    a.runtime._audit("operator", {"action": "calibrate_room", **{k: round(v, 3) for k, v in out.items()}})
    return {"ok": True, **out}


@app.post("/api/mode")
def set_mode(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    a = state()
    a.runtime.room_only = bool(body.get("room_only", False))
    a.runtime._audit("operator", {"action": "mode", "room_only": a.runtime.room_only})
    return {"ok": True, "room_only": a.runtime.room_only}


@app.post("/api/label")
def self_report(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Self-report label for the n=1 sanity set: the label plus the latest numeric tick, never media."""
    a = state()
    label = body.get("state")
    if label not in STATE_DISPLAY:
        raise HTTPException(422, f"state must be one of {list(STATE_DISPLAY)}")
    pid = int(body.get("person_id", 1))
    person = a.runtime.window.snapshot(pid)
    LABELS.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "label": label, "person_id": pid, "tick": person[-1].to_dict() if person else None}
    with (LABELS / f"{a.runtime.session_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    a.runtime._audit("operator", {"action": "self_report", "label": label, "person_id": pid})
    return {"ok": True}
