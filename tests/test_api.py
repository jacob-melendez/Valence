import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VALENCE_MODE", "replay")
    monkeypatch.setenv("VALENCE_SPEED", "100")
    monkeypatch.setenv("VALENCE_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("VALENCE_NARRATOR_URL", "http://127.0.0.1:9")  # nothing listening: template fallback
    from valence.api.app import app

    with TestClient(app) as c:
        yield c


def test_rest_endpoints(client):
    assert client.get("/api/health").json()["ok"]
    states = client.get("/api/states").json()
    assert [s["id"] for s in states] == ["Focused", "Frustrated_Masked", "Exhausted", "Collaborative_Happy"]
    assert {s["name"] for s in client.get("/api/scenarios").json()} >= {"full_demo", "context_blindness"}
    assert client.get("/api/governance/stats").json()["violations"] == 0
    assert client.get("/api/retention").json()["raw_media_stored"] == 0
    assert client.get("/api/eval/summary").status_code == 200
    assert client.post("/api/sim/face_occlusion", json={"enabled": True}).json()["overlays"]["face_occlusion"]
    assert client.post("/api/mode", json={"room_only": False}).json()["ok"]
    assert client.post("/api/demo/reset").json()["ok"]
    assert client.post("/api/replay/nope").status_code == 404
    assert client.post("/api/replay/focused", json={"speed": 0}).status_code == 422
    assert client.post("/api/replay/focused", json={"speed": "fast"}).status_code == 422
    assert client.post("/api/control/pause").json()["paused"]
    assert not client.post("/api/control/start").json()["paused"]
    assert client.get("/api/governance/verify").json()["ok"]


def test_websocket_stream(client):
    client.post("/api/replay/focused", json={"speed": 200})
    with client.websocket_connect("/ws/events") as ws:
        first = json.loads(ws.receive_text())
        assert first["type"] == "hello" and len(first["states"]) == 4
        seen = set()
        for _ in range(60):
            seen.add(json.loads(ws.receive_text())["type"])
            if {"prediction", "governance", "state_vector"} <= seen:
                break
        assert {"prediction", "governance", "state_vector"} <= seen
    hist = client.get("/api/history?minutes=15").json()
    assert any(m["type"] == "prediction" for m in hist)


def test_mic_failure_keeps_the_camera(client, monkeypatch):
    """A wrong or silent mic must not take live mode down: the edge restarts video-only."""
    import sys
    import types

    calls = []

    def fake_start(sink, *, video=True, audio=True, obscure_face=False):
        calls.append(audio)
        if audio:
            raise RuntimeError("mic is silent")
        return lambda: None

    monkeypatch.setitem(sys.modules, "valence.edge.run", types.SimpleNamespace(start=fake_start))
    res = client.post("/api/live").json()
    assert calls == [True, False]
    assert res["ok"] and res["edge_error"] is None and "mic is silent" in res["mic_error"]
    assert client.get("/api/health").json()["mic_error"]
    assert "mic" in client.get("/api/state").json()["status"]["degraded"]
