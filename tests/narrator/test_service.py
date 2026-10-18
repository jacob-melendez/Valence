from __future__ import annotations

import copy
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from valence.contracts import Explanation, FactSheet, NarrationResult
from valence.explain.narrator_client import NarratorClient
from valence.narrator.engine import Narrator, TemplateOnlyNarrator
from valence.narrator.fake_fact_sheets import example_fact_sheet
from valence.narrator.service import create_app, main
from valence.narrator.templates import render_template
from valence.narrator.validator import validate

GOOD = (
    "Model predicted Frustrated / Masked Stress (88%) for Person #1. The room had average noise of 72 dB over "
    "15 minutes, and they showed 3 clenched-fist events in the last 5 minutes after 4.2 hours since the last "
    "break. Despite a smiling expression (valence +0.62), context corrected the facial mask."
)


class FakeNarrator:
    """Minimal stand-in for the engine: counts loads and echoes a fixed answer."""

    def __init__(self) -> None:
        self.loads = 0
        self.closed = False
        self.seen: list[dict] = []

    def load(self) -> None:
        self.loads += 1

    def narrate(self, sheet: dict) -> dict:
        self.seen.append(sheet)
        return {"text": GOOD, "source": "llm", "latency_ms": 12, "attempts": 1, "reasons": [], "reused": False}

    def health(self) -> dict:
        return {"ok": True, "model": "fake", "warm": self.loads > 0, "requests": len(self.seen)}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake():
    return FakeNarrator()


@pytest.fixture
def client(fake):
    with TestClient(create_app(narrator=fake)) as test_client:
        yield test_client


def test_lifespan_loads_once_and_closes(fake):
    with TestClient(create_app(narrator=fake)) as test_client:
        test_client.get("/health")
        test_client.get("/health")
        assert fake.loads == 1
    assert fake.closed


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["warm"] is True


def test_narrate_returns_exactly_a_narration_result(client, fake):
    response = client.post("/narrate", json=example_fact_sheet())
    assert response.status_code == 200
    body = response.json()
    assert body == {"text": GOOD, "source": "llm", "latency_ms": 12, "attempts": 1}
    NarrationResult.model_validate(body)  # extra="forbid": no diagnostic fields leak
    assert fake.seen[0] == FactSheet.model_validate(example_fact_sheet()).model_dump(mode="json")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("label"),
        lambda s: s.pop("n_people"),
        lambda s: s.update(extra_field=1),
        lambda s: s["layers"][0]["drivers"][0].pop("level"),
        lambda s: s["layers"][0]["drivers"][0].update(level="extreme"),
        lambda s: s["layers"][0]["drivers"][0].update(effect="maybe"),
        lambda s: s["layers"][0]["drivers"][0].update(scope="desk"),
        lambda s: s["layers"][0].update(name="Vibes"),
        lambda s: s["data_quality"].update(unknown=True),
        lambda s: s.update(layers="none"),
    ],
    ids=["missing-label", "missing-n_people", "extra-field", "driver-level-missing", "driver-level-enum",
         "driver-effect-enum", "driver-scope-enum", "layer-name", "data-quality-extra", "layers-type"],
)
def test_narrate_rejects_off_contract_sheets(client, fake, mutate):
    sheet = copy.deepcopy(example_fact_sheet())
    mutate(sheet)
    assert client.post("/narrate", json=sheet).status_code == 422
    assert fake.seen == []


def test_narrate_rejects_non_json(client):
    response = client.post("/narrate", content=b"not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_template_only_app_end_to_end():
    with TestClient(create_app(TemplateOnlyNarrator())) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        response = test_client.get("/health")
    health = response.json()
    assert body["source"] == "template" and body["attempts"] == 0 and isinstance(body["latency_ms"], int)
    assert body["text"] == render_template(example_fact_sheet()) and validate(body["text"], example_fact_sheet()).ok
    assert health["by_source"]["template"] == 1 and health["mode"] == "template"
    # no model loaded: the backend must show "fallback", so /health is not a 200
    assert response.status_code == 503 and health["ok"] is False and health["model"] is None


def test_real_engine_with_fake_backend_through_http():
    narrator = Narrator(generate_fn=lambda messages, params, deadline: GOOD)
    with TestClient(create_app(narrator)) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        response = test_client.get("/health")
    health = response.json()
    assert set(body) == {"text", "source", "latency_ms", "attempts"}
    assert body["source"] == "llm" and body["text"] == GOOD and body["attempts"] == 1
    assert response.status_code == 200 and health["ok"] is True
    assert health["warm"] is True and health["by_source"]["llm"] == 1


def test_model_load_failure_still_serves_templates_but_health_is_503(tmp_path):
    """Review F1: a failed load used to answer 200/ok, so the backend showed narrator="ready"."""
    narrator = Narrator(model_path=str(tmp_path / "missing"))
    with TestClient(create_app(narrator)) as test_client:
        body = test_client.post("/narrate", json=example_fact_sheet()).json()
        response = test_client.get("/health")
    health = response.json()
    assert body["source"] == "template"
    assert response.status_code == 503
    assert health["ok"] is False and health["mode"] == "template" and health["warm"] is False
    assert "FileNotFoundError" in health["error"] and health["model"] is None


def test_core_client_sees_failed_load_as_unhealthy(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    narrator = Narrator(model_path=str(tmp_path / "missing"))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(narrator), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        assert NarratorClient(f"http://127.0.0.1:{port}", lambda message: None).check_health() is False
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_cli_starts_uvicorn_on_loopback(monkeypatch):
    seen: dict = {}

    def fake_run(app, host, port, **kwargs):
        seen.update(host=host, port=port, narrator=app.state.narrator)

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["--port", "8099", "--template-only"]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8099 and isinstance(seen["narrator"], TemplateOnlyNarrator)
    assert main(["--adapter", "adapters/x"]) == 0
    assert isinstance(seen["narrator"], Narrator) and seen["narrator"].adapter_path == "adapters/x"
    seen["narrator"].close()


# --- the core's real client against a live local server ---------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    uvicorn = pytest.importorskip("uvicorn")
    narrator = Narrator(generate_fn=lambda messages, params, deadline: GOOD)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(narrator), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_core_narrator_client_works_against_the_service(live_server):
    published: list[Explanation] = []
    done = threading.Event()

    def publish(message: Explanation) -> None:
        published.append(message)
        done.set()

    client = NarratorClient(live_server, publish)
    assert client.check_health() is True
    client.start()
    try:
        client.request(FactSheet.model_validate(example_fact_sheet()), person_id=1)
        assert done.wait(5), "the client never published an explanation"
    finally:
        client.stop()
    message = published[0]
    assert isinstance(message, Explanation)
    assert (message.event_id, message.person_id, message.text, message.source) == ("evt_000123", 1, GOOD, "llm")
    assert client.healthy and client.failures == 0
